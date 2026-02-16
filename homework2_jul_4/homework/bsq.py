import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = int(codebook_bits)
        self._embedding_dim = int(embedding_dim)
        self.down = torch.nn.Linear(self._embedding_dim, self._codebook_bits, bias=True)
        self.up = torch.nn.Linear(self._codebook_bits, self._embedding_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        z = self.down(x) 
        z = torch.nn.functional.normalize(z, p=2, dim=-1, eps=1e-8)
        z = diff_sign(z) 
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = int(codebook_bits)
        self.bsq = BSQ(codebook_bits=self.codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        feats = super().encode(x)          # (B, h, w, latent_dim)
        idx = self.bsq.encode_index(feats) # (B, h, w)
        return idx.long()

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.bsq.decode_index(x.long())  # (B, h, w, latent_dim)
        img = super().decode(feats)              # (B, H, W, C)
        return img

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = super().encode(x)     # (B, h, w, latent_dim)
        code = self.bsq.encode(feats) # (B, h, w, codebook_bits)
        return code

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.bsq.decode(x)    # (B, h, w, latent_dim)
        img = super().decode(feats)   # (B, H, W, C)
        return img

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        code = self.encode(x)         # (B, h, w, codebook_bits)
        recon = self.decode(code)     # (B, H, W, C)

        # Codebook usage stats (OK to use encode_index here because it's detach-only stats)
        with torch.no_grad():
            cnt = torch.bincount(
                self.encode_index(x).flatten(),
                minlength=2**self.codebook_bits
            )
            stats = {
                # fraction of codes never used in this batch
                "cb0": (cnt == 0).float().mean(),
                # fraction of codes used <=2 times (very rare / near-dead)
                "cb2": (cnt <= 2).float().mean(),
            }

        return recon, stats
