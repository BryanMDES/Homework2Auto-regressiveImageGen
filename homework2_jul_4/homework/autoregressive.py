import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.max_len = max_len
        self.tok_emb = torch.nn.Embedding(n_tokens, d_latent)
        self.start_emb = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, max_len, d_latent))

        self.drop = torch.nn.Dropout(dropout)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=4 * d_latent,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.lm_head = torch.nn.Linear(d_latent, n_tokens, bias=False)
        torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.start_emb, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(f"Expected integer token tensor, got dtype={x.dtype}")

        B, h, w = x.shape
        L = h * w
        if L > self.max_len:
            raise ValueError(f"Sequence length h*w={L} exceeds max_len={self.max_len}. Increase max_len in __init__.")

        seq = x.view(B, L).long()  
        tok = self.tok_emb(seq) 
        shifted = torch.cat([self.start_emb.expand(B, 1, -1), tok[:, :-1, :]], dim=1) 
        shifted = shifted + self.pos_emb[:, :L, :]
        z = self.drop(shifted)

        
        mask = self._causal_mask(L, device=z.device)
        z = self.tr(z, mask=mask)  

        
        logits_seq = self.lm_head(z)  
        logits = logits_seq.view(B, h, w, self.n_tokens)

        aux = {"probs": torch.softmax(logits, dim=-1)}
        return logits, aux

    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        self.eval()

        L = h * w
        if L > self.max_len:
            raise ValueError(f"Sequence length h*w={L} exceeds max_len={self.max_len}. Increase max_len in __init__.")

        
        out = torch.zeros(B, h, w, device=device, dtype=torch.long)

        for t in range(L):
            i = t // w
            j = t % w

            logits, _ = self.forward(out)               
            step_logits = logits[:, i, j, :]            
            probs = torch.softmax(step_logits, dim=-1)  

            next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            out[:, i, j] = next_tok

        return out