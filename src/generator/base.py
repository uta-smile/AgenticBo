"""The decoder always receives the full native tensor."""
from pathlib import Path
from typing import Protocol

import torch


class ProteinGenerator(Protocol):
    latent_shape: tuple[int, ...]

    def decode(self, latent: torch.Tensor, output_dir: Path) -> Path:
        """Decode one native latent and return its generated mmCIF path."""
        ...
