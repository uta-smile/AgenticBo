import math

import torch


def flatten(latent: torch.Tensor) -> torch.Tensor:
    return latent.reshape(-1)


def restore(vector: torch.Tensor, native_shape: tuple[int, ...]) -> torch.Tensor:
    if vector.ndim != 1 or vector.numel() != math.prod(native_shape):
        raise ValueError("Vector must have exactly the full native dimensionality")
    return vector.reshape(native_shape)
