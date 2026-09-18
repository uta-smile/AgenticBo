from __future__ import annotations

import torch


class LatentBox:
    """One-to-one full-D box around fixed native z0, represented in float64."""
    def __init__(self, z0: torch.Tensor, radius: float, scale: float | torch.Tensor = 1.0):
        self.native_shape = tuple(z0.shape)
        self.z0 = z0.detach().clone().to(dtype=torch.float64).reshape(-1)
        self.scale = torch.as_tensor(scale, dtype=torch.float64, device=self.z0.device)
        if self.scale.numel() == 1:
            self.scale = self.scale.expand_as(self.z0).clone()
        elif self.scale.numel() == self.z0.numel():
            self.scale = self.scale.reshape(-1).clone()
        else:
            raise ValueError("Scale must be scalar or have exactly D entries")
        if (not self.z0.numel() or not torch.isfinite(self.z0).all()
                or not torch.isfinite(self.scale).all() or (self.scale <= 0).any()
                or not torch.isfinite(torch.tensor(radius)) or radius <= 0):
            raise ValueError("Finite z0, positive finite scales and radius are required")
        self.radius = float(radius)
        self.lower, self.upper = self.z0 - radius * self.scale, self.z0 + radius * self.scale
        if not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all() or (self.lower >= self.upper).any():
            raise ValueError("Each native coordinate needs a finite nondegenerate interval")

    @property
    def dimension(self):
        return self.z0.numel()

    def to_native(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64, device=self.z0.device)
        if x.shape != self.z0.shape or not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("x must contain D finite values in [0,1]")
        return (self.z0 + self.radius * self.scale * (2*x - 1)).reshape(self.native_shape)

    def to_unit(self, z: torch.Tensor) -> torch.Tensor:
        if tuple(z.shape) != self.native_shape:
            raise ValueError("z must have the full native shape")
        flat = z.to(self.z0).reshape(-1)
        x = ((flat - self.z0) / (self.radius * self.scale) + 1) / 2
        if not torch.isfinite(x).all() or (x < -1e-12).any() or (x > 1 + 1e-12).any():
            raise ValueError("Native latent is outside the calibrated box")
        return x.clamp(0, 1)
