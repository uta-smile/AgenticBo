from __future__ import annotations

import math

import torch


class LatentBox:
    """Finite affine box used by the synthetic benchmarks."""

    def __init__(
        self,
        z0: torch.Tensor,
        radius: float,
        scale: float | torch.Tensor = 1.0,
    ):
        self.native_shape = tuple(z0.shape)

        self.z0 = (
            z0.detach()
            .clone()
            .to(dtype=torch.float64)
            .reshape(-1)
        )

        self.scale = torch.as_tensor(
            scale,
            dtype=torch.float64,
            device=self.z0.device,
        )

        if self.scale.numel() == 1:
            self.scale = (
                self.scale
                .expand_as(self.z0)
                .clone()
            )
        elif self.scale.numel() == self.z0.numel():
            self.scale = (
                self.scale
                .reshape(-1)
                .clone()
            )
        else:
            raise ValueError(
                "Scale must be scalar or have exactly D entries"
            )

        if (
            not self.z0.numel()
            or not torch.isfinite(self.z0).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
            or not torch.isfinite(torch.tensor(radius))
            or radius <= 0
        ):
            raise ValueError(
                "Finite z0, positive finite scales and radius are required"
            )

        self.radius = float(radius)

        self.lower = (
            self.z0
            - radius * self.scale
        )

        self.upper = (
            self.z0
            + radius * self.scale
        )

        if (
            not torch.isfinite(self.lower).all()
            or not torch.isfinite(self.upper).all()
            or (self.lower >= self.upper).any()
        ):
            raise ValueError(
                "Each native coordinate needs a finite nondegenerate interval"
            )

    @property
    def dimension(self):
        return self.z0.numel()

    @property
    def identity(self):
        return {
            "type": "affine_box",
            "normalized_domain": [0.0, 1.0],
            "radius": self.radius,
        }

    def to_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.as_tensor(
            x,
            dtype=torch.float64,
            device=self.z0.device,
        )

        if (
            x.shape != self.z0.shape
            or not torch.isfinite(x).all()
            or (x < 0).any()
            or (x > 1).any()
        ):
            raise ValueError(
                "x must contain D finite values in [0,1]"
            )

        return (
            self.z0
            + self.radius
            * self.scale
            * (2 * x - 1)
        ).reshape(self.native_shape)

    def to_unit(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(z.shape) != self.native_shape:
            raise ValueError(
                "z must have the full native shape"
            )

        flat = z.to(self.z0).reshape(-1)

        x = (
            (
                (flat - self.z0)
                / (self.radius * self.scale)
            )
            + 1
        ) / 2

        if (
            not torch.isfinite(x).all()
            or (x < -1e-12).any()
            or (x > 1 + 1e-12).any()
        ):
            raise ValueError(
                "Native latent is outside the calibrated box"
            )

        return x.clamp(0, 1)


class GaussianLatentSpace:
    """
    Full-dimensional Boltz latent space.

    BO works in x in [0,1]^D.

    x is transformed coordinate-wise through the inverse standard-normal
    CDF so space-filling designs in x correspond to global Gaussian
    latent exploration.

    eps truncates only the extreme Gaussian tails so x=0 and x=1 remain
    finite.
    """

    def __init__(
        self,
        native_shape,
        eps: float = 1e-6,
    ):
        self.native_shape = tuple(
            int(v)
            for v in native_shape
        )

        if (
            not self.native_shape
            or any(v < 1 for v in self.native_shape)
        ):
            raise ValueError(
                "native_shape must be nonempty and positive"
            )

        if (
            not math.isfinite(eps)
            or not 0 < eps < 0.5
        ):
            raise ValueError(
                "eps must lie in (0, 0.5)"
            )

        self.eps = float(eps)

        self.z0 = torch.zeros(
            math.prod(self.native_shape),
            dtype=torch.float64,
        )

        # Kept only for compatibility with places that inspect box.radius.
        # Protein search no longer has a native z0 +/- radius box.
        self.radius = None

    @property
    def dimension(self):
        return self.z0.numel()

    @property
    def identity(self):
        limit = float(
            (
                math.sqrt(2.0)
                * torch.erfinv(
                    torch.tensor(
                        1.0 - 2.0 * self.eps,
                        dtype=torch.float64,
                    )
                )
            ).item()
        )

        return {
            "type": "truncated_standard_normal_icdf",
            "normalized_domain": [0.0, 1.0],
            "eps": self.eps,
            "native_coordinate_limit": [
                -limit,
                limit,
            ],
        }

    def to_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.as_tensor(
            x,
            dtype=torch.float64,
        ).reshape(-1)

        if (
            x.shape != self.z0.shape
            or not torch.isfinite(x).all()
            or (x < 0).any()
            or (x > 1).any()
        ):
            raise ValueError(
                "x must contain D finite values in [0,1]"
            )

        # Uniform normalized BO coordinate -> truncated Gaussian quantile.
        u = (
            self.eps
            + (1.0 - 2.0 * self.eps) * x
        )

        z = (
            math.sqrt(2.0)
            * torch.erfinv(
                2.0 * u - 1.0
            )
        )

        if not torch.isfinite(z).all():
            raise ValueError(
                "Gaussian latent transform produced nonfinite values"
            )

        return z.reshape(
            self.native_shape
        )

    def to_unit(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(z.shape) != self.native_shape:
            raise ValueError(
                "z must have the full native shape"
            )

        z = (
            torch.as_tensor(
                z,
                dtype=torch.float64,
            )
            .reshape(-1)
        )

        if not torch.isfinite(z).all():
            raise ValueError(
                "z must be finite"
            )

        u = 0.5 * (
            1.0
            + torch.erf(
                z / math.sqrt(2.0)
            )
        )

        x = (
            (u - self.eps)
            / (1.0 - 2.0 * self.eps)
        )

        if (
            (x < -1e-12).any()
            or (x > 1.0 + 1e-12).any()
        ):
            raise ValueError(
                "Native latent lies outside the configured Gaussian tail cutoff"
            )

        return x.clamp(0, 1)