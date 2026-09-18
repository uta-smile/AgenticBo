import torch


def sobol_points(
    dimension: int,
    count: int,
    seed: int,
    *,
    skip: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:

    if not 1 <= dimension <= torch.quasirandom.SobolEngine.MAXDIM:
        raise ValueError(
            f"Native D={dimension} exceeds PyTorch Sobol support; "
            "no reduced-space fallback is allowed"
        )

    if count < 1 or skip < 0:
        raise ValueError(
            "Require positive sample count and nonnegative skip"
        )

    engine = torch.quasirandom.SobolEngine(
        dimension,
        scramble=True,
        seed=seed,
    )

    if skip:
        engine.fast_forward(skip)

    return engine.draw(
        count,
        dtype=torch.float64,
    ).to(device)


def initial_design(
    dimension: int,
    count: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    """
    Synthetic benchmark initialization:
    center followed by Sobol points.
    """

    if count < 2:
        raise ValueError(
            "Initial design needs the center and at least one Sobol point"
        )

    return torch.cat(
        [
            torch.full(
                (1, dimension),
                0.5,
                dtype=torch.float64,
            ),
            sobol_points(
                dimension,
                count - 1,
                seed,
            ),
        ]
    )


def gaussian_initial_design(
    dimension: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    """
    Protein initialization in normalized x-space.

    No artificial x=0.5 center is included because under the
    Gaussian inverse-CDF transform it would correspond to the
    degenerate all-zero Boltz noise vector.

    Every point is a scrambled Sobol point in [0,1]^D and is later
    mapped through GaussianLatentSpace.to_native().
    """

    if count < 1:
        raise ValueError(
            "Gaussian initial design requires at least one point"
        )

    return sobol_points(
        dimension,
        count,
        seed,
    )