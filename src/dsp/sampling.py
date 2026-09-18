import torch


def sobol_points(dimension: int, count: int, seed: int, *, skip: int = 0,
                 device: str | torch.device = "cpu") -> torch.Tensor:
    if not 1 <= dimension <= torch.quasirandom.SobolEngine.MAXDIM:
        raise ValueError(f"Native D={dimension} exceeds PyTorch Sobol support; no reduced-space fallback is allowed")
    if count < 1 or skip < 0:
        raise ValueError("Require positive sample count and nonnegative skip")
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    if skip:
        engine.fast_forward(skip)
    return engine.draw(count, dtype=torch.float64).to(device)


def initial_design(dimension: int, count: int = 16, seed: int = 0) -> torch.Tensor:
    if count < 2:
        raise ValueError("Initial design needs the center and at least one Sobol point")
    return torch.cat([torch.full((1, dimension), 0.5, dtype=torch.float64),
                      sobol_points(dimension, count - 1, seed)])
