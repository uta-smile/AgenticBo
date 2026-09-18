import math


def lengthscale_prior_parameters(dimension: int, *, dsp: bool = True) -> dict:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Dimension must be a positive integer")
    loc = math.sqrt(2) + (0.5 * math.log(dimension) if dsp else 0.0)
    scale = math.sqrt(3)
    return {"dimension": dimension, "loc": loc, "scale": scale,
            "mode": math.exp(loc - scale**2), "dsp": dsp}
