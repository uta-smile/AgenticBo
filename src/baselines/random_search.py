"""Independent normalized-coordinate random-search baseline."""
from __future__ import annotations

import torch


def suggest_random(state, *, initial_count: int, seed: int):
    """Suggest the next reproducible uniform point after shared initialization."""
    if state.used < initial_count:
        raise ValueError("Random baseline cannot suggest before shared initialization")
    index = state.used - initial_count
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(
        (index + 1, state.dimension),
        generator=generator,
        dtype=torch.float64,
    )
    return state.add_candidate(
        points[-1],
        {"source": "random", "random_index": index},
    )
