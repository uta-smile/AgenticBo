from dsp.sampling import sobol_points


def suggest_sobol(state, *, initial_count: int, seed: int):
    adaptive_done = state.used - initial_count
    if adaptive_done < 0:
        raise ValueError("Import all common initial observations first")
    # Initial design consists of the center plus initial_count-1 Sobol points.
    x = sobol_points(state.dimension, 1, seed, skip=initial_count - 1 + adaptive_done)[0]
    return state.add_candidate(x, {"source": "sobol", "sobol_index": initial_count - 1 + adaptive_done})
