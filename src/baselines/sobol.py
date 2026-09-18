from dsp.sampling import sobol_points


def suggest_sobol(state, *, initial_count: int, seed: int):
    """
    Full-budget Sobol baseline.

    For the paper reproduction there are no shared initial observations.
    Evaluation 1 is Sobol index 0, evaluation 2 is index 1, etc.
    """

    if initial_count != 0:
        raise ValueError(
            "Paper Figure 5 Sobol reproduction expects initial_count=0"
        )

    sobol_index = state.used

    x = sobol_points(
        state.dimension,
        1,
        seed,
        skip=sobol_index,
    )[0]

    return state.add_candidate(
        x,
        {
            "source": "sobol",
            "sobol_index": sobol_index,
        },
    )