from dsp.sampling import sobol_points


def suggest_sobol(state, *, initial_count: int, seed: int):
    """
    Full-budget Sobol baseline.

    For the paper reproduction there are no shared initial observations.
    Evaluation 1 is Sobol index 0, evaluation 2 is index 1, etc.
    """

    # Shared designs contain a center followed by Sobol points.
    sobol_index = state.used - (1 if initial_count else 0)

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
