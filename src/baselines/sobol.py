from dsp.sampling import sobol_points


def suggest_sobol(
    state,
    *,
    initial_count: int,
    seed: int,
    shared_initial_has_center: bool = False,
):
    """
    Full-budget Sobol baseline.

    If the shared initial design contains an explicit center point,
    that center is not part of the Sobol sequence and therefore
    must be excluded when determining the next Sobol index.

    Gaussian protein initialization contains only Sobol points, so
    no subtraction is needed.
    """

    if shared_initial_has_center and initial_count:
        sobol_index = state.used - 1
    else:
        sobol_index = state.used

    if sobol_index < 0:
        raise ValueError(
            "Sobol index cannot be negative"
        )

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