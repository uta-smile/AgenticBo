"""Supported, backend-owned structural optimization objectives."""
import math

DEFAULT_OBJECTIVE = "sqrt_tm_times_lddt"
OBJECTIVES = (DEFAULT_OBJECTIVE, "tm")


def validate_objective(name):
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown objective {name!r}; choose one of {OBJECTIVES}")
    return name


def objective_value(tm, lddt, name=DEFAULT_OBJECTIVE):
    validate_objective(name)
    return tm if name == "tm" else math.sqrt(tm * lddt)
