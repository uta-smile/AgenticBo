import numpy as np
from tmtools import tm_align


def tm_score(moving: np.ndarray, reference: np.ndarray, sequence: str) -> float:
    """TM-align score normalized by reference length, not its aligned subset."""
    if len(sequence) < 3:
        raise ValueError("TM-align requires at least three residues")
    result = tm_align(np.ascontiguousarray(moving, dtype=np.float64),
                      np.ascontiguousarray(reference, dtype=np.float64), sequence, sequence)
    score = float(result.tm_norm_chain2)
    if not np.isfinite(score) or not 0 <= score <= 1 + 1e-8:
        raise ValueError("TM-align returned an invalid score")
    return min(score, 1.0)
