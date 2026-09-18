import numpy as np


def align_coordinates(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Optimal proper rigid rotation for the fixed residue correspondence."""
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3 or not len(moving):
        raise ValueError("Expected equal nonempty [N, 3] coordinate arrays")
    x, y = moving - moving.mean(axis=0), reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return x @ (u @ correction @ vt) + reference.mean(axis=0)


def rmsd(moving: np.ndarray, reference: np.ndarray) -> float:
    aligned = align_coordinates(moving, reference)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1))))
