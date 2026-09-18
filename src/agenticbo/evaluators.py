"""Problem-specific evaluation; the optimizer only sees normalized coordinates."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, TypedDict

import torch


class Evaluation(TypedDict):
    objective: float | None
    valid: bool
    metrics: dict
    artifacts: dict


class Evaluator(Protocol):
    def evaluate(self, x: torch.Tensor, directory: Path) -> Evaluation: ...


PROBLEMS = {
    "branin_2d": (2, 50),
    "ackley_10d": (10, 150),
    "ackley_20d": (20, 200),
}


class SyntheticEvaluator:
    def __init__(self, name: str, seed: int):
        self.name = name
        self.dimension, self.budget = PROBLEMS[name]
        key = hashlib.sha256(f"{name}:{seed}".encode()).digest()
        rng = torch.Generator().manual_seed(int.from_bytes(key[:8], "little") % (2**63 - 1))
        self.shift = torch.rand(self.dimension, generator=rng, dtype=torch.float64) * .5 - .25
        self.optimum = -(5 / (4 * math.pi)) if name == "branin_2d" else 0.0

    @property
    def identity(self):
        return {"problem": self.name, "dimension": self.dimension, "optimum": self.optimum,
                "transform": "periodic_shift_unit_cube_v1", "shift": self.shift.tolist()}

    def raw(self, x):
        y = (x.double() + self.shift).remainder(1.0)
        if self.name == "branin_2d":
            a, b = 15 * y[0] - 5, 15 * y[1]
            return float((b - 5.1 * a.square() / (4 * math.pi**2) + 5 * a / math.pi - 6)**2
                         + 10 * (1 - 1 / (8 * math.pi)) * torch.cos(a) + 10)
        z = 10 * (y - .5)
        return float(-20 * torch.exp(-.2 * z.square().mean().sqrt())
                     - torch.exp(torch.cos(2 * math.pi * z).mean()) + math.e + 20)

    def evaluate(self, x, directory):
        raw = self.raw(x)
        return {"valid": True, "objective": -raw, "metrics": {"raw_objective": raw}, "artifacts": {}}


class ProteinEvaluator:
    """Keep reference scoring separate from generator conditioning."""
    def __init__(self, box, generator, oracle):
        self.box, self.generator, self.oracle = box, generator, oracle

    def evaluate(self, x, directory):
        path = self.generator.decode(self.box.to_native(x), directory)
        result = self.oracle.score(path)
        return {"valid": result["valid"], "objective": result["objective"] if result["valid"] else None,
                "metrics": {k: result[k] for k in ("tm", "lddt", "rmsd")},
                "details": {k: v for k, v in result.items() if k not in {"valid", "objective", "tm", "lddt", "rmsd"}},
                "artifacts": {"structure": str(path)},
                **({"error": result["error"]} if "error" in result else {})}
