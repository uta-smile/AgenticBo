"""Behavioral checks for the shared, domain-independent POC workflow."""
import builtins
import json
import math
from pathlib import Path

import pytest
import torch

from agenticbo.cli import parser, load_config
from agenticbo.evaluators import SyntheticEvaluator, ProteinEvaluator
from agenticbo.report import load_v2, curve, report
from dsp.optimize_acq import AcquisitionSettings
from dsp.state import OptimizationState, checked_result
from latent.bounds import LatentBox
from runner.engine import run_methods
from sara.agent import SaraController


class ChooseThenStop:
    def complete(self, messages, tools, seed):
        context = next(json.loads(m["content"]) for m in reversed(messages)
                       if m["role"] == "user" and m["content"].startswith("{")
                       and "advisory_candidate" in json.loads(m["content"]))
        if context["budget_used"] >= 2:
            name, args = "STOP", {"reason": "Test stopping and resume"}
        else:
            assert context["gp"]["gp_ready"] is False
            assert context["objective"] == "scalar"
            name, args = "EVALUATE", {"candidate_id": context["advisory_candidate"]["candidate_id"]}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "test", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}, None


def kwargs(tmp_path, evaluator=None, initial=0):
    evaluator = evaluator or SyntheticEvaluator("branin_2d", 0)
    return dict(directory=tmp_path, box=LatentBox(torch.full((2,), .5), .5), evaluator=evaluator,
                metadata={"kind": "benchmark", "evaluator": SyntheticEvaluator("branin_2d", 0).identity,
                          "objective_name": "negative_raw_objective"},
                target_id="opaque", target_length=2, seed=0, budget=6, initial=initial,
                methods=("sobol", "dsp_gp"), save_gp=False, fit_maxiter=10,
                acquisition_settings=AcquisitionSettings(raw_samples=8, num_restarts=2, maxiter=10))


def test_known_optima_and_shift():
    evaluator = SyntheticEvaluator("branin_2d", 7)
    x = torch.tensor([(math.pi + 5) / 15, 2.275 / 15], dtype=torch.float64)
    assert -evaluator.raw((x - evaluator.shift).remainder(1)) == pytest.approx(evaluator.optimum)
    for name in ("ackley_10d", "ackley_20d"):
        evaluator = SyntheticEvaluator(name, 7)
        x = (.5 - evaluator.shift).remainder(1)
        assert evaluator.raw(x) == pytest.approx(0, abs=1e-12)


def test_negative_objective_and_failed_trial_exclusion(tmp_path):
    state = OptimizationState(tmp_path, dimension=2, budget=2, metadata={"objective": "scalar"})
    for result in ({"valid": True, "objective": -4.0}, {"valid": False, "error": "failed"}):
        state.finish(state.reserve(state.add_candidate(torch.zeros(2))), result)
    assert state.used == 2 and state.incumbent()["objective"] == -4
    assert state.observations()[1].tolist() == [[-4.0]]
    assert state.trials()[1]["objective"] is None
    with pytest.raises(ValueError, match="finite"):
        checked_result({"valid": True, "objective": float("nan")}, "scalar")
    state.close()


def test_independent_opening_early_stop_report_and_resume(tmp_path):
    torch.set_num_threads(2)
    config = kwargs(tmp_path)
    config.update(methods=("sobol", "agentic_dsp"), controller=SaraController(
        ChooseThenStop(), prompt_path=Path("prompts/SARA_SYNTHETIC_SYSTEM.md")))
    first = run_methods(**config)
    assert first["total_physical_calls"] == 8
    assert not (tmp_path / "shared_initial").exists()
    runs = load_v2(tmp_path)
    agent = next(r for r in runs if r["method"] == "agentic_dsp")
    assert agent["budget_used"] == 2 and len(curve(agent)) == 6
    assert curve(agent)[-1] == curve(agent)[1]
    assert all("tm" not in t and t["objective"] < 0 for r in runs for t in r["trials"])
    assert report(tmp_path, tmp_path / "analysis")["runs"] == 2
    assert run_methods(**config)["total_physical_calls"] == 8
    config["budget"] = 7
    with pytest.raises(ValueError, match="context changed"):
        run_methods(**config)


def test_shared_initial_failures_are_charged_not_fitted(tmp_path):
    class FailFirst:
        calls = 0
        def evaluate(self, x, directory):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("fixture failure")
            return {"valid": True, "objective": -float(x.square().sum()), "metrics": {}, "artifacts": {}}
    evaluator = FailFirst()
    config = kwargs(tmp_path, evaluator, initial=3)
    result = run_methods(**config)
    assert evaluator.calls == result["total_physical_calls"] == 9
    runs = load_v2(tmp_path)
    assert all(not r["trials"][0]["valid"] for r in runs)
    assert len({r["shared_initial_sha256"] for r in runs}) == 1
    assert all(r["incumbent"]["objective"] < 0 for r in runs)


def test_protein_evaluator_maps_full_noise_and_keeps_metrics(tmp_path):
    box = LatentBox(torch.zeros(1, 2, 3), .25)
    class Generator:
        def decode(self, z, directory):
            assert z.shape == (1, 2, 3)
            assert torch.allclose(z, torch.full_like(z, .25))
            return directory / "model.cif"
    class Oracle:
        def score(self, path):
            return {"valid": True, "objective": .8, "tm": .8, "lddt": .7, "rmsd": 1.0}
    evaluator = ProteinEvaluator(box, Generator(), Oracle())
    result = evaluator.evaluate(torch.ones(6), tmp_path)
    assert result["objective"] == .8 and result["metrics"]["tm"] == .8
    assert result["artifacts"]["structure"].endswith("model.cif")


def test_config_overrides_and_radius_required(tmp_path):
    args = parser().parse_args(["benchmark", "--budget", "7", "--methods", "sobol"])
    _, config = load_config(args)
    assert config["budget"] == 7 and config["methods"] == ["sobol"]
    args = parser().parse_args(["protein", "--targets", "p"])
    with pytest.raises(ValueError, match="radius"):
        load_config(args)


def test_synthetic_does_not_import_protein_dependencies(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"boltz", "Bio", "tmtools", "generator"}:
            raise AssertionError(f"Synthetic imported {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    config = kwargs(tmp_path)
    config.update(methods=("sobol",), budget=2)
    assert run_methods(**config)["completed"]


def test_gp_transition_after_five_observations(tmp_path):
    torch.set_num_threads(2)
    config = kwargs(tmp_path)
    config["methods"] = ("dsp_gp",)
    run_methods(**config)
    trials = load_v2(tmp_path)[0]["trials"]
    assert all(t["candidate_diagnostics"]["source"] == "warmup_sobol" for t in trials[:5])
    assert trials[5]["candidate_diagnostics"]["source"] == "global_acquisition"
    assert not (tmp_path / "dsp_gp" / "gp").exists()


def test_ax_resume_preserves_initialization_progress(tmp_path):
    pytest.importorskip("ax")
    from baselines.ax_bo import AxSession
    torch.set_num_threads(2)
    evaluator = SyntheticEvaluator("branin_2d", 0)
    state = OptimizationState(tmp_path, dimension=2, budget=6, metadata={"objective": "scalar"})
    session = AxSession()
    for i in range(6):
        candidate = session.suggest(state, seed=0)
        info = state.candidate_info(candidate)
        assert info["phase"] == ("sobol" if i < 5 else "model_based")
        assert info["ax_trial_index"] == i
        result = evaluator.evaluate(state.candidate(candidate), tmp_path)
        state.finish(state.reserve(candidate), result)
        session.complete(candidate, result)
        # Restoring on every step must not restart Sobol or duplicate a trial.
        session = AxSession()
    assert state.used == 6
    assert len({tuple(state.candidate(t["candidate_id"]).tolist()) for t in state.trials()}) == 6
    state.close()


def test_report_rejects_modified_export(tmp_path):
    config = kwargs(tmp_path)
    config.update(methods=("sobol",), budget=2)
    run_methods(**config)
    path = tmp_path / "sobol" / "results.json"
    exported = json.loads(path.read_text())
    exported["trials"][0]["objective"] = 99
    path.write_text(json.dumps(exported))
    with pytest.raises(ValueError, match="inconsistent"):
        load_v2(tmp_path)
