"""Backend-owned full-dimensional strategy tools; no oracle edits or vectors in LLM text."""
from __future__ import annotations

import time

import torch

from dsp.acquisition import acquisition
from dsp.diagnostics import diagnostics
from dsp.model import fit_gp
from dsp.optimize_acq import AcquisitionSettings, optimize_acquisition
from dsp.state import OptimizationState, checked_result
from latent.bounds import LatentBox


class OptimizationBackend:
    def __init__(self, state: OptimizationState, box: LatentBox, generator, oracle, *, dsp: bool = True,
                 seed: int = 0, acquisition_settings: AcquisitionSettings = AcquisitionSettings(), fit_maxiter: int = 100,
                 evaluator=None, save_gp=True):
        if state.dimension != box.dimension:
            raise ValueError("Backend dimension must equal the entire native latent dimension")
        self.state, self.box, self.generator, self.oracle = state, box, generator, oracle
        self.evaluator, self.save_gp = evaluator, save_gp
        self.warmup = 5 if evaluator is not None else 2
        self.use_dsp, self.seed, self.acquisition_settings, self.fit_maxiter = dsp, seed, acquisition_settings, fit_maxiter
        self._fitted, self._fit_observations = None, -1

    def fit(self):
        x, y = self.state.observations()
        if len(x) < self.warmup:
            raise ValueError(f"GP requires {self.warmup} valid observations; request a space-filling suggestion")
        if len(x) != self._fit_observations:
            self._fitted = fit_gp(x, y, dsp=self.use_dsp, maxiter=self.fit_maxiter)
            self._fit_observations = len(x)
            if self.save_gp:
                directory = self.state.directory / "gp"
                directory.mkdir(exist_ok=True)
                torch.save({"observations": len(x), "model": self._fitted.model.state_dict(),
                            "lengthscales": self._fitted.model.covar_module.lengthscale.detach(),
                            "diagnostics": diagnostics(self._fitted, x)}, directory / f"fit_{len(x):04d}.pt")
        return self._fitted

    def _acquisition(self):
        _, y = self.state.observations()
        return acquisition(self.fit().model, float(y.max()), name=self.state.get_setting("acquisition", "log_ei"),
                           beta=self.state.get_setting("ucb_beta", 2.0))

    def _bounds(self, radius=None):
        d = self.state.dimension
        radius = self.state.get_setting("search_radius", 1.0) if radius is None else radius
        if radius == 1.0:
            return torch.stack([torch.zeros(d), torch.ones(d)]).double()
        if not isinstance(radius, (float,int)) or not 1e-6 <= radius <= 1:
            raise ValueError("Local radius must lie in [1e-6,1] of the normalized box")
        incumbent = self.state.incumbent()
        if incumbent is None:
            raise ValueError("An incumbent is required for local search")
        center = self.state.candidate(incumbent["candidate_id"])
        return torch.stack([(center-radius).clamp(0,1), (center+radius).clamp(0,1)])

    def _suggest(self, radius=None):
        if len(self.state.observations()[0]) < self.warmup:
            from dsp.sampling import sobol_points
            counter = self.state.get_setting("suggestion_count", 0)
            x = sobol_points(self.state.dimension, 1, self.seed, skip=counter)[0]
            bounds = self._bounds(radius)
            x = bounds[0] + x * (bounds[1] - bounds[0])
            candidate = self.state.add_candidate(x, {"source": "warmup_sobol"})
            self.state.set_setting("suggestion_count", counter + 1)
            return {"candidate_id": candidate, "source": "warmup_sobol", "gp_ready": False}
        incumbent = self.state.incumbent()
        if incumbent is None:
            raise ValueError("Load the common initial observations before requesting suggestions")
        center = self.state.candidate(incumbent["candidate_id"])
        counter = self.state.get_setting("suggestion_count", 0)
        x, info = optimize_acquisition(self._acquisition(), self._bounds(radius), center,
            seed=self.seed + 1009*self.state.used + counter, settings=self.acquisition_settings)
        with torch.no_grad():
            posterior = self.fit().model.posterior(x.unsqueeze(0))
        info.update(source="local_acquisition" if radius is not None or self.state.get_setting("search_radius",1.0) < 1 else "global_acquisition",
                    gp=diagnostics(self.fit(), self.state.observations()[0]),
                    posterior_mean=float(posterior.mean.squeeze()), posterior_std=float(posterior.variance.sqrt().squeeze()),
                    distance_to_incumbent=float(torch.linalg.vector_norm(x-center)),
                    distance_to_z0=float(torch.linalg.vector_norm(x-.5)),
                    acquisition_name=self.state.get_setting("acquisition", "log_ei"))
        candidate_id = self.state.add_candidate(x, info)
        self.state.set_setting("suggestion_count", counter+1)
        return {"candidate_id": candidate_id, "posterior_mean": info["posterior_mean"],
                "posterior_std": info["posterior_std"], "acquisition_value": info["acquisition_value"],
                "optimization_succeeded": info["optimization_succeeded"]}

    def suggest(self):
        return self._suggest()

    def suggest_local(self, center="incumbent", radius=0.25):
        if center != "incumbent":
            raise ValueError("Local search center must be the backend incumbent")
        return self._suggest(radius)

    def predict(self, candidate_id):
        with torch.no_grad():
            posterior = self.fit().model.posterior(self.state.candidate(candidate_id).unsqueeze(0))
        return {"mean": float(posterior.mean.squeeze()), "std": float(posterior.variance.sqrt().squeeze())}

    def acquisition_score(self, candidate_id):
        with torch.no_grad():
            score = self._acquisition()(self.state.candidate(candidate_id).reshape(1,1,-1))
        return {"value": float(score.squeeze()), "name": self.state.get_setting("acquisition", "log_ei")}

    def incumbent(self):
        trial = self.state.incumbent()
        keys = {"trial_id","candidate_id","status","objective","tm","lddt","rmsd","valid","metrics"}
        return {k:v for k,v in trial.items() if k in keys} if trial else None

    def diagnostics(self):
        if len(self.state.observations()[0]) < self.warmup:
            return {"gp_ready": False, "valid_observations": len(self.state.observations()[0]),
                    "warmup_observations": self.warmup}
        result = diagnostics(self.fit(), self.state.observations()[0])
        objectives = [t["objective"] for t in self.state.trials() if t["status"] == "completed" and t.get("valid")]
        # A recent trial improves only if it beats the entire preceding history.
        # Exclude the first observation, which has no incumbent to improve upon.
        recent_indices = range(max(1, len(objectives)-10), len(objectives))
        result["recent_improvement_rate"] = (sum(objectives[i] > max(objectives[:i]) for i in recent_indices)
                                               / len(recent_indices) if recent_indices else 0.0)
        result["active_normalized_radius"] = self.state.get_setting("search_radius",1.0)
        result["acquisition"] = self.state.get_setting("acquisition", "log_ei")
        return result

    def trials(self, last_n=5):
        # Structural details remain on disk; the controller gets concise metrics.
        keys = {"trial_id","candidate_id","status","objective","tm","lddt","rmsd","valid","metrics"}
        return [{k:v for k,v in t.items() if k in keys} for t in self.state.trials(last_n)]

    def set_search_radius(self, radius):
        self._bounds(radius)  # Validate without changing persistent state.
        self.state.set_setting("search_radius", radius)
        return {"radius": radius, "dimension": self.state.dimension}

    def reset_bounds(self):
        self.state.set_setting("search_radius", 1.0)
        return {"radius": 1.0, "dimension": self.state.dimension}

    def set_acquisition(self, name, beta=2.0):
        if name not in {"log_ei", "ucb"} or not isinstance(beta, (int, float)) or not 0 <= beta < float("inf"):
            raise ValueError("Choose log_ei or ucb with finite nonnegative beta")
        self.state.set_setting("acquisition", name)
        self.state.set_setting("ucb_beta", beta)
        return {"name": name, "beta": beta}

    def evaluate(self, candidate_id, *, agent_decision=None):
        x = self.state.candidate(candidate_id)
        incumbent = self.state.incumbent()
        previous_best = incumbent["objective"] if incumbent else 0.0
        trial_id = self.state.reserve(candidate_id)
        started = time.monotonic()
        generator_runtime = None
        try:
            directory = self.state.directory / "structures" / f"trial_{trial_id:06d}"
            if self.evaluator is not None:
                result = checked_result(self.evaluator.evaluate(x, directory), self.state.objective)
            else:
                path = self.generator.decode(self.box.to_native(x), directory)
                generator_runtime = time.monotonic()-started
                result = checked_result(self.oracle.score(path), self.state.objective)
                result["structure_path"] = str(path)
        except Exception as exc:
            result = checked_result({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, self.state.objective)
        result["evaluation_runtime_seconds"] = time.monotonic()-started
        result["generator_runtime_seconds"] = generator_runtime if generator_runtime is not None else time.monotonic()-started
        result["candidate_diagnostics"] = self.state.candidate_info(candidate_id)
        if agent_decision is not None:
            result["agent_decision"] = {**agent_decision,"improvement_after_override":
                max(0.0,result["objective"]-previous_best) if result["valid"] and incumbent and agent_decision["overrode_dsp_suggestion"] else None}
        self.state.finish(trial_id,result)
        return self.state.trials(last_n=1)[0]
