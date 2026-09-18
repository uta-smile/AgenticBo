"""Ax baseline with saved generation state for safe process restarts.

Protocol:
- Ax receives the exact shared five-point initialization used by the other
  methods.
- Trial 6 onward uses Ax's model-based generation strategy.
- Ax's model never consumes a second physical initialization budget.

Synthetic runs use initial=0. Protein comparisons can import shared initial
observations; those count toward the five-observation initialization.
"""

import json
import time
from importlib.metadata import version

import torch


def ax_identity(variant="default"):
    if variant not in {"default", "saasbo"}:
        raise ValueError(f"Unknown Ax variant: {variant}")
    return {
        "ax_platform": version("ax-platform"),
        "botorch": version("botorch"),
        "protocol": (
            "five_shared_sobol_then_saasbo"
            if variant == "saasbo"
            else "five_shared_sobol_then_ax_model_defaults"
        ),
        "variant": variant,
        "model": "SAASBO" if variant == "saasbo" else "Ax model defaults",
        "initialization_trials": 5,
        "noise_sem": 0.0,
        "device": "cpu",
        "dtype": "float64",
    }


class AxSession:
    """Keep one Ax generation strategy alive for one method segment."""

    def __init__(self, variant="default"):
        if variant not in {"default", "saasbo"}:
            raise ValueError(f"Unknown Ax variant: {variant}")
        self.variant = variant
        self.client = None
        self.pending = None
        self.names = None

    def _create(self, state, *, seed):
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties

        snapshot = state.directory / "ax" / f"before_eval_{state.used:04d}.json"
        if state.used and snapshot.exists():
            self.client = AxClient.load_from_json_file(str(snapshot))
            self.names = [f"x_{i}" for i in range(state.dimension)]
            last = state.trials()[-1]
            self.pending = (state.candidate_info(last["candidate_id"])["ax_trial_index"], last["candidate_id"])
            self.complete(last["candidate_id"], last)
            return
        if state.used and state.get_setting("initial_source_sha256") is None:
            raise ValueError("Ax resume requires its saved generation snapshot; use a new output directory")

        # IMPORTANT:
        # Do not make this depend on state.used.
        #
        # The same benchmark seed should always reconstruct the same
        # Ax random seed.
        if self.variant == "saasbo":
            from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
            from ax.modelbridge.registry import Models

            steps = []
            remaining_initial = max(0, 5 - state.used)
            if remaining_initial:
                steps.append(GenerationStep(model=Models.SOBOL, num_trials=remaining_initial))
            steps.append(GenerationStep(model=Models.SAASBO, num_trials=-1))
            generation_strategy = GenerationStrategy(steps=steps)
            client = AxClient(
                generation_strategy=generation_strategy,
                random_seed=seed,
                verbose_logging=False,
            )
        else:
            client = AxClient(
                random_seed=seed,
                verbose_logging=False,
                torch_device=torch.device("cpu"),
            )

        names = [
            f"x_{i}"
            for i in range(state.dimension)
        ]

        client.create_experiment(
            name="paper_fig5_synthetic_ax",
            parameters=[
                {
                    "name": name,
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                }
                for name in names
            ],
            objectives={
                "objective": ObjectiveProperties(
                    minimize=False
                )
            },

            # Paper protocol:
            # exactly 5 Ax-owned Sobol initialization trials.
            **({} if self.variant == "saasbo" else {
                "choose_generation_strategy_kwargs": {
                    "num_initialization_trials": max(0, 5 - state.used)
                }
            }),
        )

        # Replay the shared observations into Ax so its model sees exactly the
        # same initialization as every other method.
        for trial in state.trials():
            if trial["status"] != "completed":
                raise ValueError(
                    "Resolve pending trials before asking Ax "
                    "for a candidate"
                )

            values = state.candidate(
                trial["candidate_id"]
            )

            _, index = client.attach_trial(
                dict(
                    zip(
                        names,
                        values.tolist(),
                    )
                )
            )

            if not trial.get("valid"):
                client.log_trial_failure(index)
                continue
            client.complete_trial(
                index,
                raw_data={
                    "objective": (
                        trial["objective"],
                        0.0,
                    )
                },
            )

        self.client = client
        self.names = names

    def suggest(self, state, *, seed):
        if self.client is None:
            self._create(
                state,
                seed=seed,
            )

        if self.pending is not None:
            raise RuntimeError(
                "Ax has an uncompleted proposal"
            )

        started = time.monotonic()

        parameters, index = (
            self.client.get_next_trial()
        )

        x = torch.tensor(
            [
                parameters[name]
                for name in self.names
            ],
            dtype=torch.float64,
        )

        # -------------------------------------------------------------
        # Determine whether Ax is currently in:
        #
        #   trials 1-5   -> Sobol
        #   trials 6+    -> BoTorch GP
        #
        # Do NOT assume every Ax trial already has a GP surrogate.
        # -------------------------------------------------------------

        model_bridge = (
            self.client.generation_strategy.model
        )

        inner_model = getattr(
            model_bridge,
            "model",
            None,
        )

        phase = "sobol"
        kernel = None
        acquisition = None

        if (
            inner_model is not None
            and hasattr(inner_model, "surrogate")
        ):
            phase = "model_based"

            surrogate = inner_model.surrogate

            gp = getattr(
                surrogate,
                "model",
                None,
            )

            if gp is not None:
                covar_module = getattr(
                    gp,
                    "covar_module",
                    None,
                )

                if covar_module is not None:
                    kernel = type(
                        covar_module
                    ).__name__

            acqf_class = getattr(
                inner_model,
                "botorch_acqf_class",
                None,
            )

            if acqf_class is not None:
                acquisition = (
                    acqf_class.__name__
                )

        info = {
            "source": "ax",
            "ax": ax_identity(self.variant),
            "ax_trial_index": index,
            "phase": phase,
            "kernel": kernel,
            "acquisition": acquisition,
            "generation_strategy": str(
                self.client.generation_strategy
            ),
            "proposal_runtime_seconds": (
                time.monotonic() - started
            ),
        }

        directory = (
            state.directory / "ax"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client.save_to_json_file(
            str(
                directory
                / f"before_eval_{state.used + 1:04d}.json"
            )
        )

        (
            directory
            / f"diagnostics_{state.used:04d}.json"
        ).write_text(
            json.dumps(
                info,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        candidate_id = state.add_candidate(
            x,
            info,
        )

        self.pending = (
            index,
            candidate_id,
        )

        return candidate_id

    def complete(
        self,
        candidate_id,
        result,
    ):
        if (
            self.pending is None
            or self.pending[1] != candidate_id
        ):
            raise RuntimeError(
                "Ax completion does not match "
                "its pending candidate"
            )

        if not result.get("valid"):
            self.client.log_trial_failure(self.pending[0])
            self.pending = None
            return
        self.client.complete_trial(
            self.pending[0],
            raw_data={
                "objective": (
                    result["objective"],
                    0.0,
                )
            },
        )

        self.pending = None


def suggest_ax(state, *, seed):
    """Compatibility helper; runner normally uses AxSession."""
    return AxSession().suggest(
        state,
        seed=seed,
    )
