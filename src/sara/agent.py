from __future__ import annotations

import json
from pathlib import Path
import time

from sara.schemas import TOOLS, validate_action
from sara.history import history_messages


TERMINAL_ACTIONS = {"EVALUATE", "STOP"}


class SaraController:
    def __init__(
        self,
        client,
        *,
        prompt_path: Path,
        max_rounds: int = 20,
    ):
        if max_rounds < 1:
            raise ValueError(
                "max_rounds must be positive"
            )

        self.client = client
        self.prompt = prompt_path.read_text()
        self.max_rounds = max_rounds

    def select(
        self,
        backend,
        *,
        target_id: str,
        target_length: int,
        trace_path: Path,
        seed: int,
    ):
        trace_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        def log(event):
            with trace_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "budget_used": backend.state.used,
                            **event,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

        # Keep old deliberation bounded so long campaigns do not
        # exhaust the llama.cpp context window.
        retained_history = history_messages(
            backend.state,
            max_chars=6000,
        )

        # Fresh backend proposal available to Sara as an advisory.
        advisory = backend.suggest()

        context = {
            "target_id": target_id,
            "target_length": target_length,
            "D": backend.state.dimension,
            "objective": getattr(
                getattr(
                    backend.evaluator,
                    "oracle",
                    None,
                ),
                "objective",
                backend.state.objective,
            ),
            "latent_bounds": {
                "normalized": [0, 1],
                "native_radius": backend.box.radius,
                "center_id": "saved_z0",
                "native_shape": list(
                    backend.box.native_shape
                ),
            },
            "budget_used": backend.state.used,
            "budget_remaining": backend.state.remaining,
            "incumbent": backend.incumbent(),
            "recent_trials": backend.trials(5),
            "gp": backend.diagnostics(),
            "advisory_candidate": advisory,
        }

        current = {
            "role": "user",
            "content": json.dumps(
                context,
                allow_nan=False,
            ),
        }

        messages = [
            {
                "role": "system",
                "content": self.prompt,
            },
            *retained_history,
            current,
        ]

        turn_start = len(messages) - 1

        log(
            {
                "event": "turn_start",
                "context": context,
            }
        )

        tool_count = 0
        usage = []
        actions = []

        def action_metrics():
            return {
                "tool_calls": tool_count,
                "local_search_decisions": sum(
                    action
                    in {
                        "suggest_local",
                        "set_search_radius",
                    }
                    for action in actions
                ),
                "global_resets": actions.count(
                    "reset_bounds"
                ),
                "acquisition_switches": actions.count(
                    "set_acquisition"
                ),
            }

        for round_index in range(
            self.max_rounds
        ):
            message, tokens = self.client.complete(
                messages,
                TOOLS,
                seed + round_index,
            )

            usage.append(tokens)
            messages.append(message)

            log(
                {
                    "event": "assistant",
                    "message": message,
                    "usage": tokens,
                }
            )

            calls = (
                message.get("tool_calls")
                or []
            )

            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue reasoning with computational "
                            "tools if useful, or issue exactly one "
                            "EVALUATE(candidate_id) or STOP(reason)."
                        ),
                    }
                )
                continue

            # ---------------------------------------------------------
            # parallel_tool_calls=True is only parser tolerance.
            #
            # Sara still acts sequentially.
            #
            # If Qwen emits:
            #
            #   suggest_local(...)
            #   EVALUATE(...)
            #
            # only suggest_local is executed. EVALUATE is deferred
            # until Qwen sees the actual returned candidate.
            # ---------------------------------------------------------

            deferred_calls = []

            if len(calls) > 1:
                first_name = (
                    calls[0]
                    .get("function", {})
                    .get("name")
                )

                # Never execute a terminal action from a multi-call
                # response. No action from this batch is committed.
                if first_name in TERMINAL_ACTIONS:
                    error = {
                        "error": (
                            "Multiple tool calls were emitted. "
                            "EVALUATE and STOP must be issued alone. "
                            "No action was executed. Issue one action "
                            "on the next turn."
                        )
                    }

                    for call in calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    error
                                ),
                            }
                        )

                    log(
                        {
                            "event": (
                                "parallel_terminal_rejected"
                            ),
                            "calls": [
                                call.get(
                                    "function",
                                    {},
                                ).get("name")
                                for call in calls
                            ],
                        }
                    )

                    continue

                # Execute only the first computational call.
                deferred_calls = calls[1:]
                calls = calls[:1]

            call = calls[0]

            name = (
                call.get("function", {})
                .get("name")
            )

            try:
                arguments = json.loads(
                    call["function"]["arguments"]
                )

                validate_action(
                    name,
                    arguments,
                )

                # -----------------------------------------------------
                # STOP
                # -----------------------------------------------------
                if name == "STOP":
                    decision = {
                        "action": "STOP",
                        "reason": arguments[
                            "reason"
                        ].strip(),
                        **action_metrics(),
                        "accepted_dsp_suggestion": False,
                        "overrode_dsp_suggestion": False,
                        "usage": usage,
                        "usage_complete": all(
                            item is not None
                            for item in usage
                        ),
                    }

                    backend.state.set_setting(
                        "agent_history_pending",
                        {
                            "budget_used": (
                                backend.state.used
                            ),
                            "messages": messages[
                                turn_start:
                            ],
                        },
                    )

                    log(
                        {
                            "event": "stop_selected",
                            **decision,
                        }
                    )

                    return None, decision

                # -----------------------------------------------------
                # EVALUATE
                # -----------------------------------------------------
                if name == "EVALUATE":
                    candidate_id = arguments[
                        "candidate_id"
                    ]

                    # Ensure candidate exists.
                    backend.state.candidate(
                        candidate_id
                    )

                    if backend.state.was_evaluated(
                        candidate_id
                    ):
                        raise ValueError(
                            "Candidate has already been "
                            "evaluated; select a new proposal"
                        )

                    accepted = (
                        candidate_id
                        == advisory["candidate_id"]
                    )

                    metric = {
                        **action_metrics(),
                        "accepted_dsp_suggestion": (
                            accepted
                        ),
                        "overrode_dsp_suggestion": (
                            not accepted
                        ),
                        "advisory_candidate_id": (
                            advisory["candidate_id"]
                        ),
                        "candidate_id": candidate_id,
                        "usage": usage,
                        "usage_complete": all(
                            item is not None
                            for item in usage
                        ),
                    }

                    log(
                        {
                            "event": "selected",
                            **metric,
                        }
                    )

                    backend.state.set_setting(
                        "agent_history_pending",
                        {
                            "budget_used": (
                                backend.state.used
                            ),
                            "messages": messages[
                                turn_start:
                            ],
                        },
                    )

                    return candidate_id, metric

                # -----------------------------------------------------
                # Computational action
                # -----------------------------------------------------

                tool_count += 1

                result = getattr(
                    backend,
                    name,
                )(
                    **arguments
                )

                actions.append(name)

            except (
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                result = {
                    "error": str(exc)
                }

            # Result for the ONE action actually executed.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        result,
                        allow_nan=False,
                    ),
                }
            )

            log(
                {
                    "event": "tool",
                    "name": name,
                    "result": result,
                }
            )

            # ---------------------------------------------------------
            # Every tool_call in an assistant message requires a tool
            # response before the next assistant turn.
            #
            # We therefore acknowledge remaining calls but DO NOT
            # execute them.
            # ---------------------------------------------------------

            for deferred in deferred_calls:
                deferred_name = (
                    deferred
                    .get("function", {})
                    .get("name")
                )

                deferred_result = {
                    "deferred": True,
                    "reason": (
                        "Sara executes actions sequentially. "
                        "The previous action was executed. "
                        "Use its returned result before deciding "
                        "whether to issue this action again."
                    ),
                }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            deferred["id"]
                        ),
                        "content": json.dumps(
                            deferred_result
                        ),
                    }
                )

                log(
                    {
                        "event": (
                            "parallel_tool_deferred"
                        ),
                        "name": deferred_name,
                    }
                )

        raise RuntimeError(
            "Agent did not select EVALUATE or STOP "
            f"within {self.max_rounds} reasoning rounds"
        )