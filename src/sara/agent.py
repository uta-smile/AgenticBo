from __future__ import annotations

import json
from pathlib import Path
import time

from sara.schemas import TOOLS, validate_action
from sara.history import history_messages


class SaraController:
    def __init__(self,client,*,prompt_path:Path,max_tools:int=4,max_rounds:int=6):
        if isinstance(max_tools,bool) or not isinstance(max_tools,int) or max_tools<0 or max_rounds<1:
            raise ValueError("Agent computational-tool limit must be a nonnegative integer")
        # Keep the checked-in prompt hash stable for resumable runs while making
        # the active limit explicit to the model through the campaign context.
        prompt=prompt_path.read_text().replace(
            "Use at most four computational tool calls per\ncampaign step, conditioning each decision on the returned observations.",
            "Use at most the configured computational-tool limit per campaign step,\nconditioning each decision on the returned observations.")
        self.client,self.prompt,self.max_tools,self.max_rounds=client,prompt,max_tools,max_rounds

    def select(self,backend,*,target_id:str,target_length:int,trace_path:Path,seed:int):
        trace_path.parent.mkdir(parents=True,exist_ok=True)
        def log(event):
            with trace_path.open("a") as handle:
                handle.write(json.dumps({"time":time.time(),"budget_used":backend.state.used,**event},allow_nan=False)+"\n")
        retained_history=history_messages(backend.state)
        advisory=backend.suggest()
        context={"target_id":target_id,"target_length":target_length,"D":backend.state.dimension,
                 "objective":backend.state.objective,
                 "computational_tool_limit":self.max_tools,
                 "latent_bounds":{"normalized":[0,1],"native_radius":backend.box.radius,
                                  "center_id":"saved_z0","native_shape":list(backend.box.native_shape)},
                 "budget_used":backend.state.used,"budget_remaining":backend.state.remaining,
                 "incumbent":backend.incumbent(),"recent_trials":backend.trials(5),
                 "gp":backend.diagnostics(),"advisory_candidate":advisory}
        current={"role":"user","content":json.dumps(context,allow_nan=False)}
        messages=[{"role":"system","content":self.prompt},*retained_history,current]
        turn_start=len(messages)-1
        log({"event":"turn_start","context":context})
        tool_count,usage,actions=0,[],[]
        for round_index in range(self.max_rounds):
            message,tokens=self.client.complete(messages,TOOLS,seed+round_index)
            usage.append(tokens)
            messages.append(message)
            log({"event":"assistant","message":message,"usage":tokens})
            calls=message.get("tool_calls") or []
            terminal_calls=[c for c in calls if c.get("function",{}).get("name") in {"EVALUATE","STOP"}]
            if terminal_calls and len(calls)!=1:
                raise RuntimeError("EVALUATE or STOP must be the sole tool call; no candidate was evaluated")
            if not calls:
                messages.append({"role":"user","content":"Select one candidate ID using EVALUATE, or terminate with STOP and a reason."})
                continue
            for call in calls:
                name=call.get("function",{}).get("name")
                if name not in {"EVALUATE","STOP"}:
                    if tool_count>=self.max_tools:
                        raise RuntimeError(f"Agent exceeded the configured computational-tool limit ({self.max_tools}); no candidate was evaluated")
                    tool_count+=1
                try:
                    arguments=json.loads(call["function"]["arguments"])
                    validate_action(name,arguments)
                    if name=="STOP":
                        decision={"action":"STOP","reason":arguments["reason"].strip(),"tool_calls":tool_count,
                            "local_search_decisions":sum(a in {"suggest_local","set_search_radius"} for a in actions),
                            "global_resets":actions.count("reset_bounds"),"acquisition_switches":actions.count("set_acquisition"),
                            "accepted_dsp_suggestion":False,"overrode_dsp_suggestion":False,
                            "usage":usage,"usage_complete":all(u is not None for u in usage)}
                        backend.state.set_setting("agent_history_pending",{"budget_used":backend.state.used,
                            "messages":messages[turn_start:]})
                        log({"event":"stop_selected",**decision})
                        return None,decision
                    if name=="EVALUATE":
                        candidate_id=arguments["candidate_id"]
                        backend.state.candidate(candidate_id)
                        if backend.state.was_evaluated(candidate_id):
                            raise ValueError("Candidate has already been evaluated; select a new proposal")
                        accepted=candidate_id==advisory["candidate_id"]
                        metric={"tool_calls":tool_count,"local_search_decisions":sum(a in {"suggest_local","set_search_radius"} for a in actions),
                                "global_resets":actions.count("reset_bounds"),"acquisition_switches":actions.count("set_acquisition"),
                                "accepted_dsp_suggestion":accepted,"overrode_dsp_suggestion":not accepted,
                                "advisory_candidate_id":advisory["candidate_id"],"candidate_id":candidate_id,
                                "usage":usage,"usage_complete":all(u is not None for u in usage)}
                        log({"event":"selected",**metric})
                        backend.state.set_setting("agent_history_pending",{"budget_used":backend.state.used,
                            "messages":messages[turn_start:]})
                        return candidate_id,metric
                    result=getattr(backend,name)(**arguments)
                    actions.append(name)
                except (ValueError,KeyError,TypeError) as exc:
                    result={"error":str(exc)}
                messages.append({"role":"tool","tool_call_id":call["id"],"content":json.dumps(result,allow_nan=False)})
                log({"event":"tool","name":name,"result":result})
        raise RuntimeError("Agent did not select a valid candidate within its bounded turn")
