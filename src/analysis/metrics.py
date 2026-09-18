import numpy as np


def best_curve(trials,key):
    return np.maximum.accumulate([trial[key] for trial in trials])


def evaluations_to_threshold(trials,threshold):
    return next((i+1 for i,t in enumerate(trials) if t["tm"]>=threshold),None)


def agent_metrics(trials,stop=None):
    decisions=[t["agent_decision"] for t in trials if "agent_decision" in t]
    result={"agent_evaluations":len(decisions),"stopped":stop is not None}
    if stop is not None:
        decisions.append(stop)
    for key in ("tool_calls","local_search_decisions","global_resets","acquisition_switches",
                "accepted_dsp_suggestion","overrode_dsp_suggestion"):
        result[key]=sum(d[key] for d in decisions)
    usage=[u for d in decisions for u in d["usage"]]
    result["token_usage_complete"]=bool(decisions) and all(d["usage_complete"] for d in decisions) and all(
        isinstance(u,dict) and isinstance(u.get("total_tokens"),int) and u["total_tokens"]>=0 for u in usage)
    result["tokens"]=sum(u["total_tokens"] for u in usage) if result["token_usage_complete"] else None
    improvements=[d.get("improvement_after_override") for d in decisions if d["overrode_dsp_suggestion"]]
    result["improvement_after_override_total"]=sum(v for v in improvements if v is not None)
    result["tools_per_agent_evaluation"]=result["tool_calls"]/result["agent_evaluations"] if result["agent_evaluations"] else None
    return result
