"""Persist completed deliberations and return a bounded, tool-valid history view."""
import json


def finish_history(state, result=None):
    pending=state.get_setting("agent_history_pending")
    if not pending:
        return
    call=pending["messages"][-1]["tool_calls"][0]
    name=call["function"]["name"]
    if result is None and name=="EVALUATE":
        candidate=json.loads(call["function"]["arguments"])["candidate_id"]
        result=next((t for t in state.trials() if t["candidate_id"]==candidate and t["status"]=="completed"),None)
        if result is None:
            return
    if result is None and name=="STOP":
        result=state.get_setting("stop_decision")
    if result is None:
        return
    keys={"trial_id","candidate_id","objective","tm","lddt","rmsd","valid","metrics","error","reason","termination"}
    pending["messages"].append({"role":"tool","tool_call_id":call["id"],
        "content":json.dumps({k:v for k,v in result.items() if k in keys},allow_nan=False)})
    with state.transaction():
        history=state.get_setting("agent_history",[])
        state.set_setting("agent_history",[*history,pending])
        state.set_setting("agent_history_pending",None)


def history_messages(state, max_chars=12000):
    """Full history stays on disk; retain whole recent turns in the LLM context."""
    finish_history(state)
    turns=state.get_setting("agent_history",[])
    retained=[]
    used=0
    for turn in reversed(turns):
        messages=turn["messages"]
        # Old numeric contexts are already represented by current trials/GP state.
        compact=[{"role":"user","content":f"Previous deliberation at evaluation count {turn['budget_used']}:"},*messages[1:]]
        size=len(json.dumps(compact))
        if used+size>max_chars:
            break
        retained.insert(0,compact)
        used+=size
    result=[]
    if len(retained)<len(turns):
        omitted=turns[:len(turns)-len(retained)]
        summaries=[]
        for turn in omitted[-5:]:
            summaries.append({"evaluation_count":turn["budget_used"],"actions":[
                {"name":call["function"]["name"],"arguments":call["function"]["arguments"]}
                for message in turn["messages"] for call in message.get("tool_calls",[])]})
        result.append({"role":"user","content":json.dumps({"older_deliberations":len(omitted),
            "recent_older_actions":summaries,"note":"Full traces are saved; current context supplies measured outcomes."})})
    for turn in retained:
        result.extend(turn)
    return result
