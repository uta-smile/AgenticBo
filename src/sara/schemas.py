import math


def tool(name, description, properties=None, required=None):
    return {"type":"function", "function":{"name":name, "description":description,
        "parameters":{"type":"object", "properties":properties or {}, "required":required or [], "additionalProperties":False}}}


ID = {"type":"string", "description":"Backend candidate ID, never a latent vector"}
TOOLS = [
    tool("suggest","Propose a candidate in all native dimensions inside the active box"),
    tool("suggest_local","Propose a full-D candidate near the incumbent",{
        "center":{"type":"string","enum":["incumbent"]},
        "radius":{"type":"number","minimum":0.000001,"maximum":1}},["radius"]),
    tool("predict","Posterior mean and standard deviation",{"candidate_id":ID},["candidate_id"]),
    tool("acquisition_score","Acquisition score of an existing candidate",{"candidate_id":ID},["candidate_id"]),
    tool("incumbent","Current best observed structural scores and candidate ID"),
    tool("diagnostics","Compact GP diagnostics, never full ARD vectors"),
    tool("trials","Recent observed trials",{"last_n":{"type":"integer","minimum":1,"maximum":20}}),
    tool("set_search_radius","Set the active normalized half-width in every dimension",{
        "radius":{"type":"number","minimum":0.000001,"maximum":1}},["radius"]),
    tool("reset_bounds","Restore the original calibrated full-D box"),
    tool("set_acquisition","Choose LogEI or UCB",{
        "name":{"type":"string","enum":["log_ei","ucb"]},
        "beta":{"type":"number","exclusiveMinimum":0}},["name"]),
    tool("EVALUATE","Commit exactly one candidate ID for the next expensive evaluation",{"candidate_id":ID},["candidate_id"]),
    tool("STOP","Terminate this campaign and return its best feasible observed candidate; spends no evaluation",{
        "reason":{"type":"string","minLength":1,"maxLength":1000}},["reason"]),
]
SCHEMAS = {t["function"]["name"]:t["function"]["parameters"] for t in TOOLS}


def validate_action(name, arguments):
    if name not in SCHEMAS or not isinstance(arguments,dict):
        raise ValueError("Unknown action or non-object arguments")
    schema = SCHEMAS[name]
    if set(arguments)-set(schema["properties"]) or not set(schema["required"]) <= set(arguments):
        raise ValueError("Unexpected or missing tool arguments")
    for key,value in arguments.items():
        prop = schema["properties"][key]
        if prop["type"]=="string" and not isinstance(value,str):
            raise ValueError(f"{key} must be a string")
        if prop["type"]=="string" and (len(value.strip())<prop.get("minLength",0) or len(value)>prop.get("maxLength",float("inf"))):
            raise ValueError(f"{key} has invalid length")
        if prop["type"] in {"number","integer"}:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if prop["type"]=="integer" and not isinstance(value,int):
                raise ValueError(f"{key} must be an integer")
            if ("minimum" in prop and value<prop["minimum"] or "maximum" in prop and value>prop["maximum"]
                or "exclusiveMinimum" in prop and value<=prop["exclusiveMinimum"]):
                raise ValueError(f"{key} is outside its allowed range")
        if "enum" in prop and value not in prop["enum"]:
            raise ValueError(f"{key} is not an allowed choice")
    return arguments
