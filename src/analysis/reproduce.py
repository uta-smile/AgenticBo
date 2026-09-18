"""Compare fresh repeated experiments by backend vectors, scores, and decisions."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from analysis.aggregate import load_runs


def compare(first:Path,second:Path,atol=1e-10):
    a,b=load_runs(first),load_runs(second)
    def key(run):
        return run["target_id"],run["seed"],run["method"]
    left,right={key(r):r for r in a},{key(r):r for r in b}
    if set(left)!=set(right):
        raise ValueError("Repeated experiments must contain the same target/seed/method cells")
    reports=[]
    for identity,run in left.items():
        other=right[identity]
        vectors=[]
        for item in (run,other):
            with sqlite3.connect((item["directory"]/"state.sqlite").resolve().as_uri()+"?mode=ro",uri=True) as db:
                vectors.append(np.stack([np.frombuffer(row[0],dtype=np.float64) for row in db.execute(
                    "SELECT candidates.vector FROM trials JOIN candidates ON candidates.id=trials.candidate ORDER BY trials.id")]))
        if vectors[0].shape!=vectors[1].shape:
            raise ValueError("Repeated experiment vector shapes differ")
        vector_error=float(np.max(np.abs(vectors[0]-vectors[1])))
        score_error=max(abs(t[k]-u[k]) for t,u in zip(run["trials"],other["trials"]) for k in ("tm","lddt","objective"))
        valid_equal=all(t["valid"]==u["valid"] for t,u in zip(run["trials"],other["trials"]))
        decisions_equal=all(t.get("agent_decision",{}).get("candidate_id")==u.get("agent_decision",{}).get("candidate_id")
                            for t,u in zip(run["trials"],other["trials"]))
        reports.append({"target_id":identity[0],"seed":identity[1],"method":identity[2],
            "max_normalized_vector_difference":vector_error,"max_score_difference":score_error,
            "validity_equal":valid_equal,"selected_candidate_ids_equal":decisions_equal,
            "passed":vector_error<=atol and score_error<=atol and valid_equal and decisions_equal})
    return {"absolute_tolerance":atol,"passed":all(r["passed"] for r in reports),"runs":reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first",type=Path)
    parser.add_argument("second",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=compare(args.first,args.second)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
