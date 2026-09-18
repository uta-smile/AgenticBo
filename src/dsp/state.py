"""Persistent backend vectors and an atomic, failure-inclusive evaluation ledger."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
import platform
import numpy as np
import torch

from oracle.objectives import DEFAULT_OBJECTIVE, objective_value, validate_objective
import psutil

class BudgetExhausted(RuntimeError):
    pass


def process_identity(pid: int | None = None) -> dict:
    pid = os.getpid() if pid is None else pid

    if platform.system() == "Linux":
        return {
            "pid": pid,
            "host": socket.gethostname(),
            "boot": Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text().strip(),
            "pid_namespace": os.readlink(f"/proc/{pid}/ns/pid"),
            "start": Path(
                f"/proc/{pid}/stat"
            ).read_text().rsplit(")", 1)[1].split()[19],
        }

    # Windows/macOS fallback
    process = psutil.Process(pid)

    return {
        "pid": pid,
        "host": socket.gethostname(),
        "boot": str(psutil.boot_time()),
        "pid_namespace": None,
        "start": str(process.create_time()),
    }


def process_alive(owner: dict) -> bool | None:
    """Verify the specific owning process, including boot and PID reuse protection."""
    if owner["host"] != socket.gethostname():
        return None
    try:
        if owner.get("pid_namespace") != os.readlink("/proc/self/ns/pid"):
            return None
        if owner["boot"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
            return False
        stat = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] == owner["start"] and stat[0] not in {"Z", "X"}
    except FileNotFoundError:
        return False
    except OSError:
        return None


def checked_result(result: dict, objective: str = DEFAULT_OBJECTIVE) -> dict:
    validate_objective(objective)
    result = {k:v for k,v in result.items() if k not in {"trial_id","candidate_id","status","physical_call","started","completed","owner"}}
    if not isinstance(result.get("valid"), bool):
        raise ValueError("Oracle result needs a boolean validity field")
    if not result["valid"]:
        result.update(tm=0.0, lddt=0.0, objective=0.0, rmsd=None, gdt_ha=None)
    else:
        for key in ("tm", "lddt"):
            if not isinstance(result.get(key), (float, int)) or not 0 <= result[key] <= 1:
                raise ValueError(f"Oracle {key} must lie in [0,1]")
        if result.get("rmsd") is not None and (not math.isfinite(result["rmsd"]) or result["rmsd"] < 0):
            raise ValueError("RMSD must be finite and nonnegative")
        # The backend always computes the declared scalar objective itself.
        result["objective"] = objective_value(result["tm"], result["lddt"], objective)
    json.dumps(result, allow_nan=False)
    return result


class OptimizationState:
    def __init__(self, directory: Path, *, dimension: int, budget: int, metadata: dict):
        self.objective = validate_objective(metadata.get("objective", DEFAULT_OBJECTIVE))
        if dimension < 1 or budget < 1:
            raise ValueError("Positive dimension and budget required")
        self.directory, self.dimension, self.budget = Path(directory), dimension, budget
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "state.sqlite", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT, vector BLOB NOT NULL, info TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate INTEGER UNIQUE NOT NULL REFERENCES candidates(id),
                status TEXT NOT NULL, started REAL NOT NULL, completed REAL,
                owner TEXT, result TEXT, physical_call INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        expected = {"dimension": dimension, "budget": budget, "run": metadata}
        with self.transaction():
            rows = self.db.execute("SELECT key,value FROM metadata").fetchall()
            if rows:
                observed = {row["key"]: json.loads(row["value"]) for row in rows}
                if observed != expected:
                    raise ValueError("Existing run metadata/budget/dimension differs; refusing to reinterpret it")
            else:
                self.db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v, sort_keys=True, allow_nan=False)) for k,v in expected.items()])

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @property
    def used(self):
        return self.db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    @property
    def remaining(self):
        return self.budget - self.used

    def _vector(self, x):
        x = torch.as_tensor(x, dtype=torch.float64, device="cpu").detach()
        if x.shape != (self.dimension,) or not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("Candidate must have D finite normalized coordinates")
        return x.contiguous().numpy().tobytes()

    @staticmethod
    def _index(candidate_id):
        if not isinstance(candidate_id, str) or not re.fullmatch(r"cand_[0-9]+", candidate_id):
            raise ValueError("Expected a backend candidate ID")
        return int(candidate_id[5:])

    def add_candidate(self, x, info: dict | None = None) -> str:
        blob, encoded = self._vector(x), json.dumps(info or {}, allow_nan=False)
        row = self.db.execute("INSERT INTO candidates(vector,info) VALUES (?,?)", (blob, encoded))
        return f"cand_{row.lastrowid:06d}"

    def candidate(self, candidate_id: str) -> torch.Tensor:
        row = self.db.execute("SELECT vector FROM candidates WHERE id=?", (self._index(candidate_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown candidate ID")
        return torch.from_numpy(np.frombuffer(row[0], dtype=np.float64).copy())

    def candidate_info(self, candidate_id):
        row = self.db.execute("SELECT info FROM candidates WHERE id=?", (self._index(candidate_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown candidate ID")
        return json.loads(row[0])

    def was_evaluated(self,candidate_id):
        return self.db.execute("SELECT 1 FROM trials WHERE candidate=?",(self._index(candidate_id),)).fetchone() is not None

    def reserve(self, candidate_id: str) -> int:
        index = self._index(candidate_id)
        with self.transaction():
            if self.get_setting("stop_decision") is not None:
                raise RuntimeError("Campaign has stopped")
            if self.db.execute("SELECT 1 FROM trials WHERE status='running'").fetchone():
                raise RuntimeError("An evaluation is pending; verify its owning process before recovery")
            if self.used >= self.budget:
                raise BudgetExhausted("The exact expensive evaluation budget is exhausted")
            if not self.db.execute("SELECT 1 FROM candidates WHERE id=?", (index,)).fetchone():
                raise ValueError("Unknown candidate ID")
            if self.db.execute("SELECT 1 FROM trials WHERE candidate=?", (index,)).fetchone():
                raise ValueError("Candidate has already consumed an evaluation")
            row = self.db.execute("INSERT INTO trials(candidate,status,started,owner,physical_call) VALUES (?,'running',?,?,1)",
                                  (index, time.time(), json.dumps(process_identity())))
        return row.lastrowid

    def finish(self, trial_id: int, result: dict):
        encoded = json.dumps(checked_result(result, self.objective), allow_nan=False)
        with self.transaction():
            changed = self.db.execute("UPDATE trials SET status='completed',completed=?,result=? WHERE id=? AND status='running'",
                                      (time.time(), encoded, trial_id)).rowcount
            if changed != 1:
                raise ValueError("Trial is missing or already finalized")

    def recover_interrupted(self) -> list[int]:
        recovered = []
        for row in self.db.execute("SELECT id,owner FROM trials WHERE status='running'").fetchall():
            if process_alive(json.loads(row["owner"])) is not False:
                raise RuntimeError("Evaluation owner is live or unverified; refusing to restart or charge another call")
            self.finish(row["id"], {"valid": False, "error": "Owning process verified terminal; interrupted evaluation remains budgeted"})
            recovered.append(row["id"])
        return recovered

    def import_initial(self, x: torch.Tensor, results: list[dict], source_sha256: str):
        if x.shape != (len(results), self.dimension) or not len(results) <= self.budget:
            raise ValueError("Initial observations have the wrong full dimension or exceed budget")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("Shared initialization must have a SHA-256 provenance hash")
        blobs = [self._vector(row) for row in x]
        encoded = [json.dumps(checked_result(r, self.objective), allow_nan=False) for r in results]
        with self.transaction():
            if self.used or self.db.execute("SELECT 1 FROM candidates").fetchone():
                raise ValueError("Initialization may only be imported into an empty run")
            for blob, result in zip(blobs, encoded):
                candidate = self.db.execute("INSERT INTO candidates(vector,info) VALUES (?,?)",
                    (blob, json.dumps({"source": "shared_initial", "source_sha256": source_sha256}))).lastrowid
                self.db.execute("INSERT INTO trials(candidate,status,started,completed,result,physical_call) VALUES (?,'completed',?,?,?,0)",
                                (candidate, time.time(), time.time(), result))
            self.db.execute("INSERT INTO settings VALUES ('initial_source_sha256',?)", (json.dumps(source_sha256),))

    def trials(self, last_n: int | None = None) -> list[dict]:
        rows = self.db.execute("SELECT * FROM trials ORDER BY id").fetchall()
        if last_n is not None:
            if not 1 <= last_n <= 100:
                raise ValueError("last_n must lie between 1 and 100")
            rows = rows[-last_n:]
        return [{"trial_id": f"trial_{r['id']:06d}", "candidate_id": f"cand_{r['candidate']:06d}",
                 "status": r["status"], "physical_call": bool(r["physical_call"]),
                 **(json.loads(r["result"]) if r["result"] else {})} for r in rows]

    def observations(self):
        complete = [t for t in self.trials() if t["status"] == "completed"]
        if not complete:
            return torch.empty((0,self.dimension), dtype=torch.float64), torch.empty((0,1), dtype=torch.float64)
        return (torch.stack([self.candidate(t["candidate_id"]) for t in complete]),
                torch.tensor([[t["objective"]] for t in complete], dtype=torch.float64))

    def incumbent(self):
        complete = [t for t in self.trials() if t["status"] == "completed"]
        return max(complete, key=lambda t:t["objective"]) if complete else None

    def set_setting(self, key: str, value):
        self.db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, allow_nan=False)))

    def record_stop(self, decision):
        if decision.get("action")!="STOP" or not isinstance(decision.get("reason"),str) or not decision["reason"].strip():
            raise ValueError("STOP requires a nonempty reason")
        with self.transaction():
            if self.get_setting("stop_decision") is not None:
                raise ValueError("Campaign already stopped")
            trials=self.trials()
            if not trials or any(t["status"]!="completed" for t in trials):
                raise ValueError("Cannot stop before initialization or while an evaluation is pending")
            valid=[t for t in trials if t["valid"]]
            best=max(valid,key=lambda t:t["objective"]) if valid else None
            result={**decision,"termination":"agent_stop","budget_used":self.used,"budget_remaining":self.remaining,
                    "incumbent":{k:best[k] for k in ("trial_id","candidate_id","objective","tm","valid")} if best else None}
            self.set_setting("stop_decision",result)
        return result

    def get_setting(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def close(self):
        self.db.close()
