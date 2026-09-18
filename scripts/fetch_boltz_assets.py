#!/usr/bin/env python3
"""Fetch official Boltz2 structure weights and molecule assets to the project SSD."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


ASSETS = {
    "boltz2_conf.ckpt": "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
    "mols.tar": "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    record_path = args.cache / "assets.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, url in ASSETS.items():
        if args.checkpoint_only and name != "boltz2_conf.ckpt":
            continue
        path = args.cache / name
        if not path.exists():
            print(f"Downloading {name} to {path}", flush=True)
            part = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as handle:
                while chunk := response.read(8 * 1024 * 1024):
                    handle.write(chunk)
            part.replace(path)
        checksum = digest(path)
        if name in record and record[name]["sha256"] != checksum:
            raise ValueError(f"Saved asset checksum changed: {path}")
        record[name] = {"url": url, "sha256": checksum, "bytes": path.stat().st_size}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Verified {name}: {checksum}", flush=True)
        if name == "mols.tar" and not (args.cache / "molecules_extracted.json").exists():
            with tarfile.open(path) as archive:
                archive.extractall(args.cache, filter="data")
            (args.cache / "molecules_extracted.json").write_text(json.dumps(record[name], indent=2) + "\n")


if __name__ == "__main__":
    main()
