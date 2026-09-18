#!/usr/bin/env python3
"""Validate local FASTA/reference pairs and development/test separation."""
import argparse
import json
from pathlib import Path

from targets.manifest import load_manifest, validate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/targets/manifest.csv"))
    parser.add_argument("--poc", action="store_true", help="Require development data and five test targets of 100–200 residues")
    parser.add_argument("--reference-mode",choices=("strict","resolved"),default="strict")
    args = parser.parse_args()
    targets = load_manifest(args.root / args.manifest, args.root)
    if not targets:
        parser.error("No targets registered. Add FASTA/reference pairs to data/targets/manifest.csv")
    results = []
    dev_sequences = {t.sequence for t in targets if t.split == "dev"}
    test_sequences = {t.sequence for t in targets if t.split == "test"}
    if dev_sequences & test_sequences:
        parser.error("Development and test sequences overlap")
    if args.poc:
        test_targets = [t for t in targets if t.split == "test"]
        if not dev_sequences or len(test_targets) != 5:
            parser.error("POC requires a development target and exactly five test targets")
        if len(test_sequences) != 5 or any(not 100 <= t.length <= 200 for t in test_targets):
            parser.error("Test proteins must have distinct sequences and lengths between 100 and 200")
    for target in targets:
        results.append({**target.fingerprint(), **validate_reference(target,reference_mode=args.reference_mode), "provenance": target.provenance})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
