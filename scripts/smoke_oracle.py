#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from oracle.oracle import StructuralOracle
from targets.manifest import read_fasta


def main():
    parser = argparse.ArgumentParser(description="Score a generated structure against a fixed reference")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--generated-chain", default="A")
    args = parser.parse_args()
    oracle = StructuralOracle(args.reference, read_fasta(args.fasta)[1], args.reference_chain, args.generated_chain)
    print(json.dumps(oracle.score(args.generated), indent=2))


if __name__ == "__main__":
    main()
