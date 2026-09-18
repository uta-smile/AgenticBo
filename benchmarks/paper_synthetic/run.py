"""Compatibility entry point; implementation lives in agenticbo."""
import sys
from agenticbo.cli import main

if __name__ == "__main__":
    main(["benchmark", *sys.argv[1:]])
