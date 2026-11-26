"""Entry point kept for `run.bat` and `python translator.py`.

The code lives in the `lingo_overlay` package; see docs/ARCHITECTURE.md.
Also runnable as `python -m lingo_overlay`.
"""

from lingo_overlay.app import cli

if __name__ == "__main__":
    cli()
