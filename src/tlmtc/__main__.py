"""Run tlmtc as a module.

Enables `python -m tlmtc` to execute the CLI entrypoint.
"""

from tlmtc.cli import app

if __name__ == "__main__":
    app()
