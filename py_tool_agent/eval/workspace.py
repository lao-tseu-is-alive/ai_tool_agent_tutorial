from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
from pathlib import Path


BACKEND_SIZE_BYTES = 1121
BACKEND_MTIME = dt.datetime(2026, 5, 7, 16, 15)


def prepare_workspace(path: Path) -> Path:
    """Create a clean live-eval workspace with controlled files and metadata."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)
    backend_path = path / "backend.py"
    backend_path.write_text(_backend_content(), encoding="utf-8")
    timestamp = BACKEND_MTIME.timestamp()
    os.utime(backend_path, (timestamp, timestamp))

    return path


def main() -> None:
    """Prepare the live-eval workspace from the command line."""
    parser = argparse.ArgumentParser(description="Prepare a controlled eval workspace.")
    parser.add_argument("path", type=Path, help="Workspace directory to recreate.")
    args = parser.parse_args()

    workspace = prepare_workspace(args.path)
    print(workspace)


def _backend_content() -> str:
    """Return deterministic backend.py content with the expected byte size."""
    base = (
        '"""Controlled Python file used by live evals."""\n'
        "\n"
        "\n"
        "def handler() -> str:\n"
        '    """Return a stable fixture value."""\n'
        '    return "live-eval-backend"\n'
        "\n"
    )
    padding_size = BACKEND_SIZE_BYTES - len(base.encode("utf-8"))
    if padding_size < 0:
        raise ValueError("backend fixture base content is larger than target size")

    return base + ("#" * padding_size)


if __name__ == "__main__":
    main()
