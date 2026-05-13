from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any


def prepare_workspace(path: Path, scenarios_path: Path | None = None) -> Path:
    """Create a clean live-eval workspace from the scenario workspace config."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)
    workspace = _load_workspace_config(scenarios_path)
    reference_time = _reference_time(workspace) or dt.datetime.now()
    files = workspace.get("files", [])
    if not isinstance(files, list):
        raise ValueError("workspace.files must be a list")

    for file_config in files:
        if not isinstance(file_config, dict):
            raise ValueError("workspace.files entries must be objects")
        _create_file(path, file_config, reference_time)

    return path


def main() -> None:
    """Prepare the live-eval workspace from the command line."""
    parser = argparse.ArgumentParser(description="Prepare a controlled eval workspace.")
    parser.add_argument("path", type=Path, help="Workspace directory to recreate.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/eval_scenarios.json"),
        help="Scenario JSON file containing the workspace config.",
    )
    args = parser.parse_args()

    workspace = prepare_workspace(args.path, args.scenarios)
    print(workspace)


def _load_workspace_config(scenarios_path: Path | None) -> dict[str, Any]:
    """Load the top-level workspace config from a scenario JSON file."""
    if scenarios_path is None:
        return {}

    data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario file must be an object with an optional workspace")

    workspace = data.get("workspace", {})
    if not isinstance(workspace, dict):
        raise ValueError("workspace config must be an object")

    return workspace


def _create_file(
    workspace_path: Path,
    file_config: dict[str, Any],
    reference_time: dt.datetime | None,
) -> None:
    """Create one configured file and apply requested metadata."""
    if "path" not in file_config:
        raise ValueError("workspace file entries require a path")

    relative_path = _safe_relative_path(file_config["path"])
    file_path = workspace_path / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = file_config.get("content", "")
    if not isinstance(content, str):
        raise ValueError(f"{relative_path}: content must be a string")

    target_size = file_config.get("size_bytes")
    if target_size is not None:
        content = _content_with_size(relative_path, content, target_size)

    file_path.write_text(content, encoding="utf-8")

    modified_at = file_config.get("modified_at")
    if modified_at is not None:
        timestamp = _resolve_modified_at(modified_at, reference_time).timestamp()
        os.utime(file_path, (timestamp, timestamp))


def _content_with_size(path: Path, content: str, size_bytes: int) -> str:
    """Pad configured content so the resulting UTF-8 file has an exact size."""
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"{path}: size_bytes must be a non-negative integer")

    padding_size = size_bytes - len(content.encode("utf-8"))
    if padding_size < 0:
        raise ValueError(f"{path}: content is larger than target size")

    return content + ("#" * padding_size)


def _safe_relative_path(value: str) -> Path:
    """Return a safe workspace-relative path."""
    if not isinstance(value, str):
        raise ValueError("workspace file path must be a string")

    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"workspace file path must be relative and safe: {value!r}")

    return path


def _reference_time(workspace: dict[str, Any]) -> dt.datetime | None:
    """Return the datetime used for relative workspace dates."""
    value = workspace.get("reference_time")
    if value is None:
        return None

    return dt.datetime.fromisoformat(value)


def _resolve_modified_at(
    value: str | dict[str, Any],
    reference_time: dt.datetime | None,
) -> dt.datetime:
    """Resolve absolute or reference-relative modification time config."""
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value)

    if not isinstance(value, dict):
        raise ValueError("modified_at must be an ISO datetime or an object")

    date_value = value.get("date", "today")
    time_value = value.get("time")
    if reference_time is None:
        raise ValueError("relative modified_at values require a reference_time")

    date = _resolve_date(date_value, reference_time)
    time = (
        dt.time.fromisoformat(time_value)
        if time_value is not None
        else reference_time.time().replace(microsecond=0)
    )

    return dt.datetime.combine(date, time)


def _resolve_date(value: str, reference_time: dt.datetime) -> dt.date:
    """Resolve a relative date keyword or ISO date."""
    reference_date = reference_time.date()
    offsets = {
        "yesterday": -1,
        "today": 0,
        "tomorrow": 1,
    }

    if value in offsets:
        return reference_date + dt.timedelta(days=offsets[value])

    return dt.date.fromisoformat(value)


if __name__ == "__main__":
    main()
