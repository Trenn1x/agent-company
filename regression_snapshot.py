"""Read regression fixtures and capture stable summaries of deterministic runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from company import CompanyEngine


SCHEMA_VERSION = 1
CASE_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
COMPONENT_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
SHA256 = re.compile(r"[a-f0-9]{64}")
TASK_FIELDS = ("owner", "priority", "task", "due", "definition_of_done")


class InputError(ValueError):
    """A suite or snapshot fails to conform to its schema."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InputError(f"invalid JSON constant {value!r}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, RecursionError) as exc:
        raise InputError(f"{path}: {exc}") from exc


def _object(
    value: Any,
    location: str,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{location} must be an object")
    for key in required:
        if key not in value:
            raise InputError(f"{location} is missing {key!r}")
    for key in value:
        if key not in required and key not in optional:
            raise InputError(f"{location} has unexpected field {key!r}")
    return value


def _version(document: dict) -> None:
    version = document["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise InputError(f"schema_version must be the integer {SCHEMA_VERSION}")


def _array(value: Any, location: str) -> list:
    if not isinstance(value, list):
        raise InputError(f"{location} must be an array")
    return value


def _matches(value: Any, location: str, pattern: re.Pattern[str], description: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise InputError(f"{location} must be {description}")
    return value


def _string(value: Any, location: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        description = "a string" if allow_blank else "a nonblank string"
        raise InputError(f"{location} must be {description}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputError(f"{location} contains invalid UTF-8 text") from exc
    return value


def _unique_id(value: str, location: str, seen: set[str]) -> None:
    if value in seen:
        raise InputError(f"{location} duplicates ID {value!r}")
    seen.add(value)


def _validate_suite(value: Any) -> dict:
    suite = _object(value, "suite", ("schema_version", "scenarios"))
    _version(suite)
    scenarios = _array(suite["scenarios"], "scenarios")
    if not scenarios:
        raise InputError("scenarios must contain at least one scenario")

    normalized = []
    seen = set()
    for index, raw in enumerate(scenarios):
        location = f"scenarios[{index}]"
        scenario = _object(raw, location, ("id", "objective"), ("context", "days"))
        case_id = _matches(scenario["id"], f"{location}.id", CASE_ID, "a lowercase slug")
        _unique_id(case_id, location, seen)
        objective = _string(scenario["objective"], f"{location}.objective").strip()
        context = _string(scenario.get("context", ""), f"{location}.context", allow_blank=True)
        days = scenario.get("days", 7)
        if type(days) is not int or days < 1:
            raise InputError(f"{location}.days must be a positive integer")
        normalized.append({"id": case_id, "objective": objective, "context": context, "days": days})
    normalized.sort(key=lambda scenario: scenario["id"])
    return {"schema_version": SCHEMA_VERSION, "scenarios": normalized}


def load_suite(path: Path) -> dict:
    """Read a v1 suite, fill in its defaults, and sort scenarios by ID."""
    value = _read_json(path)
    try:
        return _validate_suite(value)
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from exc


def _validate_snapshot(value: Any) -> dict:
    snapshot = _object(value, "snapshot", ("schema_version", "provider", "cases"))
    _version(snapshot)
    _matches(snapshot["provider"], "provider", COMPONENT_ID, "a lowercase provider ID")
    case_ids = set()
    for index, raw in enumerate(_array(snapshot["cases"], "cases")):
        location = f"cases[{index}]"
        case = _object(raw, location, ("id", "scenario_sha256", "roles", "tasks"))
        case_id = _matches(case["id"], f"{location}.id", CASE_ID, "a lowercase slug")
        _unique_id(case_id, location, case_ids)
        _matches(
            case["scenario_sha256"],
            f"{location}.scenario_sha256",
            SHA256,
            "a 64-character lowercase hexadecimal digest",
        )

        role_ids = set()
        for role_index, raw_role in enumerate(_array(case["roles"], f"{location}.roles")):
            role_path = f"{location}.roles[{role_index}]"
            role = _object(raw_role, role_path, ("id", "fallback"))
            role_id = _matches(role["id"], f"{role_path}.id", COMPONENT_ID, "a lowercase role ID")
            _unique_id(role_id, role_path, role_ids)
            if not isinstance(role["fallback"], bool):
                raise InputError(f"{role_path}.fallback must be a boolean")

        for task_index, raw_task in enumerate(_array(case["tasks"], f"{location}.tasks")):
            task_path = f"{location}.tasks[{task_index}]"
            task = _object(raw_task, task_path, TASK_FIELDS)
            for field in TASK_FIELDS:
                _string(task[field], f"{task_path}.{field}")
    return snapshot


def load_snapshot(path: Path) -> dict:
    """Read and validate a v1 snapshot."""
    value = _read_json(path)
    try:
        return _validate_snapshot(value)
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from exc


def capture_suite(suite: dict) -> dict:
    """Run a suite locally and summarize its stable role and task data."""
    scenarios = _validate_suite(suite)["scenarios"]
    engine = CompanyEngine(provider="deterministic")
    cases = []
    with TemporaryDirectory(prefix="agent-company-regression-") as directory:
        for index, scenario in enumerate(scenarios):
            inputs = {key: scenario[key] for key in ("objective", "context", "days")}
            canonical = json.dumps(
                inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            # A separate root per case avoids collisions from timed or truncated run names.
            run = engine.run(
                **inputs,
                run_name="capture",
                runs_dir=Path(directory) / str(index),
                use_memory=False,
            )
            cases.append({
                "id": scenario["id"],
                "scenario_sha256": hashlib.sha256(canonical).hexdigest(),
                "roles": [
                    {"id": memo.role.role_id, "fallback": memo.fallback_used}
                    for memo in sorted(run.memos, key=lambda memo: memo.role.role_id)
                ],
                "tasks": [
                    {field: item[field] for field in TASK_FIELDS if field in item}
                    for item in run.tasks
                ],
            })
    return _validate_snapshot({
        "schema_version": SCHEMA_VERSION,
        "provider": "deterministic",
        "cases": cases,
    })
