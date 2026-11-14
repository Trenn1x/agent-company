"""Compare validated Agent Company regression snapshots."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Callable, Hashable


Task = dict[str, str]
TaskPair = tuple[Task, Task]
Finding = dict[str, str]


def _finding(severity: str, code: str, message: str, **details: str) -> Finding:
    return {"severity": severity, "code": code, "message": message, **details}


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _owner(task: Task) -> str:
    return _normalized(task["owner"])


def _priority(task: Task) -> str:
    # The board packet treats only the exact label P0 as critical.
    return task["priority"]


def _owner_priority(task: Task) -> tuple[str, str]:
    return _owner(task), _priority(task)


def _task_details(task: Task) -> tuple[str, str, str, str]:
    return (
        *_owner_priority(task),
        _normalized(task["due"]),
        _normalized(task["definition_of_done"]),
    )


def _group_tasks(tasks: list[Task]) -> dict[str, list[Task]]:
    groups = defaultdict(list)
    for task in tasks:
        groups[_normalized(task["task"])].append(task)
    for group in groups.values():
        # Raw values break ties between semantically equivalent task spellings.
        group.sort(key=lambda task: (
            _task_details(task), task["owner"], task["task"],
            task["due"], task["definition_of_done"],
        ))
    return groups


def _pair_by(
    baseline: list[Task], candidate: list[Task], *keys: Callable[[Task], Hashable]
) -> tuple[list[TaskPair], list[Task], list[Task]]:
    pairs = []
    for key in keys:
        unmatched_indexes = defaultdict(deque)
        for index, task in enumerate(candidate):
            unmatched_indexes[key(task)].append(index)

        unmatched_baseline = []
        matched_indexes = set()
        for task in baseline:
            indexes = unmatched_indexes[key(task)]
            if indexes:
                index = indexes.popleft()
                pairs.append((task, candidate[index]))
                matched_indexes.add(index)
            else:
                unmatched_baseline.append(task)
        baseline = unmatched_baseline
        candidate = [task for index, task in enumerate(candidate) if index not in matched_indexes]
    return pairs, baseline, candidate


def _pair_task_group(
    baseline: list[Task], candidate: list[Task]
) -> tuple[list[TaskPair], list[Task], list[Task]]:
    # Reserve surviving copies before interpreting unmatched copies as changed.
    # A candidate copy can satisfy only one baseline expectation.
    unchanged, baseline, candidate = _pair_by(
        baseline, candidate,
        _task_details,
        lambda task: (_owner_priority(task), _normalized(task["due"])),
        lambda task: (_owner_priority(task), _normalized(task["definition_of_done"])),
        _owner_priority,
    )
    critical = [task for task in baseline if _priority(task) == "P0"]
    ordinary = [task for task in baseline if _priority(task) != "P0"]

    critical_pairs, missing_critical, candidate = _pair_by(
        critical, candidate, _owner, _priority, lambda task: None,
    )
    ordinary_pairs, missing_ordinary, added = _pair_by(
        ordinary, candidate, _owner, _priority, lambda task: None,
    )
    return (
        unchanged + critical_pairs + ordinary_pairs,
        missing_critical + missing_ordinary,
        added,
    )


def _task_changes(before: Task, after: Task) -> list[Finding]:
    findings = []
    critical = _priority(before) == "P0"
    description = before["task"]
    fields = (
        ("priority", "priority", "p0_task_demoted", _priority),
        ("owner", "owner", "p0_owner_changed", _owner),
        ("due", "deadline", None, lambda task: _normalized(task["due"])),
        ("definition_of_done", "definition of done", None,
         lambda task: _normalized(task["definition_of_done"])),
    )
    for key, label, critical_code, comparable in fields:
        if comparable(before) == comparable(after):
            continue
        code = critical_code if critical else None
        prefix = "P0 task" if critical else "Task"
        findings.append(_finding(
            "blocking" if code else "review",
            code or f"{key}_changed",
            f"{prefix} {description!r} changed {label} from {before[key]!r} to {after[key]!r}.",
            task=description, field=key, before=before[key], after=after[key],
        ))
    return findings


def _compare_tasks(baseline: list[Task], candidate: list[Task]) -> list[Finding]:
    findings = []
    old_groups, new_groups = _group_tasks(baseline), _group_tasks(candidate)
    for description in sorted(old_groups.keys() | new_groups.keys()):
        pairs, missing, added = _pair_task_group(
            old_groups.get(description, []), new_groups.get(description, []),
        )
        for before, after in pairs:
            findings.extend(_task_changes(before, after))
        for task in missing:
            critical = _priority(task) == "P0"
            findings.append(_finding(
                "blocking" if critical else "review",
                "p0_task_removed" if critical else "task_removed",
                f"{task['priority']} task {task['task']!r} owned by {task['owner']!r} is missing.",
                task=task["task"], owner=task["owner"], priority=task["priority"],
            ))
        for task in added:
            findings.append(_finding(
                "review", "task_added",
                f"Added {task['priority']} task {task['task']!r} for {task['owner']!r}.",
                task=task["task"], owner=task["owner"], priority=task["priority"],
            ))
    return findings


def _compare_roles(baseline: list[dict], candidate: list[dict]) -> list[Finding]:
    before = {role["id"]: role["fallback"] for role in baseline}
    after = {role["id"]: role["fallback"] for role in candidate}
    findings = []
    for role in sorted(before.keys() | after.keys()):
        if role not in after:
            findings.append(_finding(
                "blocking", "role_missing", f"Expected role {role!r} is missing.", role=role,
            ))
            continue
        if role not in before:
            findings.append(_finding("review", "role_added", f"Added role {role!r}.", role=role))
        if after[role] and not before.get(role, False):
            findings.append(_finding(
                "blocking", "fallback_added", f"Role {role!r} fell back to a default memo.",
                role=role,
            ))
        elif before.get(role, False) and not after[role]:
            findings.append(_finding(
                "review", "fallback_resolved", f"Role {role!r} no longer uses a fallback memo.",
                role=role,
            ))
    return findings


def _compare_case(baseline: dict | None, candidate: dict | None) -> tuple[bool, list[Finding]]:
    if baseline is None:
        return False, [_finding("review", "scenario_added", "Added a new scenario.")]
    if candidate is None:
        return False, [_finding("blocking", "scenario_missing", "Expected scenario is missing.")]
    if baseline["scenario_sha256"] != candidate["scenario_sha256"]:
        return False, [_finding(
            "blocking", "scenario_changed",
            "The scenario inputs changed. Restore the baseline inputs or review a new baseline.",
            before=baseline["scenario_sha256"], after=candidate["scenario_sha256"],
        )]
    return True, (
        _compare_roles(baseline["roles"], candidate["roles"])
        + _compare_tasks(baseline["tasks"], candidate["tasks"])
    )


def _findings_status(findings: list[Finding]) -> str:
    if any(finding["severity"] == "blocking" for finding in findings):
        return "failed"
    return "review" if findings else "passed"


def compare_snapshots(baseline: dict, candidate: dict) -> dict:
    """Compare scenarios by ID and inputs, then roles and tasks by semantic identity."""
    before = {case["id"]: case for case in baseline["cases"]}
    after = {case["id"]: case for case in candidate["cases"]}
    findings = []
    if baseline["provider"] != candidate["provider"]:
        findings.append(_finding(
            "review", "provider_changed", "The snapshot provider changed.",
            before=baseline["provider"], after=candidate["provider"],
        ))

    cases = []
    counts = {"blocking": 0, "review": len(findings)}
    compared_cases = 0
    for case_id in sorted(before.keys() | after.keys()):
        compared, case_findings = _compare_case(before.get(case_id), after.get(case_id))
        compared_cases += int(compared)
        case_findings.sort(key=lambda finding: finding["severity"])
        for finding in case_findings:
            counts[finding["severity"]] += 1
        cases.append({
            "id": case_id, "status": _findings_status(case_findings), "findings": case_findings,
        })

    return {
        "schema_version": 1,
        "passed": counts["blocking"] == 0,
        "baseline_provider": baseline["provider"],
        "candidate_provider": candidate["provider"],
        "summary": {**counts, "total_cases": len(cases), "compared_cases": compared_cases},
        "findings": findings,
        "cases": cases,
    }


def _markdown_cell(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|~!])", r"\\\1", " ".join(value.split()))


def render_report(report: dict) -> str:
    """Render the same findings used to set the gate's exit status."""
    summary = report["summary"]
    result = "PASS" if report["passed"] else "FAIL"
    providers = " → ".join(_markdown_cell(report[key]) for key in (
        "baseline_provider", "candidate_provider",
    ))
    lines = [
        "# Agent Company Regression Report", "",
        f"**{result}** · {summary['blocking']} blocking · {summary['review']} for review", "",
        f"Compared {summary['compared_cases']} of {summary['total_cases']} scenarios ({providers}).",
        "", "## Scenarios", "",
        "| Scenario | Status | Blocking | Review |", "| --- | --- | ---: | ---: |",
    ]
    for case in report["cases"]:
        blocking = sum(finding["severity"] == "blocking" for finding in case["findings"])
        review = len(case["findings"]) - blocking
        lines.append(f"| {case['id']} | {case['status'].title()} | {blocking} | {review} |")

    entries = [("All scenarios", finding) for finding in report["findings"]]
    for case in report["cases"]:
        entries.extend((case["id"], finding) for finding in case["findings"])
    if entries:
        lines.extend([
            "", "## Findings", "", "| Scenario | Severity | Finding |",
            "| --- | --- | --- |",
        ])
        entries.sort(key=lambda entry: entry[1]["severity"])
        for case_id, finding in entries:
            message = _markdown_cell(finding["message"])
            lines.append(f"| {case_id} | {finding['severity'].title()} | {message} |")
    return "\n".join(lines) + "\n"
