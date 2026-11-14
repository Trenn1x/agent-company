# Agent regression checks

Agent Company turns objectives into plans. The regression check runs a fixed suite of objectives and detects changes such as a missing executive role, a new provider fallback, or a critical task that lost its owner or priority.

## Run the example

From the repository root, use Python 3.12 and a POSIX shell. The commands use only the Python standard library. Capture uses the deterministic provider and generates each run in a temporary directory with SQLite memory disabled.

```bash
scratch=$(mktemp -d)
python3 company.py snapshot examples/regression/suite.json --output "$scratch/candidate.json"
python3 company.py regress examples/regression/baseline.json "$scratch/candidate.json"
```

The [example suite](../examples/regression/suite.json) covers a hiring work sample, an internal agent workflow, and a local service business pilot. Each scenario has a stable `id` and an `objective`. Set `context` to include additional instructions and `days` to change the seven-day sprint default. Changing the objective, context, or sprint length can alter what the engine is supposed to generate. The comparison reports changed inputs before evaluating the resulting tasks.

Exit code `0` means there are no blocking findings, `1` means the report contains a regression, and `2` means the command or an input file is invalid. The report defaults to Markdown. Add `--format json` for scripts and `--output PATH` to save it to a file.

## See a critical task get blocked

Run this after the example to lower one `P0` task to `P1` in a copy of the candidate.

```bash
python3 - "$scratch/candidate.json" "$scratch/demoted.json" <<'PY'
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
snapshot = json.loads(source.read_text(encoding="utf-8"))
case = next(case for case in snapshot["cases"] if case["id"] == "internal-agent-loop")
task = next(task for task in case["tasks"] if task["owner"] == "CTO" and task["priority"] == "P0")
task["priority"] = "P1"
output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
PY
python3 company.py regress examples/regression/baseline.json "$scratch/demoted.json"
```

The report names `internal-agent-loop` and the CTO task to ship the task lifecycle and daily review commands. The command exits `1`. A missing baseline scenario or role, a new fallback, and removal or reassignment of a baseline `P0` task also block the check. Changes such as an added task or a revised deadline appear as review findings without failing the check. Use human review to judge whether a plan is useful and whether intentional changes are an improvement.

Task matching ignores case and whitespace in descriptions and owners. It counts each duplicate separately, preserves punctuation, and treats only the exact priority `P0` as critical, just like the board packet. If you rewrite a task description, the report shows the old task as removed and the new one as added.

## Review or refresh the baseline

The [GitHub Actions workflow](../.github/workflows/regression.yml) runs the same check on pull requests and pushes to `main`. Open the **Agent regression** workflow run and select **Summary** to read the Markdown report for a passing or blocked check. A baseline change is visible as a normal file diff in the pull request.

Compare a fresh capture with the checked-in baseline first. If a finding is unintended, fix the code and rerun the check. For an intentional behavior or scenario change, review the findings, replace the baseline using the command below, and explain the expected change in the pull request.

```bash
python3 company.py snapshot examples/regression/suite.json --output examples/regression/baseline.json
git diff -- examples/regression/suite.json examples/regression/baseline.json
```
