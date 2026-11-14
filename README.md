# Agent Company

Local prototype for a multi-agent company operating loop: a set of role-based
agents that mimic a corporate operating structure and turn a vague objective
into an execution packet.

The first version is deterministic and runs without cloud APIs. That is
intentional. It proves the workflow, artifact model, accountability structure,
and approval gates before adding LLM-backed agents.

## What It Does

`company.py run` creates a timestamped run folder with:

- CEO memo
- Product and market memo
- CTO memo
- CRO memo
- CFO/Risk memo
- COO memo
- Chief of Staff synthesis
- Board packet
- Decision log
- Product brief
- Demo script
- Outreach plan
- Validation thresholds
- Sprint calendar
- Technical plan
- Objection log
- Provider contract
- `tasks.md`
- `tasks.json`
- `run_status.md`

By default, each run is also indexed into SQLite memory at
`runs/company_memory.sqlite3`.

Each role must produce recommendations, risks, asks, and owned tasks. The Chief
of Staff reconciles those into a sprint packet.

## Run It

```bash
git clone https://github.com/Trenn1x/agent-company.git
cd agent-company
python3 company.py run "Build a repeatable local SMB MVP sales loop for Charleston service businesses" --name smb-mvp-sales-loop
```

Run with an explicit provider or memory database:

```bash
python3 company.py run "Build the multi-agent company operating loop" --provider deterministic --memory-db runs/company_memory.sqlite3
```

Inspect the generated folder under:

```bash
runs/
```

List the default roles:

```bash
python3 company.py roles
```

Compare providers on the same objective:

```bash
python3 company.py compare "Build the multi-agent company operating loop" --name provider-comparison
```

By default, `compare` runs `deterministic` and `ollama`, writes both run folders
under one comparison folder, and creates `comparison_report.md` with task,
fallback, and review guidance. For a zero-network smoke test, compare two local
deterministic runs:

```bash
python3 company.py compare "Build the multi-agent company operating loop" --providers deterministic deterministic --no-memory
```

## Catch Plan Regressions

Run a fixed suite of objectives before changing the planning templates. The
offline check fails if a role disappears or a critical (`P0`) task is dropped,
demoted, or reassigned. It also lists changes such as revised deadlines for
review. No model, API key, or database is needed.

```bash
scratch=$(mktemp -d)
python3 company.py snapshot examples/regression/suite.json --output "$scratch/candidate.json"
python3 company.py regress examples/regression/baseline.json "$scratch/candidate.json"
```

The [regression walkthrough](docs/regression.md) shows a deliberate failure,
how to read the GitHub Actions report, and when to update a baseline.

## Tests

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Operate A Run

Check the current task rollup for an existing run:

```bash
python3 company.py status runs/20260710-120000-multi-agent-company
```

The same live rollup is also maintained in `run_status.md` inside the run
folder whenever tasks are created or updated.

Close, carry, or kill a task by its stable task id:

```bash
python3 company.py task runs/20260710-120000-multi-agent-company T01 closed --note "Approved success metric."
```

Create a daily executive review artifact:

```bash
python3 company.py review runs/20260710-120000-multi-agent-company --date 2026-07-10 --note "Focus tomorrow on provider boundary and task lifecycle."
```

Task statuses are deliberately simple: `open`, `closed`, `carried`, and
`killed`. The point is to force operating decisions, not create a project
management system.

## Durable Memory

Print recent indexed runs:

```bash
python3 company.py memory
```

Re-index existing run folders into SQLite memory:

```bash
python3 company.py memory --sync-runs-dir runs
```

The SQLite memory stores run metadata, task state, decisions, and sync events.
Markdown and JSON files remain the source of truth; memory is a queryable ledger
for comparing runs over time.

## Provider Boundary

List provider contracts:

```bash
python3 company.py providers
```

Current provider:

- `deterministic`: local template-backed memos, no network or model calls.
- `ollama`: local Ollama model calls over `http://127.0.0.1:11434`; no cloud API calls.

Use the free local Ollama provider:

```bash
python3 company.py run "Decide the next engineering investment" --provider ollama --model mistral:latest --name ollama-provider-run
```

If Ollama returns incomplete role coverage, the engine keeps the run valid by
calling the model once per role, retrying a failed role once, then falling back
to a deterministic memo only for that role. Fallbacks are marked in the board
packet.

The provider contract is intentionally narrow. Providers receive role/objective
context and return structured memo content. The engine owns artifact writing,
task normalization, memory sync, and approval gates.

## Operating Model

The v1 company has seven roles:

- CEO: thesis, success bar, decision discipline
- Chief Product and Market Officer: ICP, pain, MVP, acceptance criteria
- CTO: architecture, build path, provider boundary
- CRO: pipeline, offer, outreach, conversion path
- CFO/Risk: time, cash, legal exposure, kill criteria
- COO: cadence, owners, deadlines, review rhythm
- Chief of Staff: board packet, task list, decision log

## Human Approval Gates

The system does not send email, post publicly, buy anything, or change external
state. Those actions should stay behind explicit human approval.

## Next Sensible Extensions

- Add memory-aware prompts once provider-backed role memos exist.
- Add A/B comparison reports between deterministic and Ollama runs.
- Add tool permissions per role, for example CRO can draft outreach but cannot
  send it.
