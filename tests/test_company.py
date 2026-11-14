import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from company import (
    CompanyEngine,
    OllamaMemoProvider,
    ROLES,
    classify_objective,
    compare_provider_runs,
    create_daily_review,
    extract_json_object,
    fallback_roles,
    render_status_report,
    render_memory_summary,
    slugify,
    sync_run_to_memory,
    sync_runs_dir_to_memory,
    update_task_status,
)


class CompanyEngineTests(unittest.TestCase):
    def test_slugify_keeps_safe_name(self):
        self.assertEqual(slugify("SMB MVP Sales Loop!"), "smb-mvp-sales-loop")

    def test_classify_objective_flags_agentic_work(self):
        flags = classify_objective("Build an agent automation workflow for local SMB sales")
        self.assertTrue(flags["agentic"])
        self.assertTrue(flags["local_smb"])
        self.assertTrue(flags["revenue"])

    def test_local_file_backed_objective_is_not_local_smb(self):
        flags = classify_objective("Build a local file-backed multi-agent company operating loop")
        self.assertTrue(flags["agentic"])
        self.assertFalse(flags["local_smb"])

    def test_run_writes_core_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CompanyEngine().run(
                "Build an agent company to validate local SMB MVP sales",
                runs_dir=Path(tmp),
                run_name="demo",
            )
            self.assertTrue((result.run_dir / "board_packet.md").exists())
            self.assertTrue((result.run_dir / "tasks.md").exists())
            self.assertTrue((result.run_dir / "tasks.json").exists())
            self.assertTrue((result.run_dir / "run_status.md").exists())
            self.assertTrue((result.run_dir / "provider_contract.md").exists())
            self.assertTrue((result.run_dir / "decision_log.md").exists())
            self.assertTrue((result.run_dir / "product_brief.md").exists())
            self.assertTrue((result.run_dir / "outreach_plan.md").exists())
            self.assertTrue((result.run_dir / "validation_thresholds.md").exists())
            self.assertTrue((result.run_dir / "sprint_calendar.md").exists())

            tasks = json.loads((result.run_dir / "tasks.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(tasks), 8)
            self.assertTrue(all("owner" in item for item in tasks))
            self.assertTrue(all("id" in item for item in tasks))
            self.assertTrue(all(item["status"] == "open" for item in tasks))
            self.assertTrue(any(item["priority"] == "P0" for item in tasks))
            self.assertFalse(any("Ship the local run engine" in item["task"] for item in tasks))

            board_packet = (result.run_dir / "board_packet.md").read_text(encoding="utf-8")
            self.assertIn("Board Packet", board_packet)
            self.assertIn("Success Metric", board_packet)
            self.assertIn("Approval Gates", board_packet)
            self.assertIn("T01", board_packet)

            outreach_plan = (result.run_dir / "outreach_plan.md").read_text(encoding="utf-8")
            self.assertNotIn("[archetype]", outreach_plan)

            provider_contract = (result.run_dir / "provider_contract.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Provider: deterministic", provider_contract)
            self.assertIn("Forbidden Side Effects", provider_contract)

    def test_compare_provider_runs_writes_report_and_child_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparison = compare_provider_runs(
                "Build a local file-backed multi-agent company operating loop",
                runs_dir=Path(tmp),
                run_name="provider-check",
                providers=["deterministic", "deterministic"],
                use_memory=False,
            )

            self.assertTrue(comparison.comparison_dir.exists())
            self.assertTrue(comparison.report_path.exists())
            self.assertEqual(len(comparison.runs), 2)
            self.assertIsNone(comparison.memory_db)
            self.assertTrue(all((run.run_dir / "board_packet.md").exists() for run in comparison.runs))
            self.assertTrue(all((run.run_dir / "tasks.json").exists() for run in comparison.runs))

            report = comparison.report_path.read_text(encoding="utf-8")
            self.assertIn("Provider Comparison Report", report)
            self.assertIn("## Runs", report)
            self.assertIn("## P0 Task Comparison", report)
            self.assertIn("deterministic vs deterministic", report)
            self.assertIn("provider-check-01-deterministic", report)
            self.assertIn("provider-check-02-deterministic", report)

    def test_extract_json_object_accepts_wrapped_json(self):
        payload = extract_json_object("Here is JSON:\n{\"memos\": []}\nDone")
        self.assertEqual(payload, {"memos": []})

    def test_task_status_update_rewrites_task_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CompanyEngine().run(
                "Build a multi-agent company operating loop",
                runs_dir=Path(tmp),
                run_name="ops-loop",
            )
            updated = update_task_status(result.run_dir, "1", "closed", "Success metric accepted.")
            self.assertEqual(updated["id"], "T01")
            self.assertEqual(updated["status"], "closed")

            tasks = json.loads((result.run_dir / "tasks.json").read_text(encoding="utf-8"))
            self.assertEqual(tasks[0]["status"], "closed")
            self.assertEqual(tasks[0]["status_note"], "Success metric accepted.")

            tasks_md = (result.run_dir / "tasks.md").read_text(encoding="utf-8")
            self.assertIn("- Status: closed", tasks_md)
            self.assertIn("Success metric accepted.", tasks_md)

            run_status = (result.run_dir / "run_status.md").read_text(encoding="utf-8")
            self.assertIn("closed: 1", run_status)

    def test_status_and_review_reports_include_rollup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CompanyEngine().run(
                "Build a multi-agent company operating loop",
                runs_dir=Path(tmp),
                run_name="ops-loop",
            )
            update_task_status(result.run_dir, "T01", "closed", "Done.")
            tasks = json.loads((result.run_dir / "tasks.json").read_text(encoding="utf-8"))

            report = render_status_report(result.run_dir, tasks)
            self.assertIn("Task rollup", report)
            self.assertIn("closed: 1", report)

            review_path = create_daily_review(result.run_dir, "2026-07-10", "Daily review note.")
            review = review_path.read_text(encoding="utf-8")
            self.assertIn("Daily Executive Review - 2026-07-10", review)
            self.assertIn("Daily review note.", review)

    def test_agentic_company_run_uses_internal_proof_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CompanyEngine().run(
                "Build a local file-backed multi-agent company operating loop",
                runs_dir=Path(tmp),
                run_name="agentic-company",
            )
            board_packet = (result.run_dir / "board_packet.md").read_text(encoding="utf-8")
            self.assertIn("complete one full operating loop", board_packet)
            self.assertIn("task lifecycle commands", board_packet)
            self.assertNotIn("Charleston-area owner", board_packet)

            outreach_plan = (result.run_dir / "outreach_plan.md").read_text(encoding="utf-8")
            self.assertIn("Internal Proof Plan", outreach_plan)
            self.assertIn("Run the company on one real objective.", outreach_plan)

            technical_plan = (result.run_dir / "technical_plan.md").read_text(encoding="utf-8")
            self.assertIn("Future LLM Provider Boundary", technical_plan)

            thresholds = (result.run_dir / "validation_thresholds.md").read_text(encoding="utf-8")
            self.assertIn("The operator produces this proof signal", thresholds)
            self.assertNotIn("prospect gives real workflow data", thresholds)

            friction_log = (result.run_dir / "objection_log.md").read_text(encoding="utf-8")
            self.assertIn("Operator Friction Log", friction_log)

    def test_memory_indexes_runs_and_task_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            memory_db = Path(tmp) / "memory.sqlite3"
            result = CompanyEngine().run(
                "Build a local file-backed multi-agent company operating loop",
                runs_dir=runs_dir,
                run_name="agentic-company",
                memory_db=memory_db,
            )
            self.assertTrue(memory_db.exists())

            summary = render_memory_summary(memory_db)
            self.assertIn("agentic-company", summary)
            self.assertIn("Provider: deterministic", summary)
            self.assertIn("open 11", summary)

            update_task_status(result.run_dir, "T01", "closed", "Done.", memory_db)
            summary = render_memory_summary(memory_db)
            self.assertIn("closed 1", summary)

    def test_memory_can_bulk_sync_existing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            memory_db = Path(tmp) / "memory.sqlite3"
            CompanyEngine().run(
                "Build a local file-backed multi-agent company operating loop",
                runs_dir=runs_dir,
                run_name="agentic-company",
                use_memory=False,
            )
            self.assertFalse(memory_db.exists())

            sync_runs_dir_to_memory(runs_dir, memory_db)
            summary = render_memory_summary(memory_db)
            self.assertIn("agentic-company", summary)

    def test_ollama_provider_uses_structured_local_response(self):
        class FakeResponse:
            def __init__(self, role_id):
                self.role_id = role_id

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                role = next(item for item in ROLES if item.role_id == self.role_id)
                payload = {
                    "role_id": role.role_id,
                    "summary": f"{role.title} summary.",
                    "recommendations": ["Keep the loop tight."],
                    "risks": ["Ceremony can exceed value."],
                    "asks": ["Approve the next task."],
                    "tasks": [
                        {
                            "owner": role.title,
                            "priority": "P0",
                            "task": f"{role.title} provider task.",
                            "due": "Day 1",
                            "definition_of_done": "Task is accepted.",
                        }
                    ],
                }
                transport = {"response": json.dumps(payload)}
                return json.dumps(transport).encode("utf-8")

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            match = re.search(r"role_id `([^`]+)`", body["prompt"])
            self.assertIsNotNone(match)
            return FakeResponse(match.group(1))

        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen", side_effect=fake_urlopen) as urlopen:
                result = CompanyEngine(provider="ollama", provider_model="mistral:latest").run(
                    "Build a local file-backed multi-agent company operating loop",
                    runs_dir=Path(tmp),
                    run_name="ollama-run",
                )

            self.assertEqual(result.provider, "ollama")
            self.assertTrue(urlopen.called)
            self.assertEqual(urlopen.call_count, 6)
            self.assertTrue(any("provider task" in item["task"] for item in result.tasks))
            provider_contract = (result.run_dir / "provider_contract.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Provider: ollama", provider_contract)
            self.assertIn("mistral:latest", provider_contract)

            sync_run_to_memory(result.run_dir, result.memory_db)
            summary = render_memory_summary(result.memory_db)
            self.assertIn("Provider: ollama", summary)

    def test_ollama_provider_normalizes_aliases_and_falls_back_for_missing_roles(self):
        engine = CompanyEngine()
        provider = OllamaMemoProvider()
        payload = {
            "memos": [
                {
                    "role_id": "CEO",
                    "summary": "CEO summary.",
                    "recommendations": ["Decide."],
                    "risks": ["Drift."],
                    "asks": ["Approve."],
                    "tasks": [],
                },
                {
                    "role_id": "CFO/Risk",
                    "summary": "Risk summary.",
                    "recommendations": ["Limit scope."],
                    "risks": ["Cost."],
                    "asks": ["Set budget."],
                    "tasks": [],
                },
                {
                    "role_id": "COO",
                    "summary": "Ops summary.",
                    "recommendations": ["Run cadence."],
                    "risks": ["No review."],
                    "asks": ["Review daily."],
                    "tasks": [],
                },
            ]
        }

        memos = provider._memos_from_payload(engine, payload)
        self.assertEqual([memo.role.role_id for memo in memos], [
            "ceo",
            "cpo",
            "cto",
            "cro",
            "cfo_risk",
            "coo",
        ])
        self.assertIn("CEO summary", memos[0].summary)
        self.assertIn("deterministic fallback", memos[1].summary)
        self.assertEqual(fallback_roles(memos), [
            "Chief Product and Market Officer",
            "Chief Technology Officer",
            "Chief Revenue Officer",
        ])
        self.assertIn("Risk summary", memos[4].summary)
        self.assertIn("Ops summary", memos[5].summary)

    def test_ollama_provider_retries_and_falls_back_one_role(self):
        class FakeResponse:
            def __init__(self, role_id, valid=True):
                self.role_id = role_id
                self.valid = valid

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                role = next(item for item in ROLES if item.role_id == self.role_id)
                role_id = self.role_id if self.valid else "wrong_role"
                payload = {
                    "role_id": role_id,
                    "summary": f"{role.title} summary.",
                    "recommendations": ["Keep the loop tight."],
                    "risks": ["Ceremony can exceed value."],
                    "asks": ["Approve the next task."],
                    "tasks": [
                        {
                            "owner": role.title,
                            "priority": "P0",
                            "task": f"{role.title} provider task.",
                            "due": "Day 1",
                            "definition_of_done": "Task is accepted.",
                        }
                    ],
                }
                return json.dumps({"response": json.dumps(payload)}).encode("utf-8")

        calls_by_role = {}

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            match = re.search(r"role_id `([^`]+)`", body["prompt"])
            role_id = match.group(1)
            calls_by_role[role_id] = calls_by_role.get(role_id, 0) + 1
            return FakeResponse(role_id, valid=role_id != "cto")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = CompanyEngine(provider="ollama").run(
                    "Build a local file-backed multi-agent company operating loop",
                    runs_dir=Path(tmp),
                    run_name="ollama-retry",
                )

        summaries = {memo.role.role_id: memo.summary for memo in result.memos}
        self.assertEqual(calls_by_role["cto"], 2)
        self.assertEqual(calls_by_role["ceo"], 1)
        self.assertIn("fallback used", summaries["cto"])
        self.assertIn("CEO summary", summaries["ceo"])
        self.assertEqual(fallback_roles(result.memos), ["Chief Technology Officer"])


if __name__ == "__main__":
    unittest.main()
