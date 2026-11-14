import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from regression_snapshot import InputError, capture_suite, load_snapshot, load_suite


class RegressionSnapshotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fixture.json"
        self.suite = {
            "schema_version": 1,
            "scenarios": [
                {"id": "pilot-shop", "objective": " Run a local SMB sales pilot. "},
                {
                    "id": "agent-review",
                    "objective": "Build a multi-agent operating loop.",
                    "context": "Use a café as the example.",
                    "days": 4,
                },
            ],
        }

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_capture_is_stable_offline_and_uses_no_persistent_memory(self):
        self.write(self.suite)
        suite = load_suite(self.path)
        self.assertEqual(suite["scenarios"][1]["days"], 7)
        self.assertEqual(suite["scenarios"][1]["context"], "")
        capture_roots = []

        def track_capture_root(**kwargs):
            directory = tempfile.TemporaryDirectory(**kwargs)
            capture_roots.append(Path(directory.name))
            return directory

        with (
            patch("company.urllib.request.urlopen", side_effect=AssertionError("unexpected network")),
            patch("company.sqlite3.connect", side_effect=AssertionError("unexpected memory write")),
            patch("regression_snapshot.TemporaryDirectory", side_effect=track_capture_root),
        ):
            snapshot = capture_suite(suite)
            self.assertEqual(snapshot, capture_suite(self.suite))

        self.assertEqual(len(capture_roots), 2)
        self.assertFalse(any(path.exists() for path in capture_roots))

        self.assertEqual(snapshot["provider"], "deterministic")
        self.assertEqual([case["id"] for case in snapshot["cases"]], ["agent-review", "pilot-shop"])
        for case in snapshot["cases"]:
            self.assertEqual(len(case["roles"]), 7)
            self.assertIn({"id": "chief_of_staff", "fallback": False}, case["roles"])
            self.assertTrue(any(task["priority"] == "P0" for task in case["tasks"]))
            self.assertNotIn("updated_at", case["tasks"][0])

        self.write(snapshot)
        self.assertEqual(load_snapshot(self.path), snapshot)

    def test_fingerprint_uses_context_and_defaults(self):
        updated = deepcopy(self.suite)
        updated["scenarios"][0].update({"objective": "Run a local SMB sales pilot.", "context": "", "days": 7})
        baseline = capture_suite(self.suite)
        self.assertEqual(capture_suite(updated), baseline)

        updated["scenarios"][0]["context"] = "Forecast a café pilot."
        candidate = capture_suite(updated)
        self.assertEqual(baseline["cases"][0], candidate["cases"][0])
        self.assertNotEqual(baseline["cases"][1]["scenario_sha256"], candidate["cases"][1]["scenario_sha256"])

    def test_suite_rejects_duplicate_keys_and_case_ids(self):
        self.path.write_text('{"schema_version": 1, "schema_version": 1, "scenarios": []}', encoding="utf-8")
        with self.assertRaisesRegex(InputError, r"fixture\.json: duplicate JSON key 'schema_version'"):
            load_suite(self.path)

        self.suite["scenarios"][1]["id"] = "pilot-shop"
        self.write(self.suite)
        with self.assertRaisesRegex(InputError, r"scenarios\[1\] duplicates ID 'pilot-shop'"):
            load_suite(self.path)

    def test_suite_reports_invalid_values_at_their_field(self):
        for key, value in [("days", True), ("days", 0), ("objective", " "), ("context", None), ("id", "../case")]:
            with self.subTest(field=key, value=value):
                suite = deepcopy(self.suite)
                suite["scenarios"][1][key] = value
                self.write(suite)
                with self.assertRaisesRegex(InputError, rf"scenarios\[1\]\.{key}"):
                    load_suite(self.path)

        self.suite["schema_version"] = True
        self.write(self.suite)
        with self.assertRaisesRegex(InputError, "schema_version must be the integer 1"):
            load_suite(self.path)

    def test_snapshot_rejects_malformed_role_and_task_data(self):
        snapshot = capture_suite(self.suite)

        candidate = deepcopy(snapshot)
        candidate["cases"][0]["roles"][0]["fallback"] = 0
        self.write(candidate)
        with self.assertRaisesRegex(InputError, r"cases\[0\]\.roles\[0\]\.fallback"):
            load_snapshot(self.path)

        candidate = deepcopy(snapshot)
        candidate["cases"][0]["roles"].append(candidate["cases"][0]["roles"][0])
        self.write(candidate)
        with self.assertRaisesRegex(InputError, r"roles\[7\] duplicates ID"):
            load_snapshot(self.path)

        candidate = deepcopy(snapshot)
        del candidate["cases"][1]["tasks"][0]["priority"]
        self.write(candidate)
        with self.assertRaisesRegex(InputError, r"cases\[1\]\.tasks\[0\] is missing 'priority'"):
            load_snapshot(self.path)

    def test_snapshot_accepts_historical_or_missing_roles_for_comparison(self):
        snapshot = capture_suite(self.suite)
        snapshot["provider"] = "reference-provider"
        snapshot["cases"][0]["roles"].append({"id": "historical_role", "fallback": True})
        snapshot["cases"][1]["roles"] = []
        self.write(snapshot)
        self.assertEqual(load_snapshot(self.path), snapshot)

    def test_snapshot_accepts_empty_candidate_for_missing_case_diagnostics(self):
        snapshot = {"schema_version": 1, "provider": "deterministic", "cases": []}
        self.write(snapshot)
        self.assertEqual(load_snapshot(self.path), snapshot)
