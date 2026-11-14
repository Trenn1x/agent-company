import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMPANY = Path(__file__).resolve().parents[1] / "company.py"


class RegressionCLITests(unittest.TestCase):
    def run_cli(self, cwd, *args):
        return subprocess.run(
            [sys.executable, str(COMPANY), *map(str, args)],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_pass_fail_and_bad_input_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / "suite.json"
            suite.write_text(
                json.dumps({
                    "schema_version": 1,
                    "scenarios": [{"id": "internal", "objective": "Build an agent workflow"}],
                }),
                encoding="utf-8",
            )
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            report_path = root / "report.json"

            capture = self.run_cli(root, "snapshot", suite, "--output", baseline)
            self.assertEqual(capture.returncode, 0, capture.stderr)
            self.assertEqual(capture.stdout, "")
            capture = self.run_cli(root, "snapshot", suite)
            self.assertEqual(capture.returncode, 0, capture.stderr)
            candidate.write_text(capture.stdout, encoding="utf-8")
            self.assertFalse((root / "runs").exists())

            passing = self.run_cli(root, "regress", baseline, candidate)
            self.assertEqual(passing.returncode, 0, passing.stderr)
            self.assertIn("internal", passing.stdout)

            snapshot = json.loads(capture.stdout)
            task = next(task for task in snapshot["cases"][0]["tasks"] if task["priority"] == "P0")
            task["priority"] = "P1"
            candidate.write_text(json.dumps(snapshot), encoding="utf-8")
            failing = self.run_cli(
                root, "regress", baseline, candidate, "--format", "json", "--output", report_path
            )
            self.assertEqual(failing.returncode, 1, failing.stderr)
            self.assertEqual(failing.stdout, "")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            codes = {finding["code"] for case in report["cases"] for finding in case["findings"]}
            self.assertIn("p0_task_demoted", codes)

            candidate.write_text("{}", encoding="utf-8")
            malformed = self.run_cli(root, "regress", baseline, candidate)
            self.assertEqual(malformed.returncode, 2)
            self.assertIn("candidate.json", malformed.stderr)
            self.assertEqual(malformed.stdout, "")

            baseline.write_text(
                '{"schema_version": 1, "provider": "deterministic", "cases": []}',
                encoding="utf-8",
            )
            empty = self.run_cli(root, "regress", baseline, candidate)
            self.assertEqual(empty.returncode, 2)
            self.assertIn("the baseline must contain at least one scenario", empty.stderr)

    def test_capture_cannot_overwrite_the_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = json.dumps({
                "schema_version": 1,
                "scenarios": [{"id": "example", "objective": "Build a demo"}],
            })
            (root / "suite.json").write_text(suite, encoding="utf-8")
            result = self.run_cli(root, "snapshot", "suite.json", "--output", "./suite.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("output must be different", result.stderr)
            self.assertEqual((root / "suite.json").read_text(encoding="utf-8"), suite)


if __name__ == "__main__":
    unittest.main()
