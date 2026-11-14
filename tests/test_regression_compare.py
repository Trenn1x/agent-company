import unittest

from regression_compare import compare_snapshots, render_report


def task(
    owner="CEO", priority="P0", description="Approve the metric", *,
    due="Day 0", done="Record the metric",
):
    return {
        "owner": owner, "priority": priority, "task": description,
        "due": due, "definition_of_done": done,
    }


def role(role_id, *, fallback=False):
    return {"id": role_id, "fallback": fallback}


def case(case_id, *, digest="a" * 64, roles=(), tasks=()):
    return {"id": case_id, "scenario_sha256": digest, "roles": list(roles), "tasks": list(tasks)}


def snapshot(*cases, provider="deterministic"):
    return {"schema_version": 1, "provider": provider, "cases": list(cases)}


class RegressionComparisonTests(unittest.TestCase):
    def test_ignores_order_and_harmless_whitespace_or_case(self):
        baseline = snapshot(
            case("zeta", roles=[role("ceo"), role("cto")], tasks=[task(), task("CTO")]),
            case("alpha"),
        )
        candidate = snapshot(
            case("alpha"),
            case("zeta", roles=[role("cto"), role("ceo")], tasks=[
                task("CTO"), task("ceo", description="  APPROVE\n the metric ", due="day 0"),
            ]),
        )
        report = compare_snapshots(baseline, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual([case["id"] for case in report["cases"]], ["alpha", "zeta"])
        self.assertEqual(report["summary"], {
            "blocking": 0, "review": 0, "total_cases": 2, "compared_cases": 2,
        })

    def test_blocks_missing_scenario_and_compares_only_matching_inputs(self):
        baseline = snapshot(case("deleted"), case("edited", roles=[role("ceo")], tasks=[task()]))
        candidate = snapshot(case("edited", digest="b" * 64), case("new"))
        report = compare_snapshots(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"], {
            "blocking": 2, "review": 1, "total_cases": 3, "compared_cases": 0,
        })
        self.assertEqual(
            {case["id"]: [finding["code"] for finding in case["findings"]] for case in report["cases"]},
            {"deleted": ["scenario_missing"], "edited": ["scenario_changed"], "new": ["scenario_added"]},
        )

    def test_blocks_role_loss_and_new_fallback_but_reports_recovery(self):
        baseline = snapshot(case("plan", roles=[role("ceo"), role("cto"), role("coo", fallback=True)]))
        candidate = snapshot(case("plan", roles=[role("cto", fallback=True), role("coo"), role("cpo")]))
        report = compare_snapshots(baseline, candidate)
        self.assertEqual(
            [(item["severity"], item["code"], item["role"]) for item in report["cases"][0]["findings"]],
            [
                ("blocking", "role_missing", "ceo"), ("blocking", "fallback_added", "cto"),
                ("review", "fallback_resolved", "coo"), ("review", "role_added", "cpo"),
            ],
        )

    def test_duplicate_descriptions_cannot_hide_loss_of_critical_task(self):
        baseline = snapshot(case("plan", tasks=[task(), task(), task("CTO"), task("CTO", "P1")]))
        candidate = snapshot(case("plan", tasks=[task("CTO", "P1"), task(), task("CTO")]))
        report = compare_snapshots(baseline, candidate)
        findings = report["cases"][0]["findings"]
        self.assertFalse(report["passed"])
        self.assertEqual([(item["code"], item["owner"]) for item in findings], [("p0_task_removed", "CEO")])

    def test_reports_critical_reassignment_and_demotion_with_values(self):
        baseline = snapshot(case("plan", tasks=[
            task(), task(description="Authorize rollout"), task(description="Set the deadline"),
        ]))
        candidate = snapshot(case("plan", tasks=[
            task("CTO"),
            task(priority="p0", description="Authorize rollout"),
            task(priority=" P0 ", description="Set the deadline"),
        ]))
        report = compare_snapshots(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(
            [(item["code"], item["before"], item["after"]) for item in report["cases"][0]["findings"]],
            [
                ("p0_owner_changed", "CEO", "CTO"),
                ("p0_task_demoted", "P0", "p0"),
                ("p0_task_demoted", "P0", " P0 "),
            ],
        )

    def test_preserves_duplicate_matches_before_reporting_deadline_drift(self):
        baseline = snapshot(case("plan", tasks=[task(due="Day 2"), task(due="Day 7")]))
        candidate = snapshot(case("plan", tasks=[task(due="Day 7"), task(due="Day 9")]))
        report = compare_snapshots(baseline, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual(
            [(item["code"], item["before"], item["after"]) for item in report["cases"][0]["findings"]],
            [("due_changed", "Day 2", "Day 9")],
        )

    def test_review_only_changes_pass_and_render_safely(self):
        baseline = snapshot(case("plan", tasks=[task(priority="P1", description="A | <b>")]))
        candidate = snapshot(case("plan", tasks=[
            task(priority="P2", description="A | <b>", done="Ship **notes**"),
            task(priority="P1", description="A second task"),
        ]), provider="another-provider")
        report = compare_snapshots(baseline, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual(report["findings"][0]["code"], "provider_changed")
        self.assertEqual(report["summary"]["review"], 4)
        self.assertEqual(report["cases"][0]["status"], "review")
        markdown = render_report(report)
        self.assertIn("**PASS**", markdown)
        self.assertIn(r"A \| \<b\>", markdown)
        self.assertIn(r"Ship \*\*notes\*\*", markdown)
        self.assertTrue(markdown.endswith("\n"))

    def test_does_not_fuzzy_match_task_text_that_changes_meaning(self):
        baseline = snapshot(case("plan", tasks=[task(description="Approve rollout?")]))
        candidate = snapshot(case("plan", tasks=[task(description="Approve rollout.")]))
        report = compare_snapshots(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(
            [item["code"] for item in report["cases"][0]["findings"]],
            ["p0_task_removed", "task_added"],
        )


if __name__ == "__main__":
    unittest.main()
