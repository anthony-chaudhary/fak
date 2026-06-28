#!/usr/bin/env python3
"""Hermetic tests for pm_review.py — synthetic backlog, injected clock, no gh/network.

Every test drives `build_payload` / `collect` with a fixed `now_ts` and a synthetic
issue list, so nothing here reads GitHub, the wall clock, or disk (except the
project.yaml cap-precedence test, which uses a tmp dir). Run:

    python tools/pm_review_test.py
    python -m pytest tools/pm_review_test.py
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from pathlib import Path

import pm_review as pm

# A fixed clock: 2026-06-28T00:00:00Z, as epoch seconds.
NOW_TS = dt.datetime(2026, 6, 28, tzinfo=dt.timezone.utc).timestamp()


def iso_days_ago(days: float) -> str:
    t = dt.datetime(2026, 6, 28, tzinfo=dt.timezone.utc) - dt.timedelta(days=days)
    return t.isoformat().replace("+00:00", "Z")


def issue(number, title, *, labels=None, assignees=None, age_days=1.0, body=""):
    return {
        "number": number,
        "title": title,
        "labels": [{"name": n} for n in (labels or [])],
        "assignees": [{"login": a} for a in (assignees or [])],
        "createdAt": iso_days_ago(age_days),
        "updatedAt": iso_days_ago(age_days),
        "body": body,
    }


def build(issues, **kw):
    defaults = dict(root=Path("/repo"), now_ts=NOW_TS, cap=2, cap_source="default",
                    it=pm.DEFAULT_IMPORTANCE_THRESHOLD, ut=pm.DEFAULT_URGENCY_THRESHOLD,
                    top_starts=3)
    defaults.update(kw)
    return pm.build_payload(issues=issues, **defaults)


class TestScorers(unittest.TestCase):
    def test_priority_label_parsed(self):
        self.assertEqual(pm._priority_of(["priority/P0", "bug"]), "P0")
        self.assertEqual(pm._priority_of(["priority/P2"]), "P2")
        self.assertIsNone(pm._priority_of(["enhancement"]))

    def test_importance_rises_with_priority(self):
        now = dt.datetime(2026, 6, 28, tzinfo=dt.timezone.utc)
        p0 = build([issue(1, "fix(x): a", labels=["priority/P0"])])["grid"]["issues"][0]
        p3 = build([issue(2, "fix(y): b", labels=["priority/P3"])])["grid"]["issues"][0]
        self.assertGreater(p0["importance"], p3["importance"])

    def test_trust_floor_boost(self):
        plain = build([issue(1, "feat(ui): nicer table", labels=["priority/P2"])])
        floor = build([issue(2, "fix(scrub): secret leak in guard", labels=["priority/P2"])])
        self.assertTrue(floor["grid"]["issues"][0]["factors"]["trust_floor"])
        self.assertFalse(plain["grid"]["issues"][0]["factors"]["trust_floor"])
        self.assertGreater(floor["grid"]["issues"][0]["importance"],
                           plain["grid"]["issues"][0]["importance"])

    def test_orphaned_p01_urgency(self):
        orphan = build([issue(1, "fix: a", labels=["priority/P1"], assignees=[])])
        owned = build([issue(2, "fix: b", labels=["priority/P1"], assignees=["alice"])])
        self.assertTrue(orphan["grid"]["issues"][0]["factors"]["orphaned_p01"])
        self.assertFalse(owned["grid"]["issues"][0]["factors"]["orphaned_p01"])
        self.assertGreater(orphan["grid"]["issues"][0]["urgency"],
                           owned["grid"]["issues"][0]["urgency"])

    def test_blocker_boost(self):
        blk = build([issue(1, "fix: the loop is wedged", labels=["priority/P2"])])
        self.assertTrue(blk["grid"]["issues"][0]["factors"]["blocker"])

    def test_age_increases_urgency(self):
        young = build([issue(1, "fix: a", labels=["priority/P2"], age_days=1)])
        old = build([issue(2, "fix: b", labels=["priority/P2"], age_days=30)])
        self.assertGreater(old["grid"]["issues"][0]["urgency"],
                           young["grid"]["issues"][0]["urgency"])

    def test_quadrant_assignment(self):
        # P0 + orphaned + old => DO_NOW; P3 + young + owned => BACKLOG.
        do_now = build([issue(1, "fix: fire", labels=["priority/P0"], age_days=20)])
        backlog = build([issue(2, "chore: tidy", labels=["priority/P3"],
                               assignees=["a"], age_days=0.5)])
        self.assertEqual(do_now["grid"]["issues"][0]["quadrant"], "DO_NOW")
        self.assertEqual(backlog["grid"]["issues"][0]["quadrant"], "BACKLOG")


class TestEpicSpawnGraph(unittest.TestCase):
    EPIC_BODY = (
        "## Children\n"
        "- [ ] #898 - bench: pin target\n"
        "- [x] #899 - feat: adapter\n"
        "- [ ] #900 - eval: rehearsal\n"
        "Related: #401 (out of scope, prose only)\n"
    )

    def test_children_parsed_from_checklist(self):
        kids = pm.parse_children(self.EPIC_BODY)
        self.assertEqual([k["number"] for k in kids], [898, 899, 900])
        self.assertEqual([k["done"] for k in kids], [False, True, False])
        # The prose '#401' is NOT a checklist child — never fabricated.
        self.assertNotIn(401, [k["number"] for k in kids])

    def test_true_wip_in_units(self):
        p = build([
            issue(897, "epic(tb): rank-1", labels=["epic"], body=self.EPIC_BODY),
            issue(900, "eval: rehearsal", labels=["priority/P2"]),
        ])
        g = p["epic_spawn_graph"]
        self.assertEqual(g["live_epics"], 1)
        self.assertEqual(g["true_wip_open_child_units"], 2)  # #898 + #900 open
        self.assertEqual(g["epics"][0]["done_children"], 1)

    def test_epic_without_checklist_reported_not_fabricated(self):
        p = build([issue(931, "epic(serving): qwen", labels=["epic"],
                         body="Prose only, mentions #483 and #610.")])
        g = p["epic_spawn_graph"]
        self.assertEqual(g["epics_without_child_checklist"], 1)
        self.assertEqual(g["true_wip_open_child_units"], 0)
        self.assertFalse(g["epics"][0]["has_checklist"])

    def test_epic_detected_by_title_scope(self):
        p = build([issue(1, "epic(cache): multilevel default cache")])
        self.assertTrue(p["grid"]["issues"][0]["is_epic"])


class TestWipCap(unittest.TestCase):
    def _three_epics(self):
        return [
            issue(1, "epic(a): x", labels=["epic"], body="- [ ] #11\n"),
            issue(2, "epic(b): y", labels=["epic"], body="- [ ] #21\n- [x] #22\n"),
            issue(3, "epic(c): z", labels=["epic"], body="- [ ] #31\n"),
        ]

    def test_over_cap_flagged(self):
        p = build(self._three_epics(), cap=2)
        self.assertEqual(p["wip"]["verdict"], "WIP_OVER_CAP")
        self.assertTrue(p["wip"]["over_cap"])
        self.assertFalse(p["ok"])

    def test_under_cap_ok(self):
        p = build(self._three_epics(), cap=5)
        self.assertEqual(p["wip"]["verdict"], "WIP_OK")
        self.assertTrue(p["ok"])

    def test_closest_to_done_epic_is_highest_ratio(self):
        p = build(self._three_epics(), cap=2)
        ctd = p["wip"]["closest_to_done_epic"]
        self.assertEqual(ctd["epic"], 2)  # 1/2 done is the highest done-ratio

    def test_cap_precedence_flag_over_yaml_over_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".claude").mkdir()
            (root / ".claude" / "project.yaml").write_text(
                "name: x\npm_review:\n  wip_cap: 4\n", encoding="utf-8")
            # flag wins
            self.assertEqual(pm.load_wip_cap(root, 7), (7, "flag"))
            # yaml when no flag (PyYAML present in this repo)
            cap, src = pm.load_wip_cap(root, None)
            self.assertEqual((cap, src), (4, "project.yaml"))
            # default when neither
            self.assertEqual(pm.load_wip_cap(Path(d) / "nope", None), (pm.DEFAULT_WIP_CAP, "default"))


class TestDoNext(unittest.TestCase):
    def test_starts_are_top_do_now_non_epic(self):
        issues = [
            issue(1, "fix: fire one", labels=["priority/P0"], age_days=20),
            issue(2, "fix: fire two", labels=["priority/P0"], age_days=15),
            issue(3, "chore: minor", labels=["priority/P3"], assignees=["a"], age_days=0.2),
            issue(9, "epic(x): big", labels=["epic", "priority/P0"], age_days=20),
        ]
        p = build(issues, top_starts=2)
        starts = p["do_next"]["starts"]
        self.assertEqual(len(starts), 2)
        self.assertNotIn(9, [s["number"] for s in starts])  # epics are not "starts"
        self.assertNotIn(3, [s["number"] for s in starts])  # backlog filtered out

    def test_truncation_is_logged_not_silent(self):
        issues = [issue(i, "fix: fire", labels=["priority/P0"], age_days=20)
                  for i in range(1, 6)]
        p = build(issues, top_starts=2)
        notes = " ".join(p["do_next"]["notes"])
        self.assertIn("dropped", notes)
        self.assertEqual(p["coverage"]["do_now_units"], 5)

    def test_one_blocker_prefers_fleet_bottleneck(self):
        bl = {"headline": {"title": "Rate-limit saturation", "severity": "HIGH",
                           "fix": "spread accounts"}}
        p = build([issue(1, "fix: a blocked thing", labels=["priority/P1"])], bottleneck=bl)
        b = p["do_next"]["one_blocker"]
        self.assertEqual(b["kind"], "fleet_bottleneck")
        self.assertEqual(b["title"], "Rate-limit saturation")

    def test_one_blocker_falls_back_to_issue(self):
        p = build([issue(7, "fix: the gateway is broken", labels=["priority/P1"])])
        b = p["do_next"]["one_blocker"]
        self.assertEqual(b["kind"], "issue")
        self.assertEqual(b["number"], 7)


class TestDeterminismAndPurity(unittest.TestCase):
    def _mixed(self):
        return [
            issue(897, "epic(tb): rank-1", labels=["epic"],
                  body="- [ ] #898\n- [x] #899\n"),
            issue(10, "fix(scrub): secret leak", labels=["priority/P0"], age_days=12),
            issue(11, "feat(ui): table", labels=["priority/P3"], assignees=["a"], age_days=1),
            issue(12, "fix: wedged loop", labels=["priority/P1"], age_days=30),
        ]

    def test_two_runs_byte_identical(self):
        a = json.dumps(build(self._mixed()), sort_keys=False)
        b = json.dumps(build(self._mixed()), sort_keys=False)
        self.assertEqual(a, b)

    def test_industry_context_folded_when_provided(self):
        ind = {"grade": "C", "parity_debt": 12}
        p = build(self._mixed(), industry=ind)
        self.assertEqual(p["context"]["industry"]["grade"], "C")
        self.assertEqual(p["context"]["industry"]["parity_debt"], 12)

    def test_fetch_error_surfaced(self):
        p = build([{"_error": "gh exploded"}])
        self.assertEqual(p["verdict"], "FETCH_ERROR")
        self.assertFalse(p["ok"])
        self.assertEqual(p["fetch_error"], "gh exploded")

    def test_renderers_do_not_crash(self):
        p = build(self._mixed())
        self.assertIn("PM review", pm.render(p))
        self.assertIn("# PM review", pm.render_md(p, "2026-06-28"))

    def test_collect_with_injected_fetcher_is_hermetic(self):
        # collect() must reach NO network when a fetcher is injected.
        called = {}

        def fake_fetch(root, limit):
            called["hit"] = True
            return self._mixed()

        p = pm.collect(Path("/repo"), now_ts=NOW_TS, wip_cap=2, fetcher=fake_fetch)
        self.assertTrue(called["hit"])
        self.assertEqual(p["schema"], pm.SCHEMA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
