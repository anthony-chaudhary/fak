"""Tests for dispatch_conservation.py — the worker-unit conservation ledger.

The load-bearing facts pinned here:
  - windowing keys on the SPAWN stamp in the log name (UTC), never mtime;
  - every finished in-window unit folds to exactly one closed outcome, and the
    conservation identity (spent = shipped + unwitnessed + no-commit +
    spawn-failed + leaked) holds by construction over any artifact mix;
  - a graded .witness is final; an alive .pid is live (and an UNSCANNABLE host
    counts .pid units live — a blind probe must never invent a leak);
  - a dead worker with a real log and no witness is the LEAK bucket, and a
    header-only log is spawn-failed (the 5s-probe blind spot made visible);
  - closes/holds are summed over in-window rows only;
  - re-storm churn (one issue burning 2+ units in a window) is surfaced;
  - --fail-on-leak turns the leak count into an exit code (the CI gate).
"""

import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "dispatch_conservation.py"


def load():
    spec = importlib.util.spec_from_file_location("dispatch_conservation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = load()

NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def stamp(hours_ago: float) -> str:
    dt = datetime.fromtimestamp(NOW - hours_ago * 3600, tz=timezone.utc)
    return dt.strftime("%Y%m%d-%H%M%S")


def iso(hours_ago: float) -> str:
    dt = datetime.fromtimestamp(NOW - hours_ago * 3600, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def mk_worker(self, issue: int, hours_ago: float, *, kind="resolve",
                  lane="tools", backend="claude", body="worker output\n",
                  witness: dict | None = None, pid: int | None = None) -> Path:
        name = f"{kind}-{issue}-{stamp(hours_ago)}.log"
        log = self.runs / name
        header = f"# fak-spawn {stamp(hours_ago)} issue={issue} lane={lane} backend={backend} argv0=claude.exe\n"
        log.write_text(header + body, encoding="utf-8")
        if witness is not None:
            log.with_suffix(".witness").write_text(
                json.dumps(witness, sort_keys=True), encoding="utf-8")
        if pid is not None:
            log.with_suffix(".pid").write_text(str(pid), encoding="utf-8")
        return log

    def report(self, *, alive=frozenset(), window_h=6.0):
        units = dc.collect_units(self.runs, since_ts=NOW - window_h * 3600,
                                 alive=set(alive) if alive is not None else None)
        closes = dc.windowed_closes(self.runs / "progress.jsonl", since_ts=NOW - window_h * 3600)
        holds = dc.windowed_contract_holds(self.runs / "contract-holds.jsonl",
                                           since_ts=NOW - window_h * 3600)
        return dc.fold_conservation(units, closes, holds, window_h=window_h,
                                    now_iso=iso(0))


class StampWindowingTest(Fixture):
    def test_stamp_parses_as_utc(self):
        ts = dc.parse_log_stamp_utc(f"resolve-42-{stamp(1.0)}.log")
        self.assertAlmostEqual(ts, NOW - 3600, delta=1)
        self.assertIsNone(dc.parse_log_stamp_utc("resolve-42-garbage.log"))
        self.assertIsNone(dc.parse_log_stamp_utc("unrelated.log"))

    def test_window_keys_on_spawn_stamp_not_mtime(self):
        inside = self.mk_worker(1, 2.0, witness={"claim": "CLAIM_WITNESSED", "sha": "a" * 40})
        outside = self.mk_worker(2, 30.0, witness={"claim": "CLAIM_WITNESSED", "sha": "b" * 40})
        # A fresh mtime on an OLD unit must not pull it into the window.
        os.utime(outside, (NOW, NOW))
        os.utime(inside, (NOW, NOW))
        rep = self.report()
        self.assertEqual(rep["units"]["resolve_total"], 1)
        self.assertEqual(rep["units"]["shipped_witnessed"], 1)


class OutcomeClassificationTest(Fixture):
    def test_every_bucket_and_identity(self):
        self.mk_worker(10, 1.0, witness={"claim": "CLAIM_WITNESSED", "sha": "a" * 40})
        self.mk_worker(11, 1.1, witness={"claim": "CLAIM_UNWITNESSED", "sha": "b" * 40,
                                         "verdict": "ABSTAIN", "witness": "subject-only"})
        self.mk_worker(12, 1.2, witness={"claim": "CLAIM_NO_COMMIT", "sha": None,
                                         "reason": "policy_block"})
        self.mk_worker(13, 1.3, body="")                      # header-only: spawn failed
        self.mk_worker(14, 1.4)                               # dead, real log, no witness: LEAK
        self.mk_worker(15, 1.5, pid=4242)                     # alive pid: live, not spent
        rep = self.report(alive={4242})
        u = rep["units"]
        self.assertEqual(u["resolve_total"], 6)
        self.assertEqual(u["live"], 1)
        self.assertEqual(u["spent"], 5)
        self.assertEqual(u["shipped_witnessed"], 1)
        self.assertEqual(u["committed_unwitnessed"], 1)
        self.assertEqual(u["no_commit"], 1)
        self.assertEqual(u["no_commit_reasons"], {"policy_block": 1})
        self.assertEqual(u["spawn_failed"], 1)
        self.assertEqual(u["leaked_unswept"], 1)
        self.assertTrue(rep["identity_holds"])
        self.assertEqual(rep["verdict"], "LEAKING")
        self.assertEqual(rep["leaked_units"][0]["issue"], 14)

    def test_dead_pid_with_no_witness_is_a_leak(self):
        self.mk_worker(20, 1.0, pid=999)  # pid sidecar survives but pid is dead
        rep = self.report(alive={1})
        self.assertEqual(rep["units"]["leaked_unswept"], 1)
        self.assertEqual(rep["units"]["live"], 0)

    def test_unscannable_host_counts_pid_units_live_never_leaked(self):
        self.mk_worker(21, 1.0, pid=999)
        rep = self.report(alive=None)  # alive=None: host scan unavailable
        self.assertEqual(rep["units"]["live"], 1)
        self.assertEqual(rep["units"]["leaked_unswept"], 0)
        self.assertEqual(rep["verdict"], "CONSERVED")

    def test_witness_outranks_live_pid(self):
        # The sweep only grades dead pids, so a graded unit with a leftover
        # .pid sidecar is finished — the verdict is final.
        self.mk_worker(22, 1.0, pid=4242,
                       witness={"claim": "CLAIM_NO_COMMIT", "sha": None, "reason": "auth_wall"})
        rep = self.report(alive={4242})
        self.assertEqual(rep["units"]["no_commit"], 1)
        self.assertEqual(rep["units"]["live"], 0)

    def test_unknown_witness_reason_folds_to_unknown(self):
        self.mk_worker(23, 1.0, witness={"claim": "CLAIM_NO_COMMIT", "sha": None,
                                         "reason": "not-a-real-bucket"})
        rep = self.report()
        self.assertEqual(rep["units"]["no_commit_reasons"], {"unknown": 1})

    def test_repair_units_counted_separately(self):
        self.mk_worker(30, 1.0, kind="repair", lane="contract-repair")
        rep = self.report()
        self.assertEqual(rep["units"]["repair_total"], 1)
        self.assertEqual(rep["units"]["resolve_total"], 0)
        self.assertEqual(rep["units"]["leaked_unswept"], 0)


class IssueSideTest(Fixture):
    def test_closes_and_holds_windowed(self):
        rows = [
            {"schema": "fleet-issue-resolve-progress/1", "utc": iso(1.0), "closed_now": 2,
             "open_now": 700, "baseline_open": 483},
            {"schema": "fleet-issue-resolve-progress/1", "utc": iso(0.5), "closed_now": 1,
             "open_now": 698, "baseline_open": 483},
            {"schema": "fleet-issue-resolve-progress/1", "utc": iso(30.0), "closed_now": 9},
        ]
        (self.runs / "progress.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
        holds = [
            {"utc": iso(1.0), "ts": NOW - 3600, "issue": 100, "score": 8, "reason": "x"},
            {"utc": iso(0.9), "ts": NOW - 3240, "issue": 100, "score": 8, "reason": "x"},
            {"utc": iso(40.0), "ts": NOW - 40 * 3600, "issue": 200, "score": 8, "reason": "x"},
        ]
        (self.runs / "contract-holds.jsonl").write_text(
            "\n".join(json.dumps(r) for r in holds) + "\n", encoding="utf-8")
        rep = self.report()
        self.assertEqual(rep["yield"]["issues_closed_in_window"], 3)
        self.assertEqual(rep["yield"]["open_now"], 698)
        self.assertEqual(rep["contract_holds"], {"rows": 2, "distinct_issues": 1})

    def test_churn_surfaces_issues_burning_multiple_units(self):
        self.mk_worker(50, 2.0, witness={"claim": "CLAIM_NO_COMMIT", "sha": None,
                                         "reason": "self_modify"})
        self.mk_worker(50, 1.0, witness={"claim": "CLAIM_NO_COMMIT", "sha": None,
                                         "reason": "self_modify"})
        self.mk_worker(51, 1.0, witness={"claim": "CLAIM_WITNESSED", "sha": "c" * 40})
        rep = self.report()
        self.assertEqual(rep["churn"]["issues_with_2plus_units"], 1)
        self.assertEqual(rep["churn"]["worst"], [{"issue": 50, "units": 2}])


class CLITest(Fixture):
    def run_main(self, *extra, alive=frozenset()):
        argv = ["--runs-dir", str(self.runs), "--json", *extra]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dc.main(argv, alive_provider=lambda: set(alive), now_ts=NOW)
        return rc, json.loads(buf.getvalue())

    def test_json_report_and_fail_on_leak_gate(self):
        self.mk_worker(60, 1.0)  # one leaked unit
        rc, rep = self.run_main()
        self.assertEqual(rc, 0)  # report-only by default
        self.assertEqual(rep["schema"], dc.SCHEMA)
        self.assertEqual(rep["units"]["leaked_unswept"], 1)
        rc, _ = self.run_main("--fail-on-leak", "0")
        self.assertEqual(rc, 1)
        rc, _ = self.run_main("--fail-on-leak", "1")
        self.assertEqual(rc, 0)

    def test_missing_runs_dir_degrades_to_empty_conserved(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dc.main(["--runs-dir", str(self.runs / "nope"), "--json"],
                         alive_provider=lambda: set(), now_ts=NOW)
        self.assertEqual(rc, 0)
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["units"]["resolve_total"], 0)
        self.assertEqual(rep["verdict"], "CONSERVED")


if __name__ == "__main__":
    unittest.main()
