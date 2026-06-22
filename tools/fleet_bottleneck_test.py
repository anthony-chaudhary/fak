#!/usr/bin/env python3
"""Tests for fleet_bottleneck.py — the Top-10 scorers, the Prometheus exposition,
and the cross-surface contract (dashboard + alerts only query emitted metrics).

Run from tools/ (sibling import), e.g.  python -m unittest fleet_bottleneck_test
"""
from __future__ import annotations

import os
import re
import json
import tempfile
import unittest

import fleet_bottleneck as fb

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DASH = os.path.join(HERE, "grafana", "dashboards", "fleet-bottleneck-overview.json")
GATEWAY_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-gateway-observability.json")
DOGFOOD_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-dogfood-slow-requests.json")
STARTUP_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-startup-load.json")
ALERTS = os.path.join(HERE, "grafana", "prometheus-alerts.yml")
PROMETHEUS = os.path.join(HERE, "grafana", "prometheus.yml")
# fak_gateway_/fak_kernel_ families are emitted from two files: the request/kernel
# counters in metrics.go and the one-time boot timeline in startup.go.
GATEWAY_METRICS_GO = os.path.join(HERE, "..", "fak", "internal", "gateway", "metrics.go")
STARTUP_METRICS_GO = os.path.join(HERE, "..", "fak", "internal", "gateway", "startup.go")


def make_snap(surface=4, api_error=3, hanging=2, auth=1, throttled_accts=("acctA",)):
    """A representative FleetSnapshot exercising every metric family + scorer."""
    throttle = {a: {"reset": "12:00am", "age_min": 5.0} for a in throttled_accts}
    return {
        "generated_utc": "2026-06-18T12:00:00+00:00",
        "fleet_dir": "/x", "errors": [],
        "registry": {
            "generated_utc": "2026-06-18T12:00:00+00:00", "age_min": 22.0, "window_h": 6.0,
            "n_sessions": 50, "n_accounts": 4, "n_throttled": len(throttle),
            "accounts": ["acctA", "acctB", "acctC", "acctD"],
            "category": {"AGENT": 40}, "disposition": {"DONE": 20}, "action": {"SKIP_DONE": 20},
            "throttle": throttle,
            "counts": {
                "live": 5, "done": 15, "active": 20, "hanging": hanging, "dead": 4,
                "auto_resume": 2, "supervised": 3, "auth_blocked": auth,
                "deferred_throttle": 1, "rate_limited": 1, "workers_on_throttled": 9,
                "surface": surface, "api_error": api_error,
            },
            "per_account": {"acctA": {"total": 30, "active": 18}, "acctB": {"total": 5, "active": 1},
                            "acctC": {"total": 5, "active": 1}, "acctD": {"total": 0, "active": 0}},
            "sessions": [],
            "hanging_detail": [{"account": "acctA", "project": "p", "session": "abc",
                                "disp": "PARKED_WAIT", "age_min": 200.0}],
        },
        "recovery": {"watchdog_log_age_min": 90.0, "resumed_ever": 0,
                     "transitions_log_present": True},
        "audit": {
            "n_analyzed": 12, "days": 1.5, "totals": {}, "total_cost_usd": 42.5,
            "cost_per_active_session_hr": 1.2,
            "dist": {"cache_hit_frac": {"median": 0.74}, "io_ratio": {"median": 120.0}},
            "tool_mix": {}, "top_spenders": [{"session": "11111111", "ns": "C--work", "output": 90000,
                                              "io": 100, "cache_hit": 0.7, "cost": 5.0}],
            "worst_cache": [{"session": "22222222", "ns": "C--work", "output": 1000,
                             "io": 300, "cache_hit": 0.4, "cost": 2.0}],
        },
    }


def emitted_families(snap):
    """The set of metric family names render_prometheus emits for `snap`."""
    ranked = fb.rank_bottlenecks(snap)
    text = fb.render_prometheus(snap, ranked)
    return {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE ")}, text, ranked


class AuditEmptyAlertTest(unittest.TestCase):
    """FleetTokenWaste reads fleet_cache_hit_ratio_median, which is ABSENT (not
    zero) when the session audit returns empty — so the alert silently dies in
    exactly the broken-audit case. The surface must carry a companion alert that
    fires on that absence (issue #308). This reads only the alerts file, so it is
    independent of the Go-source cross-surface checks."""

    def test_audit_empty_companion_alert_present(self):
        alerts = _read(ALERTS)
        self.assertIn("alert: FleetAuditEmpty", alerts,
                      "missing the FleetAuditEmpty companion alert")
        m = re.search(r"alert: FleetAuditEmpty.*?expr:\s*(.+)", alerts, re.DOTALL)
        self.assertIsNotNone(m, "FleetAuditEmpty has no expr")
        expr = m.group(1).splitlines()[0]
        self.assertIn("absent(", expr, f"FleetAuditEmpty must use absent(): {expr!r}")
        self.assertIn("fleet_cache_hit_ratio_median", expr,
                      "FleetAuditEmpty must watch the FleetTokenWaste source metric")


class ScorerTest(unittest.TestCase):
    def test_surface_and_api_error_classes_fire(self):
        ranked = fb.rank_bottlenecks(make_snap(surface=4, api_error=3))
        by_id = {b["id"]: b for b in ranked["bottlenecks"]}
        self.assertIn("surface_backlog", by_id)
        self.assertIn("api_error_stalls", by_id)
        self.assertEqual(by_id["surface_backlog"]["score"], 40.0)   # min(60, 10*4)
        self.assertEqual(by_id["api_error_stalls"]["score"], 36.0)  # min(60, 12*3)
        self.assertEqual(by_id["surface_backlog"]["layer"], "Recovery")
        self.assertEqual(by_id["api_error_stalls"]["layer"], "Provider")

    def test_classes_capped_at_high(self):
        by_id = {b["id"]: b for b in fb.rank_bottlenecks(make_snap(surface=99, api_error=99))["bottlenecks"]}
        self.assertEqual(by_id["surface_backlog"]["score"], 60.0)
        self.assertEqual(by_id["api_error_stalls"]["score"], 60.0)

    def test_classes_absent_when_zero(self):
        ids = {b["id"] for b in fb.rank_bottlenecks(make_snap(surface=0, api_error=0))["bottlenecks"]}
        self.assertNotIn("surface_backlog", ids)
        self.assertNotIn("api_error_stalls", ids)


class PrometheusExpositionTest(unittest.TestCase):
    def test_exposition_is_well_formed(self):
        fams, text, _ = emitted_families(make_snap())
        help_names = {l.split()[2] for l in text.splitlines() if l.startswith("# HELP ")}
        self.assertEqual(fams, help_names, "every TYPE must have a matching HELP")
        line_re = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9].*$')
        for l in text.splitlines():
            if not l or l.startswith("#"):
                continue
            self.assertRegex(l, line_re, f"malformed sample line: {l!r}")
            base = l.split("{")[0].split(" ")[0]
            self.assertIn(base, fams, f"series {base} has no TYPE")

    def test_expected_families_present(self):
        fams, _, _ = emitted_families(make_snap())
        for n in ("fleet_up", "fleet_health_state", "fleet_bottleneck_score",
                  "fleet_surface_backlog", "fleet_api_error_stalls",
                  "fleet_workers_active_per_account", "fleet_cost_usd",
                  "fleet_cache_hit_ratio_median", "fleet_top_spender_output_tokens",
                  "fleet_snapshot_timestamp_seconds", "fleet_registry_age_minutes"):
            self.assertIn(n, fams)

    def test_label_values_escaped(self):
        # a title with a quote/backslash must not break the exposition
        snap = make_snap()
        ranked = fb.rank_bottlenecks(snap)
        ranked["bottlenecks"].append({"id": "x", "title": 'a"b\\c', "layer": "L",
                                      "severity": "LOW", "score": 5.0})
        text = fb.render_prometheus(snap, ranked)
        self.assertIn(r'title="a\"b\\c"', text)

    def test_no_registry_still_emits_liveness(self):
        snap = {"generated_utc": "2026-06-18T12:00:00+00:00", "registry": None, "audit": None}
        fams, text, _ = emitted_families(snap)
        self.assertIn("fleet_up", fams)
        self.assertIn("fleet_health_state", fams)

    def test_fmt_prom_total_on_non_finite(self):
        # _fmt_prom must never raise (int(inf)/int(nan) would) — it is total.
        self.assertEqual(fb._fmt_prom(float("inf")), "+Inf")
        self.assertEqual(fb._fmt_prom(float("-inf")), "-Inf")
        self.assertEqual(fb._fmt_prom(float("nan")), "NaN")
        self.assertEqual(fb._fmt_prom(5), "5")
        self.assertEqual(fb._fmt_prom(5.0), "5")
        self.assertEqual(fb._fmt_prom(0.5), "0.5")

    def test_non_finite_values_dropped_not_crash(self):
        snap = make_snap()
        ranked = fb.rank_bottlenecks(snap)
        n_before = len([b for b in ranked["bottlenecks"]])
        ranked["bottlenecks"].append({"id": "i", "title": "i", "layer": "L", "severity": "LOW", "score": float("inf")})
        ranked["bottlenecks"].append({"id": "n", "title": "n", "layer": "L", "severity": "LOW", "score": float("nan")})
        text = fb.render_prometheus(snap, ranked)  # must not raise
        score_lines = [l for l in text.splitlines() if l.startswith("fleet_bottleneck_score{")]
        self.assertEqual(len(score_lines), n_before, "non-finite scores must be dropped, not emitted")

    def test_snapshot_count_gauges_have_no_total_suffix(self):
        # _total is reserved for monotonic counters; these snapshot counts are gauges.
        fams, _, _ = emitted_families(make_snap())
        for n in ("fleet_bottlenecks", "fleet_sessions", "fleet_accounts"):
            self.assertIn(n, fams)
        self.assertNotIn("fleet_bottlenecks_total", fams)
        self.assertNotIn("fleet_sessions_total", fams)
        self.assertNotIn("fleet_accounts_total", fams)

    def test_bottleneck_score_has_no_value_derived_severity_label(self):
        # severity is a function of the score; as a label it churns the series every
        # threshold crossing. It must NOT appear on fleet_bottleneck_score.
        _, text, _ = emitted_families(make_snap())
        score_lines = [l for l in text.splitlines() if l.startswith("fleet_bottleneck_score{")]
        self.assertTrue(score_lines, "expected fleet_bottleneck_score series")
        for l in score_lines:
            self.assertNotIn("severity=", l, f"severity must not be a label on the score: {l!r}")
            self.assertIn("id=", l)  # stable identity is still present

    def test_bottleneck_severity_is_a_separate_numeric_gauge(self):
        # severity is exposed as a stable-keyed numeric gauge (0..4), not a label.
        fams, text, _ = emitted_families(make_snap())
        self.assertIn("fleet_bottleneck_severity", fams)
        sev_lines = [l for l in text.splitlines() if l.startswith("fleet_bottleneck_severity{")]
        self.assertTrue(sev_lines)
        for l in sev_lines:
            val = float(l.rsplit(" ", 1)[1])
            self.assertIn(val, {0.0, 1.0, 2.0, 3.0, 4.0}, f"severity must be 0..4: {l!r}")
            self.assertNotIn("severity=", l)

    def test_fleet_version_fallback_does_not_crash_engine(self):
        # the import is guarded; app_version() must always return a usable string.
        self.assertTrue(fb.fleet_version.app_version())


class WriteArtifactsTest(unittest.TestCase):
    def test_writes_prom_file(self):
        snap = make_snap()
        ranked = fb.rank_bottlenecks(snap)
        with tempfile.TemporaryDirectory() as d:
            orig = fb.REG_DIR
            fb.REG_DIR = d
            try:
                fb.write_artifacts(snap, ranked)
                prom = os.path.join(d, "fleet_bottleneck.prom")
                self.assertTrue(os.path.exists(prom))
                with open(prom, encoding="utf-8") as f:
                    self.assertIn("fleet_bottleneck_score", f.read())
                self.assertTrue(os.path.exists(os.path.join(d, "bottlenecks.json")))
                self.assertTrue(os.path.exists(os.path.join(d, "BOTTLENECKS.txt")))
            finally:
                fb.REG_DIR = orig


def _fleet_metric_names(exprs):
    """fleet_ metric names in PromQL exprs, ignoring quoted label VALUES
    (e.g. up{job="fleet_bottleneck"} is the synthetic `up` metric, not a fleet_ one)."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))   # drop quoted label values
    return set(re.findall(r"fleet_[a-z0-9_]+", joined))


def _gateway_metric_names(exprs):
    """fak_gateway_/fak_kernel_ metric names in PromQL exprs, ignoring label values."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))
    return set(re.findall(r"fak_(?:gateway|kernel)_[a-z0-9_]+", joined))


def _gateway_families():
    """fak_gateway_/fak_kernel_ metric families emitted by the gateway package
    (metrics.go: request/kernel counters; startup.go: the boot timeline)."""
    src = _read(GATEWAY_METRICS_GO) + "\n" + _read(STARTUP_METRICS_GO)
    return set(re.findall(r'"(fak_(?:gateway|kernel)_[a-z0-9_]+)"', src))


def _missing_gateway(refs, families):
    missing = set()
    for name in refs:
        if name in families:
            continue
        if any(name.endswith(suffix) and name[:-len(suffix)] in families
               for suffix in ("_bucket", "_sum", "_count")):
            continue
        missing.add(name)
    return missing


def _panel_exprs(dash):
    return [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class CrossSurfaceContractTest(unittest.TestCase):
    """The dashboard + alerts must only query metrics the engine actually emits."""

    def setUp(self):
        self.fams, _, _ = emitted_families(make_snap())
        self.gateway_fams = _gateway_families()

    def test_dashboard_queries_only_emitted_metrics(self):
        dash = json.loads(_read(FLEET_DASH))
        referenced = _fleet_metric_names(_panel_exprs(dash))
        self.assertTrue(referenced, "dashboard references no fleet_ metrics?")
        missing = referenced - self.fams
        self.assertFalse(missing, f"dashboard queries phantom metrics: {sorted(missing)}")

    def test_gateway_dashboard_queries_only_emitted_metrics(self):
        for path in (GATEWAY_DASH, DOGFOOD_DASH, STARTUP_DASH):
            with self.subTest(path=os.path.basename(path)):
                dash = json.loads(_read(path))
                referenced = _gateway_metric_names(_panel_exprs(dash))
                self.assertTrue(referenced, "gateway dashboard references no fak_gateway_/fak_kernel_ metrics?")
                missing = _missing_gateway(referenced, self.gateway_fams)
                self.assertFalse(missing, f"gateway dashboard queries phantom metrics: {sorted(missing)}")

    def test_alerts_reference_only_emitted_metrics(self):
        exprs = re.findall(r"expr:\s*(.+)", _read(ALERTS))
        fleet_refs = _fleet_metric_names(exprs)
        self.assertTrue(fleet_refs, "alerts reference no fleet_ metrics?")
        missing_fleet = fleet_refs - self.fams
        self.assertFalse(missing_fleet, f"alerts query phantom fleet metrics: {sorted(missing_fleet)}")

        gateway_refs = _gateway_metric_names(exprs)
        self.assertTrue(gateway_refs, "alerts reference no fak gateway/kernel metrics?")
        missing_gateway = _missing_gateway(gateway_refs, self.gateway_fams)
        self.assertFalse(missing_gateway, f"alerts query phantom gateway metrics: {sorted(missing_gateway)}")

    def test_dashboard_json_is_valid_and_unique_ids(self):
        dash = json.loads(_read(FLEET_DASH))
        ids = [p["id"] for p in dash["panels"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate panel ids")
        self.assertEqual(dash["uid"], "fleet-bottleneck")

    def test_gateway_dashboard_json_is_valid_and_unique_ids(self):
        for path, uid in (
            (GATEWAY_DASH, "fak-gateway-observability"),
            (DOGFOOD_DASH, "fak-dogfood-slow-requests"),
            (STARTUP_DASH, "fak-startup-load"),
        ):
            with self.subTest(path=os.path.basename(path)):
                dash = json.loads(_read(path))
                ids = [p["id"] for p in dash["panels"]]
                self.assertEqual(len(ids), len(set(ids)), "duplicate gateway panel ids")
                self.assertEqual(dash["uid"], uid)

    def test_prometheus_scrapes_gateway_metrics(self):
        prom = _read(PROMETHEUS)
        self.assertIn("job_name: fak_gateway", prom)
        self.assertIn('targets: ["host.docker.internal:8080"]', prom)


if __name__ == "__main__":
    unittest.main()
