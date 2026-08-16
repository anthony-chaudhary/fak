#!/usr/bin/env python3
"""Tests for fleet_bottleneck.py — the Top-10 scorers, the Prometheus exposition,
and the cross-surface contract (dashboard + alerts only query emitted metrics).

Run from tools/ (sibling import), e.g.  python -m unittest fleet_bottleneck_test
"""
from __future__ import annotations

import os
import re
import json
import time
import tempfile
import unittest
from unittest import mock

import fleet_bottleneck as fb

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DASH = os.path.join(HERE, "grafana", "dashboards", "fleet-bottleneck-overview.json")
GATEWAY_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-gateway-observability.json")
DOGFOOD_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-dogfood-slow-requests.json")
STARTUP_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-startup-load.json")
GUARD_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-guard-adjudication.json")
ROLLUP_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-cache-value-rollup.json")
CACHEHEALTH_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-cache-health.json")
ALERTS = os.path.join(HERE, "grafana", "prometheus-alerts.yml")
PROMETHEUS = os.path.join(HERE, "grafana", "prometheus.yml")
# fak_gateway_/fak_kernel_ families are emitted from the gateway package: the
# request/kernel counters live in metrics.go, whose exposition renderer (the quoted
# family names) was split into metrics_render.go (3f1c3c83c), plus the one-time
# boot timeline in startup.go.
GATEWAY_METRICS_GO = os.path.join(HERE, "..", "internal", "gateway", "metrics.go")
GATEWAY_RENDER_GO = os.path.join(HERE, "..", "internal", "gateway", "metrics_render.go")
STARTUP_METRICS_GO = os.path.join(HERE, "..", "internal", "gateway", "startup.go")
# fak_cachevalue_/fak_ablation_ families are emitted by the `fak cachevalue metrics`
# exposition command (NOT the gateway package): renderCachevalueExposition builds them
# via w.gauge("name", ...) call sites, so the emitted set is parsed straight from there.
CACHEVALUE_METRICS_GO = os.path.join(HERE, "..", "cmd", "fak", "cachevalue_metrics.go")
# fak_fleet_ families are emitted by the `fak fleet metrics` exposition command — the
# fleet-first exporter, the only source carrying a per-session label.
FLEET_OVERVIEW_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-fleet-overview.json")
FLEET_SESSION_DASH = os.path.join(HERE, "grafana", "dashboards", "fak-fleet-session.json")
FLEET_METRICS_GO = os.path.join(HERE, "..", "cmd", "fak", "fleet_metrics.go")


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
            "storm": {"apierr_15m": 30, "apierr_30m": 60, "apierr_60m": 90,
                      "total_15m": 40, "total_30m": 80, "total_60m": 120,
                      "apierr_per_min_30m": 2.0, "apierr_frac_30m": 0.75},
            "counts_by_disp": {"DONE": 20, "STOPPED_APIERR": 60, "LIVE": 5},
            "account_health": {
                "acctA": {"apierr_30m": 40, "newest_disp": "STOPPED_APIERR", "newest_age_min": 1.0, "live": 0},
                "acctB": {"apierr_30m": 0, "newest_disp": "DONE", "newest_age_min": 3.0, "live": 2}},
            "sessions": [],
            "hanging_detail": [{"account": "acctA", "project": "p", "session": "abc",
                                "disp": "PARKED_WAIT", "age_min": 200.0}],
        },
        "recovery": {"watchdog_log_age_min": 90.0, "resume_watchdog_age_min": 90.0,
                     "resumed_ever": 0, "transitions_log_present": True},
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
    return {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE ")}, text, ranked


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


class TokenAuditNamespaceTest(unittest.TestCase):
    def test_token_audit_uses_all_namespaces_by_default(self):
        class FakeSessionAudit:
            DEFAULT_ROOTS = ["/transcripts"]

            def __init__(self):
                self.calls = []

            def discover(self, roots, since_days=None, ns_prefix=None, include_subagents=False):
                self.calls.append({
                    "roots": roots, "since_days": since_days, "ns_prefix": ns_prefix,
                    "include_subagents": include_subagents,
                })
                return [{"path": os.path.join("/transcripts", "-Users-USER-Documents-GitHub-fleet", "s.jsonl")}]

            def analyze(self, path):
                return {
                    "path": path, "session": "abcdef123456", "assistant_turns": 2, "n_tool_use": 6,
                    "wall_s": 3600, "tokens": {"output": 1200}, "io_ratio": 42.0,
                    "cache_hit_frac": 0.73, "cost_usd": 0.5,
                }

            def aggregate(self, sess):
                return {
                    "totals": {}, "total_cost_usd": 0.5,
                    "dist": {"cache_hit_frac": {"median": 0.73}, "io_ratio": {"median": 42.0}},
                    "tool_mix": {},
                }

        fake = FakeSessionAudit()
        old = fb.session_audit
        fb.session_audit = fake
        try:
            audit = fb._token_audit(days=3, cap=10)
        finally:
            fb.session_audit = old

        self.assertEqual(fake.calls[0]["ns_prefix"], "",
                         "fleet exporter must not inherit an operator-specific namespace prefix")
        self.assertEqual(audit["n_analyzed"], 1)
        self.assertEqual(audit["top_spenders"][0]["ns"], "-Users-USER-Documents-GitHub-fleet")


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


class CliParserTest(unittest.TestCase):
    """A bare invocation (no subcommand) defaults to `report`, and main() reads
    a.no_audit / a.audit_days / a.audit_max on that path — so the shared flags must
    exist on the top-level parser too. Registering them only on the subparsers
    crashed `python fleet_bottleneck.py` with AttributeError: 'Namespace' object
    has no attribute 'no_audit'."""

    def test_bare_invocation_has_shared_defaults(self):
        a = fb._build_parser().parse_args([])
        self.assertIsNone(a.cmd)  # main() maps None -> "report"
        self.assertFalse(a.no_audit)
        self.assertEqual(a.audit_days, 1.5)
        self.assertEqual(a.audit_max, 80)

    def test_bare_invocation_accepts_shared_flags(self):
        a = fb._build_parser().parse_args(["--no-audit", "--audit-days", "3"])
        self.assertIsNone(a.cmd)
        self.assertTrue(a.no_audit)
        self.assertEqual(a.audit_days, 3.0)

    def test_subcommand_form_unchanged(self):
        a = fb._build_parser().parse_args(
            ["report", "--no-audit", "--audit-days", "3", "--audit-max", "10"])
        self.assertEqual(a.cmd, "report")
        self.assertTrue(a.no_audit)
        self.assertEqual(a.audit_days, 3.0)
        self.assertEqual(a.audit_max, 10)
        a = fb._build_parser().parse_args(["serve", "--port", "9000", "--interval", "30"])
        self.assertEqual((a.cmd, a.port, a.interval, a.no_audit), ("serve", 9000, 30, False))


class CrashResumeAttributionTest(unittest.TestCase):
    """crash_resume_backlog's freshness claim ("resume watchdog last ran Xm ago")
    must be fed by the RESUME loop's own heartbeat, not the cross-loop minimum —
    a live supervisor (watchdog.log) masked a ~20h-dead resume watchdog as
    "last ran 1m ago", telling the operator the exact opposite of the truth."""

    def _backlog(self, snap):
        return {b["id"]: b for b in fb.rank_bottlenecks(snap)["bottlenecks"]}["crash_resume_backlog"]

    def test_stale_resume_reported_stale_despite_live_supervisor(self):
        snap = make_snap()
        snap["recovery"]["watchdog_log_age_min"] = 1.0        # supervisor ticked 1m ago
        snap["recovery"]["resume_watchdog_age_min"] = 1200.0  # resume loop dead ~20h
        b = self._backlog(snap)
        self.assertIn("1200m", b["symptom"])
        self.assertIn("STALE", b["symptom"])
        self.assertEqual(b["evidence"]["resume_watchdog_age_min"], 1200.0)
        self.assertEqual(b["score"], 22 * 2 + 20, "stale-resume boost must apply")

    def test_fresh_resume_reports_its_own_age(self):
        snap = make_snap()
        snap["recovery"]["watchdog_log_age_min"] = 1.0
        snap["recovery"]["resume_watchdog_age_min"] = 2.0
        b = self._backlog(snap)
        self.assertIn("resume watchdog last ran 2m ago", b["symptom"])
        self.assertNotIn("STALE", b["symptom"])
        self.assertEqual(b["score"], 22 * 2)

    def test_absent_resume_heartbeat_not_masked_by_supervisor(self):
        snap = make_snap()
        snap["recovery"].pop("resume_watchdog_age_min", None)
        snap["recovery"]["watchdog_log_age_min"] = 1.0
        b = self._backlog(snap)
        self.assertIn("not seen on disk", b["symptom"])
        self.assertEqual(b["score"], 22 * 2 + 20)


class PrometheusExpositionTest(unittest.TestCase):
    def test_exposition_is_well_formed(self):
        fams, text, _ = emitted_families(make_snap())
        help_names = {line.split()[2] for line in text.splitlines() if line.startswith("# HELP ")}
        self.assertEqual(fams, help_names, "every TYPE must have a matching HELP")
        line_re = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9].*$')
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            self.assertRegex(line, line_re, f"malformed sample line: {line!r}")
            base = line.split("{")[0].split(" ")[0]
            self.assertIn(base, fams, f"series {base} has no TYPE")

    def test_expected_families_present(self):
        fams, _, _ = emitted_families(make_snap())
        for n in ("fleet_up", "fleet_health_state", "fleet_bottleneck_score",
                  "fleet_surface_backlog", "fleet_api_error_stalls",
                  "fleet_workers_active_per_account", "fleet_cost_usd",
                  "fleet_cache_hit_ratio_median", "fleet_top_spender_output_tokens",
                  "fleet_snapshot_timestamp_seconds", "fleet_registry_age_minutes",
                  "fleet_watchdog_log_age_minutes", "fleet_resume_watchdog_age_minutes"):
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
        score_lines = [line for line in text.splitlines() if line.startswith("fleet_bottleneck_score{")]
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
        score_lines = [line for line in text.splitlines() if line.startswith("fleet_bottleneck_score{")]
        self.assertTrue(score_lines, "expected fleet_bottleneck_score series")
        for line in score_lines:
            self.assertNotIn("severity=", line, f"severity must not be a label on the score: {line!r}")
            self.assertIn("id=", line)  # stable identity is still present

    def test_bottleneck_severity_is_a_separate_numeric_gauge(self):
        # severity is exposed as a stable-keyed numeric gauge (0..4), not a label.
        fams, text, _ = emitted_families(make_snap())
        self.assertIn("fleet_bottleneck_severity", fams)
        sev_lines = [line for line in text.splitlines() if line.startswith("fleet_bottleneck_severity{")]
        self.assertTrue(sev_lines)
        for line in sev_lines:
            val = float(line.rsplit(" ", 1)[1])
            self.assertIn(val, {0.0, 1.0, 2.0, 3.0, 4.0}, f"severity must be 0..4: {line!r}")
            self.assertNotIn("severity=", line)

    def test_appversion_fallback_does_not_crash_engine(self):
        # the import is guarded; app_version() must always return a usable string.
        self.assertTrue(fb.appversion.app_version())


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
    (e.g. up{job="fleet_bottleneck"} is the synthetic `up` metric, not a fleet_ one).

    The lookbehind keeps this scoped to the PYTHON engine's namespace: without it,
    `fak_fleet_sessions_by_liveness` (the Go exporter's family) also matches from its
    fourth character and is then reported as a phantom fleet_ metric."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))   # drop quoted label values
    return set(re.findall(r"(?<![a-z0-9_])fleet_[a-z0-9_]+", joined))


def _gateway_metric_names(exprs):
    """fak_gateway_/fak_kernel_ metric names in PromQL exprs, ignoring label values."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))
    return set(re.findall(r"fak_(?:gateway|kernel)_[a-z0-9_]+", joined))


def _gateway_families():
    """fak_gateway_/fak_kernel_ metric families emitted by the gateway package
    (metrics.go + its split-out exposition renderer metrics_render.go: request/kernel
    counters; startup.go: the boot timeline)."""
    src = "\n".join(_read(p) for p in (GATEWAY_METRICS_GO, GATEWAY_RENDER_GO, STARTUP_METRICS_GO))
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


def _cachevalue_metric_names(exprs):
    """fak_cachevalue_/fak_ablation_ metric names in PromQL exprs, ignoring quoted label
    VALUES (so up{job="fak_cachevalue"} is the synthetic `up` metric, not a family)."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))
    return set(re.findall(r"fak_(?:cachevalue|ablation)_[a-z0-9_]+", joined))


def _cachevalue_families():
    """fak_cachevalue_/fak_ablation_ families emitted by the exposition command
    (cmd/fak/cachevalue_metrics.go), parsed from its w.gauge("...") call sites — the
    same emitter-source technique _gateway_families() uses for the gateway package."""
    src = _read(CACHEVALUE_METRICS_GO)
    return set(re.findall(r'w\.gauge\(\s*"(fak_(?:cachevalue|ablation)_[a-z0-9_]+)"', src))


def _fleet_exporter_metric_names(exprs):
    """fak_fleet_ metric names in PromQL exprs, ignoring quoted label VALUES (so
    up{job="fak_fleet"} is the synthetic `up` metric, not a fak_fleet_ family)."""
    joined = re.sub(r'"[^"]*"', "", " ; ".join(exprs))
    return set(re.findall(r"fak_fleet_[a-z0-9_]+", joined))


def _fleet_exporter_families():
    """fak_fleet_ families emitted by `fak fleet metrics` (cmd/fak/fleet_metrics.go).

    Two emission SHAPES have to be read, because the exporter emits family-major (all
    samples of a family contiguous), which means the family name is not always a literal
    at the w.gauge call site:

      1. direct literals -- w.gauge("fak_fleet_sessions", ...) and the {name, help, fn}
         table rows for the live per-session tier;
      2. prefix + suffix -- the historical tier writes one TIER per name prefix
         ("fak_fleet_usage_", "fak_fleet_usage_by_type_", "fak_fleet_usage_session_")
         crossed with the family suffixes in emitFleetUsageFamilies, so the fleet total
         can never be summed together with its own breakdown by accident.

    Both are parsed here, and the cross product is expanded, so a panel querying a
    phantom family still fails this test.
    """
    src = _read(FLEET_METRICS_GO)
    names = set(re.findall(r'"(fak_fleet_[a-z0-9_]*[a-z0-9])"', src))
    prefixes = set(re.findall(r'"(fak_fleet_[a-z0-9_]*_)"', src))
    # suffixes: both the literal prefix+"..." call sites and the {suffix, help, fn} table
    suffixes = set(re.findall(r'prefix\+"([a-z0-9_]+)"', src))
    suffixes |= set(re.findall(r'\{"([a-z0-9_]+)",\s*"', src))
    return names | {p + s for p in prefixes for s in suffixes}


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

    def test_storm_families_emitted(self):
        """The crash-loop storm surface (recency buckets + per-min rate + per-account
        attribution) must be exported so Grafana can graph the storm over time — the
        signal a single 'active workers' gauge hides."""
        want = {"fleet_apierr_recent", "fleet_apierr_per_min", "fleet_apierr_frac_30m",
                "fleet_sessions_by_disp", "fleet_account_apierr_recent"}
        missing = want - self.fams
        self.assertFalse(missing, f"storm metrics not emitted: {sorted(missing)}")

    def test_storm_rate_gauges_emit_zero_when_healthy(self):
        """apierr_per_min / apierr_frac_30m / apierr_recent must emit 0 (not be absent)
        when there is no storm, so a healthy fleet is distinguishable from a dead
        exporter instead of both reading 'No data'."""
        snap = make_snap()
        snap["registry"].pop("storm", None)
        fams, text, _ = emitted_families(snap)
        for name in ("fleet_apierr_per_min", "fleet_apierr_frac_30m", "fleet_apierr_recent"):
            self.assertIn(name, fams, f"{name} must still emit when healthy")
        self.assertIn("fleet_apierr_per_min 0", text)

    def test_gateway_dashboard_queries_only_emitted_metrics(self):
        for path in (GATEWAY_DASH, DOGFOOD_DASH, STARTUP_DASH, GUARD_DASH,
                     CACHEHEALTH_DASH):
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
            (GUARD_DASH, "fak-guard-adjudication"),
            (CACHEHEALTH_DASH, "fak-cache-health"),
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

    def test_prometheus_scrapes_fleet_metrics(self):
        prom = _read(PROMETHEUS)
        self.assertIn("job_name: fak_fleet", prom)
        self.assertIn('targets: ["host.docker.internal:9098"]', prom)

    def test_fleet_dashboards_query_only_emitted_metrics(self):
        """The fleet pair must not query a phantom fak_fleet_ family. This is the
        lock-step that lets the exporter be refactored (renaming a family, moving a
        tier behind a new prefix) without silently blanking a panel."""
        emitted = _fleet_exporter_families()
        self.assertIn("fak_fleet_session_info", emitted,
                      "family extractor did not find the per-session _info family?")
        for path in (FLEET_OVERVIEW_DASH, FLEET_SESSION_DASH):
            with self.subTest(path=os.path.basename(path)):
                dash = json.loads(_read(path))
                referenced = _fleet_exporter_metric_names(_panel_exprs(dash))
                self.assertTrue(referenced, "fleet dashboard references no fak_fleet_ metrics?")
                missing = referenced - emitted
                self.assertFalse(missing, f"fleet dashboard queries phantom metrics: {sorted(missing)}")

    def test_fleet_alerts_reference_only_emitted_metrics(self):
        exprs = re.findall(r"expr:\s*(.+)", _read(ALERTS))
        refs = _fleet_exporter_metric_names(exprs)
        self.assertTrue(refs, "alerts reference no fak_fleet_ metrics?")
        missing = refs - _fleet_exporter_families()
        self.assertFalse(missing, f"alerts query phantom fleet-exporter metrics: {sorted(missing)}")

    def test_fleet_dashboard_json_is_valid_and_unique_ids(self):
        for path, uid in ((FLEET_OVERVIEW_DASH, "fak-fleet-overview"),
                          (FLEET_SESSION_DASH, "fak-fleet-session")):
            with self.subTest(path=os.path.basename(path)):
                dash = json.loads(_read(path))
                ids = [p["id"] for p in dash["panels"]]
                self.assertEqual(len(ids), len(set(ids)), "duplicate fleet panel ids")
                self.assertEqual(dash["uid"], uid)

    def test_fleet_overview_drills_into_the_session_dashboard(self):
        """The two dashboards are only a PAIR if the overview actually links into the
        drill-down with the clicked session id. A data link that lost its var-session
        parameter would open an empty drill-down and look like missing data."""
        dash = json.loads(_read(FLEET_OVERVIEW_DASH))
        links = [lk
                 for p in dash["panels"]
                 for ov in p.get("fieldConfig", {}).get("overrides", [])
                 for prop in ov.get("properties", [])
                 if prop.get("id") == "links"
                 for lk in prop.get("value", [])]
        self.assertTrue(links, "no data link on the overview's session table")
        self.assertTrue(any("var-session=" in lk.get("url", "") and
                            "/d/fak-fleet-session" in lk.get("url", "") for lk in links),
                        f"session table links do not carry the session into the drill-down: {links}")

    def test_fleet_session_dashboard_has_a_session_variable(self):
        dash = json.loads(_read(FLEET_SESSION_DASH))
        names = {v["name"] for v in dash["templating"]["list"]}
        self.assertIn("session", names,
                      "the drill-down needs a `session` template variable to be drillable")
        var = next(v for v in dash["templating"]["list"] if v["name"] == "session")
        self.assertIn("fak_fleet_session_info", json.dumps(var),
                      "the session selector must be sourced from the LIVE _info family")


class RecoveryFreshnessTest(unittest.TestCase):
    """collect()'s recovery-freshness probe must read the LIVE watchdog heartbeat
    (resume_watchdog.log under %FLEET_STATE_DIR%/watchdog), not just the in-repo
    tools/_watchdog/watchdog.log. The old probe read one in-repo file written only by
    the supervisor watchdog, so whenever the supervisor was disabled but the resume
    watchdog was the live recovery plumbing it returned None -> two false CRITICAL
    bottlenecks (crash-resume backlog boost + recovery-plumbing-stale)."""

    def _age(self, state_dir, repo_watch_dir):
        with mock.patch.dict(os.environ, {"FLEET_STATE_DIR": state_dir}, clear=False), \
                mock.patch.object(fb, "WATCH_DIR", repo_watch_dir):
            return fb._recovery_watchdog_age_min()

    def _resume_age(self, state_dir, repo_watch_dir):
        with mock.patch.dict(os.environ, {"FLEET_STATE_DIR": state_dir}, clear=False), \
                mock.patch.object(fb, "WATCH_DIR", repo_watch_dir):
            return fb._resume_watchdog_age_min()

    def test_reads_live_resume_watchdog_heartbeat(self):
        # The bug case: the live resume watchdog heartbeats into the host state dir
        # under resume_watchdog.log; the old probe (in-repo watchdog.log) saw nothing.
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            wd = os.path.join(state, "watchdog")
            os.makedirs(wd)
            open(os.path.join(wd, "resume_watchdog.log"), "w").close()
            age = self._age(state, repo)
            self.assertIsNotNone(age)   # old probe returned None here
            self.assertLess(age, 5.0)

    def test_freshest_across_names_and_dirs_wins(self):
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            wd = os.path.join(state, "watchdog")
            os.makedirs(wd)
            stale = os.path.join(wd, "watchdog.log")
            fresh = os.path.join(repo, "resume_watchdog.log")
            open(stale, "w").close()
            os.utime(stale, (time.time() - 3600, time.time() - 3600))  # 60m old
            open(fresh, "w").close()                                    # now
            self.assertLess(self._age(state, repo), 5.0)

    def test_none_when_no_heartbeat_anywhere(self):
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(state, "watchdog"))
            self.assertIsNone(self._age(state, repo))
            self.assertIsNone(self._resume_age(state, repo))

    def test_resume_probe_not_masked_by_live_supervisor(self):
        # The two watchdogs are DIFFERENT recovery loops: a fresh supervisor
        # heartbeat (watchdog.log) must not make a ~20h-dead resume watchdog read
        # fresh. The cross-loop min stays available via _recovery_watchdog_age_min
        # (the "is anything self-healing?" signal), but the resume probe sees only
        # resume_watchdog.log.
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            wd = os.path.join(state, "watchdog")
            os.makedirs(wd)
            open(os.path.join(wd, "watchdog.log"), "w").close()   # supervisor: now
            dead = os.path.join(wd, "resume_watchdog.log")        # resume: 20h old
            open(dead, "w").close()
            t = time.time() - 20 * 3600
            os.utime(dead, (t, t))
            self.assertGreater(self._resume_age(state, repo), 60.0)
            self.assertLess(self._age(state, repo), 5.0)  # plumbing-wide still fresh

    def test_resume_probe_freshest_across_locations(self):
        # Same-loop cross-LOCATION min is preserved: the resume watchdog's in-repo
        # dev-path heartbeat still counts when the state-dir copy is stale.
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            wd = os.path.join(state, "watchdog")
            os.makedirs(wd)
            stale = os.path.join(wd, "resume_watchdog.log")
            open(stale, "w").close()
            t = time.time() - 3600
            os.utime(stale, (t, t))
            open(os.path.join(repo, "resume_watchdog.log"), "w").close()  # now
            self.assertLess(self._resume_age(state, repo), 5.0)

    def test_resume_probe_none_when_only_supervisor_present(self):
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as repo:
            wd = os.path.join(state, "watchdog")
            os.makedirs(wd)
            open(os.path.join(wd, "watchdog.log"), "w").close()
            self.assertIsNone(self._resume_age(state, repo))
            self.assertIsNotNone(self._age(state, repo))

    def test_fresh_recovery_suppresses_false_recovery_stale(self):
        # End-to-end contract: a fresh probe value clears the recovery-plumbing-stale
        # bottleneck; a None value (the old blind probe) raises it falsely.
        snap = make_snap()
        snap["recovery"]["watchdog_log_age_min"] = 5.0
        self.assertNotIn("recovery_stale",
                         {b["id"] for b in fb.rank_bottlenecks(snap)["bottlenecks"]})
        snap["recovery"]["watchdog_log_age_min"] = None
        self.assertIn("recovery_stale",
                      {b["id"] for b in fb.rank_bottlenecks(snap)["bottlenecks"]})


class CacheValueRollupContractTest(unittest.TestCase):
    """The cache-value & ablation roll-up dashboard must only query metrics the `fak
    cachevalue metrics` exposition actually emits. Its families come from that command
    (parsed from its w.gauge call sites), NOT the gateway package — so it is checked
    against its own emitter source and stays out of the gateway cross-surface loop."""

    def setUp(self):
        self.emitted = _cachevalue_families()

    def test_emitter_families_parsed(self):
        # Guard against a silently-empty parse making the subset checks vacuous: the
        # emitter declares ~35 families, and these load-bearing ones must be among them.
        self.assertGreater(len(self.emitted), 20, "parsed too few emitted families")
        for n in ("fak_cachevalue_cumulative_net_usd", "fak_cachevalue_saved_token_equiv",
                  "fak_cachevalue_latest_reuse_ratio", "fak_cachevalue_measured",
                  "fak_ablation_arm_speedup_ratio", "fak_ablation_report_present"):
            self.assertIn(n, self.emitted, f"{n} not found among emitted families")

    def test_rollup_dashboard_queries_only_emitted_metrics(self):
        dash = json.loads(_read(ROLLUP_DASH))
        referenced = _cachevalue_metric_names(_panel_exprs(dash))
        self.assertTrue(referenced,
                        "rollup dashboard references no fak_cachevalue_/fak_ablation_ metrics?")
        missing = referenced - self.emitted
        self.assertFalse(missing, f"rollup dashboard queries phantom metrics: {sorted(missing)}")

    def test_rollup_spans_both_tracks_and_ablation(self):
        # The dashboard's whole point is the two-surface story: cache-value P&L AND the
        # ablation arms. Assert both surfaces are actually charted.
        dash = json.loads(_read(ROLLUP_DASH))
        referenced = _cachevalue_metric_names(_panel_exprs(dash))
        self.assertTrue(any(n.startswith("fak_cachevalue_") for n in referenced),
                        "rollup must chart the cache-value tracks")
        self.assertTrue(any(n.startswith("fak_ablation_") for n in referenced),
                        "rollup must chart the ablation arms")

    def test_rollup_dashboard_valid_and_unique_ids(self):
        dash = json.loads(_read(ROLLUP_DASH))
        ids = [p["id"] for p in dash["panels"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate rollup panel ids")
        self.assertEqual(dash["uid"], "fak-cache-value-rollup")

    def test_prometheus_scrapes_cachevalue_metrics(self):
        # The dashboard's scrape-health panel reads up{job="fak_cachevalue"}; the scrape
        # job must exist or every panel is dead on arrival.
        self.assertIn("job_name: fak_cachevalue", _read(PROMETHEUS))


if __name__ == "__main__":
    unittest.main()
