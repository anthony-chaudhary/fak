#!/usr/bin/env python3
r"""gen_dashboard.py — generate the provisioned Grafana dashboard JSON.

The fleet analog of metrics-service's `internal/grafana/dashboard.go`: build
dashboards as Python dicts and dump valid JSON, so panel `expr`s stay in lock-step
with the metric names emitted by the observability sources:

  - fleet_bottleneck.py /metrics (namespace `fleet_`)
  - fak serve /metrics (namespaces `fak_gateway_` and `fak_kernel_`)

Re-run after changing either metric set:

    python tools/grafana/gen_dashboard.py     # writes dashboards/fleet-bottleneck-overview.json
                                              # and dashboards/fak-gateway-observability.json
                                              # and dashboards/fak-dogfood-slow-requests.json
                                              # and dashboards/fak-startup-load.json
                                              # and dashboards/fak-guard-adjudication.json

The committed JSON is what Grafana provisions; this generator is the maintainable
source. `METRICS` / `GATEWAY_METRICS` below are the contract —
`fleet_bottleneck_test.py` asserts dashboard/alert references against emitted
families.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dashboards", "fleet-bottleneck-overview.json")
OUT_GATEWAY = os.path.join(HERE, "dashboards", "fak-gateway-observability.json")
OUT_DOGFOOD = os.path.join(HERE, "dashboards", "fak-dogfood-slow-requests.json")
OUT_STARTUP = os.path.join(HERE, "dashboards", "fak-startup-load.json")
OUT_GUARD = os.path.join(HERE, "dashboards", "fak-guard-adjudication.json")

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}

# Every metric the dashboard queries. Kept as a manifest so the test can assert
# both directions against render_prometheus (no panel queries a phantom metric;
# every emitted family is reachable from a panel or deliberately listed here).
METRICS = [
    "fleet_health_state", "fleet_headline_score", "fleet_bottlenecks",
    "fleet_bottleneck_score", "fleet_workers_active", "fleet_accounts_throttled",
    "fleet_workers_auth_blocked", "fleet_registry_age_minutes",
    "fleet_workers_hanging", "fleet_workers_on_throttled_account",
    "fleet_resume_queue", "fleet_surface_backlog", "fleet_api_error_stalls",
    "fleet_workers_active_per_account", "fleet_cost_usd",
    "fleet_cost_per_active_session_hour", "fleet_cache_hit_ratio_median",
    "fleet_io_ratio_median", "fleet_audit_sessions",
    "fleet_top_spender_output_tokens", "fleet_worst_cache_hit_ratio",
]

GATEWAY_METRICS = [
    "fak_gateway_up", "fak_gateway_start_time_seconds",
    "fak_gateway_inflight_requests", "fak_gateway_inflight_max_age_seconds",
    "fak_gateway_inflight_requests_by_route", "fak_gateway_build_info",
    "fak_gateway_http_requests_total",
    "fak_gateway_http_request_duration_seconds",
    "fak_gateway_operations_total",
    "fak_gateway_operation_duration_seconds",
    "fak_gateway_vdso_hit_ratio",
    "fak_kernel_submits_total", "fak_kernel_vdso_hits_total",
    "fak_kernel_engine_calls_total", "fak_kernel_denies_total",
    "fak_kernel_transforms_total", "fak_kernel_quarantines_total",
    "fak_kernel_admitted_total",
]

# The one-time boot timeline + weight-load families (internal/gateway/startup.go).
# Held as gauges at their once-at-boot values for process life — see startup.go.
STARTUP_METRICS = [
    "fak_gateway_time_to_ready_seconds", "fak_gateway_ready_time_seconds",
    "fak_gateway_start_time_seconds", "fak_gateway_startup_phase_duration_seconds",
    "fak_gateway_startup_unaccounted_seconds",
    "fak_model_load_duration_seconds", "fak_model_load_bytes",
    "fak_model_load_tensors", "fak_model_load_info",
    "fak_model_load_phase_duration_seconds", "fak_model_load_phase_bytes",
    "fak_model_load_phase_tensors",
]

# The `fak guard` session view. Guard exposes the SAME /metrics surface fak serve
# does (it IS the gateway, bound on a private loopback port), so the guard dashboard
# is built entirely from already-emitted families — no new metric is introduced. It
# foregrounds the adjudication-referee story the gateway dashboard does not: what the
# kernel allowed vs blocked (by the closed refusal vocabulary), what the cache hop
# saved (the NET fak_vcache_* economics), and the history-compaction / tool-floor
# shedding — the live, in-Grafana twin of the summary `fak guard` prints on exit.
GUARD_METRICS = [
    "fak_gateway_up", "fak_gateway_operations_total",
    "fak_gateway_operation_duration_seconds",
    "fak_gateway_context_pollutions_blocked_total",
    "fak_gateway_inference_cached_prompt_tokens_total",
    "fak_vcache_saved_token_equiv", "fak_vcache_multiplier", "fak_vcache_hit_rate",
    "fak_vcache_proven", "fak_vcache_baseline_token_equiv",
    "fak_vcache_actual_token_equiv", "fak_vcache_cache_creation_tokens_total",
    "fak_gateway_compaction_attempts_total", "fak_gateway_compaction_bail_reason_total",
    "fak_gateway_compaction_shed_tokens_total",
    "fak_gateway_inbound_tools_pruned_total",
    "fak_gateway_upstream_errors_total", "fak_gateway_upstream_retries_total",
    "fak_gateway_turns_saved_total",
]

_id = [0]
def nid():
    _id[0] += 1
    return _id[0]


def steps(*pairs):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in pairs]}


def row(title, y):
    return {"type": "row", "title": title, "id": nid(), "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def stat(title, expr, x, y, w=4, h=4, unit="short", thresholds=None, mappings=None,
         desc="", color="thresholds", legend=None):
    defaults = {"color": {"mode": color}, "mappings": mappings or [],
                "thresholds": thresholds or steps((None, "green")), "unit": unit}
    tgt = {"datasource": DS, "expr": expr, "refId": "A"}
    if legend is not None:
        tgt["legendFormat"] = legend
    return {
        "datasource": DS, "title": title, "description": desc, "type": "stat",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"colorMode": "value", "graphMode": "area", "justifyMode": "auto",
                    "orientation": "auto", "textMode": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "pluginVersion": "11.5.2", "targets": [tgt],
    }


def bargauge(title, expr, legend, x, y, w=12, h=8, unit="short", thresholds=None,
             desc="", maxv=None):
    defaults = {"color": {"mode": "thresholds"}, "mappings": [],
                "thresholds": thresholds or steps((None, "green")), "unit": unit}
    if maxv is not None:
        defaults["max"] = maxv
        defaults["min"] = 0
    return {
        "datasource": DS, "title": title, "description": desc, "type": "bargauge",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"displayMode": "gradient", "orientation": "horizontal",
                    "showUnfilled": True, "minVizHeight": 10, "minVizWidth": 0,
                    "valueMode": "color",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "pluginVersion": "11.5.2",
        "targets": [{"datasource": DS, "expr": expr, "legendFormat": legend,
                     "instant": True, "refId": "A"}],
    }


def timeseries(title, targets, x, y, w=24, h=8, unit="short", desc=""):
    return {
        "datasource": DS, "title": title, "description": desc, "type": "timeseries",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineWidth": 1, "fillOpacity": 10,
                       "showPoints": "never", "axisPlacement": "auto",
                       "spanNulls": True, "pointSize": 5, "stacking": {"mode": "none"}},
            "unit": unit, "mappings": [], "thresholds": steps((None, "green"))},
            "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "calcs": []},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "pluginVersion": "11.5.2",
        "targets": [{"datasource": DS, "expr": e, "legendFormat": lg, "refId": chr(65 + i)}
                    for i, (e, lg) in enumerate(targets)],
    }


def text_panel(title, content, x, y, w=24, h=6, desc=""):
    return {
        "datasource": DS, "title": title, "description": desc, "type": "text",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"mode": "markdown", "content": content},
        "pluginVersion": "11.5.2",
    }


def table(title, expr, x, y, w=12, h=9, desc=""):
    # The score column is colored by the severity-band thresholds (SCORE_TH:
    # green/blue/yellow/red = OK/MEDIUM/HIGH/CRITICAL), so severity reads off the
    # color without a `severity` label on the series (which would churn the time
    # series every threshold crossing — see render_prometheus). The numeric
    # fleet_bottleneck_severity metric carries severity for PromQL/alerts.
    return {
        "datasource": DS, "title": title, "description": desc, "type": "table",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {"custom": {"align": "auto", "filterable": True},
                                     "mappings": [], "thresholds": steps((None, "green"))},
                        "overrides": [{
                            "matcher": {"id": "byName", "options": "score"},
                            "properties": [
                                {"id": "thresholds", "value": SCORE_TH},
                                {"id": "custom.cellOptions", "value": {"type": "color-text"}}]}]},
        "options": {"showHeader": True, "sortBy": [{"displayName": "score", "desc": True}]},
        "pluginVersion": "11.5.2",
        "targets": [{"datasource": DS, "expr": expr, "format": "table",
                     "instant": True, "refId": "A"}],
        "transformations": [{"id": "organize", "options": {
            # drop Prometheus bookkeeping + scrape labels (instance/job/fleet) so the
            # table is a clean title | layer | score.
            "excludeByName": {"Time": True, "__name__": True, "id": True,
                              "instance": True, "job": True, "fleet": True},
            "renameByName": {"Value": "score"},
            "indexByName": {"title": 0, "layer": 1, "score": 2}}}],
    }


def info_table(title, expr, x, y, w=24, h=6, desc="", exclude=None):
    """A plain label-table: render an instant series' labels as columns, dropping
    Prometheus bookkeeping + the (always-1) Value. Used for the *_info gauge."""
    drop = {"Time": True, "Value": True, "__name__": True, "job": True, "instance": True}
    for name in (exclude or []):
        drop[name] = True
    return {
        "datasource": DS, "title": title, "description": desc, "type": "table",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {
            "color": {"mode": "thresholds"},
            "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
            "mappings": [], "thresholds": steps((None, "green"))}, "overrides": []},
        "options": {"showHeader": True, "cellHeight": "sm",
                    "footer": {"show": False, "reducer": ["sum"], "fields": ""}},
        "pluginVersion": "11.5.2",
        "targets": [{"datasource": DS, "expr": expr, "format": "table",
                     "instant": True, "refId": "A"}],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": drop, "renameByName": {}}}],
    }


SCORE_TH = steps((None, "green"), (30, "blue"), (55, "yellow"), (80, "red"))
HEALTH_MAP = [{"type": "value", "options": {
    "0": {"text": "OK", "color": "green", "index": 0},
    "1": {"text": "DEGRADED", "color": "yellow", "index": 1},
    "2": {"text": "CRITICAL", "color": "red", "index": 2},
    "3": {"text": "DOWN", "color": "dark-red", "index": 3}}}]
AGE_TH = steps((None, "green"), (15, "yellow"), (30, "red"))
ONE_RED = steps((None, "green"), (1, "red"))
UP_TH = steps((None, "red"), (1, "green"))
CACHE_TH = steps((None, "red"), (0.8, "yellow"), (0.9, "green"))  # higher cache-hit = better
LAT_TH = steps((None, "green"), (0.5, "yellow"), (2, "red"))
ERR_TH = steps((None, "green"), (0.01, "yellow"), (0.05, "red"))


# Shared first-run note, shown at the top of each dashboard that has panels which
# stay "No data" until a specific subsystem is exercised. Markdown text panels carry
# no PromQL, so they always render - they are the in-dashboard explanation issue #309
# asks for (why a panel is empty on the up.sh first-run path, and what to run).
AUDIT_NOTE = (
    "Populated by `session_audit` over `~/.claude/projects`. Empty if no transcripts "
    "match the namespace filter (default `C--work`); run "
    "`python tools/session_audit.py discover --all` to check."
)
MODEL_LOAD_NOTE = (
    "Emitted only when `fak serve` is started with `--gguf`. The default `up.sh` path "
    "uses the inkernel engine against a fak-format export, so these stay empty unless "
    "you re-launch with `--gguf MODEL.gguf`."
)
DOGFOOD_NOTE = (
    "Filters `route=\"/v1/messages\"` (Claude Code traffic). Drive them by pointing a "
    "Claude Code session at `fak serve`; the `/v1/fak/syscall` smoke request does not "
    "appear here."
)


def build():
    panels = []

    panels.append(text_panel(
        "First run: why some panels read No data",
        "Bringing the stack up with `tools/grafana/up.sh` and driving the hinted "
        "`/v1/fak/syscall` smoke request populates the gateway counters but NOT the "
        "token-spend tiles below. Those read a **session audit** over local Claude "
        "Code transcripts:\n\n"
        "- **Token spend (cost & cache) row** (Window cost, Cost/session-hr, Median "
        "cache-hit, Median I:O, Transcripts analyzed, Top spenders, Lowest cache-hit): "
        + AUDIT_NOTE + "\n\n"
        "Everything else here is fed by `fleet_bottleneck.py /metrics` and fills in as "
        "the fleet runs. No panel is broken; the empty ones are waiting on the audit "
        "pass, not on traffic.",
        0, 0, w=24, h=5,
        desc="Why the token-spend tiles can read No data on a fresh up.sh stack, and "
             "how to populate them."))
    panels.append(row("Fleet health", 5))
    panels.append(stat("Health", "fleet_health_state", 0, 6, color="thresholds",
                       mappings=HEALTH_MAP, thresholds=steps((None, "green"), (1, "yellow"), (2, "red")),
                       desc="0 OK · 1 DEGRADED · 2 CRITICAL · 3 DOWN (no telemetry)."))
    panels.append(stat("#1 bottleneck score", "fleet_headline_score", 4, 6,
                       thresholds=SCORE_TH, desc="Score of the most-likely actual limiter right now."))
    panels.append(stat("Active workers", "fleet_workers_active", 8, 6,
                       desc="In-flight workers (not cleanly finished)."))
    panels.append(stat("Accounts throttled", "fleet_accounts_throttled", 12, 6,
                       thresholds=ONE_RED, desc="Accounts currently rate-limited."))
    panels.append(stat("Auth-blocked workers", "fleet_workers_auth_blocked", 16, 6,
                       thresholds=ONE_RED, desc="Workers that need /login (a human must re-auth)."))
    panels.append(stat("Telemetry age", "fleet_registry_age_minutes", 20, 6, unit="m",
                       thresholds=AGE_TH, desc="Age of the session registry — stale = flying blind (#8)."))

    panels.append(row("Ranked bottlenecks (the Top 10)", 10))
    panels.append(bargauge(
        "Bottleneck scores (ranked)", "fleet_bottleneck_score", "{{title}}", 0, 11,
        w=12, h=9, maxv=100, thresholds=SCORE_TH,
        desc="Every scored class, 0-100 - likelihood it is the actual limiter. Empty "
             "until at least one of the Top 10 classes fires; a fresh fleet with no "
             "active workers scores nothing, which is healthy, not broken."))
    panels.append(table(
        "Bottleneck detail", "fleet_bottleneck_score", 12, 11, w=12, h=9,
        desc="Title / layer / score (score color = severity band) for each scored "
             "class. Empty until a bottleneck class fires."))
    panels.append(timeseries(
        "Bottleneck score history", [("fleet_bottleneck_score", "{{title}}")], 0, 20,
        w=24, h=8, desc="How each class's score moves over time. A flat/empty chart "
                        "means no class has scored over the window."))

    panels.append(row("Capacity & accounts", 28))
    panels.append(timeseries(
        "Worker states over time", [
            ("fleet_workers_active", "active"),
            ("fleet_workers_hanging", "hanging"),
            ("fleet_workers_auth_blocked", "auth-blocked"),
            ("fleet_workers_on_throttled_account", "on throttled acct"),
            ("fleet_resume_queue", "resume queue"),
            ("fleet_surface_backlog", "surface backlog"),
            ("fleet_api_error_stalls", "api-error stalls"),
        ], 0, 29, w=16, h=8, desc="The in-flight worker buckets that drive #1-#10."))
    panels.append(bargauge(
        "Active workers per account", "fleet_workers_active_per_account", "{{account}}",
        16, 29, w=8, h=8, desc="Concentration on one account = imbalance (#7) + throttle blast radius (#1)."))

    panels.append(row("Token spend (cost & cache)", 37))
    panels.append(stat("Window cost", "fleet_cost_usd", 0, 38, unit="currencyUSD",
                       desc="Estimated token spend over the audit window (assumed "
                            "pricing). " + AUDIT_NOTE))
    panels.append(stat("Cost / session-hr", "fleet_cost_per_active_session_hour", 4, 38,
                       unit="currencyUSD", desc="Estimated cost per active session-hour. "
                       + AUDIT_NOTE))
    panels.append(stat("Median cache-hit", "fleet_cache_hit_ratio_median", 8, 38,
                       unit="percentunit", thresholds=CACHE_TH,
                       desc="Median per-session prompt-cache-hit fraction - low = token "
                            "waste (#6). " + AUDIT_NOTE))
    panels.append(stat("Median I:O", "fleet_io_ratio_median", 12, 38,
                       desc="Median per-session input:output token ratio. " + AUDIT_NOTE))
    panels.append(stat("Transcripts analyzed", "fleet_audit_sessions", 16, 38,
                       desc="Sessions in the token-spend pass. 0/No data when the audit "
                            "matched nothing. " + AUDIT_NOTE))
    panels.append(stat("Bottlenecks scored", "fleet_bottlenecks", 20, 38,
                       desc="How many of the Top 10 classes are firing now."))
    panels.append(bargauge(
        "Top spenders (output tokens)", "fleet_top_spender_output_tokens", "{{session}}",
        0, 43, w=12, h=8, desc="The heaviest sessions by output tokens (leaderboard). "
        + AUDIT_NOTE))
    panels.append(bargauge(
        "Lowest cache-hit sessions", "fleet_worst_cache_hit_ratio", "{{session}}",
        12, 43, w=12, h=8, unit="percentunit", thresholds=CACHE_TH,
        desc="Token-waste suspects - sessions with the lowest cache reuse (#6). "
        + AUDIT_NOTE))

    return {
        "uid": "fleet-bottleneck",
        "title": "Fleet Bottleneck & Visibility",
        "description": "Ranked bottlenecks (Top 10) + capacity, account, and token-spend "
                       "visibility for the autonomous Claude Code worker fleet. "
                       "Source: fleet_bottleneck.py /metrics (namespace fleet_). On a "
                       "fresh up.sh stack the token-spend tiles read No data until a "
                       "session audit returns rows (run python tools/session_audit.py "
                       "discover --all); see the First run note at the top.",
        "tags": ["fleet", "bottleneck", "observability"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {}, "links": [], "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


def build_gateway():
    panels = []

    req_rate = "sum(rate(fak_gateway_http_requests_total[5m]))"
    # `or vector(0)` so a healthy gateway (no 5xx series at all) renders 0, not "No data".
    err_ratio = (
        '(sum(rate(fak_gateway_http_requests_total{status=~"5.."}[5m])) or vector(0)) / '
        "clamp_min(sum(rate(fak_gateway_http_requests_total[5m])), 0.001)"
    )
    http_p95 = (
        "histogram_quantile(0.95, "
        "sum by (le) (rate(fak_gateway_http_request_duration_seconds_bucket[5m])))"
    )
    op_p95 = (
        "histogram_quantile(0.95, "
        "sum by (le) (rate(fak_gateway_operation_duration_seconds_bucket[5m])))"
    )

    panels.append(row("Gateway health", 0))
    panels.append(stat("Scrape", 'up{job="fak_gateway"}', 0, 1,
                       thresholds=UP_TH,
                       mappings=[{"type": "value", "options": {
                           "0": {"text": "DOWN", "color": "red", "index": 0},
                           "1": {"text": "UP", "color": "green", "index": 1}}}],
                       desc="Prometheus scrape health for fak serve /metrics."))
    panels.append(stat("Gateway up", "fak_gateway_up", 4, 1,
                       thresholds=UP_TH, desc="Self-reported gateway liveness."))
    panels.append(stat("Requests / sec", req_rate, 8, 1,
                       desc="Total HTTP request throughput over 5m."))
    panels.append(stat("5xx ratio", err_ratio, 12, 1, unit="percentunit",
                       thresholds=ERR_TH, desc="5xx responses divided by all gateway requests."))
    panels.append(stat("HTTP p95", http_p95, 16, 1, unit="s",
                       thresholds=LAT_TH, desc="Gateway HTTP p95 latency over 5m."))
    panels.append(stat("Inflight", "fak_gateway_inflight_requests", 20, 1,
                       desc="Requests currently executing in the gateway."))

    panels.append(row("HTTP surface", 5))
    panels.append(timeseries(
        "HTTP request rate by route/status",
        [('sum by (route, status) (rate(fak_gateway_http_requests_total[5m]))', "{{route}} {{status}}")],
        0, 6, w=12, h=8, desc="Which route/status classes are driving traffic."))
    panels.append(timeseries(
        "HTTP p95 by route",
        [("histogram_quantile(0.95, sum by (le, route) "
          "(rate(fak_gateway_http_request_duration_seconds_bucket[5m])))", "{{route}}")],
        12, 6, w=12, h=8, unit="s", desc="Route-level p95 latency over 5m."))
    panels.append(bargauge(
        "Current route rate", 'sum by (route) (rate(fak_gateway_http_requests_total[5m]))',
        "{{route}}", 0, 14, w=12, h=8,
        desc="Instant route throughput ranked by current rate."))
    panels.append(bargauge(
        "Current status rate", 'sum by (status) (rate(fak_gateway_http_requests_total[5m]))',
        "{{status}}", 12, 14, w=12, h=8,
        desc="Instant status-class throughput ranked by current rate."))

    panels.append(row("Kernel operations", 22))
    panels.append(timeseries(
        "Operation verdict rate",
        [('sum by (operation, verdict) (rate(fak_gateway_operations_total[5m]))',
          "{{operation}} {{verdict}}")],
        0, 23, w=12, h=8, desc="syscall/adjudicate/admit verdict mix."))
    panels.append(timeseries(
        "Operation p95 by operation",
        [("histogram_quantile(0.95, sum by (le, operation) "
          "(rate(fak_gateway_operation_duration_seconds_bucket[5m])))", "{{operation}}")],
        12, 23, w=12, h=8, unit="s", desc="Kernel operation p95 over 5m."))
    panels.append(stat("Operation p95", op_p95, 0, 31, unit="s",
                       thresholds=LAT_TH, desc="Aggregate syscall/adjudicate/admit p95 latency."))
    panels.append(stat("vDSO hit ratio", "fak_gateway_vdso_hit_ratio", 4, 31,
                       unit="percentunit", thresholds=CACHE_TH,
                       desc="Cumulative fast-path hit ratio over kernel submissions."))
    panels.append(stat("Denied / sec", '(sum(rate(fak_gateway_operations_total{verdict="DENY"}[5m])) or vector(0))',
                       8, 31, thresholds=ONE_RED, desc="Denied gateway operations per second (0 when none denied)."))
    panels.append(stat("Quarantines / sec", "rate(fak_kernel_quarantines_total[5m])",
                       12, 31, thresholds=ONE_RED,
                       desc="Result admissions quarantined by the result-side stack."))
    panels.append(stat("Engine calls / sec", "rate(fak_kernel_engine_calls_total[5m])",
                       16, 31, desc="Kernel calls that reached the engine."))
    panels.append(stat("Submits / sec", "rate(fak_kernel_submits_total[5m])",
                       20, 31, desc="Kernel submissions per second."))

    # Trust floor — the DOS-discipline view. fak_gateway_operations_total carries a
    # `reason` label (the closed refusal vocabulary from internal/abi/reasons.go:
    # TRUST_VIOLATION, UNWITNESSED, SECRET_EXFIL, SELF_MODIFY, POLICY_BLOCK,
    # DEFAULT_DENY, RATE_LIMITED, MISROUTE, UNKNOWN_TOOL, ...) and a `disposition`
    # label (DENY/WAIT/...). The verdict-mix panels above answer "how many were
    # blocked"; these answer "WHY, by the structured reason the kernel refused with"
    # — the evidence that the trust floor is actually firing, not just that traffic
    # flowed. A non-ALLOW verdict is a refusal; grouping its rate by reason is the
    # operator's break-glass "what is the floor catching right now" view. Empty
    # reason ("") = the ALLOW path (no refusal reason), excluded so the panels show
    # only the refusal vocabulary.
    nonallow = 'fak_gateway_operations_total{reason!="",verdict!="ALLOW"}'
    panels.append(row("Trust floor (DOS refusals by reason)", 35))
    panels.append(timeseries(
        "Refusal rate by reason",
        [(f'sum by (reason) (rate({nonallow}[5m]))', "{{reason}}")],
        0, 36, w=12, h=8,
        desc="Per-second rate of non-ALLOW verdicts grouped by the closed refusal "
             "vocabulary (internal/abi/reasons.go). This is the trust floor firing — "
             "0 across the board means nothing is being refused."))
    panels.append(bargauge(
        "Top refusal reasons (now)",
        f'sum by (reason) (rate({nonallow}[5m]))', "{{reason}}",
        12, 36, w=12, h=8, thresholds=ONE_RED,
        desc="Instant ranking of which refusal reasons are firing — the kernel's "
             "structured 'why I said no', ranked by current rate."))
    panels.append(timeseries(
        "Refusal disposition mix",
        [('sum by (disposition) '
          '(rate(fak_gateway_operations_total{disposition!="",verdict!="ALLOW"}[5m]))',
          "{{disposition}}")],
        0, 44, w=24, h=6,
        desc="How refusals are dispatched (DENY vs WAIT vs ...) — the action the "
             "caller should take, aggregated over all reasons."))

    panels.append(row("Kernel counters", 51))
    panels.append(timeseries(
        "Kernel call-path rates",
        [
            ("rate(fak_kernel_submits_total[5m])", "submits"),
            ("rate(fak_kernel_vdso_hits_total[5m])", "vdso hits"),
            ("rate(fak_kernel_engine_calls_total[5m])", "engine calls"),
            ("rate(fak_kernel_denies_total[5m])", "denies"),
            ("rate(fak_kernel_transforms_total[5m])", "transforms"),
            ("rate(fak_kernel_quarantines_total[5m])", "quarantines"),
            ("rate(fak_kernel_admitted_total[5m])", "admitted"),
        ],
        0, 36, w=24, h=8, desc="The kernel counters exported from fak serve /metrics."))

    return {
        "uid": "fak-gateway-observability",
        "title": "FAK Gateway Observability",
        "description": "Live HTTP, verdict, kernel-counter, and vDSO visibility for fak serve. "
                       "Source: fak serve /metrics (namespaces fak_gateway_ and fak_kernel_).",
        "tags": ["fak", "gateway", "observability"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {}, "links": [], "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


def build_dogfood():
    panels = []

    msg_filter = '{job="fak_gateway",route="/v1/messages"}'
    msg_status_filter = '{job="fak_gateway",route="/v1/messages",status=~"2..|3..|4..|5.."}'
    msg_count = f"fak_gateway_http_request_duration_seconds_count{msg_filter}"
    msg_sum = f"fak_gateway_http_request_duration_seconds_sum{msg_filter}"
    msg_bucket = f"fak_gateway_http_request_duration_seconds_bucket{msg_filter}"
    msg_requests = f"fak_gateway_http_requests_total{msg_filter}"
    msg_requests_by_status = f"fak_gateway_http_requests_total{msg_status_filter}"
    msg_inflight = (
        'fak_gateway_inflight_requests_by_route'
        '{job="fak_gateway",route="/v1/messages"} or vector(0)'
    )
    oldest_inflight = 'fak_gateway_inflight_max_age_seconds{job="fak_gateway"}'

    msg_rate = f"sum(rate({msg_requests}[$__rate_interval]))"
    msg_avg = (
        f"sum(rate({msg_sum}[$__rate_interval])) / "
        f"clamp_min(sum(rate({msg_count}[$__rate_interval])), 0.001)"
    )

    def msg_quantile(q):
        return (
            f"histogram_quantile({q}, "
            f"sum by (le) (rate({msg_bucket}[$__rate_interval])))"
        )

    def slow_over(le):
        bucket = (
            'fak_gateway_http_request_duration_seconds_bucket'
            f'{{job="fak_gateway",route="/v1/messages",le="{le}"}}'
        )
        return (
            f"(sum(rate({msg_count}[$__rate_interval])) or vector(0)) - "
            f"(sum(rate({bucket}[$__rate_interval])) or vector(0))"
        )

    panels.append(text_panel(
        "First run: this dashboard needs Claude Code traffic",
        "Every latency, status, slow-request, and in-flight panel here filters "
        "`route=\"/v1/messages\"` - **Claude Code traffic**. The `tools/grafana/up.sh` "
        "next-step hint drives `/v1/fak/syscall`, which does NOT appear on this route, "
        "so a fresh stack shows this dashboard almost entirely **No data**. " + DOGFOOD_NOTE
        + "\n\n"
        "The Scrape and Gateway up tiles populate immediately (they are route-agnostic). "
        "The Kernel activity panels at the bottom drop the route filter, so they light "
        "up on any gateway traffic - including the syscall smoke request.",
        0, 0, w=24, h=5,
        desc="Why this dashboard reads No data until a Claude Code session drives "
             "/v1/messages, and how to populate it."))

    panels.append(row("Dogfood health", 5))
    panels.append(stat("Scrape", 'up{job="fak_gateway"}', 0, 6,
                       thresholds=UP_TH,
                       mappings=[{"type": "value", "options": {
                           "0": {"text": "DOWN", "color": "red", "index": 0},
                           "1": {"text": "UP", "color": "green", "index": 1}}}],
                       desc="Prometheus scrape health for the fak gateway."))
    panels.append(stat("Gateway up", "fak_gateway_up", 4, 6,
                       thresholds=UP_TH, desc="Self-reported gateway liveness."))
    panels.append(stat("Messages in flight", msg_inflight, 8, 6,
                       thresholds=steps((None, "green"), (1, "yellow"), (3, "red")),
                       desc="/v1/messages requests currently executing in the gateway. "
                            + DOGFOOD_NOTE))
    panels.append(stat("Oldest in-flight", oldest_inflight, 12, 6, unit="s",
                       thresholds=steps((None, "green"), (60, "yellow"), (300, "red")),
                       desc="Age of the oldest currently-running gateway request."))
    panels.append(stat("/v1/messages / sec", msg_rate, 16, 6,
                       desc="Claude Code dogfood request throughput. " + DOGFOOD_NOTE))
    panels.append(stat("/v1/messages p95", msg_quantile("0.95"), 20, 6, unit="s",
                       thresholds=steps((None, "green"), (60, "yellow"), (180, "red")),
                       desc="p95 completed /v1/messages latency. Qwen3.6 local turns can "
                            "be minutes. " + DOGFOOD_NOTE))

    panels.append(row("Claude /v1/messages latency", 10))
    panels.append(timeseries(
        "Latency quantiles",
        [(msg_quantile("0.50"), "p50"),
         (msg_avg, "avg"),
         (msg_quantile("0.95"), "p95"),
         (msg_quantile("0.99"), "p99")],
        0, 11, w=12, h=8, unit="s",
        desc="Completed /v1/messages latency. Buckets extend to 1800s for local 27B "
             "prefill. " + DOGFOOD_NOTE))
    panels.append(timeseries(
        "Request rate by status",
        [(f"sum by (status) (rate({msg_requests_by_status}[$__rate_interval]))", "{{status}}")],
        12, 11, w=12, h=8,
        desc="Status mix for the Claude Code dogfood endpoint. " + DOGFOOD_NOTE))

    panels.append(row("Slow request threshold rate", 19))
    panels.append(timeseries(
        "Requests slower than threshold",
        [(slow_over("30"), ">30s"),
         (slow_over("60"), ">60s"),
         (slow_over("120"), ">120s"),
         (slow_over("300"), ">300s")],
        0, 20, w=24, h=8,
        desc="Rate of completed /v1/messages requests slower than each bucket "
             "threshold. " + DOGFOOD_NOTE))

    panels.append(row("Kernel activity during dogfood turns", 28))
    panels.append(timeseries(
        "Kernel operation verdict rate",
        [('sum by (operation, verdict) '
          '(rate(fak_gateway_operations_total{job="fak_gateway"}[$__rate_interval]))',
          "{{operation}} {{verdict}}")],
        0, 29, w=12, h=8,
        desc="Adjudication activity while Claude Code is using the gateway. This panel "
             "drops the /v1/messages filter, so it also reflects /v1/fak/syscall traffic."))
    panels.append(timeseries(
        "Kernel counter rates",
        [
            ('rate(fak_kernel_submits_total{job="fak_gateway"}[$__rate_interval])', "submits"),
            ('rate(fak_kernel_vdso_hits_total{job="fak_gateway"}[$__rate_interval])', "vdso hits"),
            ('rate(fak_kernel_engine_calls_total{job="fak_gateway"}[$__rate_interval])', "engine calls"),
            ('rate(fak_kernel_denies_total{job="fak_gateway"}[$__rate_interval])', "denies"),
            ('rate(fak_kernel_quarantines_total{job="fak_gateway"}[$__rate_interval])', "quarantines"),
        ],
        12, 29, w=12, h=8,
        desc="Kernel call-path counters during dogfood sessions. Route-agnostic, so "
             "any gateway traffic lights these up."))

    panels.append(text_panel(
        "Request/debug log pointers",
        "Grafana shows gateway metrics, not raw request bodies. For request-level debugging, "
        "the dogfood launcher defaults Claude Code to `--debug api`; probe stderr/debug goes "
        "to `/tmp/fak-claude.log`, and the gateway process log goes to `/tmp/fak-serve.log`. "
        "Set `FAK_DOGFOOD_CLAUDE_DEBUG_FILE=/path/to/file` if you want Claude's debug stream "
        "written to a dedicated file while the metrics in this dashboard show whether the "
        "request is still in flight, completed slowly, or failed by status.",
        0, 37, w=24, h=5,
        desc="Operational pointers for the logs that contain raw Claude/gateway request detail."))

    return {
        "uid": "fak-dogfood-slow-requests",
        "title": "FAK Dogfood Slow Requests",
        "description": "Focused /v1/messages latency, status, inflight, and kernel activity "
                       "for Claude Code dogfood sessions against slow local models. The "
                       "timeseries panels filter route=\"/v1/messages\", so they read No "
                       "data until a Claude Code session drives the gateway; the "
                       "/v1/fak/syscall smoke request does not appear here. See the First "
                       "run note at the top.",
        "tags": ["fak", "dogfood", "claude", "slow-requests"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {}, "links": [], "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


READY_TH = steps((None, "green"), (5, "yellow"), (30, "red"))
# Unaccounted boot time: green when every bit of startup is attributed to a named
# phase, yellow once an untimed gap appears, red when a meaningful chunk of boot
# is uninstrumented. This is the panel that says "startup is fully instrumented".
UNACCOUNTED_TH = steps((None, "green"), (0.05, "yellow"), (0.5, "red"))


def build_startup():
    """The one-time boot sequence of a fak gateway: time-to-ready, per-phase boot
    cost, the GGUF weight-load breakdown, and a coverage gauge that flags any boot
    wall-clock the named phases do NOT explain. The startup/model-load families are
    gauges held at their once-at-boot values for the life of the process, so a panel
    can show what THIS process's boot cost at any later moment — a counter/rate would
    miss it entirely on a fast restart."""
    panels = []

    panels.append(text_panel(
        "First run: the Model weight load panels need --gguf",
        "The default `tools/grafana/up.sh` path starts the gateway with `--engine "
        "inkernel --model smollm2-135m` against a fak-format export, with **no "
        "`--gguf`** - so every `fak_model_load_*` panel (Model load time, Weights "
        "loaded, Tensors, the GGUF phase bargauges, and the Loaded model table) stays "
        "**No data**. " + MODEL_LOAD_NOTE + "\n\n"
        "The Boot summary tiles and the Startup phase breakdown DO populate on the "
        "up.sh path - they time every gateway boot regardless of weights. Only the "
        "GGUF weight-load section is `--gguf`-gated.",
        0, 0, w=24, h=5,
        desc="Why the model-load panels read No data on a fresh up.sh stack (no --gguf), "
             "and what to run to populate them."))

    # Boot summary — two rows of four stat tiles (24-wide grid, w=6 each).
    panels.append(row("Boot summary", 5))
    panels.append(stat("Time to ready", "fak_gateway_time_to_ready_seconds", 0, 6,
                       unit="s", thresholds=READY_TH,
                       desc="Total wall-clock from process start to the gateway becoming "
                            "ready to serve (socket bound). 0 until the boot completes."))
    panels.append(stat("Boot unaccounted", "fak_gateway_startup_unaccounted_seconds", 6, 6,
                       unit="s", thresholds=UNACCOUNTED_TH,
                       desc="Boot wall-clock the named phases do NOT explain "
                            "(time_to_ready minus the sum of phase durations). Near-zero "
                            "means startup is FULLY instrumented; a non-zero value means a "
                            "phase is missing or work ran between New and MarkReady that "
                            "no phase records."))
    panels.append(stat("Gateway up", "fak_gateway_up", 12, 6, thresholds=UP_TH,
                       mappings=[{"type": "value", "options": {
                         "0": {"text": "DOWN", "color": "red", "index": 0},
                         "1": {"text": "UP", "color": "green", "index": 1}}}],
                       desc="Self-reported gateway liveness."))
    panels.append(stat("Uptime", "time() - fak_gateway_start_time_seconds", 18, 6,
                       unit="s", desc="Time since this gateway process started."))
    panels.append(stat("Model load time", "fak_model_load_duration_seconds", 0, 10,
                       unit="s", thresholds=READY_TH,
                       desc="Total wall-clock to load model weights at boot. No data "
                            "when serving with no --gguf weights. " + MODEL_LOAD_NOTE))
    panels.append(stat("Weights loaded", "fak_model_load_bytes", 6, 10, unit="bytes",
                       color="thresholds", thresholds=steps((None, "blue")),
                       desc="Total bytes read/materialized while loading weights at boot. "
                            + MODEL_LOAD_NOTE))
    panels.append(stat("Tensors", "fak_model_load_tensors", 12, 10,
                       thresholds=steps((None, "blue")),
                       desc="Tensors materialized while loading weights at boot. "
                            + MODEL_LOAD_NOTE))
    panels.append(stat("Ready at", "fak_gateway_ready_time_seconds * 1000", 18, 10,
                       unit="none", thresholds=steps((None, "blue")),
                       desc="Unix instant (ms) the gateway became ready to serve."))

    panels.append(row("Startup phase breakdown", 14))
    panels.append(bargauge(
        "Boot phase cost (now)", "sort_desc(fak_gateway_startup_phase_duration_seconds)",
        "{{phase}}", 0, 15, w=12, h=8, unit="s",
        desc="Wall-clock of each boot phase, bottleneck first: flag-parse, policy-load, "
             "planner-init, vdso-config, kernel-init, listener-bind, and model-load "
             "(with --gguf). The listener-bind phase is measured on the synchronous "
             "net.Listen, so it is accurate to the moment the socket is ready."))
    panels.append(timeseries(
        "Time to ready over restarts",
        [("fak_gateway_time_to_ready_seconds", "time to ready"),
         ("fak_gateway_startup_unaccounted_seconds", "unaccounted")],
        12, 15, w=12, h=8, unit="s",
        desc="Both gauges hold their once-at-boot values, so each fresh process shows "
             "as a new level — restarts and their boot cost are visible across the "
             "window. unaccounted tracking time-to-ready means a phase is missing."))

    panels.append(row("Model weight load (GGUF phases)", 23))
    panels.append(bargauge(
        "Load phase time (bottleneck first)",
        "sort_desc(fak_model_load_phase_duration_seconds)", "{{phase}}",
        0, 24, w=12, h=8, unit="s",
        desc="Wall-clock of each GGUF weight-load phase. The top bar is the bottleneck. "
             "Empty until a gateway is started with --gguf. " + MODEL_LOAD_NOTE))
    panels.append(bargauge(
        "Load phase bytes", "sort_desc(fak_model_load_phase_bytes)", "{{phase}}",
        12, 24, w=12, h=8, unit="bytes",
        desc="Bytes processed by each GGUF weight-load phase. " + MODEL_LOAD_NOTE))
    panels.append(info_table(
        "Loaded model", "fak_model_load_info", 0, 32, w=24, h=6,
        desc="Static labels for the model loaded at boot: source path, loader mode, "
             "and the slowest (bottleneck) load phase. Empty with no --gguf weights. "
             + MODEL_LOAD_NOTE))

    return {
        "uid": "fak-startup-load",
        "title": "FAK Startup & Model Load",
        "description": "The one-time boot sequence of a fak gateway: time-to-ready, "
                       "per-phase boot cost, and the GGUF weight-load breakdown. Source: "
                       "fak serve /metrics (fak_gateway_startup_* and fak_model_load_* "
                       "gauges). Populate the model-load panels with: fak serve --gguf MODEL.gguf.",
        "tags": ["fak", "gateway", "startup", "model-load"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {}, "links": [], "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


PROVEN_MAP = [{"type": "value", "options": {
    "0": {"text": "NOT YET", "color": "yellow", "index": 0},
    "1": {"text": "PROVEN", "color": "green", "index": 1}}}]
# Net cache saving: green is the win (saving repaid), red the cold-write hole a
# session sits in until its reads repay the write premium. The 0 boundary is the
# break-even line the fak_vcache_proven gate flips on.
SAVED_TH = steps((None, "red"), (0, "green"))


def build_guard():
    """The `fak guard` session view — the live Grafana twin of the adjudication +
    cache summary `fak guard` prints when the wrapped agent exits. Guard binds the
    SAME gateway fak serve does (on a private loopback port) and serves the SAME
    /metrics, so this dashboard is built entirely from already-emitted families. It
    foregrounds what the referee did: verdict mix (allowed/denied/repaired/
    quarantined), the trust floor firing by the closed refusal vocabulary, the NET
    provider-cache economics the kernel hop preserved, and the history-compaction /
    tool-floor token shedding. Scrape it by starting guard on the gateway port the
    Prometheus `fak_gateway` job already targets: `fak guard --addr 127.0.0.1:8080
    -- claude`."""
    panels = []

    # Every non-ALLOW verdict carrying a refusal reason is the trust floor firing;
    # the empty-reason ALLOW path is excluded so the refusal panels show only the
    # closed refusal vocabulary (internal/abi/reasons.go).
    nonallow = 'fak_gateway_operations_total{reason!="",verdict!="ALLOW"}'
    UP_MAP = [{"type": "value", "options": {
        "0": {"text": "DOWN", "color": "red", "index": 0},
        "1": {"text": "UP", "color": "green", "index": 1}}}]

    panels.append(row("Session at a glance", 0))
    panels.append(stat("Scrape", 'up{job="fak_gateway"}', 0, 1, thresholds=UP_TH,
                       mappings=UP_MAP,
                       desc="Prometheus scrape health. Start guard on this port with "
                            "`fak guard --addr 127.0.0.1:8080 -- claude`."))
    panels.append(stat("Kernel decisions", "sum(fak_gateway_operations_total) or vector(0)",
                       4, 1, desc="Every verdict the kernel reached this session "
                                  "(allow + deny + repair + admit)."))
    panels.append(stat("Allowed", 'sum(fak_gateway_operations_total{verdict="ALLOW"}) or vector(0)',
                       8, 1, desc="Tool calls the capability floor let through."))
    panels.append(stat("Denied", 'sum(fak_gateway_operations_total{verdict="DENY"}) or vector(0)',
                       12, 1, thresholds=ONE_RED,
                       desc="Proposed tool calls the floor refused before they ran."))
    panels.append(stat("Repaired", 'sum(fak_gateway_operations_total{verdict="TRANSFORM"}) or vector(0)',
                       16, 1, desc="Calls repaired in-flight (e.g. grammar) instead of "
                                   "costing a wasted round-trip."))
    panels.append(stat("Quarantined", "fak_gateway_context_pollutions_blocked_total or vector(0)",
                       20, 1, thresholds=ONE_RED,
                       desc="Poisoned/untrusted tool-result payloads the context-MMU "
                            "paged out before the model read them."))

    panels.append(row("Trust floor — refusals by reason", 5))
    panels.append(timeseries(
        "Refusal rate by reason",
        [(f"sum by (reason) (rate({nonallow}[5m]))", "{{reason}}")],
        0, 6, w=12, h=8,
        desc="Per-second rate of non-ALLOW verdicts grouped by the closed refusal "
             "vocabulary (internal/abi/reasons.go: TRUST_VIOLATION, SECRET_EXFIL, "
             "SELF_MODIFY, POLICY_BLOCK, DEFAULT_DENY, ...). 0 across the board means "
             "the floor refused nothing."))
    panels.append(bargauge(
        "Refusals by reason (session total)", f"sum by (reason) ({nonallow})",
        "{{reason}}", 12, 6, w=12, h=8, thresholds=ONE_RED,
        desc="Cumulative count of each structured refusal reason this session — the "
             "kernel's 'why I said no', ranked."))

    panels.append(row("Verdict & disposition mix", 14))
    panels.append(timeseries(
        "Verdict mix (rate)",
        [("sum by (operation, verdict) (rate(fak_gateway_operations_total[5m]))",
          "{{operation}} {{verdict}}")],
        0, 15, w=12, h=8,
        desc="syscall/adjudicate/admit verdict rate while the wrapped agent runs."))
    panels.append(timeseries(
        "Refusal disposition (rate)",
        [('sum by (disposition) '
          '(rate(fak_gateway_operations_total{disposition!="",verdict!="ALLOW"}[5m]))',
          "{{disposition}}")],
        12, 15, w=12, h=8,
        desc="How refusals are dispatched (DENY vs WAIT vs ...) — the action the "
             "caller should take, aggregated over all reasons."))

    panels.append(row("Cache value — what the kernel hop saved", 23))
    panels.append(stat("Net saved (tok-equiv)", "fak_vcache_saved_token_equiv", 0, 24,
                       thresholds=SAVED_TH,
                       desc="NET realized provider-cache saving (read rebate MINUS write "
                            "premium). Negative until the reads repay a cold write. Same "
                            "number as `fak vcache observe`."))
    panels.append(stat("Cache multiplier", "fak_vcache_multiplier", 4, 24,
                       desc="baseline/actual token-equiv; 1.0 = no net saving."))
    panels.append(stat("Provider hit-rate", "fak_vcache_hit_rate", 8, 24,
                       unit="percentunit", thresholds=CACHE_TH,
                       desc="cache_read share of the uncached baseline."))
    panels.append(stat("Repaid?", "fak_vcache_proven", 12, 24, thresholds=UP_TH,
                       mappings=PROVEN_MAP,
                       desc="1 once the session's observed cache reads repaid the write "
                            "premium (NET positive); else 0 (cold/write-dominated)."))
    panels.append(stat("Provider cached tokens",
                       "fak_gateway_inference_cached_prompt_tokens_total or vector(0)",
                       16, 24,
                       desc="Prompt tokens the provider served from its own cache, "
                            "preserved byte-for-byte through the kernel hop."))
    panels.append(stat("Cache writes (tok)",
                       "fak_vcache_cache_creation_tokens_total or vector(0)", 20, 24,
                       desc="cache_creation tokens — the write premium the reads repay."))
    panels.append(timeseries(
        "Cache economics over time",
        [("fak_vcache_baseline_token_equiv", "baseline (uncached)"),
         ("fak_vcache_actual_token_equiv", "actual (cached)"),
         ("fak_vcache_saved_token_equiv", "net saved")],
        0, 28, w=24, h=8,
        desc="The uncached baseline vs actual token-equiv cost and the net saving "
             "between them. Series appear once a turn carries provider cache activity."))

    panels.append(row("Token shedding — compaction & tool-floor prune", 36))
    panels.append(bargauge(
        "Compaction attempts (session)",
        "sum by (outcome) (fak_gateway_compaction_attempts_total)", "{{outcome}}",
        0, 37, w=8, h=8,
        desc="fired = body rewritten with the protected prefix shipped byte-identical; "
             "bailed = returned identity; off = budget unset."))
    panels.append(bargauge(
        "Compaction bail reasons",
        "sum by (reason) (fak_gateway_compaction_bail_reason_total)", "{{reason}}",
        8, 37, w=8, h=8, thresholds=ONE_RED,
        desc="Why a compaction bailed. under_budget/no_breakpoint are benign; "
             "prefix_mismatch / splice_failed / redecode_failed are the only fak-fault "
             "cache signals and must stay 0."))
    panels.append(stat("Tokens shed", "fak_gateway_compaction_shed_tokens_total or vector(0)",
                       16, 37, w=8, h=4,
                       desc="Estimated tokens fak removed from the outbound body across "
                            "all fires (what fak SENT, not a provider-billing claim)."))
    panels.append(stat("Tool defs pruned", "fak_gateway_inbound_tools_pruned_total or vector(0)",
                       16, 41, w=8, h=4,
                       desc="Unreachable tool definitions dropped from tools[], "
                            "cache-prefix-preserving. 0 on Claude Code, whose single "
                            "breakpoint sits on the LAST tool."))

    panels.append(row("Upstream health & turns saved", 45))
    panels.append(timeseries(
        "Upstream turn failures by kind",
        [("sum by (kind) (rate(fak_gateway_upstream_errors_total[5m]))", "{{kind}}")],
        0, 46, w=12, h=8,
        desc="Why proxied turns failed (stalled / unreachable / status_4xx / "
             "status_5xx / oom / other) — the cumulative twin of the per-turn FAILED line."))
    panels.append(bargauge(
        "Turns saved by mechanism",
        "sum by (mechanism) (fak_gateway_turns_saved_total)", "{{mechanism}}",
        12, 46, w=8, h=8,
        desc="vdso_dedup = a duplicate read served with no engine round-trip; "
             "grammar_repair = a malformed call repaired instead of a retry turn."))
    panels.append(stat("Upstream retries", "fak_gateway_upstream_retries_total or vector(0)",
                       20, 46, w=4, h=8,
                       desc="Planner 429/5xx exponential-backoff attempts this session."))

    panels.append(text_panel(
        "Scrape a fak guard session",
        "`fak guard` binds the same gateway `fak serve` does, on a private loopback "
        "port that is torn down when the wrapped agent exits — so to chart a session, "
        "pin the port the Prometheus `fak_gateway` job already scrapes:\n\n"
        "```\nfak guard --addr 127.0.0.1:8080 -- claude\n```\n\n"
        "Every panel here reads `fak guard`'s live `/metrics`. The durable, "
        "hash-chained **decision journal** (`guard-audit.jsonl`, on by default) is the "
        "tamper-evident record of the same verdicts — replay it with "
        "`fak audit verify <path>`. Grafana shows the verdicts and savings; it never "
        "shows raw request bodies (use `--log FILE` for the structured per-request / "
        "per-verdict stream).",
        0, 54, w=24, h=6,
        desc="How to point Prometheus at a guard session, and where the durable record lives."))

    return {
        "uid": "fak-guard-adjudication",
        "title": "FAK Guard — Kernel Adjudication",
        "description": "The `fak guard` session view: verdict mix (allowed/denied/"
                       "repaired/quarantined), the trust floor firing by refusal reason, "
                       "the NET provider-cache economics the kernel hop preserved, and "
                       "history-compaction / tool-floor token shedding. The live twin of "
                       "the summary `fak guard` prints on exit. Source: fak guard /metrics "
                       "(start it with --addr 127.0.0.1:8080 to reuse the fak_gateway scrape job).",
        "tags": ["fak", "guard", "adjudication", "security"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {}, "links": [], "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for path, dash in ((OUT, build()), (OUT_GATEWAY, build_gateway()),
                       (OUT_DOGFOOD, build_dogfood()), (OUT_STARTUP, build_startup()),
                       (OUT_GUARD, build_guard())):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dash, f, indent=2)
            f.write("\n")
        n_panels = sum(1 for p in dash["panels"] if p["type"] != "row")
        print(f"wrote {path}  ({n_panels} data panels, {len(dash['panels'])} total incl. rows)")


if __name__ == "__main__":
    main()
