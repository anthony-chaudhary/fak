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
                                              # and dashboards/fak-cache-value-rollup.json
                                              # and dashboards/fak-cache-health.json
                                              # and dashboards/fak-fleet-overview.json
                                              # and dashboards/fak-fleet-session.json

The committed JSON is what Grafana provisions; this generator is the maintainable
source. `METRICS` / `GATEWAY_METRICS` below are the contract —
`fleet_bottleneck_test.py` asserts dashboard/alert references against emitted
families.

The two run-operations dashboards are a PAIR: the stable-UID home keeps current
descriptor inventory separate from durable completed-run registrations, and both
tables data-link into the drill-down for one named run/session. They read the `fak_fleet_*` families from
`fak fleet metrics` (cmd/fak/fleet_metrics.go, scrape job `fak_fleet`) — a different
source from every other dashboard here, because it is the only one carrying a
per-session dimension.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dashboards", "fleet-bottleneck-overview.json")
OUT_GATEWAY = os.path.join(HERE, "dashboards", "fak-gateway-observability.json")
OUT_DOGFOOD = os.path.join(HERE, "dashboards", "fak-dogfood-slow-requests.json")
OUT_STARTUP = os.path.join(HERE, "dashboards", "fak-startup-load.json")
OUT_GUARD = os.path.join(HERE, "dashboards", "fak-guard-adjudication.json")
OUT_ROLLUP = os.path.join(HERE, "dashboards", "fak-cache-value-rollup.json")
OUT_CACHEHEALTH = os.path.join(HERE, "dashboards", "fak-cache-health.json")
OUT_FLEET_OVERVIEW = os.path.join(HERE, "dashboards", "fak-fleet-overview.json")
OUT_FLEET_SESSION = os.path.join(HERE, "dashboards", "fak-fleet-session.json")

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

# The cache-health view (#3645, under epic #3569): every family already emitted by
# fak serve / fak guard /metrics (internal/gateway/metrics_render.go) — no new metric
# is introduced. In-kernel KV-prefix reuse (WITNESSED) and the provider prompt-cache
# read/write axes (OBSERVED, relayed) are distinct signals and never double-count.
CACHE_HEALTH_METRICS = [
    "fak_gateway_up",
    "fak_gateway_kv_prefix_turns_total", "fak_gateway_kv_prefix_prompt_tokens_total",
    "fak_gateway_kv_prefix_reused_tokens_total",
    "fak_gateway_kv_prefix_turns_by_regime_total", "fak_gateway_kv_prefix_reuse_ratio",
    "fak_gateway_cache_ttl_upgrade_total",
    "fak_gateway_cache_breakpoint_placement_total",
    "fak_gateway_inference_cached_prompt_tokens_total",
    "fak_vcache_cache_creation_tokens_total",
    "fak_gateway_tool_defer_cold_total", "fak_gateway_tool_defer_turns_total",
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


def info_table(title, expr, x, y, w=24, h=6, desc="", exclude=None, links=None,
               link_field=None, sort_by=None, filterable=False):
    """A plain label-table: render an instant series' labels as columns, dropping
    Prometheus bookkeeping + the (always-1) Value. Used for the *_info gauge.

    links/link_field turn one column into the DRILL-DOWN door: `links` is a list of
    {title, url} data links and `link_field` names the column they hang off (an
    override, so only that cell is clickable rather than every cell in the row). This
    is what makes a fleet table a navigation surface instead of a read-only list.
    """
    drop = {"Time": True, "Value": True, "__name__": True, "job": True, "instance": True}
    for name in (exclude or []):
        drop[name] = True
    overrides = []
    if links and link_field:
        overrides.append({
            "matcher": {"id": "byName", "options": link_field},
            "properties": [{"id": "links", "value": links}],
        })
    options = {"showHeader": True, "cellHeight": "sm",
               "footer": {"show": False, "reducer": ["sum"], "fields": ""}}
    if sort_by:
        options["sortBy"] = [{"displayName": sort_by, "desc": False}]
    # filterable defaults OFF so adding it for one dashboard does not churn every other
    # committed info-table JSON on the next regen.
    custom = {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False}
    if filterable:
        custom["filterable"] = True
    return {
        "datasource": DS, "title": title, "description": desc, "type": "table",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {
            "color": {"mode": "thresholds"},
            "custom": custom,
            "mappings": [], "thresholds": steps((None, "green"))}, "overrides": overrides},
        "options": options,
        "pluginVersion": "11.5.2",
        "targets": [{"datasource": DS, "expr": expr, "format": "table",
                     "instant": True, "refId": "A"}],
        "transformations": [{"id": "organize", "options": {
            "excludeByName": drop, "renameByName": {}}}],
    }


def run_history_table(title, info_expr, numeric_selector, x, y, w=24, h=8, desc="",
                      links=None, link_field=None, filterable=True,
                      restrict_numeric_to_info=False):
    """Join durable run identity with low-cardinality lifecycle values.

    Reason and witness stay only on fak_fleet_run_info; timestamp and duration
    families carry only run/session. This keeps the Prometheus series safe while
    presenting one useful historical-review row in Grafana.
    """
    def metric_expr(name):
        expr = f"{name}{numeric_selector}"
        if restrict_numeric_to_info:
            expr = f"{expr} and on(run, session) {info_expr}"
        if name.endswith("_timestamp_seconds"):
            # Prometheus timestamps are seconds; Grafana date-time fields consume ms.
            expr = f"(({expr}) > 0) * 1000"
        return expr

    targets = [{
        "datasource": DS, "expr": info_expr, "format": "table",
        "instant": True, "refId": "A",
    }]
    for ref_id, metric in zip("BCDE", (
        "fak_fleet_run_created_timestamp_seconds",
        "fak_fleet_run_started_timestamp_seconds",
        "fak_fleet_run_terminal_timestamp_seconds",
        "fak_fleet_run_duration_seconds",
    )):
        targets.append({
            "datasource": DS, "expr": metric_expr(metric), "format": "table",
            "instant": True, "refId": ref_id,
        })

    overrides = [
        {"matcher": {"id": "byName", "options": field},
         "properties": [{"id": "unit", "value": "dateTimeAsIso"}]}
        for field in ("created", "started", "terminal")
    ]
    overrides.append({
        "matcher": {"id": "byName", "options": "duration"},
        "properties": [{"id": "unit", "value": "s"}],
    })
    if links and link_field:
        overrides.append({
            "matcher": {"id": "byName", "options": link_field},
            "properties": [{"id": "links", "value": links}],
        })

    custom = {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False}
    if filterable:
        custom["filterable"] = True
    exclude = {
        "Time": True, "Time #A": True, "Time #B": True, "Time #C": True,
        "Time #D": True, "Time #E": True, "Value #A": True,
        "session #B": True, "session #C": True, "session #D": True,
        "session #E": True, "__name__": True, "job": True,
        "instance": True, "service": True,
    }
    return {
        "datasource": DS, "title": title, "description": desc, "type": "table",
        "id": nid(), "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {
            "color": {"mode": "thresholds"}, "custom": custom, "mappings": [],
            "thresholds": steps((None, "green"))}, "overrides": overrides},
        "options": {"showHeader": True, "cellHeight": "sm",
                    "sortBy": [{"displayName": "terminal", "desc": True}],
                    "footer": {"show": False, "reducer": ["sum"], "fields": ""}},
        "pluginVersion": "11.5.2",
        "targets": targets,
        "transformations": [
            {"id": "joinByField", "options": {"byField": "run", "mode": "outer"}},
            {"id": "organize", "options": {
                "excludeByName": exclude,
                "renameByName": {
                    "Value #B": "created", "Value #C": "started",
                    "Value #D": "terminal", "Value #E": "duration",
                },
            }},
        ],
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


# Cache-value roll-up maps. Kept local to the builder's provenance: green = the
# honest win state, and the fence between WITNESSED and OBSERVED/projected $ is carried
# in every panel description, never in a threshold color.
MEASURED_MAP = [{"type": "value", "options": {
    "0": {"text": "INSUFFICIENT", "color": "yellow", "index": 0},
    "1": {"text": "MEASURED", "color": "green", "index": 1}}}]
BROKE_MAP = [{"type": "value", "options": {
    "0": {"text": "NOT YET", "color": "yellow", "index": 0},
    "1": {"text": "BROKE EVEN", "color": "green", "index": 1}}}]
PROVISIONAL_MAP = [{"type": "value", "options": {
    "0": {"text": "SETTLED", "color": "green", "index": 0},
    "1": {"text": "PROVISIONAL", "color": "yellow", "index": 1}}}]
PRESENT_MAP = [{"type": "value", "options": {
    "0": {"text": "NO REPORT", "color": "yellow", "index": 0},
    "1": {"text": "PRESENT", "color": "green", "index": 1}}}]
SPEEDUP_TH = steps((None, "red"), (1, "green"))  # >=1 = at least as fast as baseline

# The honesty fence, verbatim from the report, so a Grafana reader can never mistake
# the projected $ for a fak-WITNESSED claim.
ROLLUP_NOTE = (
    "**One question:** is fak's cache work paying off, and which feature moved the "
    "needle? This dashboard is the live twin of `fak cachevalue feed` (the Slack card) "
    "plus the offline `fak ablate` arms — the SAME two-track fold, re-projected as "
    "metrics.\n\n"
    "**The fence is load-bearing.** *Track 1* (`fak_cachevalue_latest_reuse_ratio`, "
    "sessions) is **WITNESSED** realized KV-prefix reuse. *Track 2* (every `_usd` / "
    "`saved_token_equiv` family) is an **OBSERVED / projected $** cost model over "
    "labelled provider cache reads — never a fak-witnessed dollar, never blended into "
    "Track 1. The `owner` label keeps provider vs fak dollars split so a provider-only "
    "corpus can never read as a fak win.\n\n"
    "Source: `fak cachevalue metrics --serve` (scrape job `fak_cachevalue`). Panels read "
    "`No data` until the `docs/nightrun/*.jsonl` ledgers exist — run `fak cachevalue "
    "feed --once` (or the nightrun) to populate them."
)
ABLATION_NOTE = (
    "**Offline, deterministic, $0.** These arms come from `fak ablate` replaying the "
    "SAME frozen trace with one feature toggled per arm — so `speedup_ratio > 1` is that "
    "feature's WITNESSED win over the `baseline` arm (marked in the table below), not a "
    "live provider measurement. Arms are labelled by `(arm, workload)`; a workload is a "
    "trace's content hash, so arms from different traces never share a bar. Populate by "
    "dropping `fak ablate` report JSONs in `experiments/ablate/` (a sample report ships "
    "with the repo; non-ablate JSONs there are skipped with a note)."
)


def build_cache_ablation_rollup():
    """The cache-value & ablation roll-up — the executive answer to 'is the cache method
    paying off, and which feature earned it?' It is the Grafana twin of the `fak
    cachevalue feed` Slack card (the two-track WITNESSED-reuse + OBSERVED-$ P&L fold) with
    the offline `fak ablate` feature arms alongside, so the realized-value story and the
    per-feature attribution live on one surface. Built entirely from the
    `fak_cachevalue_*` and `fak_ablation_*` families `fak cachevalue metrics` exposes;
    scrape it with `fak cachevalue metrics --serve --addr 127.0.0.1:9097` (job
    `fak_cachevalue`). Every $ panel is labelled OBSERVED/projected and split by owner so
    the honesty fence the report enforces survives into the dashboard."""
    panels = []

    # ---- headline: is it paying off? ----
    panels.append(row("Is fak's cache method paying off?", 0))
    panels.append(text_panel("What this dashboard answers", ROLLUP_NOTE, 0, 1, w=24, h=5))
    panels.append(stat("Scrape", 'up{job="fak_cachevalue"}', 0, 6, thresholds=UP_TH,
                       mappings=[{"type": "value", "options": {
                           "0": {"text": "DOWN", "color": "red", "index": 0},
                           "1": {"text": "UP", "color": "green", "index": 1}}}],
                       desc="Prometheus scrape health for the `fak_cachevalue` job. Start "
                            "it with `fak cachevalue metrics --serve --addr 127.0.0.1:9097`."))
    panels.append(stat("Verdict", "fak_cachevalue_measured", 4, 6, mappings=MEASURED_MAP,
                       thresholds=UP_TH,
                       desc="MEASURED once the two-track fold has enough rows; INSUFFICIENT "
                            "otherwise. `report_present` (not shown) stays 1 whenever the "
                            "exporter is alive, so a dead scrape reads as DOWN, not zero."))
    panels.append(stat("Cumulative NET $", "fak_cachevalue_cumulative_net_usd", 8, 6,
                       unit="currencyUSD", thresholds=SAVED_TH,
                       desc="OBSERVED/projected: running net $ through the latest period "
                            "(rebate + shed − write premium − spend). The P&L headline; a "
                            "cost projection, never a fak-witnessed dollar."))
    panels.append(stat("Broke even?", "fak_cachevalue_broke_even", 12, 6, thresholds=UP_TH,
                       mappings=BROKE_MAP,
                       desc="1 once the cumulative net $ has crossed zero."))
    panels.append(stat("API cost reduction", "fak_cachevalue_api_cost_reduction_pct", 16, 6,
                       unit="percent", thresholds=steps((None, "yellow"), (25, "green")),
                       desc="OBSERVED/projected: percent cheaper than the uncached/"
                            "uncompacted counterfactual."))
    panels.append(stat("fak share of $", "fak_cachevalue_fak_share_pct", 20, 6, unit="percent",
                       desc="Percent of avoided $ attributable to fak-authored cache work; "
                            "the rest is provider prompt-cache. Absent until any $ is avoided."))

    # ---- Track 1: WITNESSED realized reuse ----
    panels.append(row("Track 1 — WITNESSED realized KV-prefix reuse", 10))
    panels.append(stat("Latest reuse ratio", "fak_cachevalue_latest_reuse_ratio", 0, 11,
                       w=6, unit="percentunit", thresholds=CACHE_TH,
                       desc="WITNESSED: most recent weekly bucket's realized KV-prefix "
                            "reuse. NOT a vs-naive multiple (that family is excluded by "
                            "the report's #1066 fence)."))
    panels.append(stat("Multi-turn sessions", "fak_cachevalue_multi_turn_sessions", 6, 11,
                       w=6, desc="WITNESSED: sessions with >1 turn — the only ones that "
                                 "can realize prefix reuse."))
    panels.append(stat("Total sessions", "fak_cachevalue_total_sessions", 12, 11, w=6,
                       desc="WITNESSED: all sessions folded into Track 1."))
    panels.append(stat("Dollar-blind rows", "fak_cachevalue_dollar_blind_rows", 18, 11, w=6,
                       thresholds=steps((None, "green"), (1, "yellow")),
                       desc="Rows with saved token-equiv but no trusted price — excluded "
                            "from every $ column (honest omission, not a zero)."))
    panels.append(timeseries(
        "Realized reuse ratio over time",
        [("fak_cachevalue_latest_reuse_ratio", "latest reuse ratio")],
        0, 15, w=24, h=7, unit="percentunit",
        desc="The WITNESSED headline reuse ratio as it moves across scrapes."))

    # ---- Track 2: OBSERVED / projected $ P&L ----
    panels.append(row("Track 2 — OBSERVED / projected $ (P&L, never blended into Track 1)", 22))
    panels.append(timeseries(
        "Net $ over time (OBSERVED/projected)",
        [("fak_cachevalue_cumulative_net_usd", "cumulative net"),
         ("fak_cachevalue_latest_net_usd", "latest period net")],
        0, 23, w=12, h=8, unit="currencyUSD",
        desc="Running and per-period net $. A cost projection over labelled provider "
             "cache reads — side by side with Track 1, never summed into it."))
    panels.append(bargauge(
        "Saved token-equiv by owner", "sum by (owner) (fak_cachevalue_saved_token_equiv)",
        "{{owner}}", 12, 23, w=12, h=8,
        desc="provider = OBSERVED prompt-cache; fak = WITNESSED kernel-authored; total = "
             "their sum. Split so a provider-only corpus cannot read as a fak win."))
    panels.append(bargauge(
        "API cost avoided by owner ($)",
        "sum by (owner) (fak_cachevalue_api_cost_avoided_usd)", "{{owner}}", 0, 31,
        w=12, h=8, unit="currencyUSD", thresholds=SAVED_TH,
        desc="OBSERVED/projected $ avoided, split by owner. provider = read rebate net of "
             "write premium; fak = compaction saving (WITNESSED shed, $ projected)."))
    panels.append(stat("Observed spend", "fak_cachevalue_observed_spend_usd", 12, 31, w=6,
                       unit="currencyUSD",
                       desc="OBSERVED: actual API spend over the folded rows."))
    panels.append(stat("Counterfactual", "fak_cachevalue_counterfactual_usd", 18, 31, w=6,
                       unit="currencyUSD",
                       desc="OBSERVED/projected: uncached/uncompacted spend the reduction "
                            "is measured against."))
    panels.append(stat("Usage rows", "fak_cachevalue_usage_rows", 12, 35, w=6,
                       desc="WITNESSED: gateway/guard usage rows in the fleet aggregate."))
    panels.append(stat("Kernel decisions", "fak_cachevalue_kernel_decisions", 18, 35, w=6,
                       desc="WITNESSED: cumulative kernel admission decisions."))

    # ---- run-rate + session extension ----
    panels.append(row("Run-rate & session extension", 39))
    panels.append(bargauge(
        "$ avoided per day by owner", "sum by (owner) (fak_cachevalue_usd_avoided_per_day)",
        "{{owner}}", 0, 40, w=12, h=8, unit="currencyUSD", thresholds=SAVED_TH,
        desc="OBSERVED/projected daily run-rate over the savings-row span. PROVISIONAL "
             "under the honest-window floor — check the flag to the right."))
    panels.append(stat("Savings span (days)", "fak_cachevalue_span_days", 12, 40, w=6,
                       desc="Days of savings rows the run-rate is computed over."))
    panels.append(stat("Run-rate", "fak_cachevalue_rate_provisional", 18, 40, w=6,
                       thresholds=steps((None, "green"), (1, "yellow")),
                       mappings=PROVISIONAL_MAP,
                       desc="PROVISIONAL when the span is under the honest floor — the "
                            "30/90-day extrapolation is not yet settled."))
    panels.append(stat("Compaction shed (tok)", "fak_cachevalue_compaction_shed_tokens",
                       12, 44, w=6,
                       desc="WITNESSED: context tokens fak compaction shed — the only "
                            "source of session extension."))
    panels.append(stat("Context extension (tok)", "fak_cachevalue_context_extension_tokens",
                       18, 44, w=6,
                       desc="WITNESSED: tokens the session window was extended by. Provider "
                            "cache reads never enlarge the window, so only shed counts."))

    # ---- ablation ----
    panels.append(row("Feature ablation — which feature moved the needle", 48))
    panels.append(text_panel("How to read the ablation arms", ABLATION_NOTE, 0, 49, w=24, h=4))
    panels.append(stat("Ablate report", "fak_ablation_report_present", 0, 53, w=4,
                       thresholds=UP_TH, mappings=PRESENT_MAP,
                       desc="1 when at least one `fak ablate` report was folded."))
    panels.append(stat("Arms", "fak_ablation_arms", 4, 53, w=4,
                       desc="Total ablation arms across every folded report."))
    panels.append(bargauge(
        "Speedup vs baseline by arm", "fak_ablation_arm_speedup_ratio",
        "{{arm}} ({{workload}})", 8, 53, w=16, h=8, thresholds=SPEEDUP_TH,
        desc="WITNESSED replay: baseline mean / this arm's mean. >1 is faster than the "
             "baseline arm (which sits at exactly 1)."))
    panels.append(timeseries(
        "Mean decide latency by arm (ns)",
        [("fak_ablation_arm_mean_nanoseconds", "{{arm}} ({{workload}})")],
        0, 61, w=12, h=8, unit="ns",
        desc="WITNESSED replay: per-arm mean kernel decide latency. Lower is the win the "
             "speedup ratio expresses."))
    panels.append(bargauge(
        "vDSO fast-path hits by arm", "fak_ablation_arm_vdso_hits", "{{arm}} ({{workload}})",
        12, 61, w=12, h=8,
        desc="WITNESSED replay: vDSO (fast-path) hits per arm — the mechanism the speedup "
             "comes from when the vdso feature is on."))
    panels.append(info_table(
        "Baseline arm (speedup reference)", "fak_ablation_baseline_info",
        0, 69, w=24, h=6, exclude=["fleet"],
        desc="The arm each speedup ratio is measured against, per workload."))

    return {
        "uid": "fak-cache-value-rollup",
        "title": "FAK Cache Value — Roll-up & Ablation",
        "description": "Is fak's cache method paying off, and which feature earned it? The "
                       "Grafana twin of the `fak cachevalue feed` Slack card: Track 1 "
                       "WITNESSED KV-prefix reuse and the Track 2 OBSERVED/projected-$ P&L "
                       "(split by owner, never blended), plus the offline `fak ablate` "
                       "feature arms. Source: fak cachevalue metrics /metrics (scrape job "
                       "fak_cachevalue; start with `fak cachevalue metrics --serve`).",
        "tags": ["fak", "cache-value", "ablation", "rollup", "cost"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-30d", "to": "now"},
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


# Cache-health first-run note (#3645). Same posture as the other notes: markdown
# carries no PromQL, so the explanation renders even while every counter is 0.
CACHE_HEALTH_NOTE = (
    "**One screen: is the cache healthy right now?** Every panel reads the live "
    "`fak serve` / `fak guard` `/metrics` (scrape job `fak_gateway`) — the families "
    "already emitted by `internal/gateway/metrics_render.go`, no new metric.\n\n"
    "Two distinct signals, never double-counted: the **KV-prefix family** "
    "(`fak_gateway_kv_prefix_*`) is the WITNESSED in-kernel RadixAttention prefix "
    "match — it stays 0 on a pure proxy workload (no in-kernel model). The "
    "**provider axes** (`fak_gateway_inference_cached_prompt_tokens_total` read, "
    "`fak_vcache_cache_creation_tokens_total` write) are OBSERVED provider-relayed "
    "usage. The **authoring outcomes** (`_cache_ttl_upgrade_total`, "
    "`_cache_breakpoint_placement_total`) count what fak's managed-cache transform "
    "did; they stay 0 until `--managed-cache on` traffic flows. **Tool-defer** "
    "counters stay 0 unless `--defer-cold-tools` is enabled (off by default)."
)


def build_cache_health():
    """The cache-health board (#3645, epic #3569): the six cache signals on one
    screen — KV-prefix reuse ratio, regime buckets (the cache-cliff distribution),
    managed-cache TTL-upgrade and breakpoint-placement outcomes, the provider
    cache-read/creation token axes, and the cold-tool deferral counters. Built
    entirely from families fak serve / fak guard already emit (scrape job
    `fak_gateway`); its authoring twin for $-value questions is the cache-value
    roll-up (uid fak-cache-value-rollup), which folds ledgers, not live counters."""
    panels = []

    UP_MAP = [{"type": "value", "options": {
        "0": {"text": "DOWN", "color": "red", "index": 0},
        "1": {"text": "UP", "color": "green", "index": 1}}}]

    panels.append(row("Cache health at a glance", 0))
    panels.append(text_panel("How to read this board", CACHE_HEALTH_NOTE, 0, 1, w=24, h=5))
    panels.append(stat("Scrape", 'up{job="fak_gateway"}', 0, 6, thresholds=UP_TH,
                       mappings=UP_MAP,
                       desc="Prometheus scrape health for fak serve / fak guard /metrics."))
    panels.append(stat("KV-prefix reuse ratio", "fak_gateway_kv_prefix_reuse_ratio", 4, 6,
                       unit="percentunit", thresholds=CACHE_TH,
                       desc="WITNESSED realized in-kernel cache hit: reused / prompt "
                            "tokens across served turns. Climbs toward ~1 in the frozen "
                            "append-only regime; 0 on a pure proxy workload."))
    panels.append(stat("In-kernel turns",
                       "fak_gateway_kv_prefix_turns_total or vector(0)", 8, 6,
                       desc="Turns observed by the KV-prefix tap — the sample size "
                            "behind the reuse ratio."))
    panels.append(stat("Reused tokens",
                       "fak_gateway_kv_prefix_reused_tokens_total or vector(0)", 12, 6,
                       desc="Prompt tokens served from the cached KV prefix — prefill "
                            "work the kernel did NOT redo."))
    panels.append(stat("Provider cache-read (tok)",
                       "fak_gateway_inference_cached_prompt_tokens_total or vector(0)",
                       16, 6,
                       desc="OBSERVED: prompt tokens the provider served from its own "
                            "cache (cache_read), relayed verbatim."))
    panels.append(stat("Provider cache-write (tok)",
                       "fak_vcache_cache_creation_tokens_total or vector(0)", 20, 6,
                       desc="OBSERVED: cache_creation tokens — the write premium the "
                            "reads must repay. Net saving = read rebate minus this."))

    panels.append(row("Reuse ratio — the realized cache hit", 10))
    panels.append(timeseries(
        "KV-prefix reuse ratio",
        [("fak_gateway_kv_prefix_reuse_ratio", "session (cumulative)"),
         ("rate(fak_gateway_kv_prefix_reused_tokens_total[5m]) / "
          "rate(fak_gateway_kv_prefix_prompt_tokens_total[5m])", "5m window")],
        0, 11, w=12, h=8, unit="percentunit",
        desc="Cumulative and windowed reuse ratio. A falling 5m window while the "
             "session line holds is the first sign of leaving the frozen regime."))
    panels.append(timeseries(
        "Prefill tokens — total vs reused",
        [("rate(fak_gateway_kv_prefix_prompt_tokens_total[5m])", "prompt (prefill)"),
         ("rate(fak_gateway_kv_prefix_reused_tokens_total[5m])", "reused from cache")],
        12, 11, w=12, h=8,
        desc="Token-rate view of the same hit: the gap between the lines is prefill "
             "work actually redone."))

    panels.append(row("Regime buckets — the cache-cliff distribution", 19))
    panels.append(timeseries(
        "Turns by reuse regime (rate)",
        [("sum by (regime) (rate(fak_gateway_kv_prefix_turns_by_regime_total[5m]))",
          "{{regime}}")],
        0, 20, w=12, h=8,
        desc="frozen: reuse >= 0.90 (append-only ceiling); partial: 0.10-0.90; cold: "
             "< 0.10. Turns migrating frozen -> partial/cold is the cliff firing live."))
    panels.append(bargauge(
        "Turns by regime (session total)",
        "sum by (regime) (fak_gateway_kv_prefix_turns_by_regime_total)", "{{regime}}",
        12, 20, w=12, h=8,
        desc="Cumulative regime mix this session. A healthy single-linear agent is "
             "dominated by frozen turns after the first cold prefill."))

    panels.append(row("Managed-cache authoring — TTL upgrades & breakpoint placement", 28))
    panels.append(bargauge(
        "TTL-upgrade outcomes",
        "sum by (outcome) (fak_gateway_cache_ttl_upgrade_total)", "{{outcome}}",
        0, 29, w=12, h=8,
        desc="What fak's managed-cache transform did with the TTL upgrade per turn: "
             "upgraded = a longer-TTL breakpoint authored; the rest are structured "
             "skip reasons (no_stable_breakpoint, volatile_head, ...). All 0 until "
             "--managed-cache on traffic flows."))
    panels.append(bargauge(
        "Breakpoint-placement outcomes",
        "sum by (outcome) (fak_gateway_cache_breakpoint_placement_total)", "{{outcome}}",
        12, 29, w=12, h=8,
        desc="Where the cache_control breakpoint landed per turn: placed = fak set "
             "one; already_set = the harness's own breakpoint stood; the rest are "
             "skip reasons (no_stable_head, ...)."))

    panels.append(row("Provider cache — read vs write", 37))
    panels.append(timeseries(
        "Provider cache token rates (OBSERVED)",
        [("rate(fak_gateway_inference_cached_prompt_tokens_total[5m])",
          "cache-read (rebate)"),
         ("rate(fak_vcache_cache_creation_tokens_total[5m])",
          "cache-creation (write premium)")],
        0, 38, w=24, h=8,
        desc="The provider's own cache economics, relayed: reads must outpace writes "
             "for the cache to pay. A write spike with flat reads is a burst — a "
             "prefix that keeps getting re-authored instead of re-used."))

    panels.append(row("Tool defer — the cold-tool floor lever", 46))
    panels.append(stat("Cold tool defs deferred",
                       "fak_gateway_tool_defer_cold_total or vector(0)", 0, 47, w=6, h=8,
                       desc="WITNESSED: cold tool definitions marked defer_loading on "
                            "the outbound body (--defer-cold-tools, off by default)."))
    panels.append(stat("Turns with a deferral",
                       "fak_gateway_tool_defer_turns_total or vector(0)", 6, 47, w=6, h=8,
                       desc="Turns on which the lever fired (>=1 cold def deferred and "
                            "a tool_search_tool injected)."))
    panels.append(timeseries(
        "Tool-defer rate",
        [("rate(fak_gateway_tool_defer_cold_total[5m])", "cold defs deferred"),
         ("rate(fak_gateway_tool_defer_turns_total[5m])", "turns with a deferral")],
        12, 47, w=12, h=8,
        desc="Deferral activity over time. Flat 0 with the lever on means every "
             "advertised tool was hot."))

    return {
        "uid": "fak-cache-health",
        "title": "FAK Cache Health",
        "description": "Cache health on one screen (#3645, epic #3569): WITNESSED "
                       "in-kernel KV-prefix reuse ratio and regime buckets (the cache "
                       "cliff live), managed-cache TTL-upgrade and breakpoint-placement "
                       "outcomes, OBSERVED provider cache-read/creation token axes, and "
                       "the cold-tool deferral counters. Source: fak serve / fak guard "
                       "/metrics (scrape job fak_gateway). $-value questions live on "
                       "the cache-value roll-up instead.",
        "tags": ["fak", "cache", "prompt-caching", "observability"],
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


# The exporter is a separate scrape job (fak_fleet, :9098) with its own lifecycle, so
# "did anyone start it?" is a real operator question and both fleet dashboards lead with
# the answer instead of showing an all-zero screen that looks like a calm fleet.
FLEET_EXPORTER_NOTE = (
    "Source: **`fak fleet metrics --serve --addr 127.0.0.1:9098`** (scrape job "
    "`fak_fleet`). If every panel reads *No data*, the exporter is not running — start "
    "it, or check `up{job=\"fak_fleet\"}`.\n\n"
    "**Sources stay separate.** `fak_fleet_session_*` / `fak_fleet_sessions*` are "
    "**LIVE**, read from the durable session registry by an oracle (PCB state, heartbeat "
    "freshness, resume's idle-vs-TTL posture) — the same fold `fak session ls --durable` "
    "prints. `fak_fleet_run_*` are **DURABLE RUN HISTORY**, read from root session "
    "registrations (the default `--registration-ledger`). `fak_fleet_usage_*` are "
    "**HISTORICAL USAGE**, folded from the append-only "
    "gateway-usage ledger, and their token/cache numbers are OBSERVED (provider-relayed) "
    "except the `kv_prefix_*` pair, which fak authored itself."
)

# The drill-down door: one data link from the overview's session table into the
# per-session dashboard, carrying the clicked session id as the template variable.
# ${__url_time_range} is load-bearing, not decoration: without it Grafana opens the
# drill-down at ITS OWN default range, so an operator who zoomed the overview to the
# ten minutes around an incident would land on a dashboard showing the last 6h and
# read it as the same window. Carrying the range is what makes the two dashboards one
# surface at two zoom levels rather than two views that happen to link.
SESSION_DRILL_LINK = [{
    "title": "Drill down: this run",
    "url": "/d/fak-fleet-session/fak-run-operations-run-drill-down"
           "?var-session=${__value.raw}&${__url_time_range}",
}]

RUN_HISTORY_DRILL_LINK = [{
    "title": "Review this completed run",
    "url": "/d/fak-fleet-session/fak-run-operations-run-drill-down"
           "?var-session=${__value.raw}&${__url_time_range}",
}]


def _fleet_overview_sources_panels(panels):
    panels.append(text_panel(
        "Run Operations home",
        "**Operate current runs, then review completed ones.** The open/live table is a "
        "projection of current session inventory. The completed table is a separate "
        "projection of durable root-run registrations, so an exited process remains "
        "reviewable without being counted as live. Click either table to open the same "
        "run drill-down.\n\n" + FLEET_EXPORTER_NOTE,
        0, 0, w=24, h=7))

    panels.append(row("Run data sources & readiness", 7))
    panels.append(stat("Fleet source ready", 'max(up{job="fak_fleet"}) or vector(0)',
                       0, 8, w=6, h=4, thresholds=UP_TH,
                       desc="1 when Prometheus can scrape the exporter. Start or restart "
                            "with `tools/grafana/up.sh`; for a standalone source run "
                            "`fak fleet metrics --serve --addr 127.0.0.1:9098`."))
    panels.append(stat("Live inventory readable", "fak_fleet_registry_readable or vector(0)",
                       6, 8, w=6, h=4, thresholds=UP_TH,
                       desc="0 means the live descriptor registry could not be read. It "
                            "does not mean there are zero open runs."))
    panels.append(stat("Run history readable",
                       "fak_fleet_registration_registry_readable or vector(0)",
                       12, 8, w=6, h=4, thresholds=UP_TH,
                       desc="0 means the durable registration ledger could not be read. "
                            "It does not mean no runs have completed."))
    panels.append(stat("Fold freshness", "time() - fak_fleet_snapshot_unixtime",
                       18, 8, w=6, h=4, unit="s", thresholds=AGE_TH,
                       desc="Age of the exporter's last fold; it should stay near the "
                            "Prometheus scrape interval."))


def _fleet_overview_live_inventory_panels(panels):
    panels.append(row("Fleet right now (LIVE inventory)", 12))
    panels.append(stat("Sessions", "fak_fleet_sessions or vector(0)", 0, 13, w=3, h=4,
                       desc="Sessions in the durable inventory — the same population "
                            "`fak session ls --durable` prints."))
    panels.append(stat("Live", 'fak_fleet_sessions_by_liveness{liveness="live"} or vector(0)',
                       3, 13, w=3, h=4, color="fixed",
                       desc="Running-family sessions whose durable heartbeat is fresh: "
                            "alive AND progressing."))
    panels.append(stat("Stalled", 'fak_fleet_sessions_by_liveness{liveness="stalled"} or vector(0)',
                       6, 13, w=3, h=4, thresholds=ONE_RED,
                       desc="Sessions that CLAIM a running state but whose heartbeat has "
                            "not advanced within the stale window — claiming work, not "
                            "progressing. This is the number worth alerting on."))
    panels.append(stat("Idle", 'fak_fleet_sessions_by_liveness{liveness="idle"} or vector(0)',
                       9, 13, w=3, h=4,
                       desc="Paused / stopped sessions. Idle is a resting state, not a fault."))
    panels.append(stat("Hosts", "fak_fleet_hosts or vector(0)", 12, 13, w=3, h=4,
                       desc="Distinct hosts carrying at least one session."))
    panels.append(stat("Warm cache posture", "fak_fleet_sessions_warm or vector(0)",
                       15, 13, w=3, h=4,
                       desc="Sessions projected WARM by resume's idle-vs-TTL rule. A "
                            "projection, never a witnessed provider-cache hit."))
    panels.append(stat("Registered runs", "fak_fleet_registered_runs or vector(0)",
                       18, 13, w=3, h=4,
                       desc="Root runs retained in durable lifecycle history. This count "
                            "includes open and terminal registrations and is NOT liveness."))
    panels.append(stat("Per-session rows dropped", "fak_fleet_sessions_truncated or vector(0)",
                       21, 13, w=3, h=4, thresholds=ONE_RED,
                       desc="Sessions the --max-sessions cardinality bound kept OUT of the "
                            "table below. Non-zero means the table is INCOMPLETE."))


def _fleet_overview_open_runs_panels(panels):
    panels.append(row("All open / live runs — click one to drill down", 17))
    panels.append(info_table(
        "Open / live runs (LIVE inventory)",
        'fak_fleet_session_info{state=~"RUNNING|STALLED|THROTTLED|PAUSED|DRAINING"}',
        0, 18, w=24, h=10,
        exclude=["fleet", "service"], links=SESSION_DRILL_LINK, link_field="session",
        sort_by="session", filterable=True,
        desc="One row per non-terminal session in the LIVE descriptor inventory. `state` is the "
             "EFFECTIVE status: a RUNNING session with a lapsed heartbeat reads STALLED "
             "here, exactly as it does in the CLI headline. This table never reconstructs "
             "liveness from historical registration state."))


def _fleet_overview_completed_runs_panels(panels):
    """Completed-run review reads the DURABLE root-run registration ledger — a
    separate projection from the live inventory rows above, so an exited process
    stays reviewable without being counted as live."""
    panels.append(row("Completed-run review (DURABLE REGISTRATIONS)", 28))
    panels.append(stat("Completed registrations",
                       'sum(fak_fleet_registered_runs_by_state{state="completed"}) or vector(0)',
                       0, 29, w=6, h=4,
                       desc="Root runs whose latest durable lifecycle state is completed."))
    panels.append(stat("Failed / cancelled / lost",
                       'sum(fak_fleet_registered_runs_by_state{state=~"failed|cancelled|lost|reaped"}) or vector(0)',
                       6, 29, w=6, h=4, thresholds=ONE_RED,
                       desc="Terminal root registrations that did not complete normally."))
    panels.append(stat("Registration source ready",
                       "fak_fleet_registration_registry_readable or vector(0)",
                       12, 29, w=6, h=4, thresholds=UP_TH,
                       desc="Readiness of the durable run-history source."))
    completed_run_info = 'fak_fleet_run_info{state=~"completed|failed|cancelled|lost|reaped"}'
    panels.append(run_history_table(
        "Completed runs (DURABLE registration history)", completed_run_info, "",
        0, 33, w=24, h=10, links=RUN_HISTORY_DRILL_LINK, link_field="session",
        restrict_numeric_to_info=True,
        desc="One row per terminal root registration, retained after the live process "
             "exits. Creation, start, terminal time, duration, terminal reason, and "
             "witness reference support historical review; provenance remains "
             "`source=durable_registration`. Click its session id to review the run."))


def _fleet_overview_inventory_history_panels(panels):
    panels.append(timeseries(
        "Sessions by state over time",
        [('fak_fleet_sessions_by_state{state="RUNNING"}', "RUNNING"),
         ('fak_fleet_sessions_by_state{state="STALLED"}', "STALLED"),
         ('fak_fleet_sessions_by_state{state="THROTTLED"}', "THROTTLED"),
         ('fak_fleet_sessions_by_state{state="PAUSED"}', "PAUSED"),
         ('fak_fleet_sessions_by_state{state="DRAINING"}', "DRAINING"),
         ('fak_fleet_sessions_by_state{state="STOPPED"}', "STOPPED")],
        0, 43, w=12, h=8,
        desc="Prometheus samples of the LIVE inventory over the selected time range. "
             "This is not durable completed-run history; use the registration table above."))
    panels.append(timeseries(
        "Liveness & cache posture over time",
        [('fak_fleet_sessions_by_liveness{liveness="live"}', "live"),
         ('fak_fleet_sessions_by_liveness{liveness="stalled"}', "stalled"),
         ('fak_fleet_sessions_by_liveness{liveness="idle"}', "idle"),
         ("fak_fleet_sessions_warm", "warm posture"),
         ("fak_fleet_sessions_cold", "cold posture")],
        12, 43, w=12, h=8,
        desc="Stalled climbing while live falls is the fleet wedging."))
    panels.append(bargauge(
        "Session age", "fak_fleet_session_age_seconds", "{{session}}", 0, 51, w=12, h=8,
        unit="s",
        desc="Seconds since each session was created. A very old session that is also "
             "STALLED is the classic reap candidate."))
    panels.append(bargauge(
        "Sessions per host", "fak_fleet_sessions_by_host", "{{host}}", 12, 51, w=12, h=8,
        desc="Where the fleet is running. `unknown` is a session whose descriptor carries "
             "no host stamp, not a missing machine."))


def _fleet_overview_usage_panels(panels):
    panels.append(row("Roll-up — historical cost & cache (gateway-usage ledger)", 59))
    panels.append(stat("Sessions folded", "fak_fleet_usage_sessions or vector(0)",
                       0, 60, w=3, h=4,
                       desc="Sessions in the ledger fold. A carryforward row (written by a "
                            "ledger cut) expands to the row count it stands for, so a cut "
                            "never silently shrinks this number."))
    panels.append(stat("Turns", "fak_fleet_usage_turns or vector(0)", 3, 60, w=3, h=4,
                       desc="OBSERVED served turns across the folded corpus."))
    panels.append(stat("Cache read ratio", "fak_fleet_usage_cache_read_ratio",
                       6, 60, w=3, h=4, unit="percentunit", thresholds=CACHE_TH,
                       desc="cache_read / (input + cache_read + cache_creation). The "
                            "denominator includes tokens paid to CREATE cache entries, so "
                            "a session that only wrote cache is not credited with a hit."))
    panels.append(stat("Cache read tokens", "fak_fleet_usage_cache_read_tokens or vector(0)",
                       9, 60, w=3, h=4,
                       desc="OBSERVED prompt tokens the provider served from its cache."))
    panels.append(stat("Cache creation tokens",
                       "fak_fleet_usage_cache_creation_tokens or vector(0)", 12, 60, w=3, h=4,
                       desc="OBSERVED prompt tokens written INTO the provider's cache — a "
                            "cost, not a saving."))
    panels.append(stat("Input tokens", "fak_fleet_usage_input_tokens or vector(0)",
                       15, 60, w=3, h=4, desc="OBSERVED uncached prompt tokens."))
    panels.append(stat("Output tokens", "fak_fleet_usage_output_tokens or vector(0)",
                       18, 60, w=3, h=4, desc="OBSERVED completion tokens."))
    panels.append(stat(
        "Drill-down coverage",
        "fak_fleet_usage_sessions_identified / clamp_min("
        "fak_fleet_usage_sessions_identified + fak_fleet_usage_sessions_unidentified, 1)",
        21, 60, w=3, h=4, unit="percentunit", thresholds=CACHE_TH,
        desc="THE HONESTY GAUGE for the per-session panels below. The ledger's session_id "
             "is optional: a row written without one counts in every fleet total but can "
             "never appear in a per-session panel. Low coverage means the per-session "
             "cost view speaks for only a slice of the fleet. `fak serve` rows are "
             "deliberately unidentified — one serve process hosts a whole session table, "
             "so no single id would be true for its exit row."))
    panels.append(timeseries(
        "Token economy by session type",
        [('fak_fleet_usage_by_type_input_tokens{session_type="guard"}', "guard input"),
         ('fak_fleet_usage_by_type_cache_read_tokens{session_type="guard"}', "guard cache read"),
         ('fak_fleet_usage_by_type_output_tokens{session_type="guard"}', "guard output"),
         ('fak_fleet_usage_by_type_input_tokens{session_type="serve"}', "serve input"),
         ('fak_fleet_usage_by_type_cache_read_tokens{session_type="serve"}', "serve cache read"),
         ('fak_fleet_usage_by_type_output_tokens{session_type="serve"}', "serve output")],
        0, 64, w=12, h=8,
        desc="The guard path (one process, one agent session) and the serve path (one "
             "process, many sessions) have different economics — never read them as one."))
    panels.append(timeseries(
        "Cache read ratio by session type",
        [("fak_fleet_usage_by_type_cache_read_ratio", "{{session_type}}"),
         ("fak_fleet_usage_cache_read_ratio", "fleet")],
        12, 64, w=12, h=8, unit="percentunit",
        desc="A session type whose ratio is ABSENT folded no prompt tokens — unmeasured, "
             "not 0%."))
    panels.append(bargauge(
        "Top sessions by output tokens", "topk(10, fak_fleet_usage_session_output_tokens)",
        "{{session}}", 0, 72, w=12, h=8,
        desc="The fleet's biggest spenders, by name. Only sessions whose ledger rows carry "
             "a session_id can appear — see the drill-down coverage tile."))
    panels.append(bargauge(
        "Lowest cache read ratio by session",
        "bottomk(10, fak_fleet_usage_session_cache_read_ratio)", "{{session}}",
        12, 72, w=12, h=8, unit="percentunit", maxv=1,
        desc="Where cache reuse is worst. A session that folded no prompt tokens is "
             "ABSENT rather than ranked at 0% — unmeasured must not outrank genuinely bad."))


def _fleet_overview_adjudication_panels(panels):
    panels.append(row("Adjudication (fleet)", 80))
    panels.append(timeseries(
        "Kernel verdicts across the fleet",
        [('fak_fleet_usage_adjudications_by_verdict{verdict="allowed"}', "allowed"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="denied"}', "denied"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="transformed"}', "transformed"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="quarantined"}', "quarantined"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="deferred"}', "deferred"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="escalated"}', "escalated"),
         ('fak_fleet_usage_adjudications_by_verdict{verdict="errored"}', "errored")],
        0, 81, w=24, h=8,
        desc="The fleet-wide verdict mix, folded from the durable ledger. The per-session "
             "adjudication counts live on the drill-down dashboard."))
    panels.append(stat("Ledger rows folded", "fak_fleet_usage_rows or vector(0)",
                       0, 89, w=4, h=4,
                       desc="Rows the exporter folded after deduping by row key."))
    panels.append(stat("Duplicate rows dropped",
                       "fak_fleet_usage_duplicate_rows_dropped or vector(0)", 4, 89, w=4, h=4,
                       desc="Rows dropped as byte-identical duplicates of an already-folded "
                            "row key (a retried flush). Ledger health, not fleet activity."))
    panels.append(stat("Ledger last write",
                       "time() - fak_fleet_usage_window_end_unixtime", 8, 89, w=4, h=4,
                       unit="s", thresholds=AGE_TH,
                       desc="Age of the newest ledger row. A value that keeps climbing "
                            "means sessions stopped WRITING, which is not the same as the "
                            "fleet going quiet."))
    panels.append(stat("Fold freshness", "time() - fak_fleet_snapshot_unixtime",
                       12, 89, w=4, h=4, unit="s", thresholds=AGE_TH,
                       desc="Age of the exporter's own fold. It re-folds on every scrape, "
                            "so this should stay near the scrape interval."))


def build_fleet_overview():
    panels = []
    _fleet_overview_sources_panels(panels)
    _fleet_overview_live_inventory_panels(panels)
    _fleet_overview_open_runs_panels(panels)
    _fleet_overview_completed_runs_panels(panels)
    _fleet_overview_inventory_history_panels(panels)
    _fleet_overview_usage_panels(panels)
    _fleet_overview_adjudication_panels(panels)
    return {
        "uid": "fak-fleet-overview",
        "title": "FAK Run Operations",
        "description": "The default run-operations home: all open/live runs from the "
                       "current descriptor inventory, completed-run review from durable "
                       "root registrations, and separate historical usage economics. "
                       "Stable UID fak-fleet-overview; source: `fak fleet metrics --serve` "
                       "(scrape job fak_fleet).",
        "tags": ["fak", "fleet", "runs", "operations", "observability"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {}, "links": [{
            "asDropdown": False, "icon": "external link", "includeVars": False,
            "keepTime": True, "tags": [], "targetBlank": False, "title": "Run drill-down",
            "tooltip": "Open the per-run view", "type": "link",
            "url": "/d/fak-fleet-session/fak-run-operations-run-drill-down"}],
        "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [{
            "name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
            "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
            "regex": "", "options": [], "includeAll": False, "multi": False}]},
        "panels": panels,
    }


def build_fleet_session():
    panels = []

    panels.append(text_panel(
        "Run drill-down",
        "Everything the fleet knows about run/session **`$session`**. Pick another from "
        "the `Run / session` selector above, or return to "
        "[Run Operations](/d/fak-fleet-overview/fak-run-operations).\n\n"
        "**If the LIVE panels are populated but the HISTORICAL ones are empty**, this "
        "session simply has no ledger rows carrying its id yet — the historical row is "
        "written at session EXIT, and `fak serve` rows are deliberately unidentified. "
        "If only the durable registration panel is populated, the run has exited and is "
        "correctly absent from live inventory. That is missing usage or liveness, not a "
        "zero.\n\n" + FLEET_EXPORTER_NOTE,
        0, 0, w=24, h=8))

    panels.append(row("This session, right now (LIVE)", 8))
    panels.append(stat("Live", 'fak_fleet_session_live{session="$session"}', 0, 9, w=4, h=4,
                       thresholds=UP_TH,
                       desc="1 when the heartbeat is fresh AND the session claims a "
                            "running-family state."))
    panels.append(stat("Stalled", 'fak_fleet_session_stalled{session="$session"}',
                       4, 9, w=4, h=4, thresholds=ONE_RED,
                       desc="1 when this session claims work but its heartbeat has not "
                            "advanced within the stale window."))
    panels.append(stat("Age", 'fak_fleet_session_age_seconds{session="$session"}',
                       8, 9, w=4, h=4, unit="s",
                       desc="Seconds since the session was created."))
    panels.append(stat("Cache posture warm", 'fak_fleet_session_cache_warm{session="$session"}',
                       12, 9, w=4, h=4, thresholds=UP_TH,
                       desc="resume's idle-vs-TTL projection. A projection, never a "
                            "witnessed provider-cache hit."))
    panels.append(stat("Generation", 'fak_fleet_session_generation{session="$session"}',
                       16, 9, w=4, h=4,
                       desc="Re-continuation depth — how many budget-reset generations this "
                            "session has been through. Absent when it has never reset."))
    panels.append(stat("Priority", 'fak_fleet_session_priority{session="$session"}',
                       20, 9, w=4, h=4,
                       desc="Scheduling priority as persisted in the durable registry. "
                            "Absent at the default of 0."))
    panels.append(info_table(
        "Descriptor", 'fak_fleet_session_info{session="$session"}', 0, 13, w=24, h=5,
        exclude=["fleet", "service"],
        desc="The durable descriptor's labels. `reason` carries the closed token a "
             "THROTTLED/STOPPED session stopped for; `parent` is the re-continuation "
             "lineage parent."))
    panels.append(run_history_table(
        "Durable run registration", 'fak_fleet_run_info{session="$session"}',
        '{session="$session"}', 0, 18, w=24, h=7,
        desc="Root-run lifecycle history with creation, start, terminal time, duration, "
             "terminal reason, and witness reference. `source=durable_registration` "
             "preserves provenance after the LIVE descriptor disappears."))
    panels.append(timeseries(
        "State & liveness over time",
        [('fak_fleet_session_live{session="$session"}', "live"),
         ('fak_fleet_session_stalled{session="$session"}', "stalled"),
         ('fak_fleet_session_cache_warm{session="$session"}', "cache warm")],
        0, 23, w=12, h=8,
        desc="The moment a session went from live to stalled is the moment worth "
             "correlating against everything else on this page."))
    panels.append(timeseries(
        "Wall-clock budget",
        [('fak_fleet_session_time_elapsed_seconds{session="$session"}', "elapsed"),
         ('fak_fleet_session_time_limit_seconds{session="$session"}', "limit")],
        12, 23, w=12, h=8, unit="s",
        desc="Emitted only when a budget LIMIT is set — an unbudgeted session has no "
             "denominator and is absent here rather than reading as fully burned."))

    panels.append(row("This run's cost & cache (HISTORICAL USAGE)", 31))
    panels.append(stat("Turns", 'fak_fleet_usage_session_turns{session="$session"}',
                       0, 32, w=4, h=4, desc="OBSERVED served turns."))
    panels.append(stat("Cache read ratio",
                       'fak_fleet_usage_session_cache_read_ratio{session="$session"}',
                       4, 32, w=4, h=4, unit="percentunit", thresholds=CACHE_TH,
                       desc="cache_read / (input + cache_read + cache_creation). ABSENT "
                            "when this session folded no prompt tokens — unmeasured, not 0%."))
    panels.append(stat("Input tokens",
                       'fak_fleet_usage_session_input_tokens{session="$session"}',
                       8, 32, w=4, h=4, desc="OBSERVED uncached prompt tokens."))
    panels.append(stat("Cache read tokens",
                       'fak_fleet_usage_session_cache_read_tokens{session="$session"}',
                       12, 32, w=4, h=4,
                       desc="OBSERVED prompt tokens served from the provider's cache."))
    panels.append(stat("Cache creation tokens",
                       'fak_fleet_usage_session_cache_creation_tokens{session="$session"}',
                       16, 32, w=4, h=4,
                       desc="OBSERVED prompt tokens written INTO the cache — a cost."))
    panels.append(stat("Output tokens",
                       'fak_fleet_usage_session_output_tokens{session="$session"}',
                       20, 32, w=4, h=4, desc="OBSERVED completion tokens."))
    panels.append(timeseries(
        "Token economy for this session",
        [('fak_fleet_usage_session_input_tokens{session="$session"}', "input"),
         ('fak_fleet_usage_session_cache_read_tokens{session="$session"}', "cache read"),
         ('fak_fleet_usage_session_cache_creation_tokens{session="$session"}', "cache creation"),
         ('fak_fleet_usage_session_output_tokens{session="$session"}', "output")],
        0, 36, w=12, h=8,
        desc="Cumulative per-session totals as the ledger fold sees them; they step when "
             "a new exit row for this session lands."))
    panels.append(timeseries(
        "In-kernel KV-prefix reuse (WITNESSED)",
        [('fak_fleet_usage_session_kv_prefix_prompt_tokens{session="$session"}', "eligible"),
         ('fak_fleet_usage_session_kv_prefix_reused_tokens{session="$session"}', "reused")],
        12, 36, w=12, h=8,
        desc="fak-authored reuse from the kernel-owned KV cache — a different signal from "
             "the provider prompt-cache axes above, and never summed with them."))
    panels.append(timeseries(
        "Kernel verdicts for this session",
        [('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="allowed"}', "allowed"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="denied"}', "denied"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="transformed"}', "transformed"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="quarantined"}', "quarantined"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="deferred"}', "deferred"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="escalated"}', "escalated"),
         ('fak_fleet_usage_session_adjudications_by_verdict{session="$session",verdict="errored"}', "errored")],
        0, 44, w=16, h=8,
        desc="What the kernel decided on this session's tool calls."))
    panels.append(stat("Session wall-clock",
                       'fak_fleet_usage_session_seconds{session="$session"}',
                       16, 44, w=4, h=8, unit="s",
                       desc="Summed uptime across this session's ledger rows."))
    panels.append(stat("Ledger rows for this session",
                       'fak_fleet_usage_session_sessions{session="$session"}',
                       20, 44, w=4, h=8,
                       desc="How many folded rows carry this id. 0 / No data means the "
                            "session has not exited yet, or its rows were written without "
                            "an id."))

    return {
        "uid": "fak-fleet-session",
        "title": "FAK Run Operations — Run Drill-down",
        "description": "One run, end to end: durable registration identity, current LIVE "
                       "registry state when present, and separate HISTORICAL usage from "
                       "the gateway-usage ledger. Reached from either run table on the "
                       "Run Operations home. Source: "
                       "`fak fleet metrics --serve` (scrape job fak_fleet).",
        "tags": ["fak", "fleet", "runs", "operations", "observability"],
        "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "schemaVersion": 39, "version": 1, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {}, "links": [{
            "asDropdown": False, "icon": "external link", "includeVars": False,
            "keepTime": True, "tags": [], "targetBlank": False, "title": "Run Operations",
            "tooltip": "Back to all runs", "type": "link",
            "url": "/d/fak-fleet-overview/fak-run-operations"}],
        "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts", "type": "dashboard"}]},
        "templating": {"list": [
            {"name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource",
             "query": "prometheus", "current": {}, "hide": 2, "refresh": 1,
             "regex": "", "options": [], "includeAll": False, "multi": False},
            # The drill-down axis spans both source tiers. LIVE inventory keeps current
            # sessions selectable; durable run registrations keep completed runs in the
            # selector after their descriptors disappear.
            {"name": "session", "label": "Run / session", "type": "query",
             "datasource": DS,
             "definition": "label_values({__name__=~\"fak_fleet_session_info|fak_fleet_run_info\"}, session)",
             "query": {"qryType": 1,
                       "query": "label_values({__name__=~\"fak_fleet_session_info|fak_fleet_run_info\"}, session)",
                       "refId": "PrometheusVariableQueryEditor-VariableQuery"},
             "current": {}, "options": [], "refresh": 2, "regex": "", "sort": 1,
             "hide": 0, "includeAll": False, "multi": False, "skipUrlSync": False},
        ]},
        "panels": panels,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for path, dash in ((OUT, build()), (OUT_GATEWAY, build_gateway()),
                       (OUT_DOGFOOD, build_dogfood()), (OUT_STARTUP, build_startup()),
                       (OUT_GUARD, build_guard()), (OUT_ROLLUP, build_cache_ablation_rollup()),
                       (OUT_CACHEHEALTH, build_cache_health()),
                       (OUT_FLEET_OVERVIEW, build_fleet_overview()),
                       (OUT_FLEET_SESSION, build_fleet_session())):
        # newline="\n" pins LF so a Windows regen is byte-identical to the committed
        # (LF) dashboards instead of churning every line to CRLF — the tree stays clean
        # for a reviewer who re-runs the generator.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dash, f, indent=2)
            f.write("\n")
        n_panels = sum(1 for p in dash["panels"] if p["type"] != "row")
        print(f"wrote {path}  ({n_panels} data panels, {len(dash['panels'])} total incl. rows)")


if __name__ == "__main__":
    main()
