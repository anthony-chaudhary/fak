#!/usr/bin/env python3
"""Panel-construction tests for generated Grafana dashboards."""

import json
from pathlib import Path

import gen_dashboard as dashboard


HERE = Path(__file__).resolve().parent


def test_steps_preserves_absolute_threshold_order_and_null_floor():
    thresholds = dashboard.steps((None, "green"), (30, "yellow"), (80, "red"))
    assert thresholds == {
        "mode": "absolute",
        "steps": [
            {"color": "green", "value": None},
            {"color": "yellow", "value": 30},
            {"color": "red", "value": 80},
        ],
    }


def test_stat_panel_binds_query_geometry_and_legend():
    panel = dashboard.stat(
        "Workers",
        "fleet_workers_active",
        x=2,
        y=3,
        w=5,
        h=6,
        unit="short",
        legend="active",
    )
    assert panel["type"] == "stat"
    assert panel["gridPos"] == {"h": 6, "w": 5, "x": 2, "y": 3}
    assert panel["targets"] == [
        {
            "datasource": dashboard.DS,
            "expr": "fleet_workers_active",
            "refId": "A",
            "legendFormat": "active",
        }
    ]


def test_info_table_links_only_the_requested_field_and_drops_bookkeeping():
    links = [{"title": "session", "url": "/d/session"}]
    panel = dashboard.info_table(
        "Sessions",
        "fak_fleet_session_info",
        0,
        0,
        links=links,
        link_field="session",
        exclude=["fleet"],
        sort_by="session",
        filterable=True,
    )
    override = panel["fieldConfig"]["overrides"][0]
    assert override["matcher"]["options"] == "session"
    assert override["properties"][0]["value"] == links
    excluded = panel["transformations"][0]["options"]["excludeByName"]
    assert excluded["Value"] and excluded["fleet"]
    assert panel["fieldConfig"]["defaults"]["custom"]["filterable"] is True


def test_run_operations_is_stable_uid_default_home():
    home = dashboard.build_fleet_overview()
    drill = dashboard.build_fleet_session()

    assert home["uid"] == "fak-fleet-overview"
    assert home["title"] == "FAK Run Operations"
    assert drill["uid"] == "fak-fleet-session"
    assert drill["title"] == "FAK Run Operations — Run Drill-down"
    compose = (HERE / "docker-compose.yml").read_text(encoding="utf-8")
    assert (
        "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH="
        "/var/lib/grafana/dashboards/fak-fleet-overview.json"
    ) in compose


def test_run_operations_keeps_live_inventory_and_registration_history_separate():
    panels = {panel["title"]: panel for panel in dashboard.build_fleet_overview()["panels"]}
    live = panels["Open / live runs (LIVE inventory)"]
    history = panels["Completed runs (DURABLE registration history)"]

    assert live["targets"][0]["expr"] == (
        'fak_fleet_session_info{state=~"RUNNING|STALLED|THROTTLED|PAUSED|DRAINING"}'
    )
    completed_info = 'fak_fleet_run_info{state=~"completed|failed|cancelled|lost|reaped"}'
    assert history["targets"][0]["expr"] == completed_info
    assert [target["expr"] for target in history["targets"][1:]] == [
        f"(({metric} and on(run, session) {completed_info}) > 0) * 1000"
        for metric in (
            "fak_fleet_run_created_timestamp_seconds",
            "fak_fleet_run_started_timestamp_seconds",
            "fak_fleet_run_terminal_timestamp_seconds",
        )
    ] + [f"fak_fleet_run_duration_seconds and on(run, session) {completed_info}"]
    assert [transform["id"] for transform in history["transformations"]] == [
        "joinByField",
        "organize",
    ]
    excluded = history["transformations"][1]["options"]["excludeByName"]
    assert all(excluded[f"session #{ref}"] for ref in "BCDE")
    rename = history["transformations"][1]["options"]["renameByName"]
    assert rename == {
        "Value #B": "created",
        "Value #C": "started",
        "Value #D": "terminal",
        "Value #E": "duration",
    }
    assert {
        override["matcher"]["options"]
        for override in history["fieldConfig"]["overrides"]
    } >= {"created", "started", "terminal", "duration", "session"}
    for panel in (live, history):
        session_override = next(
            override for override in panel["fieldConfig"]["overrides"]
            if override["matcher"]["options"] == "session"
        )
        link = session_override["properties"][0]["value"][0]
        assert "/d/fak-fleet-session/fak-run-operations-run-drill-down" in link["url"]
        assert "var-session=${__value.raw}" in link["url"]
        assert "${__url_time_range}" in link["url"]

    assert panels["Fleet source ready"]["targets"][0]["expr"] == (
        'max(up{job="fak_fleet"}) or vector(0)'
    )
    assert panels["Run history readable"]["targets"][0]["expr"] == (
        "fak_fleet_registration_registry_readable or vector(0)"
    )


def test_run_drill_down_selector_keeps_completed_registered_runs_reachable():
    drill = dashboard.build_fleet_session()
    variables = {variable["name"]: variable for variable in drill["templating"]["list"]}
    query = variables["session"]["query"]["query"]
    panels = {panel["title"]: panel for panel in drill["panels"]}

    assert query == (
        'label_values({__name__=~"fak_fleet_session_info|fak_fleet_run_info"}, session)'
    )
    registration = panels["Durable run registration"]
    assert registration["targets"][0]["expr"] == 'fak_fleet_run_info{session="$session"}'
    assert [target["expr"] for target in registration["targets"][1:]] == [
        f'(({metric}{{session="$session"}}) > 0) * 1000'
        for metric in (
            "fak_fleet_run_created_timestamp_seconds",
            "fak_fleet_run_started_timestamp_seconds",
            "fak_fleet_run_terminal_timestamp_seconds",
        )
    ] + ['fak_fleet_run_duration_seconds{session="$session"}']
    assert [transform["id"] for transform in registration["transformations"]] == [
        "joinByField",
        "organize",
    ]
    assert drill["links"][0]["url"] == "/d/fak-fleet-overview/fak-run-operations"


def test_checked_in_run_dashboards_match_generator():
    # main() allocates panel IDs across dashboards in this order. Reproduce its build
    # phase without writing unrelated dashboards, so this assertion matches a normal
    # generator run while keeping the test hermetic.
    dashboard._id[0] = 0
    for builder in (
        dashboard.build,
        dashboard.build_gateway,
        dashboard.build_dogfood,
        dashboard.build_startup,
        dashboard.build_guard,
        dashboard.build_cache_ablation_rollup,
        dashboard.build_cache_health,
    ):
        builder()
    expected = {
        "fak-fleet-overview.json": dashboard.build_fleet_overview(),
        "fak-fleet-session.json": dashboard.build_fleet_session(),
    }
    for name, generated in expected.items():
        checked_in = json.loads((HERE / "dashboards" / name).read_text(encoding="utf-8"))
        assert checked_in == generated, f"{name} must be regenerated with gen_dashboard.py"


def test_guard_refusal_panels_break_down_by_reason_and_refusal_subtype():
    dashboard._id[0] = 0
    guard = dashboard.build_guard()
    panels = {panel["title"]: panel for panel in guard["panels"] if panel.get("title")}
    rate_panel = panels["Refusal rate by reason"]
    total_panel = panels["Refusals by reason (session total)"]

    assert "sum by (reason, refusal_subtype)" in rate_panel["targets"][0]["expr"]
    assert rate_panel["targets"][0]["legendFormat"] == "{{reason}} {{refusal_subtype}}"
    assert "sum by (reason, refusal_subtype)" in total_panel["targets"][0]["expr"]
    assert total_panel["targets"][0]["legendFormat"] == "{{reason}} {{refusal_subtype}}"
