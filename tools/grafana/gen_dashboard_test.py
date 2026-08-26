#!/usr/bin/env python3
"""Panel-construction tests for generated Grafana dashboards."""

import gen_dashboard as dashboard


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

