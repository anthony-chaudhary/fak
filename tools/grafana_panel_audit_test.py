#!/usr/bin/env python3
"""Tests for tools/grafana_panel_audit.py.

Pure-stdlib + unittest.mock (works both via `python tools/grafana_panel_audit_test.py`
and `python -m pytest`). Never hits a live Prometheus endpoint.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grafana_panel_audit as gpa  # noqa: E402


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class GrafanaPanelAuditTest(unittest.TestCase):
    def test_extract_panels_flattens_nested_row_panels(self):
        panels = [
            {"id": 1, "type": "stat", "title": "Top"},
            {"id": 2, "type": "row", "title": "Collapsed", "panels": [
                {"id": 3, "type": "timeseries", "title": "Nested A"},
                {"id": 4, "type": "row", "title": "Inner", "panels": [
                    {"id": 5, "type": "table", "title": "Deep B"},
                ]},
            ]},
            {"id": 6, "type": "row", "title": "EmptyRow", "panels": []},
            {"id": 7, "type": "bargauge", "title": "Bottom"},
        ]
        got = gpa.extract_panels(panels)
        self.assertEqual([p["title"] for p in got], ["Top", "Nested A", "Deep B", "Bottom"])

    def test_query_prometheus_classifies_all_response_states(self):
        seen_urls: list[str] = []

        def fake_urlopen(req, timeout=5):
            seen_urls.append(req.full_url)
            return _FakeHTTPResponse(current_payload)

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            current_payload = {
                "status": "success",
                "data": {"result": [{"value": [1, "0"]}, {"value": [1, "2.5"]}]},
            }
            status, samples = gpa.query_prometheus('rate(fak_gateway_up{job="fak"}[5m])')
            self.assertEqual(status, "NON_ZERO")
            self.assertEqual(samples, [0.0, 2.5])
            self.assertIn("rate%28fak_gateway_up", seen_urls[-1])

            current_payload = {
                "status": "success",
                "data": {"result": [{"value": [1, "0"]}, {"value": [1, "0.0"]}]},
            }
            status, samples = gpa.query_prometheus("fak_gateway_up")
            self.assertEqual(status, "ZERO")
            self.assertEqual(samples, [0.0, 0.0])

            current_payload = {"status": "success", "data": {"result": []}}
            status, detail = gpa.query_prometheus("fak_missing")
            self.assertEqual((status, detail), ("NO_DATA", None))

            current_payload = {"status": "error", "error": "parse error"}
            status, detail = gpa.query_prometheus("bad(")
            self.assertEqual((status, detail), ("ERROR", "parse error"))

        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            status, detail = gpa.query_prometheus("fak_gateway_up")
            self.assertEqual(status, "ERROR")
            self.assertIn("connection refused", detail)

    def test_audit_dashboard_resolves_variables_and_interval_macros(self):
        dash = {
            "uid": "demo-board",
            "title": "Demo Board",
            "templating": {
                "list": [
                    {"name": "env", "current": {"value": ["prod", "staging"]}},
                    {"name": "route", "current": {"value": "$__all"}},
                    {"name": "model", "current": {}},
                ]
            },
            "panels": [
                {
                    "type": "row",
                    "title": "Latency",
                    "panels": [
                        {
                            "title": "P95",
                            "type": "timeseries",
                            "targets": [
                                {"refId": "A", "expr": "   "},
                                {
                                    "refId": "B",
                                    "expr": (
                                        'sum(rate(fak_req_total{env="$env",route=~"${route}",'
                                        'model="$model"}[$__rate_interval])) / '
                                        "scalar(count_over_time(fak_up[$__range:$__interval]))"
                                    ),
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "demo.json"
            p.write_text(json.dumps(dash), encoding="utf-8")
            with mock.patch.object(gpa, "query_prometheus", return_value=("NON_ZERO", [1.0])) as m:
                uid, title, results = gpa.audit_dashboard(p, default_vars={"model": "qwen38"})

        self.assertEqual((uid, title), ("demo-board", "Demo Board"))
        self.assertEqual(len(results), 1)
        clean = results[0]["clean_expr"]
        self.assertEqual(results[0]["ref_id"], "B")
        self.assertIn('env="prod"', clean)
        self.assertIn('route=~".*"', clean)
        self.assertIn('model="qwen38"', clean)
        self.assertIn("[5m]", clean)
        self.assertIn("[1h:1m]", clean)
        m.assert_called_once_with(clean)

    def test_main_verify_all_and_details_gates_on_nodata_and_error(self):
        dash = {
            "uid": "audit-smoke",
            "title": "Audit Smoke",
            "panels": [
                {"title": "Q1", "type": "stat", "targets": [{"refId": "A", "expr": "fak_a"}]},
                {"title": "Q2", "type": "stat", "targets": [{"refId": "B", "expr": "fak_b"}]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text(json.dumps(dash), encoding="utf-8")
            with mock.patch("glob.glob", return_value=[str(p)]):
                # ZERO + NON_ZERO under --verify-all -> exit cleanly (no SystemExit)
                buf = io.StringIO()
                with mock.patch.object(gpa, "query_prometheus", side_effect=[("NON_ZERO", [1.0]), ("ZERO", [0.0])]), \
                     mock.patch.object(sys, "argv", ["grafana_panel_audit.py", "--verify-all", "--details"]), \
                     contextlib.redirect_stdout(buf):
                    gpa.main()
                out = buf.getvalue()
                self.assertIn("audit-smoke", out)
                self.assertIn("[ZERO] Q2 (B): fak_b", out)

                # NO_DATA under --verify-all -> sys.exit(1)
                buf = io.StringIO()
                with mock.patch.object(gpa, "query_prometheus", side_effect=[("NON_ZERO", [1.0]), ("NO_DATA", None)]), \
                     mock.patch.object(sys, "argv", ["grafana_panel_audit.py", "--verify-all"]), \
                     contextlib.redirect_stdout(buf):
                    with self.assertRaises(SystemExit) as cm:
                        gpa.main()
                    self.assertEqual(cm.exception.code, 1)

    def test_live_dashboards_parse_and_expand_macros_cleanly(self):
        dash_dir = Path(__file__).resolve().parent / "grafana" / "dashboards"
        paths = sorted(dash_dir.glob("*.json"))
        self.assertGreater(len(paths), 0, "no dashboards found under tools/grafana/dashboards")
        with mock.patch.object(gpa, "query_prometheus", return_value=("NON_ZERO", [1.0])):
            for path in paths:
                with self.subTest(dashboard=path.name):
                    uid, title, results = gpa.audit_dashboard(path)
                    self.assertTrue(uid, f"{path.name} missing uid")
                    self.assertTrue(title, f"{path.name} missing title")
                    for r in results:
                        self.assertNotIn("$__rate_interval", r["clean_expr"])
                        self.assertNotIn("$__range", r["clean_expr"])
                        self.assertNotIn("$__interval", r["clean_expr"])


if __name__ == "__main__":
    unittest.main()
