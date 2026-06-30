#!/usr/bin/env python3
"""Tests for the observability scorecard — the pure KPI core + the fold + the gate.

Pure-stdlib, no pytest required (though `python -m pytest` also runs it). The
self-contained ``main()`` collects every ``test_*`` function, runs each under
try/except, and returns a nonzero exit on any failure — the same shape the
sibling ``*_scorecard_test.py`` files use. Run from the repo ROOT::

    python tools/observability_scorecard_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import observability_scorecard as obs  # noqa: E402


# --- normalization + backing (the source-of-truth join) --------------------

def test_normalize_strips_one_prom_suffix():
    assert obs.normalize_family("fak_gateway_http_request_duration_seconds_bucket") == \
        "fak_gateway_http_request_duration_seconds"
    assert obs.normalize_family("fak_x_sum") == "fak_x"
    assert obs.normalize_family("fak_x_count") == "fak_x"
    # not a prom suffix -> unchanged
    assert obs.normalize_family("fak_gateway_up") == "fak_gateway_up"


def test_is_backed_direct_and_histogram_suffix():
    source = {"fak_gateway_up", "fak_gateway_http_request_duration_seconds"}
    assert obs.is_backed("fak_gateway_up", source)
    # a histogram _bucket reference is backed by its base family
    assert obs.is_backed("fak_gateway_http_request_duration_seconds_bucket", source)
    assert obs.is_backed("fak_gateway_http_request_duration_seconds_count", source)


def test_is_backed_total_forgiveness():
    # a counter cited in prose without its _total suffix is a CORRECT reference
    source = {"fak_kernel_submits_total"}
    assert obs.is_backed("fak_kernel_submits", source)
    assert obs.is_backed("fak_kernel_submits_total", source)


def test_is_backed_phantom_is_not_forgiven():
    source = {"fak_kernel_denies_total", "fak_gateway_operation_duration_seconds"}
    # real drift: the binary emits no such family under any normalization
    assert not obs.is_backed("fak_adjudication_denies_total", source)
    assert not obs.is_backed("fak_verdict_total", source)
    # dropping the gateway_ segment is still a phantom
    assert not obs.is_backed("fak_operation_duration_seconds", source)


# --- source-of-truth extraction --------------------------------------------

def test_extract_family_literals_only_quoted():
    go = (
        'writeHelpType(&b, "fak_gateway_up", "Whether...", "gauge")\n'
        'writeCounter(&b, "fak_kernel_submits_total", "Kernel...", c.Submits)\n'
        '// a comment mentioning fak_not_a_metric should NOT match\n'
        'env := os.Getenv("FAK_TOKEN")  // uppercase, not a metric\n'
    )
    fams = obs.extract_family_literals(go)
    assert "fak_gateway_up" in fams
    assert "fak_kernel_submits_total" in fams
    assert "fak_not_a_metric" not in fams  # unquoted comment word
    assert "FAK_TOKEN" not in fams


def test_extract_family_literals_precise_emitters():
    go = (
        'writeHelpType(&b, "fak_gateway_up", "Whether...", "gauge")\n'
        'writeCounter(&b, "fak_kernel_submits_total", "Kernel...", c.Submits)\n'
        'help("fak_engine_cache_events_total", "Live-engine...", "counter")\n'
        'b.WriteString("fak_engine_cache_events_total " + utoa(s.Events))\n'
        'fmt.Fprintf(&b, "fak_gateway_inflight_requests_by_route{route=\\"%s\\"} %d", r, n)\n'
    )
    fams = obs.extract_family_literals(go)
    assert {"fak_gateway_up", "fak_kernel_submits_total",
            "fak_engine_cache_events_total",
            "fak_gateway_inflight_requests_by_route"} <= fams


def test_extract_family_literals_bare_buffer_arg():
    # The writer helpers in internal/gateway/metrics.go pass the exposition buffer
    # by value (`writeCounter(b, "fak_x", ...)`), not only by pointer (`&b,`). Both
    # forms emit the family; a regex that recognized only `&b,` silently dropped the
    # by-value emitters and flagged every dashboard panel that read them as a phantom
    # (the dashboard_integrity false positive this test pins shut).
    go = (
        'writeCounter(b, "fak_gateway_compaction_shed_tokens_total", "...", n)\n'
        'writeCounter(b, "fak_gateway_upstream_retries_total", "...", n)\n'
        'writeHelpType(b, "fak_gateway_context_pollutions_blocked_total", "...", "counter")\n'
        'writeServingGaugeFamily(b, "fak_serving_x", "...", v)\n'
    )
    fams = obs.extract_family_literals(go)
    assert {"fak_gateway_compaction_shed_tokens_total",
            "fak_gateway_upstream_retries_total",
            "fak_gateway_context_pollutions_blocked_total",
            "fak_serving_x"} <= fams


def test_extract_family_literals_excludes_tool_names():
    # MCP tool names and struct fields share the fak_ prefix but are NOT metrics:
    # they are not declared via a help/write helper, nor written as exposition.
    go = (
        'switch p.Name {\n'
        'case "fak_syscall":\n'
        'case "fak_context_change":\n'
        '    return errors.New("invalid fak_context_change arguments")\n'
        '}\n'
        'row := Result{Name: "fak_native", Score: s}\n'
    )
    fams = obs.extract_family_literals(go)
    assert fams == set()


def test_is_metric_shaped():
    assert obs.is_metric_shaped("fak_gateway_up")
    assert obs.is_metric_shaped("fak_verdict_total")
    assert not obs.is_metric_shaped("fak_gateway")     # one segment (prose fragment)
    assert not obs.is_metric_shaped("fak_gateway_")    # trailing-underscore wildcard stem
    assert not obs.is_metric_shaped("fak_kernel_")     # wildcard stem
    assert not obs.is_metric_shaped("not_a_fak_metric")


def test_extract_family_tokens_from_expr():
    expr = 'sum(rate(fak_gateway_http_requests_total{status=~"5.."}[5m]))'
    toks = obs.extract_family_tokens(expr)
    assert toks == {"fak_gateway_http_requests_total"}


def test_extract_family_tokens_drops_fragments():
    # a prose mention of the wildcard family and a bare prefix are NOT references
    text = "the `fak_gateway_*` family, the fak_gateway process, fak_kernel_ counters"
    assert obs.extract_family_tokens(text) == set()


def test_dashboard_expr_text_reads_only_queries():
    dash = (
        '{"panels":[{"title":"fak_gateway latency","targets":'
        '[{"expr":"histogram_quantile(0.95, rate(fak_gateway_http_request_duration_seconds_bucket[5m]))"}]}],'
        '"templating":{"list":[{"query":"label_values(fak_gateway_startup_phase_duration_seconds, phase)",'
        '"regex":"fak_gateway_startup_.*"}]}}'
    )
    expr_text = obs.dashboard_expr_text(dash)
    toks = obs.extract_family_tokens(expr_text)
    # the panel title ("fak_gateway") and the template regex ("fak_gateway_startup_.*")
    # are NOT in an expr, so only the queried histogram family is read
    assert toks == {"fak_gateway_http_request_duration_seconds_bucket"}


def test_alert_expr_text_reads_only_exprs():
    yml = (
        "groups:\n"
        "  - name: fak_gateway_availability\n"   # group name, NOT a metric
        "    rules:\n"
        "      - alert: FakGatewayDown\n"
        "        expr: up{job=\"fak_gateway\"} == 0\n"   # fak_gateway is 1-seg label val
        "      - alert: FakGateway5xx\n"
        "        expr: sum(rate(fak_gateway_http_requests_total{status=~\"5..\"}[5m])) > 0\n"
    )
    toks = obs.extract_family_tokens(obs.alert_expr_text(yml))
    assert toks == {"fak_gateway_http_requests_total"}


# --- log privacy (the security regression guard) ---------------------------

def test_log_event_fields_clean_emitter():
    go = (
        'ev := map[string]any{\n'
        '    "event":       "gateway_http_request",\n'
        '    "method":      r.Method,\n'
        '    "route":       route,\n'
        '    "duration_ms": ms,\n'
        '}\n'
        'if traceID != "" { ev["trace_id"] = traceID }\n'
        's.logf("%s", b)\n'
    )
    keys = set(obs.log_event_field_keys(go))
    assert "event" in keys and "method" in keys and "trace_id" in keys
    assert obs.FORBIDDEN_LOG_FIELDS.isdisjoint(keys)


def test_log_event_fields_detects_leak():
    go = (
        'ev := map[string]any{\n'
        '    "event":     "gateway_operation",\n'
        '    "tool":      tool,\n'
        '    "arguments": args,\n'   # <-- leak
        '}\n'
        's.logf("%s", b)\n'
    )
    keys = set(obs.log_event_field_keys(go))
    assert "arguments" in keys
    assert not obs.FORBIDDEN_LOG_FIELDS.isdisjoint(keys)


def test_kpi_log_privacy_scores():
    clean = obs.kpi_log_privacy([], 2)
    assert clean["score"] == 100 and clean["defects"] == []
    leak = obs.kpi_log_privacy(["gateway_operation.arguments"], 2)
    assert len(leak["defects"]) == 1 and leak["score"] < 100


# --- proof witness (rigor: OPEN ok, PROVEN-without-witness is debt) ---------

_PROVEN_WITNESSED = (
    "## THEOREM 1 — something true\n"
    "**PROOF.** mechanism at file.go:1.\n"
    "**WITNESS.** `go test ./x -run TestY`\n"
    "**VERDICT.** **PROVEN** (2026-06-20).\n"
)
_OPEN_NO_WITNESS = (
    "## THEOREM 2 — not yet closed\n"
    "**PROOF / SCOPE NOTE.** the fold is out of scope.\n"
    "**VERDICT.** **OPEN** (2026-06-20).\n"
)
_PROVEN_NO_WITNESS = (
    "## THEOREM 3 — claimed proven, no command\n"
    "**PROOF.** argued in prose.\n"
    "**VERDICT.** **PROVEN** (2026-06-20).\n"
)
_NO_VERDICT = (
    "## THEOREM 4 — never adjudicated\n"
    "**PROOF.** mechanism.\n"
    "**WITNESS.** `go test ./z`\n"
)


def test_split_theorems_counts_sections():
    text = _PROVEN_WITNESSED + _OPEN_NO_WITNESS
    secs = obs.split_theorems(text)
    assert len(secs) == 2


def test_theorem_defects_open_is_clean():
    # an honestly-OPEN theorem with no witness is NOT debt
    assert obs.theorem_defects("p.md", _PROVEN_WITNESSED + _OPEN_NO_WITNESS) == []


def test_theorem_defects_proven_without_witness_is_debt():
    d = obs.theorem_defects("p.md", _PROVEN_NO_WITNESS)
    assert len(d) == 1 and "without a witness" in d[0]


def test_theorem_defects_no_verdict_is_debt():
    d = obs.theorem_defects("p.md", _NO_VERDICT)
    assert len(d) == 1 and "unadjudicated" in d[0]


def test_kpi_proof_witness_fold():
    by_doc = {"p.md": ["a", "b"], "q.md": ["c"]}
    k = obs.kpi_proof_witness(by_doc, n_theorems=10, n_docs=2)
    assert len(k["defects"]) == 3
    assert k["score"] == obs._clamp(100 - 8 * 3)


# --- reference integrity (phantom detection) -------------------------------

def test_kpi_reference_integrity_flags_phantom_only():
    source = {"fak_gateway_up", "fak_kernel_denies_total"}
    refs = {
        "good.json": {"fak_gateway_up", "fak_kernel_denies"},          # both backed
        "bad.json": {"fak_verdict_total", "fak_quarantine_total"},     # both phantom
    }
    k = obs.kpi_reference_integrity("dashboard_integrity", refs, source, "dashboard")
    assert len(k["defects"]) == 2
    assert all("bad.json" in d for d in k["defects"])
    assert k["score"] == obs._clamp(100 - 12 * 2)


def test_kpi_reference_integrity_clean():
    source = {"fak_gateway_up"}
    refs = {"d.json": {"fak_gateway_up"}}
    k = obs.kpi_reference_integrity("alert_integrity", refs, source, "alert")
    assert k["defects"] == [] and k["score"] == 100


# --- trace correlation -----------------------------------------------------

def test_kpi_trace_correlation_clean():
    k = obs.kpi_trace_correlation(
        header_present=True,
        traced_log_events=["gateway_http_request", "gateway_operation"],
        all_log_events=["gateway_http_request", "gateway_operation"],
        response_header_set=True)
    assert k["defects"] == [] and k["score"] == 100


def test_kpi_trace_correlation_gaps():
    k = obs.kpi_trace_correlation(
        header_present=False,
        traced_log_events=["gateway_http_request"],
        all_log_events=["gateway_http_request", "gateway_operation"],
        response_header_set=True)
    # missing header constant + one untraced event = 2 defects
    assert len(k["defects"]) == 2 and k["score"] < 100


# --- ship integrity (DOS, fail-open) ---------------------------------------

def test_ship_integrity_skipped_fails_open():
    k = obs.kpi_ship_integrity(None)
    assert k["score"] == 100 and k["defects"] == [] and k["soft"]


def test_ship_integrity_dos_error_fails_open():
    k = obs.kpi_ship_integrity({"error": "dos: command not found"})
    assert k["score"] == 100 and k["defects"] == [] and k["soft"]


def test_ship_integrity_counts_residual():
    dos = {"rev_range": "HEAD~5..HEAD", "checkable": 5, "cleared_rate": 0.6,
           "residual": [{"sha": "abc", "subject": "x"}, {"sha": "def", "subject": "y"}]}
    k = obs.kpi_ship_integrity(dos)
    assert len(k["defects"]) == 2 and k["score"] == obs._clamp(100 - 25 * 2)


# --- the fold + the compare gate -------------------------------------------

def _kpi(name, defects, soft=None):
    return {"kpi": name, "group": obs.KPI_GROUP[name], "score": obs._clamp(100 - 12 * len(defects)),
            "detail": "x", "defects": list(defects), "soft": list(soft or [])}


def test_build_payload_zero_debt_is_ok():
    kpis = [_kpi(n, []) for n in obs.KPI_WEIGHTS]
    p = obs.build_payload(workspace="/x", kpis=kpis, emitted_count=58)
    assert p["ok"] is True and p["corpus"]["observability_debt"] == 0
    assert p["corpus"]["grade"] == "A" and p["corpus"]["emitted_families"] == 58


def test_build_payload_counts_debt_and_groups():
    kpis = [_kpi(n, []) for n in obs.KPI_WEIGHTS]
    # inject 3 drift defects + 1 proof defect
    kpis[2] = _kpi("doc_metric_drift", ["a", "b", "c"])
    kpis[6] = _kpi("proof_witness", ["p"])
    p = obs.build_payload(workspace="/x", kpis=kpis, emitted_count=58)
    assert p["ok"] is False and p["corpus"]["observability_debt"] == 4
    assert p["corpus"]["debt_by_group"]["correlation"] == 3
    assert p["corpus"]["debt_by_group"]["verifiability"] == 1
    # heaviest KPI surfaces first in the breakdown
    assert p["corpus"]["breakdown"][0]["kpi"] == "doc_metric_drift"


def test_render_compare_10x_gate():
    base = {"corpus": {"observability_debt": 20, "score": 70, "debt_by_group": {}}}
    # 20 -> 2 is exactly the 10x bar (need <= 20//10 == 2)
    hit = {"corpus": {"observability_debt": 2, "score": 96, "debt_by_group": {}}}
    miss = {"corpus": {"observability_debt": 5, "score": 88, "debt_by_group": {}}}
    assert ">=10x" in obs.render_compare(base, hit)
    assert "not yet 10x" in obs.render_compare(base, miss)


def test_schema_and_weights_sum_to_one():
    assert obs.SCHEMA == "fak-observability-scorecard/1"
    assert abs(sum(obs.KPI_WEIGHTS.values()) - 1.0) < 1e-9
    # every weighted KPI has a group, and vice versa
    assert set(obs.KPI_WEIGHTS) == set(obs.KPI_GROUP)


# --- live smoke (tolerant: only asserts shape, not specific numbers) --------

def test_live_collect_shape():
    root = obs.repo_root()
    if not (root / "go.mod").exists():
        return  # not in the repo tree; skip
    payload = obs.collect(root, run_dos=False)
    assert payload["schema"] == obs.SCHEMA
    if payload["verdict"] == "AUDIT_ERROR":
        return  # no git / not the root; tolerated in a sandbox
    c = payload["corpus"]
    assert isinstance(c["observability_debt"], int)
    assert c["emitted_families"] > 0  # the binary really does emit fak_* families
    assert set(c["debt_by_group"]) == set(obs.GROUPS)
    assert len(payload["kpis"]) == len(obs.KPI_WEIGHTS)


def main() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
