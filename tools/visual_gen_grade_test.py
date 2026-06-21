#!/usr/bin/env python3
"""Tests for visual_gen_grade.py -- the static visual-generation quality grader.

The grader's own green suite is the SuiteGreen witness the RSI loop folds, so these
tests are load-bearing: they pin (1) each sub-check fires on the right hazard and
NOT on a clean figure, (2) the diagram-type dispatch (a bar chart is not graded by
flowchart rules), and (3) determinism (same input -> byte-identical JSON).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_gen_grade as g  # noqa: E402


# --------------------------------------------------------------------------------
# fixtures: minimal but realistic mermaid sources
# --------------------------------------------------------------------------------

INIT = "%%{init: {'theme':'base'}}%%"

CLEAN_FLOW = f"""{INIT}
flowchart TD
  a(["agent proposes a call"]):::untrusted
  b["adjudicate (in-process), keep/deny"]:::kernel
  a --> b
  classDef untrusted fill:#FFE9D6,stroke:#E8833A;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5;
"""

# A subgraph terminator `end` is legal; a node id `end[...]` is not.
FLOW_WITH_SUBGRAPH = f"""{INIT}
flowchart TD
  subgraph BIN["one binary"]
    x["inner"]:::kernel
  end
  x --> y["outer"]:::kernel
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5;
"""

UNQUOTED_RISKY = f"""{INIT}
flowchart TD
  a[cost is 500-2000 tokens, a wasted turn]:::note
  classDef note fill:#fff;
"""

RESERVED_END = f"""{INIT}
flowchart TD
  start["go"] --> end["stop"]
  classDef x fill:#fff;
  class start,end x;
"""

UNCLASSED = f"""{INIT}
flowchart TD
  a["styled"]:::kernel
  b["unstyled and unclassed"]
  a --> b
  classDef kernel fill:#DCE9FB;
"""

XYCHART = f"""{INIT}
xychart-beta
  title "break-even runs vs product N*miss*P"
  x-axis "log10( N * miss * P )" [3, 4, 5, 6, 7]
  y-axis "log10( runs )" -1 --> 6
  line [5.78, 4.78, 3.78, 2.78, 1.78]
"""

# SVG fixtures -- the two documented render hazards and their fixed form.
SVG_FOREIGNOBJECT = (
    '<svg id="s" width="100%" viewBox="0 0 100 50">'
    "<foreignObject><div>label</div></foreignObject></svg>"
)
SVG_NO_HEIGHT = '<svg id="s" width="100%" viewBox="0 0 100 50"><text>hi</text></svg>'
SVG_GOOD = (
    '<svg id="s" width="100" height="50" viewBox="0 0 100 50">'
    "<text><tspan>real label</tspan></text></svg>"
)


def write_figure(d: Path, base: str, mmd: str, svg: str | None = None,
                 meta: list | None = None) -> None:
    (d / f"{base}.mmd").write_text(mmd, encoding="utf-8")
    if svg is not None:
        (d / f"{base}.svg").write_text(svg, encoding="utf-8")
    if meta is not None:
        (d / "_meta.json").write_text(json.dumps(meta), encoding="utf-8")


# --------------------------------------------------------------------------------
# diagram-type dispatch
# --------------------------------------------------------------------------------

def test_diagram_kind_dispatch():
    assert g.diagram_kind(CLEAN_FLOW) == "flow"
    assert g.diagram_kind(XYCHART) == "chart"
    assert g.diagram_kind(f"{INIT}\nsequenceDiagram\n  a->>b: hi") == "flow"


# --------------------------------------------------------------------------------
# syntax check
# --------------------------------------------------------------------------------

def test_syntax_clean_passes():
    ok, why = g.check_syntax(CLEAN_FLOW)
    assert ok, why


def test_syntax_missing_init_block_fails():
    ok, why = g.check_syntax("flowchart TD\n  a-->b")
    assert not ok
    assert any("init" in w for w in why)


def test_syntax_unquoted_risky_label_fails():
    ok, why = g.check_syntax(UNQUOTED_RISKY)
    assert not ok
    assert any("not quoted" in w for w in why)


def test_syntax_quoted_risky_label_passes():
    # The same punctuation, but wrapped in quotes, is safe.
    quoted = UNQUOTED_RISKY.replace(
        "[cost is 500-2000 tokens, a wasted turn]",
        '["cost is 500-2000 tokens, a wasted turn"]',
    )
    ok, why = g.check_syntax(quoted)
    assert ok, why


def test_syntax_subgraph_end_is_legal():
    # `end` terminating a subgraph must NOT be flagged as a reserved-id node.
    ok, why = g.check_syntax(FLOW_WITH_SUBGRAPH)
    assert ok, why


def test_syntax_reserved_end_node_fails():
    ok, why = g.check_syntax(RESERVED_END)
    assert not ok
    assert any("reserved id `end`" in w for w in why)


def test_syntax_unbalanced_brackets_fails():
    ok, why = g.check_syntax(f"{INIT}\nflowchart TD\n  a[\"oops\"")
    assert not ok
    assert any("unbalanced" in w for w in why)


def test_chart_not_graded_by_flowchart_rules():
    # xychart `x-axis "..." [3,4,5]` and `line [..]` are legal -- the unquoted-label
    # and reserved-end rules must NOT apply.
    ok, why = g.check_syntax(XYCHART)
    assert ok, why


# --------------------------------------------------------------------------------
# class coverage
# --------------------------------------------------------------------------------

def test_class_coverage_clean_passes():
    ok, why = g.check_class_coverage(CLEAN_FLOW)
    assert ok, why


def test_class_coverage_inline_triple_colon_counts():
    # `id[...]:::class` is a valid assignment; the node must count as classed.
    src = f"{INIT}\nflowchart TD\n  a[\"x\"]:::kernel\n  classDef kernel fill:#fff;\n"
    ok, why = g.check_class_coverage(src)
    assert ok, why


def test_class_coverage_unclassed_node_fails():
    ok, why = g.check_class_coverage(UNCLASSED)
    assert not ok
    assert any("unclassed" in w for w in why)


def test_class_coverage_subgraph_id_not_a_node():
    # BIN is a subgraph container, not a node -- it must not demand a class.
    ok, why = g.check_class_coverage(FLOW_WITH_SUBGRAPH)
    assert ok, why


def test_class_coverage_chart_not_applicable():
    ok, why = g.check_class_coverage(XYCHART)
    assert ok, why


# --------------------------------------------------------------------------------
# render checks
# --------------------------------------------------------------------------------

def test_text_renderable_foreignobject_only_fails():
    ok, why = g.check_text_renderable(SVG_FOREIGNOBJECT)
    assert not ok
    assert any("foreignObject" in w for w in why)


def test_text_renderable_real_text_passes():
    ok, _ = g.check_text_renderable(SVG_GOOD)
    assert ok


def test_intrinsic_size_no_height_fails():
    ok, why = g.check_intrinsic_size(SVG_NO_HEIGHT)
    assert not ok
    assert any("height" in w for w in why)


def test_intrinsic_size_with_height_passes():
    ok, _ = g.check_intrinsic_size(SVG_GOOD)
    assert ok


# --------------------------------------------------------------------------------
# killer number + meta
# --------------------------------------------------------------------------------

def test_killer_number_present_passes():
    meta = {"killerNumber": "~1000x"}
    ok, _ = g.check_killer_number(f'{INIT}\nflowchart TD\n  a["~1000x on overhead"]', meta)
    assert ok


def test_killer_number_absent_fails():
    meta = {"killerNumber": "~1000x"}
    ok, why = g.check_killer_number(f'{INIT}\nflowchart TD\n  a["nothing here"]', meta)
    assert not ok


def test_killer_number_not_applicable_passes():
    # No killerNumber declared -> not applicable -> pass.
    ok, _ = g.check_killer_number(CLEAN_FLOW, {"killerNumber": ""})
    assert ok


def test_meta_faithful_false_fails():
    ok, why = g.check_meta_faithful({"faithful": False, "syntaxOk": True})
    assert not ok
    assert any("faithful" in w for w in why)


# --------------------------------------------------------------------------------
# whole-figure + deck grading
# --------------------------------------------------------------------------------

def test_good_figure_scores_high(tmp_path):
    meta = [{"id": "good", "faithful": True, "syntaxOk": True, "killerNumber": "42x"}]
    write_figure(tmp_path, "good",
                 f'{INIT}\nflowchart TD\n  a["42x win"]:::kernel\n  classDef kernel fill:#fff;\n',
                 SVG_GOOD, meta)
    fig = g.grade_figure("good", tmp_path, g._meta_index(tmp_path))
    assert fig["score"] == 1.0, fig


def test_broken_render_scores_low_on_render_checks(tmp_path):
    write_figure(tmp_path, "broken", CLEAN_FLOW, SVG_FOREIGNOBJECT)
    fig = g.grade_figure("broken", tmp_path, g._meta_index(tmp_path))
    assert fig["checks"]["syntax_ok"] is True
    assert fig["checks"]["text_renderable"] is False
    assert fig["checks"]["intrinsic_size"] is False
    assert fig["score"] < 0.8


def test_source_only_figure_grades_without_render_penalty(tmp_path):
    # No .svg -> render checks dropped & weights renormalized; a clean source = 1.0.
    write_figure(tmp_path, "srconly",
                 f'{INIT}\nflowchart TD\n  a["x"]:::k\n  classDef k fill:#fff;\n')
    fig = g.grade_figure("srconly", tmp_path, g._meta_index(tmp_path))
    assert fig["has_svg"] is False
    assert fig["score"] == 1.0, fig


def test_grade_deck_is_deterministic(tmp_path):
    write_figure(tmp_path, "a", CLEAN_FLOW, SVG_GOOD)
    write_figure(tmp_path, "b", CLEAN_FLOW, SVG_FOREIGNOBJECT)
    r1 = g.grade_deck(tmp_path, 0.8)
    r2 = g.grade_deck(tmp_path, 0.8)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_grade_deck_aggregate_and_below_floor(tmp_path):
    write_figure(tmp_path, "hi", CLEAN_FLOW, SVG_GOOD)
    write_figure(tmp_path, "lo", CLEAN_FLOW, SVG_FOREIGNOBJECT)
    r = g.grade_deck(tmp_path, 0.8)
    assert r["aggregate"]["n"] == 2
    assert "lo" in r["aggregate"]["below_floor"]
    assert "hi" not in r["aggregate"]["below_floor"]


def test_main_missing_dir_returns_2(tmp_path, capsys):
    rc = g.main(["--dir", str(tmp_path / "nope"), "--json"])
    assert rc == 2


def test_main_json_runs_on_real_deck():
    # Smoke: the real deck grades without crashing and emits a well-formed report.
    rc = g.main(["--json"])
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
