#!/usr/bin/env python3
"""Tests for the demo-quality scorecard.

Drives the PURE per-axis checks + grader with fixture `Demo`s (no disk needed),
covers the calibration that keeps it honest (a gold-standard demo grades A; a
README-only setup guide grades low; cleanup detection; dead-runner detection;
captured-output via EXAMPLE-OUTPUT.md OR an in-README run), then a tolerant live
smoke that `collect` folds the real committed demos.

Run: `python tools/demo_quality_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/demo_quality_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_quality_scorecard as dq  # noqa: E402


# A gold-standard demo: run.sh + a python entry with __main__ + exit code,
# README with a run command, a scope statement, prereqs, an output section, and
# a captured EXAMPLE-OUTPUT.md. Used as the "should be ~A" anchor.
GOLD_README = """# My demo — live proof

What it is: a one-line thing that proves the gate.

## Prerequisites
- Go, Python 3 (stdlib only).

## Run it
```bash
./run.sh
python3 demo.py --dry-run
```

## What you see
```
✓ check one  ALLOW
summary: kernel test passed
```

## Scope
This demo does not claim the model is good; see [CLAIMS](../../CLAIMS.md).
Exit code: 0 on pass, 1 otherwise (CI-usable).
"""

GOLD_RUNSH = """#!/bin/bash
set -e
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
go build -o "$TMP/fak" ./cmd/fak
python3 demo.py "$@"
"""

GOLD_DEMOPY = '''import tempfile, atexit, shutil, sys
sandbox = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(sandbox, ignore_errors=True))
if __name__ == "__main__":
    sys.exit(0)
'''


# A substantive captured run: >= MIN_OUTPUT_LINES non-blank lines with run-tells.
GOLD_OUTPUT = """My demo — captured run
  ✓ check one  ALLOW → ran
  ✓ check two  DENY (kernel)
summary: kernel test passed
"""


def gold_demo() -> dq.Demo:
    return dq.Demo(
        "examples/gold",
        readme=GOLD_README,
        scripts={"run.sh": GOLD_RUNSH, "demo.py": GOLD_DEMOPY},
        files={"README.md", "run.sh", "demo.py", "EXAMPLE-OUTPUT.md"},
        example_output=GOLD_OUTPUT,
    )


# --- runnable --------------------------------------------------------------

def test_runnable_gold_is_100() -> None:
    a = dq.axis_runnable(gold_demo())
    assert a["score"] == 100 and not a["defects"], a


def test_runnable_no_entry_is_defect() -> None:
    d = dq.Demo("examples/x", readme="# X\n\njust prose, no command, no script.\n")
    a = dq.axis_runnable(d)
    assert any("no runnable entry" in x for x in a["defects"]), a
    assert a["score"] < 60, a


def test_runnable_go_run_command_counts() -> None:
    d = dq.Demo("cmd/x", readme="# X\n\nRun it:\n```bash\ngo run ./cmd/x\n```\n")
    a = dq.axis_runnable(d)
    assert not a["defects"], a  # a paste-able `go run` is a runnable entry


def test_runnable_dead_runner_reference() -> None:
    # README names run.sh but the file is not present -> dead runner (HARD).
    d = dq.Demo("examples/x", readme="# X\n\nRun `./run.sh` to start.\n", files={"README.md"})
    a = dq.axis_runnable(d)
    assert any("dead runner" in x for x in a["defects"]), a


def test_runnable_script_without_readme_command_is_soft() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nA demo with a script but no shown command.\n",
                scripts={"run.sh": "#!/bin/bash\necho hi\n"}, files={"README.md", "run.sh"})
    a = dq.axis_runnable(d)
    assert not a["defects"] and a["soft"], a  # has entry; missing-command is soft only


# --- reproducible ----------------------------------------------------------

def test_reproducible_example_output_file_is_captured() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nExit code 0 on pass.\n",
                example_output="run start\n✓ ALLOW → ran\nsummary: ok\n")
    a = dq.axis_reproducible(d)
    assert not a["defects"], a


def test_reproducible_in_readme_run_counts() -> None:
    # No EXAMPLE-OUTPUT.md, but README shows a substantive captured run (>=3 lines,
    # multiple distinct run-tells).
    d = dq.Demo("examples/x",
                readme="# X\n\n## What you see\n```\n✓ PASS\n→ ran\nsummary: ok\n```\nExit code: 0.\n")
    a = dq.axis_reproducible(d)
    assert not a["defects"], a


def test_reproducible_no_output_is_defect() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nSome setup steps, no sample run shown.\n")
    a = dq.axis_reproducible(d)
    assert any("no captured example output" in x for x in a["defects"]), a


def test_reproducible_missing_exit_statement_is_soft() -> None:
    d = dq.Demo("examples/x", readme="# X\n",
                example_output="line one\n✓ ALLOW\nsummary: done\n")
    a = dq.axis_reproducible(d)
    assert not a["defects"] and any("exit-code" in s for s in a["soft"]), a


# --- content floor / anti-gaming (review-driven) ---------------------------

def test_reproducible_one_byte_output_is_not_captured() -> None:
    # A 1-byte EXAMPLE-OUTPUT.md must NOT clear the captured-output HARD defect.
    d = dq.Demo("examples/x", readme="# X\n", example_output="x\n")
    a = dq.axis_reproducible(d)
    assert any("no captured example output" in x for x in a["defects"]), a


def test_reproducible_lone_glyph_fence_is_not_captured() -> None:
    # A fenced block holding a single stray glyph is not a captured run.
    d = dq.Demo("examples/x", readme="# X\n\n```\n→\n```\n")
    assert dq.has_captured_output(d) is False


def test_captured_output_config_fence_before_output_heading_does_not_count() -> None:
    # A config fence that appears BEFORE a prose-only 'what you see' heading must
    # not be credited as the captured run (the heading must precede the fence).
    readme = ("# X\n\n## Config\n```json\n{\n  \"a\": 1,\n  \"b\": 2\n}\n```\n"
              "## What you see\nrun it and watch the verdicts.\n")
    assert dq.has_captured_output(dq.Demo("examples/x", readme=readme)) is False


def test_captured_output_fence_after_output_heading_counts() -> None:
    readme = ("# X\n\n## What you see\n```\nline a\nline b\nline c\n```\n")
    assert dq.has_captured_output(dq.Demo("examples/x", readme=readme)) is True


def test_runnable_make_sure_prose_is_not_a_command() -> None:
    # 'Make sure ...' is English prose, not a paste-able `make <target>`.
    d = dq.Demo("examples/x", readme="# X\n\nMake sure Go is installed, then explore.\n")
    assert d.has_run_command is False
    a = dq.axis_runnable(d)
    assert any("no runnable entry" in x for x in a["defects"]), a


def test_runnable_real_make_target_is_a_command() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nRun:\n```\nmake demo\n```\n")
    assert d.has_run_command is True


def test_runnable_server_launch_is_not_a_demo_run() -> None:
    # `go run ./cmd/fak serve --stdio` is a long-running server, not a demo run.
    d = dq.Demo("examples/x", readme="# X\n\n```\ngo run ./cmd/fak serve --stdio\n```\n")
    assert d.has_run_command is False


def test_runnable_dead_runner_non_canonical_name() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nRun `./start.sh` to begin.\n",
                scripts={"helper.sh": "echo hi"}, files={"README.md", "helper.sh"})
    a = dq.axis_runnable(d)
    assert any("dead runner" in x and "start.sh" in x for x in a["defects"]), a


def test_honest_scope_heading_only_is_not_enough() -> None:
    # A bare `## Scope` heading with no boundary statement does NOT clear the defect.
    d = dq.Demo("examples/x", readme="# X\n\n## Scope and goals\nWe aim to do everything.\n")
    a = dq.axis_honest_scope(d)
    assert any("no scope" in x for x in a["defects"]), a


def test_honest_scope_bare_limitation_word_is_not_enough() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nThere is no rate limitation on requests.\n")
    a = dq.axis_honest_scope(d)
    assert any("no scope" in x for x in a["defects"]), a


def test_honest_scope_matches_emphasis_wrapped_negation() -> None:
    # `does **not** demonstrate` (markdown emphasis) is a real boundary statement.
    d = dq.Demo("examples/x", readme="# X\n\nIt does **not** demonstrate containment. [c](../C.md)\n")
    a = dq.axis_honest_scope(d)
    assert not a["defects"], a


def test_cleanup_word_in_comment_does_not_discharge() -> None:
    # A comment mentioning 'cleanup' is not teardown; the create-without-cleanup
    # defect must still fire.
    d = dq.Demo("examples/x", readme="# X\n\nRequires Go.\n",
                scripts={"run.sh": "#!/bin/bash\nmkdir /tmp/leak\n# TODO: add cleanup later\n"},
                files={"README.md", "run.sh"})
    a = dq.axis_self_contained(d)
    assert any("no cleanup" in x for x in a["defects"]), a


def test_go_func_main_is_an_entry() -> None:
    d = dq.Demo("cmd/x", readme="# X\n\nRun:\n```\ngo build ./cmd/x\n```\n",
                scripts={"main.go": "package main\nfunc main() {\n  println(\"hi\")\n}\n"},
                files={"README.md", "main.go"})
    assert d.has_entry_script is True


def test_go_test_file_is_not_an_entry() -> None:
    d = dq.Demo("cmd/x", readme="# X\n",
                scripts={"main_test.go": "package main\nfunc main() {}\n"},
                files={"README.md", "main_test.go"})
    assert d.has_entry_script is False


def test_documented_h1_inside_fence_is_not_counted() -> None:
    # A shell '# comment' inside a code fence must not count as a Markdown H1.
    readme = ("# Real Title\n\n```bash\n# Use a specific model\nmodel run\n```\n")
    a = dq.axis_documented(dq.Demo("examples/x", readme=readme))
    assert not any("H1 titles" in s for s in a["soft"]), a


def test_non_utf8_read_does_not_crash(tmp_path: Path) -> None:
    # A binary/non-UTF-8 README must decode tolerantly, not crash the run.
    d = tmp_path / "examples" / "weird"
    d.mkdir(parents=True)
    (d / "README.md").write_bytes(b"\xff\xfe# binary\x00 title\n")
    demo = dq.load_demo(tmp_path, "examples/weird")
    s = dq.score_demo(demo)  # must not raise
    assert isinstance(s["score"], float), s


def test_empty_corpus_is_audit_error_not_ok(tmp_path: Path) -> None:
    # Discovering zero demos is a loud error, not a clean OK/exit-0 pass.
    p = dq.collect(tmp_path)
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR" and p["finding"] == "no_demos", p


def test_median_is_true_median_for_even_count() -> None:
    demos = [
        {"path": "a", "score": 50.0, "grade": "F", "n_defects": 1, "defects": ["x"], "soft": [], "axes": {}},
        {"path": "b", "score": 100.0, "grade": "A", "n_defects": 0, "defects": [], "soft": [], "axes": {}},
    ]
    p = dq.build_payload(workspace=".", demos=demos)
    assert p["corpus"]["median_score"] == 75.0, p


# --- honest_scope ----------------------------------------------------------

def test_honest_scope_present_no_defect() -> None:
    d = dq.Demo("examples/x",
                readme="# X\n\n## Scope\nThis demo does not claim detection works. [c](../C.md)\n")
    a = dq.axis_honest_scope(d)
    assert not a["defects"], a


def test_honest_scope_missing_is_defect() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nThe best demo. It proves everything. [c](../C.md)\n")
    a = dq.axis_honest_scope(d)
    assert any("no scope" in x for x in a["defects"]), a


def test_honest_scope_no_deeper_link_is_soft() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nThis demo does not claim much.\n")
    a = dq.axis_honest_scope(d)
    assert not a["defects"] and any("deeper doc" in s for s in a["soft"]), a


# --- self_contained --------------------------------------------------------

def test_self_contained_cleanup_present_ok() -> None:
    a = dq.axis_self_contained(gold_demo())
    assert not a["defects"], a  # mktemp + trap/atexit are discharged


def test_self_contained_create_without_cleanup_is_defect() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nRequires Go.\n",
                scripts={"run.sh": "#!/bin/bash\nmkdir /tmp/leftover\necho data > /tmp/x.txt\n"},
                files={"README.md", "run.sh"})
    a = dq.axis_self_contained(d)
    assert any("no cleanup" in x for x in a["defects"]), a


def test_self_contained_no_create_no_defect() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nRequires Go.\n",
                scripts={"run.sh": "#!/bin/bash\nexec go run ./cmd/x\n"},
                files={"README.md", "run.sh"})
    a = dq.axis_self_contained(d)
    assert not a["defects"], a  # `go run` creates no durable artifact


def test_self_contained_test_file_string_fixture_not_a_defect() -> None:
    # A `*_test.py` whose only create-tokens are commands-as-data in a string
    # fixture (a pinned `mkdir … && printf > …` chain, never executed) is a
    # maintainer unit test, not a demo runner — so it is NOT a leaves-a-mess defect.
    d = dq.Demo("examples/x", readme="# X\n\nRequires Python.\n",
                scripts={"run.sh": "#!/bin/bash\nexec python demo.py\n",
                         "demo_test.py": 'CHAIN = "mkdir -p {s} && printf hi > {s}/a.txt"\n'},
                files={"README.md", "run.sh", "demo.py", "demo_test.py"})
    a = dq.axis_self_contained(d)
    assert not a["defects"], a


def test_self_contained_missing_prereqs_is_soft() -> None:
    d = dq.Demo("examples/x", readme="# X\n\nRun:\n```\ngo run ./cmd/x\n```\n")
    a = dq.axis_self_contained(d)
    assert not a["defects"] and any("prerequisite" in s for s in a["soft"]), a


# --- documented ------------------------------------------------------------

def test_documented_gold_clean() -> None:
    a = dq.axis_documented(gold_demo())
    assert not a["defects"], a


def test_documented_missing_h1_is_defect() -> None:
    d = dq.Demo("examples/x", readme="no title here\n\n## a section\n")
    a = dq.axis_documented(d)
    assert any("no H1" in x for x in a["defects"]), a


# --- per-demo fold + grader ------------------------------------------------

def test_score_demo_gold_is_A() -> None:
    s = dq.score_demo(gold_demo())
    assert s["n_defects"] == 0 and s["grade"] == "A", s


def test_score_demo_readme_only_guide_grades_low() -> None:
    # A README-only setup guide: no runner, no captured output, no scope -> debt.
    d = dq.Demo("examples/guide", readme="# Guide\n\nCopy the config and go.\n")
    s = dq.score_demo(d)
    assert s["n_defects"] >= 3 and s["grade"] in ("D", "F"), s


def test_grade_letter_bands() -> None:
    assert dq.grade_letter(95) == "A"
    assert dq.grade_letter(85) == "B"
    assert dq.grade_letter(72) == "C"
    assert dq.grade_letter(61) == "D"
    assert dq.grade_letter(40) == "F"


def test_missing_demo_is_worst() -> None:
    d = dq.missing_demo_entry("examples/gone")
    assert d["score"] == 0.0 and d["grade"] == "F" and d["n_defects"] == 1, d


def test_payload_clean_is_ok() -> None:
    demos = [{"path": "examples/a", "score": 100.0, "grade": "A", "n_defects": 0,
              "defects": [], "soft": [], "axes": {}}]
    p = dq.build_payload(workspace=".", demos=demos)
    assert p["ok"] is True and p["verdict"] == "OK" and p["corpus"]["demo_debt"] == 0, p


def test_payload_counts_demo_debt() -> None:
    demos = [
        {"path": "examples/a", "score": 100.0, "grade": "A", "n_defects": 0,
         "defects": [], "soft": [], "axes": {}},
        {"path": "examples/b", "score": 60.0, "grade": "D", "n_defects": 2,
         "defects": ["x", "y"], "soft": [], "axes": {}},
    ]
    p = dq.build_payload(workspace=".", demos=demos)
    assert p["ok"] is False and p["corpus"]["demo_debt"] == 2, p
    assert p["corpus"]["worst"][0]["path"] == "examples/b", p


# --- captured-output helper ------------------------------------------------

def test_fenced_blocks_extraction() -> None:
    blocks = dq._fenced_blocks("intro\n```\na\nb\n```\nmid\n```py\nc\n```\n")
    assert blocks == ["a\nb", "c"], blocks


def test_has_captured_output_config_fence_does_not_count() -> None:
    # A fenced *config* block (no run-tells, no output heading) is not a captured run.
    d = dq.Demo("examples/x", readme="# X\n\nConfig:\n```json\n{\"a\": 1}\n```\n")
    assert dq.has_captured_output(d) is False


# --- live smoke ------------------------------------------------------------

def test_live_discover_finds_known_demos() -> None:
    root = dq.repo_root()
    if not (root / "examples").exists():
        return  # tolerant: not in the repo tree
    rels = dq.discover_demos(root)
    assert "examples/adjudication-demo" in rels, rels
    assert "examples/wire-proof" in rels, rels
    assert "cmd/simpledemo" in rels, rels
    assert "cmd/guarddemo" in rels, rels
    assert "cmd/demorace" in rels, rels


def test_discover_auto_includes_cmd_demo_dirs_without_manual_list(tmp_path: Path) -> None:
    ex = tmp_path / "examples" / "visible"
    ex.mkdir(parents=True)
    (ex / "README.md").write_text("# visible\n", encoding="utf-8")

    demo = tmp_path / "cmd" / "newdemo"
    demo.mkdir(parents=True)
    (demo / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")

    bench = tmp_path / "cmd" / "notbench"
    bench.mkdir(parents=True)
    (bench / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")

    rels = dq.discover_demos(tmp_path)
    assert rels == ["examples/visible", "cmd/newdemo"], rels


def test_live_collect_payload_shape() -> None:
    root = dq.repo_root()
    if not (root / "examples").exists():
        return
    p = dq.collect(root)
    assert p["schema"] == dq.SCHEMA
    assert p["corpus"]["n_demos"] >= 3
    assert isinstance(p["demos"], list) and p["demos"]


def test_live_gold_demos_grade_well() -> None:
    # The flagship adjudication-demo should be a strong, debt-free demo on disk.
    root = dq.repo_root()
    if not (root / "examples" / "adjudication-demo" / "README.md").exists():
        return
    s = dq.score_demo(dq.load_demo(root, "examples/adjudication-demo"))
    assert s["grade"] in ("A", "B") and s["n_defects"] == 0, s


def test_check_markdown_doc_accepts_fresh_generated_snapshot(tmp_path: Path) -> None:
    payload = {
        "corpus": {
            "n_demos": 0,
            "demo_debt": 0,
            "mean_score": 0,
            "median_score": 0,
            "min_score": 0,
            "max_score": 0,
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
        },
        "demos": [],
    }
    doc = tmp_path / dq.SCORECARD_DOC
    doc.parent.mkdir(parents=True)
    doc.write_text(dq.render_markdown(payload, stamp="2099-01-02") + "\n", encoding="utf-8")
    check = dq.check_markdown_doc(tmp_path, payload)
    assert check["ok"], check
    assert check["stamp"] == "2099-01-02", check


def test_check_markdown_doc_rejects_stale_snapshot(tmp_path: Path) -> None:
    payload = {
        "corpus": {
            "n_demos": 0,
            "demo_debt": 0,
            "mean_score": 0,
            "median_score": 0,
            "min_score": 0,
            "max_score": 0,
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
        },
        "demos": [],
    }
    doc = tmp_path / dq.SCORECARD_DOC
    doc.parent.mkdir(parents=True)
    stale = dq.render_markdown(payload, stamp="2099-01-02").replace("Demos scored | 0", "Demos scored | 99")
    doc.write_text(stale + "\n", encoding="utf-8")
    check = dq.check_markdown_doc(tmp_path, payload)
    assert not check["ok"], check
    assert check["diff"], check


# --- self-contained runner (mirrors docs_scorecard_test.py) ----------------

def main() -> int:
    import inspect
    import tempfile
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
