#!/usr/bin/env python3
"""Tests for the README visual-density auditor.

Drives the PURE detectors + grader with fixture strings (no disk needed), then a
tolerant live smoke that `collect` folds the real tracked READMEs.

Run: `python tools/readme_visual_audit_test.py`  (exit 0 = all pass),
or `python -m pytest tools/readme_visual_audit_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_visual_audit as rva  # noqa: E402


# --- mermaid detection -----------------------------------------------------

def test_mermaid_block_counts() -> None:
    txt = "intro\n\n```mermaid\nflowchart LR\n  A --> B\n```\n"
    assert rva.has_mermaid(txt) is True
    assert rva.audit_one(txt)["has_visual"] is True


def test_plain_code_fence_is_not_mermaid() -> None:
    txt = "```bash\necho hi\n```"
    assert rva.has_mermaid(txt) is False


# --- diagram-image vs badge ------------------------------------------------

def test_diagram_image_counts() -> None:
    txt = "![turn-tax curves](visuals/60-hero-turntax-curves.png)"
    assert rva.diagram_images(txt) == ["visuals/60-hero-turntax-curves.png"]
    assert rva.audit_one(txt)["has_visual"] is True


def test_relative_visuals_image_counts() -> None:
    txt = "![gate](../../visuals/46-two-gate-security-model.svg)"
    assert rva.diagram_images(txt), txt


def test_colab_badge_does_not_count() -> None:
    txt = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](x)"
    assert rva.diagram_images(txt) == []
    assert rva.audit_one(txt)["has_visual"] is False


def test_shields_badge_does_not_count() -> None:
    txt = "![build](https://img.shields.io/badge/ci-green.svg)"
    assert rva.diagram_images(txt) == []


# --- ASCII diagram detection -----------------------------------------------

def test_unicode_box_diagram_counts() -> None:
    txt = (
        "```text\n"
        "┌─────────┐     ┌─────────┐\n"
        "│ client  │ ──▶ │ kernel  │\n"
        "└─────────┘     └─────────┘\n"
        "```\n"
    )
    assert rva.ascii_diagram_blocks(txt) == 1
    assert rva.audit_one(txt)["has_visual"] is True


def test_bar_chart_counts_as_visual() -> None:
    # A scorecard's at-a-glance bar chart (block glyphs) is a visual too.
    txt = (
        "```text\n"
        "  durable  ████████████████············ 10\n"
        "  usable   █████████████··············· 8\n"
        "  coverage [████████████████████████████] 100%\n"
        "```\n"
    )
    assert rva.ascii_diagram_blocks(txt) == 1
    assert rva.audit_one(txt)["has_visual"] is True


def test_ascii_arrow_diagram_counts() -> None:
    txt = (
        "```\n"
        "client --> gate\n"
        "gate   --> upstream\n"
        "```\n"
    )
    assert rva.ascii_diagram_blocks(txt) == 1


def test_single_arrow_line_is_not_a_diagram() -> None:
    # One arrow line in a shell snippet must NOT register as a diagram (>=2 floor).
    txt = "```bash\nfoo --> bar.txt   # just a redirect-ish note\necho done\n```"
    assert rva.ascii_diagram_blocks(txt) == 0


def test_markdown_table_is_not_a_diagram() -> None:
    # Tables use | and - but no box/arrow glyphs, and are not even fenced.
    txt = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert rva.audit_one(txt)["has_visual"] is False


# --- grader / payload ------------------------------------------------------

def test_payload_ok_when_all_visual() -> None:
    checks = [{"check": "README.md", "status": "OK", "detail": "has image×4"}]
    p = rva.build_payload(workspace=".", checks=checks)
    assert p["ok"] is True and p["verdict"] == "OK", p


def test_payload_fails_when_any_text_only() -> None:
    checks = [
        {"check": "README.md", "status": "OK", "detail": "has image×4"},
        {"check": "docs/x/README.md", "status": "FAIL", "detail": "text-only"},
    ]
    p = rva.build_payload(workspace=".", checks=checks)
    assert p["ok"] is False and p["verdict"] == "ACTION", p
    assert "docs/x/README.md" in p["reason"], p


def test_payload_error_path() -> None:
    p = rva.build_payload(workspace=".", checks=[], error="no tracked README.md found")
    assert p["ok"] is False and p["finding"] == "tooling_error", p


# --- live smoke ------------------------------------------------------------

def test_live_collect_real_tree() -> None:
    root = rva.repo_root()
    if not (root / "README.md").exists():
        return  # tolerant: not in the repo tree
    p = rva.collect(root)
    assert p["schema"] == rva.SCHEMA
    assert isinstance(p["checks"], list) and p["checks"]
    # The front-door README.md must itself carry a visual (it leads with charts).
    front = [c for c in p["checks"] if c["check"] == "README.md"]
    assert front and front[0]["status"] == "OK", front


# --- self-contained runner (mirrors readme_freshness_audit_test.py) --------

def main() -> int:
    failures: list[str] = []
    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
