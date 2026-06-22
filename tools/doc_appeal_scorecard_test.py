#!/usr/bin/env python3
"""Tests for the doc-appeal scorecard — the prose-that-lands measuring stick.

Drives the PURE axis functions + grader with fixture strings (no disk needed),
then a tolerant live smoke that `collect` folds the real committed README. The
focus is the LLM-tell detectors on the `voice` axis (clichés, scaffolding
phrases, the em-dash flood, the forward AND reversed contrast frame, the
bold-emphasis flood), because those are the part most likely to drift or
over-fire on a real edit.

Run: `python tools/doc_appeal_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/doc_appeal_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_appeal_scorecard as das  # noqa: E402


def _voice(text: str) -> dict:
    return das.axis_voice(das.parse(text))


def _defects(axis: dict) -> str:
    return " || ".join(axis["defects"])


def _soft(axis: dict) -> str:
    return " || ".join(axis["soft"])


# --- parsing ---------------------------------------------------------------

def test_parse_classifies_heading_and_skips_code() -> None:
    doc = das.parse("# Title\n\nReal prose here that the axes see.\n\n```\nnot prose() // code\n```\n")
    assert any(lvl == 1 and "Title" in t for lvl, t, _ in doc.headings), doc.headings
    # The code-fence line must not leak into the prose corpus.
    assert "not prose()" not in doc.prose_text, doc.prose_text


def test_split_breaks_at_arrow_pointer() -> None:
    # A "→ link" pointer must not glue onto the prior sentence as one run-on.
    parts = das._split_sentences("It is real today. → One binary is the whole surface")
    assert len(parts) == 2, parts
    assert all(das._wordcount(p) < 12 for p in parts), parts


def test_split_breaks_at_middot_separator() -> None:
    parts = das._split_sentences("the demos · run your own · the showcase · the docs site")
    assert len(parts) == 4, parts


def test_horizontal_rule_does_not_merge_sections() -> None:
    # The two paragraphs sit either side of a --- rule; neither colon nor rule may
    # fuse "first ends with a colon:" onto "Second para starts.".
    doc = das.parse("# T\n\nFirst para ends with a colon:\n\n---\n\nSecond para starts here.\n")
    assert not any("First" in s and "Second" in s for s in doc.sentences), doc.sentences
    assert not any("---" in (raw) and kind == "prose" for _, raw, kind in doc.content), \
        "--- leaked into prose content"


def test_per_block_split_no_cross_block_merge() -> None:
    # Block A ends in a colon (no period); Block B opens a new paragraph. They must
    # remain separate sentences, not one phantom overlong run-on.
    doc = das.parse("Author a policy and check the call:\n\nThe gate denies by structure.\n")
    assert not any("Author" in s and "structure" in s for s in doc.sentences), doc.sentences


def test_split_at_lowercase_fak_sentence_start() -> None:
    # "(… differs). `fak` can do this" must split — fak is the product name and
    # opens many sentences, but is lower-case so the capital-only rule misses it.
    parts = das._split_sentences("Not one number differs. fak can do this because it owns it.")
    assert len(parts) == 2, parts


def test_script_block_jsonld_is_not_scored_as_prose() -> None:
    # A generated JSON-LD <script> block is machine content, not reader prose; its
    # answer text (em-dashes, run-ons and all) must not inflate the score.
    doc = das.parse(
        "# T\n\n<script type=\"application/ld+json\">\n"
        '{"text": "one — two — three — four, a, b, c, d, e, f run-on machine text"}\n'
        "</script>\n\nReal short prose here.\n")
    assert "machine text" not in doc.prose_text, doc.prose_text
    v = das.axis_voice(doc)
    assert not any("em-dash" in d for d in v["defects"]), v["defects"]


def test_front_matter_is_not_scored_as_prose() -> None:
    # A long YAML `description:` is page metadata, not reader prose; it must not
    # appear in the prose corpus nor trip a clarity run-on/overlong defect.
    fm = ("---\n"
          "title: A page\n"
          "description: " + ", ".join(["clause"] * 12) + " and a final long trailing thought\n"
          "---\n\n"
          "# Heading\n\nShort real prose.\n")
    doc = das.parse(fm)
    assert "clause" not in doc.prose_text, doc.prose_text
    c = das.axis_clarity(doc)
    assert c["defects"] == [], c["defects"]


def test_multi_item_list_is_not_a_wall() -> None:
    # A long but genuine multi-bullet list is scannable, not a wall of text.
    items = "\n".join(f"- **Item {i}.** " + " ".join(["word"] * 40) for i in range(3))
    s = das.axis_scannability(das.parse("# T\n\n" + items + "\n"))
    assert not any("wall-of-text" in d for d in s["defects"]), s["defects"]


def test_lone_giant_bullet_still_counts_as_wall() -> None:
    # A single massive bullet is a paragraph wearing a dash — still a wall.
    s = das.axis_scannability(das.parse("# T\n\n- " + " ".join(["word"] * 130) + "\n"))
    assert any("wall-of-text" in d for d in s["defects"]), s["defects"]


# --- clarity ---------------------------------------------------------------

def test_clarity_flags_overlong_sentence() -> None:
    long = "This " + "very long ".strip() + " " + " ".join(["word"] * 45) + " ends."
    c = das.axis_clarity(das.parse(long))
    assert any("overlong" in d for d in c["defects"]), c["defects"]


def test_clarity_flags_runon_by_commas() -> None:
    runon = "One, two, three, four, five, six clauses pile up here in a breathless line."
    c = das.axis_clarity(das.parse(runon))
    assert any("run-on" in d for d in c["defects"]), c["defects"]


def test_clarity_clean_short_sentences_no_defects() -> None:
    clean = "The gate denies the call. The model never sees the result. The task still finishes."
    c = das.axis_clarity(das.parse(clean))
    assert c["defects"] == [], c["defects"]


# --- voice: clichés + scaffolding -----------------------------------------

def test_voice_flags_cliche_phrase() -> None:
    v = _voice("We leverage a seamless, robust, best-in-class solution.")
    d = _defects(v)
    assert "leverage" in d and "seamless" in d, d


def test_voice_flags_llm_scaffolding() -> None:
    v = _voice("Here's the thing: at its core, it's important to note that the gate denies.")
    d = _defects(v)
    assert "here's the thing" in d.lower(), d
    assert "at its core" in d.lower(), d
    assert "important to note" in d.lower(), d


def test_voice_scaffolding_tolerates_curly_apostrophe() -> None:
    # A curly-quote "Here’s" must still trip the apostrophe-bearing phrase.
    v = _voice("Here’s the thing about the kernel: it denies by structure.")
    assert "scaffolding" in _defects(v).lower(), v["defects"]


def test_voice_spares_feynman_moves() -> None:
    # The repo voice law ENCOURAGES concrete-example framing — these must not fire.
    v = _voice("Think of it as a scratchpad. Imagine the cache as the model's notes. "
               "Picture the tool call as a syscall through a kernel.")
    d = _defects(v).lower()
    assert "scaffolding" not in d and "cliché" not in d, v["defects"]


# --- voice: em-dash flood --------------------------------------------------

def test_voice_flags_emdash_flood() -> None:
    flood = ("The gate — the lock — the wall — the audit — the proof "
             "— the trail — the metric — all of it matters a great deal here.")
    v = _voice(flood)
    assert any("em-dash" in d for d in v["defects"]), v["defects"]


def test_voice_one_emdash_under_budget_ok() -> None:
    v = _voice("The gate denies the call — by structure, with a named reason that holds.")
    assert not any("em-dash" in d for d in v["defects"]), v["defects"]


# --- voice: the contrast frame (forward + the newly-caught reversed form) ---

def test_voice_flags_forward_contrast_frame_when_over_budget() -> None:
    txt = ("It is not just a server, but a kernel. It is not merely fast, but safe. "
           "It is not only local, but auditable. It is not simply a proxy, but a gate.")
    v = _voice(txt)
    assert any("contrast frame" in d for d in v["defects"]), v["defects"]


def test_voice_flags_reversed_contrast_frame_when_over_budget() -> None:
    # The form the OLD regex missed entirely: "X, not Y".
    txt = ("Own the lock, not the screener. Evict it, not because memory got tight. "
           "It is shown, not hidden. The contrast is operational, not numerical.")
    v = _voice(txt)
    assert any("contrast frame" in d for d in v["defects"]), v["defects"]
    # detail should report a non-zero contrast-frame count
    assert "0 contrast-frame" not in v["detail"], v["detail"]


def test_voice_contrast_frame_under_budget_ok() -> None:
    # One reversed frame is good rhetoric, not a tic — must not be flagged.
    v = _voice("The kernel is the lock, not the screener; that is the whole point.")
    assert not any("contrast frame" in d for d in v["defects"]), v["defects"]


# --- voice: bold-emphasis flood is SOFT, never debt ------------------------

def test_voice_bold_flood_is_soft_not_debt() -> None:
    bold = " ".join(f"**term{i}**" for i in range(9)) + " carry the weight of the sentence."
    v = _voice(bold)
    assert any("bold span" in s for s in v["soft"]), v["soft"]
    assert not any("bold span" in d for d in v["defects"]), v["defects"]


# --- priority / scannability / organization --------------------------------

def test_priority_flags_missing_tldr() -> None:
    p = das.axis_priority(das.parse("# T\n\nfak is a kernel that denies tool calls.\n\nMore prose.\n"))
    assert any("TL;DR" in d for d in p["defects"]), p["defects"]


def test_scannability_flags_wall_of_text() -> None:
    wall = " ".join(["wordy"] * 130) + "."
    s = das.axis_scannability(das.parse(wall))
    assert any("wall-of-text" in d for d in s["defects"]), s["defects"]


def test_organization_flags_multiple_h1() -> None:
    o = das.axis_organization(das.parse("# One\n\ntext\n\n# Two\n\ntext\n"))
    assert any("H1" in d for d in o["defects"]), o["defects"]


def test_organization_flags_level_skip() -> None:
    o = das.axis_organization(das.parse("# One\n\n#### Deep\n\ntext\n"))
    assert any("skipped heading level" in d for d in o["defects"]), o["defects"]


# --- grader / payload ------------------------------------------------------

def test_score_doc_dirty_has_debt() -> None:
    dirty = ("# T\n\nWe leverage a seamless solution — robust — best-in-class — "
             "world-class — cutting-edge — game-changer — turnkey.\n")
    sc = das.score_doc(dirty, "X.md")
    assert sc["appeal_debt"] > 0, sc
    assert sc["axes"]["voice"] < 100, sc["axes"]


def test_build_payload_ok_when_zero_debt() -> None:
    doc = {"path": "X.md", "appeal_score": 95.0, "grade": "A", "appeal_debt": 0,
           "axes": {"voice": 100}, "axis_detail": {}, "axis_debt": {}, "defects": [], "soft": []}
    p = das.build_payload(workspace=".", target_rel="X.md", doc=doc)
    assert p["ok"] is True and p["verdict"] == "OK", p


def test_build_payload_action_when_debt() -> None:
    doc = {"path": "X.md", "appeal_score": 60.0, "grade": "D", "appeal_debt": 5,
           "axes": {"voice": 40, "clarity": 90}, "axis_detail": {}, "axis_debt": {},
           "defects": ["voice: x"], "soft": []}
    p = das.build_payload(workspace=".", target_rel="X.md", doc=doc)
    assert p["ok"] is False and p["verdict"] == "ACTION", p
    assert "voice" in p["reason"], p


# --- live smoke: the real committed README folds without error -------------

def test_live_collect_real_readme() -> None:
    root = das.repo_root()
    if not (root / das.DEFAULT_TARGET_REL).exists():
        return  # tolerant: not in the repo tree
    p = das.collect(root)
    assert p["schema"] == das.SCHEMA, p
    assert "ok" in p and p.get("doc") and isinstance(p["doc"]["defects"], list), p


# --- self-contained runner (mirrors readme_freshness_audit_test.py) --------

def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
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
