#!/usr/bin/env python3
"""Tests for the skill-frontmatter lint (#422).

Drives the PURE classifier: the Claude-only/opencode-honored split, the three
load-bearing reasons (read-only allowlist, model-invocation gating, not-user-
invocable), the metadata.opencode acknowledgement escape hatch, the fold/gate
verdict ladder — then a tolerant live smoke that the repo's own skills fold and
that no skill is load-bearing-and-UNacknowledged on disk.

Run: `python tools/skill_frontmatter_lint_test.py`  (exit 0 = all pass),
or `python -m pytest tools/skill_frontmatter_lint_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_frontmatter_lint as lint  # noqa: E402


def fm(text: str) -> dict:
    return lint.parse_frontmatter(text)


# --- frontmatter parsing ----------------------------------------------------

def test_parse_flat_and_quoted() -> None:
    got = fm('---\nname: foo\nargument-hint: "[--x N]"\noutput_root: none\n---\nbody\n')
    assert got["name"] == "foo"
    assert got["argument-hint"] == "[--x N]"
    assert got["output_root"] == "none"


def test_parse_nested_metadata() -> None:
    got = fm("---\nname: foo\nmetadata:\n  opencode: claude-only\n  tier: t1\n---\n")
    assert isinstance(got["metadata"], dict)
    assert got["metadata"]["opencode"] == "claude-only"
    assert got["metadata"]["tier"] == "t1"


def test_parse_no_frontmatter() -> None:
    assert fm("no frontmatter here\n") == {}


# --- allowed-tools parsing --------------------------------------------------

def test_allowed_tools_csv_and_list() -> None:
    assert lint.parse_allowed_tools("Read, Bash, Grep") == ["Read", "Bash", "Grep"]
    assert lint.parse_allowed_tools(["Read", "Write"]) == ["Read", "Write"]
    assert lint.parse_allowed_tools("[Read, Edit]") == ["Read", "Edit"]


# --- classification ---------------------------------------------------------

def test_read_only_allowlist_is_load_bearing() -> None:
    rec = lint.classify(fm("---\nname: ro\nallowed-tools: Read, Bash, Grep, Glob\n---\n"))
    assert rec["load_bearing"] is True
    assert "read_only_boundary" in rec["reasons"]
    assert rec["acknowledged"] is None
    assert rec["flagged"] is True


def test_full_allowlist_is_not_read_only() -> None:
    rec = lint.classify(fm("---\nname: rw\nallowed-tools: Read, Edit, Write, Bash\n---\n"))
    assert "read_only_boundary" not in rec["reasons"]
    assert rec["load_bearing"] is False
    assert rec["flagged"] is False


def test_invocation_gating_is_load_bearing() -> None:
    rec = lint.classify(fm(
        "---\nname: gated\ndisable-model-invocation: true\nuser-invocable: false\n"
        "allowed-tools: Read, Edit, Bash\n---\n"))
    assert "model_invocation_gated" in rec["reasons"]
    assert "not_user_invocable" in rec["reasons"]
    assert rec["load_bearing"] is True
    assert rec["flagged"] is True


def test_permissive_invocation_not_flagged() -> None:
    rec = lint.classify(fm(
        "---\nname: open\ndisable-model-invocation: false\nuser-invocable: true\n"
        "allowed-tools: Read, Edit, Write\n---\n"))
    assert rec["reasons"] == []
    assert rec["load_bearing"] is False


def test_acknowledgement_clears_flag() -> None:
    rec = lint.classify(fm(
        "---\nname: ro\nallowed-tools: Read, Bash\nmetadata:\n  opencode: claude-only\n---\n"))
    assert rec["load_bearing"] is True       # still load-bearing...
    assert rec["acknowledged"] == "claude-only"
    assert rec["flagged"] is False           # ...but acknowledged, so not flagged


def test_unknown_ack_value_does_not_clear() -> None:
    rec = lint.classify(fm(
        "---\nname: ro\nallowed-tools: Read, Bash\nmetadata:\n  opencode: maybe\n---\n"))
    assert rec["acknowledged"] is None
    assert rec["flagged"] is True


# --- fold / gate ------------------------------------------------------------

def test_fold_gate_fails_on_unacknowledged() -> None:
    recs = [
        lint.classify(fm("---\nname: a\nallowed-tools: Read, Bash\n---\n")),
        lint.classify(fm("---\nname: b\nallowed-tools: Read, Write, Edit\n---\n")),
    ]
    out = lint.fold(recs, Path("."))
    assert out["ok"] is False
    assert out["n_flagged"] == 1
    code, msg = lint.check_gate(out)
    assert code == 1 and "unacknowledged" in msg.lower()


def test_fold_gate_passes_when_all_acked() -> None:
    recs = [
        lint.classify(fm(
            "---\nname: a\nallowed-tools: Read, Bash\nmetadata:\n  opencode: claude-only\n---\n")),
        lint.classify(fm("---\nname: b\nallowed-tools: Read, Write, Edit\n---\n")),
    ]
    out = lint.fold(recs, Path("."))
    assert out["ok"] is True
    assert out["n_load_bearing"] == 1 and out["n_acknowledged"] == 1
    code, _ = lint.check_gate(out)
    assert code == 0


# --- tolerant live smoke ----------------------------------------------------

def test_live_repo_skills_fold_and_pass() -> None:
    root = lint.repo_root()
    skills = lint.find_skills(root)
    assert skills, "no .claude/skills/*/SKILL.md found in the repo"
    records = [lint.lint_skill(p, root) for p in skills]
    out = lint.fold(records, root)
    for field in ("schema", "ok", "verdict", "taxonomy", "n_skills", "flagged", "next_action"):
        assert field in out, f"missing {field} in folded payload"
    # Every real skill must parse to a name (no silent frontmatter breakage).
    nameless = [r["path"] for r in records if not r.get("name") and "error" not in r]
    assert not nameless, f"skills with unparsable frontmatter: {nameless}"
    # The repo's own skills must leave the gate GREEN — every load-bearing skill
    # is acknowledged via metadata.opencode (#422 acceptance (a)/(b)).
    code, msg = lint.check_gate(out)
    assert code == 0, msg


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
