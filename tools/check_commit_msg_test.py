#!/usr/bin/env python3
"""Tests for tools/check_commit_msg.py — the commit-subject grading gate.

Pure stdlib, no git. Exercises verdict(): valid type+verb subjects pass; an
unknown type, a noun-led description, and a malformed subject each get a reason;
Merge/Revert/fixup! are exempt; the recently-added imperatives (archive/ignore/
back) grade clean so the gate stops crying wolf on legitimate subjects.
"""
from __future__ import annotations

import check_commit_msg as mod

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok  ] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_valid_subjects_pass():
    print("valid type(scope): <verb> subjects grade clean:")
    for s in [
        "feat(gateway): add the embeddings endpoint",
        "fix(public): redact lab hostname",
        "docs(readme): clarify the install steps",
        "chore(repo): ignore public/private parity scratch",   # newly-added verb
        "feat(tools): archive .dos work-product to the private repo",  # newly-added
        "chore(tools): back up the durable .dos markdown",     # newly-added "back"
    ]:
        check(f"clean: {s!r}", mod.verdict(s) is None, f"got {mod.verdict(s)!r}")


def test_unknown_type_flagged():
    print("an unknown type is flagged (add/chore are verbs, not types):")
    # 'add(tools): ...' — 'add' is a VERB but NOT a valid TYPE; must be flagged.
    why = mod.verdict("add(tools): dos_sync the work-product")
    check("'add(' type rejected", why is not None and "unknown type" in why, f"got {why!r}")


def test_noun_led_description_flagged():
    print("a noun-led description is flagged:")
    why = mod.verdict("fix(tools): security_audit resolves the path")
    check("noun-led 'security_audit' flagged", why is not None and "not a recognized verb" in why, f"got {why!r}")


def test_malformed_subject_flagged():
    print("a subject without the type(scope): shape is flagged:")
    why = mod.verdict("just some words with no conventional prefix")
    check("no-prefix subject flagged", why is not None and "is not `type" in why, f"got {why!r}")


def test_exempt_prefixes():
    print("Merge/Revert/fixup! subjects are exempt:")
    for s in ["Merge branch 'main' into x", "Revert \"feat: y\"", "fixup! feat(x): z"]:
        check(f"exempt: {s!r}", mod.verdict(s) is None, f"got {mod.verdict(s)!r}")


def test_empty_subject():
    print("empty subject is flagged:")
    check("empty flagged", mod.verdict("") == "empty subject")


def main() -> int:
    test_valid_subjects_pass()
    test_unknown_type_flagged()
    test_noun_led_description_flagged()
    test_malformed_subject_flagged()
    test_exempt_prefixes()
    test_empty_subject()
    print(f"\ncheck_commit_msg_test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
