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
        "test(codex): isolate direct continuation override",   # #3912: isolate now accepted
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


def test_conflict_template_rejected():
    print("unedited conflict templates are rejected (#11306):")
    msg1 = "Merge remote-tracking branch 'origin/main'\n\n# Conflicts:\n#\tcmd/fak/serve.go"
    why1 = mod.verdict(msg1)
    check("conflict template '# Conflicts:' flagged", why1 is not None and "MERGE_CONFLICT_TEMPLATE_FORBIDDEN" in why1, f"got {why1!r}")

    msg2 = "# Conflicts:\n"
    why2 = mod.verdict(msg2)
    check("bare '# Conflicts:' flagged", why2 is not None and "MERGE_CONFLICT_TEMPLATE_FORBIDDEN" in why2, f"got {why2!r}")


def test_conflict_markers_rejected():
    print("conflict markers are rejected (#11306):")
    msg1 = "feat(core): update\n\n<<<<<<< HEAD\nfoo\n=======\nbar\n>>>>>>> main"
    why1 = mod.verdict(msg1)
    check("conflict markers flagged", why1 is not None and "MERGE_CONFLICT_MARKERS_FORBIDDEN" in why1, f"got {why1!r}")

    why_less = mod.verdict("<<<<<<< HEAD")
    check("'<<<<<<<' flagged", why_less is not None and "MERGE_CONFLICT_MARKERS_FORBIDDEN" in why_less, f"got {why_less!r}")

    why_eq = mod.verdict("=======")
    check("'=======' flagged", why_eq is not None and "MERGE_CONFLICT_MARKERS_FORBIDDEN" in why_eq, f"got {why_eq!r}")

    why_gt = mod.verdict(">>>>>>> main")
    check("'>>>>>>>' flagged", why_gt is not None and "MERGE_CONFLICT_MARKERS_FORBIDDEN" in why_gt, f"got {why_gt!r}")


def test_silent_drop_merge_rejection():
    print("silent drop merges are rejected without override (#11306):")
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        def git(*args):
            return subprocess.run(["git", "-C", td] + list(args),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        git("init", "-q")
        git("config", "user.name", "Test")
        git("config", "user.email", "test@example.com")
        git("config", "commit.gpgsign", "false")

        # 1. Base commit
        with open(os.path.join(td, "a.txt"), "w") as f:
            f.write("base\n")
        git("add", "a.txt")
        git("commit", "-m", "feat(core): initial base")

        # 2. Side branch with unique non-empty commit
        main_branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        git("checkout", "-q", "-b", "side-branch")
        with open(os.path.join(td, "b.txt"), "w") as f:
            f.write("side content\n")
        git("add", "b.txt")
        git("commit", "-m", "feat(side): side change")

        # 3. Main branch with commit
        git("checkout", "-q", main_branch)
        with open(os.path.join(td, "c.txt"), "w") as f:
            f.write("main content\n")
        git("add", "c.txt")
        git("commit", "-m", "feat(main): main change")

        # 4. Start merge with -s ours --no-commit
        git("merge", "-s", "ours", "--no-commit", "side-branch")

        # Test in-flight merge: tree matches HEAD while side-branch has unique commits
        why = mod.verdict("Merge branch 'side-branch'", root=td)
        check("silent drop in-flight merge rejected", why is not None and "SILENT_DROP_MERGE_FORBIDDEN" in why, f"got {why!r}")

        # Test override with Merge-Strategy: ours trailer
        msg_trailer1 = "Merge branch 'side-branch'\n\nMerge-Strategy: ours"
        check("override 'Merge-Strategy: ours' allowed", mod.verdict(msg_trailer1, root=td) is None)

        # Test override with Silent-Merge: intentional trailer
        msg_trailer2 = "Merge branch 'side-branch'\n\nSilent-Merge: intentional"
        check("override 'Silent-Merge: intentional' allowed", mod.verdict(msg_trailer2, root=td) is None)

        # Test override with ALLOW_SILENT_MERGE=1 env var
        os.environ["ALLOW_SILENT_MERGE"] = "1"
        try:
            check("override ALLOW_SILENT_MERGE=1 allowed", mod.verdict("Merge branch 'side-branch'", root=td) is None)
        finally:
            del os.environ["ALLOW_SILENT_MERGE"]

        # 5. Commit with trailer and test existing commit
        git("commit", "-m", "Merge branch 'side-branch'\n\nMerge-Strategy: ours")
        check("committed merge with trailer allowed", mod.verdict("Merge branch 'side-branch'\n\nMerge-Strategy: ours", root=td) is None)

        # But checking an existing commit without trailer rejects it
        why_no_trailer = mod.verdict("Merge branch 'side-branch'", root=td, commit_ref="HEAD")
        check("committed silent drop merge without trailer rejected", why_no_trailer is not None and "SILENT_DROP_MERGE_FORBIDDEN" in why_no_trailer, f"got {why_no_trailer!r}")


def test_historical_issue_11306_commit():
    import os
    if not os.path.exists(".git"):
        return
    print("historical issue #11306 commit 77e8525229 is rejected:")
    import subprocess
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", "77e8525229"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return
    # Message of 77e8525229 contains # Conflicts:
    r_msg = subprocess.run(["git", "log", "-1", "--format=%B", "77e8525229"],
                           stdout=subprocess.PIPE, text=True)
    why1 = mod.verdict(r_msg.stdout, root=".")
    check("77e8525229 rejected by conflict template", why1 is not None and "MERGE_CONFLICT_TEMPLATE_FORBIDDEN" in why1, f"got {why1!r}")

    # Stripped message of 77e8525229 rejected by silent drop merge
    stripped = "Merge remote-tracking branch 'origin/main'\n\nSigned-off-by: Fak Agent <fak-agent@users.noreply.github.com>"
    why2 = mod.verdict(stripped, root=".", commit_ref="77e8525229")
    check("77e8525229 stripped rejected by silent drop merge", why2 is not None and "SILENT_DROP_MERGE_FORBIDDEN" in why2, f"got {why2!r}")


def main() -> int:
    test_valid_subjects_pass()
    test_unknown_type_flagged()
    test_noun_led_description_flagged()
    test_malformed_subject_flagged()
    test_exempt_prefixes()
    test_empty_subject()
    test_conflict_template_rejected()
    test_conflict_markers_rejected()
    test_silent_drop_merge_rejection()
    test_historical_issue_11306_commit()
    print(f"\ncheck_commit_msg_test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
