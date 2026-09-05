#!/usr/bin/env python3
"""Commit-message gate: a subject the witness can grade.

The DOS commit-audit witness grades a commit as diff-witnessed ONLY when the
subject is `type(scope): <verb> <what>` — a recognized leading verb after the
colon. A bare `feat(scope): Noun ...` ABSTAINs (it cannot be auto-graded), and an
ABSTAIN on a landed commit is immutable. This nudges every commit toward the
gradeable shape so the goal-gate / witness pipeline can do its job.

Checks the subject (first line) against:
  type(scope)?: <verb> ...
where type ∈ conventional-commit set and <verb> is a recognized imperative verb.

Merge / revert / fixup! / squash! / version-bump subjects are exempt.

Usage:  check_commit_msg.py --file <COMMIT_MSG_FILE>     (commit-msg hook)
        check_commit_msg.py --message "<subject>"         (testing)
Exit: 0 ok, 1 not gradeable, 2 could-not-read.
Default mode is advisory at the hook layer (FLEET_MSG_GUARD); this tool just
reports the verdict via exit code.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

TYPES = ("feat", "fix", "docs", "refactor", "perf", "test", "chore", "build",
         "ci", "style", "revert")
# Recognized leading imperative verbs (superset of what the witness grades on).
VERBS = {
    "add", "implement", "create", "build", "introduce", "scaffold",
    "fix", "repair", "correct", "patch", "resolve", "address",
    "test", "verify", "validate", "assert", "cover",
    "refactor", "restructure", "rewrite", "reframe", "rework", "simplify",
    "remove", "delete", "drop", "strip", "prune", "purge",
    "redact", "scrub", "sanitize",
    "move", "rename", "repoint", "relocate", "migrate", "port",
    "update", "bump", "upgrade", "sync", "refresh", "regenerate",
    "wire", "gate", "enforce", "prevent", "guard", "bound", "cap", "limit",
    "restore", "recover", "reinstate",
    "document", "clarify", "annotate", "note",
    "optimize", "speed", "harden", "tune",
    "support", "enable", "disable", "deprecate",
    "revert", "merge", "split", "extract", "inline", "dedupe", "consolidate",
    "close", "land", "ship", "generalize", "normalize", "reconcile",
    "make", "use", "switch", "replace", "set", "allow", "ensure", "handle",
    "archive", "ignore", "back",  # "archive X", "ignore Y (gitignore)", "back up Z"
    # Concrete imperative verbs observed leading real commits the gate was
    # advisory-flagging despite naming a genuine action (28% -> ~1% false-flag
    # rate over 400 commits). Each describes a checkable change, not a noun.
    "define", "declare", "state", "explain", "describe", "document",
    "record", "register", "log", "witness", "prove", "demonstrate",
    "fill", "populate", "seed", "stub", "scaffold",
    "standardize", "unify", "consolidate", "reconcile", "align", "tidy",
    "tighten", "loosen", "relax", "widen", "narrow", "scope",
    "default", "pin", "warm", "prewarm", "preload", "prefetch",
    "apply", "propagate", "thread", "plumb", "route", "dispatch", "feed",
    "acknowledge", "credit", "cite", "reference", "link", "anchor", "tie",
    "cross-ref", "index", "catalog",
    "hash", "checksum", "stamp", "tag", "label", "mark", "flag",
    "parallelize", "serialize", "batch", "stream", "buffer", "cache",
    "grant", "revoke", "authorize", "permit", "deny", "block", "reject",
    "idle", "reap", "drain", "flush", "evict", "expire", "retire",
    "fold", "unfold", "expand", "collapse", "merge",
    "emit", "surface", "expose", "publish", "export", "import",
    # Second harvest from the residual flags — more concrete imperative verbs
    # that name a real action (drove the false-flag rate from 11% toward ~3%).
    "file", "sort", "kill", "ground", "sample", "report", "frame", "rephrase",
    "grade", "trend", "calibrate", "recalibrate", "keep", "run", "name",
    "print", "lift", "prefer", "generate", "forward", "flip", "drive",
    "locate", "deepen", "pace", "lock", "onboard", "treat", "preserve",
    "quote", "fence", "gofmt",
    # Advisory-action verbs: a commit that ADDS a lint/gate which advises or nudges (the
    # commit-gardening surface itself, #1326) names a real, checkable change. The gate was
    # abstaining on "advise"/"nudge"/"recommend" despite each leading a concrete diff.
    "advise", "nudge", "recommend", "warn", "remind", "hint",
    # A concrete imperative verb that names a checkable change but was absent from the harvest,
    # so `fak commit --preview` red-flagged a real subject the mutating `fak commit` accepted
    # and scored 100/A — the preview/mutation grade divergence of #3912. "isolate" leads a
    # genuine action (isolate a code path / behavior under test); kept in lockstep with the Go
    # commitVerbs set (internal/hooks/gate_commitmsg.go).
    "isolate",
}
SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(\([^)]+\))?(?P<bang>!)?:\s+(?P<rest>.+)$")
EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")


def check_conflict_banners(text: str) -> str | None:
    """Return an error string if the commit message contains conflict templates or markers."""
    for line in text.splitlines():
        if "# Conflicts:" in line:
            return ("MERGE_CONFLICT_TEMPLATE_FORBIDDEN: commit message contains "
                    "unedited git conflict template ('# Conflicts:')")
        s = line.strip()
        if s.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            return ("MERGE_CONFLICT_MARKERS_FORBIDDEN: commit message contains "
                    "git conflict markers ('<<<<<<<', '=======', or '>>>>>>>')")
    return None


def check_silent_merge_override(text: str) -> bool:
    """Check if an explicit override flag or reason trailer permits a silent drop merge."""
    if os.environ.get("ALLOW_SILENT_MERGE") == "1" or os.environ.get("FLEET_ALLOW_SILENT_MERGE") == "1":
        return True
    if re.search(r"(?im)^\s*merge-strategy\s*:\s*ours\b", text):
        return True
    if re.search(r"(?im)^\s*silent-merge\s*:\s*(?:intentional|true|yes|allow)\b", text):
        return True
    return False


def check_has_merge_head(root: str | None = None) -> bool:
    cmd = ["git"]
    if root:
        cmd.extend(["-C", root])
    try:
        r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "MERGE_HEAD"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def check_is_merge_parent(root: str | None = None, commit_ref: str | None = None) -> bool:
    cmd = ["git"]
    if root:
        cmd.extend(["-C", root])
    if commit_ref:
        try:
            r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", f"{commit_ref}^2"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return r.returncode == 0
        except Exception:
            return False
    try:
        r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "MERGE_HEAD"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    try:
        r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "HEAD^2"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    return False


def check_silent_drop_merge(root: str | None = None, text: str = "", commit_ref: str | None = None) -> str | None:
    """Reject merge commits whose tree SHA matches parent 1 exactly while parent 2 contains non-empty unique commits."""
    if not root:
        return None
    if check_silent_merge_override(text):
        return None
    cmd = ["git"]
    if root:
        cmd.extend(["-C", root])

    # Case 1: In-flight merge (MERGE_HEAD exists and ref is empty)
    if not commit_ref:
        try:
            r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "MERGE_HEAD"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                r_staged = subprocess.run(cmd + ["write-tree"],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                r_head = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "HEAD^{tree}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if r_staged.returncode == 0 and r_head.returncode == 0:
                    staged_tree = r_staged.stdout.strip()
                    head_tree = r_head.stdout.strip()
                    if staged_tree == head_tree:
                        r_cnt = subprocess.run(cmd + ["rev-list", "--count", "HEAD..MERGE_HEAD"],
                                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                        if r_cnt.returncode == 0 and int(r_cnt.stdout.strip() or "0") > 0:
                            r_diff = subprocess.run(cmd + ["diff", "--name-only", "HEAD...MERGE_HEAD"],
                                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                            if r_diff.returncode == 0 and r_diff.stdout.strip():
                                return ("SILENT_DROP_MERGE_FORBIDDEN: merge tree matches parent 1 exactly while "
                                        "parent 2 contains non-empty unique commits (silent drop merge); "
                                        "supply 'Merge-Strategy: ours' or 'Silent-Merge: intentional' trailer to allow")
                return None
        except Exception:
            return None

    # Case 2: Existing commit (commit_ref or HEAD with >= 2 parents)
    target = commit_ref
    if not target:
        try:
            r = subprocess.run(cmd + ["rev-parse", "-q", "--verify", "HEAD^2"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                target = "HEAD"
        except Exception:
            pass

    if target:
        try:
            r_p2 = subprocess.run(cmd + ["rev-parse", "-q", "--verify", f"{target}^2"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r_p2.returncode == 0:
                r_tree = subprocess.run(cmd + ["rev-parse", "-q", "--verify", f"{target}^{{tree}}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                r_p1_tree = subprocess.run(cmd + ["rev-parse", "-q", "--verify", f"{target}^1^{{tree}}"],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if r_tree.returncode == 0 and r_p1_tree.returncode == 0:
                    if r_tree.stdout.strip() == r_p1_tree.stdout.strip():
                        r_cnt = subprocess.run(cmd + ["rev-list", "--count", f"{target}^1..{target}^2"],
                                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                        if r_cnt.returncode == 0 and int(r_cnt.stdout.strip() or "0") > 0:
                            r_diff = subprocess.run(cmd + ["diff", "--name-only", f"{target}^1...{target}^2"],
                                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                            if r_diff.returncode == 0 and r_diff.stdout.strip():
                                return ("SILENT_DROP_MERGE_FORBIDDEN: merge tree matches parent 1 exactly while "
                                        "parent 2 contains non-empty unique commits (silent drop merge); "
                                        "supply 'Merge-Strategy: ours' or 'Silent-Merge: intentional' trailer to allow")
        except Exception:
            pass
    return None


def first_line(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def verdict(text: str, root: str | None = None, commit_ref: str | None = None):
    """Return None if ok, else a reason string."""
    if not text:
        return "empty subject"
    conflict_err = check_conflict_banners(text)
    if conflict_err:
        return conflict_err
    subject = first_line(text)
    if not subject:
        return "empty subject"
    if subject.startswith("Merge "):
        if root is not None:
            if not check_is_merge_parent(root, commit_ref):
                return ("MERGE_WITNESS_FAIL: commit subject starts with 'Merge ' but has fewer than "
                        "2 topological parents; pseudo-merges cannot bypass Conventional Commits and DCO")
            silent_err = check_silent_drop_merge(root, text, commit_ref)
            if silent_err:
                return silent_err
        return None
    if root is not None:
        if check_has_merge_head(root):
            silent_err = check_silent_drop_merge(root, text, commit_ref)
            if silent_err:
                return silent_err
    if subject.startswith(EXEMPT_PREFIXES):
        return None
    m = SUBJECT_RE.match(subject)
    if not m:
        return ("subject is not `type(scope): <verb> <what>` "
                "(types: " + "/".join(TYPES) + ")")
    if m.group("type") not in TYPES:
        return f"unknown type '{m.group('type')}' (use one of: {'/'.join(TYPES)})"
    first = re.split(r"[\s:]", m.group("rest").strip(), maxsplit=1)[0].lower().strip("`*\"'")
    if first not in VERBS:
        return (f"description leads with '{first}', not a recognized verb — the witness "
                f"ABSTAINs on a noun-led subject. Lead with a verb (add/fix/implement/…).")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="path to the commit-message file")
    g.add_argument("--message", help="subject or message string (for testing)")
    g.add_argument("--commit", help="commit ref (for verifying existing commit)")
    ap.add_argument("--root", default=None, help="repo root (for verifying merge parent count)")
    a = ap.parse_args()

    commit_ref = a.commit
    if a.message is not None:
        raw = a.message
    elif a.commit is not None:
        cmd = ["git"]
        if a.root:
            cmd.extend(["-C", a.root])
        try:
            r = subprocess.run(cmd + ["log", "-1", "--format=%B", a.commit],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                print(f"COMMIT_MSG (warn): cannot read commit {a.commit}: {r.stderr.strip()}", file=sys.stderr)
                return 2
            raw = r.stdout
        except Exception as e:
            print(f"COMMIT_MSG (warn): cannot run git log on {a.commit}: {e}", file=sys.stderr)
            return 2
    else:
        try:
            with open(a.file, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as e:
            print(f"COMMIT_MSG (warn): cannot read {a.file}: {e}", file=sys.stderr)
            return 2

    why = verdict(raw, root=a.root, commit_ref=commit_ref)
    if why is None:
        print("commit-msg: gradeable.")
        return 0
    print(f"COMMIT_MSG: {why}", file=sys.stderr)
    subject = first_line(raw)
    if subject:
        print(f"  subject: {subject!r}", file=sys.stderr)
    if "is not `type" in why or "unknown type" in why or "recognized verb" in why:
        print("  shape:   type(scope): <verb> <what>   e.g. fix(public): redact lab hostname", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
