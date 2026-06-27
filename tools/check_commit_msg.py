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
import re
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
}
SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(\([^)]+\))?(?P<bang>!)?:\s+(?P<rest>.+)$")
EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")


def first_line(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def verdict(subject: str):
    """Return None if ok, else a reason string."""
    if not subject:
        return "empty subject"
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
    g.add_argument("--message", help="subject string (for testing)")
    a = ap.parse_args()

    if a.message is not None:
        subject = first_line(a.message)
    else:
        try:
            with open(a.file, encoding="utf-8") as fh:
                subject = first_line(fh.read())
        except OSError as e:
            print(f"COMMIT_MSG (warn): cannot read {a.file}: {e}", file=sys.stderr)
            return 2

    why = verdict(subject)
    if why is None:
        print("commit-msg: gradeable.")
        return 0
    print(f"COMMIT_MSG: subject not witness-gradeable — {why}", file=sys.stderr)
    print(f"  subject: {subject!r}", file=sys.stderr)
    print("  shape:   type(scope): <verb> <what>   e.g. fix(public): redact lab hostname", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
