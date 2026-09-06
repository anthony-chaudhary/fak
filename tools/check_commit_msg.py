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
    # Imperative base forms the DOS commit-audit referee witnesses as a code effect
    # (dos_witness_verbs.go dosCodeEffectVerbs) that fak's gate was REJECTING as ungradeable.
    "accumulate", "arm", "attribute", "author", "bind", "bridge", "carry",
    "consume", "dequant", "dequantize", "derive", "downgrade", "floor", "hook",
    "invert", "memoize", "optimise", "price", "refuse", "require", "reserve",
    "reset", "show", "splice", "synthesize",
    # A concrete imperative verb that names a checkable change but was absent from the harvest,
    # so `fak commit --preview` red-flagged a real subject the mutating `fak commit` accepted
    # and scored 100/A — the preview/mutation grade divergence of #3912. "isolate" leads a
    # genuine action (isolate a code path / behavior under test); kept in lockstep with the Go
    # commitVerbs set (internal/hooks/gate_commitmsg.go).
    "isolate", "retain", "quarantine", "scavenge",
}
SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(?P<scope>\([^)]+\))?(?P<bang>!)?:\s+(?P<rest>.+)$")
EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")

# verbSynonyms maps common unsupported imperative verbs or synonyms to their canonical
# recognized counterpart in VERBS (#11811).
VERB_SYNONYMS = {
    "synchronize": "sync",
    "synchronise": "sync",
    "inspect":     "verify",
    "examine":     "verify",
    "audit":       "witness",
    "probe":       "sample",
    "survey":      "sample",
    "monitor":     "log",
    "terminate":   "kill",
    "persist":     "record",
    "store":       "cache",
    "broadcast":   "publish",
    "notify":      "emit",
    "signal":      "emit",
    "modify":      "update",
    "alter":       "update",
    "change":      "update",
    "adjust":      "tune",
    "reorganize":  "restructure",
    "reorganise":  "restructure",
    "rearrange":   "restructure",
    "reconfigure": "tune",
    "configure":   "set",
    "truncate":    "cap",
    "clip":        "cap",
    "clamp":       "bound",
    "intercept":   "gate",
    "prohibit":    "prevent",
    "disallow":    "deny",
    "forbid":      "deny",
    "unblock":     "allow",
    "bypass":      "ignore",
    "discard":     "drop",
    "abandon":     "drop",
    "gather":      "accumulate",
    "collect":     "accumulate",
    "rerun":       "run",
    "retry":       "run",
    "teardown":    "reap",
    "coalesce":    "consolidate",
    "orchestrate": "dispatch",
    "instantiate": "create",
    "construct":   "build",
    "setup":       "scaffold",
    "bootstrap":   "scaffold",
    "initialize":  "introduce",
    "initialise":  "introduce",
    "init":        "introduce",
}

# unsupportedImperativeVerbs is a set of known imperative verbs that are not in VERBS
# and lack an unambiguous 1:1 rewrite, but are genuine verbs rather than nouns (#11811).
UNSUPPORTED_IMPERATIVE_VERBS = {
    "calculate", "compute", "evaluate", "estimate", "quantify", "measure",
    "abort", "halt", "stop", "pause", "resume", "restart",
    "retrieve", "fetch", "query", "poll", "listen",
    "trigger", "invoke", "call", "fire",
    "parse", "tokenize", "compile", "assemble", "interpret",
    "convert", "transform", "translate", "encode", "decode",
    "marshal", "unmarshal", "deserialize",
    "allocate", "deallocate", "free",
    "attach", "detach", "unbind",
    "connect", "disconnect", "reconnect",
    "override", "overwrite",
    "inject", "eject", "insert",
    "instrument", "trace", "profile",
    "compress", "decompress", "deflate", "inflate",
    "provision", "deprovision",
    "schedule", "reschedule", "defer", "delay",
    "throttle", "rate-limit",
    "unlock", "unmount", "mount",
    "clone", "copy",
    "coordinate",
    "escape", "unescape",
    "redirect", "reroute",
    "rearchitect",
    "mitigate", "alleviate", "relieve", "offload",
    "distribute", "partition", "shard",
    "wrap", "unwrap",
    "suppress", "silence",
    "invalidate",
    "check", "observe", "watch",
    "clean", "cleanup",
}

NEAR_MISS_TYPES = {
    "feature": "feat", "features": "feat",
    "fixes": "fix", "fixed": "fix", "bugfix": "fix", "bugfixes": "fix", "hotfix": "fix",
    "doc": "docs", "documentation": "docs",
    "tests": "test", "testing": "test",
    "chores": "chore",
    "refactoring": "refactor", "refactored": "refactor",
    "performance": "perf",
    "builds": "build",
    "styling": "style", "styles": "style",
    "reverts": "revert", "reverted": "revert",
}

IRREGULAR_VERB_BASES = {
    "built": "build", "made": "make", "ran": "run", "kept": "keep",
    "drove": "drive", "driven": "drive", "fed": "feed", "showed": "show", "shown": "show",
}


def de_double_consonant(s: str) -> str:
    n = len(s)
    if n >= 2 and s[-1] == s[-2]:
        return s[:-1]
    return s


def imperative_base_forms(w: str) -> list[str]:
    out = [w]

    def add(s: str):
        if s and s != w and s not in out:
            out.append(s)

    if w.endswith("ies"):
        add(w[:-3] + "y")
    elif w.endswith("es"):
        add(w[:-2])
        add(w[:-1])
    elif w.endswith("s"):
        add(w[:-1])

    if w.endswith("ied"):
        add(w[:-3] + "y")
    elif w.endswith("ed"):
        base = w[:-2]
        add(base)
        add(w[:-1])
        add(de_double_consonant(base))

    if w.endswith("ing"):
        base = w[:-3]
        add(base)
        add(base + "e")
        add(de_double_consonant(base))

    if w in IRREGULAR_VERB_BASES:
        add(IRREGULAR_VERB_BASES[w])

    return out


def lookup_verb_synonym(w: str) -> tuple[str, bool]:
    if w in VERB_SYNONYMS:
        return VERB_SYNONYMS[w], True
    for cand in imperative_base_forms(w):
        if cand in VERB_SYNONYMS:
            return VERB_SYNONYMS[cand], True
    return "", False


def is_unsupported_imperative_verb(w: str) -> bool:
    if w in UNSUPPORTED_IMPERATIVE_VERBS or w in VERB_SYNONYMS:
        return True
    for cand in imperative_base_forms(w):
        if cand in UNSUPPORTED_IMPERATIVE_VERBS or cand in VERB_SYNONYMS:
            return True
    return False


def lint_subject_verb(first: str) -> tuple[bool, str]:
    if first in VERBS:
        return True, ""
    syn, ok = lookup_verb_synonym(first)
    if ok:
        return False, f"description leads with unsupported imperative verb '{first}' (consider '{syn}' or a recognized verb: add/fix/implement/…)."
    if is_unsupported_imperative_verb(first):
        return False, f"description leads with unsupported imperative verb '{first}', not a recognized verb. Lead with a recognized verb (add/fix/implement/…)."
    return False, f"description leads with '{first}', not a recognized verb — the witness ABSTAINs on a noun-led subject. Lead with a verb (add/fix/implement/…)."


def imperative_base(w: str) -> str:
    for cand in imperative_base_forms(w):
        if cand in VERBS:
            return cand
    return ""


def suggest_gradeable_subject(subject: str) -> str:
    subject = subject.strip()
    if not subject:
        return ""
    for p in EXEMPT_PREFIXES:
        if subject.startswith(p):
            return ""
    m = SUBJECT_RE.match(subject)
    if not m:
        return ""
    typ = m.group("type")
    bang = m.group("bang") or ""
    scope = m.group("scope") or ""
    rest = m.group("rest").strip()

    if typ not in TYPES:
        canon = NEAR_MISS_TYPES.get(typ)
        if not canon:
            return ""
        typ = canon

    first_word = re.split(r"[\s:]", rest, maxsplit=1)[0]
    first_lower = first_word.lower()
    if first_lower not in VERBS:
        if first_word.strip("`*\"'") != first_word:
            return ""
        base = imperative_base(first_lower)
        if not base:
            syn, ok = lookup_verb_synonym(first_lower)
            if ok:
                base = syn
        if not base:
            return ""
        rest = base + rest[len(first_word):]

    rebuilt = f"{typ}{scope}{bang}: {rest}"
    if rebuilt == subject:
        return ""
    if verdict(rebuilt) is None:
        return rebuilt
    return ""


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
    ok, why = lint_subject_verb(first)
    if not ok:
        return why
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
        suggestion = suggest_gradeable_subject(subject)
        if suggestion:
            print(f"  suggest: {suggestion}", file=sys.stderr)
    if "is not `type" in why or "unknown type" in why or "recognized verb" in why:
        print("  shape:   type(scope): <verb> <what>   e.g. fix(public): redact lab hostname", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
