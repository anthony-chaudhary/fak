#!/usr/bin/env python3
r"""The dispatch loop's *close-the-resolved* arm: drive OPEN_WITNESSED issues to
CLOSED, each gated on a witness the loop did not author.

``issue_closure_audit.py`` surfaces issues bucketed ``OPEN_WITNESSED`` — still
open on GitHub, yet already carrying a *diff-witnessed* resolving commit in git
ancestry. Those are exactly the tickets a correct dispatcher should close: the
work shipped, only the bookkeeping lags, and every extra OPEN_WITNESSED row drags
``closure_rate`` down. This is the deterministic half of "do N issues" — no model
worker, no code edit, no DoS — and it is the safest possible live proof that the
loop can move real issues, because the keep-bit is a git fact:

  for each candidate:
    re-run `dos commit-audit <sha> --json`     # env-authored, re-verified HERE
    iff verdict==OK and witness==diff-witnessed
        and the CLAIM KIND binds resolution (a code/test claim over non-doc
        paths, or the issue itself is a docs rung — a docs/triage commit that
        merely references #N witnesses a note, never #N's feature witness):
       gh issue close <n> --comment "<sha> (<subject>) resolves this; ..."
       gh issue view  <n> --json state   # readback: count iff state==CLOSED (#2641)

The re-verification is the whole point (.claude/rsi-loop-dod.md: "no keep on a
self-authored claim"): the closer does NOT trust the audit's bucket, it re-asks
the oracle per-SHA at close time. A close cites its witnessing SHA + subject in
the comment, so it is auditable and trivially reversible (``gh issue reopen``).

DRY-RUN BY DEFAULT — prints the exact `gh` commands and the per-issue witness.
``--live`` executes. ``--limit N`` bounds the batch (default 10).

    python tools/issue_resolve_witnessed.py                  # plan 10 closes (dry-run)
    python tools/issue_resolve_witnessed.py --limit 10 --live  # execute
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dispatch_worker import install_no_window_subprocess_defaults

install_no_window_subprocess_defaults(subprocess)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-issue-resolve-witnessed/1"
WITNESS_OK = "diff-witnessed"
# Claim kinds that can BIND a close (#2998): a diff-witnessed doc/triage claim
# witnesses only that a note shipped — it never resolves a non-docs issue. A
# code/test claim over non-doc paths binds. `dos commit-audit` emits the bare
# token `test` for a test-covering commit (verified on the live oracle); the
# historical `test_cover` is kept for backward-safety in case the vocabulary
# ever re-widens. Without `test` the closer silently held EVERY test-witnessed
# resolution (test-first / qa / regression / coverage issues, and feat commits
# dos grades test-line-dominant), booking 0 closes for the whole class.
RESOLVING_CLAIM_KINDS = {"code_effect", "test", "test_cover"}
NONRESOLVING_HOLD = "CLAIM_KIND_NONRESOLVING"
# Author-disclaimed resolution gate (#5865): every OTHER gate here reasons over the
# ISSUE (labels, body, timeline) or over the commit's DIFF SHAPE (claim kind, touched
# paths). None reads the commit MESSAGE -- which is exactly where an author states
# scope. The witnessed harm: #2694 was planned `would_close` on 2726d19, whose own body
# says "this commit does not claim it [...] No artifact is claimed under `visuals/` [...]
# The issue stays open." It got through because the #2998 claim gate has a docs-rung
# carve-out (a `docs(`-titled issue accepts a doc claim) and #2694's body carries no
# unchecked task box for the #3870 coverage gate to catch.
#
# The gate is deliberately RUNG-BLIND, because the carve-out is only one of the doors
# this class walks through: 3dbe6bf2 (#3258) is `code_effect` over real source paths and
# still says "claiming it as a served endpoint now would be claiming an integration that
# has no witness". A commit that disclaims its own resolution cannot witness one at any
# rung.
#
# Markers are AUTHOR-PLACED disclaimers, never an inference over general commit wording.
# Measured over the last 1500 commits this set matches 7 (0.47%), and every hit is a
# genuine "I did not finish this" note. A bare `follow-on` marker was REJECTED for this
# set on that measurement: it matches 56/1500 (3.7%) and would falsely refuse fec8da6f76,
# one of #2299's own witnesses ("The cmd/fak host wiring [...] is a deferred follow-on").
DISCLAIMED_HOLD = "COMMIT_DISCLAIMS_RESOLUTION"
# The witness commit message could not be read. Unlike the issue-side gates this is an
# ALLOW with an audit note, not a hold -- see ``disclaimer_binds_closure``.
DISCLAIM_UNREADABLE_NOTE = "COMMIT_BODY_UNREADABLE"
# `does not` / `doesn't` / `doesnt`, with either quote glyph (issue+commit prose uses the
# smart quote U+2019 freely, and a marker that missed it would be trivially evadable).
_NT = r"does\s*n[o’']?t"
_DISCLAIM_MARKERS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("the issue stays open", re.compile(
        r"\b(?:the|this)\s+issue\s+(?:still\s+)?(?:stays|remains|is\s+still)\s+open\b",
        re.IGNORECASE)),
    ("does not close the issue", re.compile(
        rf"\b{_NT}\s+close\s+(?:this|the)\s+issue\b", re.IGNORECASE)),
    ("leaves the issue open", re.compile(
        r"\b(?:leaves?|keeps?)\s+(?:the\s+|this\s+)?issue\s+open\b", re.IGNORECASE)),
    ("this commit does not claim it", re.compile(
        rf"\bthis\s+(?:commit|change|patch)\s+{_NT}\s+"
        r"(?:claim|close|resolve|satisfy|witness)\b", re.IGNORECASE)),
    ("no artifact is claimed", re.compile(
        r"\bno\s+(?:artifact|witness|evidence|resolution|fix)\s+is\s+claimed\b",
        re.IGNORECASE)),
    ("the work has no witness", re.compile(
        r"\b(?:has|have|with|carries)\s+no\s+witness\b", re.IGNORECASE)),
    ("not yet witnessed", re.compile(
        r"\bnot\s+yet\s+witnessed\b|\bno\s+witness\s+yet\b", re.IGNORECASE)),
    ("would be claiming", re.compile(
        r"\bwould\s+be\s+claiming\b", re.IGNORECASE)),
)
# The one NUMBERED form, scoped to the issue actually under consideration. A body saying
# "#5847 stays OPEN" disclaims *that* issue: holding every OTHER issue's close on it would
# be an over-refusal, so this marker is compiled per-candidate against its own number.
_DISCLAIM_THIS_ISSUE_TMPL = r"#{n}\s+(?:still\s+)?(?:stays|remains)\s+open\b"
# State-readback idempotency (#2641): `gh issue close` returning rc 0 proves the
# COMMAND ran, not that the issue is durably closed — a concurrent/lagging reopen
# leaves it OPEN/REOPENED. The closure loop reads the authoritative state back and
# counts a close only when GitHub reports CLOSED; a still/again-open issue is a
# distinct ``close_not_persistent`` event (never tallied), and a repeated close of
# an issue already counted THIS run is ``already_counted`` once (unless the readback
# shows a new reopen) so the progress ledger can never double-count a reopened issue.
DURABLE_CLOSED_STATE = "CLOSED"
CLOSE_NOT_PERSISTENT = "close_not_persistent"
CLOSE_ALREADY_COUNTED = "already_counted"
# Coverage gate (#3870): a single diff-witnessed commit resolves an issue only when
# nothing EXPLICITLY declares the issue to be multi-part. A first-of-many commit that
# merely references an epic (or a spine-first artifact) is diff-witnessed for its own
# small claim yet closes the WHOLE tracking ticket -- the partial-close failure. These
# are high-precision author-intent markers (an `epic` label, an unchecked task box, a
# spine-first work-unit), NOT a heuristic over every multi-bullet body, so the gate
# holds the documented partial closes without stranding ordinary single-scope issues.
PARTIAL_HOLD = "RESOLVED_PARTIAL"
COVERAGE_UNKNOWN_HOLD = "COVERAGE_UNKNOWN"
# Reopen-supersedes-witness gate (#4374): the close-resolved arm cites a witnessing
# commit, but an auto-reclose must NOT override a `reopened` event unless a commit
# landed AFTER it. Re-closing on a pre-reopen commit silently undoes a correction
# reopen -- and in the witnessed #4350 case re-marked a BROKEN main "resolved" (the
# reopen carried a red-at-HEAD regression the narrow close-gate witness never ran).
# A reopen with no newer commit stays open; an unreadable timeline fails CLOSED.
REOPEN_NO_NEW_COMMIT_HOLD = "REOPENED_NO_NEW_COMMIT"
REOPEN_UNKNOWN_HOLD = "REOPEN_UNKNOWN"
# Absent GitHub context vs UNKNOWN answer. Every gate below (#4374 reopen, #3870
# coverage, #4747 observed-effect) asks GitHub a question ABOUT AN ISSUE, which
# presupposes the workspace is bound to a GitHub repository at all. When it is not,
# gh does not return an unknown answer -- it reports the question is unaskable:
#   gh issue view <n>                     -> "no git remotes found"
#   gh api repos/{owner}/{repo}/...       -> "unable to expand placeholder in path:
#                                             no git remotes found"
# Failing CLOSED there is a category error, not caution, and it is what made the
# hermetic end-to-end smokes (cmd/fak/handoff_chain_smoke_test.go,
# cmd/fak/relay_handoff_rotate_close_test.go -- both drive this script against a
# throwaway `git init` workspace with NO remote) assert `would_close` yet always get
# a hold. A gate with no tracker to consult is INAPPLICABLE, not UNKNOWN.
# The relaxation cannot cause a wrong close: in exactly the workspaces this matches,
# `gh issue close` is unrunnable for the same reason, so a --live run there ends in
# close_failed, never a false close. Anywhere gh CAN resolve a repo -- i.e. every
# real run against a real issue -- a failed read is still a genuine unknown and
# still fails closed, so the #4374 safety property is untouched.
GH_CONTEXT_ABSENT_NOTE = "GH_CONTEXT_ABSENT"
_NO_GH_REPO_RE = re.compile(r"no git remotes? found|not a git repository",
                            re.IGNORECASE)
# Identity-compared marker returned by ``fetch_issue_meta`` for that same case. It is
# a dict so the return type stays ``dict | None``, but callers test it with ``is`` --
# a plain ``None`` keeps meaning "unreadable" (fail CLOSED), which is what every
# existing caller and stub already expects.
NO_GITHUB_CONTEXT: dict[str, Any] = {"_gh_context": "absent"}
# Observed-effect gate (#4747): a MODEL-CORRECTNESS defect (real-weight,
# architecture, coherence) may only auto-close on OBSERVED-EFFECT evidence — an
# independent real artifact demonstrating the original symptom is gone.
# Instrumentation / capture / gate code is an EARLIER resolution class
# (diagnostic shipped), not resolution evidence: #4273 and #4627 were closed
# after instrumentation landed while their required real-27B observed artifacts
# were still missing, and the defect stayed operationally present. The gate
# fires on high-precision markers only — an explicit model-defect label, a typed
# terminal-class declaration in the body (`Resolution-Class: effect-observed`,
# the issue-template marker), or the #4273/#4627 incident label signature
# (`gguf`+`generation`: a real-weight generation-correctness defect). Evidence
# is a typed block the close arm can PARSE (never "fix/resolve" commit
# wording): a checked `effect observed` task box or an `Effect-Observed:
# <artifact>` line. Missing evidence leaves the issue open with a typed hold
# reason; an unreadable body fails CLOSED (never a false close on a guess).
EFFECT_UNOBSERVED_HOLD = "MODEL_DEFECT_EFFECT_UNOBSERVED"
EFFECT_UNKNOWN_HOLD = "EFFECT_EVIDENCE_UNKNOWN"
_MODEL_DEFECT_LABELS = {"model-defect", "class:model-defect", "model-correctness"}
# The #4273/#4627 regression-fixture signature: both root incidents carry
# `gguf` + `generation` — the real-weight generation-correctness family.
_MODEL_DEFECT_LABEL_SIGNATURE = frozenset({"gguf", "generation"})
_TERMINAL_CLASS_RE = re.compile(
    r"^\s*(?:terminal-)?(?:resolution|required)-class\s*:\s*effect-observed\b",
    re.IGNORECASE | re.MULTILINE)
# Typed evidence line: `Effect-Observed: <non-empty artifact reference>`.
_EFFECT_LINE_RE = re.compile(
    r"^\s*(?:effect-observed|observed-effect)\s*:\s*\S",
    re.IGNORECASE | re.MULTILINE)
# Checked acceptance box whose text records the observed effect. An unchecked
# `- [ ]` box never matches (it is a promise, not a witness — and coverage
# #3870 already holds unchecked boxes upstream).
_EFFECT_BOX_RE = re.compile(
    r"^\s*[-*]\s+\[[xX]\]\s+.*\b(?:effect\s+observed|observed\s+effect)\b",
    re.IGNORECASE | re.MULTILINE)
# `- [ ]` / `* [ ]` GitHub task-list box, unchecked (a `[x]`/`[X]` box never matches).
_UNCHECKED_BOX_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+\S", re.MULTILINE)
_SPINE_FIRST_RE = re.compile(
    r"spine[-\s]?first|first[-\s]?spine|required first(?:[-\s]spine)?\s+artifact",
    re.IGNORECASE)
_KEEP_OPEN_LABELS = {"epic"}
# Incomplete-evidence gate: the #5865 disclaimer gate reads the commit MESSAGE, but a
# benchmark-shaped resolution states its scope in the ARTIFACT the commit adds, not in
# the subject line. The witnessed harm: on 2026-08-10 this arm closed 31 benchmark
# issues (#6122 .. #6205) on commits that each added a `docs/benchmarks/` comparison
# packet whose own header reads `Status: **INCOMPLETE**` -- only the native arm and a
# tuned baseline execute, and every external/integration arm is still a zero-measurement
# placeholder. Every gate upstream passed honestly: the commit is diff-witnessed, the
# claim is `code_effect` over real source paths, the message disclaims nothing, and the
# issue body carries no epic label or unchecked box. The evidence itself was the only
# thing that said "not done", and nothing read it. The owner had already hand-reopened
# #6097 / #6102 / #6107 for exactly this before the mass close repeated it.
#
# The gate reads only files the commit touched under `docs/benchmarks/`, and it refuses
# only on a marker BOUND TO THE ISSUE BEING CLOSED -- never on a file-global mention.
# That scoping is load-bearing: `docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md`
# is the shared registry, is touched by every packet commit, and carries an INCOMPLETE
# row for dozens of OTHER capabilities plus its own INCOMPLETE spine header. A file-level
# `"INCOMPLETE" in text` test would therefore hold every benchmark close forever,
# including a genuinely finished one. Two binding forms, both measured against the live
# corpus (they cover all 31 closes with no third heuristic):
#   1. the uppercase status token on the SAME LINE as a reference to this issue --
#      `Status: **INCOMPLETE**. Issue [#6205](...)`, `**INCOMPLETE.** ... [#6165](...)
#      tracks those witnesses`, and the registry row `| ... /issues/6135) | INCOMPLETE |`;
#   2. the packet writing `#<n> remains/stays open` -- the same author-disclaimer form
#      #5865 already trusts in a commit message, reused verbatim here.
# Lowercase prose ("the comparison is incomplete") is deliberately NOT a marker on its
# own: it is narration, and form 2 already binds the packets that use it.
EVIDENCE_INCOMPLETE_HOLD = "EVIDENCE_INCOMPLETE"
# The commit's file list or a packet blob could not be read. Commit-side, so this is an
# ALLOW with an audit note rather than a hold -- same asymmetry, and for the same reason,
# as ``disclaimer_binds_closure``.
EVIDENCE_UNREADABLE_NOTE = "EVIDENCE_STATUS_UNREADABLE"
_EVIDENCE_PREFIX = "docs/benchmarks/"
_EVIDENCE_SUFFIX = ".md"
# Uppercase, word-bounded: a status TOKEN. Packet prose says "incomplete" in lower case.
_INCOMPLETE_TOKEN_RE = re.compile(r"\bINCOMPLETE\b")
# `#6205` or the URL form `.../issues/6205`, scoped to the candidate's own number.
_EVIDENCE_ISSUE_REF_TMPL = r"(?:#|/issues/){n}\b"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_capture(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    try:
        # Pin UTF-8 (what gh/git/dos emit) with errors="replace". On Windows text=True
        # otherwise defaults to the locale codec (cp1252), which raises mid-read on the
        # em-dashes / smart-quotes ubiquitous in issue bodies -- so `gh issue view --json
        # body` would crash and the coverage gate would wrongly hold every such issue as
        # COVERAGE_UNKNOWN forever (a silent close-volume leak). Decode utf-8 so every
        # gh/git read is total.
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except (OSError, UnicodeError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def gh_context_absent(stderr: str) -> bool:
    """Did a failed gh call fail because this workspace is bound to NO GitHub repo?

    That is gh saying the QUESTION is unaskable, not that the ANSWER is unknown --
    the distinction the reopen/coverage/observed-effect gates need so an absent
    tracker reads as *gate inapplicable* instead of *fail closed*. Deliberately
    matches gh's own no-repo wording only: any other failure (auth, network, rate
    limit, a 404 on a nonexistent issue, gh missing) is a genuine unknown and keeps
    failing CLOSED, so an unrecognized message always falls back to the safe side."""
    return bool(_NO_GH_REPO_RE.search(stderr or ""))


def load_audit(root: Path, audit_json: str | None, max_commits: int) -> dict[str, Any]:
    """Get the closure audit: from a provided JSON file, else run it fresh."""
    if audit_json:
        try:
            return json.loads(Path(audit_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"_error": f"could not read --audit-json: {exc}"}
    rc, out, err = run_capture(
        [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
         "--max-commits", str(max_commits)], root, timeout=300)
    try:
        return json.loads(out)
    except ValueError:
        return {"_error": (err or out or "closure audit produced no JSON").strip()[-400:]}


def open_witnessed(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i in audit.get("issues") or []:
        if i.get("bucket") != "OPEN_WITNESSED":
            continue
        wc = i.get("witnessed_commits") or i.get("resolving_commits") or []
        first = wc[0] if wc else None
        sha = (first.get("sha") if isinstance(first, dict) else first) or ""
        subject = (first.get("subject") if isinstance(first, dict) else "") or ""
        rows.append({"number": i.get("number"), "title": i.get("title") or "",
                     "sha": str(sha), "subject": str(subject)})
    rows.sort(key=lambda r: -(r["number"] or 0))
    return rows


def reverify(root: Path, sha: str) -> dict[str, Any]:
    """Re-ask the oracle at close time — do NOT trust the audit's bucket."""
    if not sha:
        return {"witness_ok": False, "reason": "no witnessing sha"}
    rc, out, err = run_capture(
        ["dos", "commit-audit", sha, "--workspace", str(root), "--json"],
        root, timeout=60)
    # `dos commit-audit --json` emits a JSON ARRAY (one row per audited sha).
    doc: dict[str, Any] = {}
    try:
        parsed = json.loads(out.strip()) if out.strip() else []
    except ValueError:
        parsed = []
    if isinstance(parsed, list):
        # prefer the row whose sha matches; else the first row.
        doc = next((r for r in parsed if isinstance(r, dict)
                    and str(r.get("sha")) and str(sha).startswith(str(r.get("sha")))),
                   parsed[0] if parsed and isinstance(parsed[0], dict) else {})
    elif isinstance(parsed, dict):
        doc = parsed
    verdict = str(doc.get("verdict") or "")
    witness = str(doc.get("witness") or "")
    ok = verdict.upper() == "OK" and witness == WITNESS_OK
    # Carry the audit's claim kind + code-path evidence so the caller can check
    # the claim BINDS resolution (#2998). touches_code is tri-state: None when
    # the audit row carried no file lists (legacy shape — nothing to bind on).
    claim_kind = str(doc.get("claim_kind") or "") or None
    file_keys = [k for k in ("source_files", "test_files") if k in doc]
    touches_code = (any(bool(doc.get(k)) for k in file_keys)
                    if file_keys else None)
    return {"witness_ok": ok, "verdict": verdict or None, "witness": witness or None,
            "claim_kind": claim_kind, "touches_code": touches_code,
            "reason": None if ok else f"commit-audit verdict={verdict or '?'} witness={witness or '?'}"}


def issue_is_docs_rung(title: str) -> bool:
    """A docs-shaped issue MAY be resolved by a doc claim (#2998 carve-out)."""
    t = (title or "").lstrip().lower()
    return t.startswith(("docs(", "docs:", "doc(", "doc:"))


def claim_binds_resolution(rv: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does the re-verified claim KIND bind resolution of this issue? (#2998)

    audit-OK + subject-references-(#N) is NOT resolves-#N: a docs/triage commit
    is diff-witnessed for its own doc claim yet witnesses nothing about #N's
    feature. Bind iff the claim is a code/test claim over non-doc paths, or the
    issue itself is a docs rung. Unknown kind (legacy audit shape) fails OPEN —
    the hold fires only on a KNOWN non-binding claim, it never wedges the arm.
    """
    kind = rv.get("claim_kind")
    if not kind:
        return True, None
    if kind in RESOLVING_CLAIM_KINDS:
        touches = rv.get("touches_code")
        if touches is None or touches:
            return True, None
        return False, (f"{NONRESOLVING_HOLD}: {kind} claim with a docs-only "
                       f"diff cannot resolve #{row.get('number')}")
    if issue_is_docs_rung(str(row.get("title") or "")):
        return True, None
    return False, (f"{NONRESOLVING_HOLD}: {kind} claim (docs/triage-shaped) "
                   f"cannot resolve a non-docs issue")


def commit_body(root: Path, sha: str) -> str | None:
    """The witness commit's full message (subject + body), or None if unreadable."""
    if not sha:
        return None
    rc, out, _ = run_capture(["git", "show", "-s", "--format=%B", sha], root, timeout=15)
    return None if rc != 0 else (out or "")


def _excerpt(text: str, match: "re.Match[str]", pad: int = 60) -> str:
    """The matched disclaimer plus surrounding words, flattened onto one line.

    The hold reason has to be checkable by a human reading the plan without going back
    to git, so it quotes the author's own sentence rather than just naming the marker."""
    start = max(0, match.start() - pad)
    end = min(len(text), match.end() + pad)
    return " ".join(text[start:end].split())


def commit_disclaims_resolution(number: Any, body: str) -> tuple[bool, str | None]:
    """Pure commit-message -> (binds, hold_reason) author-disclaimer decision (#5865).

    Holds (returns False) only when the author EXPLICITLY wrote that this commit does not
    resolve the issue -- "The issue stays open", "this commit does not claim it", "no
    artifact is claimed", "would be claiming an integration that has no witness". Those
    are statements of fact by the one person who knew the scope; no diff-shape or
    label heuristic can outvote them, so the gate applies at every rung.

    A NUMBERED disclaimer is scoped to the issue it names: "#5847 stays OPEN" holds a
    close of #5847 and nothing else. Everything else binds -- the marker set is
    high-precision by measurement (7/1500 recent commits), never a general read of
    hedging language."""
    text = body or ""
    for label, rx in _DISCLAIM_MARKERS:
        m = rx.search(text)
        if m:
            return False, (
                f"{DISCLAIMED_HOLD}: the witness commit's own message disclaims "
                f"resolution ({label}) -- \"{_excerpt(text, m)}\"; a commit that says it "
                f"does not resolve #{number} cannot witness that it does")
    if number is not None:
        rx = re.compile(_DISCLAIM_THIS_ISSUE_TMPL.format(n=re.escape(str(number))),
                        re.IGNORECASE)
        m = rx.search(text)
        if m:
            return False, (
                f"{DISCLAIMED_HOLD}: the witness commit's own message says #{number} "
                f"stays open -- \"{_excerpt(text, m)}\"")
    return True, None


def disclaimer_binds_closure(root: Path, row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does the witness commit's own message permit closing this issue? (#5865)

    Unlike the issue-side gates, an unreadable input here ALLOWS (with an audit note)
    instead of failing closed, and that asymmetry is deliberate rather than a lapse: this
    gate only ever ADDS a refusal on POSITIVE evidence -- a disclaimer the author wrote.
    An absent or unreadable commit message is not evidence of a disclaimer. Failing
    closed on it would convert every workspace where `git show` cannot resolve the
    witness into a blanket hold, which is a large unwitnessed behavior change, and it
    would strand exactly the rows the pre-existing gates already decide correctly. The
    safety property is unchanged: a row this gate abstains on is still subject to every
    other gate, precisely as it was before #5865."""
    body = commit_body(root, str(row.get("sha") or ""))
    if body is None:
        return True, (f"{DISCLAIM_UNREADABLE_NOTE}: could not read the witness commit "
                      f"message for {str(row.get('sha') or '?')[:10]}, so there is no "
                      "disclaimer evidence either way")
    return commit_disclaims_resolution(row.get("number"), body)


def commit_touched_paths(root: Path, sha: str) -> list[str] | None:
    """Repo-relative paths the witness commit changed, or None if unreadable."""
    if not sha:
        return None
    # diff-tree lists one commit's changed paths without the `--name-only` / `-s`
    # option clash `git show` trips on (same plumbing tools/dispatch_parity_canary.py
    # settled on). Slashes are normalized so the Windows checkouts this loop runs in
    # compare against `docs/benchmarks/` the same way a POSIX one does.
    rc, out, _ = run_capture(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        root, timeout=15)
    if rc != 0:
        return None
    return [ln.strip().replace("\\", "/") for ln in (out or "").splitlines() if ln.strip()]


def file_at_commit(root: Path, sha: str, path: str) -> str | None:
    """One file's blob AS OF the witness commit, or None if unreadable.

    Read at the sha, never from the worktree: the loop runs in a shared checkout where
    HEAD moves under it, and the question is what evidence THIS commit pointed at."""
    if not sha or not path:
        return None
    rc, out, _ = run_capture(["git", "show", f"{sha}:{path}"], root, timeout=15)
    return None if rc != 0 else (out or "")


def evidence_declares_incomplete(number: Any, path: str,
                                 text: str) -> tuple[bool, str | None]:
    """Pure packet-body -> (binds, hold_reason) evidence-status decision.

    Holds (returns False) only when the packet marks ITS OWN status incomplete FOR THIS
    ISSUE -- an `INCOMPLETE` status token on the same line as a `#<n>` / `/issues/<n>`
    reference, or the packet writing `#<n> remains open`. A marker naming some other
    capability's issue, or the shared registry's own spine header, never binds: see the
    `_EVIDENCE_PREFIX` note for why file-global matching would hold every close."""
    body = text or ""
    if number is None:
        return True, None
    ref = re.compile(_EVIDENCE_ISSUE_REF_TMPL.format(n=re.escape(str(number))))
    for line in body.splitlines():
        token = _INCOMPLETE_TOKEN_RE.search(line)
        if token and ref.search(line):
            return False, (
                f"{EVIDENCE_INCOMPLETE_HOLD}: the evidence this commit points at, "
                f"`{path}`, marks its own status INCOMPLETE for #{number} -- "
                f"\"{_excerpt(line, token)}\"; a packet that says it is not finished "
                f"cannot witness that #{number} is")
    stays_open = re.compile(_DISCLAIM_THIS_ISSUE_TMPL.format(n=re.escape(str(number))),
                            re.IGNORECASE).search(body)
    if stays_open:
        return False, (
            f"{EVIDENCE_INCOMPLETE_HOLD}: the evidence this commit points at, "
            f"`{path}`, says #{number} stays open -- \"{_excerpt(body, stays_open)}\"")
    return True, None


def evidence_binds_closure(root: Path, row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does the evidence the witness commit added permit closing this issue?

    A commit that touches no `docs/benchmarks/` packet binds immediately -- this gate
    adds a refusal on POSITIVE evidence only (a packet that declares itself unfinished)
    and never converts an ordinary commit into a hold. For the same reason, and matching
    ``disclaimer_binds_closure``, an unreadable file list or blob ALLOWS with an audit
    note rather than failing closed: absent evidence is not evidence of incompleteness,
    and every other gate still applies to the row."""
    sha = str(row.get("sha") or "")
    paths = commit_touched_paths(root, sha)
    if paths is None:
        return True, (f"{EVIDENCE_UNREADABLE_NOTE}: could not list the files changed by "
                      f"{sha[:10] or '?'}, so there is no evidence-status reading "
                      "either way")
    packets = [p for p in paths
               if p.startswith(_EVIDENCE_PREFIX) and p.endswith(_EVIDENCE_SUFFIX)]
    unreadable: list[str] = []
    for path in packets:
        text = file_at_commit(root, sha, path)
        if text is None:
            # Deleted-by-this-commit or otherwise unresolvable: note it, keep checking
            # the rest -- one unreadable packet must not mask a readable INCOMPLETE one.
            unreadable.append(path)
            continue
        binds, hold = evidence_declares_incomplete(row.get("number"), path, text)
        if not binds:
            return False, hold
    if unreadable:
        return True, (f"{EVIDENCE_UNREADABLE_NOTE}: could not read "
                      f"{', '.join(unreadable[:3])} at {sha[:10]}, so their declared "
                      "status was not checked")
    return True, None


def classify_coverage(number: Any, body: str,
                      labels: set[str]) -> tuple[bool, str | None]:
    """Pure body/label -> (binds, hold_reason) coverage decision (#3870).

    Holds (returns False) only on an EXPLICIT multi-part marker: an `epic` label, an
    unchecked task-list box (an epic's child-issue checklist or an unfinished
    acceptance criterion), or a spine-first / first-artifact work-unit. Everything
    else binds. This is deliberately high-precision -- it does NOT hold on a plain
    multi-bullet ``## In scope`` (which would strand ordinary single-scope issues),
    only on markers an author placed to say 'this ticket is not done in one commit'.
    """
    hit = set(labels) & _KEEP_OPEN_LABELS
    if hit:
        return False, (f"{PARTIAL_HOLD}: #{number} is labelled {sorted(hit)} "
                       "(a parent/epic groups child issues; one commit does not close it)")
    if _UNCHECKED_BOX_RE.search(body or ""):
        return False, (f"{PARTIAL_HOLD}: #{number} has unchecked task-list box(es) "
                       "-- declared work remains, a single commit does not close it")
    if _SPINE_FIRST_RE.search(body or ""):
        return False, (f"{PARTIAL_HOLD}: #{number} is a spine-first / first-of-many "
                       "artifact -- the resolving commit is step one, not the close")
    return True, None


def fetch_issue_meta(root: Path, number: Any) -> dict[str, Any] | None:
    """The issue's body + lower-cased label names, or None when GitHub can't be read.

    None is the fail-SAFE signal: the coverage gate treats an unreadable body as
    COVERAGE_UNKNOWN and keeps the issue open (a transient gh error self-heals on the
    next tick) rather than closing something it could not inspect.

    ``NO_GITHUB_CONTEXT`` (identity-compared) is the distinct third answer: gh
    reported there is no repository bound to this workspace, so there is no issue
    body in existence to inspect and the body-reading gates are INAPPLICABLE rather
    than unknown -- see GH_CONTEXT_ABSENT_NOTE."""
    if number is None:
        return None
    rc, out, err = run_capture(
        ["gh", "issue", "view", str(number), "--json", "body,labels"], root, timeout=30)
    if rc != 0 and gh_context_absent(err):
        return NO_GITHUB_CONTEXT
    if rc != 0 or not out:
        # rc!=0, or a rc-0 call that still yielded no stdout (a rare gh hiccup):
        # unreadable -> COVERAGE_UNKNOWN (hold), never crash the live close tick.
        return None
    try:
        doc = json.loads(out.strip() or "{}")
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    labels = {str((lbl or {}).get("name") or "").strip().lower()
              for lbl in (doc.get("labels") or []) if isinstance(lbl, dict)}
    return {"body": str(doc.get("body") or ""), "labels": labels}


def coverage_binds_closure(root: Path, row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does a single resolving commit actually close this whole issue? (#3870)

    Fetches the issue's live body/labels and classifies; an unreadable body holds as
    COVERAGE_UNKNOWN -- never a silent close of an issue we could not inspect. A
    workspace bound to no GitHub repository has no body to read at all, so the gate
    is INAPPLICABLE there (allowed, with an audit note), not unknown."""
    number = row.get("number")
    meta = fetch_issue_meta(root, number)
    if meta is NO_GITHUB_CONTEXT:
        return True, (f"{GH_CONTEXT_ABSENT_NOTE}: no GitHub repository is bound to "
                      "this workspace, so the #3870 coverage gate is not applicable")
    if meta is None:
        return False, (f"{COVERAGE_UNKNOWN_HOLD}: could not read #{number} "
                       "body/labels to confirm full coverage")
    return classify_coverage(number, meta["body"], meta["labels"])


def issue_requires_observed_effect(body: str, labels: set[str]) -> bool:
    """Does this issue's resolution contract terminate at *effect observed*? (#4747)

    High-precision markers only: an explicit model-defect label, the
    #4273/#4627 incident label signature (`gguf`+`generation`), or a typed
    terminal-class declaration in the body. Ordinary issues never require the
    evidence block, so the gate cannot strand the general close arm."""
    lab = {str(lbl or "").strip().lower() for lbl in (labels or set())}
    if lab & _MODEL_DEFECT_LABELS:
        return True
    if _MODEL_DEFECT_LABEL_SIGNATURE <= lab:
        return True
    return bool(_TERMINAL_CLASS_RE.search(body or ""))


def has_observed_effect_evidence(body: str) -> bool:
    """Is there a TYPED observed-effect evidence block in the body? (#4747)

    Evidence is parseable, never inferred from commit wording: an
    `Effect-Observed: <artifact>` line or a checked `effect observed` task box.
    The terminal-class DECLARATION (`Resolution-Class: effect-observed`) is a
    requirement marker, not evidence — it deliberately does not match here."""
    b = body or ""
    return bool(_EFFECT_LINE_RE.search(b) or _EFFECT_BOX_RE.search(b))


def classify_observed_effect(number: Any, body: str,
                             labels: set[str]) -> tuple[bool, str | None]:
    """Pure body/label -> (binds, hold_reason) observed-effect decision (#4747).

    A model-correctness defect binds closure only when the body carries the
    typed observed-effect evidence block; a diagnostic/instrumentation commit
    satisfying an earlier resolution class must never close it. Everything
    without a model-defect marker binds (the gate is deliberately narrow)."""
    if not issue_requires_observed_effect(body, labels):
        return True, None
    if has_observed_effect_evidence(body):
        return True, None
    return False, (
        f"{EFFECT_UNOBSERVED_HOLD}: #{number} is a model-correctness defect "
        "(terminal resolution class: effect observed) with no observed-effect "
        "evidence in the body -- record an independent real artifact via an "
        "'Effect-Observed: <artifact>' line or a checked 'effect observed' "
        "task box; instrumentation/fix commits alone do not close it")


def observed_effect_binds_closure(root: Path,
                                  row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does observed-effect evidence permit closing this issue? (#4747)

    Fetches the issue's live body/labels and classifies; an unreadable body
    holds as EFFECT_EVIDENCE_UNKNOWN -- never a silent close of a possible
    model-defect issue we could not inspect. A workspace bound to no GitHub
    repository has no body to read at all, so the gate is INAPPLICABLE there
    (allowed, with an audit note), not unknown."""
    number = row.get("number")
    meta = fetch_issue_meta(root, number)
    if meta is NO_GITHUB_CONTEXT:
        return True, (f"{GH_CONTEXT_ABSENT_NOTE}: no GitHub repository is bound to "
                      "this workspace, so the #4747 observed-effect gate is not "
                      "applicable")
    if meta is None:
        return False, (f"{EFFECT_UNKNOWN_HOLD}: could not read #{number} "
                       "body/labels to check observed-effect evidence")
    return classify_observed_effect(number, meta["body"], meta["labels"])


def origin_main_resolvable(root: Path) -> bool:
    """Best-effort refresh + presence check for the origin/main remote-tracking ref.

    A close is only DURABLE once its resolving commit is reachable from what the
    REMOTE has. A commit that exists only in the local shared multi-session tree can
    be orphaned by a peer's reset/merge AFTER we close the issue -- exactly how #350
    became closed-but-undelivered (the close arm witnessed the commit, closed the
    issue, then a peer git op moved main off it). We refresh origin/main so the
    ancestry gate below is against current remote truth; a failed fetch (offline)
    falls back to the last-known origin/main. Returns False when origin/main cannot
    be resolved at all, in which case the caller degrades to "don't gate" rather than
    wedging the loop on a repo with no upstream.
    """
    run_capture(["git", "fetch", "origin", "main"], root, timeout=30)  # best effort
    rc, _, _ = run_capture(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main"], root, timeout=15)
    return rc == 0


def reachable_from_origin(root: Path, sha: str) -> bool:
    """Is `sha` an ancestor of origin/main -- i.e. durably pushed, not just local?"""
    if not sha:
        return False
    rc, _, _ = run_capture(
        ["git", "merge-base", "--is-ancestor", sha, "origin/main"], root, timeout=15)
    return rc == 0


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp to an aware datetime, or None if unparseable.

    Handles both the GitHub timeline shape (``2026-07-11T22:52:00Z``) and git's
    ``%cI`` strict-ISO shape (``2026-07-11T23:05:12+00:00``). A trailing ``Z`` is
    normalized to ``+00:00`` (``fromisoformat`` only learned ``Z`` in 3.11); a
    naive stamp is assumed UTC so it still compares against aware stamps rather
    than raising ``can't compare offset-naive and offset-aware``."""
    ts = (ts or "").strip()
    if not ts:
        return None
    if ts.endswith(("Z", "z")):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def latest_reopen_ts(root: Path, number: Any) -> tuple[bool | None, datetime | None]:
    """(read_status, most-recent ``reopened`` timeline datetime | None) for #4374.

    ``read_status`` is TRI-state, deliberately not a plain bool:
      * ``True``  -- the timeline was read. A successful read with no ``reopened``
        event returns ``(True, None)``.
      * ``False`` -- gh errored while a repository WAS resolvable: a genuine
        unknown, so the caller fails CLOSED (holds the close) rather than guessing
        the issue was never reopened.
      * ``None``  -- gh reported there is no repository to ask at all (a workspace
        with no git remote). There is no timeline in existence to supersede the
        witness, so the caller treats the gate as INAPPLICABLE. Safe because a
        ``gh issue close`` in that same workspace cannot run either.

    ``gh api`` substitutes ``{owner}``/``{repo}`` from the repo, and ``--paginate``
    walks every page, so a reopen on any page is seen; the per-page ``--jq`` emits
    one ``created_at`` line per reopened event and we take the max (lexical==
    chronological only within a format, so parse then max)."""
    if number is None:
        return True, None
    rc, out, err = run_capture(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/issues/{number}/timeline",
         "--paginate", "--jq", '.[] | select(.event=="reopened") | .created_at'],
        root, timeout=30)
    if rc != 0:
        return (None if gh_context_absent(err) else False), None
    stamps = [t for line in out.splitlines() if (t := _parse_iso(line))]
    return True, (max(stamps) if stamps else None)


EFFECT_REVERTED_HOLD = "EFFECT_REVERTED_AT_TIP"
EFFECT_SURVIVAL_UNKNOWN_HOLD = "EFFECT_SURVIVAL_UNKNOWN"


def closure_tip(root: Path) -> str | None:
    """Resolve the exact remote trunk tip used for closure admission."""
    rc, out, _ = run_capture(
        ["git", "rev-parse", "--verify", "origin/main^{commit}"], root, timeout=15)
    tip = (out or "").strip()
    return tip if rc == 0 and tip else None


def effect_survives_at_tip(root: Path, candidate: str,
                           tip: str) -> tuple[bool, str | None]:
    """Prove candidate-touched paths have not all returned to the parent state."""
    if not candidate or not tip:
        return False, (f"{EFFECT_SURVIVAL_UNKNOWN_HOLD}: candidate={candidate or '?'} "
                       f"tip={tip or '?'}; explicit candidate and tip are required")
    rc, out, err = run_capture(
        ["git", "rev-parse", "--verify", f"{candidate}^{{commit}}^"], root, timeout=15)
    parent = (out or "").strip()
    if rc != 0 or not parent:
        return False, (f"{EFFECT_SURVIVAL_UNKNOWN_HOLD}: candidate={candidate} tip={tip}; "
                       f"candidate parent unreadable: {(err or 'git rev-parse failed').strip()}")
    paths = commit_touched_paths(root, candidate)
    if not paths:
        return False, (f"{EFFECT_SURVIVAL_UNKNOWN_HOLD}: candidate={candidate} tip={tip}; "
                       "candidate changed-path set is empty or unreadable")
    rc, _, err = run_capture(
        ["git", "diff", "--quiet", parent, tip, "--", *paths], root, timeout=30)
    if rc == 0:
        return False, (f"{EFFECT_REVERTED_HOLD}: candidate={candidate} tip={tip}; "
                       "all candidate-touched paths equal the candidate parent at closure tip")
    if rc == 1:
        return True, None
    return False, (f"{EFFECT_SURVIVAL_UNKNOWN_HOLD}: candidate={candidate} tip={tip}; "
                   f"tip comparison failed: {(err or f'git diff rc={rc}').strip()}")


def commit_committer_ts(root: Path, sha: str) -> datetime | None:
    """The committer date of ``sha`` as an aware datetime, or None if unreadable."""
    if not sha:
        return None
    rc, out, _ = run_capture(
        ["git", "show", "-s", "--format=%cI", sha], root, timeout=15)
    if rc != 0:
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    return _parse_iso(first)


def reopen_blocks_close(root: Path, row: dict[str, Any]) -> tuple[bool, str | None]:
    """Does an unsuperseded reopen forbid re-closing this issue? (#4374)

    The close-resolved arm cites a witnessing commit; an auto-reclose may only
    override a ``reopened`` event if a commit landed AFTER it. Returns
    ``(allowed, hold_reason)``:
      - ``(True, None)``  -> never reopened, or a commit landed since the reopen
      - ``(True, GH_CONTEXT_ABSENT: ...)`` -> there is no GitHub repository bound to
        this workspace, so no timeline exists to supersede anything: the gate is
        INAPPLICABLE (allowed, with an audit note), not unknown. See
        GH_CONTEXT_ABSENT_NOTE above for why this cannot yield a wrong close.
      - ``(False, REOPENED_NO_NEW_COMMIT: ...)`` -> reopened with no newer commit;
        re-closing would silently undo the correction (the witnessed #4350 harm)
      - ``(False, REOPEN_UNKNOWN: ...)`` -> timeline (or the witness commit's date)
        unreadable in a workspace that DOES resolve a repo; fail CLOSED, since we
        cannot prove no reopen supersedes it.
    """
    number = row.get("number")
    read_ok, reopen = latest_reopen_ts(root, number)
    if read_ok is None:
        return True, (f"{GH_CONTEXT_ABSENT_NOTE}: no GitHub repository is bound to "
                      "this workspace, so the #4374 reopen gate is not applicable")
    if not read_ok:
        return False, (f"{REOPEN_UNKNOWN_HOLD}: could not read #{number} timeline "
                       "to confirm no reopen supersedes the witness")
    if reopen is None:
        return True, None  # never reopened -> the witness stands
    commit_ts = commit_committer_ts(root, str(row.get("sha") or ""))
    if commit_ts is None:
        return False, (f"{REOPEN_UNKNOWN_HOLD}: #{number} was reopened at "
                       f"{reopen.isoformat()} but the witness commit date is unreadable")
    if commit_ts <= reopen:
        return False, (
            f"{REOPEN_NO_NEW_COMMIT_HOLD}: #{number} reopened at {reopen.isoformat()} "
            f"with no commit landed since (witness {str(row.get('sha'))[:10]} dated "
            f"{commit_ts.isoformat()}); a reopen with no new work stays open")
    return True, None


def readback_state(root: Path, number: Any) -> dict[str, Any]:
    """The AUTHORITATIVE GitHub state of one issue after a close attempt (#2641).

    ``gh issue close`` returning rc 0 means the *command* ran, not that the issue is
    durably CLOSED: a close that raced a reopen (or silently no-ops on an
    already-reopened issue) leaves it OPEN with ``stateReason`` REOPENED. The
    closure loop must count a close only when the state GitHub reports back is
    CLOSED, so a reopened issue is never tallied as the loop's durable work.

    Returns ``{"state": <UPPER>, "state_reason": <str>}``. Fail-open in the
    conservative direction: an unreadable state (gh error / non-JSON) returns ``{}``
    and the caller treats it as UNCONFIRMED — surfaced as ``close_not_persistent``
    and not counted — rather than trusting the bare exit code."""
    if number is None:
        return {}
    rc, out, _ = run_capture(
        ["gh", "issue", "view", str(number), "--json", "state,stateReason"],
        root, timeout=30)
    if rc != 0:
        return {}
    try:
        doc = json.loads(out.strip() or "{}")
    except ValueError:
        return {}
    if not isinstance(doc, dict):
        return {}
    return {"state": str(doc.get("state") or "").upper(),
            "state_reason": str(doc.get("stateReason") or "")}


def close_comment(row: dict[str, Any]) -> str:
    subj = row.get("subject") or "resolving commit"
    return (f"Resolved by `{row['sha'][:10]}` ({subj}). Closed by the DOS dispatch "
            f"loop's close-resolved arm, witnessed via `dos commit-audit` "
            f"(verdict OK / diff-witnessed). Reopen if this does not fully resolve it.")


def close_cmd(row: dict[str, Any]) -> list[str]:
    return ["gh", "issue", "close", str(row["number"]), "--comment", close_comment(row)]


def note_gate(item: dict[str, Any], msg: str | None) -> None:
    """Record a gate note on a row that was ALLOWED through anyway.

    A gate that declined to apply (GH_CONTEXT_ABSENT) must not be invisible: the
    original defect was precisely that a gate's disposition was not legible from the
    plan. The note rides on the row so `--json` consumers and any later audit can see
    which gates abstained and why, without changing the row's action."""
    if msg:
        item.setdefault("gate_notes", []).append(msg)


def evaluate(root: Path, *, limit: int, live: bool, audit_json: str | None,
             max_commits: int, require_pushed: bool = True) -> dict[str, Any]:
    audit = load_audit(root, audit_json, max_commits)
    if audit.get("_error"):
        return {"schema": SCHEMA, "ok": False, "verdict": "ERROR",
                "reason": audit["_error"], "planned": [], "results": []}
    candidates = open_witnessed(audit)[:limit]
    # Durability gate: only close an issue whose resolving commit is reachable from
    # origin/main (durably pushed), never one that lives only in the local shared
    # tree. If origin/main can't be resolved (no upstream), degrade to "don't gate"
    # so a repo without a remote still closes -- the gate guards against orphaning,
    # it must not wedge the loop.
    gate_active = require_pushed and origin_main_resolvable(root)
    planned, results = [], []
    closed = skipped = skipped_nonresolving = skipped_unpushed = failed = 0
    skipped_disclaimed = skipped_incomplete_evidence = 0
    skipped_effect_reverted = skipped_effect_survival_unknown = 0
    close_not_persistent = already_counted = 0
    skipped_partial = skipped_coverage_unknown = 0
    skipped_reopened = skipped_reopen_unknown = 0
    skipped_effect_unobserved = skipped_effect_unknown = 0
    # Unique issue IDs durably closed THIS run — a repeated close tick on an issue
    # already counted here does not inflate the tally (#2641, done condition 3).
    counted: set[int] = set()
    for row in candidates:
        rv = reverify(root, row["sha"])
        item = {**row, **rv, "command": close_cmd(row)}
        planned.append(item)
        if not rv["witness_ok"]:
            item["action"] = "skip_unwitnessed"
            skipped += 1
            results.append(item)
            continue
        binds, hold_reason = claim_binds_resolution(rv, row)
        if not binds:
            item["action"] = "skip_nonresolving"
            item["reason"] = hold_reason
            skipped_nonresolving += 1
            results.append(item)
            continue
        # #5865: a commit whose own message disclaims resolution -- "The issue stays
        # open", "this commit does not claim it", "would be claiming an integration that
        # has no witness" -- can never witness that resolution, at ANY rung. Runs here,
        # ahead of every gh probe, because it is a local git read and a self-disclaimed
        # witness needs no tracker round-trip to refuse. An unreadable message ALLOWS
        # with an audit note (see disclaimer_binds_closure): the gate adds refusals on
        # positive evidence only, it never converts silence into a hold.
        undisclaimed, disclaim_hold = disclaimer_binds_closure(root, row)
        if not undisclaimed:
            item["action"] = "skip_disclaimed"
            item["reason"] = disclaim_hold
            skipped_disclaimed += 1
            results.append(item)
            continue
        note_gate(item, disclaim_hold)  # gate abstained (commit message unreadable)
        # A commit whose own EVIDENCE declares itself unfinished cannot witness a
        # resolution either. Runs beside the #5865 message gate -- both are local git
        # reads over the commit, both refuse only on positive author-written evidence,
        # and both belong ahead of every gh probe. This is the gate the 2026-08-10 mass
        # close of #6122 .. #6205 needed: 31 `docs/benchmarks/` packets headed
        # `Status: **INCOMPLETE**` closed their own issues. An unreadable packet ALLOWS
        # with an audit note (see evidence_binds_closure); silence is never a hold.
        evidence_ok, evidence_hold = evidence_binds_closure(root, row)
        if not evidence_ok:
            item["action"] = "skip_incomplete_evidence"
            item["reason"] = evidence_hold
            skipped_incomplete_evidence += 1
            results.append(item)
            continue
        note_gate(item, evidence_hold)  # gate abstained (evidence unreadable)
        if gate_active and not reachable_from_origin(root, row["sha"]):
            item["action"] = "skip_unpushed"
            item["reason"] = "resolving commit not on origin/main yet (not durable)"
            skipped_unpushed += 1
            results.append(item)
            continue
        tip = closure_tip(root) if gate_active else None
        if gate_active:
            survives, survival_reason = effect_survives_at_tip(
                root, row.get("sha", ""), tip or "")
            item["closure_tip"] = tip
            if not survives:
                item["reason"] = survival_reason
                if survival_reason and survival_reason.startswith(EFFECT_REVERTED_HOLD):
                    item["action"] = "skip_effect_reverted"
                    skipped_effect_reverted += 1
                else:
                    item["action"] = "skip_effect_survival_unknown"
                    skipped_effect_survival_unknown += 1
                results.append(item)
                continue
        # #4374: an auto-reclose may not override a `reopened` event unless a commit
        # landed AFTER it. The arm cites a witnessing commit; if that commit predates
        # the most recent reopen, re-closing silently undoes a correction reopen (and
        # in the witnessed #4350 case re-marked a BROKEN main "resolved"). Read-only
        # timeline probe, runs even in dry-run so the plan reflects the hold; an
        # unreadable timeline fails CLOSED (skip_reopen_unknown), never a false close.
        allowed, reopen_hold = reopen_blocks_close(root, row)
        if not allowed:
            unknown = str(reopen_hold or "").startswith(REOPEN_UNKNOWN_HOLD)
            item["action"] = "skip_reopen_unknown" if unknown else "skip_reopened"
            item["reason"] = reopen_hold
            if unknown:
                skipped_reopen_unknown += 1
            else:
                skipped_reopened += 1
            results.append(item)
            continue
        note_gate(item, reopen_hold)  # gate abstained (no GitHub context)
        # #3870: a diff-witnessed commit closes an issue only when the issue is not
        # EXPLICITLY multi-part. A read-only body/label probe runs even in dry-run so
        # the plan reflects the hold; an unreadable body holds (COVERAGE_UNKNOWN) and
        # never false-closes an issue we could not inspect.
        covers, cover_hold = coverage_binds_closure(root, row)
        if not covers:
            unknown = str(cover_hold or "").startswith(COVERAGE_UNKNOWN_HOLD)
            item["action"] = "skip_coverage_unknown" if unknown else "skip_partial"
            item["reason"] = cover_hold
            if unknown:
                skipped_coverage_unknown += 1
            else:
                skipped_partial += 1
            results.append(item)
            continue
        note_gate(item, cover_hold)  # gate abstained (no GitHub context)
        # #4747: a model-correctness defect closes only on OBSERVED-EFFECT
        # evidence -- an independent real artifact showing the original symptom
        # gone -- never on an instrumentation/diagnostic commit satisfying an
        # earlier resolution class (the #4273/#4627 harm). Read-only body/label
        # probe, runs even in dry-run so the plan reflects the typed hold; an
        # unreadable body fails CLOSED (skip_effect_unknown).
        effect_ok, effect_hold = observed_effect_binds_closure(root, row)
        if not effect_ok:
            unknown = str(effect_hold or "").startswith(EFFECT_UNKNOWN_HOLD)
            item["action"] = ("skip_effect_unknown" if unknown
                              else "skip_effect_unobserved")
            item["reason"] = effect_hold
            if unknown:
                skipped_effect_unknown += 1
            else:
                skipped_effect_unobserved += 1
            results.append(item)
            continue
        note_gate(item, effect_hold)  # gate abstained (no GitHub context)
        if not live:
            item["action"] = "would_close"
            results.append(item)
            continue
        number = row.get("number")
        rc, out, err = run_capture(close_cmd(row), root, timeout=60)
        item["returncode"] = rc
        if rc != 0:
            item["action"] = "close_failed"
            item["error"] = (err or out).strip()[-300:]
            failed += 1
            results.append(item)
            continue
        # #2641: rc 0 is not proof of a durable close. Read the authoritative state
        # back and count the close ONLY when GitHub reports CLOSED; an issue still or
        # again OPEN (e.g. stateReason REOPENED) is a distinct, non-persistent event
        # and is never tallied as the loop's durable work.
        state = readback_state(root, number)
        item["state_after"] = state.get("state") or None
        item["state_reason"] = state.get("state_reason") or None
        if state.get("state") != DURABLE_CLOSED_STATE:
            item["action"] = CLOSE_NOT_PERSISTENT
            item["reason"] = (
                f"gh reports state={state.get('state') or '?'} "
                f"reason={state.get('state_reason') or '?'} after close attempt; "
                "not a durable closure")
            close_not_persistent += 1
            results.append(item)
            continue
        # Authoritative state is CLOSED. Count once per issue per run: a repeated
        # close of an issue already counted this run (with no intervening reopen —
        # that path is caught above) must not inflate closed / closed_by_loop_total.
        if isinstance(number, int) and number in counted:
            item["action"] = CLOSE_ALREADY_COUNTED
            item["reason"] = f"#{number} already durably closed this run"
            already_counted += 1
            results.append(item)
            continue
        item["action"] = "closed"
        if isinstance(number, int):
            counted.add(number)
        closed += 1
        results.append(item)
    ok = failed == 0 and (live or bool(candidates))
    return {
        "schema": SCHEMA, "ok": ok,
        "verdict": ("CLOSED" if live and closed else
                    "PLANNED" if not live else "NO_CLOSES"),
        "live": live, "limit": limit,
        "candidates_total": len(open_witnessed(audit)),
        "planned_count": len(planned),
        "pushed_gate": ("active" if gate_active else
                        "disabled" if not require_pushed else "no-origin-ref"),
        "counts": {"closed": closed, "would_close": sum(
            1 for r in results if r.get("action") == "would_close"),
            "skipped_unwitnessed": skipped,
            "skipped_nonresolving": skipped_nonresolving,
            "skipped_disclaimed": skipped_disclaimed,
            "skipped_incomplete_evidence": skipped_incomplete_evidence,
            "skipped_partial": skipped_partial,
            "skipped_coverage_unknown": skipped_coverage_unknown,
            "skipped_reopened": skipped_reopened,
            "skipped_reopen_unknown": skipped_reopen_unknown,
            "skipped_effect_unobserved": skipped_effect_unobserved,
            "skipped_effect_unknown": skipped_effect_unknown,
            "skipped_unpushed": skipped_unpushed,
            "skipped_effect_reverted": skipped_effect_reverted,
            "skipped_effect_survival_unknown": skipped_effect_survival_unknown,
            "close_not_persistent": close_not_persistent,
            "already_counted": already_counted,
            "failed": failed},
        # The unique issue IDs this run drove to a readback-confirmed CLOSED state —
        # the honest durable-closure set the progress ledger should tally (#2641).
        "closed_numbers": sorted(counted),
        "closure_rate_before": audit.get("closure_rate"),
        "results": results,
    }


def render(p: dict[str, Any]) -> str:
    c = p.get("counts") or {}
    lines = [f"resolve-witnessed: {p.get('verdict')} ({'ok' if p.get('ok') else 'action'})  "
             f"live={p.get('live')}  candidates={p.get('candidates_total')} "
             f"planned={p.get('planned_count')}"]
    if p.get("results"):
        lines.append("  issue   sha        audit                  decision  reason")
    for r in p.get("results") or []:
        action = str(r.get("action") or "")
        audit = f"{r.get('verdict') or '?'}/{r.get('witness') or '?'}"
        decision = close_decision(action)
        reason = close_reason(action, r)
        lines.append(f"  #{r.get('number')!s:<6} {str(r.get('sha',''))[:10]:<10} "
                     f"{audit:<22} {decision:<9} {reason}")
    lines.append(f"  -> closed={c.get('closed')} would_close={c.get('would_close')} "
                 f"skipped={c.get('skipped_unwitnessed')} "
                 f"nonresolving={c.get('skipped_nonresolving')} "
                 f"disclaimed={c.get('skipped_disclaimed')} "
                 f"incomplete_evidence={c.get('skipped_incomplete_evidence')} "
                 f"partial={c.get('skipped_partial')} "
                 f"coverage_unknown={c.get('skipped_coverage_unknown')} "
                 f"reopened={c.get('skipped_reopened')} "
                 f"reopen_unknown={c.get('skipped_reopen_unknown')} "
                 f"effect_unobserved={c.get('skipped_effect_unobserved')} "
                 f"effect_unknown={c.get('skipped_effect_unknown')} "
                 f"unpushed={c.get('skipped_unpushed')} "
                 f"not_persistent={c.get('close_not_persistent')} "
                 f"already_counted={c.get('already_counted')} "
                 f"failed={c.get('failed')}  "
                 f"(gate={p.get('pushed_gate')}, closure_rate before={p.get('closure_rate_before')})")
    if not p.get("live"):
        lines.append("  DRY-RUN - re-run with --live to execute the gh closes")
    return "\n".join(lines)


def close_decision(action: str) -> str:
    if action in {"closed", "would_close"}:
        return "close"
    if action in {"skip_unwitnessed", "skip_nonresolving", "skip_disclaimed",
                  "skip_incomplete_evidence", "skip_partial",
                  "skip_coverage_unknown", "skip_reopened", "skip_reopen_unknown",
                  "skip_effect_unobserved", "skip_effect_unknown",
                  "skip_unpushed", "skip_effect_reverted",
                  "skip_effect_survival_unknown", CLOSE_ALREADY_COUNTED}:
        return "hold"
    if action == CLOSE_NOT_PERSISTENT:
        return "reopened"
    if action == "close_failed":
        return "failed"
    return action or "unknown"


def close_reason(action: str, row: dict[str, Any]) -> str:
    if row.get("reason"):
        return str(row["reason"])
    if action == "would_close":
        return "witness ok; dry-run only"
    if action == "closed":
        return "closed by live close arm"
    return (row.get("title") or "")[:80]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Close OPEN_WITNESSED issues, each re-verified via dos commit-audit.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--limit", type=int, default=10, help="max issues to close (default: 10)")
    ap.add_argument("--live", action="store_true", help="execute the gh closes (default: dry-run)")
    ap.add_argument("--audit-json", default=None,
                    help="path to a saved issue_closure_audit --json (else run it fresh)")
    ap.add_argument("--max-commits", type=int, default=600,
                    help="git history budget when running the audit fresh (default: 600)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--no-require-pushed", dest="require_pushed", action="store_false",
                    help="close on a local witness even if the resolving commit is not "
                         "yet on origin/main (default: require it pushed -- prevents "
                         "closing an issue whose commit a shared-tree race can orphan)")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, limit=args.limit, live=args.live,
                       audit_json=args.audit_json, max_commits=args.max_commits,
                       require_pushed=args.require_pushed)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
