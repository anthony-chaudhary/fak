#!/usr/bin/env python3
"""README front-page freshness auditor — the front door's checking layer.

``README.md`` is the one outward-facing surface read cold by everyone — an
adopter, a reviewer, a skeptic — and it is the surface most likely to rot: a
link goes dead, a version pin lags the ``VERSION`` file, a headline number drifts
from ``fak/BENCHMARK-AUTHORITY.md``, a "we beat naive" claim creeps back into the
lead. Every other claim surface in this repo already has a checking layer
(``memory_recall_audit`` re-verifies memories, ``issue_closure_audit`` grades
closures, ``BENCHMARK-AUTHORITY`` is the single source for numbers). The README
sat outside all of them — correct only as long as a human happened to tend it.

This is that missing layer. It folds read-back surfaces it does not author (the
README text, the ``VERSION`` file, the authority doc, the filesystem) and reports
one typed verdict per check, plus an ``ok`` bit the control-pane reads first.
Read-only by construction: it never edits the README; it only checks it.

The checks, and why each exists:

  links              every Markdown link target resolves on disk      FAIL on a dead link
  version_pins       every fak version string matches the VERSION file FAIL on a stale pin
  headline_authority each bolded headline number is an authority row   WARN if not mirrored
  naive_baseline     no bolded headline LEADS with a "naive" baseline   FAIL  (SOTA-not-naive law)
  freshness_stamp    the <!-- readme-verified: DATE … --> marker is     WARN if absent/older
                       present and not older than --max-age-days (14)        than the window
  jargon_density     count first-screen expert terms with no plain gloss ADVISORY count only

FAIL is a required edit; WARN/ADVISORY are judgment calls. ``ok`` is False iff
any FAIL fires (or the audit itself errored) — voice (jargon) is never a hard
gate, because plain-language is a writing judgment, not a mechanical rule.

The three operator front-page laws this enforces:
  1. SOTA-vs-us, never naive   -> naive_baseline FAIL
  2. 6th-grade / Feynman voice  -> jargon_density advisory
  3. numbers trace to one source -> headline_authority WARN

Run from the repo ROOT (``python tools/readme_freshness_audit.py``); the I/O is
pure-filesystem, no ``dos`` subprocess. The companion process is the
``/refresh-readme`` skill, which reads this audit's FAILs as its work-list and
re-stamps the marker when done.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "fleet-readme-freshness-audit/1"

# Repo-root-relative inputs.
README_REL = "README.md"
VERSION_REL = "VERSION"
AUTHORITY_REL = "fak/BENCHMARK-AUTHORITY.md"

# How long a freshness stamp stays "fresh" before we WARN.
DEFAULT_MAX_AGE_DAYS = 14

# "First screen" = the top of the page a cold reader meets before clicking
# through. We measure jargon density only here; deep-dive links may be as
# technical as they like. ~110 lines covers the headline sections through
# "Why this matters now".
FIRST_SCREEN_LINES = 110

# Expert terms that stumble a 6th-grade / Feynman reader on the first screen if
# they appear with no plain-language gloss nearby. Advisory only.
JARGON_TERMS = [
    "vDSO", "context-MMU", "IPC", "RadixAttention", "KV cache", "KV-cache",
    "prefix reuse", "append-only", "core dump", "address space",
    "fail-open", "default-deny", "adjudicat",  # adjudicate/adjudication
]

# The freshness stamp grammar the /refresh-readme skill writes.
#   <!-- readme-verified: 2026-06-20 vs VERSION 0.25.0 + BENCHMARK-AUTHORITY -->
_STAMP_RE = re.compile(
    r"<!--\s*readme-verified:\s*(\d{4}-\d{2}-\d{2})\b(?P<rest>[^>]*)-->",
    re.IGNORECASE,
)

# A Markdown inline link: [text](target). We only resolve LOCAL targets — http(s),
# mailto, and pure #anchors are out of scope (the network is not ours to witness).
_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")

# A fak version string: a bare semver, optionally v-prefixed. We compare the
# MAJOR.MINOR.PATCH against the VERSION file. A pin like "v0.3.x" or "v0.25.x"
# (a deliberate range) is matched on its leading numeric part.
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)\.(?:(\d+)|x)\b")

# A bolded headline claim: **…** (the front page leads its numbers in bold).
_BOLD_RE = re.compile(r"\*\*(?P<body>[^*]+)\*\*")

# Inside a bold span, a multiplier headline number like "60×" / "~4x" / "5.3–7.4×".
_MULT_RE = re.compile(r"~?\d[\d.,]*\s*(?:[–-]\s*\d[\d.,]*\s*)?[×x]")


# ---------------------------------------------------------------------------
# Pure check functions: each takes already-read text and returns one check dict.
# This is the testable seam — tests pass fixture strings, no disk needed.
# A check dict is {check, status (OK|WARN|FAIL|ADVISORY), detail, items?}.
# ---------------------------------------------------------------------------

def check_links(readme: str, root: Path) -> dict[str, Any]:
    dead: list[str] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(readme):
        target = m.group("target").strip()
        # Out of scope: network links, anchors, mail. Strip a trailing #anchor.
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)
        if not (root / path_part).exists():
            dead.append(path_part)
    if dead:
        return {
            "check": "links", "status": "FAIL",
            "detail": f"{len(dead)} README link target(s) do not exist on disk",
            "items": sorted(dead),
        }
    return {"check": "links", "status": "OK",
            "detail": f"all {len(seen)} local link target(s) resolve"}


def check_version_pins(readme: str, version: str) -> dict[str, Any]:
    """FAIL if any fak version pin names a version BEHIND the VERSION file.

    A pin equal to (or, for a forward-looking ``vX.Y.x`` range, covering) the
    current minor is fine; a pin naming an older minor is the stale-pin defect
    the #466 fix (`e0023ba`) corrected by hand. We compare (major, minor) and
    let an explicit ``.x`` patch range pass on the minor.
    """
    cur = _parse_version(version)
    if cur is None:
        return {"check": "version_pins", "status": "WARN",
                "detail": f"could not parse VERSION file ({version!r})"}
    cur_major, cur_minor, _ = cur
    stale: list[str] = []
    for m in _VERSION_RE.finditer(readme):
        major, minor = int(m.group(1)), int(m.group(2))
        # Only audit fak's own version line (major 0 today); ignore unrelated
        # numbers that happen to look like semver (e.g. a Go 1.26 reference is
        # not v-shaped here, but guard anyway by requiring same major).
        if major != cur_major:
            continue
        if (major, minor) < (cur_major, cur_minor):
            stale.append(m.group(0))
    if stale:
        return {
            "check": "version_pins", "status": "FAIL",
            "detail": f"version pin(s) behind VERSION {version}: refresh to v{cur_major}.{cur_minor}.x",
            "items": sorted(set(stale)),
        }
    return {"check": "version_pins", "status": "OK",
            "detail": f"no version pin behind VERSION {version}"}


def check_naive_baseline(readme: str) -> dict[str, Any]:
    """FAIL if a bolded headline LEADS with a 'naive' baseline.

    The operator law: SOTA-vs-us, never naive. A bolded multiplier whose own
    span (or the same line) names 'naive' as the comparison is the strawman to
    refuse. A 'naive' mention NOT inside a bold headline (e.g. explaining the
    cost model in prose) is fine — the rule is about what LEADS.
    """
    offenders: list[str] = []
    for line in readme.splitlines():
        for m in _BOLD_RE.finditer(line):
            body = m.group("body")
            if _MULT_RE.search(body) and re.search(r"\bnaive\b", body, re.IGNORECASE):
                offenders.append(body.strip())
    if offenders:
        return {
            "check": "naive_baseline", "status": "FAIL",
            "detail": "bolded headline leads with a 'naive' baseline — lead with the SOTA comparison",
            "items": offenders,
        }
    return {"check": "naive_baseline", "status": "OK",
            "detail": "no bolded headline leads with a naive baseline"}


def check_headline_authority(readme: str, authority: str) -> dict[str, Any]:
    """WARN if a bolded multiplier headline number is not also in the authority doc.

    Not a hard gate: prose may round or restate. We just assert the front page
    mirrors the single source of truth (BENCHMARK-AUTHORITY), surfacing any
    bolded number that has no matching figure there to be reconciled by hand.
    """
    auth_nums = {_norm_num(x) for x in _MULT_RE.findall(authority)}
    missing: list[str] = []
    for m in _BOLD_RE.finditer(readme):
        for num in _MULT_RE.findall(m.group("body")):
            if _norm_num(num) not in auth_nums:
                missing.append(num.strip())
    missing = sorted(set(missing))
    if missing:
        return {
            "check": "headline_authority", "status": "WARN",
            "detail": "bolded headline number(s) not found in BENCHMARK-AUTHORITY — reconcile",
            "items": missing,
        }
    return {"check": "headline_authority", "status": "OK",
            "detail": "every bolded multiplier headline mirrors an authority figure"}


def check_freshness_stamp(readme: str, *, today: _dt.date,
                          max_age_days: int) -> dict[str, Any]:
    m = _STAMP_RE.search(readme)
    if not m:
        return {
            "check": "freshness_stamp", "status": "WARN",
            "detail": "no <!-- readme-verified: DATE … --> stamp; add one when you verify the page",
        }
    try:
        stamped = _dt.date.fromisoformat(m.group(1))
    except ValueError:
        return {"check": "freshness_stamp", "status": "WARN",
                "detail": f"unparseable stamp date {m.group(1)!r}"}
    age = (today - stamped).days
    if age > max_age_days:
        return {
            "check": "freshness_stamp", "status": "WARN",
            "detail": f"stamp is {age}d old (> {max_age_days}d) — re-verify and re-stamp",
        }
    return {"check": "freshness_stamp", "status": "OK",
            "detail": f"verified {age}d ago (<= {max_age_days}d window)"}


def check_jargon_density(readme: str, *, first_screen_lines: int) -> dict[str, Any]:
    """ADVISORY: count first-screen jargon terms lacking a nearby plain gloss.

    Voice is judgment, not a gate, so this never FAILs. A term is 'glossed' if a
    parenthetical or an em-dash explanation sits on the same line — a cheap
    proxy for 'the writer paused to explain it'. The number is a nudge for the
    /refresh-readme pass, not a pass/fail bit.
    """
    head = "\n".join(readme.splitlines()[:first_screen_lines])
    naked: list[str] = []
    for term in JARGON_TERMS:
        for line in head.splitlines():
            if term.lower() in line.lower():
                glossed = ("(" in line) or ("—" in line) or (" - " in line)
                if not glossed:
                    naked.append(term)
                break
    naked = sorted(set(naked))
    return {
        "check": "jargon_density", "status": "ADVISORY",
        "detail": (f"{len(naked)} first-screen term(s) appear with no plain-language gloss nearby"
                   if naked else "first-screen jargon reads with plain-language glosses"),
        "items": naked,
    }


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _norm_num(s: str) -> str:
    """Normalize a multiplier token for comparison: strip ~, spaces, unify ×/x."""
    return re.sub(r"[~\s]", "", s).replace("x", "×").replace("X", "×")


# ---------------------------------------------------------------------------
# Grader: fold the check list into the standard control-pane payload
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, checks: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "ADVISORY": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]

    if error:
        ok, verdict, finding = False, "AUDIT_ERROR", "tooling_error"
        reason = error
        next_action = "fix the README/VERSION/authority read (run from repo ROOT), then re-run"
    elif fails:
        ok, verdict, finding = False, "ACTION", "readme_drift"
        names = ", ".join(c["check"] for c in fails)
        reason = f"{len(fails)} required README fix(es): {names}"
        next_action = ("invoke /refresh-readme: each FAIL is a required edit (fix the dead link / "
                       "stale pin / naive-lead headline), then re-stamp readme-verified and re-run")
    elif warns:
        ok, verdict, finding = True, "OK", "readme_fresh_with_notes"
        names = ", ".join(c["check"] for c in warns)
        reason = f"no required fix; {len(warns)} judgment-call WARN(s): {names}"
        next_action = "review each WARN at the next /refresh-readme pass; no blocking edit needed"
    else:
        ok, verdict, finding = True, "OK", "readme_fresh"
        reason = "front page is correct: links resolve, pins current, numbers traced, SOTA-led, stamp fresh"
        next_action = "no README action needed; re-run after the next front-page or VERSION change"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "counts": counts,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def run_checks(readme: str, version: str, authority: str, root: Path, *,
               today: _dt.date, max_age_days: int) -> list[dict[str, Any]]:
    """All checks over already-read text. The pure core; tests call this."""
    return [
        check_links(readme, root),
        check_version_pins(readme, version),
        check_naive_baseline(readme),
        check_headline_authority(readme, authority),
        check_freshness_stamp(readme, today=today, max_age_days=max_age_days),
        check_jargon_density(readme, first_screen_lines=FIRST_SCREEN_LINES),
    ]


def collect(workspace: Path, *, today: _dt.date | None = None,
            max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> dict[str, Any]:
    root = workspace.resolve()
    today = today or _dt.date.today()
    try:
        readme = (root / README_REL).read_text(encoding="utf-8")
    except OSError as exc:
        return build_payload(workspace=str(root), checks=[],
                             error=f"cannot read {README_REL}: {exc}")
    # VERSION and authority are best-effort: a missing one degrades a check to
    # WARN inside that check, it does not error the whole audit.
    version = _safe_read(root / VERSION_REL)
    authority = _safe_read(root / AUTHORITY_REL)
    checks = run_checks(readme, version, authority, root,
                        today=today, max_age_days=max_age_days)
    return build_payload(workspace=str(root), checks=checks)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"readme-freshness audit: {payload.get('verdict')} ({payload.get('finding')})",
        (f"checks: ok={counts.get('OK', 0)} warn={counts.get('WARN', 0)} "
         f"fail={counts.get('FAIL', 0)} advisory={counts.get('ADVISORY', 0)}"),
        f"next: {payload.get('next_action')}",
    ]
    for c in payload.get("checks", []):
        mark = {"OK": "  ok ", "WARN": " warn", "FAIL": " FAIL", "ADVISORY": " adv "}.get(
            c["status"], "  ?  ")
        lines.append(f"{mark}  {c['check']:<18} {c['detail']}")
        for it in (c.get("items") or [])[:10]:
            lines.append(f"           - {it}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="README front-page freshness auditor (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help=f"freshness-stamp WARN window (default: {DEFAULT_MAX_AGE_DAYS})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, max_age_days=args.max_age_days)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    # Exit non-zero ONLY on a required fix (FAIL) or a tooling error. WARN and
    # ADVISORY are judgment calls and stay green.
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
