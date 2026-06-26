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
one typed verdict per check, plus an ``ok`` bit AND a 0–100 ``score`` the
control-pane reads first. Read-only by construction: it never edits the README;
it only checks it.

The checks split into two tiers.

**Hygiene** — is the page *correct*? (FAIL is a required edit; these gate ``ok``)

  links              every Markdown link target resolves on disk      FAIL on a dead link
  version_pins       every fak version string matches the VERSION file FAIL on a stale pin
  naive_baseline     no bolded headline LEADS with a "naive" baseline   FAIL  (SOTA-not-naive law)
  headline_authority each bolded headline number is an authority row   WARN if not mirrored
  freshness_stamp    the <!-- readme-verified: DATE … --> marker is     WARN if absent/older
                       present and not older than --max-age-days (14)        than the window
  jargon_density     count first-screen expert terms with no plain gloss ADVISORY count only

**Substance** — does the page *do its job* for the whole audience? Each is a set
of concrete front-page affordances graded as a fraction; together they drive the
composite ``score``. A fresh-but-thin page passes every hygiene check yet still
fails the reader — so these have teeth through the number even when they don't
hard-gate ``ok``:

  guard_prominence   `fak guard` (the least-friction onramp) leads, wraps a real
                       agent, carries a one-line value + a no-key note            FAIL only if absent
  lcd_onramp         the lowest-common-denominator reader (no key, won't read) gets
                       a one-glance value, a copy-paste bare-binary command, and the
                       expected output, all above the fold                        FAIL only if absent
  speed_claim        a front-screen SPEED number (tok/s, ns, latency, ×), honestly
                       framed (traced to authority or marked relayed) and bounded  WARN
  hero_above_fold    at least one concrete headline RESULT on the first screen — a
                       skeptic/perf/casual reader sees a number before scrolling    WARN
  audience_footholds the first screen gives each reader (skeptic / security / perf /
                       casual) a foothold + an explicit who-is-this-for router      WARN

FAIL is a required edit; WARN/ADVISORY are judgment calls. ``ok`` is False iff a
*hygiene* FAIL fires (or a substance affordance is wholly absent, or the audit
itself errored). The ``score`` (0–100, A–F) is the richer signal: it weights the
substance checks heavily, so "the page is fresh" and "the page is good" are two
different numbers — the way they are in real life.

The three operator front-page laws this enforces:
  1. SOTA-vs-us, never naive   -> naive_baseline FAIL
  2. 6th-grade / Feynman voice  -> jargon_density advisory
  3. wide-audience appeal       -> audience_footholds / lcd_onramp / hero_above_fold

Run from the repo ROOT (``python tools/readme_freshness_audit.py``); the I/O is
pure-filesystem, no ``dos`` subprocess. The companion process is the
``/refresh-readme`` skill, which reads this audit's FAILs + the lowest-scoring
checks as its work-list and re-stamps the marker when done.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "fleet-readme-freshness-audit/1"

# Repo-root-relative inputs (the repo root is the Go module root, where
# BENCHMARK-AUTHORITY.md lives alongside README.md and VERSION).
README_REL = "README.md"
VERSION_REL = "VERSION"
AUTHORITY_REL = "BENCHMARK-AUTHORITY.md"

# How long a freshness stamp stays "fresh" before we WARN.
DEFAULT_MAX_AGE_DAYS = 14

# "First screen" = the top of the page a cold reader meets before clicking
# through. We measure jargon density + every substance affordance only here;
# deep-dive links may be as technical as they like. ~110 lines covers the
# headline sections through "Why now".
FIRST_SCREEN_LINES = 110

# "One glance" = the handful of lines a reader who will NOT scroll actually sees.
# The single most important sentence (what is this, why care) has to live here,
# not under a paragraph the lowest-common-denominator reader never finishes.
ONE_GLANCE_LINES = 8

# Expert terms that stumble a 6th-grade / Feynman reader on the first screen if
# they appear with no plain-language gloss nearby. Advisory only.
JARGON_TERMS = [
    "vDSO", "context-MMU", "IPC", "RadixAttention", "KV cache", "KV-cache",
    "prefix reuse", "append-only", "core dump", "address space",
    "fail-open", "default-deny", "adjudicat",  # adjudicate/adjudication
]

# Composite-score weights. Hygiene checks are necessary-but-not-sufficient
# (lower weight); the substance checks are the front page's actual job (higher).
# speed_claim and hero_above_fold carry the most weight because "a front page
# with no result and no speed number above the fold" is the single biggest gap
# for EVERY reader — the skeptic wants the number, the perf engineer wants the
# speed, the casual reader wants the wow. A check absent from this map defaults
# to 0.5 so a future check still counts.
WEIGHTS: dict[str, float] = {
    # hygiene
    "links": 1.0,
    "version_pins": 1.0,
    "naive_baseline": 1.0,
    "headline_authority": 0.75,
    "freshness_stamp": 0.75,
    "jargon_density": 0.5,
    # substance
    "guard_prominence": 1.5,
    "lcd_onramp": 1.5,
    "audience_footholds": 1.5,
    "speed_claim": 2.0,
    "hero_above_fold": 2.0,
}

# The composite SCORE measures front-page substance — does the page do its job
# for the whole audience. Correctness (links/pins/numbers/stamp) is a separate
# GATE: a hygiene FAIL flips `ok` False and caps the grade, but an all-green
# hygiene row must NOT inflate the quality score (a fresh-but-thin page is the
# exact failure this auditor exists to catch). So the score is the weighted
# average over the substance checks only; hygiene gates it, it does not pad it.
SUBSTANCE_CHECKS = {
    "guard_prominence", "lcd_onramp", "audience_footholds",
    "speed_claim", "hero_above_fold",
}

# A page with a hygiene FAIL (dead link, stale pin, naive-led headline) is
# broken — it cannot earn a passing front-page grade however good its substance.
FAIL_SCORE_CAP = 55

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

# A front-screen SPEED token: an explicit rate / latency / per-token throughput
# term, OR a unicode-× multiplier (a bare "x" is too noisy — "x86" etc). This is
# the "faster speed" signal the front page must carry for the perf reader.
_SPEED_TOKEN_RE = re.compile(
    r"\b(?:tok/s|tok/sec|tokens?\s*/\s*s(?:ec)?|tokens?\s+per\s+second|"
    r"ns(?:/op)?|µs|μs|nanoseconds?|microseconds?|latency|throughput|"
    r"time[- ]to[- ]first[- ]token|ttft|prefill|decode\b)"
    r"|\d[\d.,]*\s*×",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pure check functions: each takes already-read text and returns one check dict.
# This is the testable seam — tests pass fixture strings, no disk needed.
# A check dict is {check, status (OK|WARN|FAIL|ADVISORY), detail, items?,
# score? (a 0..1 fraction for the graded substance checks)}.
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
    /refresh-readme pass, not a pass/fail bit. It still feeds the score: each
    naked term shaves a little, floored so voice can never sink the composite.
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
    score = 1.0 if not naked else max(0.4, 1.0 - 0.15 * len(naked))
    return {
        "check": "jargon_density", "status": "ADVISORY",
        "score": round(score, 3),
        "detail": (f"{len(naked)} first-screen term(s) appear with no plain-language gloss nearby"
                   if naked else "first-screen jargon reads with plain-language glosses"),
        "items": naked,
    }


# ---------------------------------------------------------------------------
# Substance checks — does the front page do its job for the whole audience?
# Each grades a set of concrete affordances as a fraction (the `score`), and
# reports the MISSING ones in `items` as the /refresh-readme work-list.
# ---------------------------------------------------------------------------

def check_guard_prominence(readme: str, *, first_screen_lines: int) -> dict[str, Any]:
    """The least-friction onramp — `fak guard` — leads the front page.

    `fak guard -- <agent>` is the lowest-effort way to adopt fak (wrap the agent
    you already run; keep your key). A front page that buries it under the serve
    / route paths makes adoption look harder than it is. Affordances:
      present       `fak guard` appears above the fold
      wraps_agent   shown wrapping a real CLI (`-- claude` / codex / opencode / …)
      value_phrase  a one-line "why" sits next to it (drop-in / wrap the agent …)
      no_key_note   the no-key / forwards-your-credential promise is nearby
      floor_purpose its security purpose (floor / verdict / deny) is nearby
      leads_onramp  it appears BEFORE the first `fak serve` (it is the lead path)
    """
    head = readme.splitlines()[:first_screen_lines]
    headtext = "\n".join(head)
    guard_idxs = _line_idxs(head, "fak guard")
    serve_idxs = _line_idxs(head, "fak serve")
    subs = {
        "present": bool(guard_idxs),
        "wraps_agent": bool(re.search(
            r"fak guard[^\n]*--\s*(claude|codex|opencode|aider|cursor|gemini)",
            headtext, re.IGNORECASE)),
        "value_phrase": any(_near(head, i, 3, [
            "drop-in", "wrap the agent", "agent you already", "one command",
            "no rewrite", "in front of",
        ]) for i in guard_idxs),
        "no_key_note": any(_near(head, i, 6, [
            "no api key", "no key", "subscription", "forwards", "credential",
        ]) for i in guard_idxs),
        "floor_purpose": any(_near(head, i, 6, [
            "floor", "secure", "verdict", "deny", "decision", "policy",
        ]) for i in guard_idxs),
        "leads_onramp": bool(guard_idxs) and (not serve_idxs or guard_idxs[0] < serve_idxs[0]),
    }
    return _grade_subs(
        "guard_prominence", subs, fail_if_zero=True,
        label="`fak guard` (the least-friction onramp) leads the front page")


def check_lcd_onramp(readme: str, *, first_screen_lines: int,
                     one_glance_lines: int) -> dict[str, Any]:
    """The lowest-common-denominator reader gets a no-setup first command.

    The LCD reader landed from a link, will not read prose, has no key, and
    wants one line that visibly does something. Affordances:
      one_glance_value  a single plain "what is this" sentence in the first ~8 lines
      fenced_cmd        a copy-paste fenced block above the fold
      bare_binary_cmd   a bare `fak …` command (works from the binary, no clone)
      expected_output   the expected result shown inline (`# -> DENY`, →, …)
      no_setup_promise  "no key / no model / no GPU / no clone" stated above the fold
      install_reachable how to GET the binary (curl|go install|Install link) is nearby
    """
    lines = readme.splitlines()
    head = lines[:first_screen_lines]
    headtext = "\n".join(head)
    glance = "\n".join(lines[:one_glance_lines])
    subs = {
        # A one-glance value = a blockquote/bold one-liner OR an explicit "in one
        # line" marker within the first few lines (above any long paragraph).
        "one_glance_value": ("one line" in glance.lower())
        or bool(re.search(r"^\s*>\s*\*\*", glance, re.MULTILINE))
        or bool(re.search(r"^\s*\*\*[^*]+\*\*\s*$", glance, re.MULTILINE)),
        "fenced_cmd": "```" in headtext,
        "bare_binary_cmd": bool(re.search(r"^\s*fak\s+\w", headtext, re.MULTILINE)),
        "expected_output": bool(re.search(
            r"#\s*->|#\s*=>|→|->\s*(ALLOW|DENY|TRANSFORM|QUARANTINE)", headtext)),
        "no_setup_promise": any(k in headtext.lower() for k in [
            "no key", "no api key", "no model", "no gpu", "no clone",
        ]),
        "install_reachable": bool(re.search(
            r"curl[^\n]*install|go install|install\.sh|\[install\]", headtext, re.IGNORECASE)),
    }
    return _grade_subs(
        "lcd_onramp", subs, fail_if_zero=True,
        label="the lowest-common-denominator reader gets a no-setup, copy-paste first command")


def check_speed_claim(readme: str, authority: str, *,
                      first_screen_lines: int) -> dict[str, Any]:
    """A front-screen SPEED number, honestly framed and traceable.

    "Faster speed" is one of the things a perf reader scans for first, and the
    front page used to carry none above the fold. Affordances:
      speed_token   a rate/latency/throughput term (tok/s, ns, latency, ×) above the fold
      traced_or_marked  a front-screen number that is ALSO in BENCHMARK-AUTHORITY,
                        OR is explicitly marked relayed/observed/telemetry/measured
      bounded       a fence near it (vs tuned / single-stream / in-process / not wall-clock)
                        so the speed isn't overclaimed
      links_authority  the first screen links to the benchmark authority/benchmarks doc
    """
    head = "\n".join(readme.splitlines()[:first_screen_lines])
    auth = authority or ""
    nums_head = {_norm_num(x) for x in _MULT_RE.findall(head)}
    nums_auth = {_norm_num(x) for x in _MULT_RE.findall(auth)}
    traced = bool(nums_head & nums_auth)
    marked = any(k in head.lower() for k in [
        "observed", "relayed", "telemetry", "provider's own", "/metrics", "measured",
    ])
    bounded = any(k in head.lower() for k in [
        "vs tuned", "vs a tuned", "single-stream", "not wall-clock",
        "reference", "overhead", "per call", "per-call", "in-process",
    ])
    links_auth = bool(re.search(r"benchmark-authority|benchmarks?\b", head, re.IGNORECASE))
    # The honesty/boundary affordances only COUNT once a speed number actually
    # exists above the fold — otherwise a stray "vs tuned" or "benchmarks" link
    # would award credit for framing a number that isn't there.
    has_token = bool(_SPEED_TOKEN_RE.search(head))
    subs = {
        "speed_token": has_token,
        "traced_or_marked": has_token and (traced or marked),
        "bounded": has_token and bounded,
        "links_authority": has_token and links_auth,
    }
    return _grade_subs(
        "speed_claim", subs, fail_if_zero=False,
        label="a front-screen speed number, honestly framed and traceable")


def check_hero_above_fold(readme: str, authority: str, *,
                          first_screen_lines: int) -> dict[str, Any]:
    """At least one concrete headline RESULT lives on the first screen.

    A skeptic, a perf engineer, and a casual reader all want to see ONE real
    number before they scroll. Affordances:
      has_number    a multiplier or a concrete rate/count appears above the fold
      traced        a bolded multiplier above the fold is mirrored in the authority doc
      sota_framed   a SOTA-vs-us cue near it (vs tuned / vs a tuned warm-cache / parity)
                        — the headline is honest, not vs-naive
      not_only_naive  if 'naive' appears above the fold, a non-naive number does too
    """
    head_lines = readme.splitlines()[:first_screen_lines]
    head = "\n".join(head_lines)
    auth_nums = {_norm_num(x) for x in _MULT_RE.findall(authority or "")}
    head_mults = [m.group("body") for m in _BOLD_RE.finditer(head) if _MULT_RE.search(m.group("body"))]
    # A "number" above the fold: a × multiplier OR a concrete rate/count.
    has_number = bool(_MULT_RE.search(head)) or bool(re.search(
        r"\b\d[\d.,]*\s*(?:tok/s|tokens?/s|ns|µs|μs|min(?:ute)?s?|turns?|agents?)\b",
        head, re.IGNORECASE))
    bolded_traced = any(
        any(_norm_num(x) in auth_nums for x in _MULT_RE.findall(b)) for b in head_mults)
    sota_framed = any(k in head.lower() for k in [
        "vs tuned", "vs a tuned", "warm-cache", "warm cache", "parity", "sota",
    ])
    naive_present = bool(re.search(r"\bnaive\b", head, re.IGNORECASE))
    # Framing affordances only count once a number actually exists above the
    # fold — a page with no hero result scores 0, not partial credit for an
    # absent-but-honest "no naive" or a stray "parity" mention elsewhere.
    subs = {
        "has_number": has_number,
        "traced": has_number and (bolded_traced or not head_mults),  # a marked rate counts
        "sota_framed": has_number and sota_framed,
        "not_only_naive": has_number and ((not naive_present) or sota_framed or bolded_traced),
    }
    return _grade_subs(
        "hero_above_fold", subs, fail_if_zero=False,
        label="a concrete, SOTA-framed headline result above the fold")


def check_audience_footholds(readme: str, *, first_screen_lines: int) -> dict[str, Any]:
    """The first screen gives each reader a foothold (law 3, wide-audience).

    Four readers land cold; each needs a place to stand on the first screen,
    plus an explicit router so they can find their path:
      skeptic    an honesty anchor (CLAIMS / honest / ledger / what's-not) or a
                   runnable offline ALLOW/DENY proof
      security   the lock not the screener (capability floor / default-deny /
                   refused by structure / allow-list)
      perf       the reuse-or-speed win (cache+reuse/prefix, a ×, or a speed token)
      casual     the no-setup demo (no key / no GPU / copy-paste)
      persona_map  an explicit who-is-this-for router (Start here / pick / for … teams)
    """
    head = "\n".join(readme.splitlines()[:first_screen_lines]).lower()
    subs = {
        "skeptic": (any(k in head for k in [
            "claims", "honest", "ledger", "what's real", "what's not", "what it's not",
        ]) or ("offline" in head and ("allow" in head or "deny" in head))),
        "security": any(k in head for k in [
            "capability floor", "default-deny", "default_deny",
            "refused by structure", "allow-list", "allow list",
        ]),
        "perf": (("cache" in head and any(k in head for k in ["reuse", "prefix", "discount"]))
                 or bool(_SPEED_TOKEN_RE.search(head)) or "×" in head),
        "casual": any(k in head for k in [
            "no key", "no api key", "no gpu", "2-minute", "two-minute",
            "copy-paste", "copy and paste", "paste",
        ]),
        "persona_map": any(k in head for k in [
            "who is this for", "pick the line", "pick your path", "start here",
            "for security teams", "if you ",
        ]),
    }
    return _grade_subs(
        "audience_footholds", subs, fail_if_zero=False,
        label="the first screen gives each reader (skeptic / security / perf / casual) a foothold")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _grade_subs(check: str, subs: dict[str, bool], *, fail_if_zero: bool,
                label: str, warn_below: float = 0.75) -> dict[str, Any]:
    """Fold a dict of boolean affordances into a graded check dict.

    score = met/total; status = FAIL (if fail_if_zero and none met) / WARN
    (below warn_below) / OK. The MISSING affordances become the work-list.
    """
    met = sum(1 for v in subs.values() if v)
    total = len(subs) or 1
    score = met / total
    if fail_if_zero and met == 0:
        status = "FAIL"
    elif score < warn_below:
        status = "WARN"
    else:
        status = "OK"
    missing = sorted(k for k, v in subs.items() if not v)
    return {
        "check": check, "status": status, "score": round(score, 3),
        "detail": f"{label}: {met}/{total} affordances present",
        "items": missing,
    }


def _line_idxs(lines: list[str], needle: str) -> list[int]:
    nl = needle.lower()
    return [i for i, ln in enumerate(lines) if nl in ln.lower()]


def _near(lines: list[str], idx: int, radius: int, needles: list[str]) -> bool:
    lo, hi = max(0, idx - radius), min(len(lines), idx + radius + 1)
    blob = "\n".join(lines[lo:hi]).lower()
    return any(n in blob for n in needles)


def _parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _norm_num(s: str) -> str:
    """Normalize a multiplier token for comparison: strip ~, spaces, unify ×/x."""
    return re.sub(r"[~\s]", "", s).replace("x", "×").replace("X", "×")


def _check_score(c: dict[str, Any]) -> float:
    """A check's 0..1 contribution: its graded `score` if present, else by status."""
    s = c.get("score")
    if isinstance(s, (int, float)):
        return float(s)
    return {"OK": 1.0, "WARN": 0.5, "FAIL": 0.0, "ADVISORY": 1.0}.get(c["status"], 0.0)


def _grade_letter(score: int) -> str:
    return ("A" if score >= 90 else "B" if score >= 80 else
            "C" if score >= 70 else "D" if score >= 60 else "F")


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

    # Composite 0–100 score = weighted average over the SUBSTANCE checks only
    # (hygiene gates `ok`, it does not pad the score — see SUBSTANCE_CHECKS). A
    # degenerate fixture with no substance checks falls back to all checks so a
    # unit test can still produce a number.
    substance = [c for c in checks if c["check"] in SUBSTANCE_CHECKS]
    scored = substance if substance else checks
    has_substance = bool(substance)
    total_w = sum(WEIGHTS.get(c["check"], 0.5) for c in scored)
    got_w = sum(WEIGHTS.get(c["check"], 0.5) * _check_score(c) for c in scored)
    score = round(100 * got_w / total_w) if total_w else 0
    if fails:
        score = min(score, FAIL_SCORE_CAP)  # a broken page is not a passing front page
    grade = _grade_letter(score)

    # The work-list for next_action: lowest weighted contribution first (the
    # check where lifting the score buys the most), excluding already-perfect.
    ranked = sorted(
        (c for c in scored if _check_score(c) < 1.0),
        key=lambda c: WEIGHTS.get(c["check"], 0.5) * (1.0 - _check_score(c)),
        reverse=True,
    )
    worst = ", ".join(c["check"] for c in ranked[:3])

    if error:
        ok, verdict, finding = False, "AUDIT_ERROR", "tooling_error"
        reason = error
        next_action = "fix the README/VERSION/authority read (run from repo ROOT), then re-run"
    elif fails:
        ok, verdict, finding = False, "ACTION", "readme_drift"
        names = ", ".join(c["check"] for c in fails)
        reason = f"score {score}/100 ({grade}); {len(fails)} required README fix(es): {names}"
        next_action = ("invoke /refresh-readme: each FAIL is a required edit (fix the dead link / "
                       "stale pin / naive-lead headline / missing onramp), then re-stamp and re-run")
    elif has_substance and score < 90:
        ok, verdict, finding = True, "OK", "readme_fresh_thin"
        reason = (f"score {score}/100 ({grade}): front page is correct but thin — "
                  f"raise the substance checks ({worst})")
        next_action = (f"invoke /refresh-readme: lift the lowest-scoring checks ({worst}) toward 90+, "
                       "then re-stamp readme-verified and re-run")
    elif warns:
        ok, verdict, finding = True, "OK", "readme_fresh_with_notes"
        names = ", ".join(c["check"] for c in warns)
        reason = f"score {score}/100 ({grade}); no required fix; {len(warns)} judgment-call WARN(s): {names}"
        next_action = "review each WARN at the next /refresh-readme pass; no blocking edit needed"
    else:
        ok, verdict, finding = True, "OK", "readme_fresh"
        reason = (f"score {score}/100 ({grade}): front page is correct AND complete — "
                  "links resolve, pins current, numbers traced, SOTA-led, guard/speed/hero above the fold")
        next_action = "no README action needed; re-run after the next front-page or VERSION change"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "score": score,
        "grade": grade,
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
        # hygiene — is the page correct?
        check_links(readme, root),
        check_version_pins(readme, version),
        check_naive_baseline(readme),
        check_headline_authority(readme, authority),
        check_freshness_stamp(readme, today=today, max_age_days=max_age_days),
        check_jargon_density(readme, first_screen_lines=FIRST_SCREEN_LINES),
        # substance — does the page do its job for the whole audience?
        check_guard_prominence(readme, first_screen_lines=FIRST_SCREEN_LINES),
        check_lcd_onramp(readme, first_screen_lines=FIRST_SCREEN_LINES,
                         one_glance_lines=ONE_GLANCE_LINES),
        check_speed_claim(readme, authority, first_screen_lines=FIRST_SCREEN_LINES),
        check_hero_above_fold(readme, authority, first_screen_lines=FIRST_SCREEN_LINES),
        check_audience_footholds(readme, first_screen_lines=FIRST_SCREEN_LINES),
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
        f"readme-freshness audit: {payload.get('verdict')} ({payload.get('finding')})  "
        f"score {payload.get('score')}/100 ({payload.get('grade')})",
        (f"checks: ok={counts.get('OK', 0)} warn={counts.get('WARN', 0)} "
         f"fail={counts.get('FAIL', 0)} advisory={counts.get('ADVISORY', 0)}"),
        f"next: {payload.get('next_action')}",
    ]
    for c in payload.get("checks", []):
        mark = {"OK": "  ok ", "WARN": " warn", "FAIL": " FAIL", "ADVISORY": " adv "}.get(
            c["status"], "  ?  ")
        sc = c.get("score")
        sctxt = f" [{sc:.2f}]" if isinstance(sc, (int, float)) else ""
        lines.append(f"{mark}  {c['check']:<18}{sctxt} {c['detail']}")
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
    # ADVISORY are judgment calls and stay green; a thin-but-fresh page is OK
    # (the score, not the exit code, carries "thin").
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
