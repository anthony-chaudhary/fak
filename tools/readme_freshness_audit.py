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
one typed verdict per check, plus an ``ok`` bit AND a TWO-SIDED UNBOUNDED focus
``score`` (100 = a clean, complete page exactly at the size budget; it rises
without ceiling as a complete page gets leaner and falls without floor as bloat
and defects accumulate — see build_payload, so it never saturates at a "fake
perfect" 100). Read-only by construction: it never edits the README; it only
checks it.

The checks split into two tiers.

**Hygiene** — is the page *correct*? (FAIL is a required edit; these gate ``ok``)

  links              every Markdown link target resolves on disk      FAIL on a dead link
  version_pins       every fak version string matches the VERSION file FAIL on a stale pin
  naive_baseline     no bolded headline LEADS with a "naive" baseline   FAIL  (SOTA-not-naive law)
  headline_authority each bolded headline number is an authority row   WARN if not mirrored
  freshness_stamp    the <!-- readme-verified: DATE … --> marker is     WARN if absent/older
                       present and not older than --max-age-days (14)        than the window
  showcase_sync      the OTHER front door (docs/showcase.html, the repo's  FAIL on any
                       configured homepage) is linked from the README and       shortfall
                       still agrees with it on installer, first run, the
                       benchmark authority, and every version it quotes
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
  front_page_focus   the page stays SMALL — a line budget, a section-count budget, a
                       single-lead rule (pitch stated once, not thrice), and detail
                       linked OUT to an overflow sink                               WARN

The substance checks split into two forces that must both live in the tool. Five
reward ADDING an affordance; a page optimized to them alone only grows. The sixth,
``front_page_focus``, is the counterweight — it rewards concision, so "halve the
front page, then let it regrow every pass" stops being the stable equilibrium.

FAIL is a required edit; WARN/ADVISORY are judgment calls. ``ok`` is False iff a
*hygiene* FAIL fires (or a substance affordance is wholly absent, or the audit
itself errored). The ``score`` is the richer signal — a TWO-SIDED UNBOUNDED,
magnitude-aware index where 100 is the ZERO-POINT (a clean, complete page exactly
at the size budget), a leaner complete page scores ABOVE it without ceiling, and
every defect or line/section over budget subtracts real points BELOW it without
floor. So "the page is fresh" and "the page is good" stay two different numbers,
and a genuinely excellent page keeps climbing instead of parking at a fake-perfect
100 (a bounded 0-100 score frozen at "A" could not). ``readme_debt`` remains the
lower-is-better defect count for the control pane.

**The keyword-gaming limit, and the one cross-check that closes it.** Every
substance affordance above is a README-*text* heuristic — a keyword or regex
presence over the front screen. That makes them honest about *intent* but blind
to *reality*: unlike the dogfood score (which reads non-forgeable transcripts),
here the README itself is the gamed surface, so a keyword with no real affordance
behind it still scores. The scorecard anti-gaming law (``.claude/skills/scorecard/
SKILL.md``) requires the score be cross-checked against something the author cannot
forge by editing prose. We defend the single highest-leverage affordance that way:
``lcd_onramp``'s bare-binary command — the line the no-setup reader actually
pastes — only scores when the ``fak <verb>`` it names RESOLVES to a verb the
binary really dispatches, parsed live from ``cmd/fak/main.go`` (never a hand-list).
A page that stuffs ``fak <made-up-verb>`` to look runnable does not earn the point;
the reader's first paste would have died on ``unknown verb``. When
``cmd/fak/main.go`` is unreadable (the tool run outside the repo) the check abstains
to presence-only — a missing source of truth is not a README defect. The remaining
affordances stay text-only by design; this is the beachhead, not the whole wall.
``showcase_sync`` reuses that same dispatch set for the same reason: two front doors
agreeing on a command only counts when the command is one the binary really has.

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
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-readme-freshness-audit/2"

# Repo-root-relative inputs (the repo root is the Go module root, where
# BENCHMARK-AUTHORITY.md lives alongside README.md and VERSION).
README_REL = "README.md"
VERSION_REL = "VERSION"
AUTHORITY_REL = "BENCHMARK-AUTHORITY.md"
QWEN_INDEX_REL = "docs/benchmarks/QWEN-PERFORMANCE-INDEX.md"
QWEN_LATEST_REL = "docs/benchmarks/QWEN38-27B-LATEST.md"
# The binary's real dispatch table — the source of truth for which `fak <verb>`
# commands actually exist. Parsed live (never a hand-list) so the lcd_onramp
# anti-gaming cross-check stays correct as verbs are added/renamed.
MAIN_GO_REL = "cmd/fak/main.go"
# The repository's OTHER front door. GitHub's repo-level homepage setting points at
# the Pages copy of this file, so for a share of visitors it — not README.md — is the
# first fak page they see, yet it is authored by hand and no check read it. See
# check_showcase_sync.
SHOWCASE_REL = "docs/showcase.html"
# Benchmark datasets that stamp the fak version their numbers were measured at. The
# showcase quotes one of those as-of versions in a chart caption; that is legitimate
# and must NOT be mistaken for a stale product pin (see check_showcase_sync).
BENCH_DATA_GLOB = "tools/*.data.json"

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

# The size law, made checkable. The five substance checks above all reward
# ADDING an affordance (guard, lcd, speed, hero, footholds); nothing pushed back
# on growth, so every refresh pass and every "surface X on the front page" commit
# ratcheted the page UP — halve, regrow, halve, regrow. front_page_focus is the
# counterweight: a total-line budget, a section-count budget, and a single-lead
# rule (the one-binary/syscall pitch stated once in the preamble, not restated
# three times before the reader reaches a section). Generous by design — it flags
# DRIFT (debt), it does not hard-FAIL a page for being one line over. Bump the
# budgets deliberately (and say why in the commit) when the page genuinely earns
# a new section; do not bump them to silence the warning.
FRONT_PAGE_LINE_BUDGET = 250
FRONT_PAGE_SECTION_BUDGET = 12
MAX_LEAD_RESTATEMENTS = 2

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
    "showcase_sync": 1.0,
    "jargon_density": 0.5,
    # substance
    "guard_prominence": 1.5,
    "lcd_onramp": 1.5,
    "audience_footholds": 1.5,
    "front_page_focus": 1.5,
    "speed_claim": 2.0,
    "hero_above_fold": 2.0,
}

# The composite SCORE is a TWO-SIDED UNBOUNDED focus index, not a percentage. 100
# is the ZERO-POINT — a correct, complete page sitting EXACTLY at the size budget.
# The score falls WITHOUT FLOOR as defects and bloat accumulate (a badly over-budget
# or broken page scores below zero) AND rises WITHOUT CEILING as a complete page
# gets leaner (every line under budget on a complete page adds a point). This is
# deliberate, matching the cadence report's unbounded `standing_score`: a bounded
# 0-100 score saturated at "A" and floored at 0 made a page 5 lines over budget and
# one 200 over look identical, and — worse — made a complete page park at a fake
# 100 with no gradient left to reward the next trim. The index is MAGNITUDE-AWARE
# in BOTH directions — every line over budget, every extra section, every repeated
# lead, every dead link subtracts real points; every line a complete page comes in
# under budget adds one — so the number keeps moving instead of freezing. Hygiene
# still GATES `ok` (a FAIL flips it False) and VOIDS the leanness credit; it no
# longer pads the score. `readme_debt` stays the house-standard lower-is-better
# defect COUNT (unchanged); the score is 100 minus the unbounded penalty plus the
# unbounded leanness credit below.
SUBSTANCE_CHECKS = {
    "guard_prominence", "lcd_onramp", "audience_footholds",
    "front_page_focus", "speed_claim", "hero_above_fold",
}

# Score weights for the TWO-SIDED UNBOUNDED index: score = round(100 - penalty
# + credit). 100 is NOT a ceiling — it is the ZERO-POINT: a clean, complete page
# sitting EXACTLY at the line budget. Bloat and defects push BELOW it without
# floor (penalty); a leaner complete page rises ABOVE it without ceiling (credit),
# so the score never saturates at a "fake perfect" 100 — every line trimmed off a
# complete page still moves it. The size axis is SIGNED and symmetric around the
# budget: one line over costs LINE_OVER_PENALTY, one line under a *complete* page
# earns LINE_UNDER_CREDIT (see _score_credit).
#   HYGIENE_FAIL_PENALTY: one hygiene FAIL (dead link / stale pin / naive lead)
#     lands the page exactly on the old FAIL_SCORE_CAP; a second FAIL takes it
#     below, unbounded — a broken page can no longer hide near a passing grade.
#   *_OVER_PENALTY: the UNBOUNDED, magnitude-aware bloat terms — per line over the
#     line budget, per section over the section budget, per excess preamble lead.
#   LINE_UNDER_CREDIT: the mirror term — per line a COMPLETE page comes in UNDER
#     the budget, scaled by affordance completeness and voided by any hygiene FAIL
#     (a lean-but-empty or lean-but-broken page earns nothing). Bloat is always
#     punished; leanness is rewarded only once the page is correct and complete.
#   SUBSTANCE_SHORTFALL_SCALE: a fully-missing substance affordance costs its
#     WEIGHT x this (so a missing hero, weight 2.0, costs 20 points).
FAIL_SCORE_CAP = 55
HYGIENE_FAIL_PENALTY = 100 - FAIL_SCORE_CAP  # 45: one FAIL == the old cap
LINE_OVER_PENALTY = 1.0        # per README line beyond FRONT_PAGE_LINE_BUDGET
LINE_UNDER_CREDIT = 1.0        # per line a COMPLETE page is UNDER budget (x completeness)
SECTION_OVER_PENALTY = 8.0     # per `## ` section beyond FRONT_PAGE_SECTION_BUDGET
LEAD_OVER_PENALTY = 10.0       # per preamble lead restatement beyond MAX_LEAD_RESTATEMENTS
OVERFLOW_MISSING_PENALTY = 5.0  # the front page links no overflow / deep-dive sink
SUBSTANCE_SHORTFALL_SCALE = 10.0

# The freshness stamp grammar the /refresh-readme skill writes.
#   <!-- readme-verified: 2026-06-20 vs VERSION 0.25.0 + BENCHMARK-AUTHORITY -->
_STAMP_RE = re.compile(
    r"<!--\s*readme-verified:\s*(\d{4}-\d{2}-\d{2})\b(?P<rest>[^>]*)-->",
    re.IGNORECASE,
)

_QWEN_FRONTDOOR_RE = re.compile(
    r"<!--\s*qwen38-frontdoor:begin\s*-->(.*?)"
    r"<!--\s*qwen38-frontdoor:end\s*-->",
    re.IGNORECASE | re.DOTALL,
)

# A Markdown inline link: [text](target). We only resolve LOCAL targets — http(s),
# mailto, and pure #anchors are out of scope (the network is not ours to witness).
_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")

# A fak version string: a bare semver, optionally v-prefixed. We compare the
# MAJOR.MINOR.PATCH against the VERSION file. A pin like "v0.3.x" or "v0.25.x"
# (a deliberate range) is matched on its leading numeric part.
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)\.(?:(\d+)|x)\b")

# A line-leading bare `fak <verb>` command — the invocation an LCD reader pastes
# from the installed binary (a `$`/`>` prompt or fence indent is allowed before
# it; `go run ./cmd/fak …` is deliberately NOT matched — that needs the clone).
# The captured verb is cross-checked against the real cmd/fak dispatch set.
_BARE_FAK_CMD_RE = re.compile(r"^\s*(?:[$>]\s+)?fak\s+([A-Za-z][\w-]*)", re.MULTILINE)

# The main switch's verb cases. We intentionally parse string literals from the source of
# truth instead of maintaining a duplicate verb list in Python.
_MAIN_CASE_RE = re.compile(r"^\s*case\s+(?P<body>[^:]+):", re.MULTILINE)
_GO_STRING_RE = re.compile(r'"([^"]+)"')

# A bolded headline claim: **…** (the front page leads its numbers in bold).
_BOLD_RE = re.compile(r"\*\*(?P<body>[^*]+)\*\*")

# Inside a bold span, a multiplier headline number like "60×" / "~4x" / "5.3–7.4×".
_MULT_RE = re.compile(r"~?\d[\d.,]*\s*(?:[–-]\s*\d[\d.,]*\s*)?[×x]")

# A claim number that must be traceable when it appears on the front page:
# multipliers, latency figures, and concrete token-throughput rates.
_CLAIM_NUMBER_RE = re.compile(
    r"~?\d[\d.,]*\s*(?:[–-]\s*\d[\d.,]*\s*)?[×x](?!\w)"
    r"|~?\d[\d.,]*\s*(?:ns(?:/op)?|nanoseconds?|µs|μs|microseconds?|"
    r"tok/s|tok/sec|tokens?\s*/\s*s(?:ec)?|tokens?\s+per\s+second)",
    re.IGNORECASE,
)

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

# A top-level `## ` heading (NOT `### `). Counts the front page's section sprawl.
_H2_RE = re.compile(r"^##\s+\S", re.MULTILINE)

# The lead pitch's signature phrases. A page that restates "one binary in front
# of the agent you already run" three times before the first section is the
# regrowth pattern this catches — so we count these ONLY in the preamble (above
# the first `## `), where section bodies that legitimately re-say the pitch (e.g.
# "Get started") are out of scope.
_LEAD_SIGNATURE_RE = re.compile(
    r"one (?:static )?(?:go )?binary"
    r"|in front of (?:the |an? )?(?:ai )?agent"
    r"|agent you already run"
    r"|drop-?in",
    re.IGNORECASE,
)

# --- the second front door (docs/showcase.html) ----------------------------
# Raw-HTML reduction, in the ONE order that is correct (see html_text).
_HTML_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
_HTML_RAWTEXT_RE = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")

# The published one-line installer. Both front doors hand the reader this exact
# command, so a drift here sends a homepage visitor at a different script than the
# one the README stands behind.
_INSTALL_CMD_RE = re.compile(r"curl\s+-fsSL\s+(\S+?)\s*\|\s*sh")

# A `fak vX.Y.Z` version the showcase prints next to a number.
_FAK_VERSION_RE = re.compile(r"\bfak\s+v(\d+\.\d+\.\d+)\b", re.IGNORECASE)

# The as-of version a benchmark dataset stamps on its own numbers.
_DATA_FAK_VERSION_RE = re.compile(r'"fak_version"\s*:\s*"v?(\d+\.\d+\.\d+)"')

# The showcase must name the same single source of truth the README's headline
# numbers are traced against, so both front doors answer to one authority doc.
_AUTHORITY_LINK_RE = re.compile(re.escape(AUTHORITY_REL), re.IGNORECASE)


def html_text(html: str) -> str:
    """Reader-visible text of a raw HTML page (comments, script, style, tags out).

    THE ORDER IS LOAD-BEARING. Comments are stripped FIRST because
    ``docs/showcase.html``'s own header comment contains the literal string
    ``<script>`` (it tells a maintainer which block to re-sync when the benchmark
    data changes). Reducing script/style first therefore matches from *inside that
    comment* all the way to the page's real ``</script>`` and swallows the entire
    page — 42 KB in, 1 KB out. That failure is silent and it is the dangerous kind:
    an empty extraction makes every cross-check below vacuously true, so the gate
    would report a clean sync for a page it never actually read.
    """
    text = _HTML_COMMENT_RE.sub(" ", html)
    text = _HTML_RAWTEXT_RE.sub(" ", text)
    return _HTML_TAG_RE.sub(" ", text)


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
    """WARN if a bolded headline number is not also in the authority doc.

    Not a hard gate: prose may round or restate. We just assert the front page
    mirrors the single source of truth (BENCHMARK-AUTHORITY), surfacing any bolded
    multiplier / latency / rate number that has no matching figure there to be
    reconciled by hand.
    """
    # The generated Qwen block has a stronger, graph-derived authority check
    # below. Do not make its scoped values pretend to be ledger headlines.
    scan = _QWEN_FRONTDOOR_RE.sub("", readme)
    missing: list[str] = []
    for m in _BOLD_RE.finditer(scan):
        missing.extend(_trace_claim_numbers(m.group("body"), authority)["missing"])
    missing = sorted(set(missing))
    if missing:
        return {
            "check": "headline_authority", "status": "WARN",
            "detail": "bolded headline number(s) not found in BENCHMARK-AUTHORITY — reconcile",
            "items": missing,
        }
    return {"check": "headline_authority", "status": "OK",
            "detail": "every bolded headline number mirrors an authority figure"}


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


def check_hardware_front_page(readme: str, *, today: _dt.date,
                              max_age_days: int) -> dict[str, Any]:
    """Keep the README's first result view limited to the latest hardware rows."""
    match = re.search(
        r"^## Latest hardware results — (\d{4}-\d{2}-\d{2})\s*$"
        r"(.*?)(?=^##\s|\Z)",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return {"check": "hardware_front_page", "status": "FAIL",
                "detail": "missing dated 'Latest hardware results' section"}

    missing: list[str] = []
    try:
        stamped = _dt.date.fromisoformat(match.group(1))
    except ValueError:
        return {"check": "hardware_front_page", "status": "FAIL",
                "detail": "latest hardware results heading has an invalid date"}
    age = (today - stamped).days
    if age < 0 or age > max_age_days:
        missing.append("fresh_date")

    body = match.group(2)
    rows: dict[str, str] = {}
    extra_rows: list[str] = []
    for line in body.splitlines():
        if not line.startswith("|") or re.match(r"^\|\s*:?-+", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0].lower() == "platform":
            continue
        platform = re.sub(r"[*_`]", "", cells[0]).strip()
        if platform in {"Mac", "AMD", "NVIDIA"}:
            rows[platform] = line
        else:
            extra_rows.append(platform or line)

    if set(rows) != {"Mac", "AMD", "NVIDIA"}:
        missing.append("exactly Mac, AMD, NVIDIA rows")
    if extra_rows:
        missing.append("no extra result rows")
    if not re.search(r"newest\s+committed performance receipt", body, re.IGNORECASE):
        missing.append("visible latest definition")
    if "docs/benchmarks/README.md" not in body:
        missing.append("benchmark/history index")
    if "qwen38-frontdoor" in readme.lower():
        missing.append("README must not contain generated qwen38-frontdoor markers")

    row_requirements = {
        "Mac": ("docs/benchmarks/QWEN38-27B-LATEST.md", ("historical", "expired", "hold", "no accepted")),
        "AMD": ("docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md", ("microbench", "narrow", "hold", "no accepted")),
        "NVIDIA": ("docs/_witnesses/issue-8819-qwen38-cache-attribution/README.md", ("held", "failed", "diagnostic", "no accepted")),
    }
    for platform, (detail, qualifiers) in row_requirements.items():
        row = rows.get(platform, "").lower()
        if detail.lower() not in row:
            missing.append(f"{platform} detail link")
        if row and not any(word in row for word in qualifiers):
            missing.append(f"{platform} qualification/hold")

    status = "OK" if not missing else "FAIL"
    detail = (f"latest hardware view dated {match.group(1)} ({age}d old) has only Mac, AMD, NVIDIA"
              if not missing else "hardware front page needs: " + ", ".join(missing))
    return {"check": "hardware_front_page", "status": status,
            "detail": detail, "items": missing + extra_rows}


def check_qwen_result_docs(qwen_index: str, qwen_latest: str) -> dict[str, Any]:
    """Keep generated Qwen detail surfaces linked without owning the README."""
    missing: list[str] = []
    for label, body in (("index", qwen_index), ("latest", qwen_latest)):
        if not body:
            missing.append(f"{label}_document")
            continue
        if "<!-- qwen38-frontdoor:begin -->" not in body or "<!-- qwen38-frontdoor:end -->" not in body:
            missing.append(f"{label}_generated_block")
    if qwen_index and "QWEN38-27B-LATEST.md" not in qwen_index:
        missing.append("index_to_latest_link")
    if qwen_latest and "QWEN-PERFORMANCE-INDEX.md" not in qwen_latest:
        missing.append("latest_to_index_link")
    status = "OK" if not missing else "FAIL"
    detail = ("Qwen index and latest detail keep generated blocks and reciprocal routing"
              if not missing else "Qwen result docs need: " + ", ".join(missing))
    return {"check": "qwen_result_docs", "status": status,
            "detail": detail, "items": missing}

def check_showcase_sync(readme: str, showcase: str | None, *,
                        version: str = "",
                        dataset_versions: set[str] | None = None,
                        dispatch: set[str] | None = None) -> dict[str, Any]:
    """The OTHER front door cannot drift away from this one, or go unlinked.

    GitHub's repo-level homepage setting points at the Pages copy of
    ``docs/showcase.html``, so a meaningful share of visitors meet *that* page
    first. It is hand-authored, it is not Markdown, and until this check existed
    nothing read it: ``readme_freshness_audit`` audited only ``README.md``, and
    ``internal/seoaeoscore`` opens the file solely to harvest its JSON-LD blocks
    (its page scorer is ``.md``-only *by assertion* — see
    ``seoaeoscore_test.go``'s ``published(root, "docs/showcase.html")`` case). Two
    front doors, one of them invisible to the people tending the other.

    This is the missing seam. Every affordance below is MECHANICAL — no judgment,
    no prose grading — and each names a way the two doors could tell a visitor
    different things:

      linked_from_readme  the README names the showcase, so a maintainer editing
                            the front page can see the other front door exists
      install_matches     every ``curl … | sh`` installer the showcase publishes is
                            one the README publishes too (rename install.sh and the
                            homepage would keep pointing at the old script)
      first_run_matches   a line-leading ``fak <verb>`` on the showcase is BOTH a
                            verb the binary really dispatches AND one the README
                            teaches — the same anti-gaming cross-check ``lcd_onramp``
                            applies to the README, now applied to the homepage
      authority_linked    the showcase names ``BENCHMARK-AUTHORITY.md``, so both
                            doors answer to the one source ``headline_authority``
                            already traces the README's numbers against
      version_traced      every ``fak vX.Y.Z`` the showcase prints is either the
                            current ``VERSION`` or the as-of version some in-repo
                            benchmark dataset stamps on its own numbers. This is the
                            subtle one: the chart caption's ``fak v0.30.0`` is NOT a
                            stale pin — ``tools/hero_benchmark.data.json`` says the
                            run was measured at 0.30.0, and quoting the measurement's
                            version is correct. What is *not* correct is a version no
                            dataset and no VERSION file stands behind, which is
                            exactly what regenerating the dataset without re-syncing
                            the page would produce.

    Hygiene tier: ANY missing affordance is a FAIL, because each one is a concrete
    thing the homepage would be saying that the repository does not back. When the
    page cannot be read at all (``showcase is None`` — the tool run outside the
    repo) the check ABSTAINS to WARN, the same posture ``lcd_onramp`` takes when
    ``cmd/fak/main.go`` is unreadable: a missing source of truth is not a defect.
    """
    if showcase is None:
        return {
            "check": "showcase_sync", "status": "WARN",
            "detail": (f"no {SHOWCASE_REL} to cross-check (run from the repo ROOT to "
                       "audit the published homepage against this page)"),
        }
    text = html_text(showcase)
    cur = _parse_version(version)

    readme_installs = set(_INSTALL_CMD_RE.findall(readme))
    page_installs = set(_INSTALL_CMD_RE.findall(text))

    readme_verbs = set(_BARE_FAK_CMD_RE.findall(readme))
    page_verbs = set(_BARE_FAK_CMD_RE.findall(text))
    shared_verbs = page_verbs & readme_verbs
    # With a known dispatch set a shared verb must also be REAL; without one (the
    # source unreadable) abstain to "the two pages agree on some command".
    real_shared = (shared_verbs & dispatch) if dispatch else shared_verbs

    known_versions = set(dataset_versions or ())
    if cur is not None:
        known_versions.add("%d.%d.%d" % cur)
    page_versions = set(_FAK_VERSION_RE.findall(text))
    untraced_versions = sorted(page_versions - known_versions)

    subs = {
        "linked_from_readme": (SHOWCASE_REL in readme)
        or (Path(SHOWCASE_REL).name in readme),
        "install_matches": bool(page_installs) and page_installs <= readme_installs,
        "first_run_matches": bool(real_shared),
        "authority_linked": bool(_AUTHORITY_LINK_RE.search(showcase)),
        "version_traced": not untraced_versions,
    }
    out = _grade_subs(
        "showcase_sync", subs, fail_if_zero=True, warn_below=1.0,
        label=(f"the configured homepage ({SHOWCASE_REL}) is linked from the README "
               "and agrees with it"))
    if out["items"]:
        # Hygiene tier: a partially-synced front door is a required edit, not a
        # judgment call, so any shortfall is a FAIL (not the _grade_subs WARN).
        out["status"] = "FAIL"
        bits = []
        if untraced_versions:
            bits.append("version(s) no dataset or VERSION backs: "
                        + ", ".join(untraced_versions))
        if page_installs - readme_installs:
            bits.append("installer(s) the README does not publish: "
                        + ", ".join(sorted(page_installs - readme_installs)))
        if bits:
            out["detail"] += " — " + "; ".join(bits)
    return out


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
                     one_glance_lines: int,
                     dispatch: set[str] | None = None) -> dict[str, Any]:
    """The lowest-common-denominator reader gets a no-setup first command.

    The LCD reader landed from a link, will not read prose, has no key, and
    wants one line that visibly does something. Affordances:
      one_glance_value  a single plain "what is this" sentence in the first ~8 lines
      fenced_cmd        a copy-paste fenced block above the fold
      bare_binary_cmd   a bare `fak <verb>` command (works from the binary, no clone)
                          whose verb RESOLVES against cmd/fak/main.go — the one
                          anti-gaming cross-check: a made-up verb the reader's first
                          paste would die on does NOT score (see module docstring).
                          ``dispatch`` is the real verb set; when it is None/empty
                          (source unreadable) the check abstains to presence-only.
      expected_output   the expected result shown inline (`# -> DENY`, →, …)
      no_setup_promise  "no key / no model / no GPU / no clone" stated above the fold
      install_reachable how to GET the binary (curl|go install|Install link) is nearby
    """
    lines = readme.splitlines()
    head = lines[:first_screen_lines]
    headtext = "\n".join(head)
    glance = "\n".join(lines[:one_glance_lines])
    # Every line-leading bare `fak <verb>` an LCD reader would paste from the
    # front screen (NOT `go run ./cmd/fak …`, which needs the clone).
    bare_verbs = _BARE_FAK_CMD_RE.findall(headtext)
    subs = {
        # A one-glance value = a blockquote/bold one-liner OR an explicit "in one
        # line" marker within the first few lines (above any long paragraph).
        "one_glance_value": ("one line" in glance.lower())
        or bool(re.search(r"^\s*>\s*\*\*", glance, re.MULTILINE))
        or bool(re.search(r"^\s*\*\*[^*]+\*\*\s*$", glance, re.MULTILINE)),
        "fenced_cmd": "```" in headtext,
        # Anti-gaming cross-check: a bare command scores only if its verb is real.
        # With a known dispatch set, a stuffed `fak <made-up-verb>` does not score;
        # without one (tool run outside the repo) abstain to presence-only.
        "bare_binary_cmd": (any(v in dispatch for v in bare_verbs)
                            if dispatch else bool(bare_verbs)),
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
                        on the same line/sentence as that number
      bounded       a fence near it (vs tuned / single-stream / in-process / not wall-clock)
                        so the speed isn't overclaimed
      links_authority  the first screen links to the benchmark authority/benchmarks doc
    """
    head = "\n".join(readme.splitlines()[:first_screen_lines])
    trace = _trace_claim_numbers(head, authority)
    traced = bool(trace["traced"])
    marked = _claim_marked_near_number(head)
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


def check_front_page_focus(readme: str, *, line_budget: int,
                           section_budget: int, max_lead: int) -> dict[str, Any]:
    """The size law, made checkable — the counterweight to the ADD-only checks.

    Every other substance check rewards putting one more thing on the page. This
    one rewards keeping the page small and un-repetitive, so the composite score
    reflects concision, not just completeness. Affordances:
      within_line_budget   README.md total length <= line_budget
      sections_bounded     the count of top-level `## ` sections <= section_budget
      single_lead          the pitch is restated <= max_lead times in the PREAMBLE
                             (above the first `## `) — not three times before the
                             reader reaches a section
      overflow_linked      the page links an overflow/deep-dive sink (README-legacy /
                             "Going deeper"), proving detail has somewhere to flow OUT
                             instead of accreting on the front page

    Never a hard FAIL (fail_if_zero=False): an over-budget page is DRIFT, not a
    broken page. It shows up as debt + a lower score, which is the steady
    downward pressure the front page lacked.
    """
    lines = readme.splitlines()
    total = len(lines)
    n_sections = len(_H2_RE.findall(readme))
    # Preamble = everything before the first `## ` heading (the lead region).
    first_h2 = next((i for i, ln in enumerate(lines) if _H2_RE.match(ln)), len(lines))
    lead_hits = sum(1 for ln in lines[:first_h2] if _LEAD_SIGNATURE_RE.search(ln))
    # The UNBOUNDED magnitude behind the unbounded score: HOW FAR over, not just
    # a yes/no. A page 5 lines over and one 200 over both fail within_line_budget,
    # but they carry very different overage — carried out for the score penalty.
    lines_over = max(0, total - line_budget)
    # The mirror of lines_over: HOW FAR UNDER budget a page is. A complete page
    # earns leanness credit for this headroom (see _score_credit), so the score
    # keeps rising as a good page gets tighter instead of parking at 100.
    lines_under = max(0, line_budget - total)
    sections_over = max(0, n_sections - section_budget)
    lead_over = max(0, lead_hits - max_lead)
    overflow_linked = bool(re.search(
        r"README-legacy|overflow|going deeper", readme, re.IGNORECASE))
    subs = {
        "within_line_budget": lines_over == 0,
        "sections_bounded": sections_over == 0,
        "single_lead": lead_over == 0,
        "overflow_linked": overflow_linked,
    }
    out = _grade_subs(
        "front_page_focus", subs, fail_if_zero=False,
        label=(f"front page stays focused ({total}/{line_budget} lines, "
               f"{n_sections}/{section_budget} sections, "
               f"{lead_hits} preamble lead restatement(s))"))
    # Attach the raw overage (and the under-budget headroom) so the score can
    # weight both directions by magnitude: _score_penalty reads the *_over fields,
    # _score_credit reads lines_under.
    out.update({"lines_over": lines_over, "lines_under": lines_under,
                "sections_over": sections_over,
                "lead_over": lead_over, "overflow_linked": overflow_linked})
    return out


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


def _claim_numbers(s: str) -> list[str]:
    """Front-page numeric claims that need authority: multipliers, latency, rates."""
    return [m.group(0).strip() for m in _CLAIM_NUMBER_RE.finditer(s or "")]


def _norm_claim_num(s: str) -> str:
    """Normalize a claim number for README-vs-authority comparison."""
    raw = s.strip()
    if re.search(r"[×x]\s*$", raw):
        return _norm_num(raw).lower()
    t = raw.lower().replace("μ", "µ")
    t = re.sub(r"[~,\s]", "", t)
    t = t.replace("tok/sec", "tok/s")
    t = re.sub(r"tokens?/s(?:ec)?", "tok/s", t)
    t = re.sub(r"tokens?persecond", "tok/s", t)
    t = t.replace("nanoseconds", "ns").replace("nanosecond", "ns")
    t = t.replace("microseconds", "µs").replace("microsecond", "µs")
    return t


def _trace_claim_numbers(text: str, authority: str) -> dict[str, list[str]]:
    """Trace front-page numbers against BENCHMARK-AUTHORITY with one shared parser."""
    auth_nums = {_norm_claim_num(x) for x in _claim_numbers(authority or "")}
    nums = _claim_numbers(text or "")
    traced = [n for n in nums if _norm_claim_num(n) in auth_nums]
    missing = [n for n in nums if _norm_claim_num(n) not in auth_nums]
    return {
        "numbers": nums,
        "traced": sorted(set(traced)),
        "missing": sorted(set(missing)),
    }


def _claim_marked_near_number(text: str) -> bool:
    """True when an honesty marker lives on the same line/sentence as a number."""
    markers = [
        "observed", "relayed", "telemetry", "provider's own", "/metrics", "measured",
    ]
    for line in (text or "").splitlines():
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if _claim_numbers(sent) and any(k in sent.lower() for k in markers):
                return True
    return False


def _check_score(c: dict[str, Any]) -> float:
    """A check's 0..1 contribution: its graded `score` if present, else by status."""
    s = c.get("score")
    if isinstance(s, (int, float)):
        return float(s)
    return {"OK": 1.0, "WARN": 0.5, "FAIL": 0.0, "ADVISORY": 1.0}.get(c["status"], 0.0)


def _score_penalty(checks: list[dict[str, Any]]) -> float:
    """The unbounded, magnitude-aware penalty behind ``score = 100 - penalty``.

    Zero penalty is a clean, complete, LEAN page (score 100). The penalty has no
    upper bound, so the score has no lower bound:

      * each hygiene FAIL adds ``HYGIENE_FAIL_PENALTY`` (one FAIL == the old cap;
        a second takes the page below it — a broken page cannot hide near passing);
      * ``front_page_focus`` contributes the UNBOUNDED magnitude terms — per line
        over the line budget, per section over the section budget, per excess
        preamble lead, plus a fixed hit if no overflow sink is linked. This is the
        magnitude a boolean ``within_line_budget`` used to discard;
      * every other substance shortfall costs its WEIGHT x ``SUBSTANCE_SHORTFALL_SCALE``.

    A ``front_page_focus`` dict without the overage fields (a hand-built test
    fixture) contributes zero magnitude — it is treated as a lean page.
    """
    penalty = 0.0
    for c in checks:
        chk = c.get("check")
        if chk not in SUBSTANCE_CHECKS:
            if c.get("status") == "FAIL":
                penalty += HYGIENE_FAIL_PENALTY
            continue
        if chk == "front_page_focus":
            penalty += c.get("lines_over", 0) * LINE_OVER_PENALTY
            penalty += c.get("sections_over", 0) * SECTION_OVER_PENALTY
            penalty += c.get("lead_over", 0) * LEAD_OVER_PENALTY
            if not c.get("overflow_linked", True):
                penalty += OVERFLOW_MISSING_PENALTY
        else:
            penalty += WEIGHTS.get(chk, 0.5) * (1.0 - _check_score(c)) * SUBSTANCE_SHORTFALL_SCALE
    return penalty


def _add_substance_completeness(checks: list[dict[str, Any]]) -> float:
    """Mean 0..1 completeness over the five ADD substance checks.

    ``front_page_focus`` is excluded: it IS the size axis the credit rewards, so
    folding it in would be circular. A page with no add-substance checks at all
    (a bare hand-built fixture) is treated as 0% complete — it earns no credit.
    """
    scores = [
        _check_score(c) for c in checks
        if c.get("check") in SUBSTANCE_CHECKS and c.get("check") != "front_page_focus"
    ]
    return (sum(scores) / len(scores)) if scores else 0.0


def _score_credit(checks: list[dict[str, Any]]) -> float:
    """The unbounded-ABOVE leanness credit behind ``score = 100 - penalty + credit``.

    The mirror of the over-budget penalty: a COMPLETE page earns points for every
    line it comes in UNDER the size budget, so the score keeps climbing as a good
    page gets tighter instead of parking at a fake-perfect 100. Two guards keep a
    small or broken page from farming it:

      * any hygiene FAIL VOIDS the credit entirely — a page must be *correct*
        before its leanness counts (bloat, by contrast, is always penalized);
      * the credit scales by ADD-affordance completeness, so a lean-but-empty page
        (few affordances) earns proportionally little — you cannot score high by
        deleting content, only by saying the same complete thing in fewer lines.

    A ``front_page_focus`` dict without ``lines_under`` (a hand-built fixture, or
    an over-budget page) contributes zero — the page is at or over budget.
    """
    # A broken page earns no leanness reward: fix it first.
    if any(c.get("status") == "FAIL" and c.get("check") not in SUBSTANCE_CHECKS
           for c in checks):
        return 0.0
    focus = next((c for c in checks if c.get("check") == "front_page_focus"), None)
    if focus is None:
        return 0.0
    lines_under = focus.get("lines_under", 0) or 0
    if lines_under <= 0:
        return 0.0
    return LINE_UNDER_CREDIT * lines_under * _add_substance_completeness(checks)


def _as_int(v: Any) -> int:
    """Coerce a payload field to int, tolerant of None / floats / strings."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _grade_letter(score: int) -> str:
    return ("A" if score >= 90 else "B" if score >= 80 else
            "C" if score >= 70 else "D" if score >= 60 else "F")


def _payload_corpus(payload: dict[str, Any]) -> dict[str, Any]:
    corpus = payload.get("corpus")
    return corpus if isinstance(corpus, dict) else {}


def _payload_score(payload: dict[str, Any]) -> int:
    corpus = _payload_corpus(payload)
    return _as_int(corpus.get("score", payload.get("score")))


def _payload_grade(payload: dict[str, Any], score: int | None = None) -> str:
    corpus = _payload_corpus(payload)
    grade = corpus.get("grade", payload.get("grade"))
    if grade:
        return str(grade)
    return _grade_letter(score if score is not None else _payload_score(payload))


def _payload_readme_debt(payload: dict[str, Any]) -> int | None:
    corpus = _payload_corpus(payload)
    for value in (payload.get("readme_debt"), corpus.get("readme_debt")):
        if value is not None:
            return max(0, _as_int(value))
    return None


def _readme_debt_from_checks(checks: list[dict[str, Any]]) -> int:
    """Count hard README debt units without reusing the good-is-high score."""
    hygiene_fails = sum(
        1 for c in checks
        if c.get("status") == "FAIL" and c.get("check") not in SUBSTANCE_CHECKS
    )
    substance_missing = 0
    for c in checks:
        if c.get("check") not in SUBSTANCE_CHECKS:
            continue
        items = c.get("items") or []
        substance_missing += len(items) if isinstance(items, list) else 1
    return hygiene_fails + substance_missing


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

    # Composite score = 100 - an UNBOUNDED penalty + an UNBOUNDED leanness credit
    # (see _score_penalty / _score_credit). 100 is the ZERO-POINT — a clean,
    # complete page at the size budget; bloat/defects push it below without floor,
    # and a leaner complete page rises above without ceiling, so it keeps
    # discriminating instead of saturating at a fake-perfect 100. It is NOT a
    # percentage. `scored`/`has_substance` remain only for the work-list ranking.
    substance = [c for c in checks if c["check"] in SUBSTANCE_CHECKS]
    scored = substance if substance else checks
    has_substance = bool(substance)
    score = round(100 - _score_penalty(checks) + _score_credit(checks))
    grade = _grade_letter(score)
    debt = _readme_debt_from_checks(checks)
    # "Thin" is now decided by real INCOMPLETENESS (a missing add-affordance), not
    # by score < 90: the leanness credit can lift an incomplete-but-lean page well
    # above 90, so keying the verdict on the score would let it dodge the flag.
    add_missing = sum(
        (len(c.get("items") or []) if isinstance(c.get("items"), list) and c.get("items")
         else (1 if _check_score(c) < 1.0 else 0))
        for c in checks
        if c.get("check") in SUBSTANCE_CHECKS and c.get("check") != "front_page_focus"
    )

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
        reason = f"score {score} ({grade}); {len(fails)} required README fix(es): {names}"
        next_action = ("invoke /refresh-readme: each FAIL is a required edit (fix the dead link / "
                       "stale pin / naive-lead headline / missing onramp), then re-stamp and re-run")
    elif has_substance and add_missing > 0:
        ok, verdict, finding = True, "OK", "readme_fresh_thin"
        reason = (f"score {score} ({grade}): front page is correct but THIN — "
                  f"{add_missing} missing affordance(s); raise the substance checks ({worst})")
        next_action = (f"invoke /refresh-readme: add the missing affordances (worst: {worst}), "
                       "then re-stamp readme-verified and re-run")
    elif score < 100:
        # Complete (every add-affordance present) but BELOW the 100 lean-complete
        # reference: over the size budget and/or carrying judgment-call WARNs. The
        # fix is to TRIM (and/or clear the WARNs), not to add — so this is notes,
        # not thin.
        ok, verdict, finding = True, "OK", "readme_fresh_with_notes"
        names = ", ".join(c["check"] for c in warns)
        bits = [f"trim toward the {FRONT_PAGE_LINE_BUDGET}-line budget"]
        if warns:
            bits.append(f"clear {len(warns)} WARN(s): {names}")
        reason = (f"score {score} ({grade}): complete but below the 100 lean "
                  f"reference — {'; '.join(bits)}")
        next_action = ("invoke /refresh-readme: trim the front page toward budget"
                       + (" and review the WARNs" if warns else "")
                       + ", then re-stamp readme-verified and re-run")
    elif warns:
        ok, verdict, finding = True, "OK", "readme_fresh_with_notes"
        names = ", ".join(c["check"] for c in warns)
        reason = f"score {score} ({grade}); no required fix; {len(warns)} judgment-call WARN(s): {names}"
        next_action = "review each WARN at the next /refresh-readme pass; no blocking edit needed"
    else:
        ok, verdict, finding = True, "OK", "readme_fresh"
        reason = (f"score {score} ({grade}): front page is correct, complete, AND lean — "
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
        "readme_debt": debt,
        "corpus": {
            "score": score,
            "grade": grade,
            "readme_debt": debt,
        },
        "next_action": next_action,
        "workspace": workspace,
        "counts": counts,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Before/after compare — prove a refresh pass actually lifted the front page.
# Mirrors internal/dogfoodscore Compare() and tools/industry_scorecard
# render_compare: a single *_debt integer (lower is better, zero is perfect),
# folded before->after with the family's >=2x / >=3x improvement verdict.
# ---------------------------------------------------------------------------

def readme_debt(payload: dict[str, Any]) -> int:
    """Return the payload's debt integer; tolerate pre-/2 baselines."""
    debt = _payload_readme_debt(payload)
    if debt is not None:
        return debt
    # Legacy saved baselines before schema /2 had no debt field. Keep --compare
    # able to read them, but build_payload never emits score-as-debt.
    return max(0, 100 - _payload_score(payload))


def _compare_verdict(b_debt: int, c_debt: int) -> str:
    """The family's >=2x / >=3x improvement verdict over a debt before->after."""
    if b_debt <= 0:
        if c_debt > 0:
            return f"REGRESSED from a perfect baseline (debt 0 -> {c_debt}) - the front page lost ground"
        return "already perfect (debt 0 -> 0) - nothing to retire"
    if c_debt > b_debt:
        return f"REGRESSED (debt {b_debt} -> {c_debt}) - the front page got worse"
    if c_debt == b_debt:
        return f"no change (debt {b_debt} -> {c_debt})"
    if c_debt * 3 <= b_debt:
        return f">=3x improvement (debt {b_debt} -> {c_debt}, <= 1/3 of baseline)"
    if c_debt * 2 <= b_debt:
        return f">=2x improvement (debt {b_debt} -> {c_debt})"
    return f"improved but < 2x (debt {b_debt} -> {c_debt})"


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Pure before->after delta of readme_debt/score, with the family's verdict.

    ``current`` and ``baseline`` are both audit payloads (``build_payload``
    output). We fold their ``readme_debt`` + ``score`` + ``grade`` into a
    before->after report and a >=2x / >=3x improvement verdict, exactly as
    ``dogfoodscore.Compare`` and industry ``render_compare`` do. No disk, no
    clock: a pure dict fold, so the same two payloads always render the same
    string.
    """
    b_debt, c_debt = readme_debt(baseline), readme_debt(current)
    b_score, c_score = _payload_score(baseline), _payload_score(current)
    b_grade = _payload_grade(baseline, b_score)
    c_grade = _payload_grade(current, c_score)
    lines = [
        "readme-freshness compare:",
        f"  readme_debt: {b_debt} -> {c_debt}  (retired {b_debt - c_debt})",
        f"  score:       {b_score} -> {c_score}   grade {b_grade} -> {c_grade}  (100 = complete at budget; unbounded above + below)",
    ]
    # When both payloads carry hygiene counts, surface the FAIL delta too: a
    # regression that adds a dead link / stale pin shows here even if the
    # substance score happened to hold.
    if (baseline.get("counts") is not None) or (current.get("counts") is not None):
        b_fail = _as_int((baseline.get("counts") or {}).get("FAIL"))
        c_fail = _as_int((current.get("counts") or {}).get("FAIL"))
        lines.append(f"  hygiene FAILs: {b_fail} -> {c_fail}")
    lines.append("  VERDICT: " + _compare_verdict(b_debt, c_debt))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def fak_dispatch_verbs(root: Path) -> set[str] | None:
    """Parse cmd/fak/main.go for the real `fak <verb>` dispatch set.

    This is the anti-gaming witness for `lcd_onramp`: a front-screen bare command
    only earns the binary-command affordance when its verb is actually dispatched
    by the binary. If the source file cannot be read, return None so the check
    abstains to presence-only rather than punishing out-of-repo runs.
    """
    text = _safe_read(root / MAIN_GO_REL)
    if not text:
        return None
    verbs: set[str] = set()
    for m in _MAIN_CASE_RE.finditer(text):
        for verb in _GO_STRING_RE.findall(m.group("body")):
            if not verb or verb.startswith("-") or verb == "help":
                continue
            verbs.add(verb)
    return verbs or None


def run_checks(readme: str, version: str, authority: str, root: Path, *,
               today: _dt.date, max_age_days: int,
               dispatch: set[str] | None = None,
               showcase: str | None = None,
               dataset_versions: set[str] | None = None,
               qwen_index: str = "", qwen_latest: str = "") -> list[dict[str, Any]]:
    """All checks over already-read text. The pure core; tests call this."""
    return [
        # hygiene — is the page correct?
        check_links(readme, root),
        check_version_pins(readme, version),
        check_naive_baseline(readme),
        check_headline_authority(readme, authority),
        check_freshness_stamp(readme, today=today, max_age_days=max_age_days),
        check_hardware_front_page(readme, today=today, max_age_days=max_age_days),
        check_qwen_result_docs(qwen_index, qwen_latest),
        check_showcase_sync(readme, showcase, version=version,
                            dataset_versions=dataset_versions, dispatch=dispatch),
        check_jargon_density(readme, first_screen_lines=FIRST_SCREEN_LINES),
        # substance — does the page do its job for the whole audience?
        check_guard_prominence(readme, first_screen_lines=FIRST_SCREEN_LINES),
        check_lcd_onramp(readme, first_screen_lines=FIRST_SCREEN_LINES,
                         one_glance_lines=ONE_GLANCE_LINES, dispatch=dispatch),
        check_speed_claim(readme, authority, first_screen_lines=FIRST_SCREEN_LINES),
        check_hero_above_fold(readme, authority, first_screen_lines=FIRST_SCREEN_LINES),
        check_audience_footholds(readme, first_screen_lines=FIRST_SCREEN_LINES),
        check_front_page_focus(readme, line_budget=FRONT_PAGE_LINE_BUDGET,
                               section_budget=FRONT_PAGE_SECTION_BUDGET,
                               max_lead=MAX_LEAD_RESTATEMENTS),
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
    qwen_index = _safe_read(root / QWEN_INDEX_REL)
    qwen_latest = _safe_read(root / QWEN_LATEST_REL)
    dispatch = fak_dispatch_verbs(root)
    # None (not "") when the second front door is absent, so check_showcase_sync can
    # tell "no page to read" (abstain) apart from "an empty page" (a real defect).
    showcase = _safe_read(root / SHOWCASE_REL) or None
    checks = run_checks(readme, version, authority, root,
                        today=today, max_age_days=max_age_days, dispatch=dispatch,
                        showcase=showcase,
                        dataset_versions=bench_dataset_versions(root),
                        qwen_index=qwen_index, qwen_latest=qwen_latest)
    return build_payload(workspace=str(root), checks=checks)


def bench_dataset_versions(root: Path) -> set[str]:
    """Every fak version an in-repo benchmark dataset stamps as its own as-of.

    The showcase's chart caption prints ``fak v0.30.0`` beside the hero numbers.
    That is not a stale product pin: ``tools/hero_benchmark.data.json`` declares
    ``"fak_version": "0.30.0"`` as the version the run was measured at, and citing
    the measurement's version is the honest thing to do. So ``version_traced``
    accepts any version a dataset stands behind — and only those, plus the current
    ``VERSION``. Regenerate the dataset at a new version without re-syncing the
    page and the quoted version stops being backed by anything, which is precisely
    the drift the check exists to catch.
    """
    out: set[str] = set()
    for p in sorted(root.glob(BENCH_DATA_GLOB)):
        for m in _DATA_FAK_VERSION_RE.finditer(_safe_read(p)):
            out.add(m.group(1))
    return out


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"readme-freshness audit: {payload.get('verdict')} ({payload.get('finding')})  "
        f"score {payload.get('score')} ({payload.get('grade')}, 100=at-budget; "
        f"unbounded above=leaner, below=bloat/defects)",
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
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the readme_debt/score delta vs a prior baseline payload JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, max_age_days=args.max_age_days)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(compare(payload, baseline))
        # Ratchet semantics: a regression (debt rose) exits non-zero so the
        # delta can gate a refresh pass; flat or improved stays green.
        return 0 if readme_debt(payload) <= readme_debt(baseline) else 1

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
