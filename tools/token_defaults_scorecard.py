#!/usr/bin/env python3
"""Token-saving-defaults scorecard — is fak's out-of-the-box token economy *amazing*?

The sibling scorecards grade a surface a reviewer cares about: ``code_quality`` the
Go module, ``agent_readiness`` whether an agent can adopt fak, ``industry`` fak vs the
SOTA field. None of them grade the question a cost-conscious operator asks the moment
they point an agent at ``fak guard -- claude`` / ``fak serve``: **of every token-saving
method fak knows how to stack, which ones are ON by default — and are the high-value,
low-loss ones turned on out of the box, or left dark behind a flag nobody flips?**

That used to be a vibe ("we have compaction, we're fine"). This is the number.

The thesis (the operator's rule, made mechanical): a token-saving lever that **keeps
the model's working set intact and gives a large saving** belongs ON by default, with
an honest note about exactly what it sheds — not parked behind an opt-in flag. A lever
whose loss is *unbounded* stays opt-in, but must carry a documented gate naming the
witness it is waiting for. The score rewards the first and the honesty of the second.

It reads the REAL defaults straight from the entrypoint source — the ``fs.Int(...)`` /
``fs.Bool(...)`` flag defaults in ``cmd/fak/guard.go`` and ``cmd/fak/serve.go``, the
``Default*`` constants in ``internal/gateway/gateway.go``, and the audited
``servewiringData`` rows in ``cmd/fak/servewiring.go`` — so a lever's on/off state is
the binary's actual behavior, never a claim in this file. You cannot drop debt by
editing the roster: a lever the roster calls "on by default" must parse to a non-zero /
true default in the entrypoint, and a lever the roster calls "high-value, safe to
default" must carry the in-code GUARD that bounds its loss (e.g. the elision
working-set guard ``DefaultKeepRecentTurnsForElide``). Change the code, not the
detector.

  STACK       — the stacking savers are ON out of the box
    lossless_stack       every zero-loss saver (provider prompt-cache passthrough,
                         tool-floor pruning, vDSO dedup) is on by default
    high_value_defaults  every BOUNDED-LOSS saver whose in-code guard keeps the model's
                         working set intact (history compaction, oversized-result
                         elision) is on by default — the operator's "keep ~the model's
                         view, save big → default it" rule
  HONESTY     — the dark levers are dark on purpose, the on levers say what they shed
    dark_lever_gated     every OFF lever carries a documented gate (why it's off, a
                         safe value, the witness it needs)
    default_notes        every ON bounded-loss lever documents what it sheds + that it
                         is cache-prefix-safe and observable ("on by default WITH notes")
  REGRESSION  — the amazing defaults can't silently rot
    default_on_locked    every default-ON saver is pinned by a test, so a peer who
                         flips it back to off reds a gate
  PARITY      — both front doors offer the same economy
    entrypoint_parity    fak guard and fak serve agree on every shared lever's default
                         (and the audited servewiring verdict matches the real default)

The headline metric is **token_defaults_debt**: the count of concrete, re-derivable
HARD defects above. Driving it to zero means a user who runs fak with no flags gets the
full stack of safe savings, each one honestly labeled, none of it able to regress
unnoticed. The companion process — the ``/token-defaults-score`` skill — runs this,
retires the worst-first defect by **turning a high-value saver on (with notes) / writing
the lock / documenting the gate**, and re-runs to prove the drop. It folds into the
unified ``scorecard_control_pane`` alongside the other sticks.

Deterministic + read-only by construction: it reads the git-tracked tree (two clones at
one commit score identically) and edits nothing but a generated snapshot under
``--markdown``. Run from the repo ROOT::

    python tools/token_defaults_scorecard.py                 # human scorecard
    python tools/token_defaults_scorecard.py --json          # machine payload
    python tools/token_defaults_scorecard.py --markdown      # the committed snapshot body
    python tools/token_defaults_scorecard.py --compare base.json   # prove the debt moved
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "fak-token-defaults-scorecard/1"
GENERATED_SNAPSHOT = "docs/serving/token-defaults-scorecard.md"

# Source-of-truth files the scorecard parses for the REAL defaults. Each lever's on/off
# state is read from here, never from the roster below — that is the ungameable core.
GUARD_GO = "cmd/fak/guard.go"
SERVE_GO = "cmd/fak/serve.go"
GATEWAY_GO = "internal/gateway/gateway.go"
SERVEWIRING_GO = "cmd/fak/servewiring.go"

# ---------------------------------------------------------------------------
# The lever roster. Each entry NAMES a token-saving method and the in-code facts the
# scorecard cross-checks it against. The roster cannot grant a lever a status: the flag
# default is parsed from the entrypoint, the loss-bounding guard must exist in source,
# and the lock must be a real test. A fork that drops a saver or turns one off scores
# lower, by construction.
#
#   klass:
#     "lossless"  — zero model-visible change (cache reuse, pruning a provably-unreachable
#                   tool def, deduping an identical call). MUST be on by default.
#     "bounded"   — lossy, but the loss is bounded by an in-code guard that keeps the
#                   model's active working set intact (compaction drops only the
#                   un-cacheable middle; elision only shrinks results scrolled past the
#                   working-set window). High-value → SHOULD be on by default, with notes.
#     "optin"     — lossy with a broader/unbounded blast radius. Correctly OFF; must carry
#                   a documented gate naming the witness it waits for.
# ---------------------------------------------------------------------------


@dataclass
class Lever:
    key: str
    label: str
    klass: str                      # lossless | bounded | optin
    sw_feature: str                 # servewiringData feature key ("" if not a serve-wiring row)
    flag: str                       # CLI flag string ("" if structural / always-on)
    default_const: str              # gateway Default* constant resolving the default ("" if literal/structural)
    entrypoints: tuple[str, ...]    # which front doors expose it
    guard_symbol: str               # for "bounded": the source symbol that bounds the loss (must exist)
    guard_file: str                 # file the guard_symbol lives in
    note_tokens: tuple[str, ...]    # tokens the flag help / const comment must carry to count as "noted"
    sentinel_tokens: tuple[str, ...]  # tokens a *_test.go must carry to count as "locked"
    witness_tokens: tuple[str, ...] = ()  # tokens that, near a savings figure in a tracked doc, witness the tradeoff
    structural_field: str = ""      # for a no-flag lever: the gateway.Config field serve.go must set
    # populated by the impure shell:
    on: bool | None = field(default=None)        # resolved real default (None = could not resolve)
    default_repr: str = ""                        # how the default appears in source
    sw_verdict: str = ""                          # the audited servewiring verdict for sw_feature
    guard_present: bool = True                    # guard_symbol resolves in source
    gated: bool = False                           # OFF lever carries a documented gate
    noted: bool = False                           # ON bounded lever documents what it sheds
    locked: bool = False                          # a sentinel test pins the default
    witnessed: bool = False                       # a tracked doc quantifies its savings/fidelity tradeoff
    witness_note: str = ""                        # the witnessing snippet (savings figure), for the operator
    off_reason: str = ""                          # for an OFF lever: unwitnessed | witnessed_gated | ready


LEVERS: list[Lever] = [
    Lever(
        key="provider_cache",
        label="provider prompt-cache prefix (byte-faithful passthrough)",
        klass="lossless",
        sw_feature="",
        flag="",
        default_const="",
        entrypoints=("guard", "serve"),
        guard_symbol="StreamAnthropicRaw",
        guard_file="internal/gateway",
        note_tokens=(),
        # Locked by internal/gateway's passthrough test, which asserts the upstream body is
        # byte-identical to the inbound (so the client's prompt-cache prefix survives).
        sentinel_tokens=("TestAnthropicMessagesPassthroughStreamsLiveAndAdjudicates",),
        structural_field="",
    ),
    Lever(
        key="toolfloor",
        label="tool-floor pruning (drop provably-unreachable tool defs)",
        klass="lossless",
        sw_feature="toolfloor",
        flag="",
        default_const="",
        entrypoints=("guard", "serve"),
        guard_symbol="ToolFloorDenies",
        guard_file=GATEWAY_GO,
        note_tokens=(),
        sentinel_tokens=("ToolFloorDenies:",),  # the Config assignment, asserted in token_defaults_test.go
        structural_field="ToolFloorDenies",
    ),
    Lever(
        key="vdso",
        label="vDSO dedup fast path (collapse identical calls)",
        klass="lossless",
        sw_feature="vdso",
        flag="vdso",
        default_const="",
        entrypoints=("serve",),  # guard sets VDSO:true structurally (no flag)
        guard_symbol="",
        guard_file="",
        note_tokens=(),
        sentinel_tokens=('fs.Bool("vdso", true',),  # the real serve.go default, asserted in token_defaults_test.go
    ),
    Lever(
        key="compacthistory",
        label="history compaction (drop the un-cacheable middle past the budget)",
        klass="bounded",
        sw_feature="compacthistory",
        flag="compact-history-budget",
        default_const="DefaultCompactHistoryBudget",
        entrypoints=("guard", "serve"),
        guard_symbol="CompactReasonCachedSpan",  # the cache_control bail that bounds the loss
        guard_file="internal/agent",
        note_tokens=("cache_control prefix", "byte-identical"),
        sentinel_tokens=("DefaultCompactHistoryBudget",),
        witness_tokens=("compact",),  # experiments/agent-live + CLAIMS: 95.4% shed on a 100k session
    ),
    Lever(
        key="elideresult",
        label="oversized-result elision (shrink a scrolled-past tool_result to head+tail)",
        klass="bounded",
        sw_feature="elideresult",
        flag="elide-result-bytes",
        default_const="DefaultElideResultBytes",
        entrypoints=("guard", "serve"),
        guard_symbol="DefaultKeepRecentTurnsForElide",  # the working-set guard that bounds the loss
        guard_file="internal/agent",
        note_tokens=("working set", "cache_control prefix"),
        sentinel_tokens=("DefaultElideResultBytes",),
        witness_tokens=("elide-result",),  # NO savings/fidelity witness yet — ships dark on purpose (not "elision": that word also names the unrelated ctxplan KV-eviction bridge)
    ),
    Lever(
        key="ctxview",
        label="ctxplan O(1) planned view (re-materialize history under a budget)",
        klass="optin",
        sw_feature="ctxview",
        flag="ctx-view-budget",
        default_const="",
        entrypoints=("guard", "serve"),
        guard_symbol="",
        guard_file="",
        note_tokens=(),
        sentinel_tokens=('fs.Int("ctx-view-budget", 0',),  # pinned dark in token_defaults_test.go
        witness_tokens=("ctxplan", "resident"),  # CLAIMS/ctxplanbench: 13.3x fewer resident, 100% exact recall
    ),
]

GROUPS = ("stack", "honesty", "regression", "parity")
KPI_GROUP: dict[str, str] = {
    "lossless_stack": "stack",
    "high_value_defaults": "stack",
    "dark_lever_gated": "honesty",
    "default_notes": "honesty",
    "witness_status": "honesty",
    "default_on_locked": "regression",
    "entrypoint_parity": "parity",
    "stacking_depth": "stack",
}
# Weights sum to exactly 1.0 (score can reach 100). A regression test asserts both the
# sum and that the weight set == the KPI set.
KPI_WEIGHTS: dict[str, float] = {
    "lossless_stack": 0.20,
    "high_value_defaults": 0.20,
    "dark_lever_gated": 0.12,
    "default_notes": 0.08,
    "witness_status": 0.07,
    "default_on_locked": 0.16,
    "entrypoint_parity": 0.12,
    "stacking_depth": 0.05,
}


# ---------------------------------------------------------------------------
# Pure helpers (the testable core).
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


_CONST_RE = re.compile(r"\bconst\s+(Default[A-Za-z0-9]*)\s*=\s*(\d+)\b")
_FLAG_RE = re.compile(r'fs\.(Int|Bool)\(\s*"([^"]+)"\s*,\s*([^,]+?)\s*,')


def parse_const_ints(gateway_src: str) -> dict[str, int]:
    """Every ``const Default<X> = <int>`` in gateway.go — the symbolic flag defaults."""
    return {m.group(1): int(m.group(2)) for m in _CONST_RE.finditer(gateway_src or "")}


def parse_flag_defaults(src: str) -> dict[str, tuple[str, str]]:
    """Map flag name -> (kind, default_expr) for every ``fs.Int``/``fs.Bool`` in a source
    file. The default_expr is the raw second argument (``0``, ``true``,
    ``gateway.DefaultElideResultBytes``) — resolved separately so the resolver, not the
    regex, owns the symbolic lookup."""
    out: dict[str, tuple[str, str]] = {}
    for m in _FLAG_RE.finditer(src or ""):
        out[m.group(2)] = (m.group(1), m.group(3).strip())
    return out


def resolve_default(expr: str, consts: dict[str, int]) -> tuple[bool | None, str]:
    """Resolve a flag default expression to (is_on, display). An int>0 or ``true`` is ON;
    ``0``/``false`` is OFF; a ``Default*`` symbol resolves through ``consts``. Returns
    (None, expr) when it cannot be resolved (the caller abstains rather than guess)."""
    e = (expr or "").strip()
    if e in ("true", "false"):
        return (e == "true"), e
    if re.fullmatch(r"\d+", e):
        v = int(e)
        return (v > 0), e
    sym = e.split(".")[-1]  # gateway.DefaultElideResultBytes -> DefaultElideResultBytes
    if sym in consts:
        v = consts[sym]
        return (v > 0), f"{sym}={v}"
    return None, e


def parse_servewiring_verdicts(src: str) -> dict[str, str]:
    """Map each servewiringData feature -> its audited verdict glyph (WIRED /
    OFF_BY_DEFAULT_BUT_WIRED / PARTIAL / DEAD_WIRED), parsed from the row literals in
    servewiring.go. The cross-check anchor: the audited verdict must agree with the real
    flag default."""
    out: dict[str, str] = {}
    row = re.compile(r'\{\s*"([a-z]+)"\s*,[^}]*?(verdictWired|verdictOffByDefault|verdictPartial|verdictDead)\b')
    for m in row.finditer(src or ""):
        verdict = {
            "verdictWired": "WIRED",
            "verdictOffByDefault": "OFF_BY_DEFAULT_BUT_WIRED",
            "verdictPartial": "PARTIAL",
            "verdictDead": "DEAD_WIRED",
        }[m.group(2)]
        out[m.group(1)] = verdict
    return out


def sw_verdict_is_on(verdict: str) -> bool | None:
    """Does an audited servewiring verdict assert the feature is ON by default?
    WIRED = on; OFF_BY_DEFAULT_BUT_WIRED = off; others abstain (None)."""
    if verdict == "WIRED":
        return True
    if verdict == "OFF_BY_DEFAULT_BUT_WIRED":
        return False
    return None


# A witnessed tradeoff is a QUANTIFIED magnitude — an "N×"/"Nx"/"N%" savings or a
# "K/N turns" fidelity ratio — NOT a mechanism word. "elision sheds tokens" describes how
# the lever works; "13.3× fewer resident, 715/715 exact recall" is a witness. Requiring a
# number is what keeps an unmeasured lever honestly UNwitnessed (the elideresult case).
_SAVINGS_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:×|x|%)|\b\d+\s*/\s*\d+\s*turns?\b",
    re.IGNORECASE)


def find_witness(doc_texts: dict[str, str], tokens: tuple[str, ...]) -> str:
    """The lever's witnessed tradeoff: a tracked doc where one of ``tokens`` co-occurs
    with a savings/fidelity figure within a small window. Returns the witnessing snippet,
    or "" if no doc quantifies the tradeoff. The ungameable part of "is this safe to
    default": the witness must be a committed measurement, not a claim in the roster."""
    if not tokens:
        return ""
    # Same-LINE co-occurrence: a real witness states the lever AND its magnitude together
    # (each CLAIMS bullet / claim is one line). A windowed match falsely paired the
    # "Planned-elision" bullet with an unrelated "~16×" matmul figure two lines away.
    for rel, text in sorted(doc_texts.items()):
        for line in (text or "").splitlines():
            low = line.lower()
            if any(t.lower() in low for t in tokens) and _SAVINGS_RE.search(line):
                return f"{rel}: {line.strip()[:160]}"
    return ""


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of token_defaults_debt; soft = score-only nudges.
# ---------------------------------------------------------------------------

def kpi_lossless_stack(levers: list[Lever]) -> dict[str, Any]:
    """Every zero-loss saver must be ON by default. A lossless lever turned off is a free
    saving left on the table — there is no fidelity reason to gate it. One defect per
    lossless saver that is off (or whose default cannot be resolved from source)."""
    defects: list[str] = []
    pool = [v for v in levers if v.klass == "lossless"]
    on = 0
    for v in pool:
        if v.on is True:
            on += 1
        elif v.on is False:
            defects.append(f"{v.key}: lossless saver is OFF by default ({v.default_repr or 'off'}) — "
                           "turn it on; there is no fidelity cost to a zero-loss saver")
        else:
            defects.append(f"{v.key}: cannot resolve its default from source ({v.default_repr or '?'}) — "
                           "the scorecard can't confirm this lossless saver is on")
    return {"kpi": "lossless_stack", "group": "stack",
            "score": _clamp(100 * on / max(1, len(pool))),
            "detail": f"{on}/{len(pool)} lossless savers on by default",
            "defects": defects, "soft": []}


def kpi_high_value_defaults(levers: list[Lever]) -> dict[str, Any]:
    """Every BOUNDED-LOSS saver that is *demonstrably* safe must be ON by default — the
    operator's rule, made honest: a lever belongs on out of the box once it is PROVEN to
    keep the model's working set (a witness quantifies the savings/fidelity tradeoff) AND
    no remaining gate blocks it. Turning an UNWITNESSED lever on would ship an unproven
    claim, so it is NOT hard debt — it is a soft "needs a witness" until measured. CROSS-
    CHECK: hard debt requires the loss-bounding ``guard_symbol`` to resolve in source AND a
    committed witness; the roster cannot grant either. One HARD defect per bounded saver
    that is witnessed-safe, ungated, and still off (a proven win left off)."""
    defects: list[str] = []
    soft: list[str] = []
    pool = [v for v in levers if v.klass == "bounded"]
    on = 0
    eligible = 0  # the hard denominator: witnessed-safe bounded savers
    for v in pool:
        if v.on is True:
            on += 1
            eligible += 1
            continue
        if v.on is None:
            defects.append(f"{v.key}: cannot resolve its default from source ({v.default_repr or '?'})")
            continue
        # OFF — is it a proven win left off, or correctly waiting on a blocker?
        if not v.guard_present:
            soft.append(f"{v.key}: loss-bounding guard {v.guard_symbol} not found in {v.guard_file} — "
                        "cannot vouch it's safe to default-on")
        elif not v.witnessed:
            soft.append(f"{v.key}: high-value bounded-loss saver is OFF, and UNWITNESSED — its guard "
                        f"{v.guard_symbol} bounds the loss, but no committed measurement quantifies the "
                        "savings/fidelity tradeoff yet. Produce a witness, then default it on (with a note)")
        elif v.gated:
            eligible += 1  # witnessed but still gated — counts against the ratio, soft not hard
            soft.append(f"{v.key}: WITNESSED-safe and OFF, but still carries a documented gate "
                        f"({v.witness_note or 'see witness'}) — clear the gate, then default it on")
        else:
            eligible += 1
            defects.append(f"{v.key}: witnessed-safe, ungated, yet OFF by default ({v.default_repr or 'off'}) — "
                           "a proven win left off; turn it on out of the box with an honest note")
    return {"kpi": "high_value_defaults", "group": "stack",
            "score": _clamp(100 * on / max(1, eligible)) if eligible else 100,
            "detail": f"{on}/{eligible} demonstrably-safe bounded-loss savers on by default",
            "defects": defects, "soft": soft}


def kpi_witness_status(levers: list[Lever]) -> dict[str, Any]:
    """SOFT — the roadmap: for every OFF saver, WHY it's off and what unblocks a default-on.
    This is the operator's "understand where each stands" view as a live signal: a lever
    that is unwitnessed needs a measurement; a witnessed-but-gated lever (e.g. a 13.3×
    fewer-resident / 100%-recall result that hasn't been watched on a live session) is the
    NEXT default to turn on once its gate clears. Scores by how many off levers have a
    witness in hand; emits no hard debt (producing a witness is real work, not a one-line
    fix), so it never forces a reckless flip."""
    soft: list[str] = []
    off_hv = [v for v in levers if v.on is False and v.klass in ("bounded", "optin")]
    witnessed = 0
    for v in off_hv:
        if v.witnessed:
            witnessed += 1
            soft.append(f"{v.key}: WITNESSED ({v.witness_note}) — the strongest default-on candidate; "
                        "clear its gate and turn it on with notes")
        else:
            soft.append(f"{v.key}: no committed witness for its savings/fidelity tradeoff — "
                        "the blocker is a measurement, not the default value")
    return {"kpi": "witness_status", "group": "honesty",
            "score": _clamp(100 * witnessed / max(1, len(off_hv))) if off_hv else 100,
            "detail": (f"{witnessed}/{len(off_hv)} off high-value savers have a committed witness in hand" if off_hv
                       else "no off high-value savers"),
            "defects": [], "soft": soft}


def kpi_dark_lever_gated(levers: list[Lever]) -> dict[str, Any]:
    """Every OFF lever must be off ON PURPOSE: it carries a documented gate (a constant
    comment or flag-help that explains why it's dark, a safe value to enable it, and the
    witness it waits for). An undocumented OFF lever is an accident, not a guard. One
    defect per OFF lever with no documented gate."""
    defects: list[str] = []
    off = [v for v in levers if v.on is False]
    gated = 0
    for v in off:
        if v.gated:
            gated += 1
        else:
            defects.append(f"{v.key}: OFF by default but carries no documented gate — explain why it's "
                           "dark, name a safe value, and the witness it needs (or turn it on)")
    return {"kpi": "dark_lever_gated", "group": "honesty",
            "score": _clamp(100 * gated / max(1, len(off))) if off else 100,
            "detail": (f"{gated}/{len(off)} off-by-default levers carry a documented gate" if off
                       else "no off-by-default levers"),
            "defects": defects, "soft": []}


def kpi_default_notes(levers: list[Lever]) -> dict[str, Any]:
    """Every ON bounded-loss saver must document what it sheds — on by default is only
    honest WITH a note. The flag help / const comment must say what it drops AND that the
    rewrite is cache-prefix-safe (so a reader knows the saving doesn't burst the cache).
    One defect per ON bounded-loss saver missing its loss note."""
    defects: list[str] = []
    pool = [v for v in levers if v.klass == "bounded" and v.on is True]
    noted = 0
    for v in pool:
        if v.noted:
            noted += 1
        else:
            defects.append(f"{v.key}: on by default but its flag help / constant comment doesn't note what "
                           f"it sheds + that it's cache-prefix-safe (want all of: {', '.join(v.note_tokens)})")
    return {"kpi": "default_notes", "group": "honesty",
            "score": _clamp(100 * noted / max(1, len(pool))) if pool else 100,
            "detail": (f"{noted}/{len(pool)} on-by-default bounded savers carry an honest loss note" if pool
                       else "no on-by-default bounded savers"),
            "defects": defects, "soft": []}


def kpi_default_on_locked(levers: list[Lever]) -> dict[str, Any]:
    """Every default-ON saver must be pinned by a test so a peer who flips it back to off
    reds a gate — an amazing default that can silently rot is not a guarantee. One defect
    per ON saver with no sentinel test referencing its default."""
    defects: list[str] = []
    pool = [v for v in levers if v.on is True]
    locked = 0
    for v in pool:
        if v.locked:
            locked += 1
        else:
            defects.append(f"{v.key}: on by default but no test pins the default (want a *_test.go asserting "
                           f"{' + '.join(v.sentinel_tokens)}) — a regression to off would ship silently")
    return {"kpi": "default_on_locked", "group": "regression",
            "score": _clamp(100 * locked / max(1, len(pool))) if pool else 100,
            "detail": f"{locked}/{len(pool)} on-by-default savers pinned by a regression sentinel",
            "defects": defects, "soft": []}


def kpi_entrypoint_parity(levers: list[Lever], guard_defaults: dict[str, tuple[str, str]],
                          serve_defaults: dict[str, tuple[str, str]],
                          consts: dict[str, int]) -> dict[str, Any]:
    """The two front doors must offer the same economy: a flag both ``fak guard`` and
    ``fak serve`` expose must carry the SAME default, and the audited servewiring verdict
    must agree with that real default. A user gets the same out-of-the-box savings
    whichever door they enter. One defect per divergence."""
    defects: list[str] = []
    checked = 0
    for v in levers:
        # flag-default parity across the two entrypoints
        if v.flag and v.flag in guard_defaults and v.flag in serve_defaults:
            checked += 1
            g_on, g_disp = resolve_default(guard_defaults[v.flag][1], consts)
            s_on, s_disp = resolve_default(serve_defaults[v.flag][1], consts)
            if g_on is not None and s_on is not None and g_on != s_on:
                defects.append(f"{v.key}: --{v.flag} default diverges between front doors "
                               f"(guard {g_disp}={'on' if g_on else 'off'}, serve {s_disp}={'on' if s_on else 'off'})")
        # the audited servewiring verdict must match the real default
        if v.sw_feature and v.sw_verdict and v.on is not None:
            sw_on = sw_verdict_is_on(v.sw_verdict)
            if sw_on is not None and sw_on != v.on:
                defects.append(f"{v.key}: servewiring row says {v.sw_verdict} but the real default is "
                               f"{'on' if v.on else 'off'} — regenerate the wiring row "
                               "(`fak serve-wiring --md`) so the audited verdict tracks the flip")
    return {"kpi": "entrypoint_parity", "group": "parity",
            "score": _clamp(100 - 25 * len(defects)),
            "detail": (f"{len(defects)} parity/verdict divergence(s)" if defects
                       else "front doors agree + servewiring verdicts track the real defaults"),
            "defects": defects, "soft": []}


def kpi_stacking_depth(levers: list[Lever]) -> dict[str, Any]:
    """SOFT — how many savers stack ON simultaneously on the flagship passthrough. The
    headline "stacking token-saving methods on by default" number. Scores (more stacked
    savers = a better default economy) but emits no hard debt: the right count is a
    judgment about safety, not a target to chase to 100%."""
    total = len(levers)
    on = sum(1 for v in levers if v.on is True)
    soft = [f"{v.key} ({v.klass}) is not stacked on by default" for v in levers if v.on is not True]
    return {"kpi": "stacking_depth", "group": "stack",
            "score": _clamp(100 * on / max(1, total)),
            "detail": f"{on}/{total} token-saving methods stacked on by default out of the box",
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite, grade, token_defaults_debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, levers: list[Lever], kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT, with git), then re-run",
            "workspace": workspace, "corpus": {}, "levers": [], "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    score_by_group = {g: 0.0 for g in GROUPS}
    wsum_by_group = {g: 0.0 for g in GROUPS}
    for k in kpis:
        g = k["group"]
        debt_by_group[g] += len(k["defects"])
        w = KPI_WEIGHTS.get(k["kpi"], 0.0)
        score_by_group[g] += w * k["score"]
        wsum_by_group[g] += w
    group_scores = {g: (round(score_by_group[g] / wsum_by_group[g], 1)
                        if wsum_by_group[g] else 0.0) for g in GROUPS}

    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    lever_status = [lever_row(v) for v in levers]
    stacked = sum(1 for v in levers if v.on is True)

    corpus = {
        "score": score, "grade": grade, "token_defaults_debt": debt,
        "soft_signals": n_soft,
        "stacked_on": stacked, "levers_total": len(levers),
        "group_scores": group_scores,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
        "lever_status": lever_status,
    }

    gs = group_scores
    standing = (f"stack {gs['stack']:.0f} · honesty {gs['honesty']:.0f} "
                f"· regression {gs['regression']:.0f} · parity {gs['parity']:.0f}")
    if debt == 0:
        ok, verdict, finding = True, "OK", "amazing_defaults"
        reason = (f"out-of-the-box token economy is amazing: score {score}/100 (grade {grade}), "
                  f"zero token-defaults-debt across {len(kpis)} KPIs; {stacked}/{len(levers)} savers "
                  f"stacked on by default ({standing}; {n_soft} advisory)")
        next_action = ("hold the line; re-run after a change to a saver default, the servewiring rows, "
                       "or a new token-saving lever (add it to the roster)")
    else:
        ok, verdict, finding = False, "ACTION", "token_defaults_debt"
        worst = breakdown[0]
        reason = (f"{debt} unit(s) of token-defaults-debt; score {score}/100 (grade {grade}); "
                  f"{stacked}/{len(levers)} savers stacked on; heaviest: {worst['kpi']} "
                  f"({worst['debt']} defect(s)); standing {standing}")
        next_action = ("retire token-defaults-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "turn a high-value bounded-loss saver ON by default with an honest note, write the "
                       "regression lock, document the dark-lever gate, or align the front doors; re-run to "
                       "prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "levers": lever_status, "kpis": kpis,
    }


def lever_row(v: Lever) -> dict[str, Any]:
    """The per-lever STATUS row — the operator's 'where does each saver stand' view."""
    state = "ON" if v.on is True else ("OFF" if v.on is False else "?")
    return {
        "key": v.key, "label": v.label, "klass": v.klass,
        "default": state, "default_repr": v.default_repr,
        "flag": (f"--{v.flag}" if v.flag else "(structural)"),
        "entrypoints": list(v.entrypoints),
        "sw_feature": v.sw_feature, "sw_verdict": v.sw_verdict,
        "guard": v.guard_symbol, "guard_present": v.guard_present,
        "gated": v.gated, "noted": v.noted, "locked": v.locked,
        "witnessed": v.witnessed, "witness_note": v.witness_note,
        "blocker": v.off_reason or ("—" if v.on is True else ""),
    }


# ---------------------------------------------------------------------------
# Disk + git gathering (the impure shell around the pure KPIs).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_lines(args: list[str], root: Path) -> list[str]:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flag_help(src: str, flag: str) -> str:
    """The help string of a ``fs.Int/Bool("<flag>", default, "<help>")`` declaration —
    the prose the gate checks for an honest loss note / documented dark gate."""
    if not flag:
        return ""
    # The help is the third argument; it may run long, so capture up to the closing `")`.
    m = re.search(r'fs\.(?:Int|Bool)\(\s*"' + re.escape(flag) + r'"\s*,\s*[^,]+,\s*"((?:[^"\\]|\\.)*)"', src)
    return m.group(1) if m else ""


def _const_comment(src: str, const: str) -> str:
    """The doc-comment block immediately above a ``const <Name> = …`` declaration."""
    if not const:
        return ""
    idx = src.find(f"const {const} ")
    if idx < 0:
        idx = src.find(f"const {const}=")
    if idx < 0:
        return ""
    # Walk upward collecting the contiguous `//` doc-comment block directly above the
    # const; stop at the first non-comment line (Go doc comments are contiguous).
    out: list[str] = []
    for ln in reversed(src[:idx].splitlines()):
        s = ln.strip()
        if s.startswith("//"):
            out.append(s.lstrip("/").strip())
        else:
            break
    return " ".join(reversed(out))


def gather(root: Path) -> list[dict[str, Any]]:
    """Read the source-of-truth files, resolve each lever's real default + cross-checks,
    and run every pure KPI."""
    guard_src = _safe_read(root / GUARD_GO)
    serve_src = _safe_read(root / SERVE_GO)
    gw_src = _safe_read(root / GATEWAY_GO)
    sw_src = _safe_read(root / SERVEWIRING_GO)

    consts = parse_const_ints(gw_src)
    guard_defaults = parse_flag_defaults(guard_src)
    serve_defaults = parse_flag_defaults(serve_src)
    sw_verdicts = parse_servewiring_verdicts(sw_src)

    # Source + test corpus, read from the WORKING TREE (tracked OR on disk) — the binary is
    # built from the working tree, so a freshly-added (pre-commit) guard/impl counts, exactly
    # like agent_readiness's tracked-or-on-disk rule. Scoped to cmd/ + internal/ where the
    # entrypoints, gateway, and guards live, so the walk stays bounded + deterministic.
    go_blobs: dict[str, str] = {}
    for sub in ("cmd", "internal"):
        d = root / sub
        if d.is_dir():
            for p in d.rglob("*.go"):
                go_blobs[p.relative_to(root).as_posix()] = _safe_read(p)
    test_blobs = {f: t for f, t in go_blobs.items() if f.endswith("_test.go")}

    # Witness corpus: the committed docs/experiments where a savings/fidelity measurement
    # would live (NOT tests — tests prove the MECHANISM, a witness quantifies the TRADEOFF).
    doc_blobs: dict[str, str] = {}
    claims = root / "CLAIMS.md"
    if claims.exists():
        doc_blobs["CLAIMS.md"] = _safe_read(claims)
    for sub in ("docs", "experiments"):
        d = root / sub
        if d.is_dir():
            for pat in ("*.md", "*.json"):
                for p in d.rglob(pat):
                    doc_blobs[p.relative_to(root).as_posix()] = _safe_read(p)

    def symbol_present(symbol: str, where: str) -> bool:
        if not symbol:
            return True
        for f, blob in go_blobs.items():
            if where and where not in f:
                continue
            if symbol in blob:
                return True
        return False

    def locked_by_test(tokens: tuple[str, ...]) -> bool:
        if not tokens:
            return False
        for blob in test_blobs.values():
            if all(tok in blob for tok in tokens):
                return True
        return False

    for v in LEVERS:
        # --- resolve the REAL default from source (never from the roster) ---
        if v.flag:
            # prefer serve.go's declaration; fall back to guard.go (some flags live in one).
            decl = serve_defaults.get(v.flag) or guard_defaults.get(v.flag)
            if decl:
                v.on, v.default_repr = resolve_default(decl[1], consts)
            else:
                v.on, v.default_repr = None, "(flag not found)"
        elif v.structural_field:
            # a no-flag lever is on iff serve.go sets its Config field in the New(Config{}) literal.
            set_in_serve = bool(re.search(r"\b" + re.escape(v.structural_field) + r"\s*:", serve_src))
            v.on, v.default_repr = (True if set_in_serve else False), ("structural" if set_in_serve else "(field not set)")
        elif v.guard_symbol:
            # a purely structural always-on saver (provider cache): on iff its mechanism exists.
            present = symbol_present(v.guard_symbol, v.guard_file)
            v.on, v.default_repr = (True if present else None), ("structural" if present else "(mechanism not found)")
        else:
            v.on, v.default_repr = None, "?"

        # special-case vdso: guard sets it structurally true even though it's a serve flag.
        if v.key == "vdso" and v.on is None:
            v.on, v.default_repr = True, "structural (guard) / flag (serve)"

        # --- cross-checks ---
        v.sw_verdict = sw_verdicts.get(v.sw_feature, "") if v.sw_feature else ""
        v.guard_present = symbol_present(v.guard_symbol, v.guard_file) if v.guard_symbol else True

        help_guard = _flag_help(guard_src, v.flag)
        help_serve = _flag_help(serve_src, v.flag)
        const_doc = _const_comment(gw_src, v.default_const)
        prose = " ".join((help_guard, help_serve, const_doc)).lower()

        # gated: an OFF lever explains itself + names a safe value / the witness it waits for.
        v.gated = bool(re.search(r"\b(off by default|0 \(default\)|default\) = off|opt-in|ships dark|until a real-traffic|gate it|witness)\b", prose)) \
            and bool(re.search(r"\b(safe value|~?\d{3,}|witness|watched|dogfood)\b", prose))

        # noted: an ON bounded lever documents what it sheds + that it's cache-prefix-safe.
        v.noted = all(tok.lower() in prose for tok in v.note_tokens) if v.note_tokens else True

        # locked: a *_test.go pins the default.
        v.locked = locked_by_test(v.sentinel_tokens)

        # witnessed: a committed doc quantifies its savings/fidelity tradeoff. A lossless
        # lever has no tradeoff to measure → witnessed by construction.
        if v.klass == "lossless":
            v.witnessed, v.witness_note = True, "lossless (no fidelity tradeoff)"
        else:
            v.witness_note = find_witness(doc_blobs, v.witness_tokens)
            v.witnessed = bool(v.witness_note)

        # off_reason: for an OFF lever, the blocker that keeps it dark.
        if v.on is False:
            if v.witnessed and v.gated:
                v.off_reason = "witnessed_gated"
            elif v.witnessed:
                v.off_reason = "ready"  # witnessed + ungated → should be defaulted on
            else:
                v.off_reason = "unwitnessed"

    kpis = [
        kpi_lossless_stack(LEVERS),
        kpi_high_value_defaults(LEVERS),
        kpi_dark_lever_gated(LEVERS),
        kpi_default_notes(LEVERS),
        kpi_witness_status(LEVERS),
        kpi_default_on_locked(LEVERS),
        kpi_entrypoint_parity(LEVERS, guard_defaults, serve_defaults, consts),
        kpi_stacking_depth(LEVERS),
    ]
    return kpis


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), levers=[], kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    if not (root / GUARD_GO).exists() or not (root / SERVE_GO).exists():
        return build_payload(workspace=str(root), levers=[], kpis=[],
                             error=f"missing {GUARD_GO}/{SERVE_GO} — run from the fak repo ROOT")
    kpis = gather(root)
    return build_payload(workspace=str(root), levers=LEVERS, kpis=kpis)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _lever_table_lines(levers: list[dict[str, Any]]) -> list[str]:
    lines = [f"  {'lever':<16} {'class':<9} {'default':<7} {'witness':<9} {'blocker':<16} "
             f"{'noted':<6} {'locked':<6}"]
    for v in levers:
        lines.append(f"  {v['key']:<16} {v['klass']:<9} {v['default']:<7} "
                     f"{('yes' if v['witnessed'] else 'no'):<9} {(v['blocker'] or '-'):<16} "
                     f"{'yes' if v['noted'] else '-':<6} {'yes' if v['locked'] else '-':<6}")
    return lines


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    lines = [
        f"token-defaults-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· TOKEN-DEFAULTS-DEBT {c.get('token_defaults_debt', 0)} "
         f"· {c.get('stacked_on', 0)}/{c.get('levers_total', 0)} savers stacked on "
         f"· {c.get('soft_signals', 0)} advisory"),
        (f"groups:  stack {gs.get('stack', 0):.0f}  ·  honesty {gs.get('honesty', 0):.0f}  ·  "
         f"regression {gs.get('regression', 0):.0f}  ·  parity {gs.get('parity', 0):.0f}"),
        "",
        "per-lever status (where each saver stands):",
    ]
    lines += _lever_table_lines(c.get("lever_status", []))
    lines += [
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'group':<11} {'kpi':<22} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<11} "
                     f"{b['kpi']:<22} {b['detail']}")
    lines.append("")
    lines.append("token-defaults-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
    if not any_defect:
        lines.append("  (none — zero token-defaults-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak token-saving-defaults scorecard — is the out-of-the-box token economy amazing?"')
    out.append('description: "fak\'s deterministic token-saving-defaults scorecard: which stacking '
               'token-saving methods are ON by default on the fak guard / fak serve Anthropic '
               'passthrough, whether the high-value low-loss savers are turned on out of the box, '
               'and whether every default is honestly noted and locked against regression — '
               're-derived from the entrypoint source."')
    out.append("---")
    out.append("")
    out.append("# Token-saving-defaults scorecard — is fak's out-of-the-box token economy amazing?")
    out.append("")
    if stamp:
        out.append(f"<!-- token-defaults-scorecard: {stamp} · process: tools/token_defaults_scorecard.py -->")
        out.append("")
    out.append("The question a cost-conscious operator asks the moment they run "
               "`fak guard -- claude` / `fak serve`: **of every token-saving method fak knows how to "
               "stack, which ones are ON by default — and are the high-value, low-loss ones turned on "
               "out of the box, or left dark behind a flag nobody flips?** Every number below is "
               "re-derived from the entrypoint source (`cmd/fak/guard.go`, `cmd/fak/serve.go`, the "
               "`Default*` constants in `internal/gateway/gateway.go`, and the audited "
               "`servewiringData` rows) by `tools/token_defaults_scorecard.py` — a lever's on/off "
               "state is the binary's real behavior, never a claim in the roster. The headline metric "
               "is **token-defaults-debt**: the count of concrete defects — a high-value saver left "
               "off, an on-by-default saver with no honest note, a default no test locks, a front "
               "door out of step. Driving it to zero means a user who runs fak with no flags gets the "
               "full stack of safe savings, each honestly labeled, none able to regress unnoticed.")
    out.append("")
    out.append("> Regenerate: `python tools/token_defaults_scorecard.py --markdown --stamp DATE > "
               f"{GENERATED_SNAPSHOT}`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Token-defaults-debt (total HARD defects)** | **{c.get('token_defaults_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Savers stacked on by default | {c.get('stacked_on', 0)}/{c.get('levers_total', 0)} |")
    out.append(f"| Groups | stack {gs.get('stack', 0):.0f} · honesty {gs.get('honesty', 0):.0f} "
               f"· regression {gs.get('regression', 0):.0f} · parity {gs.get('parity', 0):.0f} |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## Per-lever status — where each token-saving method stands")
    out.append("")
    out.append("`class`: **lossless** = zero model-visible change (must be on); **bounded** = lossy but "
               "an in-code guard keeps the model's working set intact (high-value → should be on, with a "
               "note); **optin** = broader blast radius (correctly off, must carry a documented gate). "
               "`gated` = an off lever documents why; `noted` = an on bounded lever documents what it "
               "sheds + cache-safety; `locked` = a test pins the default.")
    out.append("")
    out.append("| Lever | Class | Default | Witness | Blocker | Flag | Gated | Noted | Locked |")
    out.append("|---|---|:--:|:--:|---|---|:--:|:--:|:--:|")
    for v in c.get("lever_status", []):
        out.append(f"| {v['key']} — {v['label']} | {v['klass']} | **{v['default']}** | "
                   f"{'✓' if v['witnessed'] else '·'} | {v['blocker'] or '—'} | `{v['flag']}` | "
                   f"{'✓' if v['gated'] else '·'} | {'✓' if v['noted'] else '·'} | "
                   f"{'✓' if v['locked'] else '·'} |")
    out.append("")
    out.append("## KPIs")
    out.append("")
    out.append("| Group | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Token-defaults-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No token-defaults-debt: every stacking saver fak can safely default is on out of "
                   "the box, honestly noted, and locked against regression. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("token_defaults_debt", 0), cur.get("token_defaults_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"token-defaults-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:               {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"savers stacked on:   {b.get('stacked_on', 0)}/{b.get('levers_total', 0)} -> "
        f"{cur.get('stacked_on', 0)}/{cur.get('levers_total', 0)}",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<11} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x token-defaults-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need token-defaults-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Token-saving-defaults scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the token-defaults-debt delta vs a prior baseline JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
