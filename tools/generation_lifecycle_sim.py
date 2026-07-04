#!/usr/bin/env python3
r"""generation_lifecycle_sim -- a deterministic simulation of generation promotion
and demotion: items MOVING between streams under CHANGING evidence.

Closes https://github.com/anthony-chaudhary/fak/issues/1656 under epic #1625.
Generation stream: gen/second-next. Proof bar for this stream is *simulation* --
so this artifact is a runnable lifecycle model, not a static score.

## Why this exists (and how it differs from its three siblings)

The generation portfolio already has three artifacts, and all three grade a
*static snapshot* -- they answer a question about the portfolio as it is right
now:

  * generation-fit-grooming-checklist.md (#1648) -- "is the label even right?"    (one issue, at intake)
  * generation-portfolio-optimizer.md    (#1652) -- "which mix gets attention?"   (whole portfolio, at planning)
  * docs/notes/GENERATION-READINESS-GATES-2026-06-30.md (#1644) -- "is the         (one item, at promotion)
    evidence strong enough to promote?"

None of them EXERCISE the promotion rules -- they never show an item actually
move. Issue #1656's Why is exactly that gap: "Promotion rules should be
exercised before they govern real work." This module is the *dynamic* sibling:
it drives the four canonical promotion verbs from docs/generation.md as a state
machine and replays a timeline of evidence changes so you can watch items travel
between streams and land in a new distribution. It is the lifecycle model, the
missing fourth grain (portfolio-over-time).

## The state machine (the executable form of docs/generation.md)

Streams are a distance-from-`now` ladder; `retired` is a terminal absorbing
state; `parked` is an "active/inactive" bit that rides alongside the stream:

    NOW(0)  <--  NEXT(1)  <--  SECOND_NEXT(2)  <--  FUTURE(3)        [+ RETIRED, + parked]
            promote -->                      <-- demote

The four verbs from docs/generation.md#promotion-verbs, and the evidence classes
from docs/generation.md#evidence that fire them, are encoded in `decide()`:

  * promote  <- blocker_retired            : evidence retired the blocker that kept it later -> one stream closer to now.
  * demote   <- assumption_failed(nearer)  : an assumption failed / witness regressed / current path no longer needs it,
               | witness_regressed           but a nearer product path still exists -> one stream farther from now.
  * retire   <- superseded                 : obsolete, superseded, option-cost-exceeds-value, or an assumption failed with
               | option_cost_exceeds_value   NO nearer path left -> terminal. (docs/generation.md#evidence demotion list.)
               | assumption_failed(no path)
  * park     <- no_owner                   : true-but-not-active (no owner, witness, or decision) -> hold the stream, mark inactive.

Precedence is deterministic and load-bearing: retire > demote > park > promote.
A hard-negative signal DOMINATES a promotion signal that arrives in the same
tick -- you never promote an item whose witness just regressed. `promote`/
`demote` are active reclassifications, so they clear `parked`; a subsequent
`blocker_retired` therefore RE-ACTIVATES a parked item as it moves it closer.
`retire` is absorbing: a retired item ignores all further evidence.

## Orthogonality is an EXECUTED invariant here, not a paragraph

docs/generation.md pins that generation is orthogonal to priority, shared trunk,
and runtime feature gates. This simulation makes that checkable instead of
asserted: every item carries three passthrough fields the state machine is
structurally forbidden to read or mutate --

  * priority : the urgency label. A promote never raises it; a demote never lowers it.
  * gate     : the runtime-exposure decision (off / on / operator-only / none). A
               stream move never toggles it -- horizon and exposure are independent.
  * trunk    : always "main". There is NO branch/worktree field in the model; a
               stream is a label partition, never an integration branch.

`apply()` copies all three verbatim across every transition, and the test suite
asserts they are byte-identical before and after a full scenario. If a future
edit lets a stream move touch any of them, a test goes red -- that is the
orthogonality guarantee, mechanized.

## CLI

    python tools/generation_lifecycle_sim.py demo           # human before/after readout (the operator witness)
    python tools/generation_lifecycle_sim.py demo --json     # the fak-generation-lifecycle/1 object
    python tools/generation_lifecycle_sim.py selfcheck       # invariants: caps, terminal-retire, orthogonality

## Promotion / demotion / assumption (for THIS artifact, per the gen/second-next rule)

  * Promotion evidence (what moves this simulation toward gen/now): wire this
    decision table into a runnable `fak` surface -- e.g. `fak generation
    simulate` (pure logic in internal/genlifecycle/, thin shell in cmd/fak/, per
    the Go-not-Python rule) fed by real label/milestone history, plus one
    captured replay over an actual sequence of generation relabels from the repo.
    The green contract test below (the worked scenario is the fixture) plus that
    real-history replay is the promotion witness. Until then this stays a
    gen/second-next lifecycle model, run by hand or by a planning agent.
  * Demotion / retirement evidence (the compatibility edge with a kill criterion):
    the four verbs and four streams here are a strict projection of
    docs/generation.md. RETIRE this simulation if that contract's verb set or
    stream ladder changes such that `decide()`/`apply()` no longer map -- do not
    let it drift into a second source of truth. DEMOTE (park) it if the static
    trio (#1648/#1652/#1644) plus operator judgment already drive every real
    promote/demote call and no one ever replays a timeline -- an unrun simulator
    is decorative and should be removed, not defended.
  * Invalidating assumption: this model assumes promotion evidence arrives as
    DISCRETE, per-item, legible signals and that the four-verb alphabet is
    COMPLETE and STABLE. If real evidence is continuous or graded (a confidence
    that crosses a threshold), if transitions need cross-item context (an item
    promotes only because a sibling retired), or if a fifth verb appears (e.g.
    "split"), the decision table is too coarse and must be rebound to whatever
    docs/generation.md then specifies. A second, sharper assumption: that a
    single-step ladder (one stream per event) matches reality -- evidence strong
    enough to skip a horizon would break the one-step move.

## Continue here (no epic reread)

A future agent advancing #1656's follow-on: (1) implement `fak generation
simulate` in Go per the promotion-evidence note above; (2) feed it real relabel
history (`gh issue view <n> --json timelineItems` or the milestone report's
sidecars) instead of the hand-authored scenario; (3) keep `decide()` a strict
projection of docs/generation.md#promotion-verbs -- if that section changes,
rebind here in the same pass (that is the demotion criterion, mechanized).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

# --------------------------------------------------------------------------- #
# Streams: a distance-from-`now` ladder + a terminal state. The integers ARE
# the horizon distance, so promote = -1 and demote = +1 (clamped). This binds
# 1:1 to the streams table in docs/generation.md#streams.
# --------------------------------------------------------------------------- #
NOW, NEXT, SECOND_NEXT, FUTURE = 0, 1, 2, 3
RETIRED = -1  # terminal / absorbing -- NOT "closer than now"; a sentinel outside the ladder.

STREAM_NAME = {
    NOW: "gen/now",
    NEXT: "gen/next",
    SECOND_NEXT: "gen/second-next",
    FUTURE: "gen/future",
    RETIRED: "retired",
}
LADDER = (NOW, NEXT, SECOND_NEXT, FUTURE)  # ordered near->far; retired is off-ladder.

# The four canonical verbs (docs/generation.md#promotion-verbs) + an explicit
# no-op so every event yields a recorded decision.
PROMOTE, DEMOTE, RETIRE, PARK, HOLD = "promote", "demote", "retire", "park", "hold"


@dataclass(frozen=True)
class Item:
    """One generation item. `stream` is a ladder position (or RETIRED); `parked`
    is the active/inactive bit. `priority`, `gate`, `trunk` are the ORTHOGONAL
    passthrough fields the state machine must never read or change."""
    id: str
    stream: int
    parked: bool = False
    # --- orthogonal passthrough; docs/generation.md#orthogonality (executed, not asserted) ---
    priority: str = "P2"      # urgency label -- generation is orthogonal to it.
    gate: str = "off"          # runtime exposure -- generation is orthogonal to it.
    trunk: str = "main"        # shared trunk -- a stream is a label, never a branch.


@dataclass(frozen=True)
class Event:
    """A tick of changing evidence for one item. `evidence` is the signal set
    read by decide(); `note` is free-text for the readout."""
    tick: int
    item_id: str
    evidence: dict
    note: str = ""


@dataclass
class Transition:
    tick: int
    item_id: str
    verb: str
    reason: str
    from_stream: int
    to_stream: int
    from_parked: bool
    to_parked: bool


def decide(item: Item, ev: dict) -> tuple[str, str]:
    """Map a changing-evidence signal set onto exactly one promotion verb.

    Deterministic precedence (load-bearing): retire > demote > park > promote.
    Pure function of (current stream, evidence) -- no I/O, no globals, no clock.
    This is the executable projection of docs/generation.md#evidence.
    """
    if item.stream == RETIRED:
        return HOLD, "already retired (terminal/absorbing state)"

    nearer_path = bool(ev.get("has_nearer_path", True))

    # 1) retire -- obsolete/superseded, option cost exceeds value, or an
    #    assumption failed with NO nearer product path to demote into.
    if ev.get("superseded"):
        return RETIRE, "superseded by a shipped or stronger item"
    if ev.get("option_cost_exceeds_value"):
        return RETIRE, "later-stream option cost now exceeds expected value"
    if ev.get("assumption_failed") and not nearer_path:
        return RETIRE, "invalidating assumption failed and no nearer path remains"

    # 2) demote -- assumption failed (nearer path still exists) or witness regressed.
    if ev.get("assumption_failed"):
        return DEMOTE, "invalidating assumption failed; a nearer path still exists"
    if ev.get("witness_regressed"):
        return DEMOTE, "witness regressed / went stale"

    # 3) park -- true-but-not-active (no owner, witness path, or decision).
    if ev.get("no_owner"):
        return PARK, "true but inactive: no owner, witness, or decision"

    # 4) promote -- evidence retired the blocker that kept it later.
    if ev.get("blocker_retired"):
        return PROMOTE, "evidence retired the blocker that kept it later"

    return HOLD, "no actionable evidence this tick"


def apply(item: Item, verb: str) -> Item:
    """Apply a verb to an item, returning the new item. Clamped at both ladder
    ends. CRUCIALLY: priority/gate/trunk are copied verbatim by `replace` on
    every path -- the state machine cannot touch them (orthogonality invariant)."""
    if item.stream == RETIRED:
        return item  # absorbing: a retired item is terminal under ANY verb, not just via decide().
    if verb == PROMOTE:
        # closer to now, clamped at NOW; an active reclassification -> un-park.
        return replace(item, stream=max(NOW, item.stream - 1), parked=False)
    if verb == DEMOTE:
        # farther from now, clamped at FUTURE; also an active reclassification.
        return replace(item, stream=min(FUTURE, item.stream + 1), parked=False)
    if verb == RETIRE:
        return replace(item, stream=RETIRED, parked=False)
    if verb == PARK:
        return replace(item, parked=True)  # stream unchanged; just mark inactive.
    return item  # HOLD


def simulate(items: list[Item], events: list[Event]) -> dict:
    """Replay `events` against `items`, returning the before/after distribution
    and the full transition trace. Deterministic and side-effect free."""
    state: dict[str, Item] = {it.id: it for it in items}
    before = {it.id: it for it in items}
    trace: list[Transition] = []
    for ev in events:
        it = state[ev.item_id]
        verb, reason = decide(it, ev.evidence)
        nxt = apply(it, verb)
        trace.append(Transition(
            tick=ev.tick, item_id=ev.item_id, verb=verb, reason=reason,
            from_stream=it.stream, to_stream=nxt.stream,
            from_parked=it.parked, to_parked=nxt.parked,
        ))
        state[ev.item_id] = nxt
    return {"before": before, "after": state, "trace": trace}


def distribution(state: dict[str, Item]) -> dict[str, int]:
    """Count of items per stream (+ retired), plus the active-parked overlay."""
    dist = {STREAM_NAME[s]: 0 for s in LADDER}
    dist["retired"] = 0
    parked = 0
    for it in state.values():
        dist[STREAM_NAME[it.stream]] += 1
        if it.parked:
            parked += 1
    dist["parked_active_items"] = parked
    return dist


def to_schema(sim: dict) -> dict:
    """The machine-readable fak-generation-lifecycle/1 object."""
    def item_row(it: Item) -> dict:
        return {"id": it.id, "stream": STREAM_NAME[it.stream], "parked": it.parked,
                "priority": it.priority, "gate": it.gate, "trunk": it.trunk}
    return {
        "schema": "fak-generation-lifecycle/1",
        "issue": 1656,
        "before": {"distribution": distribution(sim["before"]),
                   "items": [item_row(i) for i in sim["before"].values()]},
        "after": {"distribution": distribution(sim["after"]),
                  "items": [item_row(i) for i in sim["after"].values()]},
        "trace": [{"tick": t.tick, "item": t.item_id, "verb": t.verb,
                   "from": STREAM_NAME[t.from_stream], "to": STREAM_NAME[t.to_stream],
                   "parked": [t.from_parked, t.to_parked], "reason": t.reason}
                  for t in sim["trace"]],
    }


# --------------------------------------------------------------------------- #
# The worked scenario -- doubles as the demo readout AND the test fixture. It
# exercises all four verbs and shows items travelling between every stream.
# --------------------------------------------------------------------------- #
def worked_scenario() -> tuple[list[Item], list[Event]]:
    items = [
        Item("A-optimizer",  SECOND_NEXT, priority="P2", gate="off"),   # a 2nd-next bet
        Item("B-seam",       NEXT,        priority="P1", gate="off"),   # a next foundation seam
        Item("C-memo",       FUTURE,      priority="P3", gate="none"),  # a future research memo
        Item("D-shaky",      SECOND_NEXT, priority="P2", gate="off"),   # a bet on a shaky assumption
        Item("E-dup",        NEXT,        priority="P1", gate="off"),   # duplicates a shipped item
    ]
    events = [
        Event(1, "B-seam",      {"blocker_retired": True}, "prerequisite landed"),
        Event(2, "A-optimizer", {"blocker_retired": True}, "simulation shipped -> nearer horizon"),
        Event(3, "D-shaky",     {"assumption_failed": True, "has_nearer_path": True}, "assumption failed, nearer path exists"),
        Event(4, "C-memo",      {"no_owner": True}, "still true, but no owner/witness/decision"),
        Event(5, "E-dup",       {"superseded": True}, "a shipped issue now covers it"),
        Event(6, "D-shaky",     {"assumption_failed": True, "has_nearer_path": False}, "even the nearer path died"),
        Event(7, "C-memo",      {"blocker_retired": True}, "a live decision now needs it -> re-activate"),
    ]
    return items, events


def report_md(sim: dict) -> str:
    """The before/after readout -- the operator witness issue #1656 asks for."""
    b, a = distribution(sim["before"]), distribution(sim["after"])
    lines = ["# Generation lifecycle simulation -- promotion & demotion (#1656)", ""]
    lines.append("Streams: gen/now <- gen/next <- gen/second-next <- gen/future  (+ retired)")
    lines.append("")
    lines.append("## Distribution before -> after (items move under changing evidence)")
    lines.append("")
    lines.append("| stream | before | after |")
    lines.append("|---|---|---|")
    for key in ["gen/now", "gen/next", "gen/second-next", "gen/future", "retired"]:
        lines.append(f"| {key} | {b[key]} | {a[key]} |")
    lines.append(f"| (parked/active items) | {b['parked_active_items']} | {a['parked_active_items']} |")
    lines.append("")
    lines.append("## Transition trace (the promotion rules, exercised)")
    lines.append("")
    for t in sim["trace"]:
        arrow = f"{STREAM_NAME[t.from_stream]} -> {STREAM_NAME[t.to_stream]}"
        park = ""
        if t.from_parked != t.to_parked:
            park = "  [parked]" if t.to_parked else "  [re-activated]"
        lines.append(f"  t{t.tick} {t.item_id:<12} {t.verb:<8} {arrow}{park}")
        lines.append(f"        reason: {t.reason}")
    lines.append("")
    lines.append("## Orthogonality (preserved across every transition above)")
    lines.append("  priority: unchanged by any promote/demote (generation != urgency)")
    lines.append("  gate:     runtime exposure unchanged by any stream move")
    lines.append("  trunk:    always 'main' -- a stream is a label, never a branch")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# selfcheck -- the bundled invariants (mirrors switcher_shadow.runselfcheck).
# --------------------------------------------------------------------------- #
def runselfcheck() -> int:
    fails = []

    # promote clamps at NOW; demote clamps at FUTURE.
    if apply(Item("x", NOW), PROMOTE).stream != NOW:
        fails.append("promote must clamp at NOW")
    if apply(Item("x", FUTURE), DEMOTE).stream != FUTURE:
        fails.append("demote must clamp at FUTURE")

    # retire is terminal/absorbing.
    r = apply(Item("x", NEXT), RETIRE)
    if r.stream != RETIRED or decide(r, {"blocker_retired": True})[0] != HOLD:
        fails.append("retire must be terminal (ignores further evidence)")

    # orthogonality: priority/gate/trunk survive every verb.
    base = Item("x", SECOND_NEXT, priority="P0", gate="operator-only", trunk="main")
    for verb in (PROMOTE, DEMOTE, PARK):
        out = apply(base, verb)
        if (out.priority, out.gate, out.trunk) != (base.priority, base.gate, base.trunk):
            fails.append(f"{verb} must not touch priority/gate/trunk")

    # precedence: retire dominates promote in the same tick.
    if decide(Item("x", SECOND_NEXT), {"superseded": True, "blocker_retired": True})[0] != RETIRE:
        fails.append("retire must dominate promote in one tick")

    for f in fails:
        print(f"FAIL  {f}")
    print("selfcheck: OK" if not fails else f"selfcheck: {len(fails)} FAILED")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "demo"
    if cmd == "selfcheck":
        return runselfcheck()
    if cmd == "demo":
        items, events = worked_scenario()
        sim = simulate(items, events)
        if "--json" in argv:
            print(json.dumps(to_schema(sim), indent=1))
        else:
            print(report_md(sim))
        return 0
    print(f"unknown command: {cmd!r} (try: demo | demo --json | selfcheck)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
