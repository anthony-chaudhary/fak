#!/usr/bin/env python3
"""Tests for generation_lifecycle_sim -- the promotion/demotion simulation (#1656).

These pin the load-bearing properties: the four canonical verbs move items the
right direction (and CLAMP), retire is terminal, park is an active/inactive bit,
precedence is retire > demote > park > promote, and -- the headline --
generation stays ORTHOGONAL to priority, shared trunk, and runtime feature gates
(priority/gate/trunk survive every transition byte-identical). The worked
scenario's before/after distribution is the contract fixture.

Run:  python tools/generation_lifecycle_sim_test.py   (or: pytest tools/generation_lifecycle_sim_test.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generation_lifecycle_sim as g


# --------------------------------------------------------------------------- #
# decide() -- the evidence -> verb mapping (executable docs/generation.md)
# --------------------------------------------------------------------------- #
def test_blocker_retired_promotes():
    verb, _ = g.decide(g.Item("x", g.SECOND_NEXT), {"blocker_retired": True})
    assert verb == g.PROMOTE


def test_assumption_failed_with_nearer_path_demotes():
    verb, _ = g.decide(g.Item("x", g.NEXT), {"assumption_failed": True, "has_nearer_path": True})
    assert verb == g.DEMOTE


def test_assumption_failed_no_nearer_path_retires():
    verb, _ = g.decide(g.Item("x", g.NEXT), {"assumption_failed": True, "has_nearer_path": False})
    assert verb == g.RETIRE


def test_witness_regressed_demotes():
    verb, _ = g.decide(g.Item("x", g.SECOND_NEXT), {"witness_regressed": True})
    assert verb == g.DEMOTE


def test_superseded_retires():
    verb, _ = g.decide(g.Item("x", g.NEXT), {"superseded": True})
    assert verb == g.RETIRE


def test_option_cost_exceeds_value_retires():
    verb, _ = g.decide(g.Item("x", g.FUTURE), {"option_cost_exceeds_value": True})
    assert verb == g.RETIRE


def test_no_owner_parks():
    verb, _ = g.decide(g.Item("x", g.FUTURE), {"no_owner": True})
    assert verb == g.PARK


def test_no_evidence_holds():
    verb, _ = g.decide(g.Item("x", g.NEXT), {})
    assert verb == g.HOLD


# --------------------------------------------------------------------------- #
# apply() -- direction + clamping + terminal retire
# --------------------------------------------------------------------------- #
def test_promote_moves_one_closer():
    assert g.apply(g.Item("x", g.FUTURE), g.PROMOTE).stream == g.SECOND_NEXT


def test_promote_clamps_at_now():
    assert g.apply(g.Item("x", g.NOW), g.PROMOTE).stream == g.NOW


def test_demote_moves_one_farther():
    assert g.apply(g.Item("x", g.NOW), g.DEMOTE).stream == g.NEXT


def test_demote_clamps_at_future():
    assert g.apply(g.Item("x", g.FUTURE), g.DEMOTE).stream == g.FUTURE


def test_retire_is_terminal_and_absorbing():
    r = g.apply(g.Item("x", g.NEXT), g.RETIRE)
    assert r.stream == g.RETIRED
    # a retired item ignores ALL further evidence (even a promotion signal).
    assert g.decide(r, {"blocker_retired": True})[0] == g.HOLD
    assert g.apply(r, g.PROMOTE).stream == g.RETIRED  # apply is a no-op via HOLD in practice


def test_park_holds_stream_and_flags_inactive():
    p = g.apply(g.Item("x", g.FUTURE), g.PARK)
    assert p.stream == g.FUTURE and p.parked is True


def test_promote_reactivates_a_parked_item():
    parked = g.apply(g.Item("x", g.FUTURE), g.PARK)
    revived = g.apply(parked, g.PROMOTE)
    assert revived.parked is False and revived.stream == g.SECOND_NEXT


# --------------------------------------------------------------------------- #
# precedence -- retire > demote > park > promote in a single tick
# --------------------------------------------------------------------------- #
def test_retire_dominates_promote_same_tick():
    verb, _ = g.decide(g.Item("x", g.SECOND_NEXT), {"superseded": True, "blocker_retired": True})
    assert verb == g.RETIRE


def test_demote_dominates_promote_same_tick():
    verb, _ = g.decide(g.Item("x", g.SECOND_NEXT), {"witness_regressed": True, "blocker_retired": True})
    assert verb == g.DEMOTE


def test_demote_dominates_park_same_tick():
    verb, _ = g.decide(g.Item("x", g.NEXT), {"witness_regressed": True, "no_owner": True})
    assert verb == g.DEMOTE


# --------------------------------------------------------------------------- #
# ORTHOGONALITY -- the headline invariant (#1656 acceptance criterion 2).
# generation is orthogonal to priority, shared trunk, and runtime feature gates.
# --------------------------------------------------------------------------- #
def test_priority_gate_trunk_survive_every_verb():
    base = g.Item("x", g.SECOND_NEXT, priority="P0", gate="operator-only", trunk="main")
    for verb in (g.PROMOTE, g.DEMOTE, g.PARK, g.RETIRE, g.HOLD):
        out = g.apply(base, verb)
        assert out.priority == "P0", f"{verb} changed priority"
        assert out.gate == "operator-only", f"{verb} toggled the runtime gate"
        assert out.trunk == "main", f"{verb} touched shared-trunk state"


def test_full_scenario_preserves_orthogonal_fields():
    # Across a whole timeline, each item's priority/gate/trunk are byte-identical
    # before and after -- horizon moved, urgency/exposure/branch did not.
    items, events = g.worked_scenario()
    sim = g.simulate(items, events)
    for iid, before in sim["before"].items():
        after = sim["after"][iid]
        assert (after.priority, after.gate, after.trunk) == (before.priority, before.gate, before.trunk)


def test_model_has_no_branch_state():
    # Shared trunk: a stream is a label partition, never an integration branch.
    # The Item dataclass must expose no branch/worktree field, and trunk stays main.
    fields = set(g.Item("x", g.NOW).__dataclass_fields__)
    assert "branch" not in fields and "worktree" not in fields
    assert g.Item("x", g.NOW).trunk == "main"


# --------------------------------------------------------------------------- #
# the worked scenario -- the contract fixture (before/after distribution)
# --------------------------------------------------------------------------- #
def test_worked_scenario_before_after_distribution():
    items, events = g.worked_scenario()
    sim = g.simulate(items, events)
    before = g.distribution(sim["before"])
    after = g.distribution(sim["after"])

    # BEFORE: now0 next2(B,E) second_next2(A,D) future1(C), nothing parked/retired.
    assert (before["gen/now"], before["gen/next"], before["gen/second-next"],
            before["gen/future"], before["retired"]) == (0, 2, 2, 1, 0)

    # AFTER: B promoted next->now; A promoted 2nd-next->next; C parked-at-future
    # then re-activated to 2nd-next; D demoted then retired; E retired.
    assert (after["gen/now"], after["gen/next"], after["gen/second-next"],
            after["gen/future"], after["retired"]) == (1, 1, 1, 0, 2)
    assert after["parked_active_items"] == 0  # C was re-activated by the final promote

    # every event produced a recorded transition (the rules were exercised).
    assert len(sim["trace"]) == len(events)
    # all four verbs appear in the trace.
    verbs = {t.verb for t in sim["trace"]}
    assert {g.PROMOTE, g.DEMOTE, g.RETIRE, g.PARK} <= verbs


def test_schema_shape():
    items, events = g.worked_scenario()
    obj = g.to_schema(g.simulate(items, events))
    assert obj["schema"] == "fak-generation-lifecycle/1"
    assert obj["issue"] == 1656
    assert "distribution" in obj["before"] and "distribution" in obj["after"]
    assert isinstance(obj["trace"], list) and len(obj["trace"]) == len(events)


def test_selfcheck_passes():
    assert g.runselfcheck() == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERR   {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
