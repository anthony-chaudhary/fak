package loopmgr

import "testing"

// stormPolicy is a compact policy carrying both self-tuning triggers: a
// refusal-storm cap and a witness floor gated after a handful of ended runs.
func stormPolicy() Policies {
	return Policies{
		Default: Policy{
			MaxConsecutiveRefusals: 3,
			MinWitnessRate:         0.5,
			MinRunsForWitnessGate:  4,
		},
	}
}

func statusOf(loops ...LoopSnapshot) Status { return Status{Loops: loops} }

// findProposal returns the first proposal for (loopID, field), or false.
func findProposal(ps []PolicyProposal, loopID, field string) (PolicyProposal, bool) {
	for _, p := range ps {
		if p.LoopID == loopID && p.Field == field {
			return p, true
		}
	}
	return PolicyProposal{}, false
}

// A storming loop proposes a bounded cadence-floor raise, stepped once (0->60 by
// the default step) and never above the ceiling.
func TestProposeStormRaisesCadenceBounded(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 3})
	ps := ProposePolicies(st, stormPolicy(), DefaultProposeConfig())

	p, ok := findProposal(ps, "x", PolicyFieldMinInterval)
	if !ok {
		t.Fatalf("storm must propose a cadence raise, got %+v", ps)
	}
	if p.Reason != ReasonRefusalStorm {
		t.Fatalf("reason = %q, want %q", p.Reason, ReasonRefusalStorm)
	}
	if p.Before != "0" || p.After != "60" {
		t.Fatalf("bounded step wrong: before=%q after=%q (want 0->60)", p.Before, p.After)
	}
	if p.Rationale == "" {
		t.Fatalf("proposal must carry a one-line rationale")
	}
}

// A step lands on the ceiling, not past it: from 3570 with a 60s step and a 3600
// ceiling, the proposal caps at 3600.
func TestProposeStormStepCapsAtCeiling(t *testing.T) {
	pol := Policies{Default: Policy{MaxConsecutiveRefusals: 3, MinIntervalSeconds: 3570}}
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 5})
	ps := ProposePolicies(st, pol, DefaultProposeConfig())

	p, ok := findProposal(ps, "x", PolicyFieldMinInterval)
	if !ok {
		t.Fatalf("storm below ceiling must still propose, got %+v", ps)
	}
	if p.After != "3600" {
		t.Fatalf("step must cap at the 3600 ceiling, got after=%q", p.After)
	}
}

// A loop already at/over the cadence ceiling yields NO cadence proposal: a bounded
// fold cannot propose past its bound.
func TestProposeStormAtCeilingNoProposal(t *testing.T) {
	pol := Policies{Default: Policy{MaxConsecutiveRefusals: 3, MinIntervalSeconds: 3600}}
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 9})
	ps := ProposePolicies(st, pol, DefaultProposeConfig())

	if _, ok := findProposal(ps, "x", PolicyFieldMinInterval); ok {
		t.Fatalf("a loop at the ceiling must not be proposed for, got %+v", ps)
	}
}

// A witness-collapsed loop (rate below floor over enough ended runs) proposes a pause.
func TestProposeWitnessCollapseProposesPause(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", Ended: 10, Witnessed: 1})
	ps := ProposePolicies(st, stormPolicy(), DefaultProposeConfig())

	p, ok := findProposal(ps, "x", PolicyFieldPaused)
	if !ok {
		t.Fatalf("witness collapse must propose a pause, got %+v", ps)
	}
	if p.Reason != ReasonWitnessCollapse {
		t.Fatalf("reason = %q, want %q", p.Reason, ReasonWitnessCollapse)
	}
	if p.Before != "false" || p.After != "true" {
		t.Fatalf("pause proposal wrong: before=%q after=%q", p.Before, p.After)
	}
}

// An already-paused loop yields no pause proposal (nothing to nudge).
func TestProposeAlreadyPausedNoProposal(t *testing.T) {
	pol := Policies{Default: Policy{Paused: true, MinWitnessRate: 0.5, MinRunsForWitnessGate: 4}}
	st := statusOf(LoopSnapshot{LoopID: "x", Ended: 10, Witnessed: 0})
	ps := ProposePolicies(st, pol, DefaultProposeConfig())

	if len(ps) != 0 {
		t.Fatalf("already-paused loop must yield no proposal, got %+v", ps)
	}
}

// A healthy loop — under its storm cap, witness rate above floor — yields nothing.
func TestProposeHealthyLoopNoProposal(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 1, Ended: 10, Witnessed: 9})
	ps := ProposePolicies(st, stormPolicy(), DefaultProposeConfig())

	if len(ps) != 0 {
		t.Fatalf("healthy loop must yield no proposal, got %+v", ps)
	}
}

// A young loop (fewer ended runs than the witness gate) is never witness-gated, so
// never proposed for pausing — mirrors Admit's empty-denominator guard.
func TestProposeYoungLoopNotWitnessGated(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", Ended: 2, Witnessed: 0})
	ps := ProposePolicies(st, stormPolicy(), DefaultProposeConfig())

	if _, ok := findProposal(ps, "x", PolicyFieldPaused); ok {
		t.Fatalf("a young loop must not be witness-gated, got %+v", ps)
	}
}

// A loop with no gates configured (zero policy) yields nothing even while storming
// and never witnessing: the fold triggers only on a configured gate, exactly as
// Admit refuses only on a configured gate.
func TestProposeZeroPolicyNoProposal(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 99, Ended: 100, Witnessed: 0})
	ps := ProposePolicies(st, Policies{}, DefaultProposeConfig())

	if len(ps) != 0 {
		t.Fatalf("zero policy must yield no proposal, got %+v", ps)
	}
}

// Output is stable-sorted by loop id, so a readout is deterministic regardless of
// fold order.
func TestProposeStableOrder(t *testing.T) {
	st := statusOf(
		LoopSnapshot{LoopID: "zeta", ConsecutiveRefusals: 5},
		LoopSnapshot{LoopID: "alpha", ConsecutiveRefusals: 5},
	)
	ps := ProposePolicies(st, stormPolicy(), DefaultProposeConfig())
	if len(ps) != 2 {
		t.Fatalf("want 2 proposals, got %+v", ps)
	}
	if ps[0].LoopID != "alpha" || ps[1].LoopID != "zeta" {
		t.Fatalf("not stable-sorted by loop id: %+v", ps)
	}
}

// A per-loop override is honored: a loop whose override drops the storm cap is not
// proposed for even while storming, proving the fold reads the EFFECTIVE policy.
func TestProposeHonorsPerLoopOverride(t *testing.T) {
	pol := Policies{
		Default: Policy{MaxConsecutiveRefusals: 3},
		Loops:   map[string]Policy{"quiet": {MaxConsecutiveRefusals: 0}},
	}
	st := statusOf(LoopSnapshot{LoopID: "quiet", ConsecutiveRefusals: 50})
	ps := ProposePolicies(st, pol, DefaultProposeConfig())
	if len(ps) != 0 {
		t.Fatalf("override dropping the cap must suppress the proposal, got %+v", ps)
	}
}

// A zero ProposeConfig is filled from the default (step 60, ceiling 3600), so a
// caller need not supply bounds.
func TestProposeZeroConfigUsesDefaults(t *testing.T) {
	st := statusOf(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 3})
	ps := ProposePolicies(st, stormPolicy(), ProposeConfig{})
	p, ok := findProposal(ps, "x", PolicyFieldMinInterval)
	if !ok || p.After != "60" {
		t.Fatalf("zero config must default to a 60s step, got %+v", ps)
	}
}
