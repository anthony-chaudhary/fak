package loopmgr

import (
	"fmt"
	"sort"
	"strconv"
)

// Propose-only governor-policy fold (#3976). Admit is the operator-tunable gate;
// its Policy knobs are hand-tuned, and a tripped storm cap stays tripped "until an
// operator nudge" ([Policy.MaxConsecutiveRefusals]). This fold closes that loop the
// same way rsiloop.metarsi does for the RSI knobs: it reads the already-folded loop
// status, notices the SAME clustered evidence Admit refuses on, and returns a
// BOUNDED, propose-only knob nudge with a one-line rationale. It never schedules,
// never authorizes, and — the load-bearing property — never writes the policy file.
// A proposal is a hypothesis; the human gate is the edit the operator lands.

// PolicyFieldMinInterval and PolicyFieldPaused name the Policy knobs a proposal
// nudges. They match the on-disk JSON tags so a readout points an operator straight
// at the field to edit.
const (
	PolicyFieldMinInterval = "min_interval_seconds"
	PolicyFieldPaused      = "paused"
)

// ProposeConfig bounds the propose-only fold: a storming loop's cadence floor is
// nudged by one IntervalStep per proposal and never past IntervalCeiling — the same
// step+ceiling discipline as rsiloop.MetaConfig, so clustered evidence can nudge the
// gate but never slam it. The zero value is filled from [DefaultProposeConfig].
type ProposeConfig struct {
	// IntervalStep is the seconds added to a storming loop's MinIntervalSeconds per
	// proposal (a single bounded raise, not a run-to-ceiling).
	IntervalStep int64 `json:"interval_step_seconds"`
	// IntervalCeiling caps the proposed MinIntervalSeconds: a loop already at/over
	// this yields no cadence proposal, so the fold can never propose past its bound.
	IntervalCeiling int64 `json:"interval_ceiling_seconds"`
}

// DefaultProposeConfig is the conservative default: raise a storming loop's cadence
// floor by 60s per proposal, never past 1h. Both are well below any legitimate
// fleet cadence catastrophe, so a proposal only ever slows a genuinely storming loop.
func DefaultProposeConfig() ProposeConfig {
	return ProposeConfig{IntervalStep: 60, IntervalCeiling: 3600}
}

func (c ProposeConfig) withDefaults() ProposeConfig {
	d := DefaultProposeConfig()
	if c.IntervalStep <= 0 {
		c.IntervalStep = d.IntervalStep
	}
	if c.IntervalCeiling <= 0 {
		c.IntervalCeiling = d.IntervalCeiling
	}
	return c
}

// PolicyProposal is one BOUNDED, propose-only governor-policy nudge for a single
// loop: a hypothesis the operator lands by editing the policy file, never a value
// this package writes. Field names the Policy knob; Reason is the structured trigger
// drawn from the SAME closed refusal vocabulary Admit emits (REFUSAL_STORM,
// WITNESS_COLLAPSE); Before/After render the knob's current and proposed value; and
// Rationale is the one-line why an operator reads before landing the edit.
type PolicyProposal struct {
	LoopID    string `json:"loop_id"`
	Field     string `json:"field"`
	Reason    string `json:"reason"`
	Before    string `json:"before"`
	After     string `json:"after"`
	Rationale string `json:"rationale"`
}

// ProposePolicies is the PURE, propose-only governor fold: it reads a folded loop
// status and the current policy set and returns bounded policy-knob nudges for loops
// whose ledger signal has tripped a governor gate. A refusal-storming loop (at/over
// its MaxConsecutiveRefusals cap — the REFUSAL_STORM condition) proposes a bounded
// MinIntervalSeconds raise; a witness-collapsed loop (below MinWitnessRate over
// enough ended runs — the WITNESS_COLLAPSE condition) proposes Paused:true. The
// triggers are exactly the conditions Admit refuses on, so the fold proposes the fix
// for the very gate that trips.
//
// It is pure: no I/O, no clock read, no mutation of cur. It NEVER writes the policy
// file — the operator lands the edit (cmd/fak/loop.go re-reads the policy per admit
// tick, so a landed edit needs no reload). Bounds come from cfg (filled from
// [DefaultProposeConfig] when zero). Output is stable-sorted by loop id then field,
// so the readout is deterministic.
func ProposePolicies(st Status, cur Policies, cfg ProposeConfig) []PolicyProposal {
	cfg = cfg.withDefaults()
	var out []PolicyProposal
	for _, loop := range st.Loops {
		pol := cur.PolicyFor(loop.LoopID)
		if p, ok := proposeCadenceForStorm(loop, pol, cfg); ok {
			out = append(out, p)
		}
		if p, ok := proposePauseForCollapse(loop, pol); ok {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].LoopID != out[j].LoopID {
			return out[i].LoopID < out[j].LoopID
		}
		return out[i].Field < out[j].Field
	})
	return out
}

// proposeCadenceForStorm proposes a bounded MinIntervalSeconds raise when a loop is
// at/over its refusal-storm cap — the same condition Admit refuses with
// REFUSAL_STORM. The raise is one bounded step capped at the ceiling; a loop already
// at/over the ceiling yields no proposal (a bounded fold cannot propose past its
// bound). A loop with no storm cap configured (cap 0) is never gated, so never
// proposed for.
func proposeCadenceForStorm(loop LoopSnapshot, pol Policy, cfg ProposeConfig) (PolicyProposal, bool) {
	if pol.MaxConsecutiveRefusals == 0 || loop.ConsecutiveRefusals < pol.MaxConsecutiveRefusals {
		return PolicyProposal{}, false
	}
	cur := pol.MinIntervalSeconds
	next := cur + cfg.IntervalStep
	if next > cfg.IntervalCeiling {
		next = cfg.IntervalCeiling
	}
	if next <= cur {
		return PolicyProposal{}, false // already at/over the bounded ceiling
	}
	return PolicyProposal{
		LoopID: loop.LoopID,
		Field:  PolicyFieldMinInterval,
		Reason: ReasonRefusalStorm,
		Before: strconv.FormatInt(cur, 10),
		After:  strconv.FormatInt(next, 10),
		Rationale: fmt.Sprintf(
			"%d consecutive refusals at/over the %d cap; raise the cadence floor %ds->%ds (bounded by the %ds ceiling) to back the storming loop off",
			loop.ConsecutiveRefusals, pol.MaxConsecutiveRefusals, cur, next, cfg.IntervalCeiling),
	}, true
}

// proposePauseForCollapse proposes Paused:true when a loop's witnessed/claimed ratio
// has fallen below its policy floor over enough ended runs — the same condition Admit
// refuses with WITNESS_COLLAPSE. A loop already paused, or with no witness floor
// configured, or too young for the gate (fewer than MinRunsForWitnessGate ended
// runs) yields no proposal.
func proposePauseForCollapse(loop LoopSnapshot, pol Policy) (PolicyProposal, bool) {
	if pol.Paused || pol.MinWitnessRate <= 0 {
		return PolicyProposal{}, false
	}
	ended := loop.Ended
	if ended == 0 || ended < pol.MinRunsForWitnessGate {
		return PolicyProposal{}, false
	}
	rate := float64(loop.Witnessed) / float64(ended)
	if rate >= pol.MinWitnessRate {
		return PolicyProposal{}, false
	}
	return PolicyProposal{
		LoopID: loop.LoopID,
		Field:  PolicyFieldPaused,
		Reason: ReasonWitnessCollapse,
		Before: "false",
		After:  "true",
		Rationale: fmt.Sprintf(
			"witness rate %.2f below the %.2f floor over %d ended runs; propose pausing the loop until it earns independent evidence again",
			rate, pol.MinWitnessRate, ended),
	}, true
}
