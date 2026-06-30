package loopmgr

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// Governor admission. loopmgr records events and folds them; it does not
// schedule or authorize. Admit is the one pure exception to that rule by
// design: it takes an already-folded loop snapshot plus an operator-tunable
// Policy and returns an advisory decision. It performs no I/O, spawns nothing,
// and trusts no worker's self-report — every input is a ledger-derived fold.
// The caller (a producer tick, or `fak loop admit`) enforces the verdict.
//
// This is the tunable backpressure the always-on loop needs: one policy an
// operator edits to pause a loop, floor its cadence, back it off when it keeps
// getting refused, or hold it when its witnessed/claimed ratio collapses —
// without re-registering the OS scheduler task.

// Policy is the operator-tunable admission policy for one loop. The zero value
// is permissive (admit always): every gate is opt-in, so an unconfigured loop
// behaves exactly as it did before a policy existed.
type Policy struct {
	// Paused refuses every fire with REASON_LOOP_PAUSED. The operator's pause
	// switch — flip it without touching the scheduler that fires the loop.
	Paused bool `json:"paused,omitempty"`

	// Disabled refuses every fire with REASON_LOOP_DISABLED. Stronger than
	// Paused in intent (a loop retired rather than temporarily held), but the
	// mechanism is the same refusal.
	Disabled bool `json:"disabled,omitempty"`

	// MaxConcurrent is the in-flight concurrency budget: refuse a fire once this
	// loop already has this many runs started-but-not-ended. 0 disables the gate
	// (no concurrency ceiling). This is the tunable form of the schedule's binary
	// overlap-lock (which caps in-flight at 1): a budget of N lets up to N runs of
	// the loop overlap and refuses the (N+1)th with REASON_BUDGET_SPENT, so a
	// super-loop's generations cannot starve the fleet (#1653) and a dispatch loop
	// can be given a derived ceiling instead of a fixed cap (#1333). The in-flight
	// count is read straight off the fold (Started-Ended); no live-run probe.
	MaxConcurrent uint64 `json:"max_concurrent,omitempty"`

	// MinIntervalSeconds is the cadence floor: refuse a fire that lands sooner
	// than this many seconds after the loop's last event. 0 disables the gate.
	// Stops a misconfigured scheduler (or two overlapping ones) from storming
	// the same loop faster than intended.
	MinIntervalSeconds int64 `json:"min_interval_seconds,omitempty"`

	// MaxConsecutiveRefusals backs a loop off after it has been refused this
	// many times in a row with nothing admitted in between. 0 disables. This is
	// the anti-storm gate for the job-repo failure mode: re-firing one
	// un-landable unit forever. Once tripped, the loop needs an operator nudge
	// (or a successful admit from another path) to clear.
	MaxConsecutiveRefusals uint64 `json:"max_consecutive_refusals,omitempty"`

	// MinWitnessRate holds a loop whose witnessed/claimed completion ratio has
	// fallen below this fraction (0..1), but only once it has at least
	// MinRunsForWitnessGate ended runs — so a brand-new loop is never gated on
	// an empty denominator. 0 disables. This is the "stopped talking is not
	// done" gate: a loop that keeps claiming done without independent evidence
	// is held, not trusted.
	MinWitnessRate        float64 `json:"min_witness_rate,omitempty"`
	MinRunsForWitnessGate uint64  `json:"min_runs_for_witness_gate,omitempty"`
}

// Structured refusal reasons. Closed vocabulary, emittable and verifiable, in
// the spirit of the DOS refusal set: a refusal carries a reason a downstream
// can route on, never free-text drift.
const (
	ReasonLoopPaused      = "LOOP_PAUSED"
	ReasonLoopDisabled    = "LOOP_DISABLED"
	ReasonBudgetSpent     = "BUDGET_SPENT"
	ReasonCadenceFloor    = "CADENCE_FLOOR"
	ReasonRefusalStorm    = "REFUSAL_STORM"
	ReasonWitnessCollapse = "WITNESS_COLLAPSE"
	ReasonAdmitted        = "POLICY_ADMITTED"
)

// Decision is the advisory verdict Admit returns. Admit is true to proceed.
type Decision struct {
	LoopID  string `json:"loop_id"`
	Admit   bool   `json:"admit"`
	Reason  string `json:"reason"`
	Summary string `json:"summary"`
}

// Admit applies policy to a folded loop snapshot at time now. It is pure: no
// I/O, no clock read (now is supplied), no mutation. Gates are checked in a
// fixed order so the reason is deterministic; the first failing gate wins.
func Admit(loop LoopSnapshot, policy Policy, now time.Time) Decision {
	d := Decision{LoopID: loop.LoopID, Admit: true, Reason: ReasonAdmitted}

	if policy.Disabled {
		return refuse(loop.LoopID, ReasonLoopDisabled, "loop disabled by policy")
	}
	if policy.Paused {
		return refuse(loop.LoopID, ReasonLoopPaused, "loop paused by policy")
	}

	// Budget gate before cadence: a concurrency ceiling is a hard STRUCTURAL
	// refuse (like the schedule's overlap-lock), not a soft cadence smoothing, so
	// it is checked alongside the other hard refusals and ahead of the
	// timing/storm/witness gates. inFlight is read off the fold (Started-Ended);
	// at or over the budget, the (N+1)th fire is refused.
	if policy.MaxConcurrent > 0 {
		inFlight := loop.Concurrent()
		if inFlight >= policy.MaxConcurrent {
			return refuse(loop.LoopID, ReasonBudgetSpent,
				fmt.Sprintf("%d runs in flight at/over the %d concurrency budget — wait for a run to end",
					inFlight, policy.MaxConcurrent))
		}
	}

	if policy.MinIntervalSeconds > 0 && loop.LastEventUnixNano > 0 {
		elapsed := now.UTC().UnixNano() - loop.LastEventUnixNano
		floor := policy.MinIntervalSeconds * int64(time.Second)
		if elapsed >= 0 && elapsed < floor {
			return refuse(loop.LoopID, ReasonCadenceFloor,
				fmt.Sprintf("fired %ds into a %ds cadence floor",
					elapsed/int64(time.Second), policy.MinIntervalSeconds))
		}
	}

	if policy.MaxConsecutiveRefusals > 0 && loop.ConsecutiveRefusals >= policy.MaxConsecutiveRefusals {
		return refuse(loop.LoopID, ReasonRefusalStorm,
			fmt.Sprintf("%d consecutive refusals at/over the %d cap — back off until a nudge",
				loop.ConsecutiveRefusals, policy.MaxConsecutiveRefusals))
	}

	if policy.MinWitnessRate > 0 {
		ended := loop.Ended
		if ended >= policy.MinRunsForWitnessGate && ended > 0 {
			rate := float64(loop.Witnessed) / float64(ended)
			if rate < policy.MinWitnessRate {
				return refuse(loop.LoopID, ReasonWitnessCollapse,
					fmt.Sprintf("witness rate %.2f below floor %.2f over %d ended runs",
						rate, policy.MinWitnessRate, ended))
			}
		}
	}

	d.Summary = "policy admitted: no gate tripped"
	return d
}

func refuse(loopID, reason, summary string) Decision {
	return Decision{LoopID: loopID, Admit: false, Reason: reason, Summary: summary}
}

// Policies is the on-disk tunable policy set: a default applied to every loop,
// plus per-loop overrides keyed by loop id. This is the single surface an
// operator edits to tune the always-on fleet.
type Policies struct {
	Schema  string            `json:"schema"`
	Default Policy            `json:"default"`
	Loops   map[string]Policy `json:"loops,omitempty"`
}

// SchemaPolicies is the policy-document schema tag.
const SchemaPolicies = "fak.loop-policy.v1"

// LoadPolicies reads a policy document from path. A missing file is NOT an
// error: it returns an empty (permissive) policy set, so an operator who has
// not written a policy gets the same admit-always behavior they had before.
// This fail-open default is deliberate — the governor adds backpressure when
// asked, never silently throttles a loop nobody configured. A present-but-
// malformed file IS an error: a typo should be loud, not silently permissive.
func LoadPolicies(path string) (Policies, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Policies{}, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Policies{}, nil
	}
	if err != nil {
		return Policies{}, fmt.Errorf("read loop policy: %w", err)
	}
	var p Policies
	if err := json.Unmarshal(b, &p); err != nil {
		return Policies{}, fmt.Errorf("decode loop policy %s: %w", path, err)
	}
	if p.Schema != "" && p.Schema != SchemaPolicies {
		return Policies{}, fmt.Errorf("loop policy schema = %q, want %q", p.Schema, SchemaPolicies)
	}
	return p, nil
}

// PolicyFor returns the effective policy for a loop id: the per-loop override
// if present, else the default. Per-loop overrides replace the default whole;
// they are not merged field-by-field, so an override is self-contained and
// readable on its own.
func (p Policies) PolicyFor(loopID string) Policy {
	if p.Loops != nil {
		if pol, ok := p.Loops[strings.TrimSpace(loopID)]; ok {
			return pol
		}
	}
	return p.Default
}

// AdmitAll applies the policy set to every loop in a status fold, returning one
// decision per loop in stable loop-id order. Useful for `fak loop admit` with
// no specific loop named: a fleet-wide governor readout.
func AdmitAll(st Status, policies Policies, now time.Time) []Decision {
	out := make([]Decision, 0, len(st.Loops))
	for _, loop := range st.Loops {
		out = append(out, Admit(loop, policies.PolicyFor(loop.LoopID), now))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LoopID < out[j].LoopID })
	return out
}
