package loopmgr

import (
	"fmt"
	"math"
	"strings"
)

// SpeculationCandidate is the folded, per-turn input to the speculation
// governor. It is deliberately just facts and estimates: the governor does not
// schedule, dispatch, or mutate anything.
type SpeculationCandidate struct {
	LoopID string `json:"loop_id,omitempty"`
	Tool   string `json:"tool,omitempty"`

	// Effect proof. The default is deny: a caller must positively prove an
	// effect-free surface and idempotence, and destructive/write-shaped tools
	// always refuse.
	ReadOnlyHint   bool `json:"read_only_hint,omitempty"`
	DryRunHint     bool `json:"dry_run_hint,omitempty"`
	IdempotentHint bool `json:"idempotent_hint,omitempty"`
	Destructive    bool `json:"destructive,omitempty"`

	// EV inputs, in latency-equivalent milliseconds:
	// P(correct)*LatencySavedMillis > CostIfWrongMillis.
	CorrectProbability float64 `json:"correct_probability,omitempty"`
	LatencySavedMillis int64   `json:"latency_saved_millis,omitempty"`
	CostIfWrongMillis  int64   `json:"cost_if_wrong_millis,omitempty"`

	// SlackMillis is the surplus capacity available to best-effort speculative
	// work. EstimatedWorkMillis consumes that budget; <=0 is treated as 1 so a
	// caller cannot omit work cost and bypass the slack gate.
	SlackMillis         int64 `json:"slack_millis,omitempty"`
	EstimatedWorkMillis int64 `json:"estimated_work_millis,omitempty"`
}

// Closed speculation-admission refusal vocabulary. These reasons are advisory:
// a refusal means "run the normal turn path", never "block the user's turn".
const (
	ReasonSpecAdmitted         = "SPEC_ADMITTED"
	ReasonSpecEVNegative       = "SPEC_EV_NEGATIVE"
	ReasonSpecNoSlack          = "SPEC_NO_SLACK"
	ReasonSpecEffectfulRefused = "SPEC_EFFECTFUL_REFUSED"
)

// SpeculationEffectReason is the closed effect-classifier verdict carried inside
// a SPEC_EFFECTFUL_REFUSED decision.
type SpeculationEffectReason string

const (
	SpeculationEffectFree              SpeculationEffectReason = "SPEC_EFFECT_FREE"
	SpeculationEffectMissingReadOnly   SpeculationEffectReason = "SPEC_EFFECT_MISSING_READ_ONLY"
	SpeculationEffectMissingIdempotent SpeculationEffectReason = "SPEC_EFFECT_MISSING_IDEMPOTENT"
	SpeculationEffectDestructive       SpeculationEffectReason = "SPEC_EFFECT_DESTRUCTIVE"
)

// Refused reports whether the effect classifier denied speculation.
func (r SpeculationEffectReason) Refused() bool { return r != SpeculationEffectFree }

// SpeculationDecision is the advisory verdict returned by AdmitSpeculation.
type SpeculationDecision struct {
	LoopID               string                  `json:"loop_id,omitempty"`
	Tool                 string                  `json:"tool,omitempty"`
	Admit                bool                    `json:"admit"`
	Reason               string                  `json:"reason"`
	Summary              string                  `json:"summary"`
	EffectReason         SpeculationEffectReason `json:"effect_reason,omitempty"`
	ExpectedValueMillis  float64                 `json:"expected_value_millis,omitempty"`
	SlackRemainingMillis int64                   `json:"slack_remaining_millis,omitempty"`
	EstimatedWorkMillis  int64                   `json:"estimated_work_millis,omitempty"`
}

// AdmitSpeculation applies the speculation governor to one candidate turn. It is
// pure and fixed-order: correctness first (default-deny effects), then capacity
// (slack-only), then economics (positive EV). The first failing gate wins.
func AdmitSpeculation(c SpeculationCandidate) SpeculationDecision {
	effect, ok := ClassifySpeculationEffect(c)
	if !ok {
		return refuseSpec(c, ReasonSpecEffectfulRefused, effect, 0, 0,
			fmt.Sprintf("effect classifier refused speculation: %s", effect))
	}

	work := speculationWorkMillis(c)
	if c.SlackMillis <= 0 || work > c.SlackMillis {
		return refuseSpec(c, ReasonSpecNoSlack, effect, 0, work,
			fmt.Sprintf("speculation needs %dms slack but only %dms surplus is available", work, c.SlackMillis))
	}

	ev, valid := SpeculationExpectedValueMillis(c.CorrectProbability, c.LatencySavedMillis, c.CostIfWrongMillis)
	if !valid {
		return refuseSpec(c, ReasonSpecEVNegative, effect, ev, work, "expected-value inputs are invalid")
	}
	if ev <= 0 {
		return refuseSpec(c, ReasonSpecEVNegative, effect, ev, work,
			fmt.Sprintf("expected value %.2fms is not positive", ev))
	}

	return SpeculationDecision{
		LoopID:               c.LoopID,
		Tool:                 c.Tool,
		Admit:                true,
		Reason:               ReasonSpecAdmitted,
		Summary:              fmt.Sprintf("speculation admitted: expected value %.2fms, slack remaining %dms", ev, c.SlackMillis-work),
		EffectReason:         effect,
		ExpectedValueMillis:  ev,
		SlackRemainingMillis: c.SlackMillis - work,
		EstimatedWorkMillis:  work,
	}
}

// SpeculationExpectedValueMillis computes P(correct)*latency_saved - cost_if_wrong
// and reports whether the inputs were valid. Probability must be finite in [0,1],
// latency saved must be positive, and wrong-path cost cannot be negative.
func SpeculationExpectedValueMillis(pCorrect float64, latencySavedMillis, costIfWrongMillis int64) (float64, bool) {
	if math.IsNaN(pCorrect) || math.IsInf(pCorrect, 0) || pCorrect < 0 || pCorrect > 1 {
		return 0, false
	}
	if latencySavedMillis <= 0 || costIfWrongMillis < 0 {
		return 0, false
	}
	return pCorrect*float64(latencySavedMillis) - float64(costIfWrongMillis), true
}

// ClassifySpeculationEffect is the default-deny effect classifier. A candidate is
// effect-free only with a positive read-only or verified-dry-run proof, a positive
// idempotence proof, and no destructive/write-shaped signal.
func ClassifySpeculationEffect(c SpeculationCandidate) (SpeculationEffectReason, bool) {
	if !c.ReadOnlyHint && !c.DryRunHint {
		return SpeculationEffectMissingReadOnly, false
	}
	if !c.IdempotentHint {
		return SpeculationEffectMissingIdempotent, false
	}
	if c.Destructive || speculationWriteShaped(c.Tool) {
		return SpeculationEffectDestructive, false
	}
	return SpeculationEffectFree, true
}

func refuseSpec(c SpeculationCandidate, reason string, effect SpeculationEffectReason, ev float64, work int64, summary string) SpeculationDecision {
	return SpeculationDecision{
		LoopID:              c.LoopID,
		Tool:                c.Tool,
		Admit:               false,
		Reason:              reason,
		Summary:             summary,
		EffectReason:        effect,
		ExpectedValueMillis: ev,
		EstimatedWorkMillis: work,
	}
}

func speculationWorkMillis(c SpeculationCandidate) int64 {
	if c.EstimatedWorkMillis > 0 {
		return c.EstimatedWorkMillis
	}
	return 1
}

// Keep this mirror in sync with internal/vdso.WriteShapeNeedles. loopmgr is a
// tier-1 stdlib-only package, so it cannot import vdso without violating the
// architecture gate; speculation_test.go pins the mirror against the exported
// vDSO list.
var speculationWriteShapeNeedles = []string{"write", "edit", "delete", "patch", "exec", "run", "book", "update", "cancel", "send"}

func speculationWriteShaped(tool string) bool {
	t := strings.ToLower(tool)
	for _, p := range speculationWriteShapeNeedles {
		if strings.Contains(t, p) {
			return true
		}
	}
	return false
}
