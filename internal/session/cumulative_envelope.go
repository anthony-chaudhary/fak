package session

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// cumulative_envelope.go is the refusal-aware cumulative work envelope for one
// live session (#4778). It complements Budget/TimeBudget rather than replacing
// them: those types govern remaining served-session axes, while this fold meters
// work already paid across replayed model/tool turns, including the cached versus
// uncached split that a resident-context ceiling cannot see.

// SessionEnvelopeReason is the typed reason a cumulative envelope requests a
// recovery checkpoint. It is intentionally separate from both State.Reason (the
// run-state control plane) and abi.ReasonCode (one tool denial): observing a deny
// never converts it to an allow or copies SELF_MODIFY into a session stop reason.
type SessionEnvelopeReason string

const (
	ReasonEnvelopeUncachedInput   SessionEnvelopeReason = "UNCACHED_INPUT_ENVELOPE"
	ReasonEnvelopeWallTime        SessionEnvelopeReason = "WALL_TIME_ENVELOPE"
	ReasonEnvelopeSemanticRefusal SessionEnvelopeReason = "SEMANTIC_REFUSAL_ENVELOPE"
)

// CumulativeEnvelopeAction is the closed intervention vocabulary returned by
// CumulativeEnvelope.Observe.
type CumulativeEnvelopeAction uint8

const (
	EnvelopeActionContinue CumulativeEnvelopeAction = iota
	EnvelopeActionCheckpointRecovery
)

func (a CumulativeEnvelopeAction) String() string {
	switch a {
	case EnvelopeActionContinue:
		return "CONTINUE"
	case EnvelopeActionCheckpointRecovery:
		return "CHECKPOINT_RECOVERY"
	default:
		return "CUMULATIVE_ENVELOPE_ACTION_UNKNOWN"
	}
}

// CumulativeEnvelopePolicy is the resolved policy for one session. Non-positive
// maxima are unlimited/disabled, matching this package's existing unbounded
// envelope convention. MinRefusalDensity is the fraction of observed tool calls
// that must be refused before MaxEquivalentRefusals can trip; zero disables the
// density qualifier while retaining the equivalent-refusal bound.
type CumulativeEnvelopePolicy struct {
	MaxUncachedInputTokens int64   `json:"max_uncached_input_tokens,omitempty"`
	MaxWallTimeNanos       int64   `json:"max_wall_time_nanos,omitempty"`
	MaxEquivalentRefusals  int     `json:"max_equivalent_refusals,omitempty"`
	MinRefusalDensity      float64 `json:"min_refusal_density,omitempty"`
}

// CumulativeEnvelopeSample is one independently attributed delta. InputTokens
// includes CachedInputTokens; the fold charges their non-negative difference to
// UncachedInputTokens. Outcome is the latest tool outcome, if any. ToolCalls must
// include Outcome when set; a refused Outcome with ToolCalls==0 is conservatively
// counted as one call so refusal density cannot divide by zero.
type CumulativeEnvelopeSample struct {
	InputTokens         int64
	CachedInputTokens   int64
	OutputTokens        int64
	ModelCalls          int
	ToolCalls           int
	ManualContinuations int
	WallTimeNanos       int64
	Outcome             ToolCallOutcome
}

// CumulativeEnvelopeTotals is the cumulative, never-reset accounting folded from
// session-attributed deltas. Cached input remains explicit and non-zero-cost; only
// the difference is compared with the uncached envelope.
type CumulativeEnvelopeTotals struct {
	InputTokens         int64 `json:"input_tokens"`
	CachedInputTokens   int64 `json:"cached_input_tokens"`
	UncachedInputTokens int64 `json:"uncached_input_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
	ModelCalls          int   `json:"model_calls"`
	ToolCalls           int   `json:"tool_calls"`
	ManualContinuations int   `json:"manual_continuations"`
	Refusals            int   `json:"refusals"`
	WallTimeNanos       int64 `json:"wall_time_nanos"`
}

// SessionRecoveryCheckpoint is the compact drive-state pointer captured when the
// envelope trips. Goal and PendingTurn are copied from the latest State passed to
// Observe, so a caller can checkpoint/re-route without losing the active root or
// the write-ahead retry position. Reason states why recovery was requested.
type SessionRecoveryCheckpoint struct {
	Reason         SessionEnvelopeReason `json:"reason"`
	TraceID        string                `json:"trace_id"`
	Goal           Goal                  `json:"goal,omitempty,omitzero"`
	PendingTurn    PendingTurn           `json:"pending_turn,omitempty,omitzero"`
	ContinuationID string                `json:"continuation_id,omitempty"`
	Generation     int                   `json:"generation,omitempty"`
	StateRev       uint64                `json:"state_rev"`
}

// CumulativeEnvelopeDecision is the fold's current verdict. Outcome is copied
// verbatim from the latest sample: a CHECKPOINT_RECOVERY action therefore cannot
// accidentally weaken or rewrite the tool denial that contributed to it.
type CumulativeEnvelopeDecision struct {
	Action             CumulativeEnvelopeAction
	Reason             SessionEnvelopeReason
	Totals             CumulativeEnvelopeTotals
	EquivalentRefusals int
	RefusalDensity     float64
	Outcome            ToolCallOutcome
	Recovery           SessionRecoveryCheckpoint
}

// CumulativeEnvelope folds attributed deltas for exactly one session. A trip is
// latched: subsequent Observe calls return the same checkpoint receipt instead of
// repeating warnings or silently moving the recovery pointer after intervention.
type CumulativeEnvelope struct {
	policy CumulativeEnvelopePolicy
	totals CumulativeEnvelopeTotals

	lastRefusalKey semanticRefusalKey
	equivalent     int
	tripped        bool
	decision       CumulativeEnvelopeDecision
}

// NewCumulativeEnvelope returns an empty per-session fold governed by policy.
func NewCumulativeEnvelope(policy CumulativeEnvelopePolicy) *CumulativeEnvelope {
	return &CumulativeEnvelope{policy: policy}
}

// Observe folds one session-attributed delta and returns either CONTINUE or one
// latched CHECKPOINT_RECOVERY receipt. It never mutates State or Outcome, so the
// existing denial and run-state control planes remain at least as restrictive as
// they were before this accounting signal was attached.
func (e *CumulativeEnvelope) Observe(state State, sample CumulativeEnvelopeSample) CumulativeEnvelopeDecision {
	if e == nil {
		return CumulativeEnvelopeDecision{Action: EnvelopeActionContinue, Outcome: sample.Outcome}
	}
	if e.tripped {
		return e.decision
	}

	e.foldTotals(sample)
	e.foldRefusal(sample.Outcome)

	density := 0.0
	if e.totals.ToolCalls > 0 {
		density = float64(e.totals.Refusals) / float64(e.totals.ToolCalls)
	}
	decision := CumulativeEnvelopeDecision{
		Action:             EnvelopeActionContinue,
		Totals:             e.totals,
		EquivalentRefusals: e.equivalent,
		RefusalDensity:     density,
		Outcome:            sample.Outcome,
	}

	// Hard cumulative-work axes win an exact-boundary tie. The semantic refusal
	// arm otherwise trips earlier when its small threshold and density are met.
	switch {
	case e.policy.MaxUncachedInputTokens > 0 && e.totals.UncachedInputTokens >= e.policy.MaxUncachedInputTokens:
		decision.Reason = ReasonEnvelopeUncachedInput
	case e.policy.MaxWallTimeNanos > 0 && e.totals.WallTimeNanos >= e.policy.MaxWallTimeNanos:
		decision.Reason = ReasonEnvelopeWallTime
	case e.refusalEnvelopeExceeded(density):
		decision.Reason = ReasonEnvelopeSemanticRefusal
	default:
		return decision
	}

	decision.Action = EnvelopeActionCheckpointRecovery
	decision.Recovery = SessionRecoveryCheckpoint{
		Reason:         decision.Reason,
		TraceID:        state.TraceID,
		Goal:           state.Goal,
		PendingTurn:    state.PendingTurn,
		ContinuationID: state.ContinuationID,
		Generation:     state.Generation,
		StateRev:       state.Rev,
	}
	e.tripped = true
	e.decision = decision
	return decision
}

func (e *CumulativeEnvelope) foldTotals(sample CumulativeEnvelopeSample) {
	input := max64(sample.InputTokens, 0)
	cached := max64(sample.CachedInputTokens, 0)
	if cached > input {
		cached = input
	}
	e.totals.InputTokens += input
	e.totals.CachedInputTokens += cached
	e.totals.UncachedInputTokens += input - cached
	e.totals.OutputTokens += max64(sample.OutputTokens, 0)
	e.totals.ModelCalls += max(sample.ModelCalls, 0)
	e.totals.ToolCalls += max(sample.ToolCalls, 0)
	e.totals.ManualContinuations += max(sample.ManualContinuations, 0)
	e.totals.WallTimeNanos += max64(sample.WallTimeNanos, 0)

	if toolOutcomePresent(sample.Outcome) && sample.Outcome.BadForSessionControl() {
		e.totals.Refusals++
		if sample.ToolCalls <= 0 {
			e.totals.ToolCalls++
		}
	}
}

func (e *CumulativeEnvelope) foldRefusal(outcome ToolCallOutcome) {
	if !toolOutcomePresent(outcome) {
		return
	}
	if !outcome.BadForSessionControl() {
		// An unrelated allowed/no-op call is not evidence that the refused effect
		// succeeded. Only the existing explicit Progress witness clears a semantic
		// refusal run; otherwise refusal density changes but the equivalent count
		// remains available to catch command mutation around the same deny.
		if outcome.Progress {
			e.equivalent = 0
			e.lastRefusalKey = semanticRefusalKey{}
		}
		return
	}
	key := semanticKey(outcome)
	if e.equivalent > 0 && key == e.lastRefusalKey {
		e.equivalent++
		return
	}
	e.lastRefusalKey = key
	e.equivalent = 1
}

func (e *CumulativeEnvelope) refusalEnvelopeExceeded(density float64) bool {
	if e.policy.MaxEquivalentRefusals <= 0 || e.equivalent < e.policy.MaxEquivalentRefusals {
		return false
	}
	return e.policy.MinRefusalDensity <= 0 || density >= e.policy.MinRefusalDensity
}

type semanticRefusalKey struct {
	reason      abi.ReasonCode
	target      string
	effect      string
	tool        string
	disposition ToolCallDisposition
}

func semanticKey(outcome ToolCallOutcome) semanticRefusalKey {
	target := strings.TrimSpace(outcome.Target)
	effect := strings.TrimSpace(outcome.IntendedEffect)
	key := semanticRefusalKey{reason: outcome.Reason, target: target, effect: effect}
	if target == "" && effect == "" {
		// Backward-compatible identity for callers that have not yet supplied the
		// semantic coordinates. ToolCallID/command bytes are deliberately absent.
		key.tool = outcome.Tool
		key.disposition = outcome.Disposition
	}
	return key
}

func toolOutcomePresent(outcome ToolCallOutcome) bool {
	return outcome.Tool != "" || outcome.ToolCallID != "" || outcome.Kind != ToolCallOutcomeAllowed ||
		outcome.Reason != abi.ReasonNone || outcome.Disposition != ToolDispositionNone || outcome.Progress ||
		outcome.Target != "" || outcome.IntendedEffect != ""
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
