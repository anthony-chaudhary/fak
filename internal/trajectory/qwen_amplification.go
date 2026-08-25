package trajectory

import "strings"

// QwenAmplificationAction is the typed outcome of evaluating one campaign.
type QwenAmplificationAction string

const (
	QwenAmplificationObserve QwenAmplificationAction = "observe"
	QwenAmplificationAlert   QwenAmplificationAction = "alert"
	QwenAmplificationHold    QwenAmplificationAction = "hold"
)

// QwenAmplificationNonEligibleReason explains why a campaign was not evaluated.
type QwenAmplificationNonEligibleReason string

const (
	QwenAmplificationMissingUsage  QwenAmplificationNonEligibleReason = "missing_usage"
	QwenAmplificationInvalidEngine QwenAmplificationNonEligibleReason = "invalid_engine"
)

// QwenCanonicalUsage contains the canonical campaign totals supplied by the caller.
type QwenCanonicalUsage struct {
	InputTokens     uint64
	CacheReadTokens uint64
	OutputTokens    uint64
	UsefulWitnesses uint64
}

// QwenAmplificationPolicy defines campaign limits. Enforce is intentionally false by default.
// A zero limit disables that individual limit.
type QwenAmplificationPolicy struct {
	InputTokenBudget     uint64
	CacheReadTokenBudget uint64
	OutputTokenBudget    uint64
	MinimumWitnesses     uint64
	Enforce              bool
}

// QwenAmplificationOverrideReceipt records an explicit operator override.
type QwenAmplificationOverrideReceipt struct {
	Reason string
}

// QwenAmplificationDecision is a deterministic, side-effect-free policy result.
type QwenAmplificationDecision struct {
	Action            QwenAmplificationAction
	Eligible          bool
	NonEligibleReason QwenAmplificationNonEligibleReason
	Engine            string
	Usage             *QwenCanonicalUsage
	Override          *QwenAmplificationOverrideReceipt
}

// Decide evaluates canonical totals without deriving or filling in missing usage.
func (p QwenAmplificationPolicy) Decide(engine string, usage *QwenCanonicalUsage, overrideReason string) QwenAmplificationDecision {
	decision := QwenAmplificationDecision{Action: QwenAmplificationObserve, Engine: engine, Usage: usage}
	if usage == nil {
		decision.NonEligibleReason = QwenAmplificationMissingUsage
		return decision
	}
	if p.Enforce && !strings.HasPrefix(engine, "fak-native/") {
		decision.NonEligibleReason = QwenAmplificationInvalidEngine
		return decision
	}
	decision.Eligible = true

	breached := (p.InputTokenBudget != 0 && usage.InputTokens > p.InputTokenBudget) ||
		(p.CacheReadTokenBudget != 0 && usage.CacheReadTokens > p.CacheReadTokenBudget) ||
		(p.OutputTokenBudget != 0 && usage.OutputTokens > p.OutputTokenBudget) ||
		(p.MinimumWitnesses != 0 && usage.UsefulWitnesses < p.MinimumWitnesses)
	if !breached {
		return decision
	}
	decision.Action = QwenAmplificationAlert
	if overrideReason != "" {
		decision.Override = &QwenAmplificationOverrideReceipt{Reason: overrideReason}
		return decision
	}
	if p.Enforce {
		decision.Action = QwenAmplificationHold
	}
	return decision
}
