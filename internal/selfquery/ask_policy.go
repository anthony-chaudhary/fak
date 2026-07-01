// This file is the general policy #1580 asks for: a level up from any single
// ask-vs-assume decision point, it specifies the ONE rule every such decision point
// in this codebase should be checkable against, rather than each one growing its own
// bespoke threshold. Three call sites already make an ask-vs-assume judgment call
// today, each with its own local reasoning:
//
//   - ctxplan.AssessAssumptions (#1591/managed-context) — stale/unknown/low-confidence
//     assumptions become AssumptionQuery/AssumptionRefresh actions before an effect.
//   - selfquery.PromotionCandidate.IsAmbiguous (#1596) — a context->memory promotion
//     whose consent or corroboration verdict came back genuinely unresolved.
//   - the ctxmmu durability gate (docs/CONTEXT-IS-NOT-MEMORY.md, #82) — an
//     un-classified observation defaults to the shortest-lived (safest) class rather
//     than being promoted on a guess.
//
// All three converge on the same shape without saying so out loud: weigh how
// confident the system is against how expensive being WRONG would be, and ask
// (instead of silently assuming) whenever the expected cost of a silent wrong guess
// is not clearly dominated by the cost of interrupting the user. AskPolicy makes that
// convergence explicit as one small, pure decision table — Stakes x Reversibility x
// Confidence -> ShouldAsk bool — so a new call site can be checked against the same
// rule instead of inventing a fourth bespoke threshold, and so existing call sites can
// cite it as the general contract their local logic already approximates.
//
// This is deliberately NOT a replacement for #1596's PromotionClarification or
// ctxplan's AssessAssumptions — those remain the concrete, domain-shaped deciders (and
// PromotionCandidate.IsAmbiguous is used below as a conformance example, not
// duplicated). AskPolicy is the policy layer one level above them: the general rule
// those concrete deciders' thresholds are instances of.
package selfquery

// Stakes is how much is at risk if fak silently assumes wrong. The three-step scale
// mirrors ctxmmu's admit/transform/quarantine posture: most decisions are Low stakes
// (a read, a cosmetic default); Medium stakes touch state a user would notice being
// wrong (a promoted memory, a chosen default); High stakes touch effects that are hard
// to walk back or are visible outside the session (a durable belief acted on for
// months, an external side effect).
type Stakes string

const (
	StakesLow    Stakes = "low"
	StakesMedium Stakes = "medium"
	StakesHigh   Stakes = "high"
)

// Reversibility is how cheaply a wrong assumption can be corrected after the fact.
// This is the axis CONTEXT-IS-NOT-MEMORY.md names explicitly: a false-negative
// (failing to act, or asking when asking wasn't needed) is recoverable — the user
// re-states it, or answers a redundant question — while a false-positive silent wrong
// guess that is hard to reverse is the expensive direction (the "stale-as-current"
// failure: a wrong belief nobody knows to correct).
type Reversibility string

const (
	// ReversibleCheap means a wrong assumption self-corrects at near-zero cost: the
	// next turn overwrites it, or the effect is read-only and leaves no residue.
	ReversibleCheap Reversibility = "cheap"
	// ReversibleCostly means a wrong assumption persists and must be actively
	// discovered and undone: a durable memory promotion, a value that propagates into
	// later decisions before anyone notices it was wrong.
	ReversibleCostly Reversibility = "costly"
	// Irreversible means a wrong assumption cannot be undone at all once acted on: an
	// external side effect, a deletion, a message sent, a commit pushed.
	Irreversible Reversibility = "irreversible"
)

// AskInput is the closed set of factors AskPolicy weighs. Confidence is normalized to
// [0,1], the same convention ctxplan.Assumption.Confidence already uses, so a caller
// holding a ctxplan.AssumptionAssessment can pass its Confidence straight through
// without translation.
type AskInput struct {
	Confidence    float64
	Stakes        Stakes
	Reversibility Reversibility
}

// AskVerdict is the answer AskPolicy returns: whether to ask, why, and the confidence
// bar that was actually applied (so a caller can explain the decision, the same
// posture ctxplan.AssumptionAssessment.Reason and PromotionClarification's fixed-
// template text already take — no free-text model judgment, a rendered fact about
// the inputs).
type AskVerdict struct {
	ShouldAsk bool    `json:"should_ask"`
	Threshold float64 `json:"threshold"`
	Reason    string  `json:"reason"`
}

// ShouldAsk is the general decision function #1580 asks for: given how confident fak
// is and how expensive a silent wrong guess would be, decide whether to ask the user
// instead of assuming. It is pure, deterministic, and host-agnostic — no I/O, no model
// call — the same posture as ctxplan.AssessAssumptions and PromotionClarification.
//
// The rule: an Irreversible or High-stakes decision must clear a high confidence bar
// (0.90) before fak may assume; a Costly-reversibility or Medium-stakes decision must
// clear a moderate bar (matching ctxplan's own DefaultAssumptionPolicy().MinConfidence,
// 0.65 — the two policies are meant to agree, not drift); a Cheap-reversible, Low-stakes
// decision only needs to clear a low bar (0.35), because the cost of a wrong guess
// there is bounded and self-correcting. Below its applicable bar, fak asks rather than
// silently assumes — the same "un-classified means the safe class" default the
// durability gate and every fail-closed gate in this repo already take.
func ShouldAsk(in AskInput) AskVerdict {
	threshold := askThreshold(in.Stakes, in.Reversibility)
	confidence := clampConfidence(in.Confidence)
	if confidence < threshold {
		return AskVerdict{
			ShouldAsk: true,
			Threshold: threshold,
			Reason:    "confidence below the threshold for this stakes/reversibility class; ask instead of assuming",
		}
	}
	return AskVerdict{
		ShouldAsk: false,
		Threshold: threshold,
		Reason:    "confidence clears the threshold for this stakes/reversibility class; safe to assume",
	}
}

// askThreshold picks the confidence bar for a Stakes x Reversibility pair. The more
// severe axis wins: Irreversible or High stakes always demands the high bar even if
// the other axis is mild, because a single irreversible or high-stakes mistake is not
// offset by the decision usually being cheap.
func askThreshold(stakes Stakes, rev Reversibility) float64 {
	if rev == Irreversible || stakes == StakesHigh {
		return 0.90
	}
	if rev == ReversibleCostly || stakes == StakesMedium {
		return 0.65
	}
	return 0.35
}

func clampConfidence(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
