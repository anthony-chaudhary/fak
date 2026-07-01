package vcachechain

// proof.go is the deterministic, no-model proof surface for M4 — the `fak vcache
// prove-recall` witness. Like the M5 Governor's ProveStarSavings, it takes the
// caller's calibration (prefix length, unit length, read multiplier, sibling count)
// and returns a self-describing proof that the §11.0 cost gate produced the correct
// decision, with every number the gate used so a human or CI line can audit it.
//
// Default inputs reproduce the design note's §11.0 headline: a 30 000-token prefix
// replayed at r=0.1 to recall one 10-token unit is a 3000→10 (300×) LOSS, so the
// gate REFUSES it (ProofRefuted). Raising the sibling count past the break-even
// (301) flips it to ProofProven — the amortized-fan-out exception that is the only
// regime where chain rebuild wins.

// ProofStatus is the two-valued proof verdict.
type ProofStatus string

const (
	RecallProofSchema = "fak.vcache.prove-recall.v1"

	// ProofProven — the cost gate CLEARED: amortized fan-out makes one chain replay
	// cheaper than the S fresh prefills it saves (P·r < S·U). Exit 0.
	ProofProven ProofStatus = "proven"
	// ProofRefuted — the cost gate REFUSED: replay costs at least the fresh prefills
	// it would save (P·r ≥ S·U). This is the correct decision for almost every
	// single-unit recall (the §11.0 300× loss). Exit 1.
	ProofRefuted ProofStatus = "refuted"
)

// RecallProof is the self-describing output of ProveRecall.
type RecallProof struct {
	Schema               string         `json:"schema"`
	Status               ProofStatus    `json:"status"`
	Decision             RecallDecision `json:"decision"`
	Reason               string         `json:"reason"`
	PrefixTokens         int64          `json:"prefix_tokens"`
	UnitTokens           int64          `json:"unit_tokens"`
	ReadMult             float64        `json:"read_mult"`
	Siblings             int            `json:"siblings"`
	ReplayCost           float64        `json:"replay_cost"`            // P·r
	FreshPrefillCost     float64        `json:"fresh_prefill_cost"`     // U
	AmortizedSavings     float64        `json:"amortized_savings"`      // S·U
	BreakEvenSiblings    int            `json:"break_even_siblings"`    // smallest S that clears the gate
	LossRatio            float64        `json:"loss_ratio"`             // ReplayCost / FreshPrefillCost (single-unit pain)
	CorrectnessDependsOn bool           `json:"correctness_depends_on"` // always false (Law A2)
}

// ProveRecallInput is the caller-supplied calibration for the proof.
type ProveRecallInput struct {
	PrefixTokens int64   // P — the replayed warm prefix length
	UnitTokens   int64   // U — the recalled unit's fresh-prefill length
	ReadMult     float64 // r — cached-read multiplier (M1 calibration)
	Siblings     int     // S — co-recalled sibling units (amortization)
}

// ProveRecall runs the §11.0 cost gate over the given inputs and returns a proof
// showing its work. It is pure and deterministic — no DAG, no network, no clock;
// it exercises the SAME RebuildNetPositive / BreakEvenSiblings PlanRecall uses, so
// a green proof is direct evidence the gate behaves as specified.
func ProveRecall(in ProveRecallInput) RecallProof {
	siblings := floorAtOne(in.Siblings)
	replay := ReplayCost(in.PrefixTokens, in.ReadMult)
	fresh := float64(in.UnitTokens)
	if fresh < 0 {
		fresh = 0
	}
	savings := float64(siblings) * fresh
	net := RebuildNetPositive(in.PrefixTokens, in.UnitTokens, siblings, in.ReadMult)
	p := RecallProof{
		Schema:               RecallProofSchema,
		PrefixTokens:         in.PrefixTokens,
		UnitTokens:           in.UnitTokens,
		ReadMult:             in.ReadMult,
		Siblings:             siblings,
		ReplayCost:           replay,
		FreshPrefillCost:     fresh,
		AmortizedSavings:     savings,
		BreakEvenSiblings:    breakEvenSiblingCount(in.PrefixTokens, in.UnitTokens, in.ReadMult),
		CorrectnessDependsOn: false, // Law A2: a rebuild is never a license to elide resent context
	}
	if fresh > 0 {
		p.LossRatio = replay / fresh
	}
	if net {
		p.Status = ProofProven
		p.Decision = DecisionRebuild
		p.Reason = "cost gate cleared: P·r < S·U — amortized fan-out makes rebuild net-positive"
	} else {
		p.Status = ProofRefuted
		p.Decision = DecisionColdPrefill
		p.Reason = "cost gate refused: P·r >= S·U — recall-by-rebuild is net-negative (§11.0); send the unit cold"
	}
	return p
}
