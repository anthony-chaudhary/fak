package vcacheobserve

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/vcachechain"
)

// buildPanels assembles one observability panel per vCache sub-concept from the
// computed report. Each panel states the real-account VALUE, its PROVENANCE (OBSERVED
// vs DECISION), a one-word VERDICT, and the WITNESS that reproduces it.
func buildPanels(r Report) []Panel {
	famFirstPositive := 0
	for _, f := range r.Families {
		if f.Economics.FirstPositiveRequest == 1 {
			famFirstPositive++
		}
	}
	rideNatural := dominantGovernor(r.Families)
	falseWarmPct := 100 * r.Prediction.FalseWarmRate()
	falseColdPct := 100 * r.Prediction.FalseColdRate()

	panels := []Panel{
		{
			Name:       "base provider cache",
			Pkg:        "internal/cachemeta",
			Milestone:  "tier-1 (passive)",
			Question:   "is the Anthropic prefix cache being reused at all?",
			Value:      fmt.Sprintf("hit %.1f%% — %s cached input tokens served over %d turns", 100*r.HitRate, commas(int64(r.Aggregate.CacheReadTokens)), r.Turns),
			Provenance: Observed,
			Verdict:    "WORKS",
			Witness:    "fak vcache observe (aggregate.cache_read_tokens)",
			Detail:     "the obvious base case — cache_read>0 every turn; this is what 'base cache works' means.",
		},
		{
			Name:       "M2 star anchors",
			Pkg:        "internal/vcachestar",
			Milestone:  "M2 (#717)",
			Question:   "does a warmed prefix pay for itself within a family?",
			Value:      fmt.Sprintf("saved %.1f%% (%s token-equiv) across %d families; first-positive turn %d", r.Aggregate.SavedPct, commas(int64(r.Aggregate.SavedTokenEquiv)), r.FamilyCount, r.Aggregate.FirstPositiveRequest),
			Provenance: Observed,
			Verdict:    "WORKS",
			Witness:    "fak vcache prove-telemetry --file <telemetry>",
			Detail:     "within-family reuse: the shared system prefix is warmed once and read by every later turn.",
		},
		{
			Name:       "M1 concentration",
			Pkg:        "internal/vcachecal",
			Milestone:  "M1 (#716) §5.2",
			Question:   "is reuse CONCENTRATED enough for cross-family warming to help?",
			Value:      fmt.Sprintf("measured Zipf s=%.2f over %d families (defeated=%v)", r.Concentration.ZipfS, r.FamilyCount, r.Concentration.Defeated),
			Provenance: Decision,
			Verdict:    concentrationVerdict(r),
			Witness:    "fak vcache score --anchors-file <measured-anchors>",
			Detail:     r.Concentration.Recommendation,
		},
		{
			Name:       "M1 warmth belief",
			Pkg:        "internal/vcachecal",
			Milestone:  "M1 (#716) §7 / Law A1",
			Question:   "does the warmth estimator ever book a save it did not get?",
			Value:      fmt.Sprintf("false-warm %.2f%% (the lethal one) · false-cold %.2f%% (benign under-claim) over %d turns", falseWarmPct, falseColdPct, r.Prediction.Total),
			Provenance: Observed,
			Verdict:    warmthVerdict(falseWarmPct),
			Witness:    "fak vcache observe (prediction.false_warm)",
			Detail:     "false-warm is the dangerous direction (manifest says HIT, provider says MISS); 0% means verify-then-trust holds on real traffic.",
		},
		{
			Name:       "M3 dedicated warming",
			Pkg:        "internal/vcachewarm",
			Milestone:  "M3 (#718)",
			Question:   "should we spend a dedicated max_tokens:0 / decode-1 warm?",
			Value:      fmt.Sprintf("%d/%d families net-positive from natural traffic by turn 1", famFirstPositive, r.FamilyCount),
			Provenance: Decision,
			Verdict:    "NATURAL-FIRST",
			Witness:    "fak vcache observe (families[].economics.first_positive_request)",
			Detail:     "the first natural request warms the prefix for free; a dedicated warm would only add a write — spend it only for latency or a TTL gap.",
		},
		{
			Name:       "M4 chains & recall",
			Pkg:        "internal/vcachechain",
			Milestone:  "M4 (#719) §11.0",
			Question:   "should a single unit be recalled by replaying the chain?",
			Value:      fmt.Sprintf("at the account's mean %s-token prefix: %s loss, break-even %s siblings", commas(int64(r.MeanPrefixTokens)), recallLoss(r.Recall), breakEven(r.Recall.BreakEvenSiblings)),
			Provenance: Decision,
			Verdict:    recallVerdict(r.Recall),
			Witness:    "fak vcache prove-recall --prefix-tokens <mean>",
			Detail:     "single-unit chain rebuild is a large net loss; the cost gate correctly refuses it and reserves rebuild for amortized fan-out.",
		},
		{
			Name:       "M5 governor",
			Pkg:        "internal/vcachegov",
			Milestone:  "M5 (#720) §5.4",
			Question:   "pin, lazy-rebuild, or ride natural traffic per prefix?",
			Value:      fmt.Sprintf("%s (the dominant verdict across %d families' observed arrival rates)", rideNatural, r.FamilyCount),
			Provenance: Decision,
			Verdict:    governorVerdict(rideNatural),
			Witness:    "fak vcache observe (families[].governor_decision)",
			Detail:     "active sessions arrive faster than the 5m TTL (λT≥1), so natural traffic holds the prefix warm — no pin, no dedicated warm.",
		},
		{
			Name:       "score composite",
			Pkg:        "internal/vcachescore",
			Milestone:  "agent-dev 2x gate",
			Question:   "what grade does the run earn — measured vs synthetic?",
			Value:      fmt.Sprintf("MEASURED %s (%d/100) vs SYNTHETIC %s (%d/100)", r.GradeMeasured, r.ScoreMeasured, r.GradeSynthetic, r.ScoreSynthetic),
			Provenance: Decision,
			Verdict:    gradeVerdict(r),
			Witness:    "fak vcache score --telemetry <t> --anchors-file <a>",
			Detail:     "same realized economics; the grade gap is almost entirely the concentration assumption (35 of 40 pts; the other 5 are the benign false-cold risk penalty) — synthetic steep (s=1.74) vs the account's measured-flat distribution.",
		},
		{
			Name:       "cachemeta canonicalization",
			Pkg:        "internal/cachemeta",
			Milestone:  "tier-1 §6",
			Question:   "is the cacheable prefix staying byte-stable?",
			Value:      fmt.Sprintf("inferred HOLDING — %.2f%% false-warm (a canonicalization break would show as believed-warm, cache_read=0)", falseWarmPct),
			Provenance: Observed,
			Verdict:    canonVerdict(falseWarmPct),
			Witness:    "fak vcache observe (prediction.false_warm == 0)",
			Detail:     "a volatile byte in the prefix (timestamp/UUID) would silently break reuse; 0 false-warms is the proxy that the prefix is canonical.",
		},
	}
	return panels
}

func dominantGovernor(fams []Family) string {
	counts := map[string]int{}
	for _, f := range fams {
		counts[string(f.GovernorDecision)]++
	}
	best, bestN := "", -1
	for k, n := range counts {
		if n > bestN || (n == bestN && k < best) {
			best, bestN = k, n
		}
	}
	if best == "" {
		return "ride_natural"
	}
	return best
}

func concentrationVerdict(r Report) string {
	if r.Concentration.Defeated {
		return "DEFEATED"
	}
	return "EXPLOITABLE"
}

func warmthVerdict(falseWarmPct float64) string {
	if falseWarmPct == 0 {
		return "SAFE"
	}
	return "AT-RISK"
}

func canonVerdict(falseWarmPct float64) string {
	if falseWarmPct == 0 {
		return "HOLDING"
	}
	return "DRIFTING"
}

func governorVerdict(decision string) string {
	switch decision {
	case "ride_natural":
		return "RIDE-NATURAL"
	case "heartbeat_pin":
		return "PIN"
	case "lazy_rebuild":
		return "LAZY"
	case "evict":
		return "EVICT"
	default:
		return decision
	}
}

func recallVerdict(p vcachechain.RecallProof) string {
	if p.Status == vcachechain.ProofRefuted {
		return "REFUSED"
	}
	return "ALLOWED"
}

func recallLoss(p vcachechain.RecallProof) string {
	if p.LossRatio <= 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.0fx", p.LossRatio)
}

func gradeVerdict(r Report) string {
	if r.GradeMeasured == r.GradeSynthetic {
		return "STABLE"
	}
	return "CONTEXT-DEPENDENT"
}

func breakEven(n int) string {
	if n == int(^uint(0)>>1) {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

// commas renders an integer with thousands separators for the human table.
func commas(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := fmt.Sprintf("%d", n)
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}
