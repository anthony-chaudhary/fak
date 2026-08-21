package gateway

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// writeVCacheMetrics emits the fak_vcache_* family: the NET realized provider-cache
// economics (read rebate MINUS write premium) over the session's cumulative cache
// counters, computed by the SAME engine `fak vcache observe` uses
// (vcachegov.ProveTelemetrySavings over one aggregate row), so the live gauge equals
// the offline observe Aggregate on the same totals. It is the write-premium-aware
// companion to fak_gateway_inference_cached_prompt_tokens_total (the read axis): a
// cold-write-dominated session reads NEGATIVE saved / proven=0 until the reads repay
// the writes — the honest break-even the read-only surface cannot show. Every value is
// OBSERVED (provider-relayed counters); a hit is a realized rebate, never local trust.
// Until a turn carries provider cache activity it emits nothing (no phantom series).
func (m *gatewayMetrics) writeVCacheMetrics(b *strings.Builder) {
	snap := m.inferenceSnapshotData()
	if snap.cachedTok == 0 && snap.cacheCreateTok == 0 {
		return
	}
	proof := vcacheProofFromCounters(snap.promptTok, snap.cachedTok, snap.cacheCreateTok)
	writeCounter(b, "fak_vcache_cache_creation_tokens_total", "OBSERVED (provider-relayed): cumulative cache_creation_input_tokens — the provider-cache WRITE axis, companion to fak_gateway_inference_cached_prompt_tokens_total (the READ axis). Net saving = read rebate minus this write premium.", int64(snap.cacheCreateTok))
	writeHelpType(b, "fak_vcache_baseline_token_equiv", "OBSERVED-derived: input-token-equivalents the session WOULD have cost with NO provider cache (every prompt token at 1x).", "gauge")
	fmt.Fprintf(b, "fak_vcache_baseline_token_equiv %s\n", promFloat(proof.BaselineTokenEquiv))
	writeHelpType(b, "fak_vcache_actual_token_equiv", "OBSERVED-derived: input-token-equivalents the session actually cost under provider caching (read at 0.1x, unsplit write at the 5m 1.25x tier).", "gauge")
	fmt.Fprintf(b, "fak_vcache_actual_token_equiv %s\n", promFloat(proof.ActualTokenEquiv))
	writeHelpType(b, "fak_vcache_saved_token_equiv", "NET realized provider-cache saving in input-token-equivalents (baseline minus actual). NEGATIVE on a cold-write-dominated session until reads repay writes. Same engine and number as `fak vcache observe`.", "gauge")
	fmt.Fprintf(b, "fak_vcache_saved_token_equiv %s\n", promFloat(proof.SavedTokenEquiv))
	writeHelpType(b, "fak_vcache_saved_ratio", "NET saved-token-equiv as a fraction of the uncached baseline (saved_pct/100).", "gauge")
	fmt.Fprintf(b, "fak_vcache_saved_ratio %s\n", promFloat(proof.SavedPct/100))
	hit := 0.0
	if proof.BaselineTokenEquiv > 0 {
		hit = proof.CacheReadTokens / proof.BaselineTokenEquiv
	}
	writeHelpType(b, "fak_vcache_hit_rate", "OBSERVED: cache_read share of the uncached baseline (equals `fak vcache observe` Report.HitRate).", "gauge")
	fmt.Fprintf(b, "fak_vcache_hit_rate %s\n", promFloat(hit))
	mult := 0.0
	if proof.ActualTokenEquiv > 0 {
		mult = proof.BaselineTokenEquiv / proof.ActualTokenEquiv
	}
	writeHelpType(b, "fak_vcache_multiplier", "OBSERVED-derived: baseline/actual token-equiv (equals `fak vcache observe` Report.Multiplier). 1.0 = no net saving.", "gauge")
	fmt.Fprintf(b, "fak_vcache_multiplier %s\n", promFloat(mult))
	proven := 0
	if proof.Status == vcachegov.ProofProven {
		proven = 1
	}
	writeHelpType(b, "fak_vcache_proven", "1 when the session's observed cache reads repaid the write premium (NET positive); else 0 (cold/write-dominated). The honest break-even gate.", "gauge")
	fmt.Fprintf(b, "fak_vcache_proven %d\n", proven)
}

// writeVCacheWarmthMetrics emits the M1 warmth-belief prediction error over the live
// rolling provider-cache window. It is derived by the same vcacheobserve engine as
// `fak vcache observe`: predicted warm/cold is reconciled against the provider's real
// cache_read feedback, so false-warm is the live "believed warm but provider billed cold"
// signal. This is an OBSERVATION/DECISION surface only; it never gates correctness.
func (m *gatewayMetrics) writeVCacheWarmthMetrics(b *strings.Builder) {
	turns, _ := m.vcacheTurnsSnapshot()
	if len(turns) == 0 || !vcacheWindowHasCacheActivity(turns) {
		return
	}
	pred := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers()).Prediction
	if pred.Total == 0 {
		return
	}
	writeHelpType(b, "fak_vcache_warmth_prediction_outcomes", "OBSERVED provider-relayed DECISION (M1 warmth belief): rolling-window count of predicted-warm/cold outcomes reconciled against provider cache_read feedback. false_warm is the lethal believed-warm/provider-missed case; false_cold is benign missed opportunity. Derived by the same vcacheobserve engine as `fak vcache observe`.", "gauge")
	for _, row := range []struct {
		class string
		n     int
	}{
		{string(vcachecal.TrueWarm), pred.TrueWarm},
		{string(vcachecal.FalseWarm), pred.FalseWarm},
		{string(vcachecal.TrueCold), pred.TrueCold},
		{string(vcachecal.FalseCold), pred.FalseCold},
	} {
		fmt.Fprintf(b, "fak_vcache_warmth_prediction_outcomes{class=%q} %d\n", row.class, row.n)
	}
	writeHelpType(b, "fak_vcache_warmth_predictions_total", "OBSERVED provider-relayed DECISION (M1 warmth belief): total predictions reconciled in the rolling provider-cache window. Gauge because the retained window is bounded and rolling.", "gauge")
	fmt.Fprintf(b, "fak_vcache_warmth_predictions_total %d\n", pred.Total)
	writeHelpType(b, "fak_vcache_warmth_false_warm_rate", "OBSERVED provider-relayed DECISION (M1 warmth belief): false_warm / predicted_warm over the rolling provider-cache window; this is the Law A1 demote/alarm signal.", "gauge")
	fmt.Fprintf(b, "fak_vcache_warmth_false_warm_rate %s\n", promFloat(pred.FalseWarmRate()))
	writeHelpType(b, "fak_vcache_warmth_false_cold_rate", "OBSERVED provider-relayed DECISION (M1 warmth belief): false_cold / predicted_cold over the rolling provider-cache window; a missed warm opportunity, not a correctness risk.", "gauge")
	fmt.Fprintf(b, "fak_vcache_warmth_false_cold_rate %s\n", promFloat(pred.FalseColdRate()))
}

// writeVCacheGovernorMetrics emits the low-cardinality live Governor witness: how many
// active prefix families in the rolling provider-cache window classify into each M5
// steady-state decision. The per-family identities stay in /debug/vars; Prometheus gets
// only the closed decision set, derived through the same vcacheobserve.Observe engine as
// `fak vcache observe`. This is a DECISION surface only: it records what the Governor
// would do over observed traffic, but it does not warm, pin, evict, or suppress context.
func (m *gatewayMetrics) writeVCacheGovernorMetrics(b *strings.Builder) {
	turns, _ := m.vcacheTurnsSnapshot()
	counts := vcacheGovernorDecisionCounts(turns)
	if len(counts) == 0 {
		return
	}
	writeHelpType(b, "fak_vcache_governor_decision_families", "DECISION (fak policy): active provider-cache prefix families in the live rolling window, classified by the M5 Governor verdict. Derived by the same vcacheobserve engine as `fak vcache observe`; records the default-on decision witness only, not a warming side effect.", "gauge")
	for _, decision := range vcacheGovernorDecisionOrder {
		fmt.Fprintf(b, "fak_vcache_governor_decision_families{decision=%q} %d\n", decision, counts[decision])
	}
}
