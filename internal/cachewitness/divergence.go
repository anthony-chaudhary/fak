package cachewitness

import (
	"fmt"
	"sort"
)

// The WITNESSED-vs-OBSERVED reuse divergence miner (#3635, epic #3569). cachewitness
// deliberately keeps the WITNESSED in-kernel kv_prefix reuse and the OBSERVED provider
// prompt-cache in SEPARATE Record fields and NEVER derives one from the other. This miner
// adds the missing cross-check: per record it COMPARES the two reuse axes and raises a
// TRUST_CLASS_DIVERGENCE when they disagree beyond a tolerance — the signal that attribution
// has leaked across the trust classes (fak's own number and the provider's relayed number
// telling different stories about the same session).
//
// It only ever COMPARES the two ratios; it NEVER sums or blends them (explicitly out of
// scope for #3635, and forbidden by the #1066 honesty fence). Neither metric is redefined:
// the WITNESSED ratio is KVPrefix.ReuseRatio() as-is, the OBSERVED ratio is the provider
// cache_read over the same prompt denominator.

// DefaultReuseDivergenceTolerance is the absolute reuse-ratio gap (both ratios are in [0,1])
// beyond which the two trust classes are treated as disagreeing. It is a dead-band so
// ordinary provider/kernel accounting wobble does not raise a phantom leak; 0.15 mirrors the
// 15-point per-session hit-rate band the sibling cache-efficiency checks use.
const DefaultReuseDivergenceTolerance = 0.15

// ReuseDivergence is one record's WITNESSED-vs-OBSERVED reuse comparison. The two ratios are
// held side by side and their absolute difference is the divergence — they are NEVER summed.
type ReuseDivergence struct {
	Label      string `json:"label,omitempty"`
	GatewayURL string `json:"gateway_url,omitempty"`

	PromptTokens uint64 `json:"prompt_tokens"`
	// WitnessedReusedTokens is fak's OWN in-kernel kv_prefix reuse (WITNESSED).
	WitnessedReusedTokens uint64 `json:"witnessed_reused_tokens"`
	// ObservedCacheReadTokens is the provider's relayed cache_read (OBSERVED).
	ObservedCacheReadTokens uint64 `json:"observed_cache_read_tokens"`

	WitnessedReuseRatio float64 `json:"witnessed_reuse_ratio"`
	ObservedReuseRatio  float64 `json:"observed_reuse_ratio"`
	// AbsDivergence is |witnessed - observed|. It is a COMPARISON of the two axes, never a
	// sum of them.
	AbsDivergence float64 `json:"abs_divergence"`
	Diverged      bool    `json:"diverged"`
	Reason        string  `json:"reason,omitempty"`
}

// ReuseDivergenceReport is the miner's verdict over a set of records (#3635). OK is false
// ONLY when at least one record diverges — a trust-class leak is an alarm, not a thin-corpus
// non-event, so an all-single-class corpus falls open INSUFFICIENT (OK true).
type ReuseDivergenceReport struct {
	Tolerance float64 `json:"tolerance"`

	// Compared is the number of records that carried BOTH a witnessed prompt denominator and
	// a present OBSERVED provider axis, so a cross-class comparison was possible. SingleClass
	// is the number that had only the WITNESSED axis (the pure in-kernel path, provider
	// cache_read == 0): with no OBSERVED axis there is nothing to diverge FROM, so those are
	// reported, not flagged. Detecting "fak witnessed reuse the provider never corroborated"
	// (a zero OBSERVED axis) is a DIFFERENT signal (the upgrade-refused / cold-provider
	// miner), out of this miner's cross-ratio scope.
	Compared    int `json:"compared"`
	SingleClass int `json:"single_class"`

	Diverged []ReuseDivergence `json:"diverged,omitempty"`

	Verdict    string `json:"verdict"` // OK | TRUST_CLASS_DIVERGENCE | INSUFFICIENT
	OK         bool   `json:"ok"`
	Finding    string `json:"finding"`
	NextAction string `json:"next_action,omitempty"`
}

// reuseDivergenceOf computes one record's cross-class comparison. comparable is false when
// the record lacks a prompt denominator or lacks the OBSERVED provider axis (provider
// cache_read == 0, the pure in-kernel path) — there is then no second axis to compare.
func reuseDivergenceOf(r Record, tolerance float64) (ReuseDivergence, bool) {
	prompt := r.KVPrefix.PromptTokens
	if prompt == 0 || r.ProviderCacheReadTokens == 0 {
		return ReuseDivergence{}, false
	}
	witnessed := r.KVPrefix.ReuseRatio()
	observed := float64(r.ProviderCacheReadTokens) / float64(prompt)
	// Clamp the OBSERVED ratio: a provider may relay more cache_read than this window's
	// prompt denominator (cumulative vs window skew), and a ratio > 1 would inflate the gap.
	if observed > 1 {
		observed = 1
	}
	d := witnessed - observed
	if d < 0 {
		d = -d
	}
	div := ReuseDivergence{
		Label:                   r.GatewayURL,
		GatewayURL:              r.GatewayURL,
		PromptTokens:            prompt,
		WitnessedReusedTokens:   r.KVPrefix.ReusedTokens,
		ObservedCacheReadTokens: r.ProviderCacheReadTokens,
		WitnessedReuseRatio:     witnessed,
		ObservedReuseRatio:      observed,
		AbsDivergence:           d,
	}
	if d > tolerance {
		div.Diverged = true
		div.Reason = fmt.Sprintf("TRUST_CLASS_DIVERGENCE — WITNESSED reuse %.3f vs OBSERVED provider cache %.3f differ by %.3f (> tolerance %.3f); attribution has leaked across the trust classes",
			witnessed, observed, d, tolerance)
	}
	return div, true
}

// FoldReuseDivergence mines a set of witness records for WITNESSED-vs-OBSERVED reuse
// divergence (#3635). It is PURE and deterministic. A tolerance <= 0 uses
// DefaultReuseDivergenceTolerance. Records lacking a prompt denominator or an OBSERVED
// provider axis are counted as SingleClass (not comparable, never flagged). Flagged records
// are returned worst-first (largest divergence). The two axes are only ever compared, never
// summed.
func FoldReuseDivergence(records []Record, tolerance float64) ReuseDivergenceReport {
	if tolerance <= 0 {
		tolerance = DefaultReuseDivergenceTolerance
	}
	rep := ReuseDivergenceReport{Tolerance: tolerance, Verdict: "INSUFFICIENT", OK: true}
	for _, r := range records {
		div, comparable := reuseDivergenceOf(r, tolerance)
		if !comparable {
			rep.SingleClass++
			continue
		}
		rep.Compared++
		if div.Diverged {
			rep.Diverged = append(rep.Diverged, div)
		}
	}
	sort.SliceStable(rep.Diverged, func(i, j int) bool {
		a, b := rep.Diverged[i], rep.Diverged[j]
		if a.AbsDivergence != b.AbsDivergence {
			return a.AbsDivergence > b.AbsDivergence
		}
		return a.GatewayURL < b.GatewayURL
	})

	switch {
	case rep.Compared == 0:
		rep.Finding = fmt.Sprintf("INSUFFICIENT — no record carried both a WITNESSED and an OBSERVED reuse axis to compare (%d single-class record(s))", rep.SingleClass)
	case len(rep.Diverged) > 0:
		rep.Verdict = "TRUST_CLASS_DIVERGENCE"
		rep.OK = false
		rep.Finding = fmt.Sprintf("TRUST_CLASS_DIVERGENCE — %d of %d compared record(s) disagree beyond %.3f between WITNESSED kernel reuse and OBSERVED provider cache",
			len(rep.Diverged), rep.Compared, tolerance)
		rep.NextAction = "inspect each flagged session: the WITNESSED kv_prefix reuse and the OBSERVED provider cache_read tell different stories, so one trust class's number is not corroborated by the other — never reconcile them by summing (#1066/#3635)"
	default:
		rep.Verdict = "OK"
		rep.Finding = fmt.Sprintf("OK — all %d compared record(s) agree within %.3f between WITNESSED and OBSERVED reuse", rep.Compared, tolerance)
	}
	return rep
}
