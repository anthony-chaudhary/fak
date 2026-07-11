package rsiloop

// firenettune.go — SCORE each compaction fire by its net value and TUNE the fire/bail
// threshold against the scored corpus so negative-net fires are suppressed (#2817, part of
// epic #2783 workstream D: Score + RSI). It is the RSI-loop counterpart of forksaving.go:
// forksaving.go measures a per-FORK prefix-reuse saving; this file scores a per-FIRE
// compaction receipt and feeds the score back into the gate's own threshold.
//
// THE GATE THIS TUNES. internal/agent/anthropic_compact.go fires a head-anchored cache
// burst only when CacheBurstPaysBack approves it: the cached middle a compaction sheds
// forever (a per-turn read saving) must repay the one-time cold re-write of the invalidated
// suffix within the remaining session horizon. That gate decides on an EX-ANTE estimate —
// CacheBurstBreakEvenTurns prices the predicted break-even, and the gate fires when the
// predicted remaining horizon clears it. Its effective fire condition is
// `remainingTurns >= breakEven + margin`, with today's margin pinned at 0 (fire whenever the
// burst is predicted to pay back AT ALL). This file makes that margin the tunable knob.
//
// WHY A MARGIN, NOT THE RAW NET (the non-circular design, #2795 confusion-risk). The per-fire
// SCORE is the OBSERVED ex-post net — the shed read-rebate that actually accrued over the
// turns that actually followed the fire (B2), debited by the one-time suffix-burst premium
// the fire actually paid (A2). It is NOT the gate's own ex-ante 0.1x prediction: feeding the
// gate's prediction back onto the receipt would make the loop circular and tune nothing (a
// fire the gate predicted pays back would score non-negative BY CONSTRUCTION). The signal that
// makes tuning real is the GAP between the two — a fire the gate approved on a predicted
// horizon that the session did not deliver (it ended early) realizes a NEGATIVE net. Tuning
// hedges that estimation error by requiring extra predicted headroom (a positive margin)
// before firing, which bails exactly the thin-headroom fires whose realized net most often
// goes negative. The tuned margin is the fed-back fire/bail threshold.
//
// PURE, DETERMINISTIC, TESTABLE. Scoring prices on the ONE canonical cacheprice basis (#2798)
// — the same 0.1x/1.25x anchors forksaving.go and the Track-2 compaction report use — so a
// fire's net here and the report's per-fire value agree by construction. The tuning sweep is
// over the distinct predicted-margin values present in the corpus (no hyperparameter), so the
// KEEP reproduces bit-for-bit on any box (the determinism the RSI engine's regression gate
// requires). firenettune_test.go is the acceptance witness: over a fixed corpus spanning
// positive- and negative-net fires, mean per-fire net RISES after tuning and the known
// negative-net fire is bailed.
//
// GENERATION FRAME (gen/next). Promotion evidence (→ now): a corpus of REAL per-fire receipts
// off live guard sessions (the A4 receipt seam, once internal/cachevaluereport ships it) whose
// tuned margin lifts fleet-median per-fire net and stays stable across replays — that earns
// wiring the tuned margin into the live CacheBurstPaysBack gate (the "feed it back" seam this
// leaves pure ahead of, exactly as forksaving.go landed ahead of its live usage seam).
// Demotion/retirement evidence: if the tuned margin is ~0 across real corpora (the ex-ante
// estimate is already unbiased — no thin-headroom negative-net cluster to bail), the loop
// learns nothing and this folds back into the plain CacheBurstPaysBack gate. Invalidating
// assumption: that a per-fire receipt can attribute the realized horizon to the fire that shed
// it — if the shed saving cannot be isolated per fire on the live seam (interleaved fires,
// unattributable session end), the ex-post net is unmeasurable and this stays a modeled
// primitive driven by the frozen corpus, not a witnessed one.

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// CompactionFireScoreSchema tags a durable per-fire compaction score row so a reader never
// confuses it for another rsiloop journal (the fork-saving witness, the review-fire ledger,
// the rulesynth corpus). The "/1" is the row-shape version. The row is built pure
// (NewCompactionFireScoreRow) and persisted by the caller that wires this onto the live A4
// receipt seam.
const CompactionFireScoreSchema = "fak-compaction-fire-score/1"

// CompactionFireObs is the observed per-fire facts one compaction receipt records — the raw
// input the score is priced from. All token figures are in the same ~4-chars/token currency
// as the fire gate's headBurstEconomics and the provider input_tokens.
type CompactionFireObs struct {
	// Seq is the receipt's position in the corpus/journal — echoed onto the score for audit.
	Seq int

	// PredictedHorizonMargin is the EX-ANTE decision feature: the turns of headroom the gate
	// saw over the break-even at fire time, i.e. predictedRemainingTurns − CacheBurstBreakEvenTurns.
	// The untuned gate fired because this was >= 0 (CacheBurstPaysBack). It is the ONLY feature
	// the tuned threshold gates on — never the ex-post net (that would be circular).
	PredictedHorizonMargin int

	// ShedTokens is the cached middle the fire sheds FOREVER (headBurstEconomics'
	// droppedCachedTokens) — the per-turn read-rebate base (B).
	ShedTokens int
	// BurstTokens is the cached suffix the fire cold-rewrites ONCE (headBurstEconomics'
	// invalidatedSuffixTokens) — the one-time premium base (A2).
	BurstTokens int
	// ObservedHorizonTurns is the EX-POST fact: the number of turns that actually followed the
	// fire, over which the shed read-rebate actually accrued. A session that ended early makes
	// this smaller than the predicted horizon — the gap that flips a gate-approved fire negative.
	ObservedHorizonTurns int

	// WriteMultiplier is the cache-write tier the burst paid — cacheprice.Write5mMultiplier
	// (1.25x, the default, matching the Track-2 report's unattributed-creation convention) or
	// cacheprice.Write1hMultiplier (2.0x) when the creation split is 1h-attributed. <=0 defaults
	// to the 5m tier.
	WriteMultiplier float64
}

// CompactionFireReceipt is one per-fire compaction receipt (A4) with the per-fire NET-VALUE
// SCORE (#2817) attached: the observed ex-post net (B2 shed saving, A2 burst cost debited),
// in input-token-equivalents on the cacheprice basis, alongside the ex-ante feature the fire
// gate decided on. The net keeps its SIGN — a value-destroying fire reads negative (#1303: do
// not floor a net at zero).
type CompactionFireReceipt struct {
	Seq                    int     `json:"seq"`
	PredictedHorizonMargin int     `json:"predicted_horizon_margin"`
	ShedTokens             int     `json:"shed_tokens"`
	BurstTokens            int     `json:"burst_tokens"`
	ObservedHorizonTurns   int     `json:"observed_horizon_turns"`
	WriteMultiplier        float64 `json:"write_multiplier"`

	// ShedSavingTokenEquiv is B: the read rebate the shed middle earned over the realized
	// horizon — shedTokens × ReadMultiplier × observedHorizonTurns. Each surviving turn no
	// longer re-reads the shed tokens, so the rebate accrues per turn.
	ShedSavingTokenEquiv float64 `json:"shed_saving_token_equiv"`
	// BurstCostTokenEquiv is A2: the one-time excess-over-read the fire paid to cold-rewrite the
	// invalidated suffix — burstTokens × (WriteMultiplier − ReadMultiplier). A real cost that
	// nets against the shed saving.
	BurstCostTokenEquiv float64 `json:"burst_cost_token_equiv"`
	// NetScoreTokenEquiv is the HEADLINE per-fire score: shed saving − burst cost, SIGNED (B2
	// net over the A2 debit). Negative = the fire destroyed value vs not firing.
	NetScoreTokenEquiv float64 `json:"net_score_token_equiv"`
}

// NetScore is the per-fire net-value score — the signal the fire gate learns from. It is the
// stored headline, exposed as a method so callers read the score without reaching for the
// field name (and so a future re-derivation has one seam to change).
func (r CompactionFireReceipt) NetScore() float64 { return r.NetScoreTokenEquiv }

// ScoreCompactionFire is the PURE fold that prices one fire's observed facts into a scored
// receipt on the canonical cacheprice basis (#2798) — the SAME economics
// CacheBurstBreakEvenTurns prices ex-ante (perTurnSaving = shed × ReadMultiplier;
// oneTimePenalty = burst × (WriteMultiplier − ReadMultiplier)), applied EX-POST to what the
// fire realized over ObservedHorizonTurns. It does no I/O, so it stays deterministic and
// unit-testable (the #2819 discipline).
func ScoreCompactionFire(obs CompactionFireObs) CompactionFireReceipt {
	writeMult := obs.WriteMultiplier
	if writeMult <= 0 {
		writeMult = cacheprice.Write5mMultiplier
	}
	shedSaving := float64(obs.ShedTokens) * cacheprice.ReadMultiplier * float64(obs.ObservedHorizonTurns)
	burstCost := float64(obs.BurstTokens) * (writeMult - cacheprice.ReadMultiplier)
	return CompactionFireReceipt{
		Seq:                    obs.Seq,
		PredictedHorizonMargin: obs.PredictedHorizonMargin,
		ShedTokens:             obs.ShedTokens,
		BurstTokens:            obs.BurstTokens,
		ObservedHorizonTurns:   obs.ObservedHorizonTurns,
		WriteMultiplier:        writeMult,
		ShedSavingTokenEquiv:   shedSaving,
		BurstCostTokenEquiv:    burstCost,
		NetScoreTokenEquiv:     shedSaving - burstCost,
	}
}

// FirePolicy is the fire/bail threshold the RSI loop tunes. A fire is taken iff its predicted
// horizon margin meets MinHorizonMargin. The untuned gate is MinHorizonMargin 0 — fire whenever
// the burst is predicted to pay back at all (today's CacheBurstPaysBack). Raising the margin
// requires extra predicted headroom before firing, bailing the thin-headroom fires whose
// realized net most often goes negative.
type FirePolicy struct {
	MinHorizonMargin int `json:"min_horizon_margin"`
}

// Fires reports whether the policy would take this fire, given its ex-ante predicted margin.
func (p FirePolicy) Fires(r CompactionFireReceipt) bool {
	return r.PredictedHorizonMargin >= p.MinHorizonMargin
}

// MeanPerFireNet is the corpus acceptance metric (#2817): the average over ALL receipts of
// each fire's realized net UNDER the policy — a fired receipt contributes its NetScore, a
// bailed one contributes 0 (no fire = no burst, no shed, no saving). Suppressing a negative-net
// fire raises the mean; suppressing a positive-net one lowers it, so the tuner cannot cheat by
// bailing everything. An empty corpus is 0.
func MeanPerFireNet(corpus []CompactionFireReceipt, p FirePolicy) float64 {
	if len(corpus) == 0 {
		return 0
	}
	var sum float64
	for _, r := range corpus {
		if p.Fires(r) {
			sum += r.NetScore()
		}
	}
	return sum / float64(len(corpus))
}

// BaselineFirePolicy is the untuned gate: MinHorizonMargin 0, which fires every receipt in a
// corpus of ACTUAL fires (they were all gate-approved, so all have PredictedHorizonMargin >= 0).
// It is the reference the tuned policy must beat.
func BaselineFirePolicy() FirePolicy { return FirePolicy{MinHorizonMargin: 0} }

// TuneResult is the outcome of one tuning pass: the untuned baseline and the tuned policy, with
// the mean per-fire net each achieves over the corpus. TunedMeanNet >= BaselineMeanNet always
// (the baseline is in the candidate set), and the loop reports the LIFT the tuned threshold buys.
type TuneResult struct {
	Baseline        FirePolicy `json:"baseline"`
	Tuned           FirePolicy `json:"tuned"`
	BaselineMeanNet float64    `json:"baseline_mean_net"`
	TunedMeanNet    float64    `json:"tuned_mean_net"`
	Corpus          int        `json:"corpus"`
}

// Lift is the improvement in mean per-fire net the tuned threshold buys over the untuned gate.
// Zero when the ex-ante estimate is already unbiased (no negative-net cluster to bail) — the
// demotion signal named in the generation frame.
func (t TuneResult) Lift() float64 { return t.TunedMeanNet - t.BaselineMeanNet }

// TuneFirePolicy is the RSI tuning pass (#2817). Over a fixed corpus of scored per-fire
// receipts it sweeps the fire/bail threshold and returns the policy that MAXIMIZES mean
// per-fire net — the negative-net fires get bailed by the tuned MinHorizonMargin. The sweep is
// deterministic: the candidate thresholds are the distinct predicted-margin values present in
// the corpus (each partitions the corpus into fire/bail differently) plus one above the maximum
// (bail everything), so no hyperparameter is introduced and the KEEP reproduces bit-for-bit.
// Ties keep the SMALLEST margin — bail as few fires as possible for the same net, never
// over-suppress. With no signal (the ex-ante estimate is unbiased) the baseline wins its own
// tie and the tuned policy IS the baseline, so tuning never degrades the gate.
func TuneFirePolicy(corpus []CompactionFireReceipt) TuneResult {
	baseline := BaselineFirePolicy()
	res := TuneResult{
		Baseline:        baseline,
		Tuned:           baseline,
		BaselineMeanNet: MeanPerFireNet(corpus, baseline),
		Corpus:          len(corpus),
	}
	res.TunedMeanNet = res.BaselineMeanNet
	if len(corpus) == 0 {
		return res
	}

	// Candidate thresholds: every distinct predicted margin (ascending) plus max+1 (bail all).
	// A threshold equal to a distinct margin d fires iff margin >= d, so scanning d ascending
	// progressively bails the lowest-margin fires; max+1 bails everything.
	seen := map[int]bool{}
	maxMargin := corpus[0].PredictedHorizonMargin
	for _, r := range corpus {
		seen[r.PredictedHorizonMargin] = true
		if r.PredictedHorizonMargin > maxMargin {
			maxMargin = r.PredictedHorizonMargin
		}
	}
	candidates := make([]int, 0, len(seen)+1)
	for m := range seen {
		candidates = append(candidates, m)
	}
	candidates = append(candidates, maxMargin+1)
	sort.Ints(candidates)

	best := res.TunedMeanNet
	for _, m := range candidates {
		p := FirePolicy{MinHorizonMargin: m}
		mean := MeanPerFireNet(corpus, p)
		if mean > best {
			best = mean
			res.Tuned = p
			res.TunedMeanNet = mean
		}
	}
	return res
}

// CompactionFireScoreRow is one durable per-fire compaction score witness record — the row a
// caller appends onto the A4 receipt seam when this primitive is wired live, tagged
// CompactionFireScoreSchema so it never shares a row shape with the fork-saving witness or the
// review-fire ledger. It carries the observed split and the net score so the "per-fire net"
// signal becomes a re-measurable ledger the tuner replays, not a one-time number.
type CompactionFireScoreRow struct {
	Schema string `json:"schema"`
	CompactionFireReceipt
}

// NewCompactionFireScoreRow scores one fire and wraps it as a durable witness row. It is PURE
// (returns the row; the caller persists it), so wiring the live witness onto the A4 receipt
// seam is a trivial append and this primitive stays unit-testable.
func NewCompactionFireScoreRow(obs CompactionFireObs) CompactionFireScoreRow {
	return CompactionFireScoreRow{Schema: CompactionFireScoreSchema, CompactionFireReceipt: ScoreCompactionFire(obs)}
}

// Summary renders a tuning pass as a one-line operator readout — the tuned fire/bail threshold
// and the lift in mean per-fire net it buys, net figures first (the #2797 honesty order).
func (t TuneResult) Summary() string {
	return fmt.Sprintf(
		"fire-gate tune (%d receipts): mean per-fire net %+.0f → %+.0f tok-eq (lift %+.0f) at MinHorizonMargin %d→%d turns",
		t.Corpus, t.BaselineMeanNet, t.TunedMeanNet, t.Lift(),
		t.Baseline.MinHorizonMargin, t.Tuned.MinHorizonMargin,
	)
}
