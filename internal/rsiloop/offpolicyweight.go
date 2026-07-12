package rsiloop

// offpolicyweight.go — SDPO++-inspired off-policy-age trust-decay for the RSI
// ARCHIVE-MINING arms (#3917). Provenance: INSPIRE / clean-room — the idea (not a
// line of code) is borrowed from lasgroup/SDPO's `compute_self_distillation_loss`
// (an importance ratio CLAMPED to `is_clip` so one off-policy token can't dominate
// the update) and OPSD's explicit on-policy/off-policy accounting. NO code copied;
// OPSD is unlicensed and excluded from any reuse regardless.
//
// The transferable, ML-stripped rule: continual learning from stale, off-policy
// samples is stable IFF you (a) BOUND how off-policy a sample may be before it
// feeds the update (staleness-K) and (b) CLIP its contribution rather than
// penalising drift from a baseline — "constrain the step, not the destination."
//
// fak's RSI archive-mining arms fold a corpus of samples produced across many
// LEVER GENERATIONS, and that corpus is off-policy by construction: a stepping
// stone validated against an EARLIER baseline is stale w.r.t. the current config.
// The keep/evaluate arms re-measure a FRESH witness (#1021), so staleness cannot
// bite them; the archive-mining arms do NOT re-run, so they need an explicit age
// weight. This file is that primitive plus the first consumer that stops weighting
// every historical stepping stone equally (AgeDecayArchiveProposer, the smallest-
// blast-radius wiring into ProposeLoopVariants). The dojocal CalibErr fold and the
// metarsi cycle window are named follow-ons, not touched here.

import "sort"

// OffPolicyWeight is the staleness-K trust weight for a sample produced at
// sampleGen when the current lever generation is currentGen. It is
// clamp01(1 - (currentGen-sampleGen)/k): a fresh sample (delta <= 0) weighs 1, a
// sample K OR MORE generations old weighs a HARD 0 (the bounded off-policy age —
// SDPO's staleness-K), and it decays linearly in between. k <= 0 DISABLES the
// decay (every sample weighs 1) so the decay is strictly opt-in: a caller that
// passes no bound keeps today's equal-weight behaviour. A negative delta (a sample
// from a FUTURE generation, e.g. an out-of-order append) is treated as fresh,
// never weighted above 1.
func OffPolicyWeight(sampleGen, currentGen, k int) float64 {
	if k <= 0 {
		return 1
	}
	delta := currentGen - sampleGen
	if delta <= 0 {
		return 1
	}
	if delta >= k {
		return 0
	}
	return 1 - float64(delta)/float64(k)
}

// CapInfluence clips one sample's contribution to +/- clip — the `is_clip`
// analogue, so no single heavy-tailed (stale, or huge-gain) sample can dominate a
// fold. clip <= 0 disables the cap. Contributions are signed magnitudes, so the
// clip is symmetric about zero.
func CapInfluence(influence, clip float64) float64 {
	if clip <= 0 {
		return influence
	}
	switch {
	case influence > clip:
		return clip
	case influence < -clip:
		return -clip
	default:
		return influence
	}
}

// AgeWeightedInfluence is one sample's fold contribution: its raw influence
// DECAYED by off-policy age (OffPolicyWeight) and THEN clipped (CapInfluence). The
// two guards compose in the SDPO++ order — bound the age first, then cap the
// survivor's magnitude. At generation-delta <= 0 with clip <= 0 it reduces EXACTLY
// to rawInfluence, so a fresh corpus folds identically to today's equal weight.
func AgeWeightedInfluence(rawInfluence float64, sampleGen, currentGen, k int, clip float64) float64 {
	return CapInfluence(rawInfluence*OffPolicyWeight(sampleGen, currentGen, k), clip)
}

// SteppingStone is one archived kept variant paired with its off-policy-age
// influence — the mined, trust-decayed view an archive-mining proposer builds on.
type SteppingStone struct {
	Variant    LoopVariant `json:"variant"`
	Generation int         `json:"generation"` // position in the archive (0 = oldest kept)
	Gain       float64     `json:"gain"`       // raw stepping-stone influence: its point-delta gain
	Influence  float64     `json:"influence"`  // Gain after off-policy-age decay + cap; 0 == too stale to mine
}

// AgeDecayArchiveProposer is the first archive-mining proposer that does NOT
// weight every historical stepping stone equally. It reads the kept-variant
// archive as off-policy samples: a stone's raw influence is the point-delta gain
// it once demonstrated, DECAYED by how many lever generations ago it was produced
// (StalenessK) and CAPPED (InfluenceCap). Stones K+ generations old decay to 0 and
// are dropped from the mine; the survivors are emitted as candidate seeds ordered
// most-trusted first, so a single stale or heavy-tailed stone cannot dominate the
// proposal. StalenessK <= 0 disables the decay and reduces to equal-weight mining
// (today's behaviour); an empty archive proposes nothing.
type AgeDecayArchiveProposer struct {
	StalenessK   int     // lever generations past which a stepping stone contributes 0
	InfluenceCap float64 // per-stone influence ceiling (the is_clip analogue); <= 0 disables
}

// AgeDecayArchiveProposer is a drop-in LoopVariantProposer: it satisfies the same
// interface the metaloop consumes, so wiring it in is a proposer swap, not a new
// call path (the smallest-blast-radius wiring the issue asks for).
var _ LoopVariantProposer = AgeDecayArchiveProposer{}

// WeighArchive folds the kept-variant archive into off-policy-age-weighted
// stepping stones. Generation is the record's index — the archive is append-only,
// so a later row is a newer generation and the newest row is the current
// generation (fully fresh). Only KEPT records with a positive demonstrated gain
// are minable stepping stones; a REVERT or a zero-gain row is skipped.
func (p AgeDecayArchiveProposer) WeighArchive(archive []LoopVariantRecord) []SteppingStone {
	if len(archive) == 0 {
		return nil
	}
	currentGen := len(archive) - 1
	stones := make([]SteppingStone, 0, len(archive))
	for i, rec := range archive {
		if !rec.Kept {
			continue
		}
		gain := rec.After.Points - rec.Before.Points
		if gain <= 0 {
			continue
		}
		stones = append(stones, SteppingStone{
			Variant:    rec.Variant,
			Generation: i,
			Gain:       gain,
			Influence:  AgeWeightedInfluence(gain, i, currentGen, p.StalenessK, p.InfluenceCap),
		})
	}
	return stones
}

// ProposeLoopVariants mines the archive for stepping stones to build on, decays
// and caps each stone's influence, DROPS any that decayed to 0 (too stale to
// trust), and returns the survivors as candidate seeds ordered by descending
// influence (most-trusted first; ties broken newest-generation first, then by ID,
// for a deterministic order). The baseline is unused: this proposer's whole job is
// to surface WHICH archived stones are still trustworthy enough to build on.
func (p AgeDecayArchiveProposer) ProposeLoopVariants(_ LoopConfig, archive []LoopVariantRecord) ([]LoopVariant, error) {
	stones := p.WeighArchive(archive)
	kept := make([]SteppingStone, 0, len(stones))
	for _, s := range stones {
		if s.Influence > 0 {
			kept = append(kept, s)
		}
	}
	sort.SliceStable(kept, func(a, b int) bool {
		if kept[a].Influence != kept[b].Influence {
			return kept[a].Influence > kept[b].Influence
		}
		if kept[a].Generation != kept[b].Generation {
			return kept[a].Generation > kept[b].Generation
		}
		return kept[a].Variant.ID < kept[b].Variant.ID
	})
	out := make([]LoopVariant, 0, len(kept))
	for _, s := range kept {
		out = append(out, s.Variant)
	}
	return out, nil
}
