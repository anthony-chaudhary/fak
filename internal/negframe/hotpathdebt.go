package negframe

// hotpathdebt.go measures the negation tax that actually rides the hot-path injected-prose
// surface after emit-time reframing (#4419), and orders paydown by how often each surface is
// broadcast (#4408). It is purely additive: it reads the guard-runtime corpus and the existing
// ReframePass telemetry, and touches neither Build's document-corpus negframe_debt nor the
// diff-scoped ratchet, so the established card and its gate are unchanged.
//
// The honest hot-path debt is NOT "how many negatives the source contains" -- the emit sites
// already run Reframe, so a negative with a clean mechanical inverse is flipped before it ever
// reaches the model and carries zero residual. What survives into the broadcast is exactly the
// mechanical candidate the token guard REFUSED to flip (ReframePass.VerbatimFallback): a
// negatively-framed idiom fak could not rewrite without dropping a must-keep contract token, so
// it ships the original span verbatim. That count -- the reframable negation the surface is still
// broadcasting despite Reframe -- is the headline integer, and weighting it by broadcast tier
// puts the per-turn surfaces (paid every turn) ahead of the per-session and cold ones.

import "sort"

// HotPathSurfaceDebt is one surface's post-Reframe residual and its broadcast-weighted debt.
type HotPathSurfaceDebt struct {
	Name          string        `json:"name"`
	Tier          BroadcastTier `json:"tier"`
	MechResidual  int           `json:"mech_residual"`  // mechanical negatives broadcast despite Reframe (token-guard refusals)
	JudgeResidual int           `json:"judge_residual"` // judgement-tier negatives left in the broadcast (advisory)
	Weighted      int           `json:"weighted"`       // MechResidual * Tier.Weight() -- the paydown-ordering key
}

// HotPathReport folds the whole hot-path corpus: the headline mechanical residual (the hot-path
// negframe_debt integer), the advisory judgement total, the broadcast-weighted debt paydown sorts
// by, and the per-surface breakdown ordered worst (hottest-weighted) first.
type HotPathReport struct {
	Surfaces      int                  `json:"surfaces"`
	MechResidual  int                  `json:"mech_residual"`  // headline: reframable negation still broadcast on the hot path
	JudgeResidual int                  `json:"judge_residual"` // advisory total (never gates)
	WeightedDebt  int                  `json:"weighted_debt"`  // sum of per-surface Weighted (#4408 ordering signal)
	PerSurface    []HotPathSurfaceDebt `json:"per_surface"`    // worst-first: highest broadcast-weighted debt leads
}

// HotPathDebt folds corpus into a HotPathReport. Each surface is reframed exactly as the emit
// site reframes it; the surviving mechanical residual is its VerbatimFallback, weighted by the
// surface's broadcast tier. PerSurface is sorted worst-first (highest Weighted, then raw
// mechanical residual, then name) so a --suggest / paydown view leads with the hottest surface.
func HotPathDebt(corpus []HotPathString) HotPathReport {
	rep := HotPathReport{Surfaces: len(corpus)}
	for _, hp := range corpus {
		res := ReframePass(hp.Text)
		d := HotPathSurfaceDebt{
			Name:          hp.Name,
			Tier:          hp.Tier,
			MechResidual:  res.VerbatimFallback,
			JudgeResidual: res.ResidualNegatives,
			Weighted:      res.VerbatimFallback * hp.Tier.Weight(),
		}
		rep.MechResidual += d.MechResidual
		rep.JudgeResidual += d.JudgeResidual
		rep.WeightedDebt += d.Weighted
		rep.PerSurface = append(rep.PerSurface, d)
	}
	sort.SliceStable(rep.PerSurface, func(a, b int) bool {
		pa, pb := rep.PerSurface[a], rep.PerSurface[b]
		if pa.Weighted != pb.Weighted {
			return pa.Weighted > pb.Weighted
		}
		if pa.MechResidual != pb.MechResidual {
			return pa.MechResidual > pb.MechResidual
		}
		return pa.Name < pb.Name
	})
	return rep
}
