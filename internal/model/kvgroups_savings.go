package model

import "fmt"

// kvgroups_savings.go — the operator-facing READOUT of the hybrid KV memory plane (#2241,
// R1 of M9, epic #2236). kvgroups.go lands the classification + per-group budget arithmetic;
// this file turns that arithmetic into the gen/next "operator readout" artifact: a single
// grouped-vs-uniform comparison a dashboard, a preemption estimator, or the R3 resident-bytes
// benchmark reads to see how much history the window/recurrent groups let a hybrid model shed
// at a fixed context length.
//
// Generation frame (gen/next — kept behind FAK_HYBRID_KV until promotion evidence lands):
//   - promotion evidence (gen/next -> now): the R2 hybrid-radix swap/restore witnesses green
//     on a real Gated-DeltaNet model AND the R3 bench child of #2236 showing grouped resident
//     bytes beating uniform (and tracking vLLM HMA) on the same model class. Until both land,
//     this plane stays gated.
//   - demotion / retirement evidence: if the grouped budget does not beat uniform on real
//     hybrid configs, or a family's window / linear-attention layers are not classifiable from
//     Config (misclassification), retire the flag rather than promote it.
//   - invalidating assumption: classification is complete from Config.LayerTypes / Config.Window
//     plus the linear-attn dims. A hybrid family that encodes its window or state layers some
//     other way would misclassify. And at a very small context a recurrent layer's fixed O(1)
//     state can EXCEED that layer's tiny uniform per-position share, so the saving is SIGNED and
//     can be negative below the break-even context — this readout reports the true delta rather
//     than clamping it to zero.

// KVGroupSavings is the grouped-vs-uniform readout at a fixed context length: the grouped
// resident budget, the uniform allocation it replaces, and the SIGNED delta between them. It
// is the operator-facing surface over KVGroupBudget — the arithmetic lives in kvgroups.go;
// this is what a human, a sizing estimator, or the R3 benchmark reads.
type KVGroupSavings struct {
	Ctx           int           // context length the readout was computed at
	Budget        KVGroupBudget // the per-group split behind the totals
	GroupedFloats int           // resident float32 under grouped allocation (Budget.TotalFloats)
	UniformFloats int           // resident float32 under the pre-grouping uniform allocation
	SavedFloats   int           // UniformFloats - GroupedFloats; SIGNED (negative below break-even)
	SavedFraction float64       // SavedFloats / UniformFloats; 0 when UniformFloats == 0
}

// KVGroupSavingsAt computes the grouped-vs-uniform readout at context length ctx by folding
// the per-group budget (kvgroups.go) against the uniform baseline it replaces.
func (c Config) KVGroupSavingsAt(ctx int) KVGroupSavings {
	b := c.KVGroupBudgetAt(ctx)
	grouped := b.TotalFloats()
	uniform := c.UniformKVFloats(ctx)
	saved := uniform - grouped
	frac := 0.0
	if uniform > 0 {
		frac = float64(saved) / float64(uniform)
	}
	return KVGroupSavings{
		Ctx:           ctx,
		Budget:        b,
		GroupedFloats: grouped,
		UniformFloats: uniform,
		SavedFloats:   saved,
		SavedFraction: frac,
	}
}

// SavedBytes is the resident saving expressed in bytes (4 bytes per float32); SIGNED, matching
// SavedFloats.
func (s KVGroupSavings) SavedBytes() int { return s.SavedFloats * 4 }

// Readout renders a one-line operator string, e.g.
//
//	"hybrid-KV ctx=1024: grouped 49424 vs uniform 147456 float32 (saved 66.5%; layers 1 full / 1 window / 1 recurrent)"
//
// The percentage is signed, so a below-break-even context reads honestly as a negative saving.
func (s KVGroupSavings) Readout() string {
	return fmt.Sprintf(
		"hybrid-KV ctx=%d: grouped %d vs uniform %d float32 (saved %.1f%%; layers %d full / %d window / %d recurrent)",
		s.Ctx, s.GroupedFloats, s.UniformFloats, s.SavedFraction*100,
		s.Budget.FullLayers, s.Budget.WindowLayers, s.Budget.RecurrentLayers,
	)
}
