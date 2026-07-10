package memq

import (
	"context"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// MergeOnEvictKind labels the audit Effect a budget op's opt-in merge-on-evict fold
// records (#4015): a below-budget cell folded into its nearest surviving cell rather
// than tail-dropped. It is an EFFECT kind, never a pipeline op — the fold is a
// modifier on OpBudget (Op.MergeFloor), and it is proposal-only (Applied stays false;
// like OpConsolidate, the durable write-back is rung 2). One Effect is recorded per
// fold, naming [survivor, evicted] so the provenance of each absorption is auditable.
const MergeOnEvictKind = "merge_on_evict"

// applyMergeNearest is the merge-on-evict fold (#4015, borrowing LOOK-M's pivot merge,
// CAM's merge-on-evict, and KVMerger's weighted-similarity merge): rather than let the
// budget tail-drop a below-cut cell outright, route each dropped cell into its single
// most-similar SURVIVING cell and preserve its provenance there. For each dropped cell
// it embeds the body (simhash) and takes the argmax cosine over the survivors' bodies;
// if the best survivor's similarity is at or above floor, the dropped cell's Refs — and
// its own ID — are unioned onto that survivor (so a downstream consumer still reaches
// what it referenced), an audit Effect is recorded, and the cell leaves the dropped set.
// A dropped cell with no survivor above the floor stays dropped (it reaches the overflow
// verdict unchanged), so the graceful path never hides a genuine loss.
//
// The upstream tradeoff is that a bad nearest match corrupts the survivor; memq's
// guardrail is the explicit similarity floor plus the fact that the fold is
// provenance-only — no cell BYTES are rewritten (memq has no op that mutates a cell's
// body), so a mismatched fold at worst over-attaches a ref, it never garbles content.
// Sealed cells are never a fold target and never a fold source (never fold poison, and
// never fold into a body we cannot read); a cell that will not page in is skipped the
// same way computeNearDup/computeDivergence skip theirs, leaving it on the dropped list.
//
// Deterministic: survivors are embedded in working-set order, the argmax keeps the
// FIRST (lowest working-set index) survivor on a cosine tie, dropped cells are processed
// in working-set order, and each survivor's unioned Refs are sorted — no RNG, no
// map-order dependence in any decision. Returns the (possibly ref-augmented) survivor
// set, the still-dropped remainder in working-set order, and a one-line step note.
func applyMergeNearest(ctx context.Context, b Backend, res *Result, survivors, dropped []Cell, floor float64) ([]Cell, []Cell, string) {
	// Embed each eligible survivor once, remembering its index in `survivors` so a fold
	// can write the unioned Refs straight back.
	type svec struct {
		idx int
		vec simhash.Vector
	}
	svecs := make([]svec, 0, len(survivors))
	for i, c := range survivors {
		if c.Sealed {
			continue
		}
		body, err := b.Materialize(ctx, c.ID)
		if err != nil || len(body) == 0 {
			continue
		}
		svecs = append(svecs, svec{idx: i, vec: simhash.Embed(string(body))})
	}
	if len(svecs) == 0 {
		return survivors, dropped, "" // no readable survivor to fold into — nothing merges
	}

	var still []Cell
	merged := 0
	for _, d := range dropped {
		if d.Sealed {
			still = append(still, d)
			continue
		}
		body, err := b.Materialize(ctx, d.ID)
		if err != nil || len(body) == 0 {
			still = append(still, d)
			continue
		}
		dv := simhash.Embed(string(body))
		bestIdx, bestCos := -1, -2.0 // -2 is below cosine's -1 floor, so the first survivor always sets it
		for _, sv := range svecs {
			if cos := simhash.Cosine(dv, sv.vec); cos > bestCos {
				bestCos, bestIdx = cos, sv.idx
			}
		}
		if bestIdx < 0 || bestCos < floor {
			still = append(still, d) // no fold target above the floor: a genuine eviction
			continue
		}
		s := survivors[bestIdx]
		s.Refs = unionRefs(s.Refs, d.Refs, d.ID)
		survivors[bestIdx] = s
		res.Effects = append(res.Effects, Effect{
			Kind:    MergeOnEvictKind,
			Applied: false,
			Cells:   []string{s.ID, d.ID},
			Note: fmt.Sprintf("folded evicted cell %s into nearest survivor %s (cosine %.3f >= floor %.3f); refs preserved, durable write-back is rung 2",
				d.ID, s.ID, bestCos, floor),
		})
		merged++
	}
	if merged == 0 {
		return survivors, still, ""
	}
	res.Stats.MergedOnEvict += merged
	return survivors, still, fmt.Sprintf("merged %d evicted cell(s) into nearest survivor(s) (floor %.3f)", merged, floor)
}

// unionRefs returns the sorted, de-duplicated union of a survivor's existing refs, an
// evicted cell's refs, and the evicted cell's own ID (its provenance handle). Sorting
// makes the survivor's Refs a deterministic function of the inputs regardless of fold
// order, so a merged Result stays byte-identical across runs.
func unionRefs(existing, folded []string, id string) []string {
	set := make(map[string]bool, len(existing)+len(folded)+1)
	for _, r := range existing {
		set[r] = true
	}
	for _, r := range folded {
		set[r] = true
	}
	set[id] = true
	out := make([]string, 0, len(set))
	for r := range set {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}
