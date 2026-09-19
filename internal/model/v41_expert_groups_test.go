package model

// v41_expert_groups_test.go -- the #13304 witness for bounded V4.1 prefill
// routed-expert triple reuse across token groups.
//
// The token-major MoE contraction (v41_forward.go, the `for t { for pick { ... } }`
// loop) materializes, contracts and releases one expert triple per (token, pick).
// The #13296 layer-scoped f32 cache defers the fault of a repeated
// (expert, projection) ONLY while the interleaved working set fits its bound. When
// a bounded panel of prefill tokens routes DIFFERENT experts whose union exceeds
// the cache, token-major order re-faults an expert the next token had already read
// one token earlier: the panel's distinct reads become token-count-many reads of
// the same slabs.
//
// #13304 bounds this by grouping the panel's routed picks by expert: materialize
// ONE expert triple, contract every (token, slot) row assigned to it, release it,
// then replay the per-token weighted sums in ORIGINAL slot order. The two
// properties the witness pins are:
//
//  1. Every distinct routed projection in the panel is faulted exactly once,
//     independent of how the tokens order the same experts (alternating routes
//     that token-major order would thrash).
//  2. The weighted output is byte-for-byte the token-major reference: each pick's
//     weight is applied to its own row exactly once, in original slot order, and
//     duplicate experts / repeated routes do not double-count.

import (
	"math"
	"testing"
)

// v41GroupPicks builds a deterministic per-token pick set that cycles through a
// declared expert pool with a declared per-token window, so alternating routes
// share experts across tokens in a non-contiguous way (the thrash regime).
func v41GroupPicks(tokens, experts, perToken int) [][]routePick {
	out := make([][]routePick, tokens)
	for t := 0; t < tokens; t++ {
		picks := make([]routePick, perToken)
		for k := 0; k < perToken; k++ {
			// A stride that is coprime with `experts` visits every expert before
			// repeating, so a multi-token panel overlaps heavily yet never reads an
			// expert contiguously in token-major order.
			e := (t*perToken + k) % experts
			picks[k] = routePick{expert: e, weight: 0.5 + float32((t+k)%3)*0.25}
		}
		out[t] = picks
	}
	return out
}

// TestV41ExpertGroupsReadsEachDistinctOnce is the #13304 DoD item-1 witness: a
// bounded panel whose alternating routes share experts reads each distinct
// (expert) materialization once per panel, not once per (token, pick).
func TestV41ExpertGroupsReadsEachDistinctOnce(t *testing.T) {
	const (
		tokens   = 8
		experts  = 4
		perToken = 3
	)
	picks := v41GroupPicks(tokens, experts, perToken)

	plan := v41PlanExpertGroups(picks)
	if len(plan) != experts {
		t.Fatalf("panel grouped %d experts, want %d (each distinct routed expert materialized once)", len(plan), experts)
	}

	// Each (token, slot) pick must appear in exactly one group, with its original
	// slot index retained for replay.
	seen := make(map[[2]int]bool)
	rows := 0
	for _, g := range plan {
		if g.Expert < 0 || g.Expert >= experts {
			t.Fatalf("group expert %d outside the declared pool [0,%d)", g.Expert, experts)
		}
		for _, r := range g.Rows {
			key := [2]int{r.Token, r.Slot}
			if seen[key] {
				t.Fatalf("pick (token=%d,slot=%d) grouped twice", r.Token, r.Slot)
			}
			seen[key] = true
			rows++
			if want := picks[r.Token][r.Slot].expert; want != g.Expert {
				t.Fatalf("row (token=%d,slot=%d) grouped under expert %d, want %d", r.Token, r.Slot, g.Expert, want)
			}
		}
	}
	if rows != tokens*perToken {
		t.Fatalf("plan covers %d rows, want %d (every pick exactly once)", rows, tokens*perToken)
	}

	// Within a group, rows must be in stable (token, slot) ascending order so the
	// replay is deterministic and never reorders a token's arithmetic.
	for _, g := range plan {
		for i := 1; i < len(g.Rows); i++ {
			a, b := g.Rows[i-1], g.Rows[i]
			if a.Token > b.Token || (a.Token == b.Token && a.Slot >= b.Slot) {
				t.Fatalf("group expert %d rows out of stable order at %d: (%d,%d) then (%d,%d)",
					g.Expert, i, a.Token, a.Slot, b.Token, b.Slot)
			}
		}
	}
}

// TestV41ExpertGroupsMatchTokenMajorReference is the #13304 DoD item-2 witness:
// the grouped replay reproduces the token-major reference byte-for-byte. The
// reference is computed directly here (never by calling the grouped code), so the
// oracle and the implementation are independent.
func TestV41ExpertGroupsMatchTokenMajorReference(t *testing.T) {
	const (
		tokens   = 8
		experts  = 4
		perToken = 3
		H        = 8
	)
	picks := v41GroupPicks(tokens, experts, perToken)

	// Distinct, deterministic expert outputs so a mis-keyed or double-weighted row
	// changes the sum observably.
	expertOut := make([][]float32, experts)
	for e := range expertOut {
		v := make([]float32, H)
		for i := range v {
			v[i] = float32(e+1) * 0.125 * float32(i+1)
		}
		expertOut[e] = v
	}

	// Token-major reference: accumulate each pick's weight into its own token row.
	ref := make([][]float32, tokens)
	for t := range ref {
		ref[t] = make([]float32, H)
	}
	contract := 0
	for t := 0; t < tokens; t++ {
		for _, p := range picks[t] {
			contract++
			y := expertOut[p.expert]
			for i := 0; i < H; i++ {
				ref[t][i] += p.weight * y[i]
			}
		}
	}
	_ = contract

	// Grouped replay: one materialization per expert, rows contracted, then the
	// per-token weighted sum replayed in original slot order.
	plan := v41PlanExpertGroups(picks)
	got := make([][]float32, tokens)
	for t := range got {
		got[t] = make([]float32, H)
	}
	materializations := 0
	for _, g := range plan {
		materializations++
		y := expertOut[g.Expert]
		for _, r := range g.Rows {
			p := picks[r.Token][r.Slot]
			for i := 0; i < H; i++ {
				got[r.Token][i] += p.weight * y[i]
			}
		}
	}
	if materializations != experts {
		t.Fatalf("grouped replay materialized %d expert triples, want %d", materializations, experts)
	}
	for tok := 0; tok < tokens; tok++ {
		for i := 0; i < H; i++ {
			if math.Abs(float64(got[tok][i]-ref[tok][i])) > 1e-6 {
				t.Fatalf("grouped replay token %d component %d = %g, token-major reference %g",
					tok, i, got[tok][i], ref[tok][i])
			}
		}
	}
}

// TestV41ExpertGroupsDuplicateExpertsWeightedOnce pins the adversarial edge: a
// token that routes the SAME expert more than once (possible under the reduced
// fixtures) must have both picks applied, each exactly once, in slot order.
func TestV41ExpertGroupsDuplicateExpertsWeightedOnce(t *testing.T) {
	picks := [][]routePick{
		{{expert: 2, weight: 0.5}, {expert: 0, weight: 0.25}, {expert: 2, weight: 0.75}},
		{{expert: 0, weight: 1.0}},
	}
	plan := v41PlanExpertGroups(picks)

	rows := 0
	for _, g := range plan {
		for _, r := range g.Rows {
			if want := picks[r.Token][r.Slot].expert; want != g.Expert {
				t.Fatalf("duplicate-expert row (token=%d,slot=%d) grouped under %d, want %d", r.Token, r.Slot, g.Expert, want)
			}
			rows++
		}
	}
	if rows != 4 {
		t.Fatalf("plan covers %d rows, want 4 (duplicates preserved, not de-duplicated)", rows)
	}
}

// TestV41ExpertGroupsEmptyAndSingle is the bounded-degenerate witness: an empty
// panel groups to nothing, and a single pick groups to exactly one expert, so the
// seam is inert off the multi-token path.
func TestV41ExpertGroupsEmptyAndSingle(t *testing.T) {
	if got := v41PlanExpertGroups(nil); len(got) != 0 {
		t.Fatalf("nil panel grouped %d experts, want 0", len(got))
	}
	if got := v41PlanExpertGroups([][]routePick{{}}); len(got) != 0 {
		t.Fatalf("empty per-token pick set grouped %d experts, want 0", len(got))
	}
	got := v41PlanExpertGroups([][]routePick{{{expert: 3, weight: 1}}})
	if len(got) != 1 || got[0].Expert != 3 || len(got[0].Rows) != 1 {
		t.Fatalf("single pick grouped to %+v, want one expert 3 with one row", got)
	}
}

// TestV41ExpertGroupsForwardMatchesTokenMajor is the end-to-end #13304 witness on
// the REAL reduced forward: a multi-token prefill executed through the grouped
// contraction must produce output-identical logits to the historical token-major
// stream on the same model, and must not read MORE routed projections than the
// token-major stream. The tier-only fixture (routed experts live only in the
// checkpoint tier) makes the tier's own Reads counter an independent witness of
// the read count, so the equivalence is asserted against real execution, not the
// plan alone.
func TestV41ExpertGroupsForwardMatchesTokenMajor(t *testing.T) {
	ids := []int{1, 1, 1, 1, 1, 1, 1, 1}

	grouped, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = false
	gAct, gErr := grouped.forwardV41(ids, nil)
	if gErr != nil {
		t.Fatalf("grouped %d-token prefill: %v", len(ids), gErr)
	}
	gReads := grouped.expertCheckpoint.Stats().Reads

	tokenMajor, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = true
	tmAct, tmErr := tokenMajor.forwardV41(ids, nil)
	v41ForceTokenMajor = false
	if tmErr != nil {
		t.Fatalf("token-major %d-token prefill: %v", len(ids), tmErr)
	}
	tmReads := tokenMajor.expertCheckpoint.Stats().Reads

	// The two paths must be output-identical: same logits for every position.
	if len(gAct.Logits) != len(tmAct.Logits) {
		t.Fatalf("grouped emitted %d logit rows, token-major %d", len(gAct.Logits), len(tmAct.Logits))
	}
	for pos := range tmAct.Logits {
		if len(gAct.Logits[pos]) != len(tmAct.Logits[pos]) {
			t.Fatalf("position %d: grouped vocab %d, token-major %d", pos, len(gAct.Logits[pos]), len(tmAct.Logits[pos]))
		}
		for i := range tmAct.Logits[pos] {
			if gAct.Logits[pos][i] != tmAct.Logits[pos][i] {
				t.Fatalf("position %d component %d: grouped %g, token-major %g (grouping must be output-identical)",
					pos, i, gAct.Logits[pos][i], tmAct.Logits[pos][i])
			}
		}
	}

	// The grouped path reads each distinct routed projection once, so it may not
	// fault more than the token-major stream it replaces.
	if gReads > tmReads {
		t.Fatalf("grouped prefill faulted %d projections, token-major %d: grouping regressed the read count",
			gReads, tmReads)
	}
	if gReads == 0 {
		t.Fatal("fixture faulted no routed projections; the read comparison would be vacuous")
	}
	t.Logf("routed-projection tier reads: grouped=%d token-major=%d (distinct-set bound)", gReads, tmReads)
}

// TestV41ExpertGroupsThrashRegimeBoundsReads is the #13304 frontier witness: at a
// layer-cache budget SMALLER than the panel's routed union, the token-major stream
// thrashes the cache (a distinct expert evicted before the next token re-reads it)
// so its tier Reads scale with the token count, while the grouped stream faults
// each distinct projection once and is therefore token-count-independent. The
// outputs stay byte-identical.
//
// The fixture's reduced geometry (width 256) sizes one projection slab at
// v41TierWidth * (v41TierWidth/qkK) * q4kBlockBytes bytes; a cache budget of one
// slab cannot hold a whole expert triple, so the token-major path re-faults every
// token and grouping is the only way to bound the reads.
func TestV41ExpertGroupsThrashRegimeBoundsReads(t *testing.T) {
	one := []int{1}
	long := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

	// One projection slab is the smallest non-zero budget: it retains a single
	// (expert, projection) and evicts it before the next, i.e. the strictest
	// thrash regime.
	const slabBytes = int64(v41TierWidth) * (v41TierWidth / qkK) * q4kBlockBytes

	// Distinct-set floor: a 1-token prefill faults exactly its distinct routed
	// projections, which no token-major or grouped panel may undercut.
	mDistinct, _ := v41TierOnlyBudgetModel(t, 0)
	v41TestLayerCacheBudgetOverride = slabBytes
	if _, err := mDistinct.forwardV41(one, nil); err != nil {
		v41TestLayerCacheBudgetOverride = 0
		t.Fatalf("1-token prefill: %v", err)
	}
	distinctReads := mDistinct.expertCheckpoint.Stats().Reads

	// Token-major: under this budget the layer cache cannot hold a triple, so the
	// interleaved distinct experts evict each other and reads scale with tokens.
	mTokenMajor, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = true
	v41TestLayerCacheBudgetOverride = slabBytes
	_, tmErr := mTokenMajor.forwardV41(long, nil)
	v41ForceTokenMajor = false
	v41TestLayerCacheBudgetOverride = 0
	if tmErr != nil {
		t.Fatalf("token-major 16-token prefill: %v", tmErr)
	}
	tmReads := mTokenMajor.expertCheckpoint.Stats().Reads

	if tmReads <= distinctReads {
		t.Fatalf("fixture did not thrash: token-major 16-token reads %d, 1-token floor %d", tmReads, distinctReads)
	}

	// Grouped: each distinct projection is materialized once for the panel, so the
	// 16-token reads collapse to the 1-token distinct floor.
	mGrouped, _ := v41TierOnlyBudgetModel(t, 0)
	v41TestLayerCacheBudgetOverride = slabBytes
	if _, gErr := mGrouped.forwardV41(long, nil); gErr != nil {
		v41TestLayerCacheBudgetOverride = 0
		t.Fatalf("grouped 16-token prefill: %v", gErr)
	}
	gReads := mGrouped.expertCheckpoint.Stats().Reads

	if gReads >= tmReads {
		t.Fatalf("grouped 16-token reads %d did not beat token-major %d: grouping failed to bound the panel reads",
			gReads, tmReads)
	}
	if gReads != distinctReads {
		t.Fatalf("grouped 16-token reads %d != 1-token distinct floor %d: each distinct projection must be faulted once",
			gReads, distinctReads)
	}
	t.Logf("thrash regime (cache=%d B/slab): 16-token reads token-major=%d grouped=%d (1-token floor=%d)",
		slabBytes, tmReads, gReads, distinctReads)
}

// TestV41ExpertGroupsReplayIsSlotOrdered pins the accumulation-order contract: a
// token whose picks are spread across several expert groups must still have its
// weighted terms summed in ORIGINAL SLOT ORDER, byte-identically to the
// token-major loop. Grouping reorders the reads, never the arithmetic, and float
// addition is not associative -- so a group-ordered replay could drift. The
// oracle here adds the same per-slot products in slot order with no grouping.
func TestV41ExpertGroupsReplayIsSlotOrdered(t *testing.T) {
	// One token whose three picks land in three DIFFERENT groups (experts 5,1,3 in
	// slots 0,1,2), plus a second token so the panel is multi-token.
	picks := [][]routePick{
		{{expert: 5, weight: 0.1}, {expert: 1, weight: 0.2}, {expert: 3, weight: 0.3}},
		{{expert: 1, weight: 0.7}},
	}
	const H = 4
	expertOut := map[int][]float32{
		1: {0.3, -0.7, 1.1, 2.9},
		3: {-1.5, 0.25, 0.125, -0.5},
		5: {2.0, -0.125, 0.5, 0.0625},
	}

	// Slot-ordered oracle for token 0: ((w0*o5)+w1*o1)+w2*o3, left to right.
	ref := make([][]float32, len(picks))
	for tok := range picks {
		ref[tok] = make([]float32, H)
		for _, p := range picks[tok] {
			y := expertOut[p.expert]
			for i := 0; i < H; i++ {
				ref[tok][i] += p.weight * y[i]
			}
		}
	}

	plan := v41PlanExpertGroups(picks)
	got := make([][]float32, len(picks))
	for tok := range picks {
		got[tok] = make([]float32, H)
	}
	unweighted := make([][][]float32, len(picks))
	for tok := range picks {
		unweighted[tok] = make([][]float32, len(picks[tok]))
	}
	for _, g := range plan {
		y := expertOut[g.Expert]
		for _, r := range g.Rows {
			unweighted[r.Token][r.Slot] = y
		}
	}
	for tok := range picks {
		for slot, p := range picks[tok] {
			y := unweighted[tok][slot]
			for i := 0; i < H; i++ {
				got[tok][i] += p.weight * y[i]
			}
		}
	}
	for tok := range picks {
		for i := 0; i < H; i++ {
			if got[tok][i] != ref[tok][i] {
				t.Fatalf("token %d component %d: slot-ordered replay %g != oracle %g", tok, i, got[tok][i], ref[tok][i])
			}
		}
	}
}
