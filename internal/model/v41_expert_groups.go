package model

import (
	"fmt"
	"math"
)

// v41_expert_groups.go -- bounded grouped execution of the V4.1 prefill MoE
// contraction (#13304).
//
// The historical contraction is TOKEN-MAJOR: for each token, for each routed
// pick, it materializes that expert's w1/w3/w2 triple, contracts the token's
// rmsnorm row through it, weights the result, and moves to the next pick. The
// #13296 layer-scoped f32 cache defers the fault of a repeated (expert,
// projection) only while the interleaved working set fits its bound; under the
// production prefill regime (distinct prompt tokens routing a broad expert union
// whose f32 expansion exceeds the resident bound) token-major order evicts an
// expert the NEXT token routes again, so the panel re-faults the same slabs and
// the resident hit fraction the #13294 first-token seam steers by collapses.
//
// The bounded fix here is NOT a larger cache (that is the #13288 uncharged-f32
// failure mode) and NOT a transient union pre-fault (that materializes the whole
// union at once). It is a GROUPING of the panel's picks by expert:
//
//  1. Build stable (expert, token, original-pick-slot) groups for one bounded
//     panel of tokens. Rows within a group are retained in (token, slot)
//     ascending order, so the replay is deterministic.
//  2. Materialize ONE expert triple, contract every row assigned to it, release
//     it before the next expert. Peak retained expert materialization is one
//     triple, never the panel's union.
//  3. Replay the per-token weighted sums into each token's row in ORIGINAL slot
//     order using the pick's own weight, so the arithmetic is byte-identical to
//     the token-major stream (duplicate experts and repeated routes apply their
//     weight exactly once per pick).
//
// The output is keyed by (token, slot), never by expert, so two picks that route
// the same expert remain two independent contributions -- grouping reorders the
// READS, it never merges a token's arithmetic.

// v41ExpertRow is one routed pick's assignment inside a group: the token whose
// rmsnorm row the expert contracts and the pick's original slot index, which the
// replay uses to apply the pick's own weight in the token's original slot order.
type v41ExpertRow struct {
	Token int
	Slot  int
}

// v41ExpertGroup is one expert's assigned rows in first-touch order. The expert
// is materialized once for the group and released before the next group.
type v41ExpertGroup struct {
	Expert int
	Rows   []v41ExpertRow
}

// v41PlanExpertGroups builds the stable expert-major grouping of a bounded
// panel's routed picks. `picks[t][k]` is token t's k-th pick in original slot
// order. The returned groups are ordered by FIRST TOUCH (the order the token-major
// loop first reaches each expert), so the grouped plan makes the same first
// materialization decision the token-major stream would; within a group rows are
// (token, slot) ascending.
//
// Purity: the planner reads picks and allocates only the grouping arrays; it
// materializes nothing and mutates no model state, so the caller controls the
// bounded allocation exactly. An empty panel or empty per-token pick set yields
// no groups, so the seam is inert off the multi-token path.
func v41PlanExpertGroups(picks [][]routePick) []v41ExpertGroup {
	total := 0
	for _, perToken := range picks {
		total += len(perToken)
	}
	if total == 0 {
		return nil
	}
	groups := make([]v41ExpertGroup, 0, total)
	index := make(map[int]int, total)
	for tok, perToken := range picks {
		for slot := range perToken {
			expert := perToken[slot].expert
			gi, ok := index[expert]
			if !ok {
				gi = len(groups)
				groups = append(groups, v41ExpertGroup{Expert: expert})
				index[expert] = gi
			}
			groups[gi].Rows = append(groups[gi].Rows, v41ExpertRow{Token: tok, Slot: slot})
		}
	}
	// Rows are appended in token-major then slot order, so each group is already
	// (token, slot) ascending; assert the invariant cheaply by construction rather
	// than sorting (the append order IS the sort order).
	return groups
}

// v41ExpertGroupScratchBytes reports the exact grouping-array bytes a panel of
// `rows` routed picks needs, so the serve plan (leaf #13300's transient
// estimator) can charge it before allocation. It is the v41ExpertRow payload
// (two ints) plus the per-row share of the group headers and the first-touch
// index entry, rounded up conservatively. Overflow-checked: an unrepresentable
// size returns math.MaxInt64 so the estimate can only refuse MORE.
func v41ExpertGroupScratchBytes(rows int) int64 {
	if rows <= 0 {
		return 0
	}
	const perRow = int64(2 * 8) // two int64 fields (Token, Slot)
	const perRowOverhead = int64(2 * 8)
	const fixedHeader = int64(4 * 8)
	if int64(rows) > (math.MaxInt64-fixedHeader)/(perRow+perRowOverhead) {
		return math.MaxInt64
	}
	return fixedHeader + int64(rows)*(perRow+perRowOverhead)
}

// v41ForceTokenMajor, when true, disables the #13304 grouped contraction so a
// test can drive the historical token-major stream on the SAME model and compare
// the two paths' outputs. It is never set in production (the zero value keeps the
// grouped path on for multi-token panels) and exists only so the equivalence
// witness can assert the grouping is output-identical, not merely
// oracle-consistent.
var v41ForceTokenMajor bool

// v41TestLayerCacheBudgetOverride, when >0, replaces the layer-scoped expert
// cache budget a test's forward allocates. It lets the thrash witness shrink the
// cache below the panel's routed union so the token-major path's dependency on
// cache capacity is exercised at real reduced geometry (where the 810 MiB
// production default would hold the whole union and mask the difference). The
// zero value keeps the production budget; no non-test path sets it.
var v41TestLayerCacheBudgetOverride int64

// v41ContractRoutedGrouped is the #13304 expert-major routed contraction for a
// multi-token panel. It plans stable expert groups over `perTokenPicks`, then for
// each group materializes the expert triple ONCE through the same
// v41ExpertTripleInto the token-major path uses (so residency, tier faults and
// the #13296 layer cache are shared, not forked), contracts every assigned row
// read-only through v41SwiGLU, and stores the row's UNWEIGHTED output keyed by
// (token, slot). After every group has been released it replays the per-token
// weighted sums in ORIGINAL SLOT ORDER, so the float accumulation order is
// byte-identical to the token-major loop (grouping reorders the READS, never a
// token's arithmetic). `out[t]` receives token t's routed sum.
//
// #13511: before falling back to the host triple, each row is offered to the
// session's optional device gate/up seam (st.expertGateUp) exactly as the
// token-major arm does (#13358). The engine choice is per (expert, encoding), so
// it is uniform across a group, but the gate/up MatMul consumes the ROW's own
// rmsnorm input, so the seam is invoked per row. A handled row keeps the gate/up
// f32 weights off the host and runs the down contraction over the device-produced
// intermediate; a declined row falls through to the historical host triple
// byte-for-byte. A nil `st` (Model.Forward, or no device backend) leaves the
// whole panel byte-identical to the pre-#13511 grouped path.
//
// The per-contraction telemetry (#13299) is noted once per contracted row exactly
// as the token-major loop noted it per pick, so the fault/dequant-vs-contraction
// attribution is unchanged in bucket counts. Any materialization error aborts
// before the weighted replay, so a partial panel never reaches `out`.
func (m *Model) v41ContractRoutedGrouped(l int, x [][]float32, perTokenPicks [][]routePick, scratch *v41ProjScratch, ffnNorm []float32, eps float32, cfg Config, out [][]float32, st *v41ForwardState) error {
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	groups := v41PlanExpertGroups(perTokenPicks)
	// One rmsnorm row per token, computed once and reused by every group that
	// contracts a row of that token (the token-major path recomputed it per pick;
	// rmsnormCfg is pure, so the values are byte-identical).
	xnByToken := make([][]float32, len(perTokenPicks))
	for t := range perTokenPicks {
		xnByToken[t] = rmsnormCfg(x[t], ffnNorm, eps, cfg)
		out[t] = make([]float32, H)
	}
	// unweighted[token][slot] holds the contracted expert output BEFORE the pick
	// weight is applied, so the replay below can apply weights in original slot
	// order regardless of the order the groups materialized experts in.
	unweighted := make([][][]float32, len(perTokenPicks))
	for t, picks := range perTokenPicks {
		unweighted[t] = make([][]float32, len(picks))
	}
	// tripleReady guards lazy materialization of the group's host triple: when the
	// device seam handles every row, the gate/up f32 weights are never materialized
	// on the host at all (#13358's whole point). The triple is materialized once
	// per group, on the first row the device seam declines, and reused for the
	// remaining host rows -- preserving #13304's one-materialization-per-group bound.
	for _, g := range groups {
		stem := "ffn.experts." + itoa(g.Expert)
		var (
			w1, w3, w2  []float32
			tripleReady bool
		)
		for _, row := range g.Rows {
			xn := xnByToken[row.Token]
			// #13511: offer the row to the session's device gate/up seam first, the
			// same seam the token-major arm consults (#13358). A handled result
			// returns the I-wide fused intermediate from the backend, so the gate/up
			// f32 weights are never materialized on the host and only the existing
			// host down contraction runs over it (resolved per row through the same
			// residency-complete path the token-major arm uses). A selected device
			// failure must surface, never be swallowed as a decline.
			if st != nil && st.expertGateUp != nil {
				h, outcome, gerr := st.expertGateUp(l, stem, xn)
				switch outcome {
				case v41GateUpError:
					return v41StageErr(v41StageMoE, l, gerr)
				case v41GateUpHandled:
					wd, err := m.hostExpertDown(l, stem, scratch)
					if err != nil {
						return err
					}
					contractOpen := m.v41NowNanos()
					y := matRows(wd, h, H, I)
					if contractOpen != 0 {
						m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
					} else {
						m.v41NoteExpertContraction()
					}
					unweighted[row.Token][row.Slot] = y
					continue
				}
			}
			// The host arm: materialize the group's triple once (lazily, so a
			// device-handled group never touches the gate/up f32 weights), then
			// contract the row through the historical host SwiGLU. A decline (or a
			// nil state) therefore keeps the pre-#13511 grouped path byte-for-byte.
			if !tripleReady {
				var err error
				w1, w3, w2, err = m.v41ExpertTripleInto(l, stem, scratch)
				if err != nil {
					return err
				}
				tripleReady = true
			}
			// Same #13299 attribution as the token-major loop: one note per
			// contracted (token, pick) row, so the ledger's bucket counts match.
			contractOpen := m.v41NowNanos()
			y := v41SwiGLU(w1, w3, w2, xn, I, H, cfg)
			if contractOpen != 0 {
				m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
			} else {
				m.v41NoteExpertContraction()
			}
			unweighted[row.Token][row.Slot] = y
		}
	}
	// Weighted replay in ORIGINAL SLOT ORDER, byte-identical to the token-major
	// accumulation `routed[i] += pick.weight * y[i]` per slot.
	for t, picks := range perTokenPicks {
		dst := out[t]
		for slot, pick := range picks {
			y := unweighted[t][slot]
			if y == nil {
				// Unreachable for a well-formed panel: every pick is grouped. Fail
				// closed rather than silently dropping a routed contribution.
				return fmt.Errorf("v41 grouped contraction: pick (token=%d,slot=%d,expert=%d) was not materialized",
					t, slot, pick.expert)
			}
			for i := range dst {
				dst[i] += pick.weight * y[i]
			}
		}
	}
	return nil
}
