// Package dsparity is the pure, OFFLINE parity-harness SPECIFICATION for future
// DeepSeek-V4 native kernels (#3021, under the DeepSeek V4 support program #3006;
// sibling of the docs/deepseek/*.md and docs/notes/DEEPSEEK-V4-*.md plan notes).
//
// DeepSeek ships deterministic, BATCH-INVARIANT kernel libraries so a decode is
// bit-reproducible regardless of how requests are batched (V4 technical report,
// https://arxiv.org/html/2606.19348v1). V4's fused mixed-precision paths — the
// heavily-compressed attention (HCA/mHC) branch, the MoE-overlap dispatch, and the
// lightning-indexer sparse-attention top-k selector — are exactly the places where
// a naive kernel silently loses determinism (reduction-order drift, expert-permute
// races, tie-break wobble). This package encodes the PARITY ROWS and their expected
// FIELDS that any such kernel must satisfy BEFORE a perf claim can close.
//
// It is deliberately dependency-free (stdlib only, no cross-package imports) so the
// first witness needs NO real weights and no GPU: `go test ./internal/dsparity/`
// runs the schema + a synthetic order-invariance demonstration entirely offline. The
// rows that need a real native kernel are labeled Witness == WitnessHostGated; the
// rows a pure Go / synthetic fixture can already exercise are WitnessOfflineSynthetic.
//
// CLAIM DISCIPLINE (the load-bearing rule, mirrored from internal/deepseekbench and
// internal/gateway/deepseek_budget.go): no perf ticket closes with only "faster".
// A speed delta is inadmissible until the matching parity row here is witnessed —
// "faster" without parity is a regression that has not been caught yet.
package dsparity

import "fmt"

// InvarianceAxis is the CLOSED set of invariance axes the issue enumerates. Every
// parity row is anchored to exactly one; the four required axes must all be covered.
type InvarianceAxis string

const (
	// AxisBatchSize — same prompt decoded under batch sizes 1 / N / M must agree.
	AxisBatchSize InvarianceAxis = "batch-size"
	// AxisCacheHit — same prefix decoded with and without a prompt-cache hit must agree.
	AxisCacheHit InvarianceAxis = "cache-hit"
	// AxisRequestOrder — same expert routing under different request ordering must agree.
	AxisRequestOrder InvarianceAxis = "request-order"
	// AxisSeed — same sparse-attention top-k selection under a fixed seed must agree.
	AxisSeed InvarianceAxis = "seed"
)

// validAxis is the closed membership set; anything else is a spec bug.
var validAxis = map[InvarianceAxis]bool{
	AxisBatchSize:    true,
	AxisCacheHit:     true,
	AxisRequestOrder: true,
	AxisSeed:         true,
}

// RequiredAxes is the four-axis floor the issue's acceptance requires the row table
// to cover. TestAllRequiredAxesCovered fails if any is missing.
func RequiredAxes() []InvarianceAxis {
	return []InvarianceAxis{AxisBatchSize, AxisCacheHit, AxisRequestOrder, AxisSeed}
}

// ToleranceClass is the CLOSED set of comparison rules. Bitwise is the default and
// the goal (batch-invariant kernels make it achievable); the FP4/FP8 bounded classes
// exist only for genuinely mixed-precision fused paths where reduction associativity
// cannot be pinned bit-for-bit, and even then the bound must be documented, not vague.
type ToleranceClass string

const (
	// ToleranceBitwise — must match bit-for-bit; MaxAbsTol and MaxRelTol are 0.
	ToleranceBitwise ToleranceClass = "bitwise"
	// ToleranceFP8Bounded — bounded abs/rel tolerance for an FP8 accumulation path.
	ToleranceFP8Bounded ToleranceClass = "fp8-bounded"
	// ToleranceFP4Bounded — bounded abs/rel tolerance for an FP4 expert/indexer path.
	ToleranceFP4Bounded ToleranceClass = "fp4-bounded"
)

var validTolerance = map[ToleranceClass]bool{
	ToleranceBitwise:    true,
	ToleranceFP8Bounded: true,
	ToleranceFP4Bounded: true,
}

// bounded reports whether a tolerance class permits (indeed requires) a non-zero tol.
func (t ToleranceClass) bounded() bool {
	return t == ToleranceFP8Bounded || t == ToleranceFP4Bounded
}

// Witness is the CLOSED set of how a row can be witnessed. The whole point of the
// issue is that the FIRST witness needs no weights, so at least one offline row must
// exist per axis; the native-kernel numeric rows are host-gated until a GPU runs them.
type Witness string

const (
	// WitnessOfflineSynthetic — witnessable now with a pure Go / synthetic fixture, no weights, no GPU.
	WitnessOfflineSynthetic Witness = "offline-synthetic"
	// WitnessHostGated — needs a real native V4 kernel on a GPU host to witness the numeric parity.
	WitnessHostGated Witness = "host-gated"
)

var validWitness = map[Witness]bool{
	WitnessOfflineSynthetic: true,
	WitnessHostGated:        true,
}

// ParityRow is ONE parity assertion and its expected FIELDS (the issue's acceptance
// criterion 1: "a test fixture defines the parity ROWS and expected FIELDS"). The
// JSON tags are the locked schema RequiredFields() pins.
type ParityRow struct {
	// ID is the stable row identifier (unique within the table).
	ID string `json:"id"`
	// Axis is the single invariance axis this row anchors to.
	Axis InvarianceAxis `json:"axis"`
	// Kernel names the V4 fused path under test (fused-mhc | moe-overlap | sparse-attention | lightning-indexer).
	Kernel string `json:"kernel"`
	// CompareField is the observable held to parity across Variants
	// (logits | next-token-id | topk-indices | expert-routing | index-digest).
	CompareField string `json:"compare_field"`
	// Variants are the >= 2 conditions decoded and required to agree (e.g. batch=1 / batch=N / batch=M).
	Variants []string `json:"variants"`
	// Tolerance is the comparison rule for CompareField.
	Tolerance ToleranceClass `json:"tolerance"`
	// MaxAbsTol / MaxRelTol bound a non-bitwise comparison; both 0 for a bitwise row.
	MaxAbsTol float64 `json:"max_abs_tol"`
	MaxRelTol float64 `json:"max_rel_tol"`
	// Witness is offline-synthetic (witnessable now) or host-gated (needs a GPU kernel).
	Witness Witness `json:"witness"`
	// FakSeam is the nearest REAL fak seam this row maps onto (path or proposed), so the
	// harness is grounded on actual code rather than an invented API.
	FakSeam string `json:"fak_seam"`
	// Rationale is the one-line reason this invariance is load-bearing for V4.
	Rationale string `json:"rationale"`
}

// RequiredFields is the locked JSON key set every emitted row MUST carry. The
// field-lock test marshals a row and asserts these keys are present and no others,
// so the acceptance "expected FIELDS" list can never silently drift.
func RequiredFields() []string {
	return []string{
		"id", "axis", "kernel", "compare_field", "variants",
		"tolerance", "max_abs_tol", "max_rel_tol",
		"witness", "fak_seam", "rationale",
	}
}

// Invariant: DeepSeek parity checks are fail-closed and deterministic.
// All parity rows must strictly validate against closed axes and tolerance sets;
// any unmodeled axis, invalid tolerance class, or missing seam definition
// fails closed with a structured error before kernel comparisons can run.
// Guard: bitwise comparisons require zero tolerances, while bounded modes
// strictly require positive tolerances.
//
// Validate checks one row against the closed vocabularies and the internal
// consistency rules (bitwise <=> zero tolerance; bounded <=> positive tolerance;
// >= 2 variants). It returns a descriptive error so a table edit that violates the
// spec fails loud with the offending id.
func (r ParityRow) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("dsparity: row has empty id")
	}
	if !validAxis[r.Axis] {
		return fmt.Errorf("dsparity: row %q has off-vocabulary axis %q", r.ID, r.Axis)
	}
	if !validTolerance[r.Tolerance] {
		return fmt.Errorf("dsparity: row %q has off-vocabulary tolerance %q", r.ID, r.Tolerance)
	}
	if !validWitness[r.Witness] {
		return fmt.Errorf("dsparity: row %q has off-vocabulary witness %q", r.ID, r.Witness)
	}
	if len(r.Variants) < 2 {
		return fmt.Errorf("dsparity: row %q needs >= 2 variants, got %d", r.ID, len(r.Variants))
	}
	if r.Kernel == "" || r.CompareField == "" || r.FakSeam == "" {
		return fmt.Errorf("dsparity: row %q missing kernel/compare_field/fak_seam", r.ID)
	}
	if r.Tolerance == ToleranceBitwise {
		if r.MaxAbsTol != 0 || r.MaxRelTol != 0 {
			return fmt.Errorf("dsparity: bitwise row %q must have zero tolerance, got abs=%g rel=%g", r.ID, r.MaxAbsTol, r.MaxRelTol)
		}
	}
	if r.Tolerance.bounded() {
		if r.MaxAbsTol <= 0 && r.MaxRelTol <= 0 {
			return fmt.Errorf("dsparity: bounded row %q must document a positive tolerance", r.ID)
		}
	}
	return nil
}

// Rows is the parity-harness table. It is PURE DATA: the concrete parity rows every
// future V4 native kernel must satisfy, each anchored to a real fak seam. All four
// required axes are covered, bitwise is preferred, and every axis has at least one
// offline-synthetic row so the first witness needs no weights.
//
// Seam references are the ones grounded in this checkout on 2026-07-09:
//   - internal/model/dsa_index.go:66  dsaTopKIndices     (sparse top-k selector, stable tie-break)
//   - internal/model/dsa_index.go:20  dsaIndexScores     (lightning-indexer relu-scaled dot)
//   - internal/model/dsa_index.go:232 dsaIndexDigest     (sha256 over selected indices)
//   - internal/model/dsa_index.go:208 dsaIndexShare      (every-4-layer index reuse contract)
//   - internal/gateway/deepseek_budget.go                (KV budget; cache-hit prefix accounting)
//   - internal/agent (Usage CachedPromptTokens/UncachedPromptTokens; reasoning-content preservation)
func Rows() []ParityRow {
	return []ParityRow{
		// ---- Axis: batch-size (same prompt at batch 1 / N / M) ----
		{
			ID:           "batch/indexer-scores-1-N-M",
			Axis:         AxisBatchSize,
			Kernel:       "lightning-indexer",
			CompareField: "index-digest",
			Variants:     []string{"batch=1", "batch=8", "batch=64"},
			Tolerance:    ToleranceBitwise,
			Witness:      WitnessOfflineSynthetic,
			FakSeam:      "internal/model/dsa_index.go:20 dsaIndexScores + :232 dsaIndexDigest",
			Rationale:    "selected-index set must not move with batch size; the digest makes drift a bit-diff.",
		},
		{
			ID:           "batch/hca-logits-1-N-M",
			Axis:         AxisBatchSize,
			Kernel:       "fused-mhc",
			CompareField: "logits",
			Variants:     []string{"batch=1", "batch=8", "batch=64"},
			Tolerance:    ToleranceFP8Bounded,
			MaxAbsTol:    1e-3,
			MaxRelTol:    1e-3,
			Witness:      WitnessHostGated,
			FakSeam:      "proposed: HCA (rate-128) kvLayout impl over internal/model/kvlayout.go:28 kvLayout",
			Rationale:    "the fused heavily-compressed-attention accumulation is the classic batch-variant reduction; FP8 accum gets a documented bound, not a free pass.",
		},
		// ---- Axis: cache-hit (same prefix with and without a cache hit) ----
		{
			ID:           "cache/prefix-next-token",
			Axis:         AxisCacheHit,
			Kernel:       "fused-mhc",
			CompareField: "next-token-id",
			Variants:     []string{"cold-prefix", "warm-cache-hit"},
			Tolerance:    ToleranceBitwise,
			Witness:      WitnessOfflineSynthetic,
			FakSeam:      "internal/agent Usage CachedPromptTokens/UncachedPromptTokens; internal/gateway/deepseek_pricing.go hit/miss counters",
			Rationale:    "a cache hit is a compute short-cut, never a semantics change; the greedy next token must be identical to the cold recompute.",
		},
		{
			ID:           "cache/prefix-logits-bounded",
			Axis:         AxisCacheHit,
			Kernel:       "fused-mhc",
			CompareField: "logits",
			Variants:     []string{"cold-prefix", "warm-cache-hit"},
			Tolerance:    ToleranceFP8Bounded,
			MaxAbsTol:    5e-4,
			MaxRelTol:    5e-4,
			Witness:      WitnessHostGated,
			FakSeam:      "internal/gateway/deepseek_budget.go (KV budget); proposed cached-vs-recompute logit probe",
			Rationale:    "where a cached compressed-KV path cannot be pinned bitwise vs recompute, the logit delta must stay inside a documented FP8 bound.",
		},
		// ---- Axis: request-order (expert routing invariant to request permutation) ----
		{
			ID:           "order/expert-routing-permutation",
			Axis:         AxisRequestOrder,
			Kernel:       "moe-overlap",
			CompareField: "expert-routing",
			Variants:     []string{"order=identity", "order=reversed", "order=shuffled"},
			Tolerance:    ToleranceBitwise,
			Witness:      WitnessOfflineSynthetic,
			FakSeam:      "proposed: MoE dispatch over internal/ggufload/gguf_glm_tensors.go router (ffn_gate_inp + exp_probs_b); see DEEPSEEK-V4-MOE-DISPATCH-BASELINE note",
			Rationale:    "the MoE-overlap kernel batches experts across concurrent requests; a per-request top-k routing decision must be a pure function of that request, never of its neighbours' arrival order.",
		},
		{
			ID:           "order/expert-output-bounded",
			Axis:         AxisRequestOrder,
			Kernel:       "moe-overlap",
			CompareField: "logits",
			Variants:     []string{"order=identity", "order=shuffled"},
			Tolerance:    ToleranceFP4Bounded,
			MaxAbsTol:    2e-2,
			MaxRelTol:    2e-2,
			Witness:      WitnessHostGated,
			FakSeam:      "proposed: FP4 expert GEMM (DEEPSEEK-V4-FP4-QUANT-PLAN note); experts are FP4 in the V4 checkpoint",
			Rationale:    "even with identical routing, the FP4 expert GEMM's grouped accumulation may not be bit-stable under request-permuted batching; the drift must live inside a documented FP4 bound.",
		},
		// ---- Axis: seed (sparse top-k selector deterministic under a fixed seed) ----
		{
			ID:           "seed/topk-selection-fixed",
			Axis:         AxisSeed,
			Kernel:       "sparse-attention",
			CompareField: "topk-indices",
			Variants:     []string{"seed=1-runA", "seed=1-runB"},
			Tolerance:    ToleranceBitwise,
			Witness:      WitnessOfflineSynthetic,
			FakSeam:      "internal/model/dsa_index.go:66 dsaTopKIndices (stable tie-break: score desc, then key position asc)",
			Rationale:    "top-k over sparse scores is tie-break sensitive; a fixed seed must reproduce the exact key-position set, or downstream attention silently diverges.",
		},
		{
			ID:           "seed/index-share-layers",
			Axis:         AxisSeed,
			Kernel:       "sparse-attention",
			CompareField: "index-digest",
			Variants:     []string{"seed=7-runA", "seed=7-runB"},
			Tolerance:    ToleranceBitwise,
			Witness:      WitnessOfflineSynthetic,
			FakSeam:      "internal/model/dsa_index.go:208 dsaIndexShare (full layer computes, shared layers reuse prior top-k)",
			Rationale:    "the every-4-layer index-share contract must be seed-stable end to end, so a shared layer reuses exactly the full layer's decision.",
		},
	}
}

// CoveredAxes returns the distinct axes present in the row table.
func CoveredAxes() map[InvarianceAxis]bool {
	seen := make(map[InvarianceAxis]bool)
	for _, r := range Rows() {
		seen[r.Axis] = true
	}
	return seen
}

// stableTopK is the offline, dependency-free demonstration of the request-order /
// seed invariance the harness asserts: it selects the top-k key positions from
// synthetic scores using the SAME tie-break rule internal/model/dsa_index.go's
// dsaTopKIndices uses (score descending, then key position ascending). Because the
// tie-break is total and deterministic, the result is invariant to the order in
// which (position, score) candidates are supplied — which is exactly what the
// synthetic request-order / seed rows exercise with no weights and no GPU.
func stableTopK(positions []int, scores []float64, k int) []int {
	type cand struct {
		pos   int
		score float64
	}
	cands := make([]cand, 0, len(positions))
	for i := range positions {
		cands = append(cands, cand{pos: positions[i], score: scores[i]})
	}
	// insertion sort with the total order — stable and allocation-light for the
	// small candidate sets a synthetic fixture uses.
	for i := 1; i < len(cands); i++ {
		for j := i; j > 0; j-- {
			a, b := cands[j-1], cands[j]
			less := a.score < b.score || (a.score == b.score && a.pos > b.pos)
			if !less {
				break
			}
			cands[j-1], cands[j] = b, a
		}
	}
	if k > len(cands) {
		k = len(cands)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = cands[i].pos
	}
	return out
}
