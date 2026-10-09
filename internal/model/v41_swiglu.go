package model

import (
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41SwiGLU is the configured SwiGLU expert/sub-layer: down(silu(clamp(w1 x)) * clamp(w3 x)).
// v41SharedExpertSwiGLU is the streaming-safe shared-expert SwiGLU: it applies the
// three shared-expert leaves through v41ProjMatRows (resident store form, bounded
// block scratch) instead of materializing each whole f32 weight, so the shared
// expert adds no uncharged per-layer f32 expansion (#2150). Numerically identical
// to v41SwiGLU over the same weights on the f32-manifest path.
func (m *Model) v41SharedExpertSwiGLU(l int, xn []float32, cfg Config) ([]float32, error) {
	return m.v41SharedExpertSwiGLUWithProjection(l, xn, cfg, nil)
}

func (m *Model) v41SharedExpertSwiGLUWithProjection(l int, xn []float32, cfg Config, project v41DenseProjectionFunc) ([]float32, error) {
	return m.v41SharedExpertSwiGLUWithActivation(l, xn, cfg, project, nil)
}

func (m *Model) v41SharedExpertSwiGLUWithActivation(l int, xn []float32, cfg Config, project v41DenseProjectionFunc, activate v41SharedActivationFunc) ([]float32, error) {
	I, H := cfg.MoEIntermediateSize, cfg.HiddenSize
	h1, err := m.v41ProjMatRowsWithProjection(l, "ffn.shared_experts.w1.weight", xn, I, H, project)
	if err != nil {
		return nil, err
	}
	h3, err := m.v41ProjMatRowsWithProjection(l, "ffn.shared_experts.w3.weight", xn, I, H, project)
	if err != nil {
		return nil, err
	}
	if activate != nil && !cfg.ActGeluTanh && !cfg.ActGeluErf {
		values, outcome, err := activate(l, h1, h3, float32(cfg.SwigluLimit))
		switch outcome {
		case v41ProjectionHandled:
			if err == nil && len(values) != I {
				err = errV41ProjectionResult
			}
			if err == nil {
				for _, v := range values {
					if !finite32(v) {
						err = errV41ProjectionResult
						break
					}
				}
			}
			if err != nil {
				return nil, v41ProjectionOperationErr(l, "ffn.shared_experts.activation", err)
			}
			return m.v41ProjMatRowsWithProjection(l, "ffn.shared_experts.w2.weight", values, H, I, project)
		case v41ProjectionError:
			return nil, v41ProjectionOperationErr(l, "ffn.shared_experts.activation", err)
		case v41ProjectionDeclined:
		default:
			return nil, v41ProjectionOperationErr(l, "ffn.shared_experts.activation", errV41ProjectionResult)
		}
	}
	clampSwiGLUProjections(h1, h3, float32(cfg.SwigluLimit))
	return ffn.Gated(h1, h3, func(v float32) float32 { return act(v, cfg) }, func(activated []float32) ([]float32, error) {
		return m.v41ProjMatRowsWithProjection(l, "ffn.shared_experts.w2.weight", activated, H, I, project)
	})
}

// v41SwiGLUParallelCalls counts V4.1 routed-expert contractions dispatched through
// the row-parallel kernel while the test-only witness is enabled. Production pays
// zero: the counter is touched only when v41SwiGLUWitnessOn is true, mirroring the
// hostBatchWitnessOn idiom in moe_host_batch.go.
var (
	v41SwiGLUParallelCalls int64
	v41SwiGLUWitnessOn     bool
)

// enableV41SwiGLUWitness turns the parallel-dispatch witness ON for a test and
// zeroes the counter. Production never calls it.
func enableV41SwiGLUWitness() {
	v41SwiGLUWitnessOn = true
	atomic.StoreInt64(&v41SwiGLUParallelCalls, 0)
}

// v41SwiGLUDispatches reports how many large routed-expert contractions ran on the
// row-parallel kernel since enableV41SwiGLUWitness.
func v41SwiGLUDispatches() int64 { return atomic.LoadInt64(&v41SwiGLUParallelCalls) }

// v41ExpertTripleRetentions counts how many expert triples v41ExpertTripleInto
// copied back into the layer-scoped cache while the test-only retention witness is
// enabled. Production pays zero: the counter is touched only when
// v41RetentionWitnessOn is true, mirroring the v41SwiGLUWitnessOn idiom above.
// The grouped expert-major prefill (#13304) must retain ZERO triples (it re-reads
// none), while the token-major stream retains one per distinct materialization --
// which is why the witness distinguishes the two arms (fak#13697).
var (
	v41ExpertTripleRetentions int64
	v41RetentionWitnessOn     bool
)

// enableV41RetentionWitness turns the triple-retention witness ON for a test and
// zeroes the counter. Production never calls it.
func enableV41RetentionWitness() {
	v41RetentionWitnessOn = true
	atomic.StoreInt64(&v41ExpertTripleRetentions, 0)
}

// v41Retentions reports how many expert triples were copied into the layer cache
// since enableV41RetentionWitness.
func v41Retentions() int64 { return atomic.LoadInt64(&v41ExpertTripleRetentions) }

// v41NoteExpertTripleRetention records one triple retention under the test-only
// witness. Inert in production.
func v41NoteExpertTripleRetention() {
	if v41RetentionWitnessOn {
		atomic.AddInt64(&v41ExpertTripleRetentions, 1)
	}
}

// v41SwiGLU is the V4.1 routed-expert contraction: down(silu(clamp(w1 x)) *
// clamp(w3 x)). It is the package-level serial form, and it stays the reference
// every adapter witness compares against (ffn_composition_test.go,
// v41_swiglu_limit_test.go). See v41SwiGLUParallel for the bulk-prefill twin.
func v41SwiGLU(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	clampSwiGLUProjections(h1, h3, float32(cfg.SwigluLimit))
	y, err := ffn.Gated(h1, h3, func(v float32) float32 { return act(v, cfg) }, func(activated []float32) ([]float32, error) {
		return matRows(w2, activated, H, I), nil
	})
	if err != nil {
		panic(err)
	}
	return y
}

// v41SwiGLUParallel is the ROW-PARALLEL twin of v41SwiGLU, used for the BULK
// prefill routed-expert contraction (the largest single attributed bucket of the
// physical strix3 30-token prefill: contraction=57.381 s of 265.48 s, fak#13294).
//
// It is byte-for-byte v41SwiGLU's arithmetic: the only difference is the kernel
// that reduces each output row, and parMatRows splits OUTPUT ROWS across cores
// while every row keeps the identical in-order mathx.FDot reduction — the contract
// pinned by parallel.go and TestParallelMatchesSerial. Below parThreshold the
// kernel is the historical matRows, so a reduced-geometry expert is unchanged.
//
// The gate is the flat element count I*H ONE projection moves, deliberately shared
// by all three projections: at the published V4.1 expert geometry (I=2304, H=5120)
// all three are 11.8M elements — far above parThreshold — so gate/up AND down all
// ride the parallel kernel, whereas gating each projection on its own out*in would
// leave the down projection on the serial path at some geometries.
//
// CALLERS: the multi-token panel contraction (v41ContractRoutedGrouped, only
// reached for seq > 1) and the multi-token token-major arm. The one-position
// incremental decode seam (v41LayerStep / prefillV41Suffix via forwardV41Step)
// keeps the package-level serial v41SwiGLU, because at seq == 1 the three tiny
// expert GEMVs per pick would pay the parFor dispatch barrier ~3*k*40 times a
// token — a pure cost with no bandwidth to reclaim.
func v41SwiGLUParallel(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	proj := matRows
	if I*H >= parThreshold {
		if v41SwiGLUWitnessOn {
			atomic.AddInt64(&v41SwiGLUParallelCalls, 1)
		}
		proj = parMatRows
	}
	h1 := proj(w1, xn, I, H)
	h3 := proj(w3, xn, I, H)
	clampSwiGLUProjections(h1, h3, float32(cfg.SwigluLimit))
	y, err := ffn.Gated(h1, h3, func(v float32) float32 { return act(v, cfg) }, func(activated []float32) ([]float32, error) {
		return proj(w2, activated, H, I), nil
	})
	if err != nil {
		panic(err)
	}
	return y
}
