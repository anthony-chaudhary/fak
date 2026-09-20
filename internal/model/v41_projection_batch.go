package model

// v41_projection_batch.go — the #13301 V4.1 prefill projection panel.
//
// v41Layer's attention block applies attn.wq_a and attn.wq_b once per token
// (`for t := 0; t < seq; t++`, v41_forward.go), so a prefill of seq prompt
// tokens re-reads each query-projection weight seq times (GEMV-per-token). This
// adapter hoists those two projections of ONE layer into a single batched
// projection over the whole prompt panel, exactly the reuse the GLM-DSA MLA
// prefill already rides (glm_dsa.go: residentMatMulBatch reads each weight row
// once and reuses it across all seq rows).
//
// It is byte-for-byte the per-token v41ProjMatRows path:
//
//   - f32-manifest weight: residentMatMulBatch runs ONE matMulBatch, whose
//     in-order inner reduction makes Y[t*out+o] bit-identical to the per-token
//     matRows result (parallel.go: matMulBatch contract).
//   - quant-resident weight (Q8/int4/Q4_K/k-quant/GPTQ): residentMatMulBatch
//     falls back to the per-token residentMatRows call, which is exactly
//     v41ProjMatRows's own read.
//
// The adapter is deliberately narrow, per the leaf's gold-plating boundary: the
// query projections ONLY. The per-token norm/RoPE glue stays scalar at the call
// site, the KV projection and the router/shared-expert projections are not
// batched here. A single-row (decode) panel and an unsupported layout take the
// scalar path so decode is never charged a panel primitive.
//
// The panel result is produced into a fresh caller-owned [seq,out] buffer; it
// does not alias any resident store, so no lifetime escapes the call.

import (
	"fmt"
	"sync/atomic"
)

// v41ProjPanelRowChunk bounds the scratch a single panel primitive holds. A
// panel larger than this is projected in row chunks so the transient input panel
// and output panel stay bounded regardless of prompt length. It is a multiple
// that keeps the projection arithmetic identical (each row is independent).
const v41ProjPanelRowChunk = 64

// v41PanelInvocationCount counts how many times a real batched panel primitive
// ran. It is the invocation witness: a test proves the panel path was actually
// taken (presence != invokability) rather than only that the scalar path still
// returns the right bytes. Package-level so the counter needs no new field on
// Model; it is meaningful only under a test.
var v41PanelInvocationCount atomic.Int64

// v41ProjPanel projects a [seq, in] activation panel through the named
// per-layer projection, returning [seq, out] row-major. It reads residency-
// completely through the same store v41ProjMatRows does and fails closed with
// the typed ErrV41ForwardStage when the weight is in no store (#13276), so a
// physical serve reaches a NAMED refusal rather than a panic.
//
// seq <= 1 and the unsupported-layout case (out/in that the resident store does
// not admit for the batched read) stay on the scalar per-token path.
func (m *Model) v41ProjPanel(l int, leaf string, panel []float32, out, in, seq int) ([]float32, error) {
	name := layerName(l, leaf)
	if !m.hasResidentWeight(name) {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	if seq <= 0 {
		return nil, nil
	}
	if len(panel) != seq*in {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: projection panel %s len %d, want seq*in %d", ErrV41ForwardStage, name, len(panel), seq*in))
	}

	y := make([]float32, seq*out)
	// Decode (one row) is never charged a panel primitive: the scalar GEMV is
	// the established decode path and the panel adds nothing at seq==1.
	if seq == 1 {
		row, err := m.v41ProjMatRows(l, leaf, panel, out, in)
		if err != nil {
			return nil, err
		}
		copy(y, row)
		return y, nil
	}

	// Chunked batched panel: one residentMatMulBatch per row chunk. Weight-row
	// reuse is preserved across every row of the chunk, and each output row is
	// bit-identical to the scalar per-token read (residentMatMulBatch contract).
	for lo := 0; lo < seq; lo += v41ProjPanelRowChunk {
		hi := lo + v41ProjPanelRowChunk
		if hi > seq {
			hi = seq
		}
		n := hi - lo
		v41PanelInvocationCount.Add(1)
		batch := m.residentMatMulBatch(name, panel[lo*in:hi*in], out, in, n)
		copy(y[lo*out:hi*out], batch)
	}
	return y, nil
}
