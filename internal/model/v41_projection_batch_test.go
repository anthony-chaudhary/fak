package model

// v41_projection_batch_test.go — the #13301 witness for batched V4.1 attention
// query projections across prefill rows.
//
// The V4.1 forward's attention block applies attn.wq_a and attn.wq_b once per
// token in a `for t := 0; t < seq; t++` loop (v41_forward.go v41Layer), so a
// prefill of seq prompt tokens re-reads every query-projection weight seq times
// (GEMV-per-token). The existing residentMatMulBatch GEMM (qwen35_chunked.go)
// reads each weight row once and reuses it across all seq rows; for an
// f32-resident weight it runs ONE matMulBatch (bit-identical in-order reduction,
// parallel.go), and for a quant-resident weight it falls back to the per-token
// residentMatRows — byte-for-byte the v41ProjMatRows read.
//
// This file pins the four contracts the panel adapter must hold:
//
//  1. INVOCATION: for seq>1 on a supported layout, a real panel primitive runs
//     once for the whole panel instead of one GEMV per row. This is the RED
//     symptom: on the parent commit v41ProjPanel does not exist.
//  2. BYTE-IDENTITY: every batched query row is bit-for-bit the scalar
//     v41ProjMatRows row, on both the f32-manifest and resident k-quant paths.
//  3. SCALAR FALLBACK: seq==1 (decode) and an unsupported layout stay on the
//     per-token path and still produce the exact scalar row.
//  4. CHUNK TAILS: a panel size that is not a multiple of the adapter's row
//     chunk still projects every row correctly.

import (
	"errors"
	"math"
	"testing"
)

// v41PanelCalls reads the adapter's panel-invocation counter, so a test can
// prove a real panel primitive ran (presence != invokability).
func v41PanelCalls(m *Model) int {
	_ = m
	return int(v41PanelInvocationCount.Load())
}

// TestV41ProjectionBatchRunsPanelForPrefill is the #13301 acceptance witness:
// a multi-row (prefill) panel must go through the batched adapter exactly once,
// and every row must be bit-identical to the scalar per-token projection.
func TestV41ProjectionBatchRunsPanelForPrefill(t *testing.T) {
	m := v41ReducedModel(t)
	v41PanelInvocationCount.Store(0)

	const seq = 4
	shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
	out, in := shape[0], shape[1]

	panel := v41SyntheticPanel(seq, in)

	// Scalar reference: one v41ProjMatRows per row (the pre-leaf path).
	want := make([]float32, seq*out)
	for r := 0; r < seq; r++ {
		row, err := m.v41ProjMatRows(0, "attn.wq_a.weight", panel[r*in:(r+1)*in], out, in)
		if err != nil {
			t.Fatalf("scalar v41ProjMatRows row %d error = %v, want nil", r, err)
		}
		copy(want[r*out:(r+1)*out], row)
	}

	got, err := m.v41ProjPanel(0, "attn.wq_a.weight", panel, out, in, seq)
	if err != nil {
		t.Fatalf("v41ProjPanel error = %v, want nil", err)
	}
	if len(got) != seq*out {
		t.Fatalf("v41ProjPanel len = %d, want %d", len(got), seq*out)
	}
	if calls := v41PanelCalls(m); calls != 1 {
		t.Fatalf("v41ProjPanel ran %d panel primitive(s) for a %d-row panel, want exactly 1", calls, seq)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("batched panel[%d] = %v, want byte-identical scalar %v", i, got[i], want[i])
		}
	}
}

// TestV41ProjectionBatchResidentKQuantByteIdentical pins that the panel path is
// byte-identical to the scalar path when the weight is ONLY in the resident
// k-quant store (the streamed Q2_K serve shape). The reduced fixture's published
// query geometry is narrower than a Q2_K super-block, so this installs a
// synthetic [out, in] Q2_K matrix large enough to quantize (in a multiple of
// 256) under the layer's attn.wq_a leaf, exactly the #2150 fixture shape.
func TestV41ProjectionBatchResidentKQuantByteIdentical(t *testing.T) {
	m := v41ReducedModel(t)

	const seq = 3
	const out, in = 1024, 2048
	name := layerName(0, "attn.wq_a.weight")
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(v41ResidentQ2KRaw(out, in), out, in, kindQ2K)

	panel := v41SyntheticPanel(seq, in)

	want := make([]float32, seq*out)
	for r := 0; r < seq; r++ {
		row, err := m.v41ProjMatRows(0, "attn.wq_a.weight", panel[r*in:(r+1)*in], out, in)
		if err != nil {
			t.Fatalf("scalar v41ProjMatRows row %d error = %v, want nil", r, err)
		}
		copy(want[r*out:(r+1)*out], row)
	}

	got, err := m.v41ProjPanel(0, "attn.wq_a.weight", panel, out, in, seq)
	if err != nil {
		t.Fatalf("v41ProjPanel error = %v, want nil", err)
	}
	if len(got) != seq*out {
		t.Fatalf("v41ProjPanel len = %d, want %d", len(got), seq*out)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("resident-panel[%d] = %v, want byte-identical scalar %v", i, got[i], want[i])
		}
	}
}

// TestV41ProjectionBatchScalarFallback pins that a single-row (decode) panel
// stays on the scalar per-token path and returns the exact scalar row: the
// adapter must not spend a panel primitive on a decode step.
func TestV41ProjectionBatchScalarFallback(t *testing.T) {
	m := v41ReducedModel(t)
	v41PanelInvocationCount.Store(0)

	shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
	out, in := shape[0], shape[1]
	row := v41SyntheticPanel(1, in)

	want, err := m.v41ProjMatRows(0, "attn.wq_a.weight", row, out, in)
	if err != nil {
		t.Fatalf("scalar v41ProjMatRows error = %v, want nil", err)
	}
	got, err := m.v41ProjPanel(0, "attn.wq_a.weight", row, out, in, 1)
	if err != nil {
		t.Fatalf("v41ProjPanel error = %v, want nil", err)
	}
	if calls := v41PanelCalls(m); calls != 0 {
		t.Fatalf("v41ProjPanel ran %d panel primitive(s) for a single-row (decode) panel, want 0", calls)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("decode row[%d] = %v, want byte-identical scalar %v", i, got[i], want[i])
		}
	}
}

// TestV41ProjectionBatchChunkTail pins that a panel size that is not a multiple
// of the adapter's row chunk still projects every row, and that the tail rows
// are bit-identical to the scalar path.
func TestV41ProjectionBatchChunkTail(t *testing.T) {
	m := v41ReducedModel(t)

	seq := v41ProjPanelRowChunk + 2 // one full chunk plus a two-row tail
	shape := v41ProjectionShape(t, m, "attn.wq_b.weight")
	out, in := shape[0], shape[1]

	panel := v41SyntheticPanel(seq, in)

	want := make([]float32, seq*out)
	for r := 0; r < seq; r++ {
		row, err := m.v41ProjMatRows(0, "attn.wq_b.weight", panel[r*in:(r+1)*in], out, in)
		if err != nil {
			t.Fatalf("scalar v41ProjMatRows row %d error = %v, want nil", r, err)
		}
		copy(want[r*out:(r+1)*out], row)
	}

	got, err := m.v41ProjPanel(0, "attn.wq_b.weight", panel, out, in, seq)
	if err != nil {
		t.Fatalf("v41ProjPanel error = %v, want nil", err)
	}
	if len(got) != seq*out {
		t.Fatalf("v41ProjPanel len = %d, want %d", len(got), seq*out)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("chunk-tail panel[%d] = %v, want byte-identical scalar %v", i, got[i], want[i])
		}
	}
}

// TestV41ProjectionBatchFailsClosedWhenAbsent pins #13276 across the new seam:
// a projection in no store refuses with a typed ErrV41ForwardStage naming the
// tensor, never a panic through residentMatRows.
func TestV41ProjectionBatchFailsClosedWhenAbsent(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "attn.wq_a.weight")
	delete(m.manifest, name)
	if m.kqw != nil {
		delete(m.kqw, name)
	}

	shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
	out, in := shape[0], shape[1]
	panel := v41SyntheticPanel(2, in)

	_, err := m.v41ProjPanel(0, "attn.wq_a.weight", panel, out, in, 2)
	if err == nil {
		t.Fatalf("v41ProjPanel on an absent tensor returned nil error, want a typed refusal")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("v41ProjPanel error = %v, want errors.Is(err, ErrV41ForwardStage)", err)
	}
}

// v41SyntheticPanel builds a deterministic [rows, in] activation panel.
func v41SyntheticPanel(rows, in int) []float32 {
	panel := make([]float32, rows*in)
	for r := 0; r < rows; r++ {
		for i := 0; i < in; i++ {
			panel[r*in+i] = float32((((r+1)*(i%17))%23)-11) * 0.125
		}
	}
	return panel
}

// TestV41ProjectionBatchForwardInvocation drives a real two-token V4.1 forward
// and pins that the batched query projections actually run in the production
// path (one attn.wq_a panel + one attn.wq_b panel per layer) and that the
// forward still emits finite logits. The byte-identity of each batched row to
// the scalar per-token read is pinned by the tests above.
func TestV41ProjectionBatchForwardInvocation(t *testing.T) {
	m := v41ReducedModel(t)
	v41PanelInvocationCount.Store(0)

	act, err := m.forwardV41([]int{0, 1}, nil)
	if err != nil {
		t.Fatalf("forwardV41 error = %v, want nil", err)
	}
	wantPanels := 2 * m.Cfg.NumLayers // wq_a + wq_b per layer
	if calls := int(v41PanelInvocationCount.Load()); calls != wantPanels {
		t.Fatalf("forwardV41 ran %d panel primitive(s), want %d (wq_a + wq_b per layer)", calls, wantPanels)
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite after the batched query projections", r, i, v)
			}
		}
	}
}
