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
// path (query A/B, KV and router panels per layer) and that the
// forward still emits finite logits. The byte-identity of each batched row to
// the scalar per-token read is pinned by the tests above.
func TestV41ProjectionBatchForwardInvocation(t *testing.T) {
	m := v41ReducedModel(t)
	v41PanelInvocationCount.Store(0)

	act, err := m.forwardV41([]int{0, 1}, nil)
	if err != nil {
		t.Fatalf("forwardV41 error = %v, want nil", err)
	}
	wantPanels := 4 * m.Cfg.NumLayers
	if calls := int(v41PanelInvocationCount.Load()); calls != wantPanels {
		t.Fatalf("forwardV41 ran %d panel primitive(s), want %d (query A/B, KV and router per layer)", calls, wantPanels)
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite after the batched query projections", r, i, v)
			}
		}
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41ProjectionBatchKVRouterChunkCoverage(t *testing.T) {
	v41CheckKVRouterForwardPanels(t, v41ReducedModel(t), 65, []int{64, 1}, false)
}

// fak-test:runtime fast
func TestV41ProjectionBatchKVRouterScalarForward(t *testing.T) {
	v41PanelInvocationCount.Store(0)
	v41CheckKVRouterForwardPanels(t, v41ReducedModel(t), 1, []int{1}, false)
	if calls := v41PanelInvocationCount.Load(); calls != 0 {
		t.Fatalf("single-token forward invoked %d panel primitives", calls)
	}
}

// fak-test:runtime slow est=30s lane=default
func TestV41ProjectionBatchKVRouterFullRank(t *testing.T) {
	v41CheckKVRouterForwardPanels(t, v41PinnedSlateFixture(t, false), 2, []int{2}, true)
}

func v41CheckKVRouterForwardPanels(t *testing.T, m *Model, seq int, wantChunks []int, full bool) {
	t.Helper()
	ids := make([]int, seq)
	for i := range ids {
		ids[i] = i % m.Cfg.VocabSize
	}
	type call struct {
		leaf             string
		layer, rows, out int
		input, snapshot  []float32
	}
	var calls []call
	project := func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if leaf != "attn.wkv.weight" && leaf != "ffn.gate.weight" {
			return nil, v41ProjectionDeclined, nil
		}
		calls = append(calls, call{leaf: leaf, layer: layer, rows: rows, out: out, input: panel, snapshot: append([]float32(nil), panel...)})
		result := make([]float32, rows*out)
		for row := 0; row < rows; row++ {
			x, err := m.v41ProjMatRows(layer, leaf, panel[row*in:(row+1)*in], out, in)
			if err != nil {
				return nil, v41ProjectionError, err
			}
			copy(result[row*out:(row+1)*out], x)
		}
		return result, v41ProjectionHandled, nil
	}
	got, err := m.forwardV41(ids, &v41ForwardState{projectionRoute: newV41ProjectionRouteBinding(project, nil)})
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.forwardV41(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Logits) != seq || len(want.Logits) != seq {
		t.Fatalf("logit row coverage got=%d want=%d seq=%d", len(got.Logits), len(want.Logits), seq)
	}
	for row := range want.Logits {
		if !v41SameBits(got.Logits[row], want.Logits[row]) {
			t.Fatalf("scalar-row projection oracle differs at logit row %d", row)
		}
	}
	for layer := 0; layer < m.Cfg.NumLayers; layer++ {
		for _, leaf := range []string{"attn.wkv.weight", "ffn.gate.weight"} {
			var selected []call
			for _, c := range calls {
				if c.layer == layer && c.leaf == leaf {
					selected = append(selected, c)
				}
			}
			if len(selected) != len(wantChunks) {
				t.Fatalf("layer %d %s dispatched %d calls, want %d bounded chunks", layer, leaf, len(selected), len(wantChunks))
			}
			for chunk, c := range selected {
				if c.rows != wantChunks[chunk] || len(c.input) != c.rows*m.Cfg.HiddenSize {
					t.Fatalf("layer %d %s chunk %d rows=%d input=%d", layer, leaf, chunk, c.rows, len(c.input))
				}
				if full && leaf == "attn.wkv.weight" && c.out != v41KVLoraRank {
					t.Fatalf("full KV projection width=%d want %d", c.out, v41KVLoraRank)
				}
				if !v41SameBits(c.input, c.snapshot) {
					t.Fatalf("layer %d %s chunk %d input was overwritten after dispatch", layer, leaf, chunk)
				}
			}
		}
	}
}
