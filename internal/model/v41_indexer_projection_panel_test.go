package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Two full chunks plus a multi-row tail: every chunk is a real panel primitive.
const v41IndexerPanelTestRows = 2*v41ProjPanelRowChunk + 5

func v41IndexerPanelTestChunks(rows int) int {
	return (rows + v41ProjPanelRowChunk - 1) / v41ProjPanelRowChunk
}

type v41IndexerPanelTestRun struct {
	logits []float32
	topK   [][]int32
	delta  map[string]float64
	panels int64
	ops    map[string][]v41DenseTestOp
	err    error
}

// v41IndexerPanelTestPrefill runs one cold prefill on a fresh fixture. A nil
// backend is the CPU reference; otherwise the session binds the recording
// device projection.
func v41IndexerPanelTestPrefill(t *testing.T, disabled bool, ids []int, backend *v41CompressorTestBackend, mutate func(*Model)) v41IndexerPanelTestRun {
	t.Helper()
	prior := v41IndexerPanelDisabled
	v41IndexerPanelDisabled = disabled
	defer func() { v41IndexerPanelDisabled = prior }()
	m := v41IndexerTestFixture(t)
	if mutate != nil {
		mutate(m)
	}
	var s *Session
	if backend == nil {
		s = m.NewSession()
	} else {
		s = v41DenseTestSession(t, m, backend)
	}
	defer s.Close()
	before := v41IndexerTestPhase(t, m, "prefill")
	from := 0
	if backend != nil {
		from = len(backend.ops)
	}
	panelsBefore := v41PanelInvocationCount.Load()
	var run v41IndexerPanelTestRun
	run.err = panicAsError(func() { run.logits = append([]float32(nil), s.Prefill(ids)...) })
	run.panels = v41PanelInvocationCount.Load() - panelsBefore
	run.delta = v41DenseTestDelta(v41IndexerTestPhase(t, m, "prefill"), before)
	if run.err != nil {
		return run
	}
	rows, ratio, ok := s.v41Forward.attn.TopK()
	if !ok || ratio != 2 || len(rows) != len(ids) {
		t.Fatalf("prefill published %d TopK rows at ratio %d (ok=%t), want %d at ratio 2", len(rows), ratio, ok, len(ids))
	}
	for _, row := range rows {
		run.topK = append(run.topK, append([]int32(nil), row...))
	}
	if backend != nil {
		run.ops, _, _ = v41IndexerTestOps(t, s, backend, from)
	}
	return run
}

func v41IndexerPanelTestSameLogits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("panel logits len %d, per-row len %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("panel logit[%d] bits %08x, per-row %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// The CPU reference projects the indexer query and head weights once per
// chunk instead of once per row, and every logit and selection keeps its bits.
//
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41IndexerProjectionPanelCPURefBitIdentical(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" {
		t.Skip("requires the cpu-ref default backend")
	}
	ids := v41ParityPrompt(v41IndexerTestFixture(t).Cfg.VocabSize, v41IndexerPanelTestRows)
	rows := v41IndexerPanelTestPrefill(t, true, ids, nil, nil)
	panel := v41IndexerPanelTestPrefill(t, false, ids, nil, nil)
	if rows.err != nil || panel.err != nil {
		t.Fatalf("prefill errors: per-row %v, panel %v", rows.err, panel.err)
	}
	v41IndexerPanelTestSameLogits(t, panel.logits, rows.logits)
	if !reflect.DeepEqual(panel.topK, rows.topK) {
		t.Fatal("panel indexer selections differ from the per-row path")
	}
	n, chunks := len(ids), v41IndexerPanelTestChunks(len(ids))
	if rows.delta["indexer_projection_device_calls"] != 0 || panel.delta["indexer_projection_device_calls"] != 0 {
		t.Fatalf("CPU reference used a device projection: per-row %v, panel %v", rows.delta, panel.delta)
	}
	if panel.delta["indexer_projection_host_rows"] != rows.delta["indexer_projection_host_rows"] {
		t.Fatalf("projected rows changed: panel %v, per-row %v", panel.delta["indexer_projection_host_rows"], rows.delta["indexer_projection_host_rows"])
	}
	// One index layer, two leaves: 2n per-row calls become 2*ceil(n/chunk).
	if saved := rows.delta["indexer_projection_host_calls"] - panel.delta["indexer_projection_host_calls"]; saved != float64(2*(n-chunks)) {
		t.Fatalf("panel saved %v host projection calls, want %d", saved, 2*(n-chunks))
	}
	if extra := panel.panels - rows.panels; extra != int64(2*chunks) {
		t.Fatalf("panel ran %d extra panel primitives, want %d", extra, 2*chunks)
	}
}

// A selected f32 device projection receives chunk panels and returns the same
// bits as its one-row calls.
//
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41IndexerProjectionPanelDeviceF32BitIdentical(t *testing.T) {
	ids := v41ParityPrompt(v41IndexerTestFixture(t).Cfg.VocabSize, v41IndexerPanelTestRows)
	rows := v41IndexerPanelTestPrefill(t, true, ids, newV41CompressorTestBackend(), nil)
	panel := v41IndexerPanelTestPrefill(t, false, ids, newV41CompressorTestBackend(), nil)
	if rows.err != nil || panel.err != nil {
		t.Fatalf("prefill errors: per-row %v, panel %v", rows.err, panel.err)
	}
	v41IndexerPanelTestSameLogits(t, panel.logits, rows.logits)
	if !reflect.DeepEqual(panel.topK, rows.topK) {
		t.Fatal("panel indexer selections differ from the per-row path")
	}
	n, chunks := len(ids), v41IndexerPanelTestChunks(len(ids))
	var want []int
	for lo := 0; lo < n; lo += v41ProjPanelRowChunk {
		want = append(want, min(v41ProjPanelRowChunk, n-lo))
	}
	for _, leaf := range []string{"indexer.wq_b.weight", "indexer.weights_proj.weight"} {
		var got, perRow []int
		for _, op := range panel.ops[leaf] {
			got = append(got, op.rows)
		}
		for _, op := range rows.ops[leaf] {
			perRow = append(perRow, op.rows)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s panel device rows per call %v, want %v", leaf, got, want)
		}
		if len(perRow) != n {
			t.Fatalf("%s per-row path made %d device calls, want %d", leaf, len(perRow), n)
		}
	}
	if panel.delta["indexer_projection_host_calls"] != 0 || panel.delta["indexer_projection_device_rows"] != rows.delta["indexer_projection_device_rows"] {
		t.Fatalf("panel device accounting %v, per-row %v", panel.delta, rows.delta)
	}
	if saved := rows.delta["indexer_projection_device_calls"] - panel.delta["indexer_projection_device_calls"]; saved != float64(2*(n-chunks)) {
		t.Fatalf("panel saved %v device projection calls, want %d", saved, 2*(n-chunks))
	}
}

// A missing head-weight leaf still refuses at its per-row site after the
// query panel ran.
//
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41IndexerProjectionPanelKeepsPerRowRefusal(t *testing.T) {
	ids := v41ParityPrompt(v41IndexerTestFixture(t).Cfg.VocabSize, v41IndexerPanelTestRows)
	drop := func(m *Model) { delete(m.manifest, layerName(0, "indexer.weights_proj.weight")) }
	for _, disabled := range []bool{true, false} {
		run := v41IndexerPanelTestPrefill(t, disabled, ids, nil, drop)
		var stage *V41ForwardError
		if !errors.Is(run.err, ErrV41ForwardStage) || !errors.As(run.err, &stage) || stage.Stage != v41StageIndexer || stage.Layer != 0 {
			t.Fatalf("disabled=%t refusal = %v, want the layer-0 indexer stage refusal", disabled, run.err)
		}
		if run.delta["indexer_scoring_calls"] != 0 {
			t.Fatalf("disabled=%t scored %v rows before the refusal", disabled, run.delta["indexer_scoring_calls"])
		}
	}
}

func v41IndexerPanelTestRows2D(rows, width int, scale float32) [][]float32 {
	out := make([][]float32, rows)
	for t := range out {
		out[t] = make([]float32, width)
		for i := range out[t] {
			out[t][i] = scale * float32((t*31+i*7)%23-11) / 23
		}
	}
	return out
}

// Admission keeps every case it cannot prove bit-identical on the per-row path.
//
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41IndexerProjectionPanelAdmission(t *testing.T) {
	m := v41IndexerTestFixture(t)
	cfg := m.Cfg
	width := cfg.IndexNHeads * cfg.IndexHeadDim
	const seq = v41ProjPanelRowChunk + 3
	qLat := v41IndexerPanelTestRows2D(seq, cfg.QLoraRank, 1)
	hidden := v41IndexerPanelTestRows2D(seq, cfg.HiddenSize, .5)
	var calls []int
	declineAll := func(_ int, _ string, _ []float32, _, _, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		calls = append(calls, rows)
		return nil, v41ProjectionDeclined, nil
	}
	callback := &v41ProjScratch{projectionRoute: newV41ProjectionRouteBinding(declineAll, nil)}

	panel := m.v41IndexerProjectionPanels(0, seq, qLat, hidden, callback)
	if panel == nil || !panel.query || !panel.weights || len(calls) != 0 {
		t.Fatalf("f32 callback panel = %+v after %d calls; want both leaves and no projection", panel, len(calls))
	}
	for row := 0; row < seq; row++ {
		got, err := panel.row(row)
		if err != nil {
			t.Fatal(err)
		}
		q, err := m.v41ProjMatRows(0, "indexer.wq_b.weight", qLat[row], width, cfg.QLoraRank)
		if err != nil {
			t.Fatal(err)
		}
		w, err := m.v41ProjMatRows(0, "indexer.weights_proj.weight", hidden[row], cfg.IndexNHeads, cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		for i := range q {
			if math.Float32bits(got.query[i]) != math.Float32bits(q[i]) {
				t.Fatalf("query row %d[%d] differs from the per-row projection", row, i)
			}
		}
		for i := range w {
			if math.Float32bits(got.weights[i]) != math.Float32bits(w[i]) {
				t.Fatalf("weights row %d[%d] differs from the per-row projection", row, i)
			}
		}
		if cap(got.query) != width || cap(got.weights) != cfg.IndexNHeads {
			t.Fatal("panel row capacity reaches its neighbour")
		}
	}
	if want := []int{v41ProjPanelRowChunk, v41ProjPanelRowChunk, 3, 3}; !reflect.DeepEqual(calls, want) {
		t.Fatalf("callback rows per call %v, want %v", calls, want)
	}

	nonfinite := v41IndexerPanelTestRows2D(seq, cfg.HiddenSize, .5)
	nonfinite[seq-1][0] = float32(math.Inf(1))
	short := v41IndexerPanelTestRows2D(seq, cfg.QLoraRank, 1)
	short[1] = short[1][:cfg.QLoraRank-1]
	quantized := v41IndexerTestFixture(t)
	weightsName := layerName(0, "indexer.weights_proj.weight")
	delete(quantized.manifest, weightsName)
	quantized.q8w = map[string]*q8Tensor{weightsName: {out: cfg.IndexNHeads, in: cfg.HiddenSize}}
	for _, tc := range []struct {
		name           string
		m              *Model
		seq            int
		qLat, hidden   [][]float32
		scratch        *v41ProjScratch
		query, weights bool
	}{
		{"single-row", m, 1, qLat[:1], hidden[:1], callback, false, false},
		{"row-count-mismatch", m, seq, qLat, hidden[:seq-1], callback, false, false},
		{"short-query-row", m, seq, short, hidden, callback, false, true},
		{"device-nonfinite-hidden", m, seq, qLat, nonfinite, callback, true, false},
		{"host-nonfinite-hidden", m, seq, qLat, nonfinite, &v41ProjScratch{}, true, true},
		{"device-quantized-weights", quantized, seq, qLat, hidden, callback, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls = nil
			panel := tc.m.v41IndexerProjectionPanels(0, tc.seq, tc.qLat, tc.hidden, tc.scratch)
			if gotQuery, gotWeights := panel != nil && panel.query, panel != nil && panel.weights; gotQuery != tc.query || gotWeights != tc.weights {
				t.Fatalf("admitted query=%t weights=%t, want query=%t weights=%t", gotQuery, gotWeights, tc.query, tc.weights)
			}
			if !tc.query && !tc.weights && panel != nil {
				t.Fatal("declined panel is non-nil")
			}
		})
	}
}

// A bound canonical query keeps its fused one-row projection and publication.
//
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=400ms lane=default
func TestV41IndexerProjectionPanelKeepsFusedQueryPublication(t *testing.T) {
	declineAll := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionDeclined, nil
	}
	fused := &v41ProjScratch{
		projectionRoute: &v41ProjectionRouteBinding{projection: declineAll, nativeProjection: v41BoundDenseProjection{
			core: func(int, string, []float32, int, int, int, *v41ProjectionContinuation) ([]float32, v41DenseProjectionOutcome, error) {
				return nil, v41ProjectionDeclined, nil
			},
			owner: &Session{},
		}},
		gatherOps: &v41DeviceGatherOps{bf16Publish: &v41BF16PublicationBinding{}},
	}
	canonical := v41IndexerTestFixture(t)
	canonical.Cfg.IndexNHeads, canonical.Cfg.IndexHeadDim = 64, 128
	if !canonical.v41IndexQueryMayPublish(fused.projectionRoute, fused.gatherOps, 8192) {
		t.Fatal("canonical bound query publication not recognized")
	}
	ordinary := newV41ProjectionRouteBinding(declineAll, nil)
	for name, admitted := range map[string]bool{
		"no-gather-ops":     canonical.v41IndexQueryMayPublish(fused.projectionRoute, nil, 8192),
		"ordinary-callback": canonical.v41IndexQueryMayPublish(ordinary, fused.gatherOps, 8192),
		"reduced-geometry":  v41IndexerTestFixture(t).v41IndexQueryMayPublish(fused.projectionRoute, fused.gatherOps, 8192),
	} {
		if admitted {
			t.Fatalf("%s recognized as a fused query publication", name)
		}
	}
}
