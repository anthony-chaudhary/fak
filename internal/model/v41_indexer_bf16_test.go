package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// These controls pin only the BF16 tensor boundaries in the reference's BF16
// execution profile. They do not emulate FP8/FP4 quantization or qualify a
// projection/reduction implementation. Runtime estimates are unmeasured.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerBF16Boundaries(t *testing.T) {
	t.Parallel()
	config := Config{
		HiddenSize: 1, QLoraRank: 1, HeadDim: 4, IndexHeadDim: 4,
		IndexNHeads: 2, IndexTopK: 1, QKRopeHeadDim: 2, RMSNormEps: 8,
		DeepSeekV41: &DeepSeekV41Config{
			CompressRatios: []int{1}, IndexSourceLayerIDs: []int{0},
			CandidateSourceLayerID: -1, CompressRopeTheta: 10000,
		},
	}
	fixture := func(t *testing.T) *Model {
		t.Helper()
		manifest, raw := synthBuildRaw([]synthTensor{
			{layerName(0, "indexer.wk.weight"), []int{4, 4}},
			{layerName(0, "indexer.k_norm.weight"), []int{4}},
			{layerName(0, "indexer.wq_b.weight"), []int{8, 1}},
			{layerName(0, "indexer.weights_proj.weight"), []int{2, 1}},
		}, func(string, func() float32) float32 { return 1 })
		m := &Model{Cfg: config, manifest: manifest, raw: raw}
		v41WriteTensorF32(t, m, layerName(0, "indexer.wk.weight"), []float32{
			1.00390625, 0, 0, 0, 1.00390625, 0, 0, 0,
			1.00390625, 0, 0, 0, 1.00390625, 0, 0, 0,
		})
		v41WriteTensorF32(t, m, layerName(0, "indexer.k_norm.weight"), []float32{1, 2, -1, -2})
		v41WriteTensorF32(t, m, layerName(0, "indexer.wq_b.weight"), []float32{1.00390625, 1.01171875, 1.00390625, 1.00390625, -1.00390625, -2.5, 1, 0})
		return m
	}
	checkBits := func(t *testing.T, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("width %d, want %d", len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("element %d = %g (%08x), want %g (%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
	for _, selected := range []bool{false, true} {
		name := "host-projection"
		if selected {
			name = "selected-projection"
		}
		t.Run(name, func(t *testing.T) {
			m := fixture(t)
			var project v41DenseProjectionFunc
			if selected {
				project = func(l int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					values, err := m.v41ProjectionRows(l, leaf, input, out, in, rows, nil)
					return values, v41ProjectionHandled, err
				}
			}
			normCalls := 0
			normResult := []float32{1.00390625, 1.01171875, -1.00390625, -1.01171875}
			normalize := func(_ int, input, _ []float32, _ float32) ([]float32, error) {
				normCalls++
				checkBits(t, input, []float32{1, 1, 1, 1})
				// Exact halfway values exercise ties to both even neighbors.
				return normResult, nil
			}
			keys, err := m.v41IndexKeysWithOperations(0, [][]float32{{1, 0, 0, 0}}, project, normalize)
			if err != nil || normCalls != 1 || len(keys) != 1 {
				t.Fatalf("index norm calls=%d keys=%v err=%v", normCalls, keys, err)
			}
			checkBits(t, keys[0], []float32{1, 1.015625, -1, -1.015625})
			checkBits(t, normResult, []float32{1.00390625, 1.01171875, -1.00390625, -1.01171875})
			keys, err = m.v41IndexKeysWithOperations(0, [][]float32{{1, 0, 0, 0}}, project, nil)
			if err != nil || len(keys) != 1 {
				t.Fatalf("host norm keys=%v err=%v", keys, err)
			}
			// After projection copyback, variance=1 and epsilon=8, so the
			// F32 normalized values are +/-1/3 and +/-2/3 before BF16.
			checkBits(t, keys[0], []float32{0.333984375, 0.66796875, -0.333984375, -0.66796875})
			scoreCalls := 0
			score := func(_ int, q, _, _ []float32, _, _, _ int) ([]float32, error) {
				scoreCalls++
				// Projection ties precede RoPE; each head rounds only its
				// rotary tail again at copyback. All expected bits are literal.
				checkBits(t, q, []float32{1, 1.015625, -0.30078125, 1.3828125, -1, -2.5, 0.5390625, 0.83984375})
				return []float32{0}, nil
			}
			ids, err := m.v41IndexRowsWithOperations(0, 1, []float32{1}, []float32{1}, keys, project, score, nil)
			if err != nil || scoreCalls != 1 || len(ids) != 1 || ids[0] != 0 {
				t.Fatalf("score boundary ids=%v calls=%d err=%v", ids, scoreCalls, err)
			}
		})
	}
}

// A late source has already advanced an earlier source's publication when the
// host BF16 boundary refuses a finite F32 norm result. This is a model failure,
// not evidence of device loss: rollback must occur without replay or a made-up
// backend close/latch. The callback-owned result must remain unchanged.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41IndexerBF16LateFailureRollsBackWithoutReplay(t *testing.T) {
	t.Parallel()
	m := v41IndexerTestFixture(t)
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0, 1}
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Prefill([]int{1, 2, 3})
	st := s.v41State()
	if !s.v41IncrementalEligible() {
		t.Fatal("late BF16 control is not on the incremental route")
	}
	before := captureV41ForwardSnapshot(st)
	bad := make([]float32, m.Cfg.IndexHeadDim)
	bad[0], bad[len(bad)-1] = 1.00390625, math.MaxFloat32
	original := append([]float32(nil), bad...)
	calls := [2]int{}
	earlierPublicationAdvanced := false
	st.indexKeyNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
		calls[l]++
		if l == 1 {
			keys, ok := st.attn.IndexKeys(0)
			earlierPublicationAdvanced = ok && len(keys) == 2
			return bad, nil
		}
		return v41IndexKeyNormReference(input, gain, eps), nil
	}
	err := recoverError(func() { s.Step(4) })
	var operation *V41ProjectionOperationError
	if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &operation) || operation.Layer != 1 || operation.Leaf != "indexer.k_norm.weight" || operation.Stage != string(v41StageIndexer) {
		t.Fatalf("late BF16 failure lost typed no-replay attribution: %v", err)
	}
	if calls != [2]int{1, 1} || !earlierPublicationAdvanced {
		t.Fatalf("late failure replayed or preceded the earlier publication: calls=%v advanced=%v", calls, earlierPublicationAdvanced)
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(st)) || !reflect.DeepEqual(bad, original) {
		t.Fatal("late BF16 failure changed retained state or callback storage")
	}
	if s.BackendSessionClosed() || s.halFailure != nil {
		t.Fatal("host BF16 representability failure fabricated a backend failure")
	}
}

// Invalid original values must never reach bit rounding, and finite F32 values
// that overflow BF16 must never reach the next operation or lose no-replay
// attribution. The NaN payload below would round to signed zero unchecked.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerBF16RejectsNonfiniteBoundaries(t *testing.T) {
	t.Parallel()
	for _, bits := range []uint32{0x7fffffff, 0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff} {
		for _, leaf := range []string{"indexer.wk.weight", "indexer.k_norm.weight", "indexer.wq_b.weight"} {
			values := []float32{1.00390625, math.Float32frombits(bits)}
			got, err := v41IndexBF16Copyback(3, leaf, "control", values)
			var operation *V41ProjectionOperationError
			if got != nil || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &operation) || operation.Layer != 3 || operation.Leaf != leaf || operation.Stage != string(v41StageIndexer) {
				t.Fatalf("%s bits=%08x lost indexer no-replay error: %v", leaf, bits, err)
			}
			if math.Float32bits(values[0]) != 0x3f808000 || math.Float32bits(values[1]) != bits {
				t.Fatalf("%s failed copyback changed caller values", leaf)
			}
		}
	}
}
