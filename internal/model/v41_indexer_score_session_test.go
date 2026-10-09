package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41ScoreOnly(s *Session) {
	st := s.v41State()
	st.expertGateUp, st.expertDown = nil, nil
	st.denseProjection, st.groupedOutput, st.engramProjection = nil, nil, nil
	st.mhcProjection, st.finalNorm, st.queryNorm, st.kvNorm = nil, nil, nil, nil
	st.ffnNorm, st.compressorNorm, st.indexKeyNorm = nil, nil, nil
	st.sharedActivation, st.tailRoPE, st.sharedAttention = nil, nil, nil
}

// This pins the projected-input boundary, including a nonzero absolute query
// position that distinguishes missing and duplicate RoPE. No device is used.
// fak-test:runtime medium est=1s lane=default
func TestV41IndexerScoreProjectedOperands(t *testing.T) {
	m := v41IndexerTestFixture(t)
	b := newV41ScoreTestBackend()
	s := v41DenseTestSession(t, m, b)
	v41ScoreOnly(s)
	cfg := m.Cfg
	qLat, hidden := make([]float32, cfg.QLoraRank), make([]float32, cfg.HiddenSize)
	for i := range qLat {
		qLat[i] = float32(i+1) / 17
	}
	for i := range hidden {
		hidden[i] = float32(i%7+1) / 19
	}
	keys := [][]float32{make([]float32, cfg.IndexHeadDim), make([]float32, cfg.IndexHeadDim)}
	for i := range keys[0] {
		keys[0][i], keys[1][i] = float32(i%3-1)/3, float32(i%5-2)/5
	}
	width := cfg.IndexNHeads * cfg.IndexHeadDim
	unrotated, err := m.v41ProjMatRowsWithProjection(0, "indexer.wq_b.weight", qLat, width, cfg.QLoraRank, nil)
	if err != nil {
		t.Fatal(err)
	}
	once := append([]float32(nil), unrotated...)
	if err := m.v41IndexRoPE(0, 7, once, cfg.IndexNHeads); err != nil {
		t.Fatal(err)
	}
	twice := append([]float32(nil), once...)
	if err := m.v41IndexRoPE(0, 7, twice, cfg.IndexNHeads); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(once, unrotated) || reflect.DeepEqual(once, twice) {
		t.Fatal("position-sensitive rotary control is vacuous")
	}
	weights, err := m.v41ProjMatRowsWithProjection(0, "indexer.weights_proj.weight", hidden, cfg.IndexNHeads, cfg.HiddenSize, nil)
	if err != nil {
		t.Fatal(err)
	}
	for h := range weights {
		weights[h] *= float32(1/math.Sqrt(float64(cfg.IndexHeadDim))) * float32(1/math.Sqrt(float64(cfg.IndexNHeads)))
	}
	counts := map[string]int{}
	project := func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		counts[leaf]++
		values, err := m.v41ProjectionRows(layer, leaf, panel, out, in, rows, nil)
		return values, v41ProjectionHandled, err
	}
	got, err := m.v41IndexRowsWithOperations(0, 7, qLat, hidden, keys, project, s.v41State().indexerScore, s.v41State().indexScoreHealth)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.v41IndexRowsProjected(0, 7, qLat, hidden, keys, nil)
	if err != nil || !reflect.DeepEqual(got, want) || len(b.calls) != 1 || counts["indexer.wq_b.weight"] != 1 || counts["indexer.weights_proj.weight"] != 1 || len(counts) != 2 {
		t.Fatalf("projected score changed projections or selection: got=%v want=%v counts=%v err=%v", got, want, counts, err)
	}
	v41ScoreTestBits(t, b.calls[0].q, once)
	v41ScoreTestBits(t, b.calls[0].weights, weights)
	v41ScoreTestBits(t, b.calls[0].keys, append(append([]float32(nil), keys[0]...), keys[1]...))
}

// The independent bit oracle pins the current float32 factor boundaries:
// 128^-0.5 * 32^-0.5 is mathematically 1/64, but rounding each factor first
// gives 0x3c7fffff. This does not claim reference BF16 weight-product parity.
// The runtime estimate is unmeasured until an authorized execution witness.
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerScoreScaleUsesIndexGeometry(t *testing.T) {
	t.Parallel()
	base := Config{
		HeadDim: 512, IndexHeadDim: 128, IndexNHeads: 32, IndexTopK: 1,
		HiddenSize: 1, QLoraRank: 1, QKRopeHeadDim: 2,
		DeepSeekV41: &DeepSeekV41Config{
			CompressRatios: []int{2}, IndexSourceLayerIDs: []int{0},
			CandidateSourceLayerID: -1, CompressRopeTheta: 10000,
		},
	}
	manifest, raw := synthBuildRaw([]synthTensor{
		{layerName(0, "indexer.wq_b.weight"), []int{128 * 32, 1}},
		{layerName(0, "indexer.weights_proj.weight"), []int{32, 1}},
	}, func(string, func() float32) float32 { return 1 })
	for _, name := range []string{"unequal-head-widths", "attention-head-width", "attention-scalar-override", "longrope-attention-multiplier"} {
		t.Run(name, func(t *testing.T) {
			cfg := base
			switch name {
			case "attention-head-width":
				cfg.HeadDim = 2048
			case "attention-scalar-override":
				cfg.QueryPreAttnScalar = 8
			case "longrope-attention-multiplier":
				cfg.LongRope = &RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: 2}
				cfg.MaxPositionEmbeddings = 8
			}
			m := &Model{Cfg: cfg, manifest: manifest, raw: raw}
			calls := 0
			score := func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
				calls++
				if layer != 0 || heads != 32 || dim != 128 || rows != 1 || len(weights) != 32 {
					t.Fatalf("unexpected score geometry: layer=%d heads=%d dim=%d rows=%d weights=%d", layer, heads, dim, rows, len(weights))
				}
				for h, value := range weights {
					if bits := math.Float32bits(value); bits != 0x3c7fffff {
						t.Fatalf("head %d index weight bits=%08x, want 3c7fffff", h, bits)
					}
				}
				return []float32{0}, nil
			}
			selected, err := m.v41IndexRowsWithOperations(0, 0, []float32{1}, []float32{1}, [][]float32{make([]float32, 128)}, nil, score, nil)
			if err != nil || calls != 1 || !reflect.DeepEqual(selected, []int32{0}) {
				t.Fatalf("score boundary not exercised once: selected=%v calls=%d err=%v", selected, calls, err)
			}
		})
	}
}

// The recorder changes only the score producer. Scalar projections, RoPE,
// selection and attention remain intact; bit equality is required throughout.
// Estimates are unmeasured until the separately authorized execution witness.
// No parallel execution: continuation probes are package-global.
// fak-test:runtime medium est=10s lane=default
func TestV41IndexerScoreSession(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("score software witness requires cpu-ref")
	}
	newSession := func(t *testing.T, m *Model) (*Session, *v41ScoreTestBackend) {
		t.Helper()
		b := newV41ScoreTestBackend()
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		if s.v41State().indexerScore == nil {
			t.Fatal("qualified score callback missing")
		}
		return s, b
	}
	t.Run("prefill-step-suffix-and-rebind", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		s, b := newSession(t, m)
		host := m.NewSession()
		t.Cleanup(host.Close)
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
		readStep, restoreStep := v41SessionIncrementalCalls(t)
		defer restoreStep()
		v41ScoreTestBits(t, s.Step(4), host.Step(4))
		if readStep() != 2 {
			t.Fatal("selected or host Step replayed full history")
		}
		readSuffix, restoreSuffix := v41SuffixPrefillCalls(t)
		defer restoreSuffix()
		v41ScoreTestBits(t, s.Prefill([]int{5, 6}), host.Prefill([]int{5, 6}))
		if readSuffix() != 4 || len(b.calls) == 0 || len(b.allocations) != 0 {
			t.Fatal("suffix replayed, skipped score, or leaked")
		}
		before := captureV41ForwardSnapshot(s.v41Forward)
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(host.v41Forward)) {
			t.Fatal("score path changed retained index/KV/top-k state")
		}
		dataOnly := before.clone().restore()
		if dataOnly.indexerScore != nil || dataOnly.indexScoreHealth != nil || dataOnly.callbackOwner != nil {
			t.Fatal("snapshot persisted score callback or owner")
		}
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer snap.Close()
		fork, err := snap.Clone()
		if err != nil {
			t.Fatal(err)
		}
		defer fork.Close()
		target, branch := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
		for i, item := range []struct {
			snapshot *PrefixSnapshot
			session  *Session
		}{{snap, target}, {fork, branch}} {
			if err := item.snapshot.Restore(item.session); err != nil {
				t.Fatal(err)
			}
			st := item.session.v41Forward
			if st == nil || st.callbackOwner != item.session || st.indexerScore == nil || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
				t.Fatalf("restore %d did not rebind the target owner", i)
			}
			v41ScoreOnly(item.session)
		}
		s.Close()
		calls := len(b.calls)
		got := target.Step(7)
		if !reflect.DeepEqual(captureV41ForwardSnapshot(branch.v41Forward), before) {
			t.Fatal("branch continuation aliased restored source")
		}
		v41ScoreTestBits(t, got, branch.Prefill([]int{7}))
		v41ScoreTestBits(t, got, host.Step(7))
		if len(b.calls) != calls+2 || len(b.allocations) != 0 {
			t.Fatal("restored score callback replayed, leaked or used the closed source")
		}
		calls = len(b.calls)
		target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		var refused *BackendForwardOperationError
		if err := recoverError(func() { target.Step(8) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || len(b.calls) != calls {
			t.Fatalf("score seam widened whole-architecture DeviceOnly admission: %v", err)
		}
	})
	t.Run("declined-is-original-host", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41ScoreTestBackend()
		b.supported = false
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		if s.v41State().indexerScore != nil {
			t.Fatal("unsupported score callback installed")
		}
		host := m.NewSession()
		t.Cleanup(host.Close)
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
		v41ScoreTestBits(t, s.Step(4), host.Step(4))
		if len(b.calls) != 0 || b.uploadCount != 0 || b.readCount != 0 {
			t.Fatal("unsupported path performed score device I/O")
		}
		before := captureV41ForwardSnapshot(s.v41Forward)
		b.healthErr = &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassDeviceLost, Err: ErrV41ForwardStage}
		err := recoverError(func() { s.Step(5) })
		var closed *BackendForwardOperationError
		if !errors.As(err, &closed) || !errors.Is(err, b.healthErr) || closed.Stage != "admission" || !s.BackendSessionClosed() || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatalf("cached unsupported score masked later sticky fault: %v", err)
		}
		if len(b.calls) != 0 || b.uploadCount != 0 || b.readCount != 0 {
			t.Fatal("sticky health performed score I/O or retried on host")
		}
	})
	t.Run("late-suffix-retains-earlier-token", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		s, b := newSession(t, m)
		s.Prefill([]int{1, 2, 3})
		host := m.NewSession()
		t.Cleanup(host.Close)
		host.Prefill([]int{1, 2, 3})
		host.Step(4)
		want := captureV41ForwardSnapshot(host.v41Forward)
		selected, attempts := s.v41State().indexerScore, 0
		s.v41State().indexerScore = func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
			attempts++
			if attempts == 2 {
				b.forceNaN = true
			}
			return selected(layer, q, keys, weights, heads, dim, rows)
		}
		calls := len(b.calls)
		err := recoverError(func() { s.Prefill([]int{4, 5}) })
		var closed *BackendForwardOperationError
		if !errors.As(err, &closed) || closed.Stage != "selection" || !s.BackendSessionClosed() || attempts != 2 || len(b.calls) != calls+2 || len(b.allocations) != 0 {
			t.Fatalf("late suffix did not fail once on its second token: %v", err)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), want) {
			t.Fatal("late score failure lost an earlier suffix commit or kept the failed token")
		}
	})
	for _, entry := range []string{"cold", "step", "suffix"} {
		for _, fault := range []string{"dispatch", "selector NaN"} {
			t.Run(entry+"/"+fault, func(t *testing.T) {
				m := v41IndexerTestFixture(t)
				s, b := newSession(t, m)
				if entry != "cold" {
					s.Prefill([]int{1, 2, 3})
				}
				before, calls := captureV41ForwardSnapshot(s.v41Forward), len(b.calls)
				cause := &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
				b.cause = cause
				if fault == "dispatch" {
					b.fault = "dispatch"
				} else {
					b.forceNaN = true
				}
				invoke := func() {
					switch entry {
					case "cold":
						s.Prefill([]int{1, 2})
					case "step":
						s.Step(4)
					case "suffix":
						s.Prefill([]int{4, 5})
					}
				}
				err := recoverError(invoke)
				var selected *V41IndexerScoreOperationError
				var closed *BackendForwardOperationError
				if !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Path != "v41-indexer-score" || closed.Layer != 0 || !s.BackendSessionClosed() {
					t.Fatalf("selected session failure not closed/attributed: %v", err)
				}
				if fault == "dispatch" && !errors.Is(err, cause) {
					t.Fatal("dispatch cause identity changed")
				}
				if fault == "selector NaN" && closed.Stage != "selection" {
					t.Fatalf("NaN rejected before unchanged selector: %v", err)
				}
				if len(b.calls) != calls+1 || len(b.allocations) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Fatal("selected failure replayed, leaked or committed continuation")
				}
				calls, uploads, reads := len(b.calls), b.uploadCount, b.readCount
				if retry := v41IndexerTestRecover(invoke); retry != closed || len(b.calls) != calls || b.uploadCount != uploads || b.readCount != reads {
					t.Fatal("same-session retry changed latch or performed I/O")
				}
			})
		}
	}
}
