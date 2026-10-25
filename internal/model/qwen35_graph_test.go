package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Restored from #9358 (e1a86a6ce, dropped by merge b4c39338d63) and extended for P-06
// (fak#13985): the sigmoid output-gate and capture-legal GDN checks, the QSA depth bound,
// private GDN state, and no model-side host read inside a captured hybrid step.

// qwenGraphCore records graph capture calls over the Qwen GDN dispatch witness. The
// capability mixins below compose the device paths each test backend advertises.
type qwenGraphCore struct {
	*recordingQwen35Backend
	capturing, declined, gdnCaptureIllegal  bool
	began, launched, aborted, resets, reads int
}

func newQwenGraphCore(m *Model) *qwenGraphCore {
	return &qwenGraphCore{recordingQwen35Backend: newRecordingQwen35Backend(m)}
}

func (b *qwenGraphCore) GraphBegin() bool {
	if b.declined {
		return false
	}
	b.capturing = true
	b.began++
	return true
}
func (b *qwenGraphCore) GraphEndLaunch()             { b.capturing = false; b.launched++ }
func (b *qwenGraphCore) GraphAbort()                 { b.capturing = false; b.aborted++ }
func (b *qwenGraphCore) GraphReset()                 { b.capturing = false; b.resets++ }
func (b *qwenGraphCore) IsCapturing() bool           { return b.capturing }
func (b *qwenGraphCore) Qwen35GDNCaptureLegal() bool { return !b.gdnCaptureIllegal }

// Read counts model-side host reads made while a capture is open. Capability mixins read
// through the embedded backend directly, so they are not counted.
func (b *qwenGraphCore) Read(t compute.Tensor) []float32 {
	if b.capturing {
		b.reads++
	}
	return b.recordingQwen35Backend.Read(t)
}

type qwenRoPECap struct{ core *qwenGraphCore }

func (c qwenRoPECap) PartialRoPEQK(q, k compute.Tensor, pos, nQHeads, nKHeads, headDim, rotaryDim int, theta float64) (compute.Tensor, compute.Tensor) {
	be := c.core.recordingQwen35Backend.Backend
	qHost := append([]float32(nil), be.Read(q)...)
	kHost := append([]float32(nil), be.Read(k)...)
	cos, sin := ropeRowForLayer(c.core.model.Cfg, 0, pos)
	ropeRowQKInto(qHost, kHost, cos, sin, headDim, nQHeads, nKHeads)
	return compute.NewF32(be, []int{nQHeads * headDim}, qHost), compute.NewF32(be, []int{nKHeads * headDim}, kHost)
}

type qwenSplitCap struct{ core *qwenGraphCore }

func (c qwenSplitCap) SplitQwen35QueryGate(qg compute.Tensor, nHeads, headDim int) (compute.Tensor, compute.Tensor) {
	be := c.core.recordingQwen35Backend.Backend
	qData, gData := splitQwen35HeadInterleavedRows(be.Read(qg), nHeads, headDim, 1)
	return compute.NewF32(be, []int{nHeads * headDim}, qData), compute.NewF32(be, []int{nHeads * headDim}, gData)
}

type qwenSigmoidCap struct{ core *qwenGraphCore }

func (c qwenSigmoidCap) SigmoidMulInPlace(x, gate compute.Tensor) {
	be := c.core.recordingQwen35Backend.Backend
	xData, ok := be.Host(x)
	gData := be.Read(gate)
	if ok {
		for i := range xData {
			xData[i] *= sigmoidf(gData[i])
		}
	}
}

type testQwenGraphBackend struct {
	*qwenGraphCore
	qwenRoPECap
	qwenSplitCap
	qwenSigmoidCap
}

type testQwenMissingRoPEBackend struct {
	*qwenGraphCore
	qwenSplitCap
	qwenSigmoidCap
}

type testQwenMissingSplitBackend struct {
	*qwenGraphCore
	qwenRoPECap
	qwenSigmoidCap
}

type testQwenMissingSigmoidBackend struct {
	*qwenGraphCore
	qwenRoPECap
	qwenSplitCap
}

func newTestQwenGraphBackend(m *Model) *testQwenGraphBackend {
	c := newQwenGraphCore(m)
	return &testQwenGraphBackend{c, qwenRoPECap{c}, qwenSplitCap{c}, qwenSigmoidCap{c}}
}

// qwen35GraphTestCfg is the hybrid test geometry with partial RoPE, so the device
// partial-RoPE capability is part of the predicate.
func qwen35GraphTestCfg() Config {
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = 0.25
	return cfg
}

// testQwenReplaceStateBackend replaces in-place state mid-capture to trigger abort.
type testQwenReplaceStateBackend struct {
	*testQwenGraphBackend
	replaceAt int
}

func (b *testQwenReplaceStateBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	output, nextConv, nextRec, err := b.testQwenGraphBackend.Qwen35GDNDecode(
		normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon,
	)
	if err != nil {
		return output, nextConv, nextRec, err
	}
	if b.replaceAt > 0 && b.gdnCalls >= b.replaceAt {
		replacement := compute.NewF32(b.recordingQwen35Backend.Backend, convState.Shape, make([]float32, convState.Numel()))
		return output, replacement, nextRec, nil
	}
	return output, nextConv, nextRec, nil
}

type qwenDepthKV struct {
	compute.KVStore
	n int
}

func (k qwenDepthKV) Len() int { return k.n }

func newQwenGraphSession(t *testing.T, m *Model, be compute.Backend) *Session {
	t.Helper()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

// fak-test:runtime fast est=1s
func TestQwen35HybridGraphSafeGating(t *testing.T) {
	cfg := qwen35GraphTestCfg()
	m := NewSynthetic(cfg)
	be := newTestQwenGraphBackend(m)
	s := newQwenGraphSession(t, m, be)

	if !s.cudaGraphCaptureReady(false) {
		t.Fatal("fully featured hybrid session is not capture-ready")
	}
	if ok, reason := cudaGraphCaptureEligible(cfg, be); !ok || reason != CUDAGraphEligible {
		t.Fatalf("eligibility = %v %q, want eligible", ok, reason)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
		reason string
	}{
		{"scaled rope", func(c *Config) { c.RopeScaling = "linear" }, CUDAGraphQwen35ScaledRoPE},
		{"long rope", func(c *Config) { c.LongRope = &RopeScaling{Type: "longrope"} }, CUDAGraphQwen35ScaledRoPE},
	} {
		c := cfg
		tc.mutate(&c)
		if ok, reason := cudaGraphCaptureEligible(c, be); ok || reason != tc.reason {
			t.Fatalf("%s: eligibility = %v %q, want %q", tc.name, ok, reason, tc.reason)
		}
	}
	if s.cudaGraphCaptureReady(true) {
		t.Fatal("split placement is capture-ready")
	}

	c := newQwenGraphCore(m)
	for _, tc := range []struct {
		name   string
		be     compute.Backend
		reason string
	}{
		{"missing partial rope", &testQwenMissingRoPEBackend{c, qwenSplitCap{c}, qwenSigmoidCap{c}}, CUDAGraphQwen35NoPartialRoPE},
		{"missing query-gate split", &testQwenMissingSplitBackend{c, qwenRoPECap{c}, qwenSigmoidCap{c}}, CUDAGraphQwen35NoQueryGateSplit},
		{"missing sigmoid gate", &testQwenMissingSigmoidBackend{c, qwenRoPECap{c}, qwenSplitCap{c}}, CUDAGraphQwen35NoSigmoidGate},
	} {
		if ok, reason := cudaGraphCaptureEligible(cfg, tc.be); ok || reason != tc.reason {
			t.Fatalf("%s: eligibility = %v %q, want %q", tc.name, ok, reason, tc.reason)
		}
	}
	illegal := newTestQwenGraphBackend(m)
	illegal.gdnCaptureIllegal = true
	if ok, reason := cudaGraphCaptureEligible(cfg, illegal); ok || reason != CUDAGraphQwen35GDNNotCaptured {
		t.Fatalf("capture-illegal GDN: eligibility = %v %q, want %q", ok, reason, CUDAGraphQwen35GDNNotCaptured)
	}

	kv := s.halKV
	s.halKV = qwenDepthKV{kv, compute.QSADynamicGatingThreshold - 1}
	if s.cudaGraphCaptureReady(false) {
		t.Fatal("a step that reaches the QSA sparse-gather depth is capture-ready")
	}
	s.halKV = kv

	owner := s.qwen35HAL.layers[0].owner
	owner.refs++
	if s.cudaGraphCaptureReady(false) {
		t.Fatal("shared copy-on-write GDN state is capture-ready")
	}
	owner.refs--

	closed := newQwenGraphSession(t, m, be)
	closed.Close()
	if closed.cudaGraphCaptureReady(false) {
		t.Fatal("closed session is capture-ready")
	}
}

// fak-test:runtime fast est=1s
func TestQwen35HybridDecodeEagerVsGraphParity(t *testing.T) {
	m := NewSynthetic(qwen35GraphTestCfg())
	beGraph := newTestQwenGraphBackend(m)
	sGraph := newQwenGraphSession(t, m, beGraph)
	beEager := newTestQwenGraphBackend(m)
	beEager.declined = true
	sEager := newQwenGraphSession(t, m, beEager)

	for step, token := range []int{3, 7, 11, 19} {
		logitsGraph := sGraph.Step(token)
		logitsEager := sEager.Step(token)
		if len(logitsGraph) != len(logitsEager) {
			t.Fatalf("step %d logits length graph=%d eager=%d", step, len(logitsGraph), len(logitsEager))
		}
		var maxDelta float64
		for i := range logitsGraph {
			maxDelta = math.Max(maxDelta, math.Abs(float64(logitsGraph[i]-logitsEager[i])))
		}
		if maxDelta != 0 {
			t.Fatalf("step %d graph/eager logits delta=%g, want bit-exact", step, maxDelta)
		}
	}
	if beGraph.began < 1 || beGraph.launched != beGraph.began || beGraph.aborted != 0 {
		t.Fatalf("graph session began=%d launched=%d aborted=%d, want captured replays and no abort", beGraph.began, beGraph.launched, beGraph.aborted)
	}
	if beGraph.reads != 0 {
		t.Fatalf("captured hybrid steps made %d model-side host reads, want 0", beGraph.reads)
	}
	if beEager.launched != 0 {
		t.Fatalf("declined session launched %d graphs, want 0", beEager.launched)
	}
}

// fak-test:runtime fast est=1s
func TestQwen35HybridDecodeFallbackWhenUnsafe(t *testing.T) {
	for _, tc := range []struct {
		name  string
		setup func(m *Model) (compute.Backend, *qwenGraphCore)
		cfg   func(*Config)
		split bool
	}{
		{"scaled-rope", func(m *Model) (compute.Backend, *qwenGraphCore) {
			b := newTestQwenGraphBackend(m)
			return b, b.qwenGraphCore
		}, func(c *Config) { c.RopeScaling = "linear" }, false},
		{"missing-sigmoid-gate", func(m *Model) (compute.Backend, *qwenGraphCore) {
			c := newQwenGraphCore(m)
			return &testQwenMissingSigmoidBackend{c, qwenRoPECap{c}, qwenSplitCap{c}}, c
		}, nil, false},
		{"split-session", func(m *Model) (compute.Backend, *qwenGraphCore) {
			b := newTestQwenGraphBackend(m)
			return b, b.qwenGraphCore
		}, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := qwen35GraphTestCfg()
			if tc.cfg != nil {
				tc.cfg(&cfg)
			}
			m := NewSynthetic(cfg)
			be, core := tc.setup(m)
			s := newQwenGraphSession(t, m, be)
			if tc.split {
				s.DenseGPULayers = 2
			}
			for _, token := range []int{5, 12, 19} {
				if logits := s.Step(token); len(logits) != cfg.VocabSize {
					t.Fatalf("logits size %d, want %d", len(logits), cfg.VocabSize)
				}
			}
			if core.began != 0 {
				t.Fatalf("unsafe session attempted %d captures, want 0", core.began)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestQwen35HybridDecodeGraphAbortOnStateViolation(t *testing.T) {
	m := NewSynthetic(qwen35GraphTestCfg())
	be := &testQwenReplaceStateBackend{testQwenGraphBackend: newTestQwenGraphBackend(m)}
	s := newQwenGraphSession(t, m, be)

	_ = s.Step(5) // warm logits step, uncaptured
	if be.began != 0 {
		t.Fatalf("warm step captured: began=%d", be.began)
	}
	// The warm step ran the 3 linear layers; the next step's first GDN call is the 4th.
	be.replaceAt = be.gdnCalls + 1
	panicked := false
	func() {
		defer func() { panicked = recover() != nil }()
		_ = s.Step(10)
	}()
	if !panicked && s.halFailure == nil {
		t.Fatal("replaced in-place GDN state inside a capture neither panicked nor failed the session")
	}
	if be.began != 1 || be.aborted != 1 || be.capturing {
		t.Fatalf("began=%d aborted=%d capturing=%v, want the open capture aborted once", be.began, be.aborted, be.capturing)
	}
}

// fak-test:runtime fast est=1s
func TestQwen35HybridGraphResetOnCloseAndCacheRebuild(t *testing.T) {
	m := NewSynthetic(qwen35GraphTestCfg())
	be := newTestQwenGraphBackend(m)
	s := newQwenGraphSession(t, m, be)
	_ = s.Step(5)
	_ = s.Step(6)
	if !s.halLogitsWarm {
		t.Fatal("logits steps did not warm the session")
	}
	before := be.resets
	if _, err := s.RebuildCacheGeometry(CacheGeometryRequest{KVCapacityTokens: 64, DeviceBudgetBytes: math.MaxInt64}); err != nil {
		t.Fatalf("RebuildCacheGeometry: %v", err)
	}
	if be.resets != before+1 || s.halLogitsWarm {
		t.Fatalf("cache rebuild resets=%d warm=%v, want one graph reset and an unwarmed session", be.resets-before, s.halLogitsWarm)
	}
	_ = s.Step(7) // re-warm on the new geometry, uncaptured
	_ = s.Step(8) // captured again
	before = be.resets
	s.Close()
	if be.resets != before+1 {
		t.Fatalf("session close made %d graph resets, want 1", be.resets-before)
	}

	idle := newQwenGraphSession(t, m, be)
	before = be.resets
	idle.Close()
	if be.resets != before {
		t.Fatalf("closing a session that never captured made %d graph resets, want 0 (the exec may be another session's)", be.resets-before)
	}
}
