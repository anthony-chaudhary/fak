package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type testQwenGraphBackend struct {
	*recordingQwen35Backend
	capturing   bool
	beganCount  int
	launchCount int
	abortCount  int
	resetCount  int
	declined    bool
}

func newTestQwenGraphBackend(m *Model) *testQwenGraphBackend {
	return &testQwenGraphBackend{
		recordingQwen35Backend: newRecordingQwen35Backend(m),
	}
}

func (b *testQwenGraphBackend) GraphBegin() bool {
	if b.declined {
		return false
	}
	b.capturing = true
	b.beganCount++
	return true
}

func (b *testQwenGraphBackend) GraphEndLaunch() {
	b.capturing = false
	b.launchCount++
}

func (b *testQwenGraphBackend) GraphAbort() {
	b.capturing = false
	b.abortCount++
}

func (b *testQwenGraphBackend) GraphReset() {
	b.capturing = false
	b.resetCount++
}

func (b *testQwenGraphBackend) IsCapturing() bool {
	return b.capturing
}

func (b *testQwenGraphBackend) SplitQwen35QueryGate(qg compute.Tensor, nHeads, headDim int) (compute.Tensor, compute.Tensor) {
	data := b.Backend.Read(qg)
	qData, gData := splitQwen35HeadInterleavedRows(data, nHeads, headDim, 1)
	q := compute.NewF32(b.Backend, []int{nHeads * headDim}, qData)
	g := compute.NewF32(b.Backend, []int{nHeads * headDim}, gData)
	return q, g
}

func (b *testQwenGraphBackend) PartialRoPEQK(
	q, k compute.Tensor,
	pos, nQHeads, nKHeads, headDim, rotaryDim int,
	theta float64,
) (compute.Tensor, compute.Tensor) {
	qHost := append([]float32(nil), b.Backend.Read(q)...)
	kHost := append([]float32(nil), b.Backend.Read(k)...)
	cos, sin := ropeRowForLayer(b.model.Cfg, 0, pos)
	ropeRowQKInto(qHost, kHost, cos, sin, headDim, nQHeads, nKHeads)
	qOut := compute.NewF32(b.Backend, []int{nQHeads * headDim}, qHost)
	kOut := compute.NewF32(b.Backend, []int{nKHeads * headDim}, kHost)
	return qOut, kOut
}

func (b *testQwenGraphBackend) SigmoidMulInPlace(x, gate compute.Tensor) {
	xData, ok := b.Backend.Host(x)
	gData := b.Backend.Read(gate)
	if ok {
		for i := range xData {
			xData[i] *= sigmoidf(gData[i])
		}
	}
}

// testQwenMissingRoPEBackend omits PartialRoPEQK to verify refusal when partial RoPE is unsupported.
type testQwenMissingRoPEBackend struct {
	*recordingQwen35Backend
	capturing   bool
	beganCount  int
	launchCount int
	abortCount  int
	resetCount  int
}

func newTestQwenMissingRoPEBackend(m *Model) *testQwenMissingRoPEBackend {
	return &testQwenMissingRoPEBackend{
		recordingQwen35Backend: newRecordingQwen35Backend(m),
	}
}

func (b *testQwenMissingRoPEBackend) GraphBegin() bool {
	b.capturing = true
	b.beganCount++
	return true
}

func (b *testQwenMissingRoPEBackend) GraphEndLaunch() {
	b.capturing = false
	b.launchCount++
}

func (b *testQwenMissingRoPEBackend) GraphAbort() {
	b.capturing = false
	b.abortCount++
}

func (b *testQwenMissingRoPEBackend) GraphReset() {
	b.capturing = false
	b.resetCount++
}

func (b *testQwenMissingRoPEBackend) IsCapturing() bool {
	return b.capturing
}

func (b *testQwenMissingRoPEBackend) SplitQwen35QueryGate(qg compute.Tensor, nHeads, headDim int) (compute.Tensor, compute.Tensor) {
	data := b.Backend.Read(qg)
	qData, gData := splitQwen35HeadInterleavedRows(data, nHeads, headDim, 1)
	q := compute.NewF32(b.Backend, []int{nHeads * headDim}, qData)
	g := compute.NewF32(b.Backend, []int{nHeads * headDim}, gData)
	return q, g
}

func (b *testQwenMissingRoPEBackend) SigmoidMulInPlace(x, gate compute.Tensor) {
	xData, ok := b.Backend.Host(x)
	gData := b.Backend.Read(gate)
	if ok {
		for i := range xData {
			xData[i] *= sigmoidf(gData[i])
		}
	}
}

// testQwenMissingSplitBackend omits SplitQwen35QueryGate to verify refusal when split-gate is unsupported.
type testQwenMissingSplitBackend struct {
	*recordingQwen35Backend
	capturing   bool
	beganCount  int
	launchCount int
	abortCount  int
	resetCount  int
}

func newTestQwenMissingSplitBackend(m *Model) *testQwenMissingSplitBackend {
	return &testQwenMissingSplitBackend{
		recordingQwen35Backend: newRecordingQwen35Backend(m),
	}
}

func (b *testQwenMissingSplitBackend) GraphBegin() bool {
	b.capturing = true
	b.beganCount++
	return true
}

func (b *testQwenMissingSplitBackend) GraphEndLaunch() {
	b.capturing = false
	b.launchCount++
}

func (b *testQwenMissingSplitBackend) GraphAbort() {
	b.capturing = false
	b.abortCount++
}

func (b *testQwenMissingSplitBackend) GraphReset() {
	b.capturing = false
	b.resetCount++
}

func (b *testQwenMissingSplitBackend) IsCapturing() bool {
	return b.capturing
}

func (b *testQwenMissingSplitBackend) PartialRoPEQK(
	q, k compute.Tensor,
	pos, nQHeads, nKHeads, headDim, rotaryDim int,
	theta float64,
) (compute.Tensor, compute.Tensor) {
	qHost := append([]float32(nil), b.Backend.Read(q)...)
	kHost := append([]float32(nil), b.Backend.Read(k)...)
	cos, sin := ropeRowForLayer(b.model.Cfg, 0, pos)
	ropeRowQKInto(qHost, kHost, cos, sin, headDim, nQHeads, nKHeads)
	qOut := compute.NewF32(b.Backend, []int{nQHeads * headDim}, qHost)
	kOut := compute.NewF32(b.Backend, []int{nKHeads * headDim}, kHost)
	return qOut, kOut
}

func (b *testQwenMissingSplitBackend) SigmoidMulInPlace(x, gate compute.Tensor) {
	xData, ok := b.Backend.Host(x)
	gData := b.Backend.Read(gate)
	if ok {
		for i := range xData {
			xData[i] *= sigmoidf(gData[i])
		}
	}
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
		normalizedInput,
		inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj,
		convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel,
		rmsNormEpsilon,
	)
	if err != nil {
		return output, nextConv, nextRec, err
	}
	if b.replaceAt > 0 && b.gdnCalls >= b.replaceAt {
		// Return a replacement tensor for convState that breaks the in-place contract.
		replacement := compute.NewF32(b.Backend, convState.Shape, make([]float32, convState.Numel()))
		return output, replacement, nextRec, nil
	}
	return output, nextConv, nextRec, nil
}

func TestQwen35HybridGraphSafeGating(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	be := newTestQwenGraphBackend(m)

	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()

	// 1. Initial valid hybrid session satisfies graph-safety.
	if !s.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be true for fully-featured backend session")
	}

	// 2. Scaled RoPE fails graph-safety.
	s.M.Cfg.RopeScaling = "linear"
	if s.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when RopeScaling is set")
	}
	s.M.Cfg.RopeScaling = ""

	// 3. LongRope fails graph-safety.
	s.M.Cfg.LongRope = &RopeScaling{Type: "longrope"}
	if s.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when LongRope is non-nil")
	}
	s.M.Cfg.LongRope = nil

	// 4. Split session (DenseGPULayers > 0 && < NumLayers) fails graph-safety.
	s.DenseGPULayers = 2
	if s.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when session is split across GPU and CPU")
	}
	s.DenseGPULayers = 0

	// 5. Missing partial RoPE capability fails graph-safety.
	beNoRoPE := newTestQwenMissingRoPEBackend(m)
	sNoRoPE, err := m.NewBackendSessionChecked(beNoRoPE)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(beNoRoPE): %v", err)
	}
	defer sNoRoPE.Close()
	if sNoRoPE.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when backend lacks qwen35PartialRoPEBackend")
	}

	// 6. Missing query-gate split capability fails graph-safety.
	beNoSplit := newTestQwenMissingSplitBackend(m)
	sNoSplit, err := m.NewBackendSessionChecked(beNoSplit)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(beNoSplit): %v", err)
	}
	defer sNoSplit.Close()
	if sNoSplit.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when backend lacks qwen35QueryGateSplitBackend")
	}

	// 7. Non-hybrid model fails graph-safety.
	nonHybridCfg := Config{
		HiddenSize:        16,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
	mNonHybrid := NewSynthetic(nonHybridCfg)
	sNonHybrid := mNonHybrid.NewBackendSession(compute.Default())
	defer sNonHybrid.Close()
	if sNonHybrid.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false for non-hybrid model")
	}

	// 8. Closed or failed session fails graph-safety.
	sClosed, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	sClosed.Close()
	if sClosed.qwen35HybridGraphSafe() {
		t.Fatal("expected qwen35HybridGraphSafe() to be false when session is closed")
	}
}

func TestQwen35HybridDecodeEagerVsGraphParity(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)

	// Graph-enabled backend session
	beGraph := newTestQwenGraphBackend(m)
	sGraph, err := m.NewBackendSessionChecked(beGraph)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(beGraph): %v", err)
	}
	defer sGraph.Close()

	// Eager backend session (declined capture)
	beEager := newTestQwenGraphBackend(m)
	beEager.declined = true
	sEager, err := m.NewBackendSessionChecked(beEager)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(beEager): %v", err)
	}
	defer sEager.Close()

	tokens := []int{3, 7, 11, 19}
	for step, token := range tokens {
		logitsGraph := sGraph.Step(token)
		logitsEager := sEager.Step(token)

		if len(logitsGraph) != len(logitsEager) {
			t.Fatalf("step %d logits length mismatch: graph=%d eager=%d", step, len(logitsGraph), len(logitsEager))
		}

		// Verify numerical parity between eager and graph execution
		var maxDelta float64
		for i := range logitsGraph {
			d := math.Abs(float64(logitsGraph[i] - logitsEager[i]))
			if d > maxDelta {
				maxDelta = d
			}
		}
		if maxDelta > 1e-5 {
			t.Fatalf("step %d logits parity delta=%g exceeds tolerance 1e-5", step, maxDelta)
		}
	}

	// After the warmup step (step 0), subsequent steps should capture and launch
	if beGraph.beganCount < 1 {
		t.Fatalf("expected graph capture to begin, beganCount=%d", beGraph.beganCount)
	}
	if beGraph.launchCount < 1 {
		t.Fatalf("expected graph launch to execute, launchCount=%d", beGraph.launchCount)
	}
	if beGraph.abortCount != 0 {
		t.Fatalf("expected no graph aborts, abortCount=%d", beGraph.abortCount)
	}

	// Eager session must never launch graph captures
	if beEager.launchCount != 0 {
		t.Fatalf("eager session launched %d graphs, want 0", beEager.launchCount)
	}
}

func TestQwen35HybridDecodeFallbackWhenUnsafe(t *testing.T) {
	t.Run("scaled-rope-fallback", func(t *testing.T) {
		cfg := qwen35HybridTestCfg()
		cfg.RopeScaling = "linear"
		m := NewSynthetic(cfg)
		be := newTestQwenGraphBackend(m)

		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatalf("NewBackendSessionChecked: %v", err)
		}
		defer s.Close()

		if s.qwen35HybridGraphSafe() {
			t.Fatal("expected qwen35HybridGraphSafe() to be false for scaled RoPE")
		}

		// Verify steps execute successfully via eager fallback without attempting graph capture
		for _, token := range []int{5, 12, 19} {
			logits := s.Step(token)
			if len(logits) != cfg.VocabSize {
				t.Fatalf("unexpected logits size: %d, want %d", len(logits), cfg.VocabSize)
			}
		}
		if be.beganCount != 0 {
			t.Fatalf("scaled RoPE attempted graph capture: beganCount=%d, want 0", be.beganCount)
		}
	})

	t.Run("missing-partial-rope-fallback", func(t *testing.T) {
		cfg := qwen35HybridTestCfg()
		m := NewSynthetic(cfg)
		be := newTestQwenMissingRoPEBackend(m)

		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatalf("NewBackendSessionChecked: %v", err)
		}
		defer s.Close()

		if s.qwen35HybridGraphSafe() {
			t.Fatal("expected qwen35HybridGraphSafe() to be false for missing partial RoPE")
		}

		for _, token := range []int{5, 12, 19} {
			logits := s.Step(token)
			if len(logits) != cfg.VocabSize {
				t.Fatalf("unexpected logits size: %d, want %d", len(logits), cfg.VocabSize)
			}
		}
		if be.beganCount != 0 {
			t.Fatalf("missing partial RoPE attempted graph capture: beganCount=%d, want 0", be.beganCount)
		}
	})

	t.Run("split-session-fallback", func(t *testing.T) {
		cfg := qwen35HybridTestCfg()
		m := NewSynthetic(cfg)
		be := newTestQwenGraphBackend(m)

		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatalf("NewBackendSessionChecked: %v", err)
		}
		defer s.Close()

		// Split 2 layers on GPU, 2 layers on CPU
		s.DenseGPULayers = 2
		if s.qwen35HybridGraphSafe() {
			t.Fatal("expected qwen35HybridGraphSafe() to be false for split session")
		}

		for _, token := range []int{5, 12} {
			logits := s.Step(token)
			if len(logits) != cfg.VocabSize {
				t.Fatalf("unexpected logits size: %d, want %d", len(logits), cfg.VocabSize)
			}
		}
		if be.beganCount != 0 {
			t.Fatalf("split session attempted graph capture: beganCount=%d, want 0", be.beganCount)
		}
	})
}

func TestQwen35HybridDecodeGraphAbortOnStateViolation(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	baseBE := newTestQwenGraphBackend(m)
	be := &testQwenReplaceStateBackend{
		testQwenGraphBackend: baseBE,
		replaceAt:            0, // Initially disabled
	}

	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()

	// Step 0: Warmup step (uncaptured)
	_ = s.Step(5)
	if be.beganCount != 0 {
		t.Fatalf("warmup step should not capture: beganCount=%d", be.beganCount)
	}

	// Arm replacement at step 1 during graph capture
	// Step 0 executed 3 GDN layers (gdnCalls = 3).
	// Step 1 will execute layers 0, 1, 2 (gdnCalls = 4, 5, 6).
	// Setting replaceAt = 4 triggers on layer 0 of step 1 while capture is active.
	be.replaceAt = 4

	var panicked bool
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		_ = s.Step(10)
	}()

	if !panicked {
		t.Fatal("expected forward to panic on replaced persistent in-place state")
	}

	// The panic must have cleanly aborted the open graph capture
	if be.abortCount != 1 {
		t.Fatalf("expected GraphAbort() to be called once: abortCount=%d", be.abortCount)
	}
	if be.capturing {
		t.Fatal("capturing state must be false after GraphAbort()")
	}
}
