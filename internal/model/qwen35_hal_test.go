package model

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var (
	errFakeGDNSequenceAlloc = errors.New("injected GDN sequence allocation failure")
	errFakeGDNSequenceRun   = errors.New("injected GDN sequence operation failure")
)

type fakeQwen35GDNSequenceBackend struct {
	path         string
	next         Qwen35GDNAuxHandle
	allocCalls   int
	runCalls     int
	failAllocAt  int
	failRun      bool
	replace      bool
	allocated    []Qwen35GDNAuxState
	replacements []Qwen35GDNAuxState
	requests     []Qwen35GDNPreprojectedSequenceRequest
	freed        map[Qwen35GDNAuxState]int
	freeErr      error
	freeHook     func()
}

func newFakeQwen35GDNSequenceBackend() *fakeQwen35GDNSequenceBackend {
	return &fakeQwen35GDNSequenceBackend{
		path:  Qwen35GDNPreprojectedSequencePath,
		next:  1,
		freed: make(map[Qwen35GDNAuxState]int),
	}
}

func (b *fakeQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequencePath() string {
	return b.path
}

func (b *fakeQwen35GDNSequenceBackend) NewQwen35GDNAuxState(_ int, _ Qwen35GDNSequenceGeometry) (Qwen35GDNAuxState, error) {
	b.allocCalls++
	if b.failAllocAt > 0 && b.allocCalls == b.failAllocAt {
		return Qwen35GDNAuxState{}, errFakeGDNSequenceAlloc
	}
	state := Qwen35GDNAuxState{Convolution: b.next, Recurrent: b.next + 1}
	b.next += 2
	b.allocated = append(b.allocated, state)
	return state, nil
}

func (b *fakeQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequence(req Qwen35GDNPreprojectedSequenceRequest) (Qwen35GDNPreprojectedSequenceResult, error) {
	b.runCalls++
	b.requests = append(b.requests, req)
	if b.failRun {
		return Qwen35GDNPreprojectedSequenceResult{}, errFakeGDNSequenceRun
	}
	state := req.State
	if b.replace {
		state = Qwen35GDNAuxState{Convolution: b.next, Recurrent: b.next + 1}
		b.next += 2
		b.replacements = append(b.replacements, state)
	}
	return Qwen35GDNPreprojectedSequenceResult{
		Core:  make([]float32, req.Tokens*req.Geometry.NumValueHeads*req.Geometry.ValueHeadDim),
		State: state,
	}, nil
}

func (b *fakeQwen35GDNSequenceBackend) FreeQwen35GDNAuxState(state Qwen35GDNAuxState) error {
	b.freed[state]++
	if b.freeHook != nil {
		b.freeHook()
	}
	return b.freeErr
}

type markerOnlyQwen35GDNSequence struct{ path string }

func (m markerOnlyQwen35GDNSequence) Qwen35GDNPreprojectedSequencePath() string { return m.path }

func linearQwen35Layers(cfg Config) int {
	n := 0
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if cfg.isLinearAttnLayer(layer) {
			n++
		}
	}
	return n
}

func assertEachAuxStateFreedOnce(t *testing.T, backend *fakeQwen35GDNSequenceBackend, states []Qwen35GDNAuxState) {
	t.Helper()
	for _, state := range states {
		if got := backend.freed[state]; got != 1 {
			t.Errorf("state %#v free count=%d, want exactly 1", state, got)
		}
	}
}

func TestQwen35GDNPreprojectedSequenceAdmissionIsFailClosed(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	s := m.NewSession()
	defer s.Close()

	if accepted, err := s.initQwen35GDNPreprojectedSequence(nil); accepted || err != nil {
		t.Fatalf("absent capability = accepted %v err %v, want false nil", accepted, err)
	}
	for name, candidate := range map[string]any{
		"marker-only": markerOnlyQwen35GDNSequence{path: Qwen35GDNPreprojectedSequencePath},
		"wrong-path": func() *fakeQwen35GDNSequenceBackend {
			b := newFakeQwen35GDNSequenceBackend()
			b.path = "wrong/path"
			return b
		}(),
	} {
		t.Run(name, func(t *testing.T) {
			accepted, err := s.initQwen35GDNPreprojectedSequence(candidate)
			if !accepted {
				t.Fatal("advertised capability was not recognized")
			}
			var unsupported *UnsupportedGDNPreprojectedSequenceError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error=%T %v, want typed pre-allocation refusal", err, err)
			}
			if backend, ok := candidate.(*fakeQwen35GDNSequenceBackend); ok && backend.allocCalls != 0 {
				t.Fatalf("refusal allocated %d state pairs", backend.allocCalls)
			}
		})
	}

	backend := newFakeQwen35GDNSequenceBackend()
	accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
	if !accepted || err != nil {
		t.Fatalf("exact capability = accepted %v err %v", accepted, err)
	}
	if s.Backend != nil {
		t.Fatalf("backend-neutral capability changed Session.Backend to %T", s.Backend)
	}
	if got, want := backend.allocCalls, linearQwen35Layers(m.Cfg); got != want {
		t.Fatalf("allocated states=%d, want one for each of %d linear layers", got, want)
	}
}

func TestQwen35GDNPreprojectedSequenceSessionIsolationStableIdentityAndNoRetry(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	backend := newFakeQwen35GDNSequenceBackend()
	a, b := m.NewSession(), m.NewSession()
	defer a.Close()
	defer b.Close()
	for _, s := range []*Session{a, b} {
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("attach = accepted %v err %v", accepted, err)
		}
	}

	seenA := make(map[Qwen35GDNAuxHandle]bool)
	for _, state := range a.qwen35HAL.sequenceLayers {
		if state.valid() {
			seenA[state.Convolution], seenA[state.Recurrent] = true, true
		}
	}
	for _, state := range b.qwen35HAL.sequenceLayers {
		if state.valid() && (seenA[state.Convolution] || seenA[state.Recurrent]) {
			t.Fatalf("sessions share auxiliary identity %#v", state)
		}
	}

	req := Qwen35GDNPreprojectedSequenceRequest{Layer: 0, Tokens: 2}
	for i := 0; i < 2; i++ {
		result, accepted, err := a.tryQwen35GDNPreprojectedSequence(req)
		if !accepted || err != nil {
			t.Fatalf("run %d = accepted %v err %v", i, accepted, err)
		}
		if result.State != a.qwen35HAL.sequenceLayers[0] {
			t.Fatalf("run %d changed state identity: %#v", i, result.State)
		}
	}
	if backend.requests[0].State != backend.requests[1].State {
		t.Fatalf("backend observed unstable handles: %#v then %#v", backend.requests[0].State, backend.requests[1].State)
	}

	backend.failRun = true
	before := backend.runCalls
	_, accepted, err := a.tryQwen35GDNPreprojectedSequence(req)
	var operation *Qwen35GDNSequenceOperationError
	if !accepted || !errors.As(err, &operation) || !errors.Is(err, errFakeGDNSequenceRun) {
		t.Fatalf("failed run = accepted %v err %T %v, want accepted typed failure", accepted, err, err)
	}
	if backend.runCalls != before+1 {
		t.Fatalf("failed operation calls=%d, want exactly one submit", backend.runCalls-before)
	}
	_, accepted, err = a.tryQwen35GDNPreprojectedSequence(req)
	if !accepted || !errors.As(err, &operation) || backend.runCalls != before+1 {
		t.Fatalf("poisoned session retried/fell back: accepted=%v err=%v runCalls=%d", accepted, err, backend.runCalls)
	}
	assertEachAuxStateFreedOnce(t, backend, backend.allocated[:linearQwen35Layers(m.Cfg)])
	for _, state := range backend.allocated[linearQwen35Layers(m.Cfg):] {
		if backend.freed[state] != 0 {
			t.Fatalf("failure in session A freed session B state %#v", state)
		}
	}
}

func TestQwen35GDNPreprojectedSequenceCleanupCloseResetAndAllocationFailure(t *testing.T) {
	t.Run("close-exactly-once", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := newFakeQwen35GDNSequenceBackend()
		s := m.NewSession()
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("attach = accepted %v err %v", accepted, err)
		}
		states := append([]Qwen35GDNAuxState(nil), backend.allocated...)
		s.Close()
		s.Close()
		assertEachAuxStateFreedOnce(t, backend, states)
	})

	t.Run("cleanup-error-and-reentry-stay-exactly-once", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := newFakeQwen35GDNSequenceBackend()
		s := m.NewSession()
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("attach = accepted %v err %v", accepted, err)
		}
		states := append([]Qwen35GDNAuxState(nil), backend.allocated...)
		backend.freeErr = errors.New("injected cleanup report")
		backend.freeHook = s.closeQwen35HALState
		s.Close()
		s.Close()
		assertEachAuxStateFreedOnce(t, backend, states)
	})

	t.Run("cache-reset", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := newFakeQwen35GDNSequenceBackend()
		s := m.NewSession()
		defer s.Close()
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("attach = accepted %v err %v", accepted, err)
		}
		states := append([]Qwen35GDNAuxState(nil), backend.allocated...)
		_, err := s.RebuildCacheGeometry(CacheGeometryRequest{KVCapacityTokens: 8, DeviceBudgetBytes: math.MaxInt64})
		if err != nil {
			t.Fatalf("RebuildCacheGeometry: %v", err)
		}
		assertEachAuxStateFreedOnce(t, backend, states)
	})

	t.Run("allocation-rollback", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := newFakeQwen35GDNSequenceBackend()
		backend.failAllocAt = 2
		s := m.NewSession()
		defer s.Close()
		accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
		var operation *Qwen35GDNSequenceOperationError
		if !accepted || !errors.As(err, &operation) || !errors.Is(err, errFakeGDNSequenceAlloc) {
			t.Fatalf("failed attach = accepted %v err %T %v", accepted, err, err)
		}
		if s.qwen35HAL != nil {
			t.Fatalf("failed transaction attached state: %#v", s.qwen35HAL)
		}
		assertEachAuxStateFreedOnce(t, backend, backend.allocated)
	})

	t.Run("replacement-state-fails-closed", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := newFakeQwen35GDNSequenceBackend()
		backend.replace = true
		s := m.NewSession()
		defer s.Close()
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("attach = accepted %v err %v", accepted, err)
		}
		_, accepted, err := s.tryQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequenceRequest{Layer: 0, Tokens: 1})
		var operation *Qwen35GDNSequenceOperationError
		if !accepted || !errors.As(err, &operation) {
			t.Fatalf("replacement = accepted %v err %T %v", accepted, err, err)
		}
		assertEachAuxStateFreedOnce(t, backend, backend.allocated)
		assertEachAuxStateFreedOnce(t, backend, backend.replacements)
	})
}

func TestQwen35GDNPreprojectedSequenceLeavesDefaultAndCUDALifecyclesUnchanged(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	legacy := m.NewSession()
	if legacy.Backend != nil || legacy.qwen35HAL != nil {
		t.Fatalf("default session unexpectedly activated capability: backend=%T state=%#v", legacy.Backend, legacy.qwen35HAL)
	}
	legacy.Close()

	cuda := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(cuda)
	if err != nil {
		t.Fatalf("existing CUDA session refused: %v", err)
	}
	if s.qwen35HAL == nil || s.qwen35HAL.backend != cuda || s.qwen35HAL.sequenceAccepted {
		t.Fatalf("CUDA state changed: %#v", s.qwen35HAL)
	}
	var cudaStateBuffers []compute.Buffer
	for _, state := range s.qwen35HAL.layers {
		if state.conv.Buf() != nil {
			cudaStateBuffers = append(cudaStateBuffers, state.conv.Buf(), state.recurrent.Buf())
		}
	}
	s.Close()
	s.Close()
	for _, buffer := range cudaStateBuffers {
		if got := cuda.freeCalls[buffer]; got != 1 {
			t.Errorf("existing CUDA state buffer %p free count=%d, want exactly 1", buffer, got)
		}
	}
}

type qkNormRecordingBackend struct {
	*recordingQwen35Backend
	uploadedBySite map[string][]float32
}

func newQKNormRecordingBackend(m *Model) *qkNormRecordingBackend {
	return &qkNormRecordingBackend{
		recordingQwen35Backend: newRecordingQwen35Backend(m),
		uploadedBySite:         make(map[string][]float32),
	}
}

func (b *qkNormRecordingBackend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	data := b.Backend.Read(t)
	b.uploadedBySite[site] = append([]float32(nil), data...)
	return b.recordingQwen35Backend.UploadClass(t, as, class, site)
}

func TestQwen35FullAttentionHAL_QKNorm(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = 0.25
	cfg.RopeTheta = 10_000_000
	cfg.RMSNormEps = 1e-5
	cfg.QKNorm = false

	m := NewSynthetic(cfg)
	fullLayer := 3
	qVals := make([]float32, cfg.HeadDim)
	kVals := make([]float32, cfg.HeadDim)
	for i := 0; i < cfg.HeadDim; i++ {
		qVals[i] = 0.85 + 0.3*float32(i+1)/float32(cfg.HeadDim)
		kVals[i] = 0.90 + 0.2*float32(i+1)/float32(cfg.HeadDim)
	}
	tpInjectTensors(m, map[string]tpTensor{
		layerName(fullLayer, "self_attn.q_norm.weight"): {shape: []int{cfg.HeadDim}, vals: qVals},
		layerName(fullLayer, "self_attn.k_norm.weight"): {shape: []int{cfg.HeadDim}, vals: kVals},
	})

	pos := 0
	eps := float32(cfg.RMSNormEps)
	scale := cfg.attnScale()
	grp := cfg.GroupSize()
	residualHost := make([]float32, cfg.HiddenSize)
	for i := range residualHost {
		residualHost[i] = float32(i+1) * 0.1
	}

	// 1. Prove that when QKNorm: false, no QK normalization is applied.
	m.Cfg.QKNorm = false
	beFalse := newQKNormRecordingBackend(m)
	sFalse, err := m.NewBackendSessionChecked(beFalse)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(false): %v", err)
	}
	defer sFalse.Close()

	resFalse := beFalse.Upload(compute.NewF32(beFalse, []int{cfg.HiddenSize}, append([]float32(nil), residualHost...)), compute.F32)
	sFalse.qwen35FullAttentionHAL(fullLayer, pos, resFalse, eps, scale, grp)

	if recordedClassSite(beFalse.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-q") {
		t.Fatal("QKNorm=false uploaded qwen35-full-attn-norm-q")
	}
	if recordedClassSite(beFalse.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-k") {
		t.Fatal("QKNorm=false uploaded qwen35-full-attn-norm-k")
	}
	keysFalse := beFalse.Read(sFalse.halKV.KeysView(0))

	// 2. Prove that when QKNorm: true with explicit QKNormEps, normalization is applied before RoPE and matches CPU reference.
	m.Cfg.QKNorm = true
	m.Cfg.QKNormEps = 2e-4 // explicitly distinct from RMSNormEps (1e-5)
	beTrue := newQKNormRecordingBackend(m)
	sTrue, err := m.NewBackendSessionChecked(beTrue)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(true): %v", err)
	}
	defer sTrue.Close()

	resTrue := beTrue.Upload(compute.NewF32(beTrue, []int{cfg.HiddenSize}, append([]float32(nil), residualHost...)), compute.F32)
	sTrue.qwen35FullAttentionHAL(fullLayer, pos, resTrue, eps, scale, grp)

	if recordedClassSite(beTrue.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-q") {
		t.Fatal("QKNorm=true bounced qwen35-full-attn-norm-q to host")
	}
	if recordedClassSite(beTrue.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-k") {
		t.Fatal("QKNorm=true bounced qwen35-full-attn-norm-k to host")
	}

	// Compute unnormalized Q and K directly from the original residual via the layer's projection to prove CPU reference match
	p := func(suffix string) string { return layerName(fullLayer, suffix) }
	resOrig := beTrue.Upload(compute.NewF32(beTrue, []int{cfg.HiddenSize}, append([]float32(nil), residualHost...)), compute.F32)
	xn := beTrue.Read(beTrue.RMSNorm(resOrig, sTrue.normWeightHAL(p("input_layernorm.weight")), eps))
	qWeight, _ := sTrue.qwen35QueryWeightsHAL(fullLayer)
	xnTensor := compute.NewF32(beTrue, []int{cfg.HiddenSize}, xn)
	qRaw := beTrue.Read(beTrue.MatMul(qWeight, xnTensor))
	kRaw := beTrue.Read(beTrue.MatMul(sTrue.matWeightHAL(p("self_attn.k_proj.weight")), xnTensor))

	qExpected := append([]float32(nil), qRaw...)
	kExpected := append([]float32(nil), kRaw...)
	m.applyLayerQKNorm(fullLayer, qExpected, kExpected)

	// Verify that normalized K is passed to RoPE and stored in KV cache:
	// Apply RoPE on kExpected and verify it matches the keys in the HAL KV store.
	keysTrue := beTrue.Read(sTrue.halKV.KeysView(0))
	cos, sin := ropeRowForLayer(cfg, fullLayer, pos)
	qDummy := make([]float32, len(qExpected))
	kRopeExpected := append([]float32(nil), kExpected...)
	ropeRowQKInto(qDummy, kRopeExpected, cos, sin, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads)

	if d := maxAbsDelta(keysTrue, kRopeExpected); d > 1e-6 {
		t.Fatalf("HAL KV keys differ from post-norm RoPE keys, max|delta|=%g", d)
	}
	if d := maxAbsDelta(keysTrue, keysFalse); d < 1e-3 {
		t.Fatalf("HAL KV keys with QKNorm do not differ from QKNorm=false keys, max|delta|=%g", d)
	}

	// 3. Prove full session parity between HAL session and CPU reference session with QKNorm: true
	prompt := []int{3, 7, 11, 5}
	cpuRef := m.NewSession()
	defer cpuRef.Close()
	wantPrefill := cpuRef.Prefill(prompt)

	halSession, err := m.NewBackendSessionChecked(newRecordingQwen35Backend(m))
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(halSession): %v", err)
	}
	defer halSession.Close()
	gotPrefill := halSession.Prefill(prompt)

	if d := maxAbsDelta(wantPrefill, gotPrefill); d > 2e-5 {
		t.Fatalf("full-session Prefill differs between HAL and CPU reference under QKNorm, max|delta|=%g", d)
	}
}

func TestQwen35HALQKNormMissingResidentPathFailsClosed(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.QKNorm = false
	m := NewSynthetic(cfg)
	fullLayer := 3
	qVals := make([]float32, cfg.HeadDim)
	kVals := make([]float32, cfg.HeadDim)
	for i := range qVals {
		qVals[i] = 0.9 + float32(i)/100
		kVals[i] = 1.1 - float32(i)/100
	}
	tpInjectTensors(m, map[string]tpTensor{
		layerName(fullLayer, "self_attn.q_norm.weight"): {shape: []int{cfg.HeadDim}, vals: qVals},
		layerName(fullLayer, "self_attn.k_norm.weight"): {shape: []int{cfg.HeadDim}, vals: kVals},
	})
	m.Cfg.QKNorm = true
	m.Cfg.QKNormPerHeadWeight = true
	be := newQKNormRecordingBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()
	q := be.Upload(compute.NewF32(compute.Default(), []int{cfg.NumHeads * cfg.HeadDim}, make([]float32, cfg.NumHeads*cfg.HeadDim)), compute.F32)
	k := be.Upload(compute.NewF32(compute.Default(), []int{cfg.NumKVHeads * cfg.HeadDim}, make([]float32, cfg.NumKVHeads*cfg.HeadDim)), compute.F32)

	_, _, err = s.qwen35ResidentQKNorm(fullLayer, q, k)
	var residentErr *Qwen35QKNormResidencyError
	if !errors.As(err, &residentErr) {
		t.Fatalf("error=%T %v, want typed Qwen35QKNormResidencyError", err, err)
	}
	if residentErr.Layer != fullLayer || be.readCalls != 0 || be.hostCalls != 0 {
		t.Fatalf("refusal layer=%d read_calls=%d host_calls=%d, want %d/0/0", residentErr.Layer, be.readCalls, be.hostCalls, fullLayer)
	}
	if recordedClassSite(be.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-q") ||
		recordedClassSite(be.recordingQwen35Backend, compute.MemoryActivation, "qwen35-full-attn-norm-k") {
		t.Fatal("typed resident refusal must not re-upload Q/K activations")
	}
}

func TestQwen35QSASparseRowGather(t *testing.T) {
	// Requirements verified:
	// - N_kv >= 16,384 triggers QSA sparse row gather
	// - N_kv < 16,384 preserves dense attention
	// - Multi-sequence (batch > 1) preserves dense attention
	// - Non-QSA layers preserve dense attention
	// - Scratch size fits in 32MB MALL Infinity Cache
	// - Numerical equivalence (perplexity delta <= 0.05% / QSAPerplexityDeltaTolerance) via ValidateQSAPerplexityParity

	cfg := Config{
		HiddenSize:       256,
		HeadDim:          64,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 512,
		VocabSize:        1000,
		NumLayers:        4,
		LayerTypes: []string{
			"linear_attention",
			"linear_attention",
			"linear_attention",
			"full_attention",
		},
		LinearConvKernelDim: 3,
		LinearKeyHeadDim:    64,
		LinearNumKeyHeads:   2,
		LinearValueHeadDim:  64,
		LinearNumValueHeads: 4,
		AttnOutputGate:      false,
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
	}

	fullLayer := 3

	// 1. Non-QSA layers preserve dense attention
	for l := 0; l < 3; l++ {
		if cfg.ShouldUseQSASparseGather(l, 16384, 1) {
			t.Errorf("layer %d (linear) should not use QSA gather", l)
		}
		if cfg.ShouldUseQSASparseGather(l, 20000, 1) {
			t.Errorf("layer %d (linear) should not use QSA gather at 20k", l)
		}
	}

	// 2. Multi-sequence (batch > 1) preserves dense attention
	if cfg.ShouldUseQSASparseGather(fullLayer, 16384, 2) {
		t.Error("batchSize=2 should not use QSA gather (multi-sequence safety)")
	}
	if cfg.ShouldUseQSASparseGather(fullLayer, 20000, 4) {
		t.Error("batchSize=4 should not use QSA gather (multi-sequence safety)")
	}

	// 3. N_kv < 16,384 preserves dense attention
	if cfg.ShouldUseQSASparseGather(fullLayer, 10000, 1) {
		t.Error("nKV=10000 should not use QSA gather (dynamic gating)")
	}
	if cfg.ShouldUseQSASparseGather(fullLayer, 16383, 1) {
		t.Error("nKV=16383 should not use QSA gather (boundary)")
	}

	// 4. N_kv >= 16,384 triggers QSA sparse row gather
	if !cfg.ShouldUseQSASparseGather(fullLayer, 16384, 1) {
		t.Error("nKV=16384 should use QSA gather")
	}
	if !cfg.ShouldUseQSASparseGather(fullLayer, 20000, 1) {
		t.Error("nKV=20000 should use QSA gather")
	}

	m := NewSynthetic(cfg)
	be := newQKNormRecordingBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()

	totalTokens := 16384
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	grp := nH / nKV
	w := nKV * hd
	scale := cfg.attnScale()
	eps := float32(cfg.RMSNormEps)
	kvLayer := qwen35HALKVLayer(cfg, fullLayer)

	// Query vector Q
	Q := make([]float32, nH*hd)
	for i := 0; i < nH*hd; i++ {
		Q[i] = float32(math.Sin(float64(i+1)*0.1)) + 0.5
	}

	// Populate s.halKV
	// High affinity keys align with Q; low affinity oppose Q
	highK := make([]float32, w)
	lowK := make([]float32, w)
	val := make([]float32, w)
	for i := 0; i < w; i++ {
		h := i / hd
		dim := i % hd
		highK[i] = Q[h*hd+dim]
		lowK[i] = -Q[h*hd+dim] * 2.0
		val[i] = float32(math.Cos(float64(i) * 0.013))
	}

	highKTensor := be.Upload(compute.NewF32(be, []int{w}, highK), compute.F32)
	lowKTensor := be.Upload(compute.NewF32(be, []int{w}, lowK), compute.F32)
	valTensor := be.Upload(compute.NewF32(be, []int{w}, val), compute.F32)

	blockSize := compute.QSABlockSize
	totalBlocks := (totalTokens + blockSize - 1) / blockSize
	topKBlocks := compute.QSABaseTopKTokens / blockSize
	tailBlocks := compute.QSALocalTailTokens / blockSize

	for p := 0; p < totalTokens; p++ {
		b := p / blockSize
		isSelected := (b < topKBlocks) || (b >= totalBlocks-tailBlocks)
		if isSelected {
			s.halKV.AppendKV(kvLayer, highKTensor, highKTensor, valTensor, p)
		} else {
			s.halKV.AppendKV(kvLayer, lowKTensor, lowKTensor, valTensor, p)
		}
	}

	if s.halKV.Len() != totalTokens {
		t.Fatalf("halKV length = %d, want %d", s.halKV.Len(), totalTokens)
	}

	// Reference: dense masked attention output
	qTensor := be.Upload(compute.NewF32(be, []int{nH * hd}, append([]float32(nil), Q...)), compute.F32)
	denseOutTensor := be.Attention(qTensor, s.halKV, kvLayer, true, grp, scale)
	denseScores := be.Read(denseOutTensor)

	// QSA sparse attention execution via HAL
	sparseOutTensor, err := s.qwen35QSASparseAttentionHAL(fullLayer, kvLayer, qTensor, grp, scale)
	if err != nil {
		t.Fatalf("qwen35QSASparseAttentionHAL failed: %v", err)
	}
	sparseScores := be.Read(sparseOutTensor)

	// Verify QSA sparse row gather triggered
	receipt, ok := s.LastQSABlockSelectionReceipt()
	if !ok {
		t.Fatal("LastQSABlockSelectionReceipt() returned false, want true")
	}
	if receipt.DynamicGatingBypassed {
		t.Fatal("receipt.DynamicGatingBypassed must be false for 16,384 tokens")
	}
	if receipt.SelectedBlocks != 36 {
		t.Fatalf("receipt.SelectedBlocks = %d, want 36 (32 top-k + 4 tail)", receipt.SelectedBlocks)
	}

	// Verify scratch size fits in 32MB MALL Infinity Cache
	scratchBytes, ok := s.LastQSAScratchBytes()
	if !ok {
		t.Fatal("LastQSAScratchBytes() returned false, want true")
	}
	if scratchBytes > compute.StrixHaloInfinityCacheBytes {
		t.Fatalf("scratchBytes %d exceeds 32MB MALL Infinity Cache cap (%d)",
			scratchBytes, compute.StrixHaloInfinityCacheBytes)
	}
	t.Logf("QSA scratch memory: %d bytes (%.2f MB <= 32MB MALL cap %d)",
		scratchBytes, float64(scratchBytes)/(1024*1024), compute.StrixHaloInfinityCacheBytes)

	// Verify numerical equivalence (perplexity delta <= 0.05% / QSAPerplexityDeltaTolerance) via ValidateQSAPerplexityParity
	relL2, err := ValidateQSAPerplexityParity(denseScores, sparseScores)
	if err != nil {
		t.Fatalf("ValidateQSAPerplexityParity returned error: %v", err)
	}
	if relL2 > compute.QSAPerplexityDeltaTolerance {
		t.Fatalf("ValidateQSAPerplexityParity relL2 %e > QSAPerplexityDeltaTolerance %e (0.05%%)",
			relL2, compute.QSAPerplexityDeltaTolerance)
	}
	t.Logf("QSA sparse vs dense perplexity delta = %e (tolerance = %e, <= 0.05%%)",
		relL2, compute.QSAPerplexityDeltaTolerance)

	// Also verify end-to-end qwen35FullAttentionHAL execution
	residualHost := make([]float32, cfg.HiddenSize)
	for i := range residualHost {
		residualHost[i] = float32(i+1) * 0.01
	}
	residualTensor := be.Upload(compute.NewF32(be, []int{cfg.HiddenSize}, residualHost), compute.F32)
	s.qwen35FullAttentionHAL(fullLayer, s.halKV.Len()-1, residualTensor, eps, scale, grp)

	fullReceipt, ok := s.LastQSABlockSelectionReceipt()
	if !ok || fullReceipt.DynamicGatingBypassed {
		t.Fatal("qwen35FullAttentionHAL failed to trigger QSA sparse row gather")
	}
}

// TestQwen35MTPDepth4SpeculativeLoop witnesses native MTP depth K=4 causal tree verification
// evaluating 4 candidate tokens in parallel during a single base-model weight read pass.
func TestQwen35MTPDepth4SpeculativeLoop(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	ctx := context.Background()

	prompt := []int{5, 12, 19, 26}
	s := m.NewSession()
	_ = s.Prefill(prompt)
	basePos := s.Cache.Len()

	// 1. Execute speculative verification pass with K=4 draft proposals
	drafts := [4]int{31, 37, 41, 43}
	res, err := s.Qwen35MTPDepth4CausalTreeVerifyResult(ctx, drafts)
	if err != nil {
		t.Fatalf("Qwen35MTPDepth4CausalTreeVerifyResult failed: %v", err)
	}

	// 2. Validate K=4 and single-pass execution contract
	if res.DraftDepthK != 4 {
		t.Errorf("DraftDepthK = %d, want 4", res.DraftDepthK)
	}
	if !res.SinglePass {
		t.Errorf("SinglePass = false, want true (single weight read pass)")
	}

	// 3. Validate packed 4x4 causal verification tree mask in LDS
	canonicalMask := compute.MTPK4CausalVerificationTreeMask()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if res.TreeMask[i][j] != canonicalMask[i][j] {
				t.Fatalf("TreeMask[%d][%d] = %f, want %f", i, j, res.TreeMask[i][j], canonicalMask[i][j])
			}
			if j <= i && res.TreeMask[i][j] != 1.0 {
				t.Errorf("causal connection masked at (%d, %d)", i, j)
			}
			if j > i && res.TreeMask[i][j] != 0.0 {
				t.Errorf("non-causal connection unmasked at (%d, %d)", i, j)
			}
		}
	}

	// 4. Validate Strix Halo single-pass weight reuse and 40 CU occupancy
	if !res.Audit.CausalTreeMaskApplied {
		t.Errorf("Audit.CausalTreeMaskApplied = false, want true")
	}
	if res.Audit.ComputeUnitsEngaged != compute.StrixHaloComputeUnits {
		t.Errorf("Audit.ComputeUnitsEngaged = %d, want %d", res.Audit.ComputeUnitsEngaged, compute.StrixHaloComputeUnits)
	}
	if res.Audit.WeightReuseRatio < 1.0 {
		t.Errorf("Audit.WeightReuseRatio = %f, want >= 1.0", res.Audit.WeightReuseRatio)
	}

	// 5. Validate atomic rollback and KV cache invariant: Len == basePos + accepted
	if s.Cache.Len() != basePos+res.AcceptedCount {
		t.Errorf("s.Cache.Len() = %d, want basePos (%d) + accepted (%d) = %d",
			s.Cache.Len(), basePos, res.AcceptedCount, basePos+res.AcceptedCount)
	}
	if res.RollbackCount != 4-res.AcceptedCount {
		t.Errorf("RollbackCount = %d, want %d", res.RollbackCount, 4-res.AcceptedCount)
	}

	// 6. Validate sustained throughput scaling on >= 80% draft acceptance
	if res.AcceptanceRate >= 0.80 && res.ThroughputTokS < 34.8 {
		t.Errorf("throughput = %f tok/s, want >= 34.8 tok/s at >= 80%% acceptance", res.ThroughputTokS)
	}

	// 7. Verify helper Qwen35MTPDepth4CausalTreeVerify returns matched counts
	s2 := m.NewSession()
	_ = s2.Prefill(prompt)
	acc2, next2, err2 := s2.Qwen35MTPDepth4CausalTreeVerify(ctx, drafts)
	if err2 != nil {
		t.Fatalf("s2.Qwen35MTPDepth4CausalTreeVerify failed: %v", err2)
	}
	if acc2 != res.AcceptedCount || len(next2) != len(res.NextTokens) {
		t.Errorf("helper mismatch: acc2=%d want %d, len(next2)=%d want %d",
			acc2, res.AcceptedCount, len(next2), len(res.NextTokens))
	}

	// 8. Verify Model method wrapper
	s3 := m.NewSession()
	_ = s3.Prefill(prompt)
	acc3, next3, err3 := m.Qwen35MTPDepth4CausalTreeVerify(ctx, s3, drafts)
	if err3 != nil {
		t.Fatalf("m.Qwen35MTPDepth4CausalTreeVerify failed: %v", err3)
	}
	if acc3 != res.AcceptedCount || len(next3) != len(res.NextTokens) {
		t.Errorf("model wrapper mismatch: acc3=%d want %d, len(next3)=%d want %d",
			acc3, res.AcceptedCount, len(next3), len(res.NextTokens))
	}

	// 9. Verify context cancellation fail-closed
	cancCtx, cancel := context.WithCancel(context.Background())
	cancel()
	_, errCanc := s.Qwen35MTPDepth4CausalTreeVerifyResult(cancCtx, drafts)
	if errCanc == nil {
		t.Errorf("expected error on cancelled context, got nil")
	}
}
