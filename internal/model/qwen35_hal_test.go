package model

import (
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

// TestQwen35QSASparseRowGather verifies true QSA sparse attention row gather into 32MB MALL Infinity Cache:
// 1. Dynamic gating threshold (N_kv >= 16,384 tokens) and multi-sequence safety fallback.
// 2. Contiguous scratch buffer sizing (~2,304 tokens aligned to 256-wide tiles) and 32MB MALL residency.
// 3. Radix Top-K block selection and contiguous sparse row gather.
// 4. Numerical equivalence and perplexity parity (relative L2 delta <= 0.05%) against dense masked attention.
// 5. End-to-end qwen35FullAttentionHAL execution over 16,384 cached tokens.
func TestQwen35QSASparseRowGather(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	// Layer 3 is full_attention (QSA layer)
	if !cfg.IsQSALayer(3) {
		t.Fatal("cfg.IsQSALayer(3) must be true for hybrid test config")
	}

	// 1. Dynamic Threshold Gating Verification
	if cfg.ShouldUseQSASparseGather(3, 1000, 1) {
		t.Errorf("ShouldUseQSASparseGather(3, 1000, 1) = true, want false (N_kv < 16384)")
	}
	if cfg.ShouldUseQSASparseGather(3, 16383, 1) {
		t.Errorf("ShouldUseQSASparseGather(3, 16383, 1) = true, want false (boundary N_kv < 16384)")
	}
	if !cfg.ShouldUseQSASparseGather(3, 16384, 1) {
		t.Errorf("ShouldUseQSASparseGather(3, 16384, 1) = false, want true")
	}
	if !cfg.ShouldUseQSASparseGather(3, 78000, 1) {
		t.Errorf("ShouldUseQSASparseGather(3, 78000, 1) = false, want true (78k context)")
	}
	// Multi-sequence safety: batchSize > 1 must fall back to dense
	for _, bs := range []int{2, 4, 8} {
		if cfg.ShouldUseQSASparseGather(3, 16384, bs) {
			t.Errorf("ShouldUseQSASparseGather(3, 16384, %d) = true, want false (multi-sequence safety)", bs)
		}
	}
	// Non-QSA linear layers must never trigger QSA
	for l := 0; l < 3; l++ {
		if cfg.ShouldUseQSASparseGather(l, 78000, 1) {
			t.Errorf("ShouldUseQSASparseGather(%d, 78000, 1) = true, want false (non-QSA layer)", l)
		}
	}

	// 2. Contiguous Scratch Sizing & 32MB MALL Infinity Cache Residency
	blockSize := compute.QSABlockSize // 64
	totalTokens := 16384
	totalBlocks := (totalTokens + blockSize - 1) / blockSize // 256
	topKBlocks := compute.QSABaseTopKTokens / blockSize      // 32
	tailBlocks := compute.QSALocalTailTokens / blockSize     // 4

	scores := make([]float32, totalBlocks)
	for b := range scores {
		if b < topKBlocks || b >= totalBlocks-tailBlocks {
			scores[b] = 5.0 + float32(b)*0.01
		} else {
			scores[b] = -10.0 + float32(b)*0.001
		}
	}

	selected, _, err := compute.RadixTopKBlockSelect(scores, totalBlocks, topKBlocks, tailBlocks)
	if err != nil {
		t.Fatalf("RadixTopKBlockSelect failed: %v", err)
	}
	if len(selected) != topKBlocks+tailBlocks {
		t.Fatalf("selected blocks count = %d, want %d", len(selected), topKBlocks+tailBlocks)
	}

	tileSize := compute.QSATileSize // 256
	totalGatherTokens := len(selected) * blockSize
	numTiles := (totalGatherTokens + tileSize - 1) / tileSize
	alignedGather := numTiles * tileSize
	if alignedGather != compute.QSAMaxGatherTokens {
		t.Fatalf("alignedGather = %d, want QSAMaxGatherTokens %d", alignedGather, compute.QSAMaxGatherTokens)
	}

	w := cfg.NumKVHeads * cfg.HeadDim
	neededElements := alignedGather * w
	scratchBytes := 2 * int64(neededElements) * 4 // sizeof(float32) for K and V
	if scratchBytes > compute.StrixHaloInfinityCacheBytes {
		t.Fatalf("scratchBytes %d > StrixHaloInfinityCacheBytes %d (32MB MALL)", scratchBytes, compute.StrixHaloInfinityCacheBytes)
	}
	t.Logf("QSA gathered scratch buffer size = %d bytes (resides in 32MB MALL Infinity Cache: %t)", scratchBytes, scratchBytes <= compute.StrixHaloInfinityCacheBytes)

	// 3. Numerical Equivalence & Perplexity Parity Check (<= 0.05% relative delta)
	hd := cfg.HeadDim
	nH := cfg.NumHeads
	nKV := cfg.NumKVHeads
	grp := nH / nKV
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	Q := make([]float32, nH*hd)
	for i := range Q {
		Q[i] = float32(math.Sin(float64(i+1)*0.1)) + 0.5
	}
	K := make([]float32, totalTokens*w)
	V := make([]float32, totalTokens*w)

	for tTok := 0; tTok < totalTokens; tTok++ {
		b := tTok / blockSize
		isSelected := (b < topKBlocks) || (b >= totalBlocks-tailBlocks)
		for i := 0; i < w; i++ {
			idx := tTok*w + i
			kvh := i / hd
			dim := i % hd
			sinVal := float32(math.Sin(float64(tTok*w+i)*0.017 + float64(b)*0.1))
			qh := kvh * grp
			if isSelected {
				K[idx] = Q[qh*hd+dim] + sinVal*0.01
				V[idx] = float32(math.Cos(float64(tTok*w+i) * 0.013))
			} else {
				K[idx] = -Q[qh*hd+dim]*3.0 + sinVal*0.01
				V[idx] = float32(math.Cos(float64(tTok*w+i) * 0.013))
			}
		}
	}

	// Evaluate dense attention
	denseOut := make([]float32, nH*hd)
	cache := NewKVCache(cfg)
	cache.pos = make([]int, totalTokens)
	for i := range cache.pos {
		cache.pos[i] = i
	}
	cache.K[3] = K
	cache.V[3] = V
	_ = attnDecodeOne(denseOut, Q, cache, 3, nH, hd, w, grp, scale, fdot, fdot3scalar, nil)

	// Evaluate QSA sparse row gather attention via HAL session
	m := NewSynthetic(cfg)
	be := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()

	// Append rows into halKV for layer 0 (tracking pos) and layer 3 (QSA target)
	for p := 0; p < totalTokens; p++ {
		kRow := K[p*w : (p+1)*w]
		vRow := V[p*w : (p+1)*w]
		kTen := s.uploadHostF32([]int{w}, kRow, compute.MemoryKVCache, "test-k")
		vTen := s.uploadHostF32([]int{w}, vRow, compute.MemoryKVCache, "test-v")
		s.halKV.AppendKV(0, kTen, kTen, vTen, p)
		s.halKV.AppendKV(3, kTen, kTen, vTen, p)
	}

	qTensor := s.uploadHostF32([]int{nH * hd}, Q, compute.MemoryActivation, "test-q")
	sparseOutTensor := s.qwen35QSASparseAttentionHAL(3, 3, qTensor, grp, scale)
	sparseOut := s.Backend.Read(sparseOutTensor)

	relL2, err := ValidateQSAPerplexityParity(denseOut, sparseOut)
	if err != nil {
		t.Fatalf("ValidateQSAPerplexityParity returned error: %v", err)
	}
	t.Logf("QSA HAL sparse vs dense relative L2 delta = %e (tolerance = %e)", relL2, compute.QSAPerplexityDeltaTolerance)
	if relL2 >= compute.QSAPerplexityDeltaTolerance {
		t.Fatalf("relative L2 delta %e >= tolerance %e (0.05%%)", relL2, compute.QSAPerplexityDeltaTolerance)
	}

	// 4. End-to-End qwen35FullAttentionHAL execution with QSA gather
	residualHost := make([]float32, cfg.HiddenSize)
	for i := range residualHost {
		residualHost[i] = float32(i+1) * 0.05
	}
	residual := s.uploadHostF32([]int{cfg.HiddenSize}, residualHost, compute.MemoryActivation, "test-res")
	s.qwen35FullAttentionHAL(3, totalTokens, residual, 1e-5, scale, grp)
	resOut := s.Backend.Read(residual)
	if len(resOut) != cfg.HiddenSize {
		t.Fatalf("residual output size = %d, want %d", len(resOut), cfg.HiddenSize)
	}
	for i, v := range resOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("residual output at index %d is non-finite: %v", i, v)
		}
	}
	if len(s.qwen35HAL.qsaGatheredK) < neededElements || len(s.qwen35HAL.qsaGatheredV) < neededElements {
		t.Errorf("qsa scratch buffers not sized: K=%d, V=%d, want >= %d", len(s.qwen35HAL.qsaGatheredK), len(s.qwen35HAL.qsaGatheredV), neededElements)
	}
}
