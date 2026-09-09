package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type refusingQwen35CUDA struct {
	compute.Backend
	matmuls   int
	attention int
}

func (b *refusingQwen35CUDA) Name() string                    { return "cuda" }
func (b *refusingQwen35CUDA) Tier() string                    { return "sm_80" }
func (b *refusingQwen35CUDA) Class() compute.CorrectnessClass { return compute.Approx }
func (b *refusingQwen35CUDA) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *refusingQwen35CUDA) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.attention++
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}

type markerOnlyQwen35CUDA struct{ *refusingQwen35CUDA }

func (*markerOnlyQwen35CUDA) Qwen35GDNPath() string { return Qwen35GDNCUDAPath }

var errInjectedQwen35GDN = errors.New("injected Qwen35 GDN operation failure")
var errInjectedQwen35Clone = errors.New("injected Qwen35 recurrent clone failure")

// recordingQwen35Backend is a model-dispatch witness, not a second GDN oracle. Its
// Qwen35GDNDecode implementation delegates the normalized input to the existing
// Session.linearAttnStep CPU/reference object, while the outer HAL remains the production
// implementation under test. Calls made by the operation use the embedded backend directly,
// so wrapper Read/Host counters expose only forbidden model-side fallback.
type recordingQwen35Backend struct {
	compute.Backend
	lifecycleMu     sync.Mutex
	model           *Model
	reference       *Session
	linearLayers    []int
	gdnCalls        int
	attentionCalls  int
	attentionLayers []int
	hostCalls       int
	readCalls       int
	matmulSites     []string
	classes         []compute.MemoryClass
	sites           []string
	tensorSites     map[compute.Buffer]string
	freeCalls       map[compute.Buffer]int
	stateIdentity   map[int][2]compute.Buffer
	stateTicks      map[compute.Buffer]float32
	stateContinuous bool
	cloneCalls      int
	cloneFailAt     int
	clonedBuffers   []compute.Buffer
	freeHook        func()
	badRoute        string
	failAt          int
	deviceMemory    bool
}

func newRecordingQwen35Backend(m *Model) *recordingQwen35Backend {
	b := &recordingQwen35Backend{
		Backend:         compute.Default(),
		model:           m,
		reference:       m.NewSession(),
		tensorSites:     make(map[compute.Buffer]string),
		freeCalls:       make(map[compute.Buffer]int),
		stateIdentity:   make(map[int][2]compute.Buffer),
		stateTicks:      make(map[compute.Buffer]float32),
		stateContinuous: true,
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		if m.Cfg.isLinearAttnLayer(l) {
			b.linearLayers = append(b.linearLayers, l)
		}
	}
	return b
}

func (b *recordingQwen35Backend) Name() string                    { return "recording-cuda" }
func (b *recordingQwen35Backend) Tier() string                    { return "recording" }
func (b *recordingQwen35Backend) Class() compute.CorrectnessClass { return compute.Approx }
func (*recordingQwen35Backend) Qwen35GDNPath() string             { return Qwen35GDNCUDAPath }
func (b *recordingQwen35Backend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = b.deviceMemory
	return caps
}

func (b *recordingQwen35Backend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	out := b.Backend.Upload(t, as)
	b.classes = append(b.classes, class)
	b.sites = append(b.sites, site)
	b.tensorSites[out.Buf()] = site
	return out
}

func (b *recordingQwen35Backend) Host(t compute.Tensor) ([]float32, bool) {
	b.hostCalls++
	return b.Backend.Host(t)
}

func (b *recordingQwen35Backend) Read(t compute.Tensor) []float32 {
	b.readCalls++
	return b.Backend.Read(t)
}

func (b *recordingQwen35Backend) Free(t compute.Tensor) {
	b.lifecycleMu.Lock()
	b.freeCalls[t.Buf()]++
	hook := b.freeHook
	b.freeHook = nil
	b.lifecycleMu.Unlock()
	b.Backend.Free(t)
	if hook != nil {
		hook()
	}
}

func (b *recordingQwen35Backend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	b.lifecycleMu.Lock()
	b.cloneCalls++
	call := b.cloneCalls
	failAt := b.cloneFailAt
	b.lifecycleMu.Unlock()
	if failAt > 0 && call == failAt {
		return compute.Tensor{}, errInjectedQwen35Clone
	}
	cloner, ok := b.Backend.(compute.TensorCloner)
	if !ok {
		return compute.Tensor{}, errors.New("recording backend cannot clone tensor")
	}
	out, err := cloner.CloneTensor(t)
	if err == nil {
		b.lifecycleMu.Lock()
		b.clonedBuffers = append(b.clonedBuffers, out.Buf())
		b.lifecycleMu.Unlock()
	}
	return out, err
}

func (b *recordingQwen35Backend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmulSites = append(b.matmulSites, b.tensorSites[w.Buf()])
	return b.Backend.MatMul(w, x)
}

func (b *recordingQwen35Backend) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.attentionCalls++
	b.attentionLayers = append(b.attentionLayers, layer)
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}

func (b *recordingQwen35Backend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	b.gdnCalls++
	if b.failAt > 0 && b.gdnCalls == b.failAt {
		return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, errInjectedQwen35GDN
	}
	if len(b.linearLayers) == 0 {
		return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, errors.New("recording backend has no linear layers")
	}
	layer := b.linearLayers[(b.gdnCalls-1)%len(b.linearLayers)]
	wantSite := layerName(layer, "linear_attn.in_proj_qkv.weight")
	if got := b.tensorSites[inProjQKV.Buf()]; got != "hal-weight "+wantSite {
		b.badRoute = "layer " + itoa(layer) + " received " + got
	}
	identity := [2]compute.Buffer{convState.Buf(), recurrentState.Buf()}
	if prior, ok := b.stateIdentity[layer]; ok && prior != identity {
		b.stateContinuous = false
	} else {
		b.stateIdentity[layer] = identity
	}
	// Mutate a marker in each backend-owned state buffer and verify the next call sees it.
	// These reads deliberately bypass b.Read: they are the test backend's operation body,
	// not model-side state readback.
	for _, state := range []compute.Tensor{convState, recurrentState} {
		data := b.Backend.Read(state)
		if len(data) == 0 {
			continue
		}
		if data[0] != b.stateTicks[state.Buf()] {
			b.stateContinuous = false
		}
		data[0]++
		b.stateTicks[state.Buf()] = data[0]
	}
	xn := append([]float32(nil), b.Backend.Read(normalizedInput)...)
	out := b.reference.linearAttnStep(layer, xn, residentKernel{b.model})
	return compute.NewF32(b.Backend, []int{b.model.Cfg.HiddenSize}, out), convState, recurrentState, nil
}

type pathQwen35Backend struct {
	*recordingQwen35Backend
	path string
}

func (b *pathQwen35Backend) Qwen35GDNPath() string { return b.path }

type wrongPathQwen35Backend struct{ *recordingQwen35Backend }

func (*wrongPathQwen35Backend) Qwen35GDNPath() string { return "cuda/qwen35-gdn-wrong-v0" }

func TestValidateBackendForwardConfigAcceptsExplicitVulkanContract(t *testing.T) {
	cfg := Config{ModelType: "qwen3_5", LayerTypes: []string{"linear_attention"}}
	be := &pathQwen35Backend{recordingQwen35Backend: &recordingQwen35Backend{}, path: Qwen35GDNVulkanPath}
	if err := ValidateBackendForwardConfig(cfg, be); err != nil {
		t.Fatalf("explicit Vulkan structural backend refused: %v", err)
	}
}

func TestValidateBackendForwardConfigRefusesUnknownStructuralPath(t *testing.T) {
	cfg := Config{ModelType: "qwen3_5", LayerTypes: []string{"linear_attention"}}
	be := &pathQwen35Backend{recordingQwen35Backend: &recordingQwen35Backend{}, path: "vulkan/qwen35-gdn-unknown-v0"}
	err := ValidateBackendForwardConfig(cfg, be)
	var unsupported *UnsupportedBackendForwardError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error=%T %v, want typed refusal", err, err)
	}
	for _, want := range []string{Qwen35GDNCUDAPath, Qwen35GDNVulkanPath, be.path} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal missing %q: %v", want, err)
		}
	}
}
func TestValidateBackendForwardConfigQwen35ExactPathAdmission(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	missing := &refusingQwen35CUDA{Backend: compute.Default()}
	marker := &markerOnlyQwen35CUDA{refusingQwen35CUDA: &refusingQwen35CUDA{Backend: compute.Default()}}
	exact := newRecordingQwen35Backend(m)
	wrong := &wrongPathQwen35Backend{recordingQwen35Backend: newRecordingQwen35Backend(m)}

	for name, be := range map[string]compute.Backend{"missing": missing, "marker-only": marker, "wrong-path": wrong} {
		t.Run(name, func(t *testing.T) {
			err := ValidateBackendForwardConfig(cfg, be)
			var unsupported *UnsupportedBackendForwardError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error=%T %v, want *UnsupportedBackendForwardError", err, err)
			}
			if unsupported.Forward != ForwardQwen35GDN || unsupported.IntendedPath != Qwen35GDNCUDAPath+" or "+Qwen35GDNVulkanPath {
				t.Fatalf("wrong refusal identity: %#v", unsupported)
			}
			for _, text := range []string{Qwen35GDNCUDAPath, Qwen35GDNVulkanPath, "generic QKV/CPU fallback", "0.999"} {
				if !strings.Contains(err.Error(), text) {
					t.Errorf("refusal missing %q: %v", text, err)
				}
			}
		})
	}
	if err := ValidateBackendForwardConfig(cfg, exact); err != nil {
		t.Fatalf("exact structural backend refused: %v", err)
	}
	if err := ValidateBackendForwardConfig(cfg, nil); err != nil {
		t.Fatalf("legacy CPU/reference selection refused: %v", err)
	}
	plain := cfg
	plain.LayerTypes = nil
	if err := ValidateBackendForwardConfig(plain, missing); err != nil {
		t.Fatalf("non-hybrid backend admission changed: %v", err)
	}
	if got := m.NewSession().Prefill([]int{3, 7}); len(got) != cfg.VocabSize {
		t.Fatalf("legacy CPU/reference path logits=%d, want %d", len(got), cfg.VocabSize)
	}
}

func TestQwen35BackendConstructorRefusesBeforeAnyFallbackOrStateUpload(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	for name, be := range map[string]compute.Backend{
		"missing":     &refusingQwen35CUDA{Backend: compute.Default()},
		"marker-only": &markerOnlyQwen35CUDA{refusingQwen35CUDA: &refusingQwen35CUDA{Backend: compute.Default()}},
		"wrong-path":  &wrongPathQwen35Backend{recordingQwen35Backend: newRecordingQwen35Backend(m)},
	} {
		t.Run(name, func(t *testing.T) {
			s, err := m.NewBackendSessionChecked(be)
			if s != nil {
				t.Fatalf("refused backend returned session %#v", s)
			}
			var unsupported *UnsupportedBackendForwardError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error=%T %v, want typed refusal", err, err)
			}
			switch got := be.(type) {
			case *refusingQwen35CUDA:
				if got.matmuls != 0 || got.attention != 0 {
					t.Fatalf("refusal executed fallback: matmul=%d attention=%d", got.matmuls, got.attention)
				}
			case *markerOnlyQwen35CUDA:
				if got.matmuls != 0 || got.attention != 0 {
					t.Fatalf("refusal executed fallback: matmul=%d attention=%d", got.matmuls, got.attention)
				}
			case *wrongPathQwen35Backend:
				if len(got.classes) != 0 || got.gdnCalls != 0 {
					t.Fatalf("wrong path allocated state or ran operation: classes=%v calls=%d", got.classes, got.gdnCalls)
				}
			}
		})
	}
}

func TestQwen35HybridHALFullLogitParityDispatchAndStateLifecycle(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = 0.25
	cfg.RopeTheta = 10_000_000
	m := NewSynthetic(cfg)
	be := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}

	var stateBuffers []compute.Buffer
	for l, state := range s.qwen35HAL.layers {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		stateBuffers = append(stateBuffers, state.conv.Buf(), state.recurrent.Buf())
		for _, site := range []string{"qwen35-gdn-conv-state layer " + itoa(l), "qwen35-gdn-recurrent-state layer " + itoa(l)} {
			if !recordedClassSite(be, compute.MemoryKVCache, site) {
				t.Fatalf("missing persistent KV-cache allocation %q; classes=%v sites=%v", site, be.classes, be.sites)
			}
		}
	}

	prompt := []int{3, 7, 11, 5, 17, 19, 23}
	want := m.NewSession().Prefill(prompt)
	s.Prefill(prompt[:4])
	var got []float32
	for _, id := range prompt[4:] {
		got = s.Step(id)
	}
	cosine := cosineF32(t, want, got)
	if cosine < Qwen35GDNParityCosineMin {
		t.Fatalf("full-logit cosine %.9f < %.3f", cosine, Qwen35GDNParityCosineMin)
	}
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("greedy argmax=%d, want %d (cosine %.9f)", argmaxF32(got), argmaxF32(want), cosine)
	}

	tokens, linearLayers, fullLayers := len(prompt), 0, 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			linearLayers++
		} else {
			fullLayers++
		}
	}
	if wantCalls := tokens * linearLayers; be.gdnCalls != wantCalls {
		t.Fatalf("GDN operations=%d, want every token x linear layer = %d", be.gdnCalls, wantCalls)
	}
	if wantCalls := tokens * fullLayers; be.attentionCalls != wantCalls {
		t.Fatalf("generic Attention calls=%d, want full layers only=%d", be.attentionCalls, wantCalls)
	}
	for _, layer := range be.attentionLayers {
		if layer != 0 { // hybrid full layers are compacted into the backend KV plane.
			t.Fatalf("full-attention KV layer=%d, want compact backend layer 0", layer)
		}
	}
	if !be.stateContinuous || be.badRoute != "" {
		t.Fatalf("persistent state/operation routing failed: continuous=%v bad_route=%q", be.stateContinuous, be.badRoute)
	}
	if s.halKV.Len() != tokens {
		t.Fatalf("hybrid backend KV length=%d, want %d", s.halKV.Len(), tokens)
	}
	for _, site := range be.matmulSites {
		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) && strings.Contains(site, layerName(l, "self_attn.q_proj")) {
				t.Fatalf("linear layer %d fell through q_proj via %q", l, site)
			}
		}
	}
	if be.hostCalls != 0 {
		t.Fatalf("model called Backend.Host %d times", be.hostCalls)
	}
	// Per full layer/token: q+k partial-RoPE reads plus gate+attention reads. Only the
	// split Prefill result and three Step results add final-logit reads. Any state or
	// linear fallback read makes this exact count fail.
	wantReads := tokens*fullLayers*4 + 1 + len(prompt[4:])
	if be.readCalls != wantReads {
		t.Fatalf("backend Read calls=%d, want %d (full-attention bridge + returned logits only)", be.readCalls, wantReads)
	}

	s.Close()
	if s.qwen35HAL != nil || !s.halClosed {
		t.Fatalf("Close did not clear backend GDN state: state=%#v closed=%v", s.qwen35HAL, s.halClosed)
	}
	for _, buffer := range stateBuffers {
		if be.freeCalls[buffer] != 1 {
			t.Fatalf("persistent state %p freed %d times, want once", buffer, be.freeCalls[buffer])
		}
	}
	freeBefore := totalFreeCalls(be)
	s.Close()
	if totalFreeCalls(be) != freeBefore {
		t.Fatal("Session.Close is not idempotent")
	}
}

func TestQwen35GDNOperationErrorClosesSessionWithoutRetry(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	be.failAt = 2
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	var stateBuffers []compute.Buffer
	for l, state := range s.qwen35HAL.layers {
		if m.Cfg.isLinearAttnLayer(l) {
			stateBuffers = append(stateBuffers, state.conv.Buf(), state.recurrent.Buf())
		}
	}

	panicErr := recoverError(func() { _ = s.Prefill([]int{3}) })
	var operation *BackendForwardOperationError
	if !errors.As(panicErr, &operation) || !errors.Is(panicErr, errInjectedQwen35GDN) {
		t.Fatalf("operation panic=%T %v, want wrapped injected error", panicErr, panicErr)
	}
	if !strings.Contains(operation.Error(), "session closed, no CPU retry") || operation.Layer != 1 {
		t.Fatalf("wrong fail-closed verdict: %#v (%v)", operation, operation)
	}
	if !s.halClosed || s.qwen35HAL != nil || be.gdnCalls != 2 || be.attentionCalls != 0 {
		t.Fatalf("failure lifecycle: closed=%v state=%#v gdn=%d attention=%d", s.halClosed, s.qwen35HAL, be.gdnCalls, be.attentionCalls)
	}
	for _, buffer := range stateBuffers {
		if be.freeCalls[buffer] != 1 {
			t.Fatalf("failed session state %p freed %d times, want once", buffer, be.freeCalls[buffer])
		}
	}
	for _, site := range be.matmulSites {
		if strings.Contains(site, "self_attn.q_proj") {
			t.Fatalf("failed linear path retried through generic q_proj: %q", site)
		}
	}
	calls := be.gdnCalls
	reuseErr := recoverError(func() { _ = s.Step(7) })
	if !errors.Is(reuseErr, errInjectedQwen35GDN) || be.gdnCalls != calls {
		t.Fatalf("failed session reuse=%v calls=%d, want same failure and no retry", reuseErr, be.gdnCalls)
	}
}

func recordedClassSite(be *recordingQwen35Backend, class compute.MemoryClass, site string) bool {
	for i := range be.classes {
		if be.classes[i] == class && be.sites[i] == site {
			return true
		}
	}
	return false
}

func cosineF32(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("logit lengths=%d/%d", len(a), len(b))
	}
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			t.Fatalf("non-finite logits at %d: %v/%v", i, a[i], b[i])
		}
		dot += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		t.Fatalf("zero-norm logits: %g/%g", aa, bb)
	}
	return dot / math.Sqrt(aa*bb)
}

func recoverError(fn func()) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = asError(recovered)
		}
	}()
	fn()
	return nil
}

func asError(v any) error {
	if err, ok := v.(error); ok {
		return err
	}
	return nil
}

func totalFreeCalls(be *recordingQwen35Backend) int {
	total := 0
	for _, calls := range be.freeCalls {
		total += calls
	}
	return total
}

func TestQwen35PrefixSnapshotClonesAndRestoresAllHybridDeviceState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Prefill([]int{3, 7, 11})
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	clone, err := snap.Clone()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	if err := clone.Restore(restored); err != nil {
		t.Fatal(err)
	}
	clone.Close()
	defer restored.Close()
	if restored.Cache.Len() != s.Cache.Len() || restored.halKV.Len() != s.halKV.Len() {
		t.Fatalf("restored positions host=%d/%d device=%d/%d", restored.Cache.Len(), s.Cache.Len(), restored.halKV.Len(), s.halKV.Len())
	}
	if restored.qwen35HAL == nil || len(restored.qwen35HAL.layers) != len(s.qwen35HAL.layers) {
		t.Fatal("recurrent state omitted")
	}
	if be.cloneCalls != 0 {
		t.Fatalf("snapshot clone eagerly copied recurrent tensors: clone calls=%d, want 0", be.cloneCalls)
	}
}

func TestQwen35RecurrentSnapshotCopyOnWriteIsolation(t *testing.T) {
	type pairImage struct {
		conv, recurrent         compute.Buffer
		convData, recurrentData []float32
	}

	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	root, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	root.Prefill([]int{3, 7, 11})
	linear := len(be.linearLayers)
	if linear == 0 {
		t.Fatal("test model has no recurrent layers")
	}
	image := func(s *Session, layer int) pairImage {
		t.Helper()
		s.qwen35HAL.mu.Lock()
		defer s.qwen35HAL.mu.Unlock()
		state := &s.qwen35HAL.layers[layer]
		return pairImage{
			conv: state.conv.Buf(), recurrent: state.recurrent.Buf(),
			convData:      append([]float32(nil), be.Backend.Read(state.conv)...),
			recurrentData: append([]float32(nil), be.Backend.Read(state.recurrent)...),
		}
	}
	assertBitsEqual := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s elements=%d, want %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s changed at %d: got %08x want %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	allStateBuffers := make(map[compute.Buffer]struct{})
	captureBuffers := func(s *Session) {
		t.Helper()
		for _, layer := range be.linearLayers {
			got := image(s, layer)
			allStateBuffers[got.conv] = struct{}{}
			allStateBuffers[got.recurrent] = struct{}{}
		}
	}

	baseline := make(map[int]pairImage, linear)
	for _, layer := range be.linearLayers {
		baseline[layer] = image(root, layer)
	}
	captureBuffers(root)
	root.qwen35HAL.sequenceAccepted = true
	root.qwen35HAL.sequenceFailure = errors.New("session-local sentinel")
	root.qwen35HAL.prefillRoute.RequestedPath = "session-local"
	root.qwen35HAL.qsaGatheredK = []float32{1}

	snap, err := root.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	sibling, err := snap.Clone()
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	if be.cloneCalls != 0 {
		t.Fatalf("read-only sibling snapshots made %d eager tensor copies, want 0", be.cloneCalls)
	}
	for name, q := range map[string]*qwen35HALState{"snapshot": snap.qwen35, "sibling": sibling.qwen35} {
		if q.sequenceAccepted || q.sequenceFailure != nil || q.prefillRoute.RequestedPath != "" || q.qsaGatheredK != nil {
			t.Fatalf("%s shared session-local sequence/QSA/receipt state: %#v", name, q)
		}
	}

	branchA, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	captureBuffers(branchA)
	branchB, err := m.NewBackendSessionChecked(be)
	if err != nil {
		branchA.Close()
		t.Fatal(err)
	}
	captureBuffers(branchB)
	if err := snap.Restore(branchA); err != nil {
		t.Fatal(err)
	}
	if err := sibling.Restore(branchB); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	snap.Close()
	sibling.Close()
	sibling.Close()

	beforeFirstWrite := be.cloneCalls
	branchA.Step(13)
	if got, want := be.cloneCalls-beforeFirstWrite, 2*linear; got != want {
		t.Fatalf("branch A first write tensor copies=%d, want one pair per layer (%d)", got, want)
	}
	captureBuffers(branchA)
	for _, layer := range be.linearLayers {
		if got, original := image(branchA, layer), baseline[layer]; got.conv == original.conv || got.recurrent == original.recurrent {
			t.Fatalf("branch A layer %d retained shared state on first write", layer)
		}
	}
	branchA.Step(17)
	if be.cloneCalls != beforeFirstWrite+2*linear {
		t.Fatalf("branch A second write cloned again: clone calls=%d want %d", be.cloneCalls, beforeFirstWrite+2*linear)
	}
	branchABeforeB := make(map[int]pairImage, linear)
	for _, layer := range be.linearLayers {
		branchABeforeB[layer] = image(branchA, layer)
		got, original := image(branchB, layer), baseline[layer]
		if got.conv != original.conv || got.recurrent != original.recurrent {
			t.Fatalf("untouched branch B layer %d changed identity before first write", layer)
		}
		assertBitsEqual("untouched branch B convolution layer "+itoa(layer), got.convData, original.convData)
		assertBitsEqual("untouched branch B recurrent layer "+itoa(layer), got.recurrentData, original.recurrentData)
	}

	beforeBranchB := be.cloneCalls
	branchB.Step(19)
	if got, want := be.cloneCalls-beforeBranchB, 2*linear; got != want {
		t.Fatalf("branch B first write tensor copies=%d, want %d", got, want)
	}
	captureBuffers(branchB)
	for _, layer := range be.linearLayers {
		gotA, beforeB := image(branchA, layer), branchABeforeB[layer]
		if gotA.conv != beforeB.conv || gotA.recurrent != beforeB.recurrent {
			t.Fatalf("branch B write changed branch A layer %d identity", layer)
		}
		assertBitsEqual("branch A convolution after branch B write layer "+itoa(layer), gotA.convData, beforeB.convData)
		assertBitsEqual("branch A recurrent after branch B write layer "+itoa(layer), gotA.recurrentData, beforeB.recurrentData)
		gotB := image(branchB, layer)
		if gotA.conv == gotB.conv || gotA.recurrent == gotB.recurrent || (reflect.DeepEqual(gotA.convData, gotB.convData) && reflect.DeepEqual(gotA.recurrentData, gotB.recurrentData)) {
			t.Fatalf("branch A/B layer %d did not isolate identity and bytes", layer)
		}
		got, original := image(root, layer), baseline[layer]
		if got.conv != original.conv || got.recurrent != original.recurrent {
			t.Fatalf("untouched root layer %d changed identity", layer)
		}
		assertBitsEqual("untouched convolution layer "+itoa(layer), got.convData, original.convData)
		assertBitsEqual("untouched recurrent layer "+itoa(layer), got.recurrentData, original.recurrentData)
	}

	// Two siblings may reach first write together. Per-owner locking serializes the
	// refcount transition while each branch receives its own physical pair.
	concurrentSnap, err := root.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	concurrentSibling, err := concurrentSnap.Clone()
	if err != nil {
		t.Fatal(err)
	}
	concurrentA, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	captureBuffers(concurrentA)
	concurrentB, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	captureBuffers(concurrentB)
	if err := concurrentSnap.Restore(concurrentA); err != nil {
		t.Fatal(err)
	}
	if err := concurrentSibling.Restore(concurrentB); err != nil {
		t.Fatal(err)
	}
	concurrentSnap.Close()
	concurrentSibling.Close()
	firstLayer := be.linearLayers[0]
	start := make(chan struct{})
	errs := make(chan error, 2)
	mutatePair := func(conv, recurrent compute.Tensor, delta float32) {
		convData := be.Backend.Read(conv)
		recurrentData := be.Backend.Read(recurrent)
		convData[0] += delta
		recurrentData[0] += delta
	}
	mutateScalar := func(s *Session, delta float32) {
		<-start
		_, _, _, mutateErr := s.qwen35HAL.mutateLayer(be, firstLayer, func(conv, recurrent compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
			mutatePair(conv, recurrent, delta)
			return compute.Tensor{}, conv, recurrent, nil
		})
		errs <- mutateErr
	}
	mutateWholeSequence := func(s *Session, delta float32) {
		<-start
		_, mutateErr := s.qwen35HAL.mutateSequence(be, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
			state := states[firstLayer]
			mutatePair(state.Conv, state.Recurrent, delta)
			return compute.Qwen35SequencePrefillResult{}, nil
		})
		errs <- mutateErr
	}
	beforeConcurrent := be.cloneCalls
	go mutateWholeSequence(concurrentA, 2)
	go mutateScalar(concurrentB, 3)
	close(start)
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if got, want := be.cloneCalls-beforeConcurrent, 2*linear+2; got != want {
		t.Fatalf("concurrent sequence/scalar first writes copied %d tensors, want %d", got, want)
	}
	captureBuffers(concurrentA)
	captureBuffers(concurrentB)
	aImage, bImage := image(concurrentA, firstLayer), image(concurrentB, firstLayer)
	if aImage.conv == bImage.conv || aImage.recurrent == bImage.recurrent {
		t.Fatal("concurrent branches share a post-write state handle")
	}

	stale, err := root.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	staleTarget, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	captureBuffers(staleTarget)
	root.cacheGeometryMu.Lock()
	root.cacheGeometryEpoch++
	root.cacheGeometryMu.Unlock()
	if err := stale.Restore(staleTarget); err == nil || !strings.Contains(err.Error(), "stale prefix snapshot") {
		t.Fatalf("stale restore error=%v, want ownership-preserving refusal", err)
	}
	if stale.qwen35 == nil || stale.Cache == nil {
		t.Fatal("stale restore transferred ownership before refusing")
	}

	// The shared owner remembers its allocating backend; final release does not
	// depend on the snapshot/session wrapper still carrying that pointer.
	branchA.qwen35HAL.free(nil)
	branchA.qwen35HAL = nil
	branchA.Close()
	branchA.Close()
	branchB.Close()
	branchB.Close()
	concurrentA.Close()
	concurrentB.Close()
	staleTarget.Close()
	stale.Close()
	stale.Close()
	root.Close()
	root.Close()
	for buffer := range allStateBuffers {
		if got := be.freeCalls[buffer]; got != 1 {
			t.Fatalf("recurrent state %p freed %d times, want exactly once", buffer, got)
		}
	}

	// A transactional pair clone may need to discard the convolution clone when
	// cloning recurrent state fails. That compensating Free must run only after
	// the handle and owner locks are released because backend cleanup may re-enter
	// Session.closeQwen35HALState.
	failureModel := NewSynthetic(qwen35HybridTestCfg())
	failureBackend := newRecordingQwen35Backend(failureModel)
	failureRoot, err := failureModel.NewBackendSessionChecked(failureBackend)
	if err != nil {
		t.Fatal(err)
	}
	failureRoot.Prefill([]int{5})
	failureBuffers := make(map[compute.Buffer]struct{})
	rememberFailureState := func(s *Session) {
		t.Helper()
		for _, layer := range failureBackend.linearLayers {
			state := s.qwen35HAL.layers[layer]
			failureBuffers[state.conv.Buf()] = struct{}{}
			failureBuffers[state.recurrent.Buf()] = struct{}{}
		}
	}
	rememberFailureState(failureRoot)
	failureSnapshot, err := failureRoot.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	failureBranch, err := failureModel.NewBackendSessionChecked(failureBackend)
	if err != nil {
		t.Fatal(err)
	}
	rememberFailureState(failureBranch)
	if err := failureSnapshot.Restore(failureBranch); err != nil {
		t.Fatal(err)
	}
	failureSnapshot.Close()
	failureBackend.cloneFailAt = failureBackend.cloneCalls + 2
	reentered := make(chan struct{})
	failureBackend.freeHook = func() {
		failureBranch.closeQwen35HALState()
		close(reentered)
	}
	mutationDone := make(chan error, 1)
	failureState := failureBranch.qwen35HAL
	go func() {
		_, _, _, mutationErr := failureState.mutateLayer(failureBackend, failureBackend.linearLayers[0], func(conv, recurrent compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
			return compute.Tensor{}, conv, recurrent, nil
		})
		mutationDone <- mutationErr
	}()
	select {
	case mutationErr := <-mutationDone:
		if !errors.Is(mutationErr, errInjectedQwen35Clone) {
			t.Fatalf("recurrent clone failure=%v, want injected error", mutationErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("recurrent clone cleanup deadlocked during re-entrant close")
	}
	select {
	case <-reentered:
	default:
		t.Fatal("partial-clone cleanup did not invoke re-entrant Free callback")
	}
	if failureBranch.qwen35HAL != nil {
		t.Fatal("re-entrant close retained failed branch recurrent ownership")
	}
	if len(failureBackend.clonedBuffers) != 1 {
		t.Fatalf("successful partial clones=%d, want convolution only", len(failureBackend.clonedBuffers))
	}
	partialClone := failureBackend.clonedBuffers[0]
	failureBuffers[partialClone] = struct{}{}
	if got := failureBackend.freeCalls[partialClone]; got != 1 {
		t.Fatalf("partial convolution clone freed %d times, want once", got)
	}
	failureBranch.Close()
	failureRoot.Close()
	for buffer := range failureBuffers {
		if got := failureBackend.freeCalls[buffer]; got != 1 {
			t.Fatalf("failure-path state %p freed %d times, want exactly once", buffer, got)
		}
	}
}

func TestQwen35PrefixSnapshotHostRoundTripOwnsCompleteHybridState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	be.deviceMemory = true
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Prefill([]int{3, 7, 11})
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	hostBytes, deviceBytes := snap.ResidencyBytes()
	if hostBytes <= 0 || deviceBytes <= 0 {
		snap.Close()
		t.Fatalf("physical residency host=%d device=%d, want split metadata/payload ownership", hostBytes, deviceBytes)
	}
	host, err := snap.CloneToHost()
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	defer host.Close()
	if host.ResidentBytes() <= 0 || host.TransferBytes() <= 0 {
		snap.Close()
		t.Fatalf("host image resident=%d transfer=%d, want owned payload", host.ResidentBytes(), host.TransferBytes())
	}
	freeBefore := totalFreeCalls(be)
	snap.Close()
	if got := totalFreeCalls(be); got != freeBefore {
		t.Fatalf("closing a shared read-only snapshot freed live recurrent tensors: frees=%d, want %d", got, freeBefore)
	}

	restored, err := host.Restore()
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	roundTrip, err := restored.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer roundTrip.Close()
	if !reflect.DeepEqual(roundTrip.cache, host.cache) {
		t.Fatal("host model cache drifted across host→backend→host round trip")
	}
	if !reflect.DeepEqual(roundTrip.kv, host.kv) {
		t.Fatal("attention K/Kraw/V or positions drifted across host round trip")
	}
	if !reflect.DeepEqual(roundTrip.qwen35.layers, host.qwen35.layers) {
		t.Fatal("Qwen convolution/recurrent state drifted across host round trip")
	}
}

type mockThirdPartyGDNBackend struct {
	compute.Backend
	name string
	path string
}

func (b *mockThirdPartyGDNBackend) Name() string {
	if b.name != "" {
		return b.name
	}
	return "synthetic-npu"
}

func (b *mockThirdPartyGDNBackend) Qwen35GDNPath() string {
	if b.path != "" {
		return b.path
	}
	return Qwen35GDNCapabilityIdentity
}

func (b *mockThirdPartyGDNBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	return compute.Tensor{}, convState, recurrentState, nil
}

func TestQwen35GDNCapabilityIdentity(t *testing.T) {
	for _, p := range []string{Qwen35GDNCapabilityIdentity, Qwen35GDNCUDAPath, Qwen35GDNVulkanPath} {
		if !IsSupportedQwen35GDNPath(p) {
			t.Errorf("IsSupportedQwen35GDNPath(%q) = false, want true", p)
		}
	}
	for _, p := range []string{"legacy/unsupported", "unknown", ""} {
		if IsSupportedQwen35GDNPath(p) {
			t.Errorf("IsSupportedQwen35GDNPath(%q) = true, want false", p)
		}
	}

	cfg := qwen35HybridTestCfg()
	npuBackend := &mockThirdPartyGDNBackend{
		Backend: compute.Default(),
		name:    "synthetic-npu",
		path:    Qwen35GDNCapabilityIdentity,
	}

	// Prove ValidateBackendForwardConfig admits it for a Qwen35 hybrid config.
	if err := ValidateBackendForwardConfig(cfg, npuBackend); err != nil {
		t.Fatalf("ValidateBackendForwardConfig refused synthetic GDN backend: %v", err)
	}

	// Prove a backend with an unsupported path (e.g. "legacy/unsupported") is refused with UnsupportedBackendForwardError.
	unsupportedBackend := &mockThirdPartyGDNBackend{
		Backend: compute.Default(),
		name:    "synthetic-npu",
		path:    "legacy/unsupported",
	}
	err := ValidateBackendForwardConfig(cfg, unsupportedBackend)
	var unsupported *UnsupportedBackendForwardError
	if !errors.As(err, &unsupported) {
		t.Fatalf("ValidateBackendForwardConfig error=%T %v, want *UnsupportedBackendForwardError", err, err)
	}

	// Prove prefix snapshot decode accepts the synthetic backend.
	m := NewSynthetic(cfg)
	recBe := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(recBe)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()
	s.Prefill([]int{3, 7, 11})
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	defer snap.Close()
	host, err := snap.CloneToHost()
	if err != nil {
		t.Fatalf("CloneToHost: %v", err)
	}
	defer host.Close()
	wire, err := host.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	decoded, err := DecodeHostPrefixSnapshot(wire, npuBackend, cfg)
	if err != nil {
		t.Fatalf("DecodeHostPrefixSnapshot refused synthetic GDN backend: %v", err)
	}
	defer decoded.Close()

	if decoded.qwen35 == nil || decoded.qwen35.backend != npuBackend {
		t.Fatalf("decoded snapshot did not bind synthetic backend: %+v", decoded.qwen35)
	}

	// Prove prefix snapshot decode refuses unsupported backend.
	if _, err := DecodeHostPrefixSnapshot(wire, unsupportedBackend, cfg); err == nil {
		t.Fatal("DecodeHostPrefixSnapshot accepted backend with unsupported GDN path")
	}
}
