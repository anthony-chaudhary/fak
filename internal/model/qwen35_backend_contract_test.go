package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

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

// recordingQwen35Backend is a model-dispatch witness, not a second GDN oracle. Its
// Qwen35GDNDecode implementation delegates the normalized input to the existing
// Session.linearAttnStep CPU/reference object, while the outer HAL remains the production
// implementation under test. Calls made by the operation use the embedded backend directly,
// so wrapper Read/Host counters expose only forbidden model-side fallback.
type recordingQwen35Backend struct {
	compute.Backend
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
	b.freeCalls[t.Buf()]++
	b.Backend.Free(t)
}

func (b *recordingQwen35Backend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	b.cloneCalls++
	cloner, ok := b.Backend.(compute.TensorCloner)
	if !ok {
		return compute.Tensor{}, errors.New("recording backend cannot clone tensor")
	}
	return cloner.CloneTensor(t)
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

type wrongPathQwen35Backend struct{ *recordingQwen35Backend }

func (*wrongPathQwen35Backend) Qwen35GDNPath() string { return "cuda/qwen35-gdn-wrong-v0" }

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
			if unsupported.Forward != ForwardQwen35GDN || unsupported.IntendedPath != Qwen35GDNCUDAPath {
				t.Fatalf("wrong refusal identity: %#v", unsupported)
			}
			for _, text := range []string{Qwen35GDNCUDAPath, "generic QKV/CPU fallback", "0.999"} {
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
	linear := len(be.linearLayers)
	if be.cloneCalls < 4*linear {
		t.Fatalf("clone calls=%d want at least %d", be.cloneCalls, 4*linear)
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
	if totalFreeCalls(be) <= freeBefore {
		t.Fatal("closing the hot snapshot did not release its backend-owned tensors")
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
