package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RouterGateRecorder is a CPU-backed device that offers the separate
// projection + router primitives and the fused router gate, and counts which
// ran. It records ownership, not device performance.
type v41RouterGateRecorder struct {
	compute.Backend
	mu                                             sync.Mutex
	live                                           map[compute.Buffer]string
	allocated                                      map[compute.Buffer]bool
	frees                                          map[compute.Buffer]int
	matmuls, routerCalls, gateCalls, gateAvailable int
	biasUploads                                    int
	gateUnavailable                                bool
	gateFault                                      error
	gateBias                                       compute.Buffer
	afterGate                                      func()
}

func newV41RouterGateRecorder() *v41RouterGateRecorder {
	return &v41RouterGateRecorder{Backend: compute.Default(), live: map[compute.Buffer]string{}, allocated: map[compute.Buffer]bool{}, frees: map[compute.Buffer]int{}}
}

func (b *v41RouterGateRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41RouterGateRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}
func (b *v41RouterGateRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}
func (b *v41RouterGateRecorder) V41RouterTopKAvailable() bool { return true }
func (b *v41RouterGateRecorder) V41RouterGateTopKAvailable() bool {
	b.gateAvailable++
	return !b.gateUnavailable
}

func (b *v41RouterGateRecorder) own(shape []int, values []float32, site string) compute.Tensor {
	out := compute.NewF32(b, append([]int(nil), shape...), values)
	b.live[out.Buf()] = site
	b.allocated[out.Buf()] = true
	return out
}

func (b *v41RouterGateRecorder) Upload(t compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(t, dt, compute.MemoryWeights, "unclassified")
}

func (b *v41RouterGateRecorder) UploadClass(t compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("fixture only supports F32")
	}
	if site == "V4.1 router correction bias" {
		b.biasUploads++
	}
	return b.own(t.Shape, append([]float32(nil), b.Backend.Read(t)...), site)
}

func (b *v41RouterGateRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	tmp := b.Backend.MatMul(w, x)
	defer b.Backend.Free(tmp)
	return b.own(tmp.Shape, append([]float32(nil), b.Backend.Read(tmp)...), "projection output")
}

func (b *v41RouterGateRecorder) Read(t compute.Tensor) []float32 {
	if !b.allocated[t.Buf()] {
		return b.Backend.Read(t)
	}
	if _, ok := b.live[t.Buf()]; !ok {
		panic("read after free")
	}
	return b.Backend.Read(t)
}

func (b *v41RouterGateRecorder) Free(t compute.Tensor) {
	if !b.allocated[t.Buf()] {
		b.Backend.Free(t)
		return
	}
	if _, ok := b.live[t.Buf()]; !ok {
		panic("double or foreign free")
	}
	delete(b.live, t.Buf())
	b.frees[t.Buf()]++
}

func (b *v41RouterGateRecorder) route(logits []float32, bias compute.Tensor, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	var correction []float32
	if hasBias {
		correction = b.Read(bias)
	}
	picks, err := v41RouteForGeometry(logits, correction, v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: scale}, full)
	if err != nil {
		return compute.V41RouterPicks{}, err
	}
	var out compute.V41RouterPicks
	for _, p := range picks {
		out.Experts = append(out.Experts, int32(p.expert))
		out.Weights = append(out.Weights, p.weight)
	}
	return out, nil
}

func (b *v41RouterGateRecorder) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.routerCalls++
	if _, ok := b.live[logits.Buf()]; !ok || !reflect.DeepEqual(logits.Shape, []int{1, experts}) {
		return compute.V41RouterPicks{}, errors.New("router input is not live/exact")
	}
	return b.route(b.Read(logits), bias, experts, topK, scale, full, hasBias)
}

func (b *v41RouterGateRecorder) V41RouterGateTopK(weight, x, bias compute.Tensor, experts, hidden, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.gateCalls++
	b.gateBias = bias.Buf()
	for _, operand := range []compute.Tensor{weight, x, bias} {
		if _, ok := b.live[operand.Buf()]; !ok {
			return compute.V41RouterPicks{}, errors.New("router gate operand is not live")
		}
	}
	if !reflect.DeepEqual(weight.Shape, []int{experts, hidden}) || !reflect.DeepEqual(x.Shape, []int{hidden}) {
		return compute.V41RouterPicks{}, errors.New("router gate operand shape")
	}
	if b.afterGate != nil {
		b.afterGate()
	}
	if b.gateFault != nil {
		return compute.V41RouterPicks{}, b.gateFault
	}
	tmp := b.Backend.MatMul(weight, x)
	defer b.Backend.Free(tmp)
	return b.route(b.Backend.Read(tmp), bias, experts, topK, scale, full, hasBias)
}

func v41RouterGateFixture(t *testing.T) (*Model, *Session, *v41RouterGateRecorder, *v41ProjScratch, []float32, func() []float32, []routePick) {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	b := newV41RouterGateRecorder()
	t.Cleanup(func() {
		if len(b.live) != 0 {
			t.Errorf("model/session cleanup retained %d buffers", len(b.live))
		}
		for buffer := range b.allocated {
			if b.frees[buffer] != 1 {
				t.Errorf("tracked allocation freed %d times, want 1", b.frees[buffer])
			}
		}
	})
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	})
	s := v41DenseTestSession(t, m, b)
	st := s.v41State()
	scope := s.beginBackendExecution()
	t.Cleanup(scope.finish)
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%7-3) / 16
	}
	bias := func() []float32 { return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")) }
	cfg := v41DefaultRouterConfig()
	logits, err := m.v41ProjMatRows(0, "ffn.gate.weight", x, cfg.Experts, m.Cfg.HiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	want, err := v41RouteForGeometry(logits, bias(), cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	return m, s, b, &v41ProjScratch{projectionRoute: st.projectionRoute}, x, bias, want
}

// fak-test:runtime medium est=2s lane=default
func TestV41RouterGateFusedStep(t *testing.T) {
	t.Run("one-fused-call-replaces-projection-and-router", func(t *testing.T) {
		m, _, b, scratch, x, bias, want := v41RouterGateFixture(t)
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		if err != nil || !reflect.DeepEqual(got, want) || b.gateCalls != 1 || b.matmuls != 0 || b.routerCalls != 0 {
			t.Fatalf("fused result=%v err=%v gate=%d matmuls=%d router=%d", got, err, b.gateCalls, b.matmuls, b.routerCalls)
		}
		if b.biasUploads != 0 || b.live[b.gateBias] == "" {
			t.Fatalf("canonical bias must stay the borrowed checkpoint tensor: uploads=%d", b.biasUploads)
		}
	})
	t.Run("caller-bias-is-transient", func(t *testing.T) {
		m, _, b, scratch, x, bias, want := v41RouterGateFixture(t)
		copied := append([]float32(nil), bias()...)
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, func() []float32 { return copied })
		if err != nil || !reflect.DeepEqual(got, want) || b.biasUploads != 1 || b.frees[b.gateBias] != 1 {
			t.Fatalf("copied bias result=%v err=%v uploads=%d frees=%d", got, err, b.biasUploads, b.frees[b.gateBias])
		}
	})
	for name, setup := range map[string]func(*testing.T, *v41RouterGateRecorder){
		"env-zero-keeps-two-ops":    func(t *testing.T, _ *v41RouterGateRecorder) { t.Setenv(v41RouterGateFusedEnv, "0") },
		"unavailable-keeps-two-ops": func(_ *testing.T, b *v41RouterGateRecorder) { b.gateUnavailable = true },
	} {
		t.Run(name, func(t *testing.T) {
			m, _, b, scratch, x, bias, want := v41RouterGateFixture(t)
			setup(t, b)
			got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
			if err != nil || !reflect.DeepEqual(got, want) || b.gateCalls != 0 || b.matmuls != 1 || b.routerCalls != 1 {
				t.Fatalf("separate result=%v err=%v gate=%d matmuls=%d router=%d", got, err, b.gateCalls, b.matmuls, b.routerCalls)
			}
		})
	}
	t.Run("numeric-fault-closes-without-replay", func(t *testing.T) {
		m, s, b, scratch, x, bias, _ := v41RouterGateFixture(t)
		b.gateFault = &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageLogit, Expert: 3}
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		var route *v4RouteError
		var stage *V41ForwardError
		if got != nil || !errors.As(err, &route) || route.Field != "logits" || !errors.As(err, &stage) || !s.halClosed ||
			b.gateCalls != 1 || b.matmuls != 0 || b.routerCalls != 0 {
			t.Fatalf("fault result=%v err=%v closed=%t gate=%d matmuls=%d router=%d", got, err, s.halClosed, b.gateCalls, b.matmuls, b.routerCalls)
		}
	})
	t.Run("backend-failure-keeps-identity", func(t *testing.T) {
		m, s, b, scratch, x, bias, _ := v41RouterGateFixture(t)
		cause := &compute.BackendError{Backend: "vulkan", Site: "V41RouterGateTopK", Class: compute.VulkanClassDeviceLost, Err: compute.ErrVulkanDeviceLost}
		b.gateFault = cause
		_, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		var closed *BackendForwardOperationError
		if !errors.Is(err, compute.ErrVulkanDeviceLost) || !errors.As(err, &closed) || closed.Path != "v41-router-gate-topk" || !s.halClosed || b.matmuls != 0 || b.routerCalls != 0 {
			t.Fatalf("backend failure err=%v closed=%t matmuls=%d router=%d", err, s.halClosed, b.matmuls, b.routerCalls)
		}
	})
	t.Run("scope-change-has-no-replay", func(t *testing.T) {
		m, s, b, scratch, x, bias, _ := v41RouterGateFixture(t)
		b.afterGate = func() { s.cacheGeometryEpoch++ }
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		if got != nil || !errors.Is(err, errV41RouterProjectionScope) || b.gateCalls != 1 || b.matmuls != 0 || b.routerCalls != 0 {
			t.Fatalf("scope change result=%v err=%v gate=%d matmuls=%d router=%d", got, err, b.gateCalls, b.matmuls, b.routerCalls)
		}
	})
	t.Run("bias-panic-keeps-identity", func(t *testing.T) {
		m, _, b, scratch, x, _, _ := v41RouterGateFixture(t)
		primary := errors.New("bias lookup")
		got := recoverError(func() {
			_, _ = m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, func() []float32 { panic(primary) })
		})
		if got != primary || b.gateCalls != 0 || b.matmuls != 0 {
			t.Fatalf("bias panic=%v gate=%d matmuls=%d", got, b.gateCalls, b.matmuls)
		}
	})
}
