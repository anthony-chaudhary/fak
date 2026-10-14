package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage records ownership and avoided boundaries, not device performance.
type v41RouterResidentRecorder struct {
	allocated map[compute.Buffer]bool

	afterAvailable func()

	compute.Backend
	mu                                                                      sync.Mutex
	live                                                                    map[compute.Buffer]string
	frees                                                                   map[compute.Buffer]int
	output                                                                  compute.Buffer
	matmuls, routerCalls, outputReads, logitUploads, probes, availableCalls int
	deny, unavailable                                                       bool
	fault                                                                   error
	panicFault, cleanup                                                     any
	afterMatMul                                                             func()
	badShape                                                                bool
	nativeInputs                                                            int
}

func newV41RouterResidentRecorder() *v41RouterResidentRecorder {
	return &v41RouterResidentRecorder{Backend: compute.Default(), allocated: map[compute.Buffer]bool{}, live: map[compute.Buffer]string{}, frees: map[compute.Buffer]int{}}
}
func (b *v41RouterResidentRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41RouterResidentRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}
func (b *v41RouterResidentRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	b.probes++
	return !b.deny && dt == compute.F32
}
func (b *v41RouterResidentRecorder) V41RouterTopKAvailable() bool {
	b.availableCalls++
	if b.afterAvailable != nil {
		b.afterAvailable()
	}
	return !b.unavailable
}
func (b *v41RouterResidentRecorder) Upload(t compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(t, dt, compute.MemoryActivation, "unclassified")
}
func (b *v41RouterResidentRecorder) UploadClass(t compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("fixture only supports F32")
	}
	out := compute.NewF32(b, append([]int(nil), t.Shape...), append([]float32(nil), b.Backend.Read(t)...))
	b.live[out.Buf()] = site
	b.allocated[out.Buf()] = true
	if site == "V4.1 router logits" {
		b.logitUploads++
	}
	return out
}
func (b *v41RouterResidentRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	tmp := b.Backend.MatMul(w, x)
	out := compute.NewF32(b, append([]int(nil), tmp.Shape...), append([]float32(nil), b.Backend.Read(tmp)...))
	b.Backend.Free(tmp)
	b.live[out.Buf()] = "projection output"
	b.allocated[out.Buf()] = true
	b.output = out.Buf()
	if b.afterMatMul != nil {
		b.afterMatMul()
	}
	if b.badShape {
		out.Shape = []int{1, out.Shape[0]}
	}
	return out
}
func (b *v41RouterResidentRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	tmp := b.Backend.BatchedMatMul(w, x, rows)
	out := compute.NewF32(b, append([]int(nil), tmp.Shape...), append([]float32(nil), b.Backend.Read(tmp)...))
	b.Backend.Free(tmp)
	b.live[out.Buf()] = "projection output"
	b.allocated[out.Buf()] = true
	b.output = out.Buf()
	return out
}
func (b *v41RouterResidentRecorder) Read(t compute.Tensor) []float32 {
	// Inherited CPU primitives own their outputs. Never let changed tensor
	// metadata bypass this recorder's allocation history after retirement.
	if t.Backend() == b.Backend && !b.allocated[t.Buf()] {
		return b.Backend.Read(t)
	}
	if _, ok := b.live[t.Buf()]; !ok {
		panic("read after free")
	}
	if t.Buf() == b.output {
		b.outputReads++
	}
	return b.Backend.Read(t)
}
func (b *v41RouterResidentRecorder) Free(t compute.Tensor) {
	if t.Backend() == b.Backend && !b.allocated[t.Buf()] {
		b.Backend.Free(t)
		return
	}
	site, ok := b.live[t.Buf()]
	if !ok {
		panic("double or foreign free")
	}
	delete(b.live, t.Buf())
	b.frees[t.Buf()]++
	b.Backend.Free(t)
	if b.cleanup != nil && (site == "projection output" || site == "V4.1 dense projection activation") {
		panic(b.cleanup)
	}
}
func (b *v41RouterResidentRecorder) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.routerCalls++
	if _, ok := b.live[logits.Buf()]; !ok || logits.Backend() != b || !reflect.DeepEqual(logits.Shape, []int{1, experts}) {
		return compute.V41RouterPicks{}, errors.New("router loan is not live/exact")
	}
	if logits.Buf() == b.output {
		b.nativeInputs++
	}
	if b.panicFault != nil {
		panic(b.panicFault)
	}
	if b.fault != nil {
		return compute.V41RouterPicks{}, b.fault
	}
	var correction []float32
	if hasBias {
		correction = b.Backend.Read(bias)
	}
	picks, err := v41RouteForGeometry(b.Backend.Read(logits), correction, v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: scale}, full)
	if err != nil {
		return compute.V41RouterPicks{}, err
	}
	out := compute.V41RouterPicks{}
	for _, p := range picks {
		out.Experts = append(out.Experts, int32(p.expert))
		out.Weights = append(out.Weights, p.weight)
	}
	return out, nil
}

// Register before Session construction: scope finish and Session.Close run
// first, then the held model weights retire, then all allocations are checked.
func v41RouterResidentModelCleanup(t *testing.T, m *Model, backends ...*v41RouterResidentRecorder) {
	t.Helper()
	t.Cleanup(func() {
		for _, b := range backends {
			if len(b.live) != 0 {
				t.Errorf("model/session cleanup retained %d buffers", len(b.live))
			}
			for buffer := range b.allocated {
				if b.frees[buffer] != 1 {
					t.Errorf("tracked allocation freed %d times, want 1", b.frees[buffer])
				}
			}
		}
	})
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	})
}

func v41RouterResidentFixture(t *testing.T) (*Model, *Session, *v41RouterResidentRecorder, *v41ProjScratch, []float32, func() []float32) {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	b := newV41RouterResidentRecorder()
	v41RouterResidentModelCleanup(t, m, b)
	s := v41DenseTestSession(t, m, b)
	st := s.v41State()
	scope := s.beginBackendExecution()
	t.Cleanup(scope.finish)
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%7-3) / 16
	}
	return m, s, b, &v41ProjScratch{projectionRoute: st.projectionRoute}, x, func() []float32 { return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")) }
}

// fak-test:runtime medium est=3s lane=default
func TestV41RouterResidentBinding(t *testing.T) {
	t.Run("same-picks-and-one-owned-loan", func(t *testing.T) {
		m, _, b, scratch, x, bias := v41RouterResidentFixture(t)
		cfg := v41DefaultRouterConfig()
		logits, err := m.v41ProjMatRows(0, "ffn.gate.weight", x, cfg.Experts, m.Cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		want, err := v41RouteForGeometry(logits, bias(), cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, cfg, true, scratch, bias)
		if err != nil || !reflect.DeepEqual(got, want) || b.matmuls != 1 || b.routerCalls != 1 || b.nativeInputs != 1 || b.outputReads != 0 || b.logitUploads != 0 || b.frees[b.output] != 1 {
			t.Fatalf("native result=%v err=%v matmuls=%d router=%d native=%d reads=%d uploads=%d frees=%d", got, err, b.matmuls, b.routerCalls, b.nativeInputs, b.outputReads, b.logitUploads, b.frees[b.output])
		}
	})
	t.Run("wrapped-projection-and-router-stay-ordinary", func(t *testing.T) {
		m, s, b, scratch, x, bias := v41RouterResidentFixture(t)
		st := s.v41State()
		projection, router := st.denseProjection(), st.routerTopK()
		pc, rc := 0, 0
		st.setDenseProjection(func(l int, leaf string, p []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			pc++
			y, o, e := projection(l, leaf, p, out, in, rows)
			if len(y) > 0 {
				y[0] += 20
			}
			return y, o, e
		})
		st.setRouterTopK(func(l int, logits, correction []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
			rc++
			picks, e := router(l, logits, correction, cfg, full)
			if len(picks) > 0 {
				picks[0].weight += 3
			}
			return picks, e
		})
		scratch.projectionRoute = st.projectionRoute
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		if err != nil || pc != 1 || rc != 1 || len(got) != 6 || got[0].weight < 3 || b.nativeInputs != 0 || b.outputReads != 1 || b.logitUploads != 1 {
			t.Fatalf("wrapper result=%v err=%v pc=%d rc=%d native=%d", got, err, pc, rc, b.nativeInputs)
		}
	})
	t.Run("one-call-decline-and-device-only", func(t *testing.T) {
		for _, deviceOnly := range []bool{false, true} {
			m, s, b, scratch, x, bias := v41RouterResidentFixture(t)
			b.deny = true
			if deviceOnly {
				s.v41Forward.liftDeviceOnlyDense(s)
				scratch.projectionRoute = s.v41Forward.projectionRoute
			}
			before := b.probes
			got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
			if b.probes-before != 1 || b.matmuls != 0 || b.nativeInputs != 0 {
				t.Fatal("declined projection was invoked twice or selected native work")
			}
			if deviceOnly {
				if err == nil || got != nil || !s.halClosed || !s.hostFallbackObserved || b.routerCalls != 0 {
					t.Fatal("DeviceOnly decline escaped")
				}
			} else if err != nil || len(got) != 6 || b.routerCalls != 1 || b.logitUploads != 1 {
				t.Fatalf("decline continuation=%v err=%v", got, err)
			}
		}
	})
	t.Run("projection-admission-precedes-bias", func(t *testing.T) {
		m, _, b, scratch, x, _ := v41RouterResidentFixture(t)
		calls := 0
		_, err := m.v41RouterPicksForStep(0, x[:len(x)-1], m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, func() []float32 { calls++; panic("early bias") })
		if err == nil || calls != 0 || b.matmuls != 0 || b.routerCalls != 0 {
			t.Fatal("invalid projection observed later bias/router")
		}
	})
	t.Run("bias-panic-keeps-identity-through-free", func(t *testing.T) {
		m, _, b, scratch, x, _ := v41RouterResidentFixture(t)
		primary := errors.New("bias lookup")
		b.cleanup = errors.New("secondary free")
		got := recoverError(func() {
			_, _ = m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, func() []float32 { panic(primary) })
		})
		b.cleanup = nil
		if got != primary || b.routerCalls != 0 || b.frees[b.output] != 1 {
			t.Fatalf("bias panic=%v router=%d frees=%d", got, b.routerCalls, b.frees[b.output])
		}
	})
	t.Run("router-primary-survives-native-cleanup", func(t *testing.T) {
		for _, panicked := range []bool{false, true} {
			m, s, b, scratch, x, bias := v41RouterResidentFixture(t)
			primary := &BackendForwardOperationError{Path: "injected-router", Cause: errors.New("primary")}
			b.cleanup = errors.New("secondary free")
			if panicked {
				b.panicFault = primary
			} else {
				b.fault = primary
			}
			got := recoverError(func() {
				_, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
				if err != nil {
					panic(err)
				}
			})
			b.cleanup = nil
			if (panicked && got != primary) || !errors.Is(got, primary) || !s.halClosed || b.routerCalls != 1 || b.matmuls != 1 || b.frees[b.output] != 1 {
				t.Fatalf("primary=%v closed=%t calls=%d", got, s.halClosed, b.routerCalls)
			}
		}
	})
	t.Run("epoch-change-has-no-replay", func(t *testing.T) {
		m, s, b, scratch, x, bias := v41RouterResidentFixture(t)
		b.afterMatMul = func() { s.cacheGeometryEpoch++ }
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		if got != nil || err == nil || b.matmuls != 1 || b.routerCalls != 0 || b.outputReads != 0 || b.logitUploads != 0 || b.frees[b.output] != 1 {
			t.Fatal("changed epoch replayed or escaped")
		}
	})
	t.Run("malformed-output-does-not-enter-router", func(t *testing.T) {
		m, _, b, scratch, x, bias := v41RouterResidentFixture(t)
		b.badShape = true
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
		var projection *V41ProjectionOperationError
		if got != nil || !errors.As(err, &projection) || b.routerCalls != 0 || b.matmuls != 1 || b.frees[b.output] != 1 {
			t.Fatal("malformed native output escaped")
		}
	})
}

// fak-test:runtime medium est=3s lane=default
func TestV41RouterResidentBindingOwnership(t *testing.T) {
	t.Run("factory-unavailable-router-is-not-reopened", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41RouterResidentRecorder()
		b.unavailable = true
		v41RouterResidentModelCleanup(t, m, b)
		s := v41DenseTestSession(t, m, b)
		if b.availableCalls != 0 {
			t.Fatal("constructor observed router availability before state binding")
		}
		st := s.v41State()
		binding := st.projectionRoute
		if b.availableCalls != 1 || binding == nil || binding.projection == nil || binding.nativeProjection.core == nil ||
			binding.nativeProjection.owner != s || binding.nativeProjection.model != m || binding.nativeProjection.backend != b ||
			binding.router != nil || binding.nativeRouter.core != nil {
			t.Fatal("original factories did not capture the unavailable router half")
		}
		x := make([]float32, m.Cfg.HiddenSize)
		for i := range x {
			x[i] = float32(i%7-3) / 16
		}
		cfg := v41DefaultRouterConfig()
		logits, err := m.v41ProjMatRows(0, "ffn.gate.weight", x, cfg.Experts, m.Cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		bias := func() []float32 { return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")) }
		want, err := v41RouteForGeometry(logits, bias(), cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		scope := s.beginBackendExecution()
		t.Cleanup(scope.finish)
		scratch := &v41ProjScratch{projectionRoute: binding}
		b.unavailable = false // No setter, callback replacement, factory call or rebind.
		m.v41SetExpertFaultPhase(V41PhaseDecode)
		before := m.V41ExpertFaultAttribution().Decode
		got, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, cfg, true, scratch, bias)
		after := m.V41ExpertFaultAttribution().Decode
		if err != nil || !reflect.DeepEqual(got, want) || b.matmuls != 1 || b.outputReads != 1 ||
			b.routerCalls != 0 || b.nativeInputs != 0 || b.logitUploads != 0 || b.availableCalls != 1 {
			t.Fatalf("captured unavailable route changed: err=%v matmuls=%d reads=%d router=%d native=%d uploads=%d probes=%d", err, b.matmuls, b.outputReads, b.routerCalls, b.nativeInputs, b.logitUploads, b.availableCalls)
		}
		if s.v41State() != st || st.projectionRoute != binding || st.callbackOwner != s || b.availableCalls != 1 || s.halClosed || s.halFailure != nil {
			t.Fatal("ordinary continuation rebound or retired the live session")
		}
		if after.DenseProjectionDeviceCalls-before.DenseProjectionDeviceCalls != 1 || after.DenseProjectionDeviceRows-before.DenseProjectionDeviceRows != 1 ||
			after.DenseProjectionReadbackBytes-before.DenseProjectionReadbackBytes != int64(4*cfg.Experts) || after.DenseProjectionActivationUploadBytes-before.DenseProjectionActivationUploadBytes != int64(4*len(x)) ||
			after.DenseProjectionHostCalls-before.DenseProjectionHostCalls != 0 || after.DenseProjectionHostRows-before.DenseProjectionHostRows != 0 {
			t.Fatalf("ordinary projection ledger changed: before=%+v after=%+v", before, after)
		}
	})
	t.Run("inherited-cpu-output-cannot-hide-retired-recorder-buffer", func(t *testing.T) {
		b := newV41RouterResidentRecorder()
		x := compute.NewF32(b.Backend, []int{2}, []float32{1, 2})
		gain := compute.NewF32(b.Backend, []int{2}, []float32{1, 1})
		y := b.RMSNorm(x, gain, 1e-5)
		if y.Backend() != b.Backend || b.allocated[y.Buf()] || !reflect.DeepEqual(b.Read(y), b.Backend.Read(y)) {
			t.Fatal("inherited RMSNorm did not retain its CPU owner")
		}
		b.Free(y)
		owned := b.Upload(x, compute.F32)
		b.Backend.Free(x)
		b.Backend.Free(gain)
		b.Free(owned)
		// Tensor owner metadata is private to compute. Model the adversarial
		// alias directly: a CPU-labelled buffer with recorder allocation history.
		alias := compute.NewF32(b.Backend, []int{2}, []float32{3, 4})
		b.allocated[alias.Buf()] = true
		b.live[alias.Buf()] = "recorded CPU-labelled alias"
		b.Free(alias)
		unknown := compute.NewF32(b, []int{2}, []float32{5, 6})
		recoverValue := func(fn func()) (value any) {
			defer func() { value = recover() }()
			fn()
			return nil
		}
		for i, tensor := range []compute.Tensor{owned, alias, unknown} {
			readPanic := recoverValue(func() { b.Read(tensor) })
			freePanic := recoverValue(func() { b.Free(tensor) })
			if readPanic != "read after free" || freePanic != "double or foreign free" {
				t.Fatalf("buffer %d lifetime refusal: Read=%v Free=%v", i, readPanic, freePanic)
			}
		}
		if len(b.live) != 0 || b.frees[owned.Buf()] != 1 || b.frees[alias.Buf()] != 1 || b.frees[unknown.Buf()] != 0 {
			t.Fatal("rejected accesses changed recorder retirement counts")
		}
		b.Backend.Free(unknown)
	})
	t.Run("ordinary-dense-is-dynamic-router-op-is-captured", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		first, second := newV41RouterResidentRecorder(), newV41RouterResidentRecorder()
		v41RouterResidentModelCleanup(t, m, first, second)
		s := &Session{M: m, Backend: first, halW: map[string]compute.Tensor{}}
		projection, router := s.v41DenseProjectionFunc(), s.v41RouterTopKFunc()
		if first.availableCalls != 1 {
			t.Fatal("router availability was not observed once at binding")
		}
		first.unavailable = true
		s.Backend = second
		x := make([]float32, m.Cfg.HiddenSize)
		logits, outcome, err := projection(0, "ffn.gate.weight", x, m.Cfg.NumExperts, m.Cfg.HiddenSize, 1)
		if err != nil || outcome != v41ProjectionHandled || first.matmuls != 0 || second.matmuls != 1 {
			t.Fatal("ordinary dense stopped following Session.Backend")
		}
		_, _ = router(0, logits, nil, v41DefaultRouterConfig(), true)
		if first.routerCalls != 1 || second.routerCalls != 0 || first.availableCalls != 1 {
			t.Fatal("ordinary router reopened its captured operation")
		}
	})
	t.Run("availability-mutation-cannot-forge-native-provenance", func(t *testing.T) {
		m, s, original, scratch, _, _ := v41RouterResidentFixture(t)
		replacement := newV41RouterResidentRecorder()
		defer func() { s.Backend = original; original.afterAvailable = nil }()
		s.Backend = replacement
		projection := s.v41BindDenseProjection()
		s.Backend = original
		original.afterAvailable = func() { s.Backend = replacement }
		before := original.availableCalls
		router := s.v41BindRouterTopK()
		if router.backend != original || s.Backend != replacement || original.availableCalls != before+1 {
			t.Fatal("router provenance did not retain its type-asserted backend")
		}
		st := s.v41Forward
		st.bindCanonicalDense(projection)
		st.bindCanonicalRouter(router)
		scratch.projectionRoute = st.projectionRoute
		if st.projectionRoute.nativeOwner(m, scratch, v41DefaultRouterConfig(), true, 0) != nil || original.availableCalls != before+1 || replacement.availableCalls != 0 {
			t.Fatal("availability mutation selected native or performed a second probe")
		}
	})

	t.Run("scope-provenance-and-transform-gates", func(t *testing.T) {
		m, s, _, scratch, _, _ := v41RouterResidentFixture(t)
		binding := scratch.projectionRoute
		cfg := v41DefaultRouterConfig()
		if binding.nativeOwner(m, scratch, cfg, true, 0) != s {
			t.Fatal("canonical owner was not eligible")
		}
		oldBackend := s.Backend
		s.Backend = newV41RouterResidentRecorder()
		if binding.nativeOwner(m, scratch, cfg, true, 0) != nil {
			t.Fatal("backend mismatch inherited native authority")
		}
		s.Backend = oldBackend
		scope := s.backendExecutionScope
		s.backendExecutionScope = nil
		if binding.nativeOwner(m, scratch, cfg, true, 0) != nil {
			t.Fatal("missing execution scope selected")
		}
		s.backendExecutionScope = scope
		m.lora = NewLoRASet()
		if binding.nativeOwner(m, scratch, cfg, true, 0) != nil {
			t.Fatal("host LoRA transform was bypassed")
		}
		m.lora = nil
		old := s.v41Forward.projectionRoute
		s.v41Forward.setDenseProjection(s.v41Forward.denseProjection())
		if binding.nativeOwner(m, scratch, cfg, true, 0) != nil {
			t.Fatal("stale scratch binding selected")
		}
		s.v41Forward.projectionRoute = old
		name := layerName(0, "ffn.gate.e_score_correction_bias")
		meta := m.manifest[name]
		bad := meta
		bad.Offset = len(m.raw)
		m.manifest[name] = bad
		if binding.nativeOwner(m, scratch, cfg, true, 0) != nil {
			t.Fatal("unsafe delayed bias view selected")
		}
		m.manifest[name] = meta
	})
	t.Run("completed-versus-transport-unknown-accounting", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			fault    error
			wantRows int
		}{
			{"success", nil, 1},
			{"numeric", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageLogit, Expert: 2}, 1},
			{"transport", &compute.BackendError{Backend: "router-recorder", Site: "V41RouterTopK", Err: errors.New("transport unknown")}, 0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				m, _, b, scratch, x, bias := v41RouterResidentFixture(t)
				b.fault = tc.fault
				m.v41SetExpertFaultPhase(V41PhaseDecode)
				before := m.V41ExpertFaultAttribution().Decode
				_, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, v41DefaultRouterConfig(), true, scratch, bias)
				after := m.V41ExpertFaultAttribution().Decode
				if (err != nil) != (tc.fault != nil) {
					t.Fatalf("fault result=%v", err)
				}
				if tc.name == "numeric" {
					var route *v4RouteError
					if !errors.As(err, &route) || route.Field != "logits" {
						t.Fatalf("numeric mapping=%v", err)
					}
				}
				if tc.name == "transport" && !errors.Is(err, tc.fault) {
					t.Fatalf("transport cause lost: %v", err)
				}
				if after.DenseProjectionDeviceCalls-before.DenseProjectionDeviceCalls != 1 || after.DenseProjectionDeviceRows-before.DenseProjectionDeviceRows != tc.wantRows ||
					after.DenseProjectionReadbackBytes-before.DenseProjectionReadbackBytes != 0 || after.DenseProjectionActivationUploadBytes-before.DenseProjectionActivationUploadBytes != int64(4*len(x)) ||
					after.DenseProjectionHostCalls-before.DenseProjectionHostCalls != 0 || after.DenseProjectionHostRows-before.DenseProjectionHostRows != 0 ||
					b.outputReads != 0 || b.logitUploads != 0 || b.nativeInputs != 1 || b.routerCalls != 1 {
					t.Fatalf("completion accounting before=%+v after=%+v rows=%d", before, after, tc.wantRows)
				}
			})
		}
	})
}

// Both production incremental branches must select only after cold Prefill has
// kept its ordinary panel projection and host-row router behavior.
// fak-test:runtime medium est=5s lane=default
func TestV41RouterResidentBothStepBranches(t *testing.T) {
	for _, role := range []bool{false, true} {
		t.Run("role="+itoa(boolToIntV41Expert(role)), func(t *testing.T) {
			m := v41IncrementalExpertFixture(t, true, false)
			if !role {
				m.Cfg.DeepSeekV41.CompressRatios = []int{0}
				m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
				m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
			}
			b, ordinary := newV41RouterResidentRecorder(), newV41RouterResidentRecorder()
			v41RouterResidentModelCleanup(t, m, b, ordinary)
			s, control := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, ordinary)
			controlState := control.v41State()
			// Replacing the ordinary projection with itself deliberately drops
			// the whole native pair while retaining the exact bound callbacks.
			controlState.setDenseProjection(controlState.denseProjection())
			if got, want := s.Prefill([]int{1, 2}), control.Prefill([]int{1, 2}); !reflect.DeepEqual(got, want) ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(s.v41State()), captureV41ForwardSnapshot(control.v41State())) {
				t.Fatal("cold Prefill differs from ordinary-only pair")
			}
			if b.nativeInputs != 0 || ordinary.nativeInputs != 0 || b.logitUploads == 0 || ordinary.logitUploads == 0 {
				t.Fatal("cold prefill unexpectedly changed route boundary")
			}
			for _, token := range []int{3, 4} {
				beforeNative, beforeUploads, beforeOrdinaryUploads := b.nativeInputs, b.logitUploads, ordinary.logitUploads
				got, want := s.Step(token), control.Step(token)
				if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41State()), captureV41ForwardSnapshot(control.v41State())) {
					t.Fatalf("token %d native logits or continuation snapshot differs from ordinary-only pair", token)
				}
				if b.nativeInputs-beforeNative != m.Cfg.NumLayers || b.logitUploads != beforeUploads ||
					ordinary.nativeInputs != 0 || ordinary.logitUploads-beforeOrdinaryUploads != m.Cfg.NumLayers {
					t.Fatalf("token %d did not exercise both paths: native=%d uploads=%d ordinary native=%d uploads=%d", token, b.nativeInputs-beforeNative, b.logitUploads-beforeUploads, ordinary.nativeInputs, ordinary.logitUploads-beforeOrdinaryUploads)
				}
			}
		})
	}
}
