package model

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RoutedSwiGLUTestBackend is a cpu-ref recorder for the fused routed seam.
// Its activation uses the model's host primitives, so a session through it is
// a wiring witness only, never a physical device qualification (Ref #13815).
type v41RoutedSwiGLUTestBackend struct {
	*v41HalSeamBackend
	available bool
	calls     int
	reads     int
	fault     *compute.V41RoutedSwiGLUFault
	failure   error
}

func newV41RoutedSwiGLUTestBackend() *v41RoutedSwiGLUTestBackend {
	return &v41RoutedSwiGLUTestBackend{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}, available: true}
}

func (b *v41RoutedSwiGLUTestBackend) V41RoutedSwiGLUAvailable() bool { return b.available }

func (b *v41RoutedSwiGLUTestBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.v41HalSeamBackend.Read(x)
}

func (b *v41RoutedSwiGLUTestBackend) V41RoutedSwiGLU(gate, up compute.Tensor, limit, weight float32) (compute.Tensor, error) {
	b.calls++
	if b.failure != nil {
		return compute.Tensor{}, b.failure
	}
	if b.fault != nil {
		return compute.Tensor{}, b.fault
	}
	g := append([]float32(nil), b.Backend.Read(gate)...)
	u := append([]float32(nil), b.Backend.Read(up)...)
	out := make([]float32, len(g))
	for i := range g {
		a, c := v41RoundBF16(g[i]), v41RoundBF16(u[i])
		if limit > 0 {
			a = min(a, limit)
			c = max(-limit, min(c, limit))
		}
		out[i] = v41RoundBF16(silu(a) * c * weight)
	}
	return b.Backend.Upload(compute.NewF32(compute.Default(), []int{len(out)}, out), compute.F32), nil
}

// fak-test:runtime medium est=5s lane=default
func TestV41RoutedSwiGLUDeviceParity(t *testing.T) {
	for _, limit := range []float64{0, 0.01} {
		m := v41IncrementalExpertFixture(t, true, false)
		if limit > 0 {
			m.Cfg.SwigluLimit = limit
		}
		cfg := m.Cfg
		b := newV41RoutedSwiGLUTestBackend()
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		routed := s.v41RoutedExpertFunc()
		if routed == nil {
			t.Fatal("full-profile session left the fused routed seam unbound")
		}
		x := make([]float32, cfg.HiddenSize)
		for i := range x {
			x[i] = v41OracleBF16(float32(i%5-2) * .23)
		}
		stem := "ffn.experts.1"
		w1, w3, w2, err := m.v41ExpertTripleInto(0, stem, &v41ProjScratch{}, true)
		if err != nil {
			t.Fatal(err)
		}
		forbiddenGate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			t.Fatal("handled fused seam offered gate/up")
			return nil, v41GateUpError, nil
		}
		forbiddenDown := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
			t.Fatal("handled fused seam offered down")
			return nil, v41DownError, nil
		}
		weights := []float32{.501953125, 1, 2.5}
		for _, weight := range weights {
			want := v41RoutedExpertBF16Oracle(w1, w3, w2, x, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit), weight)
			got, err := m.v41FullRoutedExpertWithDevice(0, stem, x, weight, cfg, routed, forbiddenGate, forbiddenDown, func() ([]float32, []float32, []float32, error) {
				t.Fatal("handled fused seam loaded the host triple")
				return nil, nil, nil, nil
			}, func() ([]float32, error) { t.Fatal("handled fused seam loaded the host down"); return nil, nil }, false)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("limit=%g weight=%g: fused output differs from the BF16 oracle (err=%v)", limit, weight, err)
			}
		}
		// Gate, up and down GEMMs run per pick; only the H-wide down output is read.
		if b.calls != len(weights) || b.matmuls != 3*len(weights) || b.reads != len(weights) {
			t.Fatalf("limit=%g: activations=%d matmuls=%d reads=%d, want %d/%d/%d", limit, b.calls, b.matmuls, b.reads, len(weights), 3*len(weights), len(weights))
		}
		s.Close()
	}
}

// Every production prefill (grouped and token-major) and decode path offers the
// pick to the fused seam, and the session matches the two-seam route bit for bit.
// fak-test:runtime slow est=30s lane=default
func TestV41RoutedSwiGLUSessionMatchesSeams(t *testing.T) {
	for _, tokenMajor := range []bool{false, true} {
		previous := v41ForceTokenMajor
		v41ForceTokenMajor = tokenMajor
		func() {
			defer func() { v41ForceTokenMajor = previous }()
			m := v41IncrementalExpertFixture(t, true, false)
			fused := newV41RoutedSwiGLUTestBackend()
			sf, err := m.NewBackendSessionChecked(fused)
			if err != nil {
				t.Fatal(err)
			}
			defer sf.Close()
			sp, err := m.NewBackendSessionChecked(&v41HalSeamBackend{Backend: compute.Default()})
			if err != nil {
				t.Fatal(err)
			}
			defer sp.Close()
			if sp.v41RoutedExpertFunc() != nil {
				t.Fatal("a backend without the routed module bound the fused seam")
			}
			ids := []int{1, 2, 3}
			if got, want := sf.Prefill(ids), sp.Prefill(ids); !reflect.DeepEqual(got, want) {
				t.Fatalf("tokenMajor=%t prefill logits differ between the fused and two-seam routes", tokenMajor)
			}
			rows := len(ids) * m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers
			if fused.calls != rows {
				t.Fatalf("tokenMajor=%t prefill fused activations=%d want %d", tokenMajor, fused.calls, rows)
			}
			if got, want := sf.Step(4), sp.Step(4); !reflect.DeepEqual(got, want) {
				t.Fatalf("tokenMajor=%t decode logits differ between the fused and two-seam routes", tokenMajor)
			}
			if fused.calls != rows+m.Cfg.NumExpertsPerTok*m.Cfg.NumLayers {
				t.Fatalf("tokenMajor=%t decode fused activations=%d", tokenMajor, fused.calls)
			}
		}()
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41RoutedSwiGLUSelectedFailures(t *testing.T) {
	sentinel := &compute.BackendError{Backend: "test", Site: "V41RoutedSwiGLU", Err: errors.New("routed activation fault")}
	for name, c := range map[string]struct {
		fault   *compute.V41RoutedSwiGLUFault
		failure error
		stage   string
		closed  bool
	}{
		"activation fault":   {fault: &compute.V41RoutedSwiGLUFault{Stage: compute.V41RoutedSwiGLUStageActivation, Index: 3}, stage: "gate/up"},
		"down operand fault": {fault: &compute.V41RoutedSwiGLUFault{Stage: compute.V41RoutedSwiGLUStageDownOperand}, stage: "down"},
		"backend failure":    {failure: sentinel, closed: true},
	} {
		t.Run(name, func(t *testing.T) {
			m := v41IncrementalExpertFixture(t, true, false)
			b := newV41RoutedSwiGLUTestBackend()
			b.fault, b.failure = c.fault, c.failure
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			host := 0
			gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				host++
				return nil, v41GateUpDeclined, nil
			}
			down := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
				host++
				return nil, v41DownDeclined, nil
			}
			got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", make([]float32, m.Cfg.HiddenSize), .5, m.Cfg, s.v41RoutedExpertFunc(), gate, down,
				func() ([]float32, []float32, []float32, error) {
					host++
					return nil, nil, nil, errors.New("host triple")
				},
				func() ([]float32, error) { host++; return nil, errors.New("host down") }, false)
			if got != nil || host != 0 || b.calls != 1 {
				t.Fatalf("selected failure published=%v host=%d activations=%d", got, host, b.calls)
			}
			var operation *V41ExpertOperationError
			var closed *BackendForwardOperationError
			switch {
			case c.closed:
				if !errors.As(err, &closed) || !errors.Is(err, sentinel) || closed.Path != "v41-routed-expert" || closed.Stage != "activation" || !s.BackendSessionClosed() {
					t.Fatalf("backend failure err=%v closed=%t", err, s.BackendSessionClosed())
				}
			default:
				if !errors.As(err, &operation) || operation.Stage != c.stage || !errors.Is(err, errV41ExpertResult) || s.BackendSessionClosed() {
					t.Fatalf("numeric fault err=%v closed=%t, want host-typed stage %q", err, s.BackendSessionClosed(), c.stage)
				}
			}
		})
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestV41RoutedSwiGLUDeclineFallsThrough(t *testing.T) {
	cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	declined := func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
		return nil, v41RoutedExpertDeclined, nil
	}
	gateCalls := 0
	gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		gateCalls++
		return []float32{1}, v41GateUpHandled, nil
	}
	down := func(_ int, _ string, h []float32) ([]float32, v41ExpertDownOutcome, error) {
		return []float32{h[0] * 1.5}, v41DownHandled, nil
	}
	got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", []float32{1}, .501953125, cfg, declined, gate, down, nil, nil, false)
	want, wantErr := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, .501953125, cfg, gate, down, nil, nil, false)
	if err != nil || wantErr != nil || gateCalls != 2 || !reflect.DeepEqual(got, want) {
		t.Fatalf("declined fused seam got=%v want=%v gate=%d err=%v/%v", got, want, gateCalls, err, wantErr)
	}
	for _, outcome := range []v41RoutedExpertOutcome{v41RoutedExpertHandled, v41RoutedExpertOutcome(255)} {
		malformed := func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
			return nil, outcome, nil
		}
		var named *V41ExpertOperationError
		if got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", []float32{1}, .5, cfg, malformed, nil, nil, nil, nil, false); got != nil || !errors.As(err, &named) {
			t.Fatalf("malformed fused outcome %v err=%v", outcome, err)
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41RoutedSwiGLUDeviceOnlyBinding(t *testing.T) {
	if slices.ContainsFunc(v41DeviceOnlyHostOps, func(op v41HostOp) bool { return op.name == "routed-swiglu-clamp" }) {
		t.Fatal("routed-swiglu-clamp is still a static host op")
	}
	routedUnbound := func(s *Session) bool {
		return slices.Contains(v41HostOpNames(s.v41DeviceOnlyUnboundOps()), "routed-swiglu-clamp")
	}
	m := v41IncrementalExpertFixture(t, true, false)
	with := newV41RoutedSwiGLUTestBackend()
	without := newV41RoutedSwiGLUTestBackend()
	without.available = false
	for _, c := range []struct {
		b       *v41RoutedSwiGLUTestBackend
		unbound bool
	}{{with, false}, {without, true}} {
		s, err := m.NewBackendSessionChecked(c.b)
		if err != nil {
			t.Fatal(err)
		}
		if got := routedUnbound(s); got != c.unbound {
			t.Fatalf("available=%t routed-swiglu-clamp unbound=%t want %t", c.b.available, got, c.unbound)
		}
		s.Close()
	}
	// The reduced profile clamps through the generic activation instead.
	reduced := v41IncrementalExpertFixture(t, false, false)
	reduced.Cfg.SwigluLimit = 0.01
	rs, err := reduced.NewBackendSessionChecked(newV41RoutedSwiGLUTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	if !routedUnbound(rs) {
		t.Fatal("reduced session without SwiGLUWithLimit admitted a host clamp")
	}
	rs.Close()

	refused := func(t *testing.T, s *Session, err error) {
		t.Helper()
		if !errors.Is(err, errV41DeviceOnlyHostOp) || !s.BackendSessionClosed() || !s.HostFallbackObserved() {
			t.Fatalf("err=%v closed=%t observed=%t, want the typed device-only refusal", err, s.BackendSessionClosed(), s.HostFallbackObserved())
		}
	}
	for name, f := range map[string]v41RoutedExpertFunc{
		"unbound": nil,
		"declined": func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
			return nil, v41RoutedExpertDeclined, nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			s, err := m.NewBackendSessionChecked(newV41RoutedSwiGLUTestBackend())
			if err != nil {
				t.Fatal(err)
			}
			_, outcome, err := v41DeviceOnlyRoutedExpert(s, f)(0, "ffn.experts.0", nil, 1)
			if outcome != v41RoutedExpertError {
				t.Fatalf("outcome %v", outcome)
			}
			refused(t, s, err)
		})
	}
	t.Run("handled passes through", func(t *testing.T) {
		s, err := m.NewBackendSessionChecked(newV41RoutedSwiGLUTestBackend())
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		y, outcome, err := v41DeviceOnlyRoutedExpert(s, func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
			return []float32{7}, v41RoutedExpertHandled, nil
		})(0, "ffn.experts.0", nil, 1)
		if err != nil || outcome != v41RoutedExpertHandled || len(y) != 1 || y[0] != 7 || s.BackendSessionClosed() {
			t.Fatalf("handled call changed: y=%v outcome=%v err=%v", y, outcome, err)
		}
	})
	t.Run("device-only session binds the refusing wrapper", func(t *testing.T) {
		b := newV41RoutedSwiGLUTestBackend()
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		st := s.v41State()
		// Per-pick admission declines a GELU expert after the seam is bound.
		m.Cfg.ActGeluTanh = true
		defer func() { m.Cfg.ActGeluTanh = false }()
		_, outcome, err := st.routedExpert(0, "ffn.experts.0", make([]float32, m.Cfg.HiddenSize), 1)
		if outcome != v41RoutedExpertError || b.calls != 0 {
			t.Fatalf("device-only per-pick decline outcome=%v activations=%d", outcome, b.calls)
		}
		refused(t, s, err)
	})
}
