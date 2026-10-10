package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedActivationRecorder struct {
	*v41DenseTestBackend
	outputs                                           map[compute.Buffer]bool
	calls, limited, readbacks, uploadBytes, readBytes int
	opsAtFault                                        int
	fault                                             error
	badRead                                           bool
}

func newV41SharedActivationRecorder() *v41SharedActivationRecorder {
	return &v41SharedActivationRecorder{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
}

func (b *v41SharedActivationRecorder) activate(g, u compute.Tensor, limit float32) compute.Tensor {
	// Shared activation receives two uploaded host rows. Routed activations
	// receive MatMul outputs; those calls must not inflate this witness.
	shared := b.uploads[g.Buf()] > 0 && b.uploads[u.Buf()] > 0
	if shared {
		b.calls++
		b.uploadBytes += b.uploads[g.Buf()] + b.uploads[u.Buf()]
		if limit > 0 {
			b.limited++
		}
		if b.fault != nil {
			b.opsAtFault = len(b.ops)
			panic(b.fault)
		}
	}
	gate, up := b.Backend.Read(g), b.Backend.Read(u)
	values := make([]float32, len(gate))
	for i := range values {
		a, c := gate[i], up[i]
		if limit > 0 {
			a = min(a, limit)
			c = max(-limit, min(c, limit))
		}
		values[i] = a / (1 + float32(math.Exp(float64(-a)))) * c
	}
	y := compute.NewF32(b, append([]int(nil), g.Shape...), values)
	b.live[y.Buf()] = true
	if shared {
		b.outputs[y.Buf()] = true
	}
	return y
}

func (b *v41SharedActivationRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	return b.activate(g, u, 0)
}

func (b *v41SharedActivationRecorder) Read(x compute.Tensor) []float32 {
	values := b.v41DenseTestBackend.Read(x)
	if b.outputs[x.Buf()] {
		b.readbacks++
		b.readBytes += 4 * len(values)
		if b.badRead {
			b.opsAtFault = len(b.ops)
			return values[:len(values)-1]
		}
	}
	return values
}

func (b *v41SharedActivationRecorder) Free(x compute.Tensor) {
	if b.outputs[x.Buf()] {
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		delete(b.outputs, x.Buf())
	}
	b.v41DenseTestBackend.Free(x)
}

type v41SharedLimitedRecorder struct{ *v41SharedActivationRecorder }

func (b *v41SharedLimitedRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	return b.activate(g, u, limit)
}

// Software dispatch, semantic and lifetime assertions only. The recorder uses
// scalar host arithmetic, and cannot establish hardware execution or net benefit.
// fak-test:runtime medium est=10s lane=default
func TestV41SharedActivationDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("shared activation fixture requires cpu-ref")
	}
	for _, role := range []bool{false, true} {
		for _, scenario := range []struct {
			capable bool
			limit   float64
		}{{false, .01}, {true, .01}, {false, 0}} {
			capable := scenario.capable
			name := "plain"
			if role {
				name = "role"
			}
			if scenario.limit == 0 {
				name += "/zero-limit-device"
			} else if capable {
				name += "/device"
			} else {
				name += "/declined"
			}
			t.Run(name, func(t *testing.T) {
				m := v41IncrementalExpertFixture(t, role, false)
				m.Cfg.SwigluLimit = scenario.limit
				b := newV41SharedActivationRecorder()
				var backend compute.Backend = b
				if capable {
					backend = &v41SharedLimitedRecorder{b}
				}
				s := v41DenseTestSession(t, m, backend)
				v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
				if !s.v41IncrementalEligible() {
					t.Fatal("shared activation fixture did not seed incremental state")
				}
				v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
				want := 0
				if capable || scenario.limit == 0 {
					want = 3 * m.Cfg.NumLayers
				}
				wantLimited := want
				if scenario.limit == 0 {
					wantLimited = 0
				}
				if b.calls != want || b.limited != wantLimited || b.readbacks != want || b.uploadBytes != 8*want*m.Cfg.MoEIntermediateSize || b.readBytes != 4*want*m.Cfg.MoEIntermediateSize || len(b.outputs) != 0 {
					t.Fatalf("shared activation calls/limited/readbacks=%d/%d/%d want=%d upload/read=%d/%d outputs=%d", b.calls, b.limited, b.readbacks, want, b.uploadBytes, b.readBytes, len(b.outputs))
				}
				s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
				var refused *BackendForwardOperationError
				if err := recoverError(func() { s.Step(4) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.calls != want {
					t.Fatalf("strict architecture refusal changed: calls=%d err=%v", b.calls, err)
				}
			})
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41SharedActivationClampAndOwnership(t *testing.T) {
	for _, limit := range []float32{0, -2, 2} {
		b := &v41SharedLimitedRecorder{newV41SharedActivationRecorder()}
		s := &Session{M: &Model{Cfg: Config{MoEIntermediateSize: 4}}, Backend: b, halW: map[string]compute.Tensor{}}
		gate, up := []float32{4, -4, 1, -1}, []float32{-4, 4, 3, -3}
		gateBefore, upBefore := append([]float32(nil), gate...), append([]float32(nil), up...)
		got, outcome, err := s.v41SharedActivationFunc()(0, gate, up, limit)
		if err != nil || outcome != v41ProjectionHandled || len(got) != 4 {
			t.Fatalf("shared activation outcome=%v err=%v", outcome, err)
		}
		for i := range gate {
			g, u := gate[i], up[i]
			if limit > 0 {
				g, u = min(g, limit), max(-limit, min(u, limit))
			}
			want := g / (1 + float32(math.Exp(float64(-g)))) * u
			if !finite32(got[i]) || math.Abs(float64(got[i]-want)) > 1e-6 {
				t.Fatalf("shared activation[%d]=%g want=%g", i, got[i], want)
			}
		}
		if !reflect.DeepEqual(gate, gateBefore) || !reflect.DeepEqual(up, upBefore) || len(b.live) != 0 {
			t.Fatal("shared activation mutated operands or leaked transient tensors")
		}
		s.Close()
	}
}

// fak-test:runtime medium est=6s lane=default
func TestV41SharedActivationSelectedFailure(t *testing.T) {
	for _, fault := range []error{
		&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage},
		&compute.CUDAOpError{Op: "SwiGLU", Err: ErrV41ForwardStage},
		nil,
	} {
		m := v41IncrementalExpertFixture(t, true, false)
		m.Cfg.SwigluLimit = .01
		b := &v41SharedLimitedRecorder{newV41SharedActivationRecorder()}
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		b.live = map[compute.Buffer]bool{}
		calls := b.calls
		b.fault, b.badRead = fault, fault == nil
		err := recoverError(func() { s.Step(3) })
		var operation *V41ProjectionOperationError
		var closed *BackendForwardOperationError
		stage := "limited swiglu"
		if fault == nil {
			stage = "readback"
		}
		if !errors.As(err, &operation) || operation.Stage != string(v41StageMoE) || !errors.As(err, &closed) || closed.Path != "v41-shared-activation" || closed.Stage != stage {
			t.Fatalf("selected shared activation lost failure identity: %v", err)
		}
		if fault != nil && !errors.Is(err, fault) {
			t.Fatal("selected shared activation lost backend cause")
		}
		if b.calls != calls+1 || len(b.ops) != b.opsAtFault || !s.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
			t.Fatalf("shared failure retried, projected down, leaked, or advanced state: calls=%d/%d ops=%d/%d live=%d", b.calls, calls, len(b.ops), b.opsAtFault, len(b.live))
		}
		if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.calls != calls+1 {
			t.Fatalf("closed shared-activation session retried: %v", retry)
		}
	}
}

// Independent source-order oracle; no production projection, activation or
// BF16 helper is used. Generic activation flags do not alter full V4.1.
func v41SharedExpertBF16Oracle(w1, w3, w2, input []float32, I, H int, limit float32) []float32 {
	x := make([]float32, len(input))
	for i, v := range input {
		x[i] = v41OracleBF16(v)
	}
	gate, up := cpuOracleMatVec(w1, x, I, H), cpuOracleMatVec(w3, x, I, H)
	values := make([]float32, I)
	for i := range values {
		g, u := v41OracleBF16(gate[i]), v41OracleBF16(up[i])
		if limit > 0 {
			g = min(g, limit)
			u = max(-limit, min(u, limit))
		}
		activated := g / (1 + float32(math.Exp(float64(-g))))
		values[i] = v41OracleBF16(activated * u)
	}
	out := cpuOracleMatVec(w2, values, H, I)
	for i := range out {
		out[i] = v41OracleBF16(out[i])
	}
	return out
}

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestV41FullSharedExpertHostBF16(t *testing.T) {
	m := v41FullStepModel(t)
	cfg := m.Cfg
	cfg.ActGeluTanh, cfg.ActGeluErf = true, true
	input := make([]float32, cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%7-3) * .137
	}
	for _, limit := range []float64{0, .125} {
		cfg.SwigluLimit = limit
		got, err := m.v41SharedExpertSwiGLUWithActivation(0, input, cfg, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := v41SharedExpertBF16Oracle(cpuOracleTensor(t, m, layerName(0, "ffn.shared_experts.w1.weight")), cpuOracleTensor(t, m, layerName(0, "ffn.shared_experts.w3.weight")), cpuOracleTensor(t, m, layerName(0, "ffn.shared_experts.w2.weight")), input, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(limit))
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("full shared expert BF16 host mismatch, limit=%g", limit)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestV41FullSharedExpertCallbackBF16Ownership(t *testing.T) {
	m := v41FullStepModel(t)
	cfg := m.Cfg
	cfg.ActGeluTanh, cfg.ActGeluErf = true, true
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	input := make([]float32, H)
	input[0] = 1.00390625
	inputBefore := append([]float32(nil), input...)
	storage := make([]float32, max(H, I))
	activation := make([]float32, I)
	for i := range activation {
		activation[i] = []float32{1.00390625, 1.01171875, -1.00390625, -1.01171875}[i%4]
	}
	activationBefore := append([]float32(nil), activation...)
	calls, activated := 0, 0
	var gateSaved, upSaved []float32
	project := func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		calls++
		switch leaf {
		case "ffn.shared_experts.w1.weight":
			if panel[0] != 1 {
				t.Fatal("projection input not BF16")
			}
			for i := 0; i < out; i++ {
				storage[i] = 1.00390625
			}
		case "ffn.shared_experts.w3.weight":
			for i := 0; i < out; i++ {
				storage[i] = 2.0234375
			}
		case "ffn.shared_experts.w2.weight":
			for i, v := range panel {
				if v != v41OracleBF16(activationBefore[i]) {
					t.Fatal("down input not BF16")
				}
			}
			for i := 0; i < out; i++ {
				storage[i] = 1.01171875
			}
		default:
			t.Fatalf("unexpected projection %s", leaf)
		}
		return storage[:out], v41ProjectionHandled, nil
	}
	activate := func(layer int, gate, up []float32, limit float32) ([]float32, v41DenseProjectionOutcome, error) {
		activated++
		gateSaved, upSaved = gate, up
		for i := range gate {
			if gate[i] != 1 || up[i] != 2.03125 {
				t.Fatal("gate/up BF16 publication or projection alias")
			}
		}
		return activation, v41ProjectionHandled, nil
	}
	got, err := m.v41SharedExpertSwiGLUWithActivation(0, input, cfg, project, activate)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 || activated != 1 {
		t.Fatalf("calls=%d activated=%d", calls, activated)
	}
	for i := range storage {
		storage[i] = 77
	}
	for _, v := range got {
		if v != 1.015625 {
			t.Fatal("down output not owned BF16")
		}
	}
	if !reflect.DeepEqual(input, inputBefore) || !reflect.DeepEqual(activation, activationBefore) {
		t.Fatal("borrowed input/activation changed")
	}
	if gateSaved[0] != 1 || upSaved[0] != 2.03125 {
		t.Fatal("activation operands changed")
	}
}

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestV41FullSharedExpertBF16Failures(t *testing.T) {
	for _, leaf := range []string{"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight", "ffn.shared_experts.activation", "ffn.shared_experts.w2.weight"} {
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
			m := v41FullStepModel(t)
			cfg := m.Cfg
			calls := 0
			project := func(layer int, name string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				calls++
				values := make([]float32, out)
				if name == leaf {
					values[0] = bad
				}
				return values, v41ProjectionHandled, nil
			}
			activate := func(layer int, gate, up []float32, limit float32) ([]float32, v41DenseProjectionOutcome, error) {
				values := make([]float32, len(gate))
				if leaf == "ffn.shared_experts.activation" {
					values[0] = bad
				}
				return values, v41ProjectionHandled, nil
			}
			got, err := m.v41SharedExpertSwiGLUWithActivation(0, make([]float32, cfg.HiddenSize), cfg, project, activate)
			var named *V41ProjectionOperationError
			if got != nil || !errors.As(err, &named) || named.Leaf != leaf || named.Stage != string(v41StageMoE) {
				t.Fatalf("leaf=%s got=%v err=%v", leaf, got, err)
			}
			want := 3
			if leaf == "ffn.shared_experts.w1.weight" {
				want = 1
			} else if leaf != "ffn.shared_experts.w2.weight" {
				want = 2
			}
			if calls != want {
				t.Fatalf("leaf=%s projection calls=%d want=%d", leaf, calls, want)
			}
		}
	}
}

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestV41FullSharedExpertActivationDeclineAndError(t *testing.T) {
	m := v41FullStepModel(t)
	cfg := m.Cfg
	input := make([]float32, cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%3-1) * .2
	}
	want, err := m.v41SharedExpertSwiGLUWithActivation(0, input, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	decline := func(int, []float32, []float32, float32) ([]float32, v41DenseProjectionOutcome, error) {
		calls++
		return nil, v41ProjectionDeclined, nil
	}
	got, err := m.v41SharedExpertSwiGLUWithActivation(0, input, cfg, nil, decline)
	if err != nil || calls != 1 || !reflect.DeepEqual(got, want) {
		t.Fatalf("decline calls=%d err=%v", calls, err)
	}
	sentinel := errors.New("selected shared activation failed")
	for _, outcome := range []v41DenseProjectionOutcome{v41ProjectionError, v41ProjectionHandled} {
		projections := 0
		project := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			projections++
			return nil, v41ProjectionDeclined, nil
		}
		fail := func(int, []float32, []float32, float32) ([]float32, v41DenseProjectionOutcome, error) {
			return nil, outcome, sentinel
		}
		got, err := m.v41SharedExpertSwiGLUWithActivation(0, input, cfg, project, fail)
		var named *V41ProjectionOperationError
		if got != nil || !errors.Is(err, sentinel) || !errors.As(err, &named) || named.Leaf != "ffn.shared_experts.activation" || projections != 2 {
			t.Fatalf("selected failure replay: calls=%d err=%v", projections, err)
		}
	}
}

// Full graph consumer coverage uses the existing F32 software backend seam.
// It establishes source dispatch/ownership only, not GPU execution parity.
// fak-test:runtime medium est=4s lane=default
// Estimated only, not measured: two synthetic session layouts with prefill and step.
func TestV41FullSharedExpertSessionBF16(t *testing.T) {
	for _, tokenMajor := range []bool{false, true} {
		previous := v41ForceTokenMajor
		v41ForceTokenMajor = tokenMajor
		func() {
			defer func() { v41ForceTokenMajor = previous }()
			m := v41FullStepPatchedNorms(t)
			m.Cfg.SwigluLimit = .125
			b := newV41SharedActivationRecorder()
			s := v41DenseTestSession(t, m, &v41SharedLimitedRecorder{b})
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() {
				t.Fatal("full fixture did not seed incremental state")
			}
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			want := 3 * m.Cfg.NumLayers
			if b.calls != want || b.limited != want || b.readbacks != want || len(b.outputs) != 0 {
				t.Fatalf("full shared calls/limited/readbacks=%d/%d/%d want=%d", b.calls, b.limited, b.readbacks, want)
			}
		}()
	}
}

// A software-only fault recorder. Releases remove recorder ownership before
// panicking; native reclamation after a failed Free is deliberately not modeled.
type v41SharedActivationCleanupRecorder struct {
	*v41SharedLimitedRecorder
	primary, cleanup any
	readFault        bool
	freeAttempts     int
}

func (b *v41SharedActivationCleanupRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	if b.primary != nil && !b.readFault {
		panic(b.primary)
	}
	return b.v41SharedLimitedRecorder.SwiGLUWithLimit(g, u, limit)
}

func (b *v41SharedActivationCleanupRecorder) Read(x compute.Tensor) []float32 {
	if b.primary != nil && b.readFault && b.outputs[x.Buf()] {
		panic(b.primary)
	}
	return b.v41SharedLimitedRecorder.Read(x)
}

func (b *v41SharedActivationCleanupRecorder) Free(x compute.Tensor) {
	b.freeAttempts++
	b.v41SharedLimitedRecorder.Free(x)
	if b.cleanup != nil {
		panic(b.cleanup)
	}
}

// fak-test:runtime fast est=100ms lane=default
// Estimate is unmeasured. This proves a callback contract in source-authored
// controls; it does not claim an observed native double fault or hardware parity.
func TestV41SharedActivationPrimaryFailureSurvivesCleanup(t *testing.T) {
	typed := &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: errors.New("activation failure")}
	plain := errors.New("rocm-style plain error")
	unknown := &struct{ marker int }{19}
	for _, cleanup := range []any{&compute.BackendError{Backend: "cleanup", Class: compute.VulkanClassExecutionFailed, Err: errors.New("release failure")}, "unknown cleanup panic"} {
		for _, tc := range []struct {
			name                 string
			primary              any
			readFault, malformed bool
			owned                int
		}{
			{"typed activation", typed, false, false, 2},
			{"plain activation", plain, false, false, 2},
			{"unknown activation", unknown, false, false, 2},
			{"typed read", typed, true, false, 3},
			{"plain read", plain, true, false, 3},
			{"unknown read", unknown, true, false, 3},
			{"returned malformed read", nil, false, true, 3},
			{"cleanup only", nil, false, false, 3},
		} {
			t.Run(tc.name, func(t *testing.T) {
				b := &v41SharedActivationCleanupRecorder{v41SharedLimitedRecorder: &v41SharedLimitedRecorder{newV41SharedActivationRecorder()}, primary: tc.primary, cleanup: cleanup, readFault: tc.readFault}
				b.badRead = tc.malformed
				s := &Session{M: &Model{Cfg: Config{MoEIntermediateSize: 4}}, Backend: b, halW: map[string]compute.Tensor{}}
				t.Cleanup(s.Close)
				callback := s.v41SharedActivationFunc()
				gate, up := []float32{1, -2, 3, -4}, []float32{-4, 3, -2, 1}
				gateBefore, upBefore := append([]float32(nil), gate...), append([]float32(nil), up...)
				var result []float32
				var outcome v41DenseProjectionOutcome
				var cause error
				caught := v41IndexerTestRecover(func() { result, outcome, cause = callback(2, gate, up, 2) })
				want := tc.primary
				stage := "limited swiglu"
				if tc.readFault || tc.malformed || tc.primary == nil {
					stage = "readback"
				}
				if tc.malformed {
					if caught != nil || outcome != v41ProjectionError || !errors.Is(cause, errV41ProjectionResult) {
						t.Fatalf("returned primary lost: panic=%v outcome=%v err=%v", caught, outcome, cause)
					}
				} else {
					if want == nil {
						want = cleanup
					}
					if _, classified := want.(*compute.BackendError); classified {
						if caught != nil || outcome != v41ProjectionError || !errors.Is(cause, want.(error)) {
							t.Fatalf("typed primary lost: panic=%v outcome=%v err=%v", caught, outcome, cause)
						}
					} else if caught != want {
						t.Fatalf("original panic identity lost: got=%v want=%v", caught, want)
					}
				}
				var closed *BackendForwardOperationError
				if result != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Path != "v41-shared-activation" || closed.Layer != 2 || closed.Stage != stage || b.freeAttempts != tc.owned || len(b.live) != 0 || len(b.outputs) != 0 {
					t.Fatalf("cleanup/retirement contract: result=%v closed=%v attempts=%d live=%d", result, s.halFailure, b.freeAttempts, len(b.live))
				}
				if tc.malformed && !errors.Is(s.halFailure, errV41ProjectionResult) {
					t.Fatal("latched validation cause lost")
				}
				if primary, ok := tc.primary.(error); ok && !errors.Is(s.halFailure, primary) {
					t.Fatal("latched backend cause lost")
				}
				if !reflect.DeepEqual(gate, gateBefore) || !reflect.DeepEqual(up, upBefore) {
					t.Fatal("caller operands changed")
				}
				ops, calls, frees := len(b.ops), b.calls, b.freeAttempts
				retry := v41IndexerTestRecover(func() { _, _, _ = callback(2, gate, up, 2) })
				if retry != s.halFailure || len(b.ops) != ops || b.calls != calls || b.freeAttempts != frees {
					t.Fatal("retired callback retried device work or cleanup")
				}
			})
		}
	}
}
