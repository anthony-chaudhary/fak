package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41FinalNormTestBackend struct {
	*v41DenseTestBackend
	norms, readsAfterNorm, opsAtFailure int
	failNorm                            error
	normalized                          map[compute.Buffer]bool
}

func (b *v41FinalNormTestBackend) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	b.norms++
	if b.failNorm != nil {
		b.opsAtFailure = len(b.ops)
		panic(b.failNorm)
	}
	y := b.Backend.RMSNorm(x, weight, eps)
	b.live[y.Buf()] = true
	b.normalized[y.Buf()] = true
	return y
}

func (b *v41FinalNormTestBackend) Read(x compute.Tensor) []float32 {
	if b.normalized[x.Buf()] {
		b.readsAfterNorm++
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41FinalNormTestBackend) Free(x compute.Tensor) {
	delete(b.normalized, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// The recorder delegates to cpu-ref. This witnesses dispatch and final-norm
// semantics only; physical device execution and performance remain unqualified.
// fak-test:runtime fast est=500ms lane=default
func TestV41FinalNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("final-norm dispatch fixture requires cpu-ref")
	}
	m := v41IncrementalPlainModel(t, 1)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	newSession := func(t *testing.T) (*Session, *v41FinalNormTestBackend) {
		b := &v41FinalNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), normalized: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		if s.v41State().finalNorm == nil {
			t.Fatal("device session did not bind final RMSNorm")
		}
		return s, b
	}
	t.Run("prefill-decode-and-strict-refusal", func(t *testing.T) {
		s, b := newSession(t)
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-5)
		if b.norms != 2 || b.readsAfterNorm != 2 {
			t.Fatalf("prefill norms/readbacks=%d/%d, want 2/2", b.norms, b.readsAfterNorm)
		}
		if !s.v41IncrementalEligible() {
			t.Fatal("fixture did not seed incremental decode")
		}
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-5)
		if b.norms != 3 || b.readsAfterNorm != 3 || len(b.normalized) != 0 {
			t.Fatalf("decode norms/readbacks/live results=%d/%d/%d", b.norms, b.readsAfterNorm, len(b.normalized))
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := recoverError(func() { s.Step(4) })
		var refused *BackendForwardOperationError
		if !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.norms != 3 {
			t.Fatalf("strict architecture guard changed: norms=%d err=%v", b.norms, err)
		}
	})
	for _, tc := range []struct {
		name         string
		err          error
		unclassified bool
	}{
		{"vulkan", &compute.BackendError{Backend: "cpu-ref", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}, false},
		{"cuda", &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}, false},
		{"wrapped-cuda", fmt.Errorf("norm dispatch: %w", &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}), false},
		{"allocation", &compute.DeviceAllocError{Bytes: 64, Site: "rmsnorm", Class: compute.MemoryScratchpad}, false},
		{"fault", &compute.DeviceFaultError{Backend: "cuda", Site: "rmsnorm"}, false},
		{"unclassified", fmt.Errorf("rocm: rmsnorm: native call failed"), true},
	} {
		t.Run("selected-failure-"+tc.name, func(t *testing.T) {
			s, b := newSession(t)
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.live = map[compute.Buffer]bool{}
			b.failNorm = tc.err
			err := recoverError(func() { s.Step(3) })
			var operation *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			if tc.unclassified {
				if err != tc.err {
					t.Fatalf("unclassified panic identity changed: %v", err)
				}
				if !errors.As(s.halFailure, &closed) {
					t.Fatalf("unclassified failure left no closed-session error: %v", s.halFailure)
				}
			} else if !errors.As(err, &operation) || operation.Stage != string(v41StageFinalNorm) || !errors.As(err, &closed) {
				t.Fatalf("selected final-norm failure lost identity: %v", err)
			}
			if closed == nil || closed.Path != "v41-final-norm" || closed.Stage != "rmsnorm" || !errors.Is(closed, tc.err) {
				t.Fatalf("session did not retain selected failure: %v", closed)
			}
			if b.norms != 3 || len(b.ops) != b.opsAtFailure || !s.BackendSessionClosed() || len(b.live) != 0 {
				t.Fatalf("failed norm retried, projected, or leaked: norms=%d ops=%d/%d closed=%t live=%d", b.norms, len(b.ops), b.opsAtFailure, s.BackendSessionClosed(), len(b.live))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("selected final-norm failure advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 {
				t.Fatalf("closed session retried final norm: norms=%d err=%v", b.norms, retry)
			}
		})
	}
}

// Pinned model.py:288–294,1008–1012,1268–1269: the final learned RMSNorm
// publishes BF16; the head widens it to F32. Estimates below are unmeasured.
// fak-test:runtime fast est=100ms lane=default
func TestV41FullFinalNormBF16Owner(t *testing.T) {
	t.Parallel()
	m := v41FullStepModel(t)
	H := m.Cfg.HiddenSize
	m.Cfg.RMSNormEps, m.Cfg.LogitScale = 8, 1
	input, gain, want := make([]float32, H), make([]float32, H), make([]float32, H)
	for i := range input {
		input[i] = 1.01171875 // odd BF16 tie: rounds to 1.015625 before norm
		gain[i] = []float32{1, 2, -1, -2}[i%4]
		want[i] = []float32{.337890625, .67578125, -.337890625, -.67578125}[i%4]
	}
	v41WriteTensorF32(t, m, "model.norm.weight", gain)
	original := append([]float32(nil), input...)
	if !reflect.DeepEqual(want, v41AttentionInputNormOracle(input, gain, 8)) {
		t.Fatal("fixed norm values disagree with independent F32/BF16 scalar equation")
	}
	for _, useCallback := range []bool{false, true} {
		var captured, borrowed []float32
		calls := 0
		var normalize v41FinalNormFunc
		if useCallback {
			normalize = func(staged []float32) ([]float32, error) {
				calls++
				for _, value := range staged {
					if value != 1.015625 {
						t.Fatal("callback did not receive owned BF16 input")
					}
				}
				borrowed = cpuOracleRMSNorm(staged, gain, 8)
				return borrowed, nil
			}
		}
		project := func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			if layer != -1 || leaf != m.headName() || rows != 1 || in != H {
				t.Fatal("wrong head projection identity")
			}
			captured = panel // Keep the actual operand to catch borrowed callback output.
			values := make([]float32, out)
			for i := range values {
				values[i] = .1234567
			}
			return values, v41ProjectionHandled, nil
		}
		logits, err := m.v41HeadWithFinalNorm(input, project, normalize)
		if err != nil || !reflect.DeepEqual(captured, want) || !reflect.DeepEqual(input, original) {
			t.Fatalf("full final norm owner callback=%t: %v", useCallback, err)
		}
		if useCallback {
			if calls != 1 {
				t.Fatal("selected norm did not run exactly once")
			}
			borrowed[0] = 99
			if captured[0] != want[0] {
				t.Fatal("head operand aliases borrowed callback output")
			}
		}
		for _, value := range logits {
			if value != float32(.1234567) {
				t.Fatal("head logits were rounded away from F32")
			}
		}
	}
	// Direct device callback ABI is still F32; only the full graph owner rounds.
	b := newV41DenseTestBackend()
	device := v41DenseTestSession(t, m, b)
	direct, err := device.v41State().finalNorm(input)
	if err != nil {
		t.Fatal(err)
	}
	assertV41RowsClose(t, "direct F32 final norm callback", direct, cpuOracleRMSNorm(input, gain, 8), 1e-5)
	if reflect.DeepEqual(direct, want) {
		t.Fatal("direct callback was rounded or the ABI control is vacuous")
	}
	// Exact output ties discriminate ties-to-even independently of RMS arithmetic.
	ties, tiesWant := make([]float32, H), make([]float32, H)
	for i := range ties {
		ties[i] = []float32{1.00390625, 1.01171875, -1.00390625, -1.01171875}[i%4]
		tiesWant[i] = []float32{1, 1.015625, -1, -1.015625}[i%4]
	}
	var tieOperand []float32
	_, err = m.v41HeadWithFinalNorm(input, func(_ int, _ string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		tieOperand = panel
		return make([]float32, out), v41ProjectionHandled, nil
	}, func([]float32) ([]float32, error) { return ties, nil })
	if err != nil || !reflect.DeepEqual(tieOperand, tiesWant) {
		t.Fatalf("full final norm output ties: %v", err)
	}
	ties[0] = 99
	if !reflect.DeepEqual(tieOperand, tiesWant) {
		t.Fatal("output ties retained borrowed storage")
	}
	for _, fault := range []string{"selected error", "wrong width", "NaN", "BF16 overflow"} {
		stop := errors.New("selected full final norm failure")
		calls, heads := 0, 0
		normalize := func([]float32) ([]float32, error) {
			calls++
			values := append([]float32(nil), want...)
			switch fault {
			case "selected error":
				return nil, stop
			case "wrong width":
				return values[:H-1], nil
			case "NaN":
				values[0] = float32(math.NaN())
			case "BF16 overflow":
				values[0] = math.MaxFloat32
			}
			return values, nil
		}
		project := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			heads++
			return nil, v41ProjectionDeclined, nil
		}
		_, err := m.v41HeadWithFinalNorm(input, project, normalize)
		var selected *V41ProjectionOperationError
		if !errors.As(err, &selected) || selected.Layer != -1 || selected.Leaf != "model.norm.weight" || selected.Stage != string(v41StageFinalNorm) || calls != 1 || heads != 0 {
			t.Fatalf("%s lost norm identity or replayed: calls=%d heads=%d err=%v", fault, calls, heads, err)
		}
		if fault == "selected error" && !errors.Is(err, stop) {
			t.Fatal("selected cause lost")
		}
	}
	for _, value := range []float32{float32(math.NaN()), math.MaxFloat32} {
		bad := append([]float32(nil), input...)
		bad[0] = value
		bits := math.Float32bits(bad[0])
		calls := 0
		_, err := m.v41HeadWithFinalNorm(bad, nil, func([]float32) ([]float32, error) { calls++; return want, nil })
		var selected *V41ProjectionOperationError
		if !errors.As(err, &selected) || selected.Stage != string(v41StageFinalNorm) || calls != 0 || math.Float32bits(bad[0]) != bits {
			t.Fatal("invalid full norm input escaped, selected callback, or changed caller")
		}
	}
}
