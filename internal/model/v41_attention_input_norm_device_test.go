package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cpu-ref-backed recording witnesses production dispatch, BF16 operands,
// session ownership and cleanup; it makes no physical-device or speed claim.
// fak-test:runtime medium est=5s lane=default
func TestV41AttentionInputNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("attention-input norm software witness requires cpu-ref")
	}
	const leaf = "attn_norm.weight"
	newSession := func(t *testing.T, m *Model) (*Session, *v41QueryNormTestBackend) {
		b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, leaf)] }
		return s, b
	}
	for _, tc := range []struct {
		name             string
		role, tokenMajor bool
	}{
		{"plain-grouped", false, false},
		{"plain-token-major", false, true},
		{"compressed-grouped", true, false},
		{"compressed-token-major", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prior := v41ForceTokenMajor
			v41ForceTokenMajor = tc.tokenMajor
			defer func() { v41ForceTokenMajor = prior }()
			m := v41FullStepPatchedNorms(t)
			if tc.role {
				m = v41IncrementalExpertFixture(t, true, false)
			}
			s, b := newSession(t, m)
			if s.v41State().attentionNormFunc() == nil {
				t.Fatal("device session did not bind attention-input RMSNorm")
			}
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() || b.norms != 2 {
				t.Fatalf("prefill calls=%d incremental=%t", b.norms, s.v41IncrementalEligible())
			}
			weight := s.halW[layerName(0, leaf)]
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			if b.norms != 3 || b.readbacks != 3 || len(b.outputs) != 0 || b.uploadBytes != 3*4*m.Cfg.HiddenSize || b.readBytes != b.uploadBytes || weight.Buf() != s.halW[layerName(0, leaf)].Buf() {
				t.Fatalf("attention norm dispatch/ownership: norms=%d reads=%d outputs=%d upload=%d read=%d", b.norms, b.readbacks, len(b.outputs), b.uploadBytes, b.readBytes)
			}
			for _, value := range b.input {
				if math.Float32bits(value)&0xffff != 0 {
					t.Fatal("device consumed an unrounded collapse")
				}
			}
			snap, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.Close()
			target := v41DenseTestSession(t, m, b)
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			s.Close()
			b.queryWeight = func() compute.Tensor { return target.halW[layerName(0, leaf)] }
			v41GroupedParity(t, target.Prefill([]int{4}), lastLogits(m.Forward([]int{1, 2, 3, 4})), 1e-4)
			if b.norms != 4 || target.v41Forward.callbackOwner != target || target.v41Forward.attentionNormFunc() == nil {
				t.Fatal("suffix restore did not execute target-owned attention normalization")
			}
			before := captureV41ForwardSnapshot(target.v41Forward)
			b.live = map[compute.Buffer]bool{}
			fault := &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			b.fault = fault
			err = recoverError(func() { target.Step(5) })
			var selected *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &selected) || selected.Leaf != leaf || selected.Stage != string(v41StageAttention) || !errors.As(err, &closed) || closed.Path != "v41-attention-input-norm" || closed.Stage != "rmsnorm" || !errors.Is(err, fault) {
				t.Fatalf("selected norm lost failure identity: %v", err)
			}
			if b.norms != 5 || len(b.ops) != b.opsAtFault || !target.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41Forward), before) {
				t.Fatal("failed attention norm replayed, leaked, or advanced state")
			}
			if retry := recoverError(func() { target.Step(5) }); retry != target.halFailure || b.norms != 5 {
				t.Fatal("closed attention norm session retried")
			}
		})
	}
	t.Run("portable and reduced", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		s, b := newSession(t, m)
		if s.v41State().attentionNormFunc() != nil {
			t.Fatal("reduced graph bound full-profile input normalization")
		}
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-5)
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-5)
		var absent *Session
		if b.norms != 0 || absent.v41AttentionInputNormFunc() != nil || (&Session{M: m, Backend: compute.Default()}).v41AttentionInputNormFunc() != nil {
			t.Fatal("portable/reduced session selected device attention-input norm")
		}
	})
}

// The resident mHC branch has separate call sites in grouped and one-token
// forward. Both must reach the same normalization callback as host-held streams.
// fak-test:runtime medium est=3s lane=default
func TestV41AttentionInputNormResidentDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("attention-input norm software witness requires cpu-ref")
	}
	for _, role := range []bool{false, true} {
		m := v41FullStepPatchedNorms(t)
		if role {
			m = v41IncrementalExpertFixture(t, true, false)
		}
		b := newV41MHCTestBackend(true)
		s := v41DenseTestSession(t, m, b)
		st := s.v41State()
		normalize, calls := st.attentionNormFunc(), 0
		if normalize == nil {
			t.Fatal("resident session did not bind attention normalization")
		}
		st.setAttentionNorm(func(layer int, input []float32) ([]float32, error) {
			calls++
			return normalize(layer, input)
		})
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
		if calls != 3 || b.calls["rms-split"] == 0 || len(b.outputs) != 0 {
			t.Fatalf("role=%t resident path missing norm or leaked: norms=%d mhc=%v live=%d", role, calls, b.calls, len(b.outputs))
		}
	}
}

// An odd BF16 tie makes both dtype boundaries observable independently of the
// later attention projections. The callback's output remains caller-owned.
// fak-test:runtime fast est=100ms lane=default
func TestV41AttentionInputNormDeviceBoundaries(t *testing.T) {
	t.Parallel()
	m := &Model{Cfg: Config{HiddenSize: 4, RMSNormEps: 8}}
	input := []float32{1.01171875, 1.01171875, 1.01171875, 1.01171875}
	gain := []float32{1, 2, -1, -2}
	before := append([]float32(nil), input...)
	var output []float32
	got, err := m.v41AttentionInputNormWithDevice(0, input, 8, func(layer int, staged []float32) ([]float32, error) {
		if layer != 0 || !reflect.DeepEqual(staged, []float32{1.015625, 1.015625, 1.015625, 1.015625}) {
			t.Fatal("callback consumed wrong BF16 collapse")
		}
		output = cpuOracleRMSNorm(staged, gain, 8)
		return output, nil
	})
	if err != nil || !reflect.DeepEqual(got, v41AttentionInputNormOracle(input, gain, 8)) || !reflect.DeepEqual(input, before) {
		t.Fatalf("device boundary got=%v err=%v", got, err)
	}
	output[0] = 99
	if got[0] == 99 {
		t.Fatal("BF16 result aliases callback output")
	}
}
