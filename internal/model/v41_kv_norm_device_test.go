package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Reuse the query norm recorder with the KV gain as its selected identity. It
// counts only this norm's uploads/readbacks and poisons its output on Free.
// cpu-ref arithmetic establishes dispatch and lifetime semantics, not physical
// device qualification or a throughput gain.
// fak-test:runtime medium est=5s lane=default
func TestV41KVNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("KV-norm dispatch fixture requires cpu-ref")
	}
	const leaf = "attn.kv_norm.weight"
	newSession := func(t *testing.T, m *Model) (*Session, *v41QueryNormTestBackend) {
		b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, leaf)] }
		if s.v41State().kvNorm == nil {
			t.Fatal("device session did not bind KV RMSNorm")
		}
		return s, b
	}
	for _, role := range []bool{false, true} {
		name := "plain"
		if role {
			name = "compressed-role"
		}
		t.Run(name, func(t *testing.T) {
			var m *Model
			if role {
				m = v41IncrementalExpertFixture(t, true, false)
			} else {
				m = v41FullStepPatchedNorms(t)
			}
			s, b := newSession(t, m)
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() || b.norms != 2 || b.readbacks != 2 {
				t.Fatalf("prefill KV calls/readbacks=%d/%d incremental=%t", b.norms, b.readbacks, s.v41IncrementalEligible())
			}
			weight := s.halW[layerName(0, leaf)]
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			if b.norms != 3 || b.readbacks != 3 || len(b.outputs) != 0 || b.uploadBytes != 3*4*v41KVLoraRank || b.readBytes != b.uploadBytes {
				t.Fatalf("KV dispatch/transfer/cleanup mismatch: norms=%d reads=%d outputs=%d upload=%d read=%d", b.norms, b.readbacks, len(b.outputs), b.uploadBytes, b.readBytes)
			}
			if weight.Buf() == nil || s.halW[layerName(0, leaf)].Buf() != weight.Buf() || !reflect.DeepEqual(b.gain, m.tensor(layerName(0, leaf))) {
				t.Fatal("KV norm did not reuse its cached learned gain")
			}
			// A direct oracle catches a wrong norm even if later attention masks
			// it. Free poisoning detects borrowed, rather than owned, readback.
			input := append([]float32(nil), b.input...)
			got, err := s.v41State().kvNorm(0, input)
			if err != nil {
				t.Fatal(err)
			}
			v41GroupedParity(t, got, cpuOracleRMSNorm(input, b.gain, float32(m.Cfg.RMSNormEps)), 1e-5)
			if len(input) != v41KVLoraRank || !reflect.DeepEqual(input, b.input) {
				t.Fatal("KV norm changed its host input or latent width")
			}

			// Restore onto the same backend, close the source, then append via
			// suffix prefill. A captured source closure would use a closed session.
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
			beforeNorms := b.norms
			v41GroupedParity(t, target.Prefill([]int{4}), lastLogits(m.Forward([]int{1, 2, 3, 4})), 1e-4)
			if target.v41Forward.callbackOwner != target || target.v41Forward.kvNorm == nil || b.norms != beforeNorms+1 || target.halW[layerName(0, leaf)].Buf() == nil {
				t.Fatal("restored KV norm did not bind and execute on the target session")
			}
			beforeNorms = b.norms
			target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			var refused *BackendForwardOperationError
			if err := recoverError(func() { target.Step(5) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || b.norms != beforeNorms {
				t.Fatalf("strict architecture guard changed: calls=%d/%d err=%v", b.norms, beforeNorms, err)
			}
		})
	}
	t.Run("reduced-profile", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		s, b := newSession(t, m)
		s.Prefill([]int{1, 2})
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-5)
		if b.norms != 0 || b.readbacks != 0 {
			t.Fatal("reduced profile selected a full-profile KV norm")
		}
	})
	for _, tc := range []struct {
		name    string
		role    bool
		fault   error
		badRead bool
	}{
		{"selected-vulkan", false, &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}, false},
		{"selected-cuda-role", true, &compute.CUDAOpError{Op: "RMSNorm", Err: ErrV41ForwardStage}, false},
		{"invalid-readback", false, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var m *Model
			if tc.role {
				m = v41IncrementalExpertFixture(t, true, false)
			} else {
				m = v41FullStepPatchedNorms(t)
			}
			s, b := newSession(t, m)
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.live = map[compute.Buffer]bool{}
			b.fault, b.badRead = tc.fault, tc.badRead
			err := recoverError(func() { s.Step(3) })
			var selected *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			stage := "rmsnorm"
			if tc.badRead {
				stage = "readback"
			}
			if !errors.As(err, &selected) || selected.Leaf != leaf || selected.Stage != string(v41StageAttention) || !errors.As(err, &closed) || closed.Path != "v41-kv-norm" || closed.Layer != 0 || closed.Stage != stage {
				t.Fatalf("selected KV-norm failure lost identity: %v", err)
			}
			if tc.fault != nil && !errors.Is(err, tc.fault) {
				t.Fatalf("KV-norm failure lost its cause: %v", err)
			}
			if b.norms != 3 || len(b.ops) != b.opsAtFault || !s.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 {
				t.Fatalf("KV failure retried, projected, or leaked: norms=%d ops=%d/%d closed=%t live=%d outputs=%d", b.norms, len(b.ops), b.opsAtFault, s.BackendSessionClosed(), len(b.live), len(b.outputs))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("failed KV norm advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 {
				t.Fatalf("closed session retried KV norm: norms=%d err=%v", b.norms, retry)
			}
		})
	}
}
