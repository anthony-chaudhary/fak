package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Reuse the query norm recorder with the FFN gain as its selected identity. It
// counts only this norm's activation uploads/readbacks and poisons its output on Free.
// cpu-ref arithmetic establishes dispatch and lifetime semantics, not physical
// device qualification or a throughput gain.
// fak-test:runtime medium est=5s lane=default
func TestV41FFNNormDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("FFN-norm dispatch fixture requires cpu-ref")
	}
	const leaf = "ffn_norm.weight"
	newSession := func(t *testing.T, m *Model) (*Session, *v41QueryNormTestBackend) {
		b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, leaf)] }
		full, err := v41ForwardGeometry(m.Cfg)
		if err != nil || (s.v41State().ffnNorm != nil) != full {
			t.Fatalf("FFN callback selection disagrees with full geometry %t: %v", full, err)
		}
		return s, b
	}
	for _, tc := range []struct {
		name             string
		role, tokenMajor bool
	}{
		{"plain-grouped", false, false},
		{"plain-token-major", false, true},
		{"compressed-role", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prior := v41ForceTokenMajor
			v41ForceTokenMajor = tc.tokenMajor
			defer func() { v41ForceTokenMajor = prior }()
			var m *Model
			if tc.role {
				m = v41IncrementalExpertFixture(t, true, false)
			} else {
				m = v41FullStepPatchedNorms(t)
			}
			gain := make([]float32, m.Cfg.HiddenSize)
			for i := range gain {
				gain[i] = 0.5 + float32(i)/float32(len(gain))
			}
			v41WriteTensorF32(t, m, layerName(0, leaf), gain)
			s, b := newSession(t, m)
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() || b.norms != 2 || b.readbacks != 2 {
				t.Fatalf("prefill FFN calls/readbacks=%d/%d incremental=%t", b.norms, b.readbacks, s.v41IncrementalEligible())
			}
			weight := s.halW[layerName(0, leaf)]
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			if b.norms != 3 || b.readbacks != 3 || len(b.outputs) != 0 || b.uploadBytes != 3*4*m.Cfg.HiddenSize || b.readBytes != b.uploadBytes {
				t.Fatalf("FFN dispatch/transfer/cleanup mismatch: norms=%d reads=%d outputs=%d upload=%d read=%d", b.norms, b.readbacks, len(b.outputs), b.uploadBytes, b.readBytes)
			}
			if weight.Buf() == nil || s.halW[layerName(0, leaf)].Buf() != weight.Buf() || !reflect.DeepEqual(b.gain, m.tensor(layerName(0, leaf))) {
				t.Fatal("FFN norm did not reuse its cached learned gain")
			}
			// A direct oracle catches a wrong norm even if later projections mask
			// it. Free poisoning detects borrowed, rather than owned, readback.
			input := append([]float32(nil), b.input...)
			got, err := s.v41State().ffnNorm(0, input)
			if err != nil {
				t.Fatal(err)
			}
			v41GroupedParity(t, got, cpuOracleRMSNorm(input, b.gain, float32(m.Cfg.RMSNormEps)), 1e-5)
			if len(input) != m.Cfg.HiddenSize || !reflect.DeepEqual(input, b.input) {
				t.Fatal("FFN norm changed its host input or hidden width")
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
			if target.v41Forward.callbackOwner != target || target.v41Forward.ffnNorm == nil || b.norms != beforeNorms+1 || target.halW[layerName(0, leaf)].Buf() == nil {
				t.Fatal("restored FFN norm did not bind and execute on the target session")
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
			t.Fatal("reduced profile selected a full-profile FFN norm")
		}
	})
	t.Run("eligibility", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		var absent *Session
		if absent.v41FFNNormFunc() != nil || (&Session{M: m}).v41FFNNormFunc() != nil ||
			(&Session{M: m, Backend: compute.Default()}).v41FFNNormFunc() != nil {
			t.Fatal("portable or absent session selected device FFN normalization")
		}
		original := m.Cfg
		defer func() { m.Cfg = original }()
		for _, tc := range []struct {
			name string
			edit func(*Config, *v41DenseTestBackend)
		}{
			{"layernorm", func(c *Config, _ *v41DenseTestBackend) { c.LayerNorm = true }},
			{"gain-plus-one", func(c *Config, _ *v41DenseTestBackend) { c.NormGain1p = true }},
			{"zero-epsilon", func(c *Config, _ *v41DenseTestBackend) { c.RMSNormEps = 0 }},
			{"nonfinite-epsilon", func(c *Config, _ *v41DenseTestBackend) { c.RMSNormEps = math.NaN() }},
			{"unsupported-dtype", func(_ *Config, b *v41DenseTestBackend) { b.deny = true }},
		} {
			m.Cfg = original
			b := newV41DenseTestBackend()
			tc.edit(&m.Cfg, b)
			if (&Session{M: m, Backend: b}).v41FFNNormFunc() != nil {
				t.Fatalf("ineligible %s selected device FFN normalization", tc.name)
			}
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
			if !errors.As(err, &selected) || selected.Leaf != leaf || selected.Stage != string(v41StageMoE) || !errors.As(err, &closed) || closed.Path != "v41-ffn-norm" || closed.Layer != 0 || closed.Stage != stage {
				t.Fatalf("selected FFN-norm failure lost identity: %v", err)
			}
			if tc.fault != nil && !errors.Is(err, tc.fault) {
				t.Fatalf("FFN-norm failure lost its cause: %v", err)
			}
			if b.norms != 3 || len(b.ops) != b.opsAtFault || !s.BackendSessionClosed() || len(b.live) != 0 || len(b.outputs) != 0 {
				t.Fatalf("FFN failure retried, projected, or leaked: norms=%d ops=%d/%d closed=%t live=%d outputs=%d", b.norms, len(b.ops), b.opsAtFault, s.BackendSessionClosed(), len(b.live), len(b.outputs))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("failed FFN norm advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.norms != 3 {
				t.Fatalf("closed session retried FFN norm: norms=%d err=%v", b.norms, retry)
			}
		})
	}
}

// The callback remains F32. Its owner supplies the BF16 collapse of attention's
// updated residuals, then owns and rounds the callback's result before MoE.
// fak-test:runtime fast est=100ms lane=default
func TestV41FFNNormGraphOwner(t *testing.T) {
	t.Parallel()
	m := v41FullStepModel(t)
	H := m.Cfg.HiddenSize
	streams := make([][]float32, 4)
	attn := make([]float32, H)
	for h := range streams {
		streams[h] = make([]float32, H)
		for d := range streams[h] {
			streams[h][d] = float32((d+3*h)%13-6) / 8
		}
	}
	for d := range attn {
		attn[d] = float32(d%7-3)/16 + .003
	}
	mix := v41MHCMix{pre: []float32{.125, .25, .625, .875}, post: []float32{.5, .75, 1.25, 1.5}, comb: []float32{.1, .2, .3, .4, .4, .3, .2, .1, .25, .25, .25, .25, .3, .1, .4, .2}}
	before := sysFlatten(streams)
	wantResidual := v41GraphOraclePost(attn, streams, mix.post, mix.comb)
	wantInput := v41GraphOracleCollapse(wantResidual, mix.pre)
	gain := cpuOracleTensor(t, m, layerName(0, "ffn_norm.weight"))
	var callbackInput, callbackOutput []float32
	calls := 0
	scratch := &v41ProjScratch{ffnNorm: func(layer int, input []float32) ([]float32, error) {
		calls++
		callbackInput = append([]float32(nil), input...)
		callbackOutput = cpuOracleRMSNorm(input, gain, float32(m.Cfg.RMSNormEps))
		return callbackOutput, nil
	}}
	projected := false
	residual, _, got, err := m.v41FullFFNInput(0, streams, mix, attn, func(flat []float32) ([]float32, error) {
		projected = true
		if !reflect.DeepEqual(flat, sysFlatten(wantResidual)) {
			t.Fatal("FFN mHC did not project attention-post streams")
		}
		return make([]float32, 24), nil
	}, scratch)
	if err != nil || !projected || calls != 1 || !reflect.DeepEqual(residual, wantResidual) || !reflect.DeepEqual(callbackInput, wantInput) {
		t.Fatalf("FFN graph operand/order/callback: calls=%d err=%v", calls, err)
	}
	if !reflect.DeepEqual(got, v41LatentNormOracleBF16(callbackOutput)) || !reflect.DeepEqual(before, sysFlatten(streams)) {
		t.Fatal("FFN owner omitted BF16 publication or changed incoming residuals")
	}
	callbackOutput[0] = 99
	if got[0] == 99 {
		t.Fatal("FFN owner retained borrowed callback output")
	}
	if reflect.DeepEqual(callbackInput, streams[0]) {
		t.Fatal("FFN input discriminator is vacuous")
	}
}
