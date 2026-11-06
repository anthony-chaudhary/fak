package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type qwen35VLLMFusedQKStore struct {
	compute.KVStore
	decline, empty bool
	err            error
	calls          int
	eps            float32
	gain           []float32
}

func (k *qwen35VLLMFusedQKStore) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}

func (k *qwen35VLLMFusedQKStore) AppendQKNormRoPE(q, key, value, qw, kw compute.Tensor, layer, pos, heads, rotary int, eps float32, theta float64) (compute.Tensor, bool, error) {
	k.calls++
	k.eps = eps
	k.gain = append([]float32(nil), compute.Default().Read(qw)...)
	if k.err != nil {
		return compute.Tensor{}, true, k.err
	}
	if k.decline {
		return compute.Tensor{}, false, nil
	}
	if k.empty {
		return compute.Tensor{}, true, nil
	}
	// This fixture witnesses routing, not CUDA arithmetic. Publish exactly once
	// so a mistakenly repeated ordinary append remains observable.
	k.KVStore.AppendKV(layer, key, key, value, pos)
	return q, true, nil
}

// fak-test:runtime fast est=30ms lane=default
func TestQwen35VLLMFusedQKOrdinaryRoute(t *testing.T) {
	for _, tc := range []struct {
		name           string
		decline, empty bool
		failure        error
	}{
		{name: "selected"},
		{name: "unsupported", decline: true},
		{name: "selected-error", failure: errors.New("selected kernel failure")},
		{name: "invalid-result", empty: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, be, _, _, layer := newQwen35QKNormOutputSession(t, 0)
			s.M.Cfg.QKNormEps = 3e-4
			store := &qwen35VLLMFusedQKStore{KVStore: s.halKV, decline: tc.decline, empty: tc.empty, err: tc.failure}
			s.halKV = store
			cfg := s.M.Cfg
			x := be.Upload(compute.NewF32(compute.Default(), []int{cfg.HiddenSize}, make([]float32, cfg.HiddenSize)), compute.F32)
			var raised any
			func() {
				defer func() { raised = recover() }()
				s.qwen35FullAttentionHAL(layer, 0, x, float32(cfg.RMSNormEps), cfg.attnScale(), cfg.GroupSize())
			}()
			if store.calls != 1 {
				t.Fatalf("fused calls=%d, want 1 from the ordinary full-attention path", store.calls)
			}
			if store.eps != float32(cfg.QKNormEps) || len(store.gain) != cfg.HeadDim {
				t.Fatalf("effective RMS contract: epsilon=%g gain=%v", store.eps, store.gain)
			}
			for _, gain := range store.gain {
				if gain != 1 {
					t.Fatalf("NormGain1p was not applied exactly once: %g", gain)
				}
			}
			wantNorms := 1
			if tc.decline {
				wantNorms = 3
			}
			if be.calls != wantNorms {
				t.Fatalf("RMSNorm calls=%d, want %d; selected operation must not replay", be.calls, wantNorms)
			}
			if tc.failure != nil || tc.empty {
				if raised == nil || !s.halClosed {
					t.Fatalf("selected failure did not close the forward: panic=%v closed=%v", raised, s.halClosed)
				}
			} else if raised != nil || store.Len() != 1 {
				t.Fatalf("ordinary publication: panic=%v length=%d, want nil/1", raised, store.Len())
			}
		})
	}
}
