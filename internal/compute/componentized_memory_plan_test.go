package compute

import (
	"testing"
)

func TestComponentizedMemoryBreakdownWitness(t *testing.T) {
	// First witness requirements (#9921):
	// 1. Emits machine-readable componentized memory receipt (weights, activations, signal_buffers, kv_cache, scratchpad).
	// 2. Proves each component can INDEPENDENTLY push fit from ALLOW to REFUSE.
	// 3. Reports culpable component in refusal receipt.

	capacity := int64(100 * 1024 * 1024) // 100 MiB
	headroom := 0.10                     // 10% -> 90 MiB effective budget

	// Baseline ALLOW setup: each component is 10 MiB -> total 50 MiB <= 90 MiB
	base := ComponentizedMemoryBreakdown{
		WeightsBytes:       10 * 1024 * 1024,
		ActivationsBytes:   10 * 1024 * 1024,
		SignalBuffersBytes: 10 * 1024 * 1024,
		KVCacheBytes:       10 * 1024 * 1024,
		ScratchpadBytes:    10 * 1024 * 1024,
	}

	receipt, err := EvaluateComponentizedMemoryFit(base, capacity, headroom)
	if err != nil {
		t.Fatalf("EvaluateComponentizedMemoryFit failed: %v", err)
	}
	if !receipt.Allowed {
		t.Fatalf("expected ALLOW for baseline, got refusal: %s", receipt.RefusalReason)
	}

	// 2. Verify EACH component independently pushes fit from ALLOW to REFUSE
	tests := []struct {
		name        string
		modifier    func(ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown
		wantCulprit string
	}{
		{
			name: "weights_overflow",
			modifier: func(b ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown {
				b.WeightsBytes = 60 * 1024 * 1024 // 60 + 40 = 100 > 90
				return b
			},
			wantCulprit: "weights",
		},
		{
			name: "activations_overflow",
			modifier: func(b ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown {
				b.ActivationsBytes = 60 * 1024 * 1024
				return b
			},
			wantCulprit: "activations",
		},
		{
			name: "signal_buffers_overflow",
			modifier: func(b ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown {
				b.SignalBuffersBytes = 60 * 1024 * 1024
				return b
			},
			wantCulprit: "signal_buffers",
		},
		{
			name: "kv_cache_overflow",
			modifier: func(b ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown {
				b.KVCacheBytes = 60 * 1024 * 1024
				return b
			},
			wantCulprit: "kv_cache",
		},
		{
			name: "scratchpad_overflow",
			modifier: func(b ComponentizedMemoryBreakdown) ComponentizedMemoryBreakdown {
				b.ScratchpadBytes = 60 * 1024 * 1024
				return b
			},
			wantCulprit: "scratchpad",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := tc.modifier(base)
			rec, err := EvaluateComponentizedMemoryFit(b, capacity, headroom)
			if err != nil {
				t.Fatalf("evaluation failed: %v", err)
			}
			if rec.Allowed {
				t.Fatalf("expected REFUSE for %s, got ALLOW", tc.name)
			}
			if rec.ViolatingComponent != tc.wantCulprit {
				t.Fatalf("expected culprit %q, got %q", tc.wantCulprit, rec.ViolatingComponent)
			}
			if rec.ExcessBytes <= 0 {
				t.Fatalf("expected positive excess bytes, got %d", rec.ExcessBytes)
			}
			if rec.RefusalReason == "" {
				t.Fatal("expected non-empty refusal reason")
			}
		})
	}
}
