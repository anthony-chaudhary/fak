package modelengine

import (
	"errors"
	"testing"
)

// TestSwaBoundedReplayGate pins the closed contract of the host-side SWA
// bounded-replay admission gate: each refusal branch matches exactly its own
// sentinel via errors.Is, and the all-clear configuration is admitted. The
// assertions bind sentinels, not error prose.
func TestSwaBoundedReplayGate(t *testing.T) {
	allClear := SwaBoundedReplayConfig{}

	t.Run("all-clear-admitted", func(t *testing.T) {
		if err := SwaBoundedReplayGate(allClear); err != nil {
			t.Fatalf("SwaBoundedReplayGate(allClear) = %v, want nil", err)
		}
	})

	refusals := []struct {
		name string
		cfg  SwaBoundedReplayConfig
		want error
	}{
		{
			name: "refuses-prompt-logprobs",
			cfg:  SwaBoundedReplayConfig{PromptLogprobs: true},
			want: ErrSwaReplayPromptLogprobs,
		},
		{
			name: "refuses-prefill-cuda-graph",
			cfg:  SwaBoundedReplayConfig{PrefillCudaGraph: true},
			want: ErrSwaReplayPrefillCudaGraph,
		},
		{
			name: "refuses-dp-attention",
			cfg:  SwaBoundedReplayConfig{DPAttention: true},
			want: ErrSwaReplayDPAttention,
		},
	}
	for _, tc := range refusals {
		t.Run(tc.name, func(t *testing.T) {
			err := SwaBoundedReplayGate(tc.cfg)
			if !errors.Is(err, tc.want) {
				t.Fatalf("SwaBoundedReplayGate(%+v) = %v, want errors.Is %v", tc.cfg, err, tc.want)
			}
			for _, other := range []error{ErrSwaReplayPromptLogprobs, ErrSwaReplayPrefillCudaGraph, ErrSwaReplayDPAttention} {
				if other == tc.want {
					continue
				}
				if errors.Is(err, other) {
					t.Fatalf("SwaBoundedReplayGate(%+v) also matches %v, want only %v", tc.cfg, other, tc.want)
				}
			}
		})
	}

	t.Run("reports-every-violation", func(t *testing.T) {
		err := SwaBoundedReplayGate(SwaBoundedReplayConfig{
			PromptLogprobs:   true,
			PrefillCudaGraph: true,
			DPAttention:      true,
		})
		for _, want := range []error{ErrSwaReplayPromptLogprobs, ErrSwaReplayPrefillCudaGraph, ErrSwaReplayDPAttention} {
			if !errors.Is(err, want) {
				t.Fatalf("SwaBoundedReplayGate(all set) = %v, want errors.Is %v", err, want)
			}
		}
	})
}
