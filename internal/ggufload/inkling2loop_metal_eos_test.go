package ggufload

import (
	"slices"
	"testing"
)

// inkling2EndSamplingToken is Inkling-2's `<|content_model_end_sampling|>` EOS
// token ID (200006), which the resident Metal path previously ignored when
// running to max_tokens.
const inkling2EndSamplingToken = 200006

type fakeMetalBackendName struct{}

func (fakeMetalBackendName) Name() string { return "metal" }

// TestInkling2MetalResidentEOSStopAndPrefillDefaults is a cross-platform
// regression witness for the two Metal resident bugs recorded in the M3 Pro
// ratification:
//  1. Resident greedy decode must stop immediately at EOS 200006
//     (`<|content_model_end_sampling|>`) instead of running to max_tokens.
//  2. Metal resident prefill defaults to batched prefill with a 2048-token
//     chunk width when FAK_INKLING2_PREFILL_CHUNK and FAK_INKLING2_BATCHED_PREFILL
//     are unset.
func TestInkling2MetalResidentEOSStopAndPrefillDefaults(t *testing.T) {
	t.Run("stops at 200006 before max_tokens", func(t *testing.T) {
		script := []int{101, 202, 303, inkling2EndSamplingToken, 404, 505}
		s := &scriptedSession{script: script}
		got, st, err := inkling2GreedyStepsUntil(s, []int{1, 2, 3}, 256, inkling2EndSamplingToken)
		if err != nil {
			t.Fatal(err)
		}
		want := []int{101, 202, 303}
		if !slices.Equal(got, want) {
			t.Fatalf("ids = %v, want %v (must exclude EOS %d and stop early)", got, want, inkling2EndSamplingToken)
		}
		if st.DecodeSteps != 3 || s.calls != 4 {
			t.Fatalf("decode steps = %d over %d forwards, want 3 over 4", st.DecodeSteps, s.calls)
		}
	})

	t.Run("stops on prefill EOS 200006 with zero decode steps", func(t *testing.T) {
		s := &scriptedSession{script: []int{inkling2EndSamplingToken, 999}}
		got, st, err := inkling2GreedyStepsUntil(s, []int{1, 2}, 64, inkling2EndSamplingToken)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 || st.DecodeSteps != 0 || s.calls != 1 {
			t.Fatalf("got %v (%d steps, %d calls), want empty with 0 steps / 1 call", got, st.DecodeSteps, s.calls)
		}
	})

	t.Run("metal defaults to batched prefill at 2048-token chunk", func(t *testing.T) {
		t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "")
		if !inkling2BatchedPrefill() {
			t.Fatal("inkling2BatchedPrefill() = false when unset, want true")
		}
		chunk := inkling2PrefillChunkWantedOn(fakeMetalBackendName{}, func(string) (string, bool) { return "", false })
		if chunk != inkling2PrefillChunkMetal {
			t.Fatalf("metal default prefill chunk = %d, want %d", chunk, inkling2PrefillChunkMetal)
		}
		if inkling2PrefillChunkMetal < 1024 {
			t.Fatalf("inkling2PrefillChunkMetal = %d, want >= 1024", inkling2PrefillChunkMetal)
		}
		if got := inkling2PrefillChunkFor(8137, chunk); got != 2048 {
			t.Fatalf("chunk for 8137-token prompt = %d, want 2048", got)
		}
	})
}
