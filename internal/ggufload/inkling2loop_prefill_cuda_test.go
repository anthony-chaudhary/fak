//go:build cuda && cgo

package ggufload

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

// inkling2PrefillPrompt is n tokens of the passphrase oracle prompt, repeated past
// its end.
func inkling2PrefillPrompt(n int) []int {
	src := inkling2OracleCases[2].prompt
	out := make([]int, n)
	for i := range out {
		out[i] = src[i%len(src)]
	}
	return out
}

// FAK_INKLING2_PREFILL_BENCH=19,512,2048 logs prefill tok/s at each prompt length on
// the arm FAK_INKLING2_BATCHED_PREFILL selects (1 = batched, else the per-token step).
func TestInkling2CUDAPrefillBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFILL_BENCH")
	defer m.Close()
	arm := "batched"
	if !inkling2BatchedPrefill() {
		arm = "per-token"
	}
	for _, f := range strings.Split(os.Getenv("FAK_INKLING2_PREFILL_BENCH"), ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n < 1 {
			t.Fatalf("bad prompt length %q", f)
		}
		got, st, err := Inkling2LoopGreedyTimed(m, be, inkling2PrefillPrompt(n), 2)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("inkling2 prefill bench: arm %s pp%d prefill %s = %.2f tok/s; next %v",
			arm, n, st.Prefill, float64(n)/st.Prefill.Seconds(), got)
	}
}

// The batched prefill leaves the same caches as the per-token step: the greedy
// continuation of a prompt that spans several chunks and wraps the local window
// matches the per-token path. Each arm also proves the forward width it claims, so a
// wide chunk the device quietly refused cannot pass as a base-chunk run.
func TestInkling2CUDABatchedPrefillMatchesPerToken(t *testing.T) {
	// the chunk setting sizes the rings at bind, so each arm opens its own model
	for _, arm := range []struct {
		env   string
		chunk int
	}{{"", 512}, {"512", 512}, {"1024", 1024}, {"auto", 1024}} {
		t.Run("chunk="+arm.env, func(t *testing.T) {
			t.Setenv(inkling2PrefillChunkEnv, arm.env)
			m, be := openInkling2CUDAForTest(t, "")
			defer m.Close()
			for _, n := range []int{513, 1024, 1025, 1300} {
				prompt := inkling2PrefillPrompt(n)
				run := func(batched string, widest int) []int {
					t.Setenv("FAK_INKLING2_BATCHED_PREFILL", batched)
					got, st, err := Inkling2LoopGreedyTimed(m, be, prompt, 8)
					if err != nil {
						t.Fatal(err)
					}
					t.Logf("pp%d batched=%s chunk %d prefill %s: %v", n, batched, st.PrefillChunk, st.Prefill, got)
					if st.PrefillChunk != widest {
						t.Fatalf("pp%d batched=%s: widest prefill forward %d tokens, want %d", n, batched, st.PrefillChunk, widest)
					}
					return got
				}
				want, got := run("0", 0), run("1", min(n, arm.chunk))
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("pp%d greedy token %d: batched %d, per-token %d", n, i, got[i], want[i])
					}
				}
			}
		})
	}
}
