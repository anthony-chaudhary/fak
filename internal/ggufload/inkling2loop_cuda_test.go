//go:build cuda && cgo

package ggufload

import (
	"fmt"
	"hash/fnv"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func openInkling2CUDAForTest(t *testing.T, env string) (*Inkling2LoopModel, compute.Backend) {
	t.Helper()
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" || (env != "" && os.Getenv(env) == "") {
		t.Skipf("FAK_INKLING2_GGUF%s not set", map[bool]string{true: " / " + env}[env != ""])
	}
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		t.Fatal("FAK_INKLING2_GGUF set but the cuda backend is not registered")
	}
	start := time.Now()
	m, err := OpenInkling2LoopCUDA(path, be)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("cuda load %s on %s (resident forward %v)", time.Since(start), be.Tier(), inkling2ResidentEligible(m, be))
	return m, be
}

func TestInkling2CUDAOracleGreedyMatchesLlamaCpp(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	for _, c := range inkling2OracleCases {
		t.Run(c.name, func(t *testing.T) {
			runInkling2Oracle(t, c, func(prompt []int, n int) ([]int, error) {
				got, st, err := Inkling2LoopGreedyTimed(m, be, prompt, n)
				if err == nil {
					t.Logf("cuda: %d prompt tokens prefill %s; %d decode steps %s = %.2f tok/s",
						len(prompt), st.Prefill, st.DecodeSteps, st.Decode, st.DecodeTokensPerSecond())
				}
				return got, err
			})
		})
	}
}

// FAK_INKLING2_BENCH=N decodes N (default 128) greedy tokens after the
// capitals prompt and logs decode tok/s, prefill excluded.
// FAK_INKLING2_BENCH_DEPTH=D (at most 1946) prefills the first D tokens of the
// passphrase-recall prompt instead, so decode runs at context depth D.
func TestInkling2CUDADecodeBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_BENCH")
	defer m.Close()
	n := 128
	if v, err := strconv.Atoi(os.Getenv("FAK_INKLING2_BENCH")); err == nil && v > 1 {
		n = v
	}
	prompt := inkling2OracleCases[0].prompt
	if d, err := strconv.Atoi(os.Getenv("FAK_INKLING2_BENCH_DEPTH")); err == nil && d > 0 {
		long := inkling2OracleCases[2].prompt
		prompt = long[:min(d, len(long))]
	}
	got, st, err := Inkling2LoopGreedyTimed(m, be, prompt, n)
	if err != nil {
		t.Fatal(err)
	}
	// The fnv64 of every generated id lets two arms (say FAK_INKLING2_PDL=0 and =1) be
	// compared over the whole decode, not just the printed prefix.
	h := fnv.New64a()
	for _, id := range got {
		_, _ = fmt.Fprintf(h, "%d,", id)
	}
	t.Logf("inkling2 cuda bench: prompt %d prefill %s (%.2f tok/s); decode %d steps %s = %.2f tok/s; first tokens %v; tokens %d fnv64 %016x",
		len(prompt), st.Prefill, float64(len(prompt))/st.Prefill.Seconds(), st.DecodeSteps, st.Decode,
		st.DecodeTokensPerSecond(), got[:min(8, len(got))], len(got), h.Sum64())
	if d := inkling2SpecDepth(); d > 0 {
		sp := inkling2LastSpec
		t.Logf("inkling2 cuda bench spec: depth %d, %d verifies, accepted %d/%d (%.2f tok/verify), draft %.3f ms/call, verify %.3f ms/call",
			d, sp.Verifies, sp.Accepted, sp.Drafted, float64(sp.Accepted+sp.Verifies)/float64(max(1, sp.Verifies)),
			ms(sp.Draft)/float64(max(1, sp.Verifies)), ms(sp.Verify)/float64(max(1, sp.Verifies)))
	}
}
