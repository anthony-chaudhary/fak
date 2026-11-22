//go:build cuda && cgo

package ggufload

import (
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
	t.Logf("cuda load %s on %s", time.Since(start), be.Tier())
	return m, be
}

func TestInkling2CUDAOracleGreedyMatchesLlamaCpp(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	for _, c := range inkling2OracleCases {
		t.Run(c.name, func(t *testing.T) {
			got, st, err := Inkling2LoopGreedyTimed(m, be, c.prompt, len(c.greedy))
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("cuda: %d prompt tokens prefill %s; %d decode steps %s = %.2f tok/s",
				len(c.prompt), st.Prefill, st.DecodeSteps, st.Decode, st.DecodeTokensPerSecond())
			for i := range got {
				if got[i] != c.greedy[i] {
					t.Fatalf("greedy token %d = %d, llama.cpp = %d (got %v)", i, got[i], c.greedy[i], got)
				}
			}
		})
	}
}

// FAK_INKLING2_BENCH=N decodes N (default 128) greedy tokens after the
// capitals prompt and logs decode tok/s, prefill excluded.
func TestInkling2CUDADecodeBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_BENCH")
	defer m.Close()
	n := 128
	if v, err := strconv.Atoi(os.Getenv("FAK_INKLING2_BENCH")); err == nil && v > 1 {
		n = v
	}
	prompt := inkling2OracleCases[0].prompt
	got, st, err := Inkling2LoopGreedyTimed(m, be, prompt, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("inkling2 cuda bench: prompt %d prefill %s (%.2f tok/s); decode %d steps %s = %.2f tok/s; first tokens %v",
		len(prompt), st.Prefill, float64(len(prompt))/st.Prefill.Seconds(), st.DecodeSteps, st.Decode,
		st.DecodeTokensPerSecond(), got[:min(8, len(got))])
}
