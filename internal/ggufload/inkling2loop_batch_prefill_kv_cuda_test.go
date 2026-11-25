//go:build cuda && cgo

package ggufload

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// inkling2KVArm sets one cohort cache arm: fp32 caches with cohort prefill, f16 caches
// with cohort prefill, or f16 caches with lockstep prompts.
func inkling2KVArm(t *testing.T, arm string) {
	t.Helper()
	switch arm {
	case "f32cohort":
		t.Setenv("FAK_INKLING2_BATCH_KV", "f32")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL_KV", "")
	case "f16cohort":
		t.Setenv("FAK_INKLING2_BATCH_KV", "f16")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL_KV", "f16")
	case "f16lock":
		t.Setenv("FAK_INKLING2_BATCH_KV", "f16")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
		t.Setenv("FAK_INKLING2_COHORT_PREFILL_KV", "")
	default:
		t.Fatalf("unknown cache arm %q", arm)
	}
}

func inkling2BatchCtxFor(prompts [][]int, n int) int {
	ctx := 0
	for _, pr := range prompts {
		ctx = max(ctx, len(pr)+n)
	}
	return (ctx + inkling2BatchCtxAlign - 1) / inkling2BatchCtxAlign * inkling2BatchCtxAlign
}

// A cohort prefilled over f16 caches gives every row the greedy tokens the scalar
// resident session gives its prompt, up to an admissible near tie, at eight rows beside
// the fp32 cohort and at sixteen pp512 rows, which fit only with f16 caches. Rows that
// differ from the fp32 cohort are logged: f16 decode reads round where fp32 ones do not.
func TestInkling2CUDACohortPrefillF16MatchesF32(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	const n = 16
	var prompts [][]int
	for _, c := range inkling2OracleCases {
		if len(c.prompt) < 256 {
			prompts = append(prompts, c.prompt)
		}
	}
	for len(prompts) < 8 {
		prompts = append(prompts, inkling2PrefillPrompt(40+7*len(prompts)))
	}
	prompts = prompts[:8]
	wide := make([][]int, 16)
	for i := range wide {
		pr := inkling2PrefillPrompt(512 + i)
		wide[i] = pr[i:]
	}
	run := func(arm string, prompts [][]int) [][]int {
		inkling2KVArm(t, arm)
		got, st, err := inkling2BatchGreedy(m, be, prompts, n, inkling2BatchCtxFor(prompts, n))
		var refused *Inkling2BatchMemoryError
		if chunk := inkling2PrefillChunkWanted(os.LookupEnv); errors.As(err, &refused) && len(prompts) > 8 && chunk > 256 {
			// sixteen pp512 rows fit beside the 512-token prefill arenas only with a smaller chunk
			t.Logf("%s C=%d refused at prefill chunk %d (rerun with FAK_INKLING2_PREFILL_CHUNK=256): %v", arm, len(prompts), chunk, err)
			return nil
		}
		if err != nil {
			t.Fatalf("%s C=%d: %v", arm, len(prompts), err)
		}
		t.Logf("%s C=%d prompt %s; %d decode steps %s; state %.1f MiB", arm, st.Sequences, st.Prompt, st.DecodeSteps, st.Decode,
			float64(st.StateBytes)/(1<<20))
		return got
	}
	check := func(label string, prompts, got [][]int, ref [][]int) {
		ties := 0
		for i, prompt := range prompts {
			want, runnerUp, margin, err := inkling2ResidentGreedyMargins(m, be, prompt, n)
			if err != nil {
				t.Fatal(err)
			}
			at, nearTie, why := inkling2CohortDivergence(want, runnerUp, margin, got[i])
			switch {
			case at < 0:
				t.Logf("COHORT_VERDICT %s row %d PASS (%d prompt tokens)", label, i, len(prompt))
			case nearTie:
				ties++
				t.Logf("COHORT_VERDICT %s row %d NEAR_TIE (%d prompt tokens) %s", label, i, len(prompt), why)
			default:
				t.Errorf("COHORT_VERDICT %s row %d FAIL (%d prompt tokens): %s", label, i, len(prompt), why)
			}
			if ref == nil {
				continue
			}
			for j := range ref[i] {
				if got[i][j] != ref[i][j] {
					t.Logf("%s row %d: f16 cohort differs from the fp32 cohort at token %d (%d vs %d)", label, i, j, got[i][j], ref[i][j])
					break
				}
			}
		}
		if ties > inkling2OracleMaxNearTies {
			t.Errorf("%s: %d of %d rows are near ties, more than %d", label, ties, len(prompts), inkling2OracleMaxNearTies)
		}
	}
	f32 := run("f32cohort", prompts)
	f16 := run("f16cohort", prompts)
	check("C8", prompts, f16, f32)
	if got := run("f16cohort", wide); got != nil {
		check("C16pp512", wide, got, nil)
	}
}

// FAK_INKLING2_KVBENCH=8,16 times the cache arms of FAK_INKLING2_KVBENCH_ARMS (default
// f16cohort,f16lock,f32cohort) at each C over FAK_INKLING2_KVBENCH_PP prompt tokens
// per row (default 512) and FAK_INKLING2_KVBENCH_N greedy tokens (default 32), with
// identical prompts, FAK_INKLING2_KVBENCH_ROUNDS ABBA rounds (default 2). Each run logs
// one KVBENCH line; a refused arm logs its memory error.
func TestInkling2CUDACohortPrefillKVBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_KVBENCH")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	envInt := func(k string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
			return v
		}
		return def
	}
	pp, n, rounds := envInt("FAK_INKLING2_KVBENCH_PP", 512), envInt("FAK_INKLING2_KVBENCH_N", 32), envInt("FAK_INKLING2_KVBENCH_ROUNDS", 2)
	arms := strings.Split(os.Getenv("FAK_INKLING2_KVBENCH_ARMS"), ",")
	if os.Getenv("FAK_INKLING2_KVBENCH_ARMS") == "" {
		arms = []string{"f16cohort", "f16lock", "f32cohort"}
	}
	for _, f := range strings.Split(os.Getenv("FAK_INKLING2_KVBENCH"), ",") {
		c, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || c < 1 {
			t.Fatalf("bad cohort width %q", f)
		}
		prompts := make([][]int, c)
		for i := range prompts {
			prompts[i] = inkling2PrefillPrompt(pp + i)[i:]
		}
		ctx := inkling2BatchCtxFor(prompts, n)
		var order []string
		for r := 0; r < rounds; r++ {
			if r%2 == 0 {
				order = append(order, arms...)
			} else {
				for i := len(arms) - 1; i >= 0; i-- {
					order = append(order, arms[i])
				}
			}
		}
		for k, arm := range order {
			inkling2KVArm(t, strings.TrimSpace(arm))
			start := time.Now()
			got, st, err := inkling2BatchGreedy(m, be, prompts, n, ctx)
			wall := time.Since(start)
			var refused *Inkling2BatchMemoryError
			if errors.As(err, &refused) {
				t.Logf("KVBENCH arm=%s C=%d pp=%d run=%d refused: %v", arm, c, pp, k, err)
				continue
			}
			if err != nil {
				t.Fatalf("%s C=%d: %v", arm, c, err)
			}
			toks := 0
			for _, row := range got {
				toks += len(row)
			}
			t.Logf("KVBENCH arm=%s C=%d pp=%d n=%d run=%d prompt_s=%.4f decode_s=%.4f wall_s=%.4f prompt_tok_s=%.1f e2e_tok_s=%.2f decode_tok_s=%.2f need_mib=%.1f state_mib=%.1f free_after_mib=%.1f first=%v",
				arm, c, pp, n, k, st.Prompt.Seconds(), st.Decode.Seconds(), wall.Seconds(), float64(c*pp)/st.Prompt.Seconds(),
				float64(toks)/wall.Seconds(), st.AggregateTokensPerSecond(), float64(st.NeedBytes)/(1<<20), float64(st.StateBytes)/(1<<20),
				float64(st.FreeAfter)/(1<<20), got[0][:min(4, len(got[0]))])
		}
	}
}
