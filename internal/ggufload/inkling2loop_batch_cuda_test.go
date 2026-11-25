//go:build cuda && cgo

package ggufload

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

// The batched decode gives every sequence the greedy continuation the llama.cpp
// oracle gives its prompt alone, with sequences of different prompt lengths (so
// different positions every step) sharing each step, and two copies of one prompt
// agreeing token for token. The 1946-token passphrase case is left to the
// single-sequence oracle test: its cache would not fit beside the others' on 16 GB.
func TestInkling2CUDABatchDecodeMatchesOracle(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	var cases = inkling2OracleCases[:0:0]
	for _, c := range inkling2OracleCases {
		if len(c.prompt) < 256 {
			cases = append(cases, c)
		}
	}
	cases = append(cases, cases[0])
	prompts := make([][]int, len(cases))
	n := 0
	for i, c := range cases {
		prompts[i] = c.prompt
		n = max(n, len(c.greedy))
	}
	got, st, err := inkling2BatchGreedy(m, be, prompts, n, 512)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("C=%d state %.1f MiB; prompt %s; %d decode steps %s = %.2f aggregate tok/s", st.Sequences,
		float64(st.StateBytes)/(1<<20), st.Prompt, st.DecodeSteps, st.Decode, st.AggregateTokensPerSecond())
	for i, c := range cases {
		want := c.greedy
		for j := range want {
			if got[i][j] != want[j] {
				t.Errorf("sequence %d (%s): greedy token %d = %d, llama.cpp = %d (got %v)", i, c.name, j, got[i][j], want[j], got[i][:len(c.greedy)])
				break
			}
		}
	}
	last := len(cases) - 1
	for j := range got[0] {
		if got[last][j] != got[0][j] {
			t.Fatalf("two copies of one prompt diverge at token %d: %v vs %v", j, got[0], got[last])
		}
	}
}

// FAK_INKLING2_BATCH=1,2,4,8,12,16 decodes FAK_INKLING2_BENCH (default 64) greedy
// tokens after the capitals prompt in every one of C lockstep sequences, and logs
// the aggregate and per-sequence decode tok/s at each C.
func TestInkling2CUDABatchDecodeBench(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_BATCH")
	defer m.Close()
	n := 64
	if v, err := strconv.Atoi(os.Getenv("FAK_INKLING2_BENCH")); err == nil && v > 1 {
		n = v
	}
	ctx := inkling2BatchCtx
	if v, err := strconv.Atoi(os.Getenv("FAK_INKLING2_BATCH_CTX")); err == nil && v > 0 {
		ctx = v
	}
	prompt := inkling2OracleCases[0].prompt
	t.Logf("hparams %+v", m.HP)
	for _, f := range strings.Split(os.Getenv("FAK_INKLING2_BATCH"), ",") {
		c, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || c < 1 {
			t.Fatalf("bad batch size %q", f)
		}
		prompts := make([][]int, c)
		for i := range prompts {
			prompts[i] = prompt
		}
		got, st, err := inkling2BatchGreedy(m, be, prompts, n, ctx)
		var refused *Inkling2BatchMemoryError
		if errors.As(err, &refused) {
			t.Logf("inkling2 batch bench: C=%d refused: %v (device %s before the resident)", c, err, inkling2MiB(st.FreeBefore, st.DeviceTotal))
			continue
		}
		if err != nil {
			t.Fatalf("C=%d: %v", c, err)
		}
		t.Logf("inkling2 batch bench: C=%d memory: need %.1f MiB; device %s before the resident, %s after the batch",
			c, float64(st.NeedBytes)/(1<<20), inkling2MiB(st.FreeBefore, st.DeviceTotal), inkling2MiB(st.FreeAfter, st.DeviceTotal))
		same := true
		for i := range got {
			for j := range got[i] {
				same = same && got[i][j] == got[0][j]
			}
		}
		agg := st.AggregateTokensPerSecond()
		t.Logf("inkling2 batch bench: C=%d decode %d steps %s = %.2f aggregate tok/s, %.2f tok/s per sequence, %.3f ms/step; state %.1f MiB; prompt %s; rows identical %v; first tokens %v",
			c, st.DecodeSteps, st.Decode, agg, agg/float64(c), st.Decode.Seconds()*1000/float64(max(st.DecodeSteps, 1)),
			float64(st.StateBytes)/(1<<20), st.Prompt, same, got[0][:min(8, len(got[0]))])
	}
}

// inkling2MiB renders a free/total probe; an unprobed free prints as unknown.
func inkling2MiB(free, total int64) string {
	if free < 0 {
		return fmt.Sprintf("free unknown of %.1f MiB", float64(total)/(1<<20))
	}
	return fmt.Sprintf("%.1f of %.1f MiB free", float64(free)/(1<<20), float64(total)/(1<<20))
}
