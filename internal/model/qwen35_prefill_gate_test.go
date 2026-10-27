package model

import (
	"math"
	"slices"
	"testing"
)

func qwen35PrefillGateSerial(out, gate []float32) {
	for i := range out {
		out[i] *= sigmoidf(gate[i])
	}
}

func TestQwen35PrefillGateParallelKeepsScalarBits(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	for _, workers := range []int{1, 4} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, shape := range [][2]int{{1, 7}, {17, 65}, {257, 256}} {
			rows, width := shape[0], shape[1]
			n := rows * width
			storage, gate := make([]float32, n+2), make([]float32, n)
			storage[0], storage[n+1] = 17, -19
			out := storage[1 : n+1 : n+1]
			for i := range out {
				out[i], gate[i] = float32(i%41-20)/13, float32(i%79-39)/7
			}
			for i, bits := range []uint32{0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345} {
				if i < n {
					out[i], gate[n-1-i] = math.Float32frombits(bits), math.Float32frombits(bits)
				}
			}
			before, want := slices.Clone(gate), slices.Clone(out)
			qwen35PrefillGateSerial(want, gate)
			qwen35PrefillGateDisjoint(out, gate, rows, width)
			for i := range out {
				if math.Float32bits(out[i]) != math.Float32bits(want[i]) || math.Float32bits(gate[i]) != math.Float32bits(before[i]) {
					t.Fatalf("workers%d shape%v element%d changed math or input", workers, shape, i)
				}
			}
			if storage[0] != 17 || storage[n+1] != -19 {
				t.Fatal("output canary changed")
			}
		}
	}
}

func TestQwen35PrefillGateMalformedKeepsSerialMutation(t *testing.T) {
	for _, shape := range [][4]int{{17, 11, 1, 17}, {13, 13, 2, 6}, {13, 13, -1, 13}, {13, 13, 13, 0}} {
		run := func(parallel bool) (out []float32, panicked bool) {
			out = make([]float32, shape[0])
			gate := make([]float32, shape[1])
			for i := range out {
				out[i] = float32(i + 1)
			}
			defer func() { panicked = recover() != nil }()
			if parallel {
				qwen35PrefillGateDisjoint(out, gate, shape[2], shape[3])
			} else {
				qwen35PrefillGateSerial(out, gate)
			}
			return
		}
		got, panicGot := run(true)
		want, panicWant := run(false)
		if panicGot != panicWant || !slices.Equal(got, want) {
			t.Fatalf("shape%v changed original partial mutation/panic", shape)
		}
	}
}

func TestQwen35HybridPrefillTokenLoopEscapeHatch(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	if !q8Qwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) ||
		!q4kQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) {
		t.Fatal("baseline Q8/Q4_K hybrid prefill gates unexpectedly declined")
	}
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "1")
	if q8Qwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) ||
		q4kQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) {
		t.Fatal("FAK_QWEN35_PREFILL_TOKEN_LOOP must force both Q8 and Q4_K token-loop paths")
	}
}
