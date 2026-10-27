package model

import (
	"math"
	"slices"
	"testing"
)

// Original channel/tap loop retained as the arithmetic and missing-history oracle.
func conv4Original(out, mixed, weights []float32, history [][]float32, rows, width int) {
	for t := 0; t < rows; t++ {
		outRow := out[t*width : (t+1)*width]
		for c := 0; c < width; c++ {
			var acc float32
			cb := c * 4
			for j := 0; j < 4; j++ {
				src := t + j - 3
				var row []float32
				if src >= 0 {
					row = mixed[src*width : (src+1)*width]
				} else {
					idx := len(history) + src
					if idx < 0 {
						continue
					}
					row = history[idx]
				}
				acc += weights[cb+j] * row[c]
			}
			outRow[c] = silu(acc)
		}
	}
}

func TestPrefillConv4KeepsScalarOrderAndSkippedTaps(t *testing.T) {
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
		for _, shape := range [][2]int{{1, 7}, {2, 7}, {3, 64}, {17, 64}, {257, 513}} {
			rows, width := shape[0], shape[1]
			for kept := 0; kept <= 3; kept++ {
				mixed, weights := make([]float32, rows*width), make([]float32, 4*width)
				for i := range mixed {
					mixed[i] = float32(i%29-14) / 13
				}
				for i := range weights {
					weights[i] = float32(i%23-11) / 17
				}
				// For cold t=0 these leading nonfinite weights must never be
				// multiplied by fabricated zero rows. Only tap3 actually exists.
				weights[0], weights[1], weights[2] = float32(math.Inf(1)), math.Float32frombits(0x7fc12345), float32(math.Inf(-1))
				mixed[len(mixed)-1] = math.Float32frombits(0x80000000)
				history := make([][]float32, kept)
				historyBefore := make([][]float32, kept)
				for i := range history {
					history[i] = make([]float32, width)
					for j := range history[i] {
						history[i][j] = float32(i+j%13-6) / 7
					}
					historyBefore[i] = slices.Clone(history[i])
				}
				before, weightsBefore := slices.Clone(mixed), slices.Clone(weights)
				storage, want := make([]float32, len(mixed)+2), make([]float32, len(mixed))
				storage[0], storage[len(storage)-1] = 123, -456
				got := storage[1 : len(storage)-1 : len(storage)-1]
				conv4Original(want, mixed, weights, history, rows, width)
				if !qwen35PrefillConv4(got, mixed, weights, history, rows, width, workers) {
					t.Fatal("valid immutable panel declined")
				}
				if !convHistoryBitsEqual(got, want) || !convHistoryBitsEqual(mixed, before) || !convHistoryBitsEqual(weights, weightsBefore) {
					t.Fatalf("changed math or inputs: workers%d shape%v history%d", workers, shape, kept)
				}
				if kept == 0 && math.IsNaN(float64(got[0])) {
					t.Fatal("evaluated a missing leading nonfinite tap")
				}
				for i := range history {
					if !convHistoryBitsEqual(history[i], historyBefore[i]) {
						t.Fatal("changed history")
					}
				}
				if storage[0] != 123 || storage[len(storage)-1] != -456 {
					t.Fatal("changed output canary")
				}
			}
		}
	}
}

func TestPrefillConv4DeclinesMalformedAndOutputAliases(t *testing.T) {
	const rows, width = 4, 7
	for _, name := range []string{"mixed alias", "weight alias", "history alias", "short weights", "short mixed", "short output", "short history", "excess history"} {
		mixed, weights, out := make([]float32, rows*width), make([]float32, 4*width), make([]float32, rows*width)
		history := [][]float32{make([]float32, width)}
		for i := range out {
			mixed[i], weights[i], out[i] = 11, 13, 17
		}
		switch name {
		case "mixed alias":
			out = mixed
		case "weight alias":
			out = weights
		case "history alias":
			history[0] = out[:width]
		case "short weights":
			weights = weights[:len(weights)-1]
		case "short mixed":
			mixed = mixed[:len(mixed)-1]
		case "short output":
			out = out[:len(out)-1]
		case "short history":
			history[0] = history[0][:width-1]
		case "excess history":
			history = append(history, make([]float32, width), make([]float32, width), make([]float32, width))
		}
		before := slices.Clone(out)
		if qwen35PrefillConv4(out, mixed, weights, history, rows, width, 4) || !convHistoryBitsEqual(out, before) {
			t.Fatalf("failed no-write decline: %s", name)
		}
	}
}

func TestPrefillConv4MalformedFallbackKeepsPartialPanic(t *testing.T) {
	run := func(useFast bool) (out []float32, panicked bool) {
		out = make([]float32, 2*7)
		mixed, weights := make([]float32, len(out)), make([]float32, 4*7-1)
		for i := range mixed {
			mixed[i] = 1
		}
		for i := range weights {
			weights[i] = 1
		}
		defer func() { panicked = recover() != nil }()
		if !useFast || !qwen35PrefillConv4(out, mixed, weights, nil, 2, 7, 1) {
			conv4Original(out, mixed, weights, nil, 2, 7)
		}
		return
	}
	got, gotPanic := run(true)
	want, wantPanic := run(false)
	if !gotPanic || gotPanic != wantPanic || !convHistoryBitsEqual(got, want) {
		t.Fatal("changed malformed convolution prefix writes/panic")
	}
}
