package agent

import (
	"context"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeTopLogprobsSortedNormalizedAndTieOrdered(t *testing.T) {
	logits := []float32{1, 3, 3, -2, 0.5, 2}
	chosen, row, err := nativeTopLogprobs(logits, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got := []int{row[0].TokenID, row[1].TokenID, row[2].TokenID, row[3].TokenID}; !reflect.DeepEqual(got, []int{1, 2, 5, 0}) {
		t.Fatalf("top ids=%v, want [1 2 5 0] (logit desc, id asc)", got)
	}
	if chosen.TokenID != 1 || chosen.Logprob != row[0].Logprob {
		t.Fatalf("chosen=%+v, want row[0]=%+v", chosen, row[0])
	}
	var z float64
	for _, v := range logits {
		z += math.Exp(float64(v))
	}
	if want := 3 - math.Log(z); math.Abs(chosen.Logprob-want) > 1e-12 {
		t.Fatalf("chosen logprob=%v, want full-vocab log_softmax %v", chosen.Logprob, want)
	}
	for i, c := range row {
		if c.Logprob > 0 || (i > 0 && c.Logprob > row[i-1].Logprob) {
			t.Fatalf("row[%d]=%+v not non-positive and non-increasing: %+v", i, c, row)
		}
	}
	if _, _, err := nativeTopLogprobs([]float32{0, float32(math.NaN())}, 0, 2); err == nil {
		t.Fatal("non-finite logits accepted")
	}
}

func TestNativeDecodeTopLogprobsGreedyTop1IsChosenAndOutputUnchanged(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	ids := synthIDs(cfg.VocabSize, 12, 9070)
	const generated = 6
	p := &InKernelPlanner{m: m, modelID: "synthetic-top5", quant: false}

	run := func(m *nativeInferenceMeasurement) []int {
		t.Helper()
		var emitted []int
		_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
			context.Background(), ids, generated, 0, 0, 0, nil, 0, 0, map[int]bool{},
			func(tok int) bool { emitted = append(emitted, tok); return false }, m,
		)
		if err != nil {
			t.Fatal(err)
		}
		return emitted
	}
	off := run(nil)
	measurement := &nativeInferenceMeasurement{startedAt: time.Now(), inferenceDisabled: true, topLogprobsK: 5}
	on := run(measurement)
	if !reflect.DeepEqual(on, off) || len(on) != generated {
		t.Fatalf("greedy ids with top-k=%v, without=%v; want identical %d tokens", on, off, generated)
	}
	receipt := buildNativeDecodeTopLogprobs(measurement, nil)
	if receipt == nil || receipt.Schema != NativeDecodeTopLogprobsSchema || receipt.K != 5 || !reflect.DeepEqual(receipt.TokenIDs, on) || len(receipt.Rows) != len(on) {
		t.Fatalf("receipt=%+v, want schema/k=5 and one row per generated id %v", receipt, on)
	}
	for i, row := range receipt.Rows {
		if len(row) != 5 || row[0].TokenID != on[i] || receipt.Chosen[i].Logprob != row[0].Logprob {
			t.Fatalf("row %d=%+v chosen=%+v, want 5 candidates led by greedy id %d", i, row, receipt.Chosen[i], on[i])
		}
		seen := map[int]bool{}
		for j, c := range row {
			if seen[c.TokenID] || c.Logprob > 0 || math.IsNaN(c.Logprob) || (j > 0 && c.Logprob > row[j-1].Logprob) {
				t.Fatalf("row %d=%+v: want distinct ids, finite non-positive non-increasing logprobs", i, row)
			}
			seen[c.TokenID] = true
		}
	}
	if (&nativeInferenceMeasurement{inferenceDisabled: true}).topLogprobsK != 0 || buildNativeDecodeTopLogprobs(&nativeInferenceMeasurement{}, nil) != nil {
		t.Fatal("top logprobs must stay off by default")
	}
}

func TestWithNativeDecodeTopLogprobsClampsAndDefaultsOff(t *testing.T) {
	for in, want := range map[int]int{-3: 0, 0: 0, 5: 5, 99: MaxNativeDecodeTopLogprobs} {
		var sp SampleParams
		WithNativeDecodeTopLogprobs(in)(&sp)
		if sp.NativeDecodeTopLogprobs != want {
			t.Fatalf("k=%d -> %d, want %d", in, sp.NativeDecodeTopLogprobs, want)
		}
	}
}
