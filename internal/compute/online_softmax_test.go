package compute

import (
	"math"
	"math/rand"
	"testing"
)

func referenceStandardSoftmax(logits []float32) []float32 {
	out := append([]float32(nil), logits...)
	mx := out[0]
	for _, v := range out {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range out {
		e := math.Exp(float64(v - mx))
		out[i] = float32(e)
		sum += e
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}

func referenceSinkSoftmax(logits []float32, sinkLogit float32) ([]float32, float32) {
	out := append([]float32(nil), logits...)
	mx := sinkLogit
	for _, v := range out {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range out {
		e := math.Exp(float64(v - mx))
		out[i] = float32(e)
		sum += e
	}
	sinkTerm := math.Exp(float64(sinkLogit - mx))
	sum += sinkTerm

	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	sinkProb := float32(sinkTerm / sum)
	return out, sinkProb
}

func TestSinkAwareCPUOnlineSoftmaxWitness(t *testing.T) {
	// First witness requirements (#9927):
	// 1. Correctness against reference for both sink and no-sink
	// 2. Numerical stability under extreme logits (+1000, -1000)
	// 3. Long context streaming (up to 16,384 positions)
	// 4. Bitwise repeatability on repeats
	// 5. Conservation of probability: sum(tokens) + sink_prob == 1.0

	rng := rand.New(rand.NewSource(101))

	// 1. No-sink equivalence with standard softmax
	t.Run("no_sink_matches_standard_softmax", func(t *testing.T) {
		lengths := []int{1, 16, 64, 128, 512, 1024, 4096}
		for _, n := range lengths {
			logits := make([]float32, n)
			for i := range logits {
				logits[i] = rng.Float32()*20 - 10
			}
			want := referenceStandardSoftmax(logits)

			gotLogits := append([]float32(nil), logits...)
			receipt, err := OnlineSoftmaxSinkInPlace(gotLogits, false, 0)
			if err != nil {
				t.Fatalf("n=%d: OnlineSoftmaxSinkInPlace failed: %v", n, err)
			}

			if receipt.HasSink {
				t.Fatal("expected HasSink = false")
			}
			if math.Abs(float64(receipt.SumProbabilities-1.0)) > 1e-5 {
				t.Fatalf("n=%d: sum of probabilities %v != 1.0", n, receipt.SumProbabilities)
			}

			for i := range want {
				if math.Abs(float64(gotLogits[i]-want[i])) > 1e-6 {
					t.Fatalf("n=%d: mismatch at index %d: got %v, want %v", n, i, gotLogits[i], want[i])
				}
			}
		}
	})

	// 2. Sink awareness matching reference sink formulation
	t.Run("sink_aware_matches_reference", func(t *testing.T) {
		lengths := []int{4, 32, 128, 512, 2048}
		sinkLogits := []float32{-5.0, 0.0, 2.5, 10.0}

		for _, n := range lengths {
			for _, sink := range sinkLogits {
				logits := make([]float32, n)
				for i := range logits {
					logits[i] = rng.Float32()*10 - 5
				}
				wantLogits, wantSinkProb := referenceSinkSoftmax(logits, sink)

				gotLogits := append([]float32(nil), logits...)
				receipt, err := OnlineSoftmaxSinkInPlace(gotLogits, true, sink)
				if err != nil {
					t.Fatalf("n=%d sink=%v: failed: %v", n, sink, err)
				}

				if !receipt.HasSink {
					t.Fatal("expected HasSink = true")
				}
				if math.Abs(float64(receipt.SinkProbability-wantSinkProb)) > 1e-6 {
					t.Fatalf("n=%d sink=%v: sink prob mismatch: got %v, want %v", n, sink, receipt.SinkProbability, wantSinkProb)
				}

				totalMass := receipt.SumProbabilities + receipt.SinkProbability
				if math.Abs(float64(totalMass-1.0)) > 1e-5 {
					t.Fatalf("n=%d sink=%v: probability conservation violated: %v", n, sink, totalMass)
				}

				for i := range wantLogits {
					if math.Abs(float64(gotLogits[i]-wantLogits[i])) > 1e-6 {
						t.Fatalf("n=%d sink=%v: logit mismatch at %d: got %v, want %v", n, sink, i, gotLogits[i], wantLogits[i])
					}
				}
			}
		}
	})

	// 3. Extreme logits numerical stability
	t.Run("extreme_logits_stability", func(t *testing.T) {
		extreme := []float32{1000.0, 999.0, 500.0, -1000.0, 0.0, -500.0}
		receipt, err := OnlineSoftmaxSinkInPlace(extreme, true, 998.0)
		if err != nil {
			t.Fatalf("extreme logits failed: %v", err)
		}
		if math.IsNaN(float64(receipt.Denominator)) || math.IsInf(float64(receipt.Denominator), 0) {
			t.Fatalf("extreme denominator is non-finite: %v", receipt.Denominator)
		}
		for i, p := range extreme {
			if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
				t.Fatalf("extreme output %d is non-finite: %v", i, p)
			}
		}
		if extreme[0] <= 0 || extreme[0] > 1.0 {
			t.Fatalf("max element prob out of range: %v", extreme[0])
		}
	})

	// 4. Long context streaming (16k)
	t.Run("long_context_streaming", func(t *testing.T) {
		n := 16384
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = rng.Float32()*4 - 2
		}
		sink := float32(1.5)
		wantLogits, wantSinkProb := referenceSinkSoftmax(logits, sink)

		gotLogits := append([]float32(nil), logits...)
		receipt, err := OnlineSoftmaxSinkInPlace(gotLogits, true, sink)
		if err != nil {
			t.Fatalf("16k long context failed: %v", err)
		}
		if math.Abs(float64(receipt.SinkProbability-wantSinkProb)) > 1e-5 {
			t.Fatalf("16k sink prob mismatch: got %v, want %v", receipt.SinkProbability, wantSinkProb)
		}
		for i := 0; i < 100; i++ { // spot check first 100
			if math.Abs(float64(gotLogits[i]-wantLogits[i])) > 1e-5 {
				t.Fatalf("16k mismatch at %d: got %v, want %v", i, gotLogits[i], wantLogits[i])
			}
		}
	})

	// 5. Bitwise repeatability
	t.Run("bitwise_repeatability", func(t *testing.T) {
		n := 128
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = rng.Float32()*6 - 3
		}

		run1 := append([]float32(nil), logits...)
		rec1, err1 := OnlineSoftmaxSinkInPlace(run1, true, 0.5)
		if err1 != nil {
			t.Fatal(err1)
		}

		run2 := append([]float32(nil), logits...)
		rec2, err2 := OnlineSoftmaxSinkInPlace(run2, true, 0.5)
		if err2 != nil {
			t.Fatal(err2)
		}

		if rec1 != rec2 {
			t.Fatalf("receipt mismatch: %+v vs %+v", rec1, rec2)
		}
		for i := range run1 {
			if run1[i] != run2[i] {
				t.Fatalf("bitwise drift at index %d: %v vs %v", i, run1[i], run2[i])
			}
		}
	})
}
