package beam

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func beamTestLogits(p ...float64) []float32 {
	out := make([]float32, len(p))
	for i, x := range p {
		out[i] = float32(math.Log(x))
	}
	return out
}

// fak-test:runtime fast est=10ms lane=default
func TestSearchSurvivorsBeatGreedyAndOwnInputs(t *testing.T) {
	prompt := []int{2}
	source := append([]int(nil), prompt...)
	next := func(_ context.Context, prefix []int) ([]float32, error) {
		history := append([]int(nil), prefix...)
		prefix[0] = 0 // A provider cannot mutate the search's prompt or beam histories.
		if len(history) == 1 {
			return beamTestLogits(.6, .399999, .000001), nil
		}
		if history[1] == 0 {
			return beamTestLogits(.34, .33, .33), nil
		}
		return beamTestLogits(.99, .005, .005), nil
	}
	cfg := SearchConfig{Width: 2, MaxTokens: 2, VocabSize: 3, LengthPenalty: 0}
	got, err := Search(context.Background(), prompt, cfg, next)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Sequences[0].TokenIDs, []int{1, 0}) {
		t.Fatalf("beam winner: %+v", got.Sequences)
	}
	if !reflect.DeepEqual(prompt, source) {
		t.Fatal("prompt mutated")
	}
	if got.ForwardCalls != 3 || got.ReplayedTokens != 5 || got.ReplayTokenBound != 5 {
		t.Fatalf("work %+v", got)
	}
	cfg.Width = 1
	greedy, err := Search(context.Background(), prompt, cfg, next)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(greedy.Sequences[0].TokenIDs, []int{0, 0}) {
		t.Fatalf("greedy %+v", greedy)
	}
	if !(got.Sequences[0].CumulativeLogprob > greedy.Sequences[0].CumulativeLogprob) {
		t.Fatal("winner is not more probable")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSearchEOSLengthPenaltyAndCompletedLifecycle(t *testing.T) {
	next := func(_ context.Context, p []int) ([]float32, error) {
		if len(p) == 1 {
			return beamTestLogits(.6, .3, .1), nil
		}
		if p[1] == 0 {
			return beamTestLogits(.1, .1, .8), nil
		}
		return beamTestLogits(.9, .05, .05), nil
	}
	got, err := Search(context.Background(), []int{0}, SearchConfig{Width: 2, MaxTokens: 2, VocabSize: 3, LengthPenalty: 1, EOS: []int{2}}, next)
	if err != nil {
		t.Fatal(err)
	}
	best := got.Sequences[0]
	if !reflect.DeepEqual(best.TokenIDs, []int{0}) || best.GeneratedTokens != 2 || best.FinishReason != "stop" {
		t.Fatalf("EOS accounting %+v", best)
	}
	if math.Abs(best.Score-best.CumulativeLogprob/2) > 1e-12 {
		t.Fatal("EOS excluded from score length")
	}
	if math.Abs(best.CumulativeLogprob-math.Log(.48)) > 1e-6 {
		t.Fatalf("cumulative=%g", best.CumulativeLogprob)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSearchStableTiesAndFiniteLogSoftmax(t *testing.T) {
	cfg := SearchConfig{Width: 3, MaxTokens: 2, VocabSize: 3, LengthPenalty: 0}
	want := [][]int{{0, 0}, {0, 1}, {0, 2}}
	for repeat := 0; repeat < 3; repeat++ {
		got, err := Search(context.Background(), []int{1}, cfg, func(context.Context, []int) ([]float32, error) { return []float32{3e38, 3e38, 3e38}, nil })
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range got.Sequences {
			if !reflect.DeepEqual(row.TokenIDs, want[i]) {
				t.Fatalf("ties: %+v", got)
			}
			if math.Abs(row.CumulativeLogprob+2*math.Log(3)) > 1e-12 {
				t.Fatalf("unstable log-softmax %g", row.CumulativeLogprob)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSearchRefusesBeforeForwardAndNeverReturnsPartial(t *testing.T) {
	base := SearchConfig{Width: 2, MaxTokens: 2, VocabSize: 3, LengthPenalty: 1}
	for _, change := range []func(*SearchConfig){
		func(c *SearchConfig) { c.Width = 0 }, func(c *SearchConfig) { c.Width = 9 },
		func(c *SearchConfig) { c.MaxTokens = 129 }, func(c *SearchConfig) { c.VocabSize = MaxSearchVocab + 1 },
		func(c *SearchConfig) { c.LengthPenalty = math.NaN() }, func(c *SearchConfig) { c.EOS = []int{3} },
		func(c *SearchConfig) { c.EOS = []int{0, 1, 2, 0} },
	} {
		cfg := base
		change(&cfg)
		calls := 0
		got, err := Search(context.Background(), []int{0}, cfg, func(context.Context, []int) ([]float32, error) { calls++; return nil, nil })
		if err == nil || got != nil || calls != 0 {
			t.Fatalf("refusal got=%v err=%v calls=%d", got, err, calls)
		}
	}
	large := make([]int, 3900)
	cfg := base
	cfg.Width = 8
	cfg.MaxTokens = 128
	if _, err := ValidateSearch(large, cfg); err == nil {
		t.Fatal("unbounded replay admitted")
	}
	sentinel := errors.New("forward failed")
	for _, row := range [][]float32{{1, 2}, {1, float32(math.NaN()), 2}, {1, float32(math.Inf(1)), 2}} {
		got, err := Search(context.Background(), []int{0}, base, func(context.Context, []int) ([]float32, error) { return row, nil })
		if err == nil || got != nil {
			t.Fatalf("bad row accepted: %v", row)
		}
	}
	calls := 0
	got, err := Search(context.Background(), []int{0}, base, func(context.Context, []int) ([]float32, error) {
		calls++
		if calls > 1 {
			return nil, sentinel
		}
		return []float32{1, 2, 3}, nil
	})
	if got != nil || !errors.Is(err, sentinel) {
		t.Fatalf("partial result: %v %v", got, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSearchCancellationClosesPublication(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	got, err := Search(ctx, []int{0}, SearchConfig{Width: 2, MaxTokens: 4, VocabSize: 3, LengthPenalty: 1}, func(context.Context, []int) ([]float32, error) {
		calls++
		cancel()
		return []float32{1, 2, 3}, nil
	})
	if got != nil || !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("got=%v err=%v calls=%d", got, err, calls)
	}
}
