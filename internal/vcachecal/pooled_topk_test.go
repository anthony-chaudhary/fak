package vcachecal

import (
	"math"
	"reflect"
	"testing"
)

func TestPooledPerHeadTopKAllocation(t *testing.T) {
	t.Run("avg pooling counts padding and keeps window", func(t *testing.T) {
		// prefix 6, window 2, budget 5 => keep 3 prefix positions per head.
		scores := [][]float64{
			{0, 0, 9, 0, 0, 3},
			{3, 0, 0, 0, 0, 0},
		}
		got, err := PooledPerHeadTopK(scores, PooledTopKConfig{Window: 2, Budget: 5, KernelSize: 3, Pooling: PoolAvg})
		if err != nil {
			t.Fatal(err)
		}
		// head0 pooled 0,3,3,3,1,1: the three tied 3s win.
		// head1 pooled 1,1,0,0,0,0 (edge divides by 3, not 2): 0,1, then lowest tie 2.
		want := [][]int{{1, 2, 3, 6, 7}, {0, 1, 2, 6, 7}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("pooling changes the pick versus raw top-k", func(t *testing.T) {
		scores := [][]float64{{5, 0, 0, 4, 4, 4, 0}}
		got, err := PooledPerHeadTopK(scores, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 3, Pooling: PoolAvg})
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]int{{4, 7}}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
		raw, err := PooledPerHeadTopK(scores, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 1, Pooling: PoolAvg})
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]int{{0, 7}}; !reflect.DeepEqual(raw, want) {
			t.Fatalf("kernel 1 should be raw top-k: got %v want %v", raw, want)
		}
	})
	t.Run("max pooling ignores padding with negative scores", func(t *testing.T) {
		scores := [][]float64{{-1, -5, -5, -5, -2}}
		got, err := PooledPerHeadTopK(scores, PooledTopKConfig{Window: 1, Budget: 3, KernelSize: 3, Pooling: PoolMax})
		if err != nil {
			t.Fatal(err)
		}
		// pooled -1,-1,-5,-2,-2; zero padding would have made both edges 0.
		if want := [][]int{{0, 1, 5}}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("NaN ranks highest", func(t *testing.T) {
		scores := [][]float64{{1, 2, math.NaN(), 3}}
		got, err := PooledPerHeadTopK(scores, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 1, Pooling: PoolMax})
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]int{{2, 4}}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("short sequence keeps everything", func(t *testing.T) {
		got, err := PooledPerHeadTopK([][]float64{{1, 2}}, PooledTopKConfig{Window: 2, Budget: 5, KernelSize: 3})
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]int{{0, 1, 2, 3}}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("sequence equal to budget keeps everything", func(t *testing.T) {
		got, err := PooledPerHeadTopK([][]float64{{3, 1, 2}}, PooledTopKConfig{Window: 2, Budget: 5, KernelSize: 3})
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]int{{0, 1, 2, 3, 4}}; !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("no heads", func(t *testing.T) {
		got, err := PooledPerHeadTopK(nil, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 1})
		if err != nil || got != nil {
			t.Fatalf("got %v, %v", got, err)
		}
	})
	t.Run("rejects invalid config", func(t *testing.T) {
		ok := [][]float64{{1, 2, 3}}
		cases := []struct {
			name   string
			scores [][]float64
			cfg    PooledTopKConfig
		}{
			{"zero window", ok, PooledTopKConfig{Window: 0, Budget: 2, KernelSize: 1}},
			{"budget not above window", ok, PooledTopKConfig{Window: 2, Budget: 2, KernelSize: 1}},
			{"even kernel", ok, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 4}},
			{"zero kernel", ok, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 0}},
			{"bad pooling", ok, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 1, Pooling: PoolingMode(9)}},
			{"ragged heads", [][]float64{{1, 2}, {1}}, PooledTopKConfig{Window: 1, Budget: 2, KernelSize: 1}},
		}
		for _, c := range cases {
			if got, err := PooledPerHeadTopK(c.scores, c.cfg); err == nil {
				t.Errorf("%s: want error, got %v", c.name, got)
			}
		}
	})
}
