package ggufload

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPlanLoadConcurrencyPartitionsBudget(t *testing.T) {
	tests := []struct {
		name                  string
		total, outer, tensors int
		want                  loadConcurrencyPlan
	}{
		{name: "default cap", total: 32, outer: 16, tensors: 100, want: loadConcurrencyPlan{Budget: 32, Outer: 16, Inner: 2}},
		{name: "tuned outer", total: 32, outer: 4, tensors: 100, want: loadConcurrencyPlan{Budget: 32, Outer: 4, Inner: 8}},
		{name: "serial outer uses inner budget", total: 32, outer: 1, tensors: 100, want: loadConcurrencyPlan{Budget: 32, Outer: 1, Inner: 32}},
		{name: "few tensors redistribute", total: 32, outer: 16, tensors: 2, want: loadConcurrencyPlan{Budget: 32, Outer: 2, Inner: 16}},
		{name: "explicit oversubscription remains visible", total: 32, outer: 64, tensors: 100, want: loadConcurrencyPlan{Budget: 64, Outer: 64, Inner: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := planLoadConcurrency(tt.total, tt.outer, tt.tensors)
			if got != tt.want {
				t.Fatalf("planLoadConcurrency(%d, %d, %d) = %+v, want %+v", tt.total, tt.outer, tt.tensors, got, tt.want)
			}
			if got.Outer*got.Inner > got.Budget {
				t.Fatalf("partition exceeds budget: %+v", got)
			}
		})
	}
}

func TestHierarchicalLoadConcurrencyBudget(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	oldProcs := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(oldProcs) })

	const (
		budget  = 4
		tensors = 4
	)
	s := &WeightSource{File: &File{Tensors: make([]TensorInfo, tensors)}}
	started := make(chan struct{}, 32)
	release := make(chan struct{})
	var active atomic.Int32
	var peak atomic.Int32

	compute := func(_ TensorInfo, innerWorkers int) tensorWork {
		dequantBlocksLimited(make([]float32, 2*dequantParallelMinBlocks), make([]byte, 2*dequantParallelMinBlocks), 1, 1, innerWorkers,
			func([]float32, []byte) {
				n := active.Add(1)
				for old := peak.Load(); n > old && !peak.CompareAndSwap(old, n); old = peak.Load() {
				}
				started <- struct{}{}
				<-release
				active.Add(-1)
			})
		return tensorWork{}
	}

	var wg sync.WaitGroup
	wg.Add(1)
	var loadErr error
	go func() {
		defer wg.Done()
		loadErr = s.parallelQuantLoadContextBudget(context.Background(), compute, func(tensorWork) error { return nil })
	}()

	for i := 0; i < budget; i++ {
		<-started
	}
	select {
	case <-started:
		close(release)
		wg.Wait()
		t.Fatalf("nested dequant exceeded budget %d; peak active bodies = %d", budget, peak.Load())
	case <-time.After(250 * time.Millisecond):
		// Both outer workers are blocked in their sole dequant body. If nested work
		// multiplied the pool, another body would have entered before this deadline.
	}
	close(release)
	wg.Wait()
	if loadErr != nil {
		t.Fatalf("parallelQuantLoad: %v", loadErr)
	}
	if got := peak.Load(); got != budget {
		t.Fatalf("peak active dequant bodies = %d, want %d", got, budget)
	}
}

func TestHierarchicalLoadConcurrencyBudgetReleasesOnError(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	oldProcs := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(oldProcs) })

	wantErr := errors.New("first tensor failed")
	s := &WeightSource{File: &File{Tensors: make([]TensorInfo, 2)}}
	var calls atomic.Int32
	err := s.parallelQuantLoadContextBudget(context.Background(), func(TensorInfo, int) tensorWork {
		if calls.Add(1) == 1 {
			return tensorWork{err: wantErr}
		}
		return tensorWork{}
	}, func(tensorWork) error { return nil })
	if !errors.Is(err, wantErr) {
		t.Fatalf("parallelQuantLoad error = %v, want %v", err, wantErr)
	}
	var innerSeen atomic.Int32
	err = s.parallelQuantLoadContextBudget(context.Background(), func(_ TensorInfo, innerWorkers int) tensorWork {
		if innerWorkers != 2 {
			return tensorWork{err: errors.New("unexpected inner worker budget")}
		}
		innerSeen.Add(1)
		return tensorWork{}
	}, func(tensorWork) error { return nil })
	if err != nil {
		t.Fatalf("second load after error: %v", err)
	}
	if got := innerSeen.Load(); got != 2 {
		t.Fatalf("second load compute calls = %d, want 2", got)
	}
}
