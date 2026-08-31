package ggufload

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestHierarchicalLoadConcurrencyBudget(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")

	const (
		budget  = 2
		tensors = 4
	)
	s := &WeightSource{File: &File{Tensors: make([]TensorInfo, tensors)}}
	started := make(chan struct{}, 32)
	release := make(chan struct{})
	var active atomic.Int32
	var peak atomic.Int32

	compute := func(TensorInfo) tensorWork {
		dequantBlocks(make([]float32, 2*dequantParallelMinBlocks), make([]byte, 2*dequantParallelMinBlocks), 1, 1,
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
		loadErr = s.parallelQuantLoad(compute, func(tensorWork) error { return nil })
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

	wantErr := errors.New("first tensor failed")
	s := &WeightSource{File: &File{Tensors: make([]TensorInfo, 2)}}
	var calls atomic.Int32
	err := s.parallelQuantLoad(func(TensorInfo) tensorWork {
		if calls.Add(1) == 1 {
			return tensorWork{err: wantErr}
		}
		return tensorWork{}
	}, func(tensorWork) error { return nil })
	if !errors.Is(err, wantErr) {
		t.Fatalf("parallelQuantLoad error = %v, want %v", err, wantErr)
	}
	if got := activeParallelLoads.Load(); got != 0 {
		t.Fatalf("active parallel load scopes after error = %d, want 0", got)
	}
}
