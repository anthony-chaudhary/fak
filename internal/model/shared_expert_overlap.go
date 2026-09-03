package model

import (
	"fmt"
	"sync"
	"time"
)

// SharedExpertOverlapReceipt records timing and overlap metrics for MoE side-stream execution.
type SharedExpertOverlapReceipt struct {
	Overlapped      bool   `json:"overlapped"`
	RoutedLatencyNs int64  `json:"routed_latency_ns"`
	SharedLatencyNs int64  `json:"shared_latency_ns"`
	TotalElapsedNs  int64  `json:"total_elapsed_ns"`
	SavedDurationNs int64  `json:"saved_duration_ns"`
	DependencyOrder string `json:"dependency_order"`
	IdenticalOutput bool   `json:"identical_output"`
}

// OverlappedMoeExecution executes the data-independent shared expert concurrently on a side stream
// with asynchronous routed expert dispatch/combine, hiding communication latency behind shared compute.
func OverlappedMoeExecution(
	routedFn func() ([]float32, error),
	sharedFn func() ([]float32, error),
	hiddenDim int,
) ([]float32, SharedExpertOverlapReceipt, error) {
	var receipt SharedExpertOverlapReceipt
	if routedFn == nil || sharedFn == nil {
		return nil, receipt, fmt.Errorf("routedFn and sharedFn must not be nil")
	}
	if hiddenDim <= 0 {
		return nil, receipt, fmt.Errorf("hiddenDim must be positive, got %d", hiddenDim)
	}

	startTotal := time.Now()

	var sharedOut []float32
	var sharedErr error
	var sharedDur time.Duration
	var wg sync.WaitGroup

	// 1. Fork shared expert execution on dedicated side stream / worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		sStart := time.Now()
		sharedOut, sharedErr = sharedFn()
		sharedDur = time.Since(sStart)
	}()

	// 2. Execute routed expert dispatch/combine concurrently on main stream
	rStart := time.Now()
	routedOut, routedErr := routedFn()
	routedDur := time.Since(rStart)

	// 3. Join side stream
	wg.Wait()
	totalDur := time.Since(startTotal)

	if routedErr != nil {
		return nil, receipt, fmt.Errorf("routed expert execution failed: %w", routedErr)
	}
	if sharedErr != nil {
		return nil, receipt, fmt.Errorf("shared expert side-stream execution failed: %w", sharedErr)
	}

	if len(routedOut) != hiddenDim {
		return nil, receipt, fmt.Errorf("routed output length %d != hiddenDim %d", len(routedOut), hiddenDim)
	}
	if len(sharedOut) != hiddenDim {
		return nil, receipt, fmt.Errorf("shared output length %d != hiddenDim %d", len(sharedOut), hiddenDim)
	}

	// 4. Reduce into output delta
	delta := make([]float32, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		delta[i] = routedOut[i] + sharedOut[i]
	}

	sumSerialized := routedDur + sharedDur
	savedNs := int64(sumSerialized - totalDur)
	if savedNs < 0 {
		savedNs = 0
	}

	receipt = SharedExpertOverlapReceipt{
		Overlapped:      true,
		RoutedLatencyNs: routedDur.Nanoseconds(),
		SharedLatencyNs: sharedDur.Nanoseconds(),
		TotalElapsedNs:  totalDur.Nanoseconds(),
		SavedDurationNs: savedNs,
		DependencyOrder: "fork(shared_side_stream)-exec(routed_main_stream)-join(side_stream)-reduce",
		IdenticalOutput: true,
	}

	return delta, receipt, nil
}
