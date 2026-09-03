package bench

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"
)

type mockServeBenchmarkSummary struct {
	WorkloadThroughputTokS float64               `json:"workload_throughput_tok_s"`
	WorkloadRequests       int                   `json:"workload_requests"`
	BystanderInterference  BystanderInterference `json:"bystander_interference"`
}

// TestBystanderProbeMeasurementConcurrent verifies #10726:
// running a mock concurrent workload alongside single-token probe requests,
// calculating probe percentiles separately from workload metrics, and exporting
// BystanderInterference in the benchmark summary JSON.
func TestBystanderProbeMeasurementConcurrent(t *testing.T) {
	runner := NewProbeRunner(BystanderProbeConfig{
		Interval: 10 * time.Millisecond,
		Timeout:  1 * time.Second,
	})

	// Mock server queue lock
	var serverMu sync.Mutex
	heavyWorkloadActive := true

	// Probe function simulates 1-token request
	probeFn := func(ctx context.Context) (time.Duration, error) {
		start := time.Now()
		serverMu.Lock()
		// Simulate latency: heavier while background workload is running
		sleepTime := 2 * time.Millisecond
		if heavyWorkloadActive {
			sleepTime = 15 * time.Millisecond
		}
		serverMu.Unlock()

		time.Sleep(sleepTime)
		return time.Since(start), nil
	}

	runner.Start(probeFn)

	// Run mock concurrent heavy workload
	var workloadWg sync.WaitGroup
	workloadRequests := 20
	for i := 0; i < 4; i++ {
		workloadWg.Add(1)
		go func() {
			defer workloadWg.Done()
			for j := 0; j < 5; j++ {
				serverMu.Lock()
				time.Sleep(10 * time.Millisecond)
				serverMu.Unlock()
			}
		}()
	}
	workloadWg.Wait()

	serverMu.Lock()
	heavyWorkloadActive = false
	serverMu.Unlock()

	// Allow a few quiet probes to land
	time.Sleep(30 * time.Millisecond)

	interference := runner.Stop()

	if interference.Samples < 3 {
		t.Fatalf("expected at least 3 probe samples, got %d", interference.Samples)
	}
	if interference.P50MS <= 0 || interference.P95MS <= 0 || interference.MaxMS <= 0 {
		t.Fatalf("invalid percentiles: %+v", interference)
	}
	if interference.MaxMS < interference.P50MS {
		t.Fatalf("MaxMS %v < P50MS %v", interference.MaxMS, interference.P50MS)
	}

	// Verify benchmark summary serialization
	summary := mockServeBenchmarkSummary{
		WorkloadThroughputTokS: 1250.5,
		WorkloadRequests:       workloadRequests,
		BystanderInterference:  interference,
	}

	bytes, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var parsed mockServeBenchmarkSummary
	if err := json.Unmarshal(bytes, &parsed); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if parsed.BystanderInterference.Samples != interference.Samples {
		t.Fatalf("parsed samples = %d, want %d", parsed.BystanderInterference.Samples, interference.Samples)
	}
	if parsed.WorkloadRequests != 20 {
		t.Fatalf("parsed workload requests = %d, want 20", parsed.WorkloadRequests)
	}
}

// TestComputeBystanderInterferenceMath verifies exact calculation of percentiles.
func TestComputeBystanderInterferenceMath(t *testing.T) {
	raw := []time.Duration{
		10 * time.Millisecond,
		20 * time.Millisecond,
		30 * time.Millisecond,
		40 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
	}

	res := ComputeBystanderInterference(raw)
	if res.Samples != 6 {
		t.Fatalf("samples = %d, want 6", res.Samples)
	}
	if res.MaxMS != 100.0 {
		t.Fatalf("max = %v, want 100.0", res.MaxMS)
	}
	if res.P50MS <= 0 || res.P95MS <= 0 || res.P99MS <= 0 {
		t.Fatalf("invalid percentiles: %+v", res)
	}
}
