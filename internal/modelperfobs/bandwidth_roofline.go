package modelperfobs

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

const RooflineMeasurementSchema = "fak-host-memory-roofline/1"

const (
	MinRooflineWorkingSet = 1 << 20
	MaxRooflineWorkingSet = 512 << 20
	MinRooflineTrials     = 3
	MaxRooflineTrials     = 15
)

type RooflineBenchmarkOptions struct {
	WorkingSetBytes uint64
	Trials          int
	TargetDuration  time.Duration
	Threads         int
}

type RooflineTrial struct {
	Index        int     `json:"index"`
	Iterations   uint64  `json:"iterations"`
	DurationMS   float64 `json:"duration_ms"`
	TrafficBytes uint64  `json:"traffic_bytes"`
	GBS          float64 `json:"gb_s"`
}

type RooflineMeasurement struct {
	Schema                 string          `json:"schema"`
	Scope                  string          `json:"scope"`
	MachineClass           string          `json:"machine_class"`
	Method                 string          `json:"method"`
	Aggregation            string          `json:"aggregation"`
	TrafficAccounting      string          `json:"traffic_accounting"`
	WorkingSetBytes        uint64          `json:"working_set_bytes"`
	TargetDurationMS       int64           `json:"target_duration_ms"`
	Threads                int             `json:"threads"`
	MeasuredSustainableGBS float64         `json:"measured_sustainable_gb_s"`
	Trials                 []RooflineTrial `json:"trials"`
}

func DefaultRooflineBenchmarkOptions() RooflineBenchmarkOptions {
	return RooflineBenchmarkOptions{WorkingSetBytes: 64 << 20, Trials: 5, TargetDuration: 100 * time.Millisecond, Threads: runtime.GOMAXPROCS(0)}
}

func ValidateRooflineBenchmarkOptions(o RooflineBenchmarkOptions) error {
	if o.WorkingSetBytes < MinRooflineWorkingSet || o.WorkingSetBytes > MaxRooflineWorkingSet {
		return fmt.Errorf("roofline working set must be %d..%d bytes", MinRooflineWorkingSet, MaxRooflineWorkingSet)
	}
	if o.Trials < MinRooflineTrials || o.Trials > MaxRooflineTrials {
		return fmt.Errorf("roofline trials must be %d..%d", MinRooflineTrials, MaxRooflineTrials)
	}
	if o.Threads < 1 || o.Threads > 256 {
		return fmt.Errorf("roofline threads must be 1..256")
	}
	if uint64(o.Threads) > o.WorkingSetBytes {
		return fmt.Errorf("roofline threads exceed working set bytes")
	}
	if o.TargetDuration < 10*time.Millisecond || o.TargetDuration > 2*time.Second {
		return fmt.Errorf("roofline target duration must be 10ms..2s")
	}
	return nil
}

// MeasureHostMemoryRoofline measures sustained copy bandwidth. Each copy counts
// one byte read and one byte written. It is a host-memory benchmark, not a live
// process DRAM counter and not a GPU HBM measurement.
func MeasureHostMemoryRoofline(ctx context.Context, o RooflineBenchmarkOptions) (RooflineMeasurement, error) {
	if err := ValidateRooflineBenchmarkOptions(o); err != nil {
		return RooflineMeasurement{}, err
	}
	n := int(o.WorkingSetBytes)
	src, dst := make([]byte, n), make([]byte, n)
	// First-touch in the same parallel placement envelope as the benchmark so
	// page faults and single-thread NUMA placement do not become trial traffic.
	var warm sync.WaitGroup
	for worker := 0; worker < o.Threads; worker++ {
		lo := n * worker / o.Threads
		hi := n * (worker + 1) / o.Threads
		warm.Add(1)
		go func(lo, hi int) {
			defer warm.Done()
			for i := lo; i < hi; i++ {
				src[i] = byte(i*131 + 17)
			}
			copy(dst[lo:hi], src[lo:hi])
		}(lo, hi)
	}
	warm.Wait()
	trials := make([]RooflineTrial, 0, o.Trials)
	values := make([]float64, 0, o.Trials)
	for trial := 0; trial < o.Trials; trial++ {
		select {
		case <-ctx.Done():
			return RooflineMeasurement{}, ctx.Err()
		default:
		}
		iterations := uint64(1)
		var elapsed time.Duration
		for {
			start := time.Now()
			var wg sync.WaitGroup
			for worker := 0; worker < o.Threads; worker++ {
				lo := n * worker / o.Threads
				hi := n * (worker + 1) / o.Threads
				wg.Add(1)
				go func(lo, hi int) {
					defer wg.Done()
					for i := uint64(0); i < iterations; i++ {
						copy(dst[lo:hi], src[lo:hi])
					}
				}(lo, hi)
			}
			wg.Wait()
			elapsed = time.Since(start)
			if elapsed >= o.TargetDuration {
				break
			}
			if elapsed <= 0 {
				iterations *= 2
				continue
			}
			next := uint64(float64(iterations) * float64(o.TargetDuration) / float64(elapsed) * 1.05)
			if next <= iterations {
				next = iterations + 1
			}
			iterations = next
		}
		traffic := o.WorkingSetBytes * 2 * iterations
		gbs := float64(traffic) / elapsed.Seconds() / 1e9
		trials = append(trials, RooflineTrial{Index: trial + 1, Iterations: iterations, DurationMS: float64(elapsed) / float64(time.Millisecond), TrafficBytes: traffic, GBS: gbs})
		values = append(values, gbs)
		runtime.KeepAlive(dst)
	}
	sort.Float64s(values)
	median := values[len(values)/2]
	return RooflineMeasurement{Schema: RooflineMeasurementSchema, Scope: "host-memory", MachineClass: runtime.GOOS + "/" + runtime.GOARCH, Method: "parallel-copy", Aggregation: "median", TrafficAccounting: "read-plus-write-2-bytes-per-copied-byte", WorkingSetBytes: o.WorkingSetBytes, TargetDurationMS: o.TargetDuration.Milliseconds(), Threads: o.Threads, MeasuredSustainableGBS: median, Trials: trials}, nil
}
