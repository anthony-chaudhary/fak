package harnessbench

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agentsched"
	"github.com/anthony-chaudhary/fak/internal/workspaceslot"
)

// RunSpawningInvariantWitness executes consecutive synthetic agent spawn, step, and
// slot-recycling cycles, verifying O(1) launch latency and bounded memory drift (#11183).
func RunSpawningInvariantWitness(cycles int, baseDir string) (*SpawningInvariantReceipt, error) {
	if cycles < 2000 {
		cycles = 2000
	}
	if baseDir == "" {
		tmp, err := os.MkdirTemp("", "fak-harnessbench-slots-*")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(tmp)
		baseDir = tmp
	}

	ring, err := workspaceslot.NewSlotRing(baseDir, 8)
	if err != nil {
		return nil, fmt.Errorf("harnessbench: init slot ring: %w", err)
	}
	defer ring.Close()

	ctx := context.Background()
	var epoch1TotalNS int64
	var epochNTotalNS int64
	const sampleCount = 100
	const warmupCycles = 200

	var allocAt1000 uint64
	var allocAtEnd uint64

	for i := 0; i < cycles; i++ {
		t0 := time.Now()
		slot, err := ring.Acquire(ctx, fmt.Sprintf("agent-%d", i))
		launchElapsed := time.Since(t0).Nanoseconds()
		if err != nil {
			return nil, fmt.Errorf("cycle %d acquire: %w", i, err)
		}

		// Simulate step writing scratch data
		scratchFile := filepath.Join(slot.ScratchDir, "step.log")
		_ = os.WriteFile(scratchFile, []byte("step completed"), 0644)

		if err := ring.Release(slot); err != nil {
			return nil, fmt.Errorf("cycle %d release: %w", i, err)
		}

		if i >= warmupCycles && i < warmupCycles+sampleCount {
			epoch1TotalNS += launchElapsed
		} else if i >= cycles-sampleCount {
			epochNTotalNS += launchElapsed
		}

		if i == 1000 {
			runtime.GC()
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			allocAt1000 = m.Alloc
		}
	}

	runtime.GC()
	var mFinal runtime.MemStats
	runtime.ReadMemStats(&mFinal)
	allocAtEnd = mFinal.Alloc

	avgEpoch1 := epoch1TotalNS / sampleCount
	avgEpochN := epochNTotalNS / sampleCount
	ratio := float64(avgEpochN) / float64(avgEpoch1)
	if avgEpoch1 == 0 {
		ratio = 1.0
	}

	drift := int64(allocAtEnd) - int64(allocAt1000)

	// Invariant: O(1) launch latency - ratio <= 1.05 (allowing for sub-microsecond timer noise floor within 500ns)
	passLatency := ratio <= 1.05 || (avgEpochN-avgEpoch1) < 500
	// Invariant: zero net memory leak (< 5MB heap growth after GC across cycles)
	passMemory := drift < 5*1024*1024

	return &SpawningInvariantReceipt{
		Schema:          SpawningInvariantSchema,
		TotalCycles:     cycles,
		Epoch1LatencyNS: avgEpoch1,
		EpochNLatencyNS: avgEpochN,
		LatencyRatio:    ratio,
		InitialAllocB:   allocAt1000,
		FinalAllocB:     allocAtEnd,
		AllocDriftBytes: drift,
		Pass:            passLatency && passMemory,
		CompletedAt:     time.Now().UTC(),
	}, nil
}

// RunThunderingHerdWitness executes a sudden burst of concurrent agent tasks against
// a worker pool, verifying that the 512 queue capacity is enforced with non-blocking
// 429 backpressure and zero thread leaks (#11183).
func RunThunderingHerdWitness(burstSize, poolWorkers, queueCap int) (*ThunderingHerdReceipt, error) {
	if burstSize <= 0 {
		burstSize = 1000
	}
	if poolWorkers <= 0 {
		poolWorkers = 16
	}
	if queueCap <= 0 {
		queueCap = abi.MaxQueueCapacity
	}

	gov := agentsched.NewGovernor(agentsched.GovernorConfig{
		BaseConcurrency: poolWorkers,
		QueueCapacity:   queueCap,
	})

	initialThreads := runtime.NumGoroutine()

	var enqueuedCount atomic.Int64
	var rejected429 atomic.Int64

	var burstWg sync.WaitGroup
	// Dispatch sudden burst concurrently
	for i := 0; i < burstSize; i++ {
		burstWg.Add(1)
		go func(taskID int) {
			defer burstWg.Done()
			prio := abi.ThreadPriority(taskID % 4)
			err := gov.Submit(&agentsched.Task{
				ID:       fmt.Sprintf("burst-task-%d", taskID),
				Priority: prio,
			})
			if err != nil {
				var admErr *abi.AdmissionError
				if errors.As(err, &admErr) && admErr.Code == abi.AdmissionCodeQueueFull {
					rejected429.Add(1)
				}
			} else {
				enqueuedCount.Add(1)
			}
		}(i)
	}

	burstWg.Wait()

	// Drain enqueued tasks using a worker pool
	var processedCount atomic.Int64
	var workersWg sync.WaitGroup

	for w := 0; w < poolWorkers; w++ {
		workersWg.Add(1)
		go func() {
			defer workersWg.Done()
			for {
				task, verdict, err := gov.TryAdmit()
				if err != nil || !verdict.Admitted || task == nil {
					if gov.Queue().Len() == 0 {
						return
					}
					time.Sleep(500 * time.Microsecond)
					continue
				}

				// Simulate task execution
				time.Sleep(100 * time.Microsecond)
				processedCount.Add(1)
				gov.Release(task)
			}
		}()
	}

	workersWg.Wait()
	finalThreads := runtime.NumGoroutine()

	cleanRecovery := gov.InFlight() == 0 && gov.Queue().Len() == 0
	passQueueCap := int(enqueuedCount.Load()) <= queueCap
	passBackpressure := int(rejected429.Load()) == (burstSize - int(enqueuedCount.Load()))
	passClean := cleanRecovery && (finalThreads <= initialThreads+5)

	return &ThunderingHerdReceipt{
		Schema:         ThunderingHerdSchema,
		BurstSize:      burstSize,
		PoolWorkers:    poolWorkers,
		QueueCap:       queueCap,
		EnqueuedCount:  int(enqueuedCount.Load()),
		Rejected429:    int(rejected429.Load()),
		ProcessedCount: int(processedCount.Load()),
		InitialThreads: initialThreads,
		FinalThreads:   finalThreads,
		CleanRecovery:  cleanRecovery,
		Pass:           passQueueCap && passBackpressure && passClean,
		CompletedAt:    time.Now().UTC(),
	}, nil
}

// RunThermalTelemetryWitness tests dynamic load shedding and recovery under simulated thermal pressure.
func RunThermalTelemetryWitness() (*ThermalSheddingReceipt, error) {
	const baseK = 16
	gov := agentsched.NewGovernor(agentsched.GovernorConfig{
		BaseConcurrency: baseK,
		DropP3OnStress:  true,
	})

	initialK := gov.EffectiveConcurrency()

	// Submit mixed tasks
	_ = gov.Submit(&agentsched.Task{ID: "p1-work", Priority: abi.ThreadPriorityP1Interactive})
	_ = gov.Submit(&agentsched.Task{ID: "p3-spec", Priority: abi.ThreadPriorityP3Speculative})

	// Inject thermal & CPU stress (> 85%)
	peakCPU := 93.5
	gov.UpdateTelemetry(agentsched.HostTelemetry{
		CPUPct:          peakCPU,
		ThermalPressure: true,
	})

	throttledK := gov.EffectiveConcurrency()
	p3Paused := gov.IsP3Paused()

	// Inject recovery telemetry
	for i := 0; i < 20; i++ {
		gov.UpdateTelemetry(agentsched.HostTelemetry{
			CPUPct:          35.0,
			ThermalPressure: false,
		})
	}

	restoredK := gov.EffectiveConcurrency()

	pass := initialK == baseK && throttledK == baseK/2 && p3Paused && restoredK == baseK

	return &ThermalSheddingReceipt{
		Schema:          ThermalSheddingSchema,
		InitialK:        initialK,
		ThrottledK:      throttledK,
		RestoredK:       restoredK,
		P3TasksPaused:   p3Paused,
		PeakObservedCPU: peakCPU,
		Pass:            pass,
		CompletedAt:     time.Now().UTC(),
	}, nil
}
