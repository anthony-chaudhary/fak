package director_test

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/director"
)

var (
	benchDigestSink DirectorDigestSinkHolder
	benchRecsSink   []director.SteeringRecommendation
	benchWorkerSink director.WorkerDigestRow
)

type DirectorDigestSinkHolder struct {
	digest director.DirectorDigest
}

// TestBenchmarkOperationsSanity verifies that all benchmarked production paths execute
// cleanly and adhere to closed steering contracts before performance measurement loops run.
func TestBenchmarkOperationsSanity(t *testing.T) {
	// 1. Sanity check DirectorEvaluate
	digest := buildBenchmarkDigest(32)
	recs := director.EvaluateFleetSteering(digest)
	if len(recs) == 0 {
		t.Fatalf("expected non-empty recommendations from benchmark digest")
	}

	// 2. Sanity check StallTimeoutAutomaticTransition
	engine := director.NewRollupEngine()
	engine.SetStallTimeoutMs(500)
	now := int64(1700000000000)
	engine.SetNowFunc(func() int64 { return now })
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:         "RID-SANITY",
		Lane:          "gateway",
		State:         director.WorkerHealthy,
		LastWitnessMs: now - 1000, // 1000ms old > 500ms timeout
	})
	d := engine.CompileDigest()
	if d.StalledWorkers != 1 || d.Workers[0].State != director.WorkerStalled {
		t.Fatalf("expected worker to automatically transition to stalled: %+v", d.Workers)
	}

	// 3. Sanity check CompileAndEvaluate
	d2, recs2 := engine.CompileAndEvaluate()
	if d2.TotalWorkers != 1 || len(recs2) == 0 {
		t.Fatalf("expected valid digest and recs from CompileAndEvaluate")
	}
}

func buildBenchmarkDigest(numWorkers int) director.DirectorDigest {
	now := int64(1700000000000)
	workers := make([]director.WorkerDigestRow, numWorkers)
	for i := 0; i < numWorkers; i++ {
		runID := fmt.Sprintf("RID-%03d", i)
		lane := fmt.Sprintf("lane-%d", i%8)
		issue := fmt.Sprintf("#%d", 10000+i)

		var state director.WorkerState
		var stepCount, verifiedCommits, treeTouches int
		var velocity float64

		switch i % 8 {
		case 0, 1, 2, 3:
			state = director.WorkerHealthy
			stepCount = 15
			verifiedCommits = 2
			treeTouches = 4
			velocity = 2.0
		case 4:
			state = director.WorkerStalled
			stepCount = 8
			verifiedCommits = 0
			treeTouches = 2
			velocity = 0.0
		case 5:
			state = director.WorkerBlocked
			stepCount = 12
			verifiedCommits = 0
			treeTouches = 3
			velocity = 0.0
		case 6:
			// Runaway thrashing worker
			state = director.WorkerHealthy
			stepCount = 60
			verifiedCommits = 0
			treeTouches = 15
			velocity = 0.0
		case 7:
			// Tree expansion needed (widen)
			state = director.WorkerHealthy
			stepCount = 35
			verifiedCommits = 0
			treeTouches = 30
			velocity = 0.0
		}

		workers[i] = director.WorkerDigestRow{
			RunID:           runID,
			Lane:            lane,
			Issue:           issue,
			State:           state,
			StepCount:       stepCount,
			VerifiedCommits: verifiedCommits,
			TreeTouches:     treeTouches,
			VelocityScore:   velocity,
			LastWitnessMs:   now,
		}
	}

	leases := make([]director.LeaseSnapshot, 10)
	for i := 0; i < 10; i++ {
		lane := fmt.Sprintf("lane-%d", i)
		holder := fmt.Sprintf("RID-%03d", i)
		if i == 9 {
			holder = "" // Idle lease triggers ActionSpawn
		}
		leases[i] = director.LeaseSnapshot{
			Lane:     lane,
			LaneKind: director.LaneKindCluster,
			Tree:     []string{"internal/" + lane + "/**"},
			Holder:   holder,
			Mode:     director.LeaseModeExclusive,
		}
	}

	totalCommits := 16
	commitsPerHour := 8.0
	blockRate := 0.125
	stallRate := 0.125

	d := director.DirectorDigest{
		Schema:           director.DigestSchema,
		Timestamp:        now,
		TotalWorkers:     numWorkers,
		ActiveWorkers:    numWorkers / 2,
		StalledWorkers:   numWorkers / 8,
		CompletedWorkers: 0,
		FleetVelocity: director.FleetVelocityScore{
			TotalCommits:   totalCommits,
			CommitsPerHour: commitsPerHour,
			BlockRate:      blockRate,
			StallRate:      stallRate,
		},
		Workers: workers,
		Leases:  leases,
	}
	return d
}

// BenchmarkDirectorEvaluate measures evaluation of closed supervisor steering
// recommendations across multi-agent worker health states and active lane leases.
func BenchmarkDirectorEvaluate(b *testing.B) {
	digest := buildBenchmarkDigest(64)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRecsSink = director.EvaluateFleetSteering(digest)
	}
}

// BenchmarkHighConcurrencyRollupEngine measures in-process RollupEngine throughput
// under concurrent reader and writer contention across 60+ workers.
func BenchmarkHighConcurrencyRollupEngine(b *testing.B) {
	engine := director.NewRollupEngine()
	const numWorkers = 64

	// Pre-seed workers and leases
	for i := 0; i < numWorkers; i++ {
		runID := fmt.Sprintf("RID-%03d", i)
		lane := fmt.Sprintf("lane-%d", i%8)
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           runID,
			Lane:            lane,
			Issue:           fmt.Sprintf("#%d", 10000+i),
			State:           director.WorkerHealthy,
			StepCount:       10,
			VerifiedCommits: 1,
			TreeTouches:     2,
			VelocityScore:   1.0,
			LastWitnessMs:   time.Now().UnixMilli(),
		})
		engine.RecordLease(director.LeaseSnapshot{
			Lane:     lane,
			LaneKind: director.LaneKindCluster,
			Tree:     []string{"internal/" + lane + "/**"},
			Holder:   runID,
			Mode:     director.LeaseModeExclusive,
		})
	}

	var stop int32
	var bgWg sync.WaitGroup
	numBgWorkers := 8

	for g := 0; g < numBgWorkers; g++ {
		bgWg.Add(1)
		go func(gid int) {
			defer bgWg.Done()
			step := 0
			for atomic.LoadInt32(&stop) == 0 {
				step++
				workerIdx := (step + gid) % numWorkers
				runID := fmt.Sprintf("RID-%03d", workerIdx)
				lane := fmt.Sprintf("lane-%d", workerIdx%8)
				if gid%2 == 0 {
					engine.RecordWorker(director.WorkerDigestRow{
						RunID:           runID,
						Lane:            lane,
						Issue:           fmt.Sprintf("#%d", 10000+workerIdx),
						State:           director.WorkerHealthy,
						StepCount:       step,
						VerifiedCommits: step / 5,
						TreeTouches:     step % 5,
						VelocityScore:   1.2,
						LastWitnessMs:   time.Now().UnixMilli(),
					})
				} else {
					_ = engine.CompileDigest()
				}
				runtime.Gosched()
			}
		}(g)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		switch i % 4 {
		case 0:
			workerIdx := i % numWorkers
			engine.RecordWorker(director.WorkerDigestRow{
				RunID:           fmt.Sprintf("RID-%03d", workerIdx),
				Lane:            fmt.Sprintf("lane-%d", workerIdx%8),
				Issue:           fmt.Sprintf("#%d", 10000+workerIdx),
				State:           director.WorkerHealthy,
				StepCount:       i,
				VerifiedCommits: i / 10,
				TreeTouches:     i % 4,
				VelocityScore:   1.5,
				LastWitnessMs:   time.Now().UnixMilli(),
			})
		case 1:
			benchDigestSink.digest = engine.CompileDigest()
		case 2:
			benchRecsSink = engine.EvaluateFleetSteering(benchDigestSink.digest)
		case 3:
			d, recs := engine.CompileAndEvaluate()
			benchDigestSink.digest = d
			benchRecsSink = recs
		}
	}

	atomic.StoreInt32(&stop, 1)
	bgWg.Wait()
}

// BenchmarkStallTimeoutAutomaticTransition measures the automatic transition latency
// and throughput when worker silence exceeds configured StallTimeoutMs.
func BenchmarkStallTimeoutAutomaticTransition(b *testing.B) {
	engine := director.NewRollupEngine()
	engine.SetStallTimeoutMs(500)
	var nowMs int64 = 1700000000000
	engine.SetNowFunc(func() int64 { return atomic.LoadInt64(&nowMs) })

	const numWorkers = 48
	for i := 0; i < numWorkers; i++ {
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           fmt.Sprintf("RID-%03d", i),
			Lane:            fmt.Sprintf("lane-%d", i%6),
			Issue:           fmt.Sprintf("#%d", 10000+i),
			State:           director.WorkerHealthy,
			StepCount:       10,
			VerifiedCommits: 1,
			TreeTouches:     2,
			VelocityScore:   1.0,
			LastWitnessMs:   1700000000000,
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			atomic.StoreInt64(&nowMs, 1700000000000+1000) // 1000ms elapsed > 500ms -> stalled
		} else {
			atomic.StoreInt64(&nowMs, 1700000000000+200)  // 200ms elapsed < 500ms -> active
		}
		d, recs := engine.CompileAndEvaluate()
		benchDigestSink.digest = d
		benchRecsSink = recs
	}
}

// BenchmarkCompileDigest measures compiling zero-self-report DirectorDigests
// including worker sorting, fleet velocity scoring, and cryptographic rollup hash.
func BenchmarkCompileDigest(b *testing.B) {
	engine := director.NewRollupEngine()
	const numWorkers = 50

	for i := 0; i < numWorkers; i++ {
		runID := fmt.Sprintf("RID-%03d", i)
		lane := fmt.Sprintf("lane-%d", i%8)
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           runID,
			Lane:            lane,
			Issue:           fmt.Sprintf("#%d", 10000+i),
			State:           director.WorkerHealthy,
			StepCount:       20,
			VerifiedCommits: 3,
			TreeTouches:     5,
			VelocityScore:   2.5,
			LastWitnessMs:   time.Now().UnixMilli(),
		})
		engine.RecordLease(director.LeaseSnapshot{
			Lane:     lane,
			LaneKind: director.LaneKindCluster,
			Tree:     []string{"internal/" + lane + "/**"},
			Holder:   runID,
			Mode:     director.LeaseModeExclusive,
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDigestSink.digest = engine.CompileDigest()
	}
}

// BenchmarkCompileAndEvaluate measures the combined digest compilation and
// supervisor steering evaluation pipeline in a single call.
func BenchmarkCompileAndEvaluate(b *testing.B) {
	engine := director.NewRollupEngine()
	const numWorkers = 50

	for i := 0; i < numWorkers; i++ {
		runID := fmt.Sprintf("RID-%03d", i)
		lane := fmt.Sprintf("lane-%d", i%8)
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           runID,
			Lane:            lane,
			Issue:           fmt.Sprintf("#%d", 10000+i),
			State:           director.WorkerHealthy,
			StepCount:       15,
			VerifiedCommits: 2,
			TreeTouches:     3,
			VelocityScore:   1.8,
			LastWitnessMs:   time.Now().UnixMilli(),
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d, recs := engine.CompileAndEvaluate()
		benchDigestSink.digest = d
		benchRecsSink = recs
	}
}

// BenchmarkRecordWorker measures worker state ingestion and registration throughput.
func BenchmarkRecordWorker(b *testing.B) {
	engine := director.NewRollupEngine()
	const numWorkers = 64

	row := director.WorkerDigestRow{
		RunID:           "RID-BENCH",
		Lane:            "core",
		Issue:           "#11411",
		State:           director.WorkerHealthy,
		StepCount:       10,
		VerifiedCommits: 1,
		TreeTouches:     2,
		VelocityScore:   1.0,
		LastWitnessMs:   time.Now().UnixMilli(),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row.StepCount = i
		row.RunID = fmt.Sprintf("RID-%03d", i%numWorkers)
		engine.RecordWorker(row)
	}
}
