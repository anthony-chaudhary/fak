package looprecover

import (
	"fmt"
	"testing"
)

// generateBenchRuns produces a deterministic slice of RunFacts simulating
// diverse production fleet conditions.
func generateBenchRuns(count int, now int64) []RunFact {
	runs := make([]RunFact, count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("run-%05d", i)
		loopID := fmt.Sprintf("loop-tenant-%02d", i%5)
		unit := fmt.Sprintf("issue-%d", 1000+i)
		pid := 10000 + (i % 500)
		start := fmt.Sprintf("boot-%04d", pid)

		switch i % 10 {
		case 0:
			// Witnessed complete
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, Ended: true, Witnessed: true,
				LastEventUnix: now - 3600,
			}
		case 1:
			// Running with recent activity (within 600s stale window)
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started:       true,
				LastEventUnix: now - 120,
			}
		case 2:
			// Confirmed alive worker, running
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, WorkerKnown: true, WorkerLive: true,
				WorkerPID: pid, WorkerStart: start,
				LastEventUnix: now - 300,
			}
		case 3:
			// Confirmed alive worker, but silent past stale window (alive_silent)
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, WorkerKnown: true, WorkerLive: true,
				WorkerPID: pid, WorkerStart: start,
				LastEventUnix: now - 1500,
			}
		case 4:
			// Confirmed dead worker (orphaned immediately)
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, WorkerKnown: true, WorkerLive: false,
				WorkerPID: pid, WorkerStart: start,
				LastEventUnix: now - 100,
			}
		case 5:
			// Silent past stale window without worker info (orphaned by staleness)
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started:       true,
				LastEventUnix: now - 2500,
			}
		case 6:
			// Ended but unwitnessed
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, Ended: true, Witnessed: false,
				LastEventUnix: now - 800,
			}
		case 7:
			// Claimed done but unwitnessed
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, Claimed: true, Witnessed: false,
				LastEventUnix: now - 450,
			}
		case 8:
			// Failed terminal
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, Failed: true,
				LastEventUnix: now - 900,
			}
		case 9:
			// Canceled terminal
			runs[i] = RunFact{
				RunID: id, LoopID: loopID, Unit: unit,
				Started: true, Canceled: true,
				LastEventUnix: now - 950,
			}
		}
	}
	return runs
}

// generateHomogeneousRuns produces a batch of runs all sharing a target state.
func generateHomogeneousRuns(count int, now int64, dispo Disposition) []RunFact {
	runs := make([]RunFact, count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("homo-%05d", i)
		loopID := "loop-homo"
		switch dispo {
		case DispComplete:
			runs[i] = RunFact{
				RunID: id, LoopID: loopID,
				Started: true, Ended: true, Witnessed: true,
				LastEventUnix: now - int64(100+i),
			}
		case DispOrphaned:
			runs[i] = RunFact{
				RunID: id, LoopID: loopID,
				Started: true, WorkerKnown: true, WorkerLive: false,
				LastEventUnix: now - int64(50+i*5),
			}
		default:
			runs[i] = RunFact{
				RunID: id, LoopID: loopID,
				Started:       true,
				LastEventUnix: now - 30,
			}
		}
	}
	return runs
}

// TestBenchmarkFixtureSanity verifies that generateBenchRuns and generateHomogeneousRuns
// construct properly distributed and valid data for benchmarking.
func TestBenchmarkFixtureSanity(t *testing.T) {
	const testNow = 1_000_000
	runs := generateBenchRuns(50, testNow)
	res := Plan(Input{NowUnix: testNow, StaleSeconds: 600, Runs: runs})

	if res.OrphanedCount == 0 || res.UnwitnessedCount == 0 || res.RunningCount == 0 ||
		res.CompleteCount == 0 || res.FailedCount == 0 {
		t.Fatalf("benchmark fixture missing disposition categories: %+v", res)
	}

	if len(res.Recover) != res.OrphanedCount+res.UnwitnessedCount {
		t.Fatalf("worklist length %d does not match orphaned(%d) + unwitnessed(%d)",
			len(res.Recover), res.OrphanedCount, res.UnwitnessedCount)
	}

	homoComplete := Plan(Input{NowUnix: testNow, Runs: generateHomogeneousRuns(20, testNow, DispComplete)})
	if homoComplete.CompleteCount != 20 || len(homoComplete.Recover) != 0 {
		t.Fatalf("expected 20 complete, got %+v", homoComplete)
	}

	homoOrphan := Plan(Input{NowUnix: testNow, Runs: generateHomogeneousRuns(20, testNow, DispOrphaned)})
	if homoOrphan.OrphanedCount != 20 || len(homoOrphan.Recover) != 20 {
		t.Fatalf("expected 20 orphaned, got %+v", homoOrphan)
	}
}

// BenchmarkPlan measures the core recovery planning algorithm across production batch sizes
// and disposition workloads.
func BenchmarkPlan(b *testing.B) {
	const benchNow = 1_000_000
	const stale = 600

	batch10 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateBenchRuns(10, benchNow)}
	batch50 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateBenchRuns(50, benchNow)}
	batch200 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateBenchRuns(200, benchNow)}
	batch1000 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateBenchRuns(1000, benchNow)}
	allComplete50 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateHomogeneousRuns(50, benchNow, DispComplete)}
	allOrphaned50 := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateHomogeneousRuns(50, benchNow, DispOrphaned)}

	b.Run("Batch10_Mixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(batch10)
			if len(res.Runs) != 10 {
				b.Fatal("unexpected run count")
			}
		}
	})

	b.Run("Batch50_Mixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(batch50)
			if len(res.Runs) != 50 {
				b.Fatal("unexpected run count")
			}
		}
	})

	b.Run("Batch200_FleetSweep", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(batch200)
			if len(res.Runs) != 200 {
				b.Fatal("unexpected run count")
			}
		}
	})

	b.Run("Batch1000_BacklogStress", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(batch1000)
			if len(res.Runs) != 1000 {
				b.Fatal("unexpected run count")
			}
		}
	})

	b.Run("AllComplete_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(allComplete50)
			if res.CompleteCount != 50 {
				b.Fatal("unexpected complete count")
			}
		}
	})

	b.Run("AllOrphaned_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Plan(allOrphaned50)
			if res.OrphanedCount != 50 {
				b.Fatal("unexpected orphaned count")
			}
		}
	})
}

// BenchmarkProbe measures the performance of the pure PID liveness and reuse-defense probe.
func BenchmarkProbe(b *testing.B) {
	liveSame := Liveness(func(int) (string, bool) { return "boot-1234", true })
	gone := Liveness(func(int) (string, bool) { return "", false })
	reused := Liveness(func(int) (string, bool) { return "boot-5678", true })

	b.Run("AliveMatching", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Probe(1234, "boot-1234", liveSame)
			if v != ProbeAlive {
				b.Fatal("expected alive")
			}
		}
	})

	b.Run("DeadProcess", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Probe(1234, "boot-1234", gone)
			if v != ProbeDead {
				b.Fatal("expected dead")
			}
		}
	})

	b.Run("ReusedPID", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Probe(1234, "boot-1234", reused)
			if v != ProbeDead {
				b.Fatal("expected dead on reused pid")
			}
		}
	})

	b.Run("UnknownPID", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Probe(0, "boot-1234", liveSame)
			if v != ProbeUnknown {
				b.Fatal("expected unknown")
			}
		}
	})
}

// BenchmarkProbeAndPlanPipeline measures the combined end-to-end operation of probing
// candidate runs for liveness and folding them into a recovery plan.
func BenchmarkProbeAndPlanPipeline(b *testing.B) {
	const benchNow = 1_000_000
	const stale = 600

	baseRuns := generateBenchRuns(50, benchNow)
	// Create an OS liveness lookup simulator where even PIDs are alive with same boot,
	// odd PIDs are dead, and PIDs divisible by 7 have had PID reuse.
	osLive := Liveness(func(pid int) (string, bool) {
		if pid%2 == 1 {
			return "", false
		}
		if pid%7 == 0 {
			return fmt.Sprintf("boot-reused-%d", pid), true
		}
		return fmt.Sprintf("boot-%04d", pid), true
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probed := make([]RunFact, len(baseRuns))
		for j, r := range baseRuns {
			verdict := ProbeRun(r, osLive)
			probed[j] = r.ApplyProbe(verdict)
		}
		res := Plan(Input{NowUnix: benchNow, StaleSeconds: stale, Runs: probed})
		if len(res.Runs) != 50 {
			b.Fatal("unexpected pipeline result size")
		}
	}
}

// BenchmarkPlanParallel measures concurrent throughput of Plan under multi-goroutine load.
func BenchmarkPlanParallel(b *testing.B) {
	const benchNow = 1_000_000
	const stale = 600
	in := Input{NowUnix: benchNow, StaleSeconds: stale, Runs: generateBenchRuns(50, benchNow)}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res := Plan(in)
			if len(res.Runs) != 50 {
				b.Fatal("unexpected run count in parallel execution")
			}
		}
	})
}
