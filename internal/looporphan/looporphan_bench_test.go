package looporphan

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchReportSink Report
	benchPIDsSink   []int
)

func makeSyntheticCensus(count int, lanesCount int) []Supervisor {
	census := make([]Supervisor, count)
	for i := 0; i < count; i++ {
		lane := fmt.Sprintf("lane-%03d", i%lanesCount)
		parent := ParentDead
		liveWorkers := 0
		if i%5 == 0 {
			liveWorkers = 1
			parent = ParentAlive
		} else if i%3 == 0 {
			parent = ParentAlive
		} else if i%7 == 0 {
			parent = ParentUnknown
		}

		start := fmt.Sprintf("start-%06d", 100000+i)
		if i%11 == 0 {
			start = "" // unfenced
		}

		census[i] = Supervisor{
			PID:         10000 + i,
			PPID:        1000 + (i % 10),
			Start:       start,
			Cmdline:     fmt.Sprintf("fak loop drive --lane %s --session %d", lane, i),
			Lane:        lane,
			Parent:      parent,
			LiveWorkers: liveWorkers,
		}
	}
	return census
}

func makeSyntheticCmdlineCensus(count int, groupsCount int) []Supervisor {
	census := make([]Supervisor, count)
	for i := 0; i < count; i++ {
		group := fmt.Sprintf("cmdline-group-%03d", i%groupsCount)
		parent := ParentDead
		liveWorkers := 0
		if i%5 == 0 {
			liveWorkers = 1
			parent = ParentAlive
		}
		census[i] = Supervisor{
			PID:         20000 + i,
			PPID:        2000 + (i % 10),
			Start:       fmt.Sprintf("start-%06d", 200000+i),
			Cmdline:     fmt.Sprintf("fak loop drive --region %s", group),
			Lane:        "",
			Parent:      parent,
			LiveWorkers: liveWorkers,
		}
	}
	return census
}

func TestBenchmarkFixtures(t *testing.T) {
	c1 := makeSyntheticCensus(50, 5)
	if len(c1) != 50 {
		t.Fatalf("expected 50 supervisors, got %d", len(c1))
	}
	rep1 := Plan(c1, DefaultConfig())
	if len(rep1.Verdicts) != 50 {
		t.Fatalf("expected 50 verdicts, got %d", len(rep1.Verdicts))
	}
	if rep1.Keep == 0 || rep1.Reap == 0 {
		t.Fatalf("expected non-zero keep (%d) and reap (%d)", rep1.Keep, rep1.Reap)
	}

	c2 := makeSyntheticCmdlineCensus(30, 3)
	if len(c2) != 30 {
		t.Fatalf("expected 30 supervisors, got %d", len(c2))
	}
	rep2 := Plan(c2, DefaultConfig())
	if len(rep2.Verdicts) != 30 {
		t.Fatalf("expected 30 verdicts, got %d", len(rep2.Verdicts))
	}
}

func TestPlanAllocationBudget(t *testing.T) {
	census := makeSyntheticCensus(50, 10)
	cfg := DefaultConfig()

	allocs := testing.AllocsPerRun(100, func() {
		rep := Plan(census, cfg)
		if len(rep.Verdicts) != 50 {
			t.Fatalf("expected 50 verdicts, got %d", len(rep.Verdicts))
		}
	})

	t.Logf("Plan 50 supervisors allocations per run: %.1f", allocs)
	// Plan creates group maps, slices, sort closures, and verdicts.
	// Bound allocations to <= 2.5 allocs per supervisor (<= 125 allocs for 50 supervisors).
	allocsPerSup := allocs / float64(len(census))
	if allocsPerSup > 2.5 {
		t.Fatalf("allocations per supervisor %.2f exceeds budget 2.5 (total %.1f)", allocsPerSup, allocs)
	}
}

func TestPlanThroughputLinear(t *testing.T) {
	censusSmall := makeSyntheticCensus(50, 10)
	censusLarge := makeSyntheticCensus(500, 50)
	cfg := DefaultConfig()

	// Warmup
	for i := 0; i < 5; i++ {
		_ = Plan(censusSmall, cfg)
		_ = Plan(censusLarge, cfg)
	}

	const iters = 100
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		_ = Plan(censusSmall, cfg)
	}
	durSmall := time.Since(t0)

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		_ = Plan(censusLarge, cfg)
	}
	durLarge := time.Since(t1)

	ratio := float64(durLarge) / float64(durSmall)
	t.Logf("runtime ratio 500/50 supervisors (10x workload): %.2fx (durSmall=%v, durLarge=%v)", ratio, durSmall, durLarge)
	// Sorting 500 vs 50 items is O(N log N). Ratio should scale comfortably under 30x.
	if ratio > 30.0 {
		t.Fatalf("throughput scales super-linearly: 10x workload took %.2fx time", ratio)
	}
}

func BenchmarkPlan_RealisticWorkstation(b *testing.B) {
	census := makeSyntheticCensus(15, 4)
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Plan(census, cfg)
		if len(rep.Verdicts) != 15 {
			b.Fatalf("expected 15 verdicts, got %d", len(rep.Verdicts))
		}
		benchReportSink = rep
	}
}

func BenchmarkPlan_CensusScaling(b *testing.B) {
	sizes := []struct {
		name        string
		supervisors int
		lanes       int
	}{
		{name: "10_procs", supervisors: 10, lanes: 3},
		{name: "50_procs", supervisors: 50, lanes: 10},
		{name: "200_procs", supervisors: 200, lanes: 25},
		{name: "1000_procs", supervisors: 1000, lanes: 100},
	}

	cfg := DefaultConfig()

	for _, tc := range sizes {
		census := makeSyntheticCensus(tc.supervisors, tc.lanes)
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := Plan(census, cfg)
				benchReportSink = rep
			}
			b.StopTimer()

			nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
			nsPerSupervisor := nsPerOp / float64(tc.supervisors)
			procsPerSec := float64(tc.supervisors*b.N) / b.Elapsed().Seconds()
			b.ReportMetric(procsPerSec, "supervisors/s")
			b.ReportMetric(nsPerSupervisor, "ns/supervisor")
		})
	}
}

func BenchmarkPlan_HighDuplication(b *testing.B) {
	// 100 duplicate supervisors on a single lane: 1 live keeper, 99 idle duplicates.
	census := make([]Supervisor, 100)
	for i := 0; i < 100; i++ {
		live := 0
		parent := ParentDead
		if i == 0 {
			live = 1
			parent = ParentAlive
		}
		census[i] = Supervisor{
			PID:         50000 + i,
			PPID:        5000,
			Start:       fmt.Sprintf("start-%d", i),
			Lane:        "contested-lane",
			Parent:      parent,
			LiveWorkers: live,
		}
	}
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Plan(census, cfg)
		if rep.Keep != 1 || rep.Reap != 99 {
			b.Fatalf("unexpected verdict counts: keep=%d reap=%d", rep.Keep, rep.Reap)
		}
		benchReportSink = rep
	}
}

func BenchmarkPlan_CollisionSprawl(b *testing.B) {
	// 50 supervisors on a single lane where 10 are live workers (COLLISION).
	census := make([]Supervisor, 50)
	for i := 0; i < 50; i++ {
		live := 0
		if i < 10 {
			live = 1
		}
		census[i] = Supervisor{
			PID:         60000 + i,
			PPID:        6000,
			Start:       fmt.Sprintf("start-%d", i),
			Lane:        "colliding-lane",
			Parent:      ParentDead,
			LiveWorkers: live,
		}
	}
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Plan(census, cfg)
		if rep.Collision != 10 || rep.Reap != 40 {
			b.Fatalf("unexpected verdict counts: collision=%d reap=%d", rep.Collision, rep.Reap)
		}
		benchReportSink = rep
	}
}

func BenchmarkPlan_WideDisjointLanes(b *testing.B) {
	// 200 supervisors, each on a unique lane.
	census := makeSyntheticCensus(200, 200)
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Plan(census, cfg)
		benchReportSink = rep
	}
}

func BenchmarkPlan_CmdlineFallback(b *testing.B) {
	census := makeSyntheticCmdlineCensus(100, 10)
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Plan(census, cfg)
		benchReportSink = rep
	}
}

func BenchmarkReport_ReapPIDs(b *testing.B) {
	sizes := []int{20, 100, 1000}
	for _, size := range sizes {
		census := makeSyntheticCensus(size, size/5)
		rep := Plan(census, DefaultConfig())

		b.Run(fmt.Sprintf("%d_verdicts", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pids := rep.ReapPIDs()
				benchPIDsSink = pids
			}
		})
	}
}

func BenchmarkPlan_Parallel(b *testing.B) {
	census := makeSyntheticCensus(50, 10)
	cfg := DefaultConfig()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			rep := Plan(census, cfg)
			benchReportSink = rep
		}
	})
}
