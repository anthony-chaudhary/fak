package growthgate

import "testing"

func benchmarkCensus() []Artifact {
	return []Artifact{
		{Path: `C:\work\fak\.dos\metrics\observations.jsonl`, Size: 119546680, ModAgeSec: 3},
		{Path: `./.dos/_dos_park/metrics/observations.jsonl`, Size: 116788783, ModAgeSec: 90000},
		{Path: `./.dispatch-runs/superloop-200-wide.log`, Size: 47019668, ModAgeSec: 500000},
		{Path: `./.dispatch-runs/dispatcher-live.log`, Size: 18000000, ModAgeSec: 120},
		{Path: `./.dos/lane-journal.jsonl`, Size: 22459548, ModAgeSec: 10},
		{Path: `./.fak/loops.jsonl`, Size: 23340033, ModAgeSec: 10},
		{Path: `./.fak/toolproc/journal.jsonl`, Size: 9500000, ModAgeSec: 50},
		{Path: `./.fak/toolproc/old.jsonl`, Size: 30000000, ModAgeSec: 86400},
		{Path: `./.goal-runs/marathon-runner.log`, Size: 68000000, ModAgeSec: 7200},
		{Path: `fleet-runs/nightrun/watch.out.log`, Size: 70000000, ModAgeSec: 30000},
		{Path: `./build/output.log`, Size: 12000000, ModAgeSec: 600},
		{Path: `./build/error.err`, Size: 2000000, ModAgeSec: 600},
		{Path: `./benchmarks/measurements.jsonl`, Size: 35000000, ModAgeSec: 4000},
		{Path: `./fresh-ledger.jsonl`, Size: 1048576, ModAgeSec: 1},
		{Path: `./README.md`, Size: 4096, ModAgeSec: 100},
	}
}

// BenchmarkGrowthGate benchmarks the combined classification and reap planning
// cycle across a realistic census of workspace artifacts.
func BenchmarkGrowthGate(b *testing.B) {
	arts := benchmarkCensus()
	budget := DefaultBudget()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Classify(arts, budget)
		reap, protected := ReapPlan(rep)
		if len(reap) == 0 || len(protected) == 0 {
			b.Fatal("unexpected empty partition")
		}
	}
}

// BenchmarkClassify benchmarks artifact classification and severity sorting.
func BenchmarkClassify(b *testing.B) {
	arts := benchmarkCensus()
	budget := DefaultBudget()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Classify(arts, budget)
		if rep.Verdict != SevAction {
			b.Fatalf("unexpected verdict %v", rep.Verdict)
		}
	}
}

// BenchmarkReapPlan benchmarks partitioning findings into reapable and protected sets.
func BenchmarkReapPlan(b *testing.B) {
	arts := benchmarkCensus()
	budget := DefaultBudget()
	rep := Classify(arts, budget)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reap, protected := ReapPlan(rep)
		if len(reap) == 0 || len(protected) == 0 {
			b.Fatal("unexpected empty partition")
		}
	}
}

// BenchmarkClassifyPath benchmarks path categorization throughput across different path styles.
func BenchmarkClassifyPath(b *testing.B) {
	paths := [...]string{
		`C:\work\fak\.dos\metrics\observations.jsonl`,
		`./.dos/metrics/observations.jsonl`,
		`./.dos/lane-journal.jsonl`,
		`./.dispatch-runs/superloop-200-wide.log`,
		`./.goal-runs/marathon.log`,
		`./.fak/loops.jsonl`,
		`./.fak/toolproc/journal.jsonl`,
		`fleet-runs/nightrun/watch.out.log`,
		`logs/worker.log`,
		`logs/worker.err`,
		`data/records.jsonl`,
		`docs/README.md`,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range paths {
			_ = ClassifyPath(p)
		}
	}
}

func TestBenchmarkGrowthGateSanity(t *testing.T) {
	arts := benchmarkCensus()
	budget := DefaultBudget()
	rep := Classify(arts, budget)
	if rep.Verdict != SevAction {
		t.Fatalf("expected action verdict, got %s", rep.Verdict)
	}
	reap, protected := ReapPlan(rep)
	if len(reap) == 0 {
		t.Fatal("expected reapable findings")
	}
	if len(protected) == 0 {
		t.Fatal("expected protected findings")
	}
}
