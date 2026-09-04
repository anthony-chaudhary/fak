package goalregistry

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

// BenchmarkGoalRegistry measures end-to-end lookup and resolution throughput across pre-populated goals.
func BenchmarkGoalRegistry(b *testing.B) {
	dir := b.TempDir()
	store := Store{
		Path: filepath.Join(dir, "goals.json"),
		Now:  func() time.Time { return time.Unix(1700000000, 0).UTC() },
	}
	p := Provenance{Actor: "benchmark-runner", Authority: "automated-test"}

	seedGoal, err := store.Create("Primary benchmark goal", "Durable intent under benchmark load", p, nil)
	if err != nil {
		b.Fatalf("seed goal creation failed: %v", err)
	}

	for i := 0; i < 10; i++ {
		extID := fmt.Sprintf("bench-obj-%d", i)
		if _, err := store.Bind(seedGoal.GoalID, "fak:trajctl", extID, "", p); err != nil {
			b.Fatalf("seed binding failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extID := fmt.Sprintf("bench-obj-%d", i%10)
		goal, binding, err := store.Resolve("fak:trajctl", extID, "")
		if err != nil {
			b.Fatalf("resolve failed: %v", err)
		}
		if goal.GoalID != seedGoal.GoalID || binding.ExternalID != extID {
			b.Fatalf("unexpected resolve mismatch")
		}
	}
}

// BenchmarkGoalRegistryCreate measures the throughput of creating and persisting new canonical goals.
func BenchmarkGoalRegistryCreate(b *testing.B) {
	dir := b.TempDir()
	store := Store{
		Path: filepath.Join(dir, "goals.json"),
		Now:  func() time.Time { return time.Unix(1700000000, 0).UTC() },
	}
	p := Provenance{Actor: "benchmark-runner", Authority: "automated-test"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := store.Create(fmt.Sprintf("Benchmark intent %d", i), "Synthetic benchmark workload", p, nil)
		if err != nil {
			b.Fatalf("create failed: %v", err)
		}
	}
}

// BenchmarkGoalRegistryShow measures goal inspection and binding aggregation throughput.
func BenchmarkGoalRegistryShow(b *testing.B) {
	dir := b.TempDir()
	store := Store{
		Path: filepath.Join(dir, "goals.json"),
		Now:  func() time.Time { return time.Unix(1700000000, 0).UTC() },
	}
	p := Provenance{Actor: "benchmark-runner", Authority: "automated-test"}

	goal, err := store.Create("Inspection target", "Goal for show benchmarking", p, nil)
	if err != nil {
		b.Fatalf("setup goal failed: %v", err)
	}
	for i := 0; i < 5; i++ {
		if _, err := store.Bind(goal.GoalID, fmt.Sprintf("ns-%d", i), "id-1", "", p); err != nil {
			b.Fatalf("setup binding failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g, bindings, err := store.Show(goal.GoalID)
		if err != nil {
			b.Fatalf("show failed: %v", err)
		}
		if g.GoalID != goal.GoalID || len(bindings) != 5 {
			b.Fatalf("unexpected show result")
		}
	}
}

// BenchmarkGoalRegistryTransition measures lifecycle state advancement with typed outcome evidence.
func BenchmarkGoalRegistryTransition(b *testing.B) {
	dir := b.TempDir()
	store := Store{
		Path: filepath.Join(dir, "goals.json"),
		Now:  func() time.Time { return time.Unix(1700000000, 0).UTC() },
	}
	p := Provenance{Actor: "benchmark-runner", Authority: "automated-test"}

	goal, err := store.Create("Transition goal", "Goal for lifecycle transitions", p, nil)
	if err != nil {
		b.Fatalf("setup goal failed: %v", err)
	}

	ev := OutcomeEvidence{
		Class:     IndependentWitness,
		Author:    "bench-witness",
		Reference: "proof-ref-1",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g, err := store.Transition(goal.GoalID, Achieved, ev)
		if err != nil {
			b.Fatalf("transition failed: %v", err)
		}
		if g.Lifecycle != Achieved {
			b.Fatalf("unexpected lifecycle: %s", g.Lifecycle)
		}

		g, err = store.Reopen(goal.GoalID, "operator", "bench-reopen")
		if err != nil {
			b.Fatalf("reopen failed: %v", err)
		}
		if g.Lifecycle != Active {
			b.Fatalf("unexpected reopen lifecycle: %s", g.Lifecycle)
		}
	}
}

// BenchmarkGoalRegistryLoad measures deserialization and schema validation of persisted registry state.
func BenchmarkGoalRegistryLoad(b *testing.B) {
	dir := b.TempDir()
	store := Store{
		Path: filepath.Join(dir, "goals.json"),
		Now:  func() time.Time { return time.Unix(1700000000, 0).UTC() },
	}
	p := Provenance{Actor: "benchmark-runner", Authority: "automated-test"}

	for i := 0; i < 20; i++ {
		g, err := store.Create(fmt.Sprintf("Preloaded goal %d", i), "State dataset", p, nil)
		if err != nil {
			b.Fatalf("preload create failed: %v", err)
		}
		if _, err := store.Bind(g.GoalID, "fak:trajctl", fmt.Sprintf("obj-%d", i), "", p); err != nil {
			b.Fatalf("preload bind failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reg, err := store.Load()
		if err != nil {
			b.Fatalf("load failed: %v", err)
		}
		if len(reg.Goals) != 20 || len(reg.Bindings) != 20 {
			b.Fatalf("unexpected registry size: %d goals, %d bindings", len(reg.Goals), len(reg.Bindings))
		}
	}
}
