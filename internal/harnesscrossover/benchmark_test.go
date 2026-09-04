package harnesscrossover

import (
	"encoding/json"
	"fmt"
	"testing"
)

func benchmarkStudy() Study {
	tasks := []Task{
		{ID: "task-code-1", Domain: "coding"},
		{ID: "task-legal-1", Domain: "legal"},
		{ID: "task-integ-1", Domain: "integrated"},
		{ID: "task-code-2", Domain: "coding"},
		{ID: "task-legal-2", Domain: "legal"},
		{ID: "task-integ-2", Domain: "integrated"},
	}
	runsNative := []Run{
		{TaskID: "task-code-1", SwitchActions: 1, WrongLayerIncidents: 0, Explanation: Seconds{Value: 12.0, Provenance: "observed"}, ContextTokens: 1400, Provenance: "observed"},
		{TaskID: "task-legal-1", SwitchActions: 2, WrongLayerIncidents: 1, Explanation: Seconds{Value: 24.0, Provenance: "observed"}, ContextTokens: 3200, Provenance: "observed"},
		{TaskID: "task-integ-1", SwitchActions: 2, WrongLayerIncidents: 1, Explanation: Seconds{Value: 30.0, Provenance: "observed"}, ContextTokens: 4800, Provenance: "observed"},
		{TaskID: "task-code-2", SwitchActions: 1, WrongLayerIncidents: 0, Explanation: Seconds{Value: 15.0, Provenance: "observed"}, ContextTokens: 1600, Provenance: "observed"},
		{TaskID: "task-legal-2", SwitchActions: 3, WrongLayerIncidents: 2, Explanation: Seconds{Value: 28.0, Provenance: "observed"}, ContextTokens: 3500, Provenance: "observed"},
		{TaskID: "task-integ-2", SwitchActions: 2, WrongLayerIncidents: 1, Explanation: Seconds{Value: 32.0, Provenance: "observed"}, ContextTokens: 5100, Provenance: "observed"},
	}
	runsContextual := []Run{
		{TaskID: "task-code-1", SwitchActions: 0, WrongLayerIncidents: 0, Explanation: Seconds{Value: 5.0, Provenance: "witnessed"}, ContextTokens: 900, Provenance: "witnessed"},
		{TaskID: "task-legal-1", SwitchActions: 0, WrongLayerIncidents: 0, Explanation: Seconds{Value: 6.0, Provenance: "witnessed"}, ContextTokens: 1200, Provenance: "witnessed"},
		{TaskID: "task-integ-1", SwitchActions: 1, WrongLayerIncidents: 0, Explanation: Seconds{Value: 8.0, Provenance: "witnessed"}, ContextTokens: 1500, Provenance: "witnessed"},
		{TaskID: "task-code-2", SwitchActions: 0, WrongLayerIncidents: 0, Explanation: Seconds{Value: 5.0, Provenance: "witnessed"}, ContextTokens: 950, Provenance: "witnessed"},
		{TaskID: "task-legal-2", SwitchActions: 0, WrongLayerIncidents: 0, Explanation: Seconds{Value: 7.0, Provenance: "witnessed"}, ContextTokens: 1300, Provenance: "witnessed"},
		{TaskID: "task-integ-2", SwitchActions: 1, WrongLayerIncidents: 0, Explanation: Seconds{Value: 9.0, Provenance: "witnessed"}, ContextTokens: 1600, Provenance: "witnessed"},
	}

	return Study{
		Schema: Schema,
		ID:     "crossover-bench-v1",
		Tasks:  tasks,
		Weights: Weights{
			SwitchActionSeconds: 15.0,
			WrongLayerSeconds:   60.0,
			ContextTokenSeconds: 0.002,
		},
		Alternatives: []Alternative{
			{
				ID:   "host-native-profile",
				Kind: "native-profile",
				Documentation: []Source{
					{URL: "https://example.test/docs/native", Retrieved: "2026-09-01", Note: "Native tuned profile"},
				},
				Setup:       Seconds{Value: 120.0, Provenance: "observed"},
				Maintenance: Seconds{Value: 45.0, Provenance: "observed"},
				Runs:        runsNative,
			},
			{
				ID:   "fak-contextual-harness",
				Kind: "contextual-harness",
				Documentation: []Source{
					{URL: "https://example.test/docs/fak", Retrieved: "2026-09-01", Note: "Contextual harness layer"},
				},
				Setup:       Seconds{Value: 360.0, Provenance: "witnessed"},
				Maintenance: Seconds{Value: 90.0, Provenance: "witnessed"},
				Runs:        runsContextual,
			},
		},
	}
}

// BenchmarkHarnessCrossover measures the end-to-end evaluation and break-even computation for a multi-domain study.
func BenchmarkHarnessCrossover(b *testing.B) {
	study := benchmarkStudy()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Evaluate(study)
		if report.Winner == "" || report.Crossover == nil {
			b.Fatal("unexpected empty report winner or crossover")
		}
	}
}

// BenchmarkParse measures JSON deserialization, schema gating, and constraint validation throughput.
func BenchmarkParse(b *testing.B) {
	raw, err := json.Marshal(benchmarkStudy())
	if err != nil {
		b.Fatalf("failed to marshal benchmark study: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		study, err := Parse(raw)
		if err != nil {
			b.Fatalf("unexpected parse error: %v", err)
		}
		if len(study.Tasks) == 0 {
			b.Fatal("unexpected empty tasks")
		}
	}
}

// BenchmarkCrossoverScaling measures evaluation throughput across a scaled task set with frequent domain switches.
func BenchmarkCrossoverScaling(b *testing.B) {
	domains := []string{"coding", "legal", "integrated"}
	taskCount := 60
	tasks := make([]Task, taskCount)
	runsNative := make([]Run, taskCount)
	runsContextual := make([]Run, taskCount)

	for i := 0; i < taskCount; i++ {
		id := fmt.Sprintf("task-%03d", i)
		dom := domains[i%len(domains)]
		tasks[i] = Task{ID: id, Domain: dom}
		runsNative[i] = Run{
			TaskID:              id,
			SwitchActions:       2,
			WrongLayerIncidents: 1,
			Explanation:         Seconds{Value: 20.0, Provenance: "observed"},
			ContextTokens:       2500,
			Provenance:          "observed",
		}
		runsContextual[i] = Run{
			TaskID:              id,
			SwitchActions:       0,
			WrongLayerIncidents: 0,
			Explanation:         Seconds{Value: 5.0, Provenance: "witnessed"},
			ContextTokens:       800,
			Provenance:          "witnessed",
		}
	}

	study := Study{
		Schema: Schema,
		ID:     "scaling-study",
		Tasks:  tasks,
		Weights: Weights{
			SwitchActionSeconds: 15.0,
			WrongLayerSeconds:   60.0,
			ContextTokenSeconds: 0.002,
		},
		Alternatives: []Alternative{
			{
				ID:            "native",
				Kind:          "native-profile",
				Documentation: []Source{{URL: "https://example.test/native"}},
				Setup:         Seconds{Value: 100.0, Provenance: "observed"},
				Maintenance:   Seconds{Value: 50.0, Provenance: "observed"},
				Runs:          runsNative,
			},
			{
				ID:            "contextual",
				Kind:          "contextual-harness",
				Documentation: []Source{{URL: "https://example.test/contextual"}},
				Setup:         Seconds{Value: 500.0, Provenance: "witnessed"},
				Maintenance:   Seconds{Value: 120.0, Provenance: "witnessed"},
				Runs:          runsContextual,
			},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Evaluate(study)
		if report.Crossover == nil {
			b.Fatal("expected non-nil crossover")
		}
	}
}

// TestBenchmarkHarnessCrossoverSanity verifies that BenchmarkHarnessCrossover executes without error.
func TestBenchmarkHarnessCrossoverSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessCrossover)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
