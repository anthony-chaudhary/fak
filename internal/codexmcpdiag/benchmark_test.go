package codexmcpdiag

import (
	"fmt"
	"testing"
)

// BenchmarkDiagnosticRun benchmarks evaluation throughput across multiple server instances
// and synthetic log event streams with mixed ready, error, and cancellation telemetry.
func BenchmarkDiagnosticRun(b *testing.B) {
	names := []string{"codex_apps", "dos", "fak", "openaiDeveloperDocs", "context_mmu", "rulesynth"}
	events := make([]Event, 0, len(names)*20)
	for idx, name := range names {
		for j := 0; j < 15; j++ {
			events = append(events, Event{
				Level:  "INFO",
				Target: "codex_core::mcp::bootstrap",
				Body:   fmt.Sprintf("processing configuration entry %d for component %s", j, name),
			})
		}
		switch idx % 3 {
		case 0:
			events = append(events, Event{
				Level:  "INFO",
				Target: "codex_core::mcp",
				Body:   fmt.Sprintf("MCP client for `%s` initialized successfully", name),
			})
		case 1:
			events = append(events, Event{
				Level:  "ERROR",
				Target: "codex_core::mcp",
				Body:   fmt.Sprintf("server `%s` failed to initialize: timeout after 30s", name),
			})
		case 2:
			events = append(events, Event{
				Level:  "WARN",
				Target: "codex_core::mcp",
				Body:   fmt.Sprintf("server `%s` startup cancelled during runtime refresh", name),
			})
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Classify(names, events)
		if rep.Verdict == "" || len(rep.Servers) != len(names) {
			b.Fatalf("invalid report produced: %+v", rep)
		}
	}
}

// TestBenchmarkDiagnosticRunSanity ensures the benchmark workload executes correctly
// and classifies expected mixed verdicts under unit testing.
func TestBenchmarkDiagnosticRunSanity(t *testing.T) {
	names := []string{"codex_apps", "fak"}
	events := []Event{
		{Level: "INFO", Target: "mcp", Body: "codex_apps initialized"},
		{Level: "ERROR", Target: "mcp", Body: "fak failed to initialize"},
	}
	rep := Classify(names, events)
	if rep.Verdict != VerdictServerFailure {
		t.Fatalf("expected VerdictServerFailure, got %s", rep.Verdict)
	}
	if len(rep.Servers) != 2 {
		t.Fatalf("expected 2 servers, got %d", len(rep.Servers))
	}
}
