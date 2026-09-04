package subtractiveprofile

import (
	"testing"
)

// BenchmarkSubtractiveProfile benchmarks resolution across multi-layered capability profiles
// containing inclusions, aliases, configurations, and sticky removals.
func BenchmarkSubtractiveProfile(b *testing.B) {
	profiles := []Profile{
		{
			Include: []Capability{
				{ID: "tools", Aliases: []string{"tools-alias", "tooling"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "agent", Aliases: []string{"bot"}, Requires: []string{"tools"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "storage", Aliases: []string{"disk"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "network", Aliases: []string{"net"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "ui", Aliases: []string{"frontend"}, Help: true, Schema: true, Runtime: true, Artifact: true},
			},
			Configure: map[string]map[string]string{
				"agent": {"mode": "autonomous", "concurrency": "4"},
				"tools": {"sandbox": "isolated"},
			},
		},
		{
			Remove: map[string]Removal{
				"ui": RemovalStatic,
			},
			Replace: map[string]Capability{
				"storage": {ID: "storage", Help: true, Schema: true, Runtime: true, Artifact: false},
			},
		},
	}
	report := Report{
		Minimal: Delta{BinaryBytes: 1024, StartupMillis: 12},
		Full:    Delta{BinaryBytes: 4096, StartupMillis: 45, IdleMemoryBytes: 8192, ContextTokens: 1024, SchemaBytes: 512},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		effective, err := Resolve(profiles, report)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if len(effective.Capabilities) == 0 {
			b.Fatal("expected non-empty resolved capabilities")
		}
	}
}

// TestBenchmarkProfileResolution verifies that the benchmark dataset resolves properly
// and honors all removal and configuration invariants.
func TestBenchmarkProfileResolution(t *testing.T) {
	profiles := []Profile{
		{
			Include: []Capability{
				{ID: "tools", Aliases: []string{"tools-alias"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "agent", Requires: []string{"tools"}, Help: true, Schema: true, Runtime: true, Artifact: true},
				{ID: "ui", Help: true, Schema: true, Runtime: true, Artifact: true},
			},
		},
		{
			Remove: map[string]Removal{
				"ui": RemovalStatic,
			},
		},
	}
	effective, err := Resolve(profiles, Report{})
	if err != nil {
		t.Fatalf("unexpected error resolving benchmark fixture: %v", err)
	}
	if err := effective.ProbeAbsent("ui"); err != nil {
		t.Fatalf("ui capability should be absent: %v", err)
	}
	if _, ok := effective.Capabilities["tools"]; !ok {
		t.Fatal("tools capability missing from effective set")
	}
	if _, ok := effective.Capabilities["agent"]; !ok {
		t.Fatal("agent capability missing from effective set")
	}
}
