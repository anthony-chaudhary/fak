package harnessinspect

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

func benchmarkLock() harnessresolve.Lock {
	return harnessresolve.Lock{
		Schema:      harnessresolve.LockSchema,
		ID:          "prod-agent-harness-lock-4f2a",
		Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "contract-v2"},
		Budget:      harnessresolve.Budget{ContextTokens: 131072, MemoryMiB: 16384, Workers: 8},
		Components: []harnessresolve.LockedComponent{
			{ID: "telemetry-agent", Version: "v1.0.0", Source: "manifest", Reason: "root component", Provides: []string{"metrics", "tracing"}},
			{ID: "planner-core", Version: "v1.2.0", Source: "manifest", Reason: "root component", Provides: []string{"plan", "decompose"}},
			{ID: "policy-guard", Version: "v1.4.2", Source: "layer:security", Reason: "security boundary", Provides: []string{"guard", "audit"}},
			{ID: "memory-store", Version: "v0.9.0", Source: "layer:state", Reason: "dependency", Provides: []string{"kv", "vector"}},
			{ID: "execution-engine", Version: "v2.0.1", Source: "layer:exec", Reason: "dependency", Provides: []string{"exec", "sandbox"}},
		},
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "model", ID: "qwen-coder", Value: "14b-instruct", Mandatory: true, Source: "manifest"},
			{Kind: "policy", ID: "readonly-tools", Ref: "policies/readonly.json", Locked: true, Source: "layer:security"},
			{Kind: "memory", ID: "kv-cache", Boundary: "8GiB", Source: "layer:state"},
			{Kind: "tool", ID: "filesystem", Value: "chroot:/workspace", Source: "manifest", Grants: []string{"read", "stat"}, Denies: []string{"write", "exec"}},
			{Kind: "tool", ID: "network", Boundary: "egress-allowlist", Source: "layer:security", Denies: []string{"inbound"}},
			{Kind: "instruction", ID: "system-prompt", Value: "concise operator mode", Source: "manifest"},
			{Kind: "prompt", ID: "safety-preamble", Ref: "prompts/preamble.txt", Source: "layer:security"},
			{Kind: "context", ID: "window-budget", Boundary: "128k", Source: "manifest"},
		},
		Decisions: []stackresolve.Decision{
			{From: "manifest", Chosen: "planner-core"},
			{From: "manifest", Chosen: "telemetry-agent"},
			{From: "layer:exec", Chosen: "execution-engine"},
			{From: "layer:state", Chosen: "memory-store"},
			{From: "layer:security", Chosen: "policy-guard"},
		},
	}
}

func scaleLock(n int) harnessresolve.Lock {
	components := make([]harnessresolve.LockedComponent, n)
	assets := make([]harnesscompose.EffectiveAsset, n)
	decisions := make([]stackresolve.Decision, n)
	for i := 0; i < n; i++ {
		// Interleave names so sorting work is non-trivial.
		key := fmt.Sprintf("comp-%04d", (i*37)%n)
		components[i] = harnessresolve.LockedComponent{
			ID:       key,
			Version:  "v1.0.0",
			Source:   "manifest",
			Reason:   "scale component",
			Provides: []string{"cap-" + key},
		}
		assetKey := fmt.Sprintf("asset-%04d", (i*37)%n)
		assets[i] = harnesscompose.EffectiveAsset{
			Kind:      "tool",
			ID:        assetKey,
			Value:     "/path/" + assetKey,
			Mandatory: i%3 == 0,
			Locked:    i%3 == 1,
			Source:    "manifest",
			Grants:    []string{"grant-" + key},
			Denies:    []string{"deny-" + key},
		}
		decisions[i] = stackresolve.Decision{
			From:   "manifest",
			Chosen: key,
		}
	}
	return harnessresolve.Lock{
		Schema:      harnessresolve.LockSchema,
		ID:          "scale-lock",
		Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "scale"},
		Budget:      harnessresolve.Budget{ContextTokens: 65536, MemoryMiB: 8192, Workers: 4},
		Components:  components,
		Assets:      assets,
		Decisions:   decisions,
	}
}

// BenchmarkHarnessInspect benchmarks projecting, categorizing, and sorting a resolved product lock.
func BenchmarkHarnessInspect(b *testing.B) {
	lock := benchmarkLock()
	const lockPath = "locks/prod-agent.lock.json"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Inspect(lock, lockPath)
		if len(report.Components) == 0 {
			b.Fatal("unexpected empty components")
		}
	}
}

// BenchmarkRender benchmarks formatting an inspection report into the operator text view.
func BenchmarkRender(b *testing.B) {
	lock := benchmarkLock()
	report := Inspect(lock, "locks/prod-agent.lock.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := Render(report)
		if len(out) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}

// BenchmarkInspectAndRender benchmarks the complete inspection and rendering pipeline.
func BenchmarkInspectAndRender(b *testing.B) {
	lock := benchmarkLock()
	const lockPath = "locks/prod-agent.lock.json"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Inspect(lock, lockPath)
		out := Render(report)
		if len(out) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}

// BenchmarkHarnessInspectScale measures inspect performance across varying component and asset counts.
func BenchmarkHarnessInspectScale(b *testing.B) {
	for _, n := range []int{10, 50, 100} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			lock := scaleLock(n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				report := Inspect(lock, "locks/scale.lock.json")
				if len(report.Components) != n {
					b.Fatalf("expected %d components, got %d", n, len(report.Components))
				}
			}
		})
	}
}
