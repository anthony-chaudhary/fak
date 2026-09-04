package harnessoverride

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func benchmarkLock() harnessresolve.Lock {
	return harnessresolve.Lock{
		Schema: harnessresolve.LockSchema,
		ID:     "sha256:benchmark-lock-id",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "model", ID: "coder", Value: "qwen-coder-14b", Source: "manifest"},
			{Kind: "tool", ID: "filesystem", Value: "chroot:/workspace", Source: "layer:base", Grants: []string{"read", "stat"}},
			{Kind: "tool", ID: "network", Value: "egress-restricted", Source: "layer:security", Locked: true},
			{Kind: "policy", ID: "guardrails", Source: "company", Grants: []string{"search", "read"}, Denies: []string{"exec"}},
			{Kind: "workflow", ID: "audit-trail", Source: "company", Mandatory: true},
			{Kind: "instruction", ID: "system", Value: "expert assistant", Source: "manifest"},
		},
	}
}

func scaleLock(n int) harnessresolve.Lock {
	assets := make([]harnesscompose.EffectiveAsset, n)
	for i := 0; i < n; i++ {
		assets[i] = harnesscompose.EffectiveAsset{
			Kind:      "tool",
			ID:        fmt.Sprintf("tool-%04d", i),
			Value:     fmt.Sprintf("/opt/tools/%04d", i),
			Source:    "manifest",
			Mandatory: i%5 == 0,
			Locked:    i%5 == 1,
		}
	}
	return harnessresolve.Lock{
		Schema: harnessresolve.LockSchema,
		ID:     "sha256:scale-lock-id",
		Assets: assets,
	}
}

// BenchmarkHarnessOverride benchmarks the combined proposal generation and rendering pipeline.
func BenchmarkHarnessOverride(b *testing.B) {
	lock := benchmarkLock()
	req := Request{
		Capability: "tool:filesystem",
		Value:      "chroot:/sandboxes/agent-01",
		LayerID:    "operator-custom",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proposal, err := Propose(lock, req)
		if err != nil {
			b.Fatalf("Propose failed: %v", err)
		}
		rendered := Render(proposal)
		if len(rendered) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}

// BenchmarkPropose benchmarks generating proposals across different capability types.
func BenchmarkPropose(b *testing.B) {
	lock := benchmarkLock()

	b.Run("ToolReplace", func(b *testing.B) {
		req := Request{
			Capability: "tool:filesystem",
			Value:      "chroot:/sandboxes/agent-01",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := Propose(lock, req); err != nil {
				b.Fatalf("Propose failed: %v", err)
			}
		}
	})

	b.Run("PolicyNarrow", func(b *testing.B) {
		req := Request{
			Capability: "policy:guardrails",
			Denies:     []string{"network", "raw_exec", "network"},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := Propose(lock, req); err != nil {
				b.Fatalf("Propose failed: %v", err)
			}
		}
	})
}

// BenchmarkRender benchmarks formatting an override proposal into operator text.
func BenchmarkRender(b *testing.B) {
	lock := benchmarkLock()
	proposal, err := Propose(lock, Request{
		Capability: "policy:guardrails",
		Denies:     []string{"raw_exec", "spawn"},
		LayerID:    "audit-boundary",
	})
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rendered := Render(proposal)
		if len(rendered) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}

// BenchmarkProposeScale measures Propose performance as the lock asset count scales.
func BenchmarkProposeScale(b *testing.B) {
	for _, n := range []int{10, 50, 100} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			lock := scaleLock(n)
			targetIdx := n - 1
			for targetIdx%5 == 0 || targetIdx%5 == 1 {
				targetIdx--
			}
			req := Request{
				Capability: fmt.Sprintf("tool:tool-%04d", targetIdx),
				Value:      "/custom/override",
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := Propose(lock, req); err != nil {
					b.Fatalf("Propose failed: %v", err)
				}
			}
		})
	}
}
