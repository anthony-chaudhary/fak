package harnessverify

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func BenchmarkHarnessVerify(b *testing.B) {
	lock := harnessresolve.Lock{
		ID: "sha256:lock-verified",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "tool", ID: "search", Source: "repo", Value: "kb"},
			{Kind: "tool", ID: "bash", Source: "repo", Value: "safe"},
			{Kind: "policy", ID: "sandbox", Source: "policy", Grants: []string{"read", "exec"}, Denies: []string{"net"}},
			{Kind: "instruction", ID: "system", Source: "repo", Value: "concise"},
			{Kind: "secret", ID: "token", Source: "vault", Ref: "secret/token"},
			{Kind: "route", ID: "primary", Source: "config", Value: "fast"},
		},
	}
	obs := Observation{
		Schema: ObservationSchema,
		LockID: lock.ID,
		RunID:  "run-100",
		Capabilities: []Capability{
			{Capability: "tool:search", Source: "repo", Value: "kb"},
			{Capability: "tool:bash", Source: "repo", Value: "safe"},
			{Capability: "policy:sandbox", Source: "policy", Grants: []string{"read", "exec"}, Denies: []string{"net"}},
			{Capability: "instruction:system", Source: "repo", Value: "concise"},
			{Capability: "secret:token", Source: "vault", Ref: "secret/token"},
			{Capability: "route:primary", Source: "config", Value: "fast"},
		},
		Events: []Event{
			{Kind: "invoke", Capability: "tool:search", Source: "repo", Outcome: "allow"},
			{Kind: "invoke", Capability: "tool:bash", Source: "repo", Outcome: "allow"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Verify(lock, obs)
		if err != nil {
			b.Fatalf("verify failed: %v", err)
		}
		if report.Verdict != "verified" {
			b.Fatalf("expected verified verdict, got %s", report.Verdict)
		}
	}
}

func BenchmarkVerifyDeviations(b *testing.B) {
	lock := harnessresolve.Lock{
		ID: "sha256:lock-drift",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "tool", ID: "search", Source: "repo", Value: "kb"},
			{Kind: "tool", ID: "bash", Source: "repo", Value: "strict"},
			{Kind: "policy", ID: "egress", Source: "base", Denies: []string{"all"}},
			{Kind: "workflow", ID: "lint", Source: "repo"},
		},
	}
	obs := Observation{
		Schema: ObservationSchema,
		LockID: lock.ID,
		RunID:  "run-drift",
		Capabilities: []Capability{
			{Capability: "tool:search", Source: "repo", Value: "kb"},
			{Capability: "tool:bash", Source: "runtime-override", Value: "permissive"},
			{Capability: "tool:extra", Source: "agent", Value: "unlocked"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Verify(lock, obs)
		if err != nil {
			b.Fatalf("verify failed: %v", err)
		}
		if report.Verdict != "deviation" {
			b.Fatalf("expected deviation verdict, got %s", report.Verdict)
		}
	}
}

func BenchmarkVerifyScale(b *testing.B) {
	const n = 64
	assets := make([]harnesscompose.EffectiveAsset, n)
	caps := make([]Capability, n)
	for i := 0; i < n; i++ {
		id := strconv.Itoa(i)
		assets[i] = harnesscompose.EffectiveAsset{
			Kind:   "tool",
			ID:     "tool-" + id,
			Source: "manifest",
			Value:  "val-" + id,
		}
		caps[i] = Capability{
			Capability: fmt.Sprintf("tool:tool-%s", id),
			Source:     "manifest",
			Value:      "val-" + id,
		}
	}
	lock := harnessresolve.Lock{ID: "sha256:scale-lock", Assets: assets}
	obs := Observation{
		Schema:       ObservationSchema,
		LockID:       lock.ID,
		RunID:        "run-scale",
		Capabilities: caps,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Verify(lock, obs)
		if err != nil {
			b.Fatalf("verify scale failed: %v", err)
		}
		if report.Matched != n {
			b.Fatalf("expected %d matches, got %d", n, report.Matched)
		}
	}
}

func BenchmarkRender(b *testing.B) {
	report := Report{
		Schema:  ReportSchema,
		Verdict: "deviation",
		LockID:  "sha256:lock-report",
		RunID:   "run-report",
		Findings: []Finding{
			{Status: "matched", Capability: "tool:search", ExpectedSource: "repo", RuntimeSource: "repo"},
			{Status: "changed", Capability: "tool:bash", ExpectedSource: "repo", RuntimeSource: "override", Difference: "source,value"},
			{Status: "added", Capability: "tool:extra", RuntimeSource: "agent"},
			{Status: "omitted", Capability: "policy:egress", ExpectedSource: "base"},
		},
		Events: []Event{
			{Kind: "invoke", Capability: "tool:search", Source: "repo", Outcome: "allow"},
			{Kind: "gate", Capability: "tool:extra", Source: "agent", Outcome: "deny"},
		},
		Matched: 1,
		Changed: 1,
		Added:   1,
		Omitted: 1,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := Render(report)
		if len(s) == 0 {
			b.Fatal("unexpected empty render")
		}
	}
}

func TestBenchmarkHarnessVerifySanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessVerify)
	if res.N <= 0 {
		t.Fatalf("expected iterations > 0, got %d", res.N)
	}
	devRes := testing.Benchmark(BenchmarkVerifyDeviations)
	if devRes.N <= 0 {
		t.Fatalf("expected deviations iterations > 0, got %d", devRes.N)
	}
}
