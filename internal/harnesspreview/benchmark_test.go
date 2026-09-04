package harnesspreview

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessclassify"
	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
)

func BenchmarkHarnessPreview(b *testing.B) {
	current := fixtureLock("sha256:current",
		policy("company", []string{"search"}, []string{"shell"}, true),
		harnesscompose.EffectiveAsset{Kind: "tool", ID: "logger", Source: "base"},
		harnesscompose.EffectiveAsset{Kind: "secret", ID: "api-token", Ref: "vault:token-v1", Source: "base"},
		harnesscompose.EffectiveAsset{Kind: "workflow", ID: "lint", Source: "base", Mandatory: true},
	)
	candidate := fixtureLock("sha256:candidate",
		policy("task-override", []string{"search", "shell", "exec"}, nil, false),
		harnesscompose.EffectiveAsset{Kind: "tool", ID: "payments", Source: "task-override"},
		harnesscompose.EffectiveAsset{Kind: "secret", ID: "api-token", Ref: "vault:token-v2", Source: "task-override"},
		harnesscompose.EffectiveAsset{Kind: "workflow", ID: "lint", Source: "task-override", Mandatory: false},
		harnesscompose.EffectiveAsset{Kind: "instruction", ID: "guidelines", Source: "task-override", Value: "strict"},
	)
	class := harnessclassify.Result{
		Confidence:    0.55,
		NeedsDecision: true,
		DecisionRequest: &harnessclassify.DecisionRequest{
			Scope:  "project:finance-core",
			Reason: "finance and compliance signals tie",
		},
	}
	in := Input{
		Current:         &current,
		Candidate:       &candidate,
		CurrentDomain:   "coding",
		CandidateDomain: "finance",
		Classification:  &class,
		Conflict:        "contract version mismatch on api-v2",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := Compare(in)
		if !p.RequiresDecision || p.Verdict != VerdictDecision {
			b.Fatalf("expected decision-required, got %s", p.Verdict)
		}
	}
}

func BenchmarkCompareQuiet(b *testing.B) {
	lock := fixtureLock("sha256:steady",
		policy("company", []string{"search"}, []string{"shell"}, true),
		harnesscompose.EffectiveAsset{Kind: "tool", ID: "logger", Source: "base"},
	)
	in := Input{
		Current:         &lock,
		Candidate:       &lock,
		CurrentDomain:   "coding",
		CandidateDomain: "coding",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := Compare(in)
		if p.RequiresDecision || p.Verdict != VerdictQuiet {
			b.Fatalf("expected quiet verdict, got %s", p.Verdict)
		}
	}
}

func BenchmarkComparePrivilegeWidening(b *testing.B) {
	old := fixtureLock("sha256:v1", policy("base", []string{"read"}, []string{"write", "exec"}, true))
	next := fixtureLock("sha256:v2", policy("custom", []string{"read", "write"}, []string{"exec"}, false))
	in := Input{Current: &old, Candidate: &next}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := Compare(in)
		if len(p.Changes) == 0 {
			b.Fatal("expected privilege widening changes")
		}
	}
}

func BenchmarkRender(b *testing.B) {
	old := fixtureLock("sha256:old")
	next := fixtureLock("sha256:new",
		harnesscompose.EffectiveAsset{Kind: "tool", ID: "deploy", Source: "task:release"},
		harnesscompose.EffectiveAsset{Kind: "secret", ID: "prod-creds", Source: "task:release"},
	)
	p := Compare(Input{
		Current:         &old,
		Candidate:       &next,
		CurrentDomain:   "dev",
		CandidateDomain: "prod",
	})

	b.Run("CLI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := RenderCLI(p)
			if len(s) == 0 {
				b.Fatal("unexpected empty CLI render")
			}
		}
	})

	b.Run("TUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := RenderTUI(p)
			if len(s) == 0 {
				b.Fatal("unexpected empty TUI render")
			}
		}
	})
}

func TestBenchmarkHarnessPreviewSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessPreview)
	if res.N <= 0 {
		t.Fatalf("expected iterations > 0, got %d", res.N)
	}
	quietRes := testing.Benchmark(BenchmarkCompareQuiet)
	if quietRes.N <= 0 {
		t.Fatalf("expected quiet iterations > 0, got %d", quietRes.N)
	}
}
