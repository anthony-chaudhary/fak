package cacheheadlines

import (
	"strings"
	"testing"
)

func BenchmarkCheckLine(b *testing.B) {
	lines := []string{
		"We achieved a huge 99% cache win in production.",
		"Ship it: this is a huge cache win.",
		"Honestly the cache is 99% of the speedup story here.",
		"provider prompt-cache rebate: 99% cache read, OBSERVED (cost/latency only)",
		"kernel KV reuse was the cache win here — WITNESSED",
		"O(1) context saved 99% cache prefill work",
		"- \"cache win\" as a bare headline is legacy language to remove.",
		"Reviewing the '99% cache' pattern in documentation.",
		"the bounded cache window is argmax-identical to the full-cache windowed decode",
		"fak treats the model like an untrusted program.",
	}
	var violations int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		violations = 0
		for _, line := range lines {
			if bad, _ := CheckLine(line); bad {
				violations++
			}
		}
	}
	if violations != 3 {
		b.Fatalf("expected 3 violations, got %d", violations)
	}
}

func BenchmarkCheckLine_Violation(b *testing.B) {
	line := "We hit a 99% cache — the biggest win of the quarter!"
	var bad bool
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bad, _ = CheckLine(line)
	}
	if !bad {
		b.Fatal("expected violation")
	}
}

func BenchmarkCheckLine_Labeled(b *testing.B) {
	line := "provider prompt-cache rebate: 99% cache read, OBSERVED (cost/latency only, not fak-owned reuse)"
	var bad bool
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bad, _ = CheckLine(line)
	}
	if bad {
		b.Fatal("unexpected violation")
	}
}

func BenchmarkCheckLine_Clean(b *testing.B) {
	line := "fak is an agent kernel: one Go binary that sits between an AI agent and the tools it calls."
	var bad bool
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bad, _ = CheckLine(line)
	}
	if bad {
		b.Fatal("unexpected violation")
	}
}

func BenchmarkScanText(b *testing.B) {
	doc := strings.Repeat(
		"# Cache Performance Analysis\n\n"+
			"This report examines performance metrics across recent runs.\n"+
			"We observed significant speedups in the benchmark suite.\n"+
			"We hit a 99% cache in testing.\n"+
			"provider prompt-cache: 99% cache achieved with low latency.\n"+
			"The model context is preserved across calls.\n"+
			"Reviewing the '99% cache' pattern in docs.\n"+
			"Reported a cache-win without attribution.\n"+
			"kernel KV reuse was the cache win here — WITNESSED.\n"+
			"All evaluations completed within tolerances.\n\n",
		20,
	)
	var findings []Finding
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findings = ScanText(doc, "benchmark.md")
	}
	if len(findings) == 0 {
		b.Fatal("expected findings in benchmark document")
	}
}

func BenchmarkScanText_Clean(b *testing.B) {
	cleanDoc := strings.Repeat(
		"# Compliant Cache Performance Report\n\n"+
			"provider prompt-cache rebate: 99% cache read, OBSERVED (cost/latency only).\n"+
			"kernel KV reuse was the cache win here — WITNESSED, bit-identical to full prefill.\n"+
			"O(1) context saved 99% cache prefill work (WITNESSED resident-view elision).\n"+
			"- \"cache win\" as a bare headline is legacy language to remove.\n"+
			"Reviewing the '99% cache' pattern in documentation.\n"+
			"the bounded cache window is argmax-identical to the full-cache windowed decode.\n"+
			"fak treats the model like an untrusted program.\n\n",
		25,
	)
	var findings []Finding
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findings = ScanText(cleanDoc, "clean.md")
	}
	if len(findings) != 0 {
		b.Fatalf("expected 0 findings, got %d", len(findings))
	}
}

func BenchmarkScanText_Violations(b *testing.B) {
	violationDoc := strings.Repeat(
		"# Unlabeled Cache Claims\n\n"+
			"We achieved a huge 99% cache across all workloads.\n"+
			"The new pipeline delivered a massive cache win for users.\n"+
			"Honestly the cache is 99% of what made this fast.\n"+
			"Our speedup metrics show 95% of the story came from the cache layer.\n"+
			"Reported a bare cache-win in the release summary.\n"+
			"The system cache wins here without qualification.\n\n",
		25,
	)
	var findings []Finding
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		findings = ScanText(violationDoc, "violations.md")
	}
	if len(findings) == 0 {
		b.Fatal("expected violations in document")
	}
}

func BenchmarkShouldSkip(b *testing.B) {
	paths := []string{
		"docs/releases/v0.10.0.md",
		"vendor/github.com/pkg/file.md",
		"node_modules/pkg/README.md",
		"llms-full.txt",
		"docs/overview.md",
		"internal/cacheheadlines/cacheheadlines.go",
		"cmd/fak/main.go",
		"README.md",
	}
	var skipped int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		skipped = 0
		for _, p := range paths {
			if ShouldSkip(p) {
				skipped++
			}
		}
	}
	if skipped != 5 {
		b.Fatalf("expected 5 skipped paths, got %d", skipped)
	}
}

func BenchmarkMatchesScanGlob(b *testing.B) {
	paths := []string{
		"docs/guide.md",
		"index.html",
		"notes.txt",
		"main.go",
		"config.json",
		"script.sh",
	}
	var matches int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matches = 0
		for _, p := range paths {
			if MatchesScanGlob(p) {
				matches++
			}
		}
	}
	if matches != 3 {
		b.Fatalf("expected 3 matching paths, got %d", matches)
	}
}

func TestBenchmarkSanity(t *testing.T) {
	if bad, _ := CheckLine("bare 99% cache"); !bad {
		t.Fatal("expected violation")
	}
	findings := ScanText("bare 99% cache\nclean line\n", "test.md")
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(findings))
	}
}
