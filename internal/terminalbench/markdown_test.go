package terminalbench

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRenderMarkdownGolden(t *testing.T) {
	t.Parallel()

	report := &Report{
		GeneratedAt:   "2026-08-27T12:34:56Z",
		Benchmark:     "terminal-bench-command-smoke",
		Model:         "test-model",
		EvidenceClass: "benchmark-native",
		Tasks: []TaskReport{{
			ID: "task-1",
			Raw: ArmResult{
				TestSuccess:        true,
				SafeResolve:        false,
				NormalizedCommands: []NormalizedCommand{{Tool: "terminal.exec"}},
			},
			Fak: ArmResult{
				TestSuccess:        true,
				SafeResolve:        true,
				DeniedCommands:     3,
				DangerousBlocks:    []CommandEvent{{Tool: "terminal.exec"}},
				UnnecessaryBlocks:  []CommandEvent{{Tool: "terminal.exec"}, {Tool: "terminal.exec"}},
				NormalizedCommands: []NormalizedCommand{{Tool: "terminal.exec"}, {Tool: "terminal.exec"}},
			},
		}},
		Summary: Summary{
			TaskCount: 1,
			Raw: ArmSummary{
				Pass1:                0.125,
				SafeResolveRate:      0.250,
				PolicyBreaches:       4,
				MinefieldHits:        5,
				DangerousBlocks:      6,
				UnnecessaryBlocks:    7,
				DeniedCommands:       8,
				EvidenceCompleteness: 0.375,
			},
			Fak: ArmSummary{
				Pass1:                0.500,
				SafeResolveRate:      0.625,
				PolicyBreaches:       1,
				MinefieldHits:        2,
				DangerousBlocks:      3,
				UnnecessaryBlocks:    4,
				DeniedCommands:       5,
				EvidenceCompleteness: 0.750,
			},
		},
		OfficialHarness: OfficialHarness{
			Required:  true,
			Available: false,
			Reason:    "fixture unavailable",
		},
		PromotionRequirements: []string{"capture official grader output", "bind the evidence join"},
		ResultClaimAllowed:    false,
		ClaimBoundary:         "adapter smoke only",
	}

	assertMarkdownGolden(t, "render_markdown.golden.md", RenderMarkdown(report))
}

func assertMarkdownGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("create golden dir: %v", err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("update golden: %v", err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if got != string(want) {
		t.Fatalf("RenderMarkdown bytes changed\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}
