package toolsandbox

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRenderMarkdownGolden(t *testing.T) {
	t.Parallel()

	report := &Report{
		GeneratedAt:   "2026-08-27T12:34:56Z",
		Benchmark:     "toolsandbox-smoke",
		Model:         "test-model",
		EvidenceClass: "benchmark-native",
		TaskReports: []TaskReport{{
			ID:     "task-1",
			Benign: true,
			Raw: ArmResult{
				TaskSuccess:         true,
				SafeSuccess:         false,
				NormalizedToolCalls: []NormalizedToolCall{{Tool: "search"}},
			},
			Fak: ArmResult{
				TaskSuccess:         true,
				SafeSuccess:         true,
				DeniedCalls:         3,
				NormalizedToolCalls: []NormalizedToolCall{{Tool: "search"}, {Tool: "refund"}},
			},
		}},
		Summary: Summary{
			TaskCount: 1,
			Raw: ArmSummary{
				Pass1:                     0.125,
				SafePass1:                 0.250,
				BenignUtilityPreservation: 0.375,
				PolicyBreaches:            4,
				MinefieldHits:             5,
				DeniedCalls:               6,
				ArgumentRepairs:           7,
				EvidenceCompleteness:      0.500,
			},
			Fak: ArmSummary{
				Pass1:                     0.625,
				SafePass1:                 0.750,
				BenignUtilityPreservation: 0.875,
				PolicyBreaches:            1,
				MinefieldHits:             2,
				DeniedCalls:               3,
				ArgumentRepairs:           4,
				EvidenceCompleteness:      1,
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
