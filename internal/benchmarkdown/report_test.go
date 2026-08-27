package benchmarkdown

import (
	"strings"
	"testing"
)

func TestRenderAdapterReportLayout(t *testing.T) {
	t.Parallel()

	got := RenderAdapterReport(AdapterReport{
		Title: "Adapter Report",
		Metadata: Metadata{
			GeneratedAt:              "2026-08-27T00:00:00Z",
			Benchmark:                "bench",
			TaskCount:                1,
			OfficialHarnessRequired:  true,
			OfficialHarnessAvailable: true,
			ResultClaimAllowed:       false,
			ClaimBoundary:            "fixture only",
		},
		Summary: Table{
			Header:    "| Arm | Score |",
			Separator: "|---|---:|",
			Rows:      []string{"| raw | 1 |", "| fak | 2 |"},
		},
		Tasks: Table{
			Header:    "| Task | OK |",
			Separator: "|---|:---:|",
			Rows:      []string{"| one | true |"},
		},
	})
	want := "# Adapter Report\n\n" +
		"- Generated: `2026-08-27T00:00:00Z`\n" +
		"- Benchmark: `bench`\n" +
		"- Tasks: `1`\n" +
		"- Official harness: required=true available=true\n" +
		"- Result claim allowed: `false`\n" +
		"- Boundary: fixture only\n\n" +
		"| Arm | Score |\n|---|---:|\n| raw | 1 |\n| fak | 2 |\n" +
		"\n## Tasks\n\n" +
		"| Task | OK |\n|---|:---:|\n| one | true |\n"
	if got != want {
		t.Fatalf("RenderAdapterReport mismatch\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestRenderAdapterReportOptionalMetadataAndPromotions(t *testing.T) {
	t.Parallel()

	got := RenderAdapterReport(AdapterReport{
		Title: "Adapter Report",
		Metadata: Metadata{
			GeneratedAt:           "now",
			Benchmark:             "bench",
			Model:                 "model",
			EvidenceClass:         "native",
			OfficialHarnessReason: "not installed",
			ResultClaimAllowed:    true,
		},
		Summary:               Table{Header: "s", Separator: "-"},
		Tasks:                 Table{Header: "t", Separator: "-"},
		PromotionRequirements: []string{"capture proof"},
	})
	for _, want := range []string{
		"- Model: `model`\n",
		"- Evidence class: `native`\n",
		"- Official harness: required=false available=false (not installed)\n",
		"- Result claim allowed: `true`\n",
		"\n## Promotion Requirements\n\n- capture proof\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("RenderAdapterReport missing %q:\n%s", want, got)
		}
	}
}
