package issuepolicy

import (
	"strings"
	"testing"
)

// corruptGenerationDraft mirrors the contract_test.go:195 fixture: unexpanded
// batch-filer tokens confined to the top "## Generation stream" header block,
// with an intact human-authored body below.
func corruptGenerationDraft() IssueDraft {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=second-next; title=...; labels=...; why=...; scope=...}.gen)",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"- Parent: #1625",
		"- Source: $source, Phase 2",
		"",
		"## Why",
		"The generated body below the corrupt header is intact.",
		"",
		"## Initial scope",
		"Repair only the generated metadata header.",
		"",
		"## Witness",
		"Captured dry-run output lists affected issue, marker, and replacement header.",
	}, "\n")
	return IssueDraft{
		Number: 1727,
		Title:  "generation(second-next): build the multi-generation portfolio optimizer",
		Body:   body,
		Labels: []IssueLabel{{Name: "generation"}, {Name: "gen/second-next"}},
	}
}

func TestApplyTemplateRepairSafeMergeReplacesHeaderPreservesBody(t *testing.T) {
	res, ok := ApplyTemplateRepair(corruptGenerationDraft())
	if !ok {
		t.Fatal("ApplyTemplateRepair returned ok=false for a corrupt header body")
	}
	if !res.Safe {
		t.Fatalf("merge not marked safe: unsafe=%q", res.Unsafe)
	}
	// The recomputed header replaced the corrupt block.
	for _, want := range []string{
		"## Generation stream",
		"- Generation: gen/second-next",
		"- Parent: #1625",
	} {
		if !strings.Contains(res.NewBody, want) {
			t.Fatalf("repaired body missing header line %q:\n%s", want, res.NewBody)
		}
	}
	// No unexpanded token survives.
	if strings.Contains(res.NewBody, "$(") || strings.Contains(res.NewBody, "$source") {
		t.Fatalf("repaired body still carries template tokens:\n%s", res.NewBody)
	}
	if got := UnexpandedTemplateMarkers(res.NewBody); len(got) != 0 {
		t.Fatalf("UnexpandedTemplateMarkers(newBody) = %v, want none", got)
	}
	// Every human-authored section below the header is preserved verbatim,
	// including the final line (no net deletion of body content).
	tail := strings.Join([]string{
		"## Why",
		"The generated body below the corrupt header is intact.",
		"",
		"## Initial scope",
		"Repair only the generated metadata header.",
		"",
		"## Witness",
		"Captured dry-run output lists affected issue, marker, and replacement header.",
	}, "\n")
	if !strings.Contains(res.NewBody, tail) {
		t.Fatalf("repaired body dropped human-authored tail:\n%s", res.NewBody)
	}
	if !strings.HasSuffix(res.NewBody, "replacement header.") {
		t.Fatalf("repaired body truncated the final body line:\n%s", res.NewBody)
	}
}

func TestApplyTemplateRepairIsIdempotent(t *testing.T) {
	res, ok := ApplyTemplateRepair(corruptGenerationDraft())
	if !ok || !res.Safe {
		t.Fatalf("first pass not a safe repair: ok=%v unsafe=%q", ok, res.Unsafe)
	}
	// Feeding the repaired body back reports nothing to repair: the fix is
	// self-detecting, so a re-run of the sweep never edits the same issue twice.
	if _, ok2 := ApplyTemplateRepair(IssueDraft{Number: 1727, Title: "x", Body: res.NewBody}); ok2 {
		t.Fatalf("repaired body still classified as repairable (not idempotent):\n%s", res.NewBody)
	}
}

func TestApplyTemplateRepairNoMarkersReturnsNotOK(t *testing.T) {
	clean := IssueDraft{Number: 5, Title: "clean", Body: "## Why\nAll good.\n"}
	if _, ok := ApplyTemplateRepair(clean); ok {
		t.Fatal("clean body should return ok=false (nothing to repair)")
	}
}

func TestApplyTemplateRepairRefusesProseInHeaderBlock(t *testing.T) {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=x}.gen)",
		"An operator note wedged into the header block, not a bullet.",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"",
		"## Why",
		"Intact body.",
	}, "\n")
	res, ok := ApplyTemplateRepair(IssueDraft{
		Number: 9, Title: "x", Body: body,
		Labels: []IssueLabel{{Name: "generation"}},
	})
	if !ok {
		t.Fatal("expected ok=true (markers present) with an unsafe verdict")
	}
	if res.Safe {
		t.Fatal("prose in the header block must fail closed, not auto-apply")
	}
	if res.Unsafe != TemplateUnsafeUnexpectedProse {
		t.Fatalf("unsafe reason = %q, want %q", res.Unsafe, TemplateUnsafeUnexpectedProse)
	}
}

func TestApplyTemplateRepairRefusesMarkersSpanningSections(t *testing.T) {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=x}.gen)",
		"",
		"## Why",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"",
		"## Witness",
		"Intact.",
	}, "\n")
	res, ok := ApplyTemplateRepair(IssueDraft{
		Number: 11, Title: "x", Body: body,
		Labels: []IssueLabel{{Name: "generation"}},
	})
	if !ok {
		t.Fatal("expected ok=true (markers present) with an unsafe verdict")
	}
	if res.Safe {
		t.Fatal("markers spanning two sections must fail closed, not auto-apply")
	}
	if res.Unsafe != TemplateUnsafeMarkersSpanSections {
		t.Fatalf("unsafe reason = %q, want %q", res.Unsafe, TemplateUnsafeMarkersSpanSections)
	}
}
