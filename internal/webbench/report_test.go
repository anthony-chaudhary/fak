package webbench

import (
	"strings"
	"testing"
)

// TestBuildSuccessRateFamilyGatedWhenNoPredictions pins the default: with no --predictions
// the "Task success rate + safety" family stays the honest gated placeholder — the
// comparison never claims a success rate it did not grade.
func TestBuildSuccessRateFamilyGatedWhenNoPredictions(t *testing.T) {
	fam := buildSuccessRateFamily("")
	if fam.Name != "Task success rate + safety" {
		t.Fatalf("family name = %q", fam.Name)
	}
	if fam.Provenance != "gated" {
		t.Errorf("no-predictions family must be gated, got %q", fam.Provenance)
	}
	if len(fam.Rows) != 1 || fam.Rows[0].Values != nil {
		t.Errorf("no-predictions family must be the placeholder row, got %+v", fam.Rows)
	}
}

// TestBuildSuccessRateFamilyFoldsPredictions proves the wire: given a predictions path,
// the family is folded from RunEval rather than left as the static placeholder. On a box
// without the official harness (the test environment) RunEval returns a gated result, so
// the row must carry the SPECIFIC reason + the predictions path — never a fabricated rate.
// The provenance stays "gated" (honest) precisely because the rate was not measured here.
func TestBuildSuccessRateFamilyFoldsPredictions(t *testing.T) {
	fam := buildSuccessRateFamily("/nonexistent/preds.json")
	if fam.Name != "Task success rate + safety" {
		t.Fatalf("family name = %q", fam.Name)
	}
	if len(fam.Rows) != 1 {
		t.Fatalf("expected exactly one folded row, got %d: %+v", len(fam.Rows), fam.Rows)
	}
	row := fam.Rows[0]
	// The folded row (gated or measured or errored) must NAME the predictions input — this
	// is the proof the flag is no longer discarded. The static placeholder carries no Values.
	if row.Values == nil {
		t.Fatalf("a predictions path must produce a folded row with Values, got the static placeholder: %+v", row)
	}
	if got, ok := row.Values["predictions"]; !ok || got != "/nonexistent/preds.json" {
		t.Errorf("folded row must carry the predictions path, got %+v", row.Values)
	}
	// On this harness-less box the fold must be gated/errored and never claim "measured".
	if fam.Provenance == "measured" {
		t.Errorf("a box without the harness must not report a measured success rate: %+v", fam)
	}
}

// TestBuildComparisonAlwaysHasSuccessFamily guards that BuildComparison still emits the
// success-rate family in every comparison (the fold is additive, not a replacement of the
// four-family shape).
func TestBuildComparisonAlwaysHasSuccessFamily(t *testing.T) {
	d := NewDataset([]Instance{{TaskID: "t1", Description: "Find a contact email."}})
	c := BuildComparison(CompareInputs{Dataset: d, Geometry: DefaultGeometryModel(), Workers: []int{1}})
	var found bool
	for _, fam := range c.Families {
		if fam.Name == "Task success rate + safety" {
			found = true
		}
	}
	if !found {
		t.Error("BuildComparison must always include the Task success rate + safety family")
	}
}

// TestRenderMarkdownIncludesMeasurementStatus pins issue #72's doc contract for
// generated WebBench reports: the deterministic geometry table must be labeled as
// theoretical/model-only before any result table appears.
func TestRenderMarkdownIncludesMeasurementStatus(t *testing.T) {
	d := NewDataset([]Instance{{TaskID: "t1", Description: "Find a contact email."}})
	c := BuildComparison(CompareInputs{
		Dataset:     d,
		Geometry:    DefaultGeometryModel(),
		Workers:     []int{1},
		GeneratedAt: "2026-06-30T00:00:00Z",
	})
	md := RenderMarkdown(c)
	for _, want := range []string{
		"## Measurement Status",
		"- Dataset:",
		"- Model:",
		"- Runs:",
		"- Artifacts:",
		"- Status: THEORETICAL (MODELED)",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("rendered markdown missing %q:\n%s", want, md)
		}
	}
	if strings.Index(md, "## Measurement Status") > strings.Index(md, "## Prefill-Token Work-Elimination") {
		t.Fatalf("measurement status must precede result tables:\n%s", md)
	}
}
