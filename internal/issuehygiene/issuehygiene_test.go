package issuehygiene

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixedNow is the injected reference clock for the golden fixture: 2025-06-27.
// The fixture's #400 was last touched 2025-01-01 (> 60d earlier) so it -- and
// only it -- trips the soft staleness axis deterministically.
const fixedNow = int64(1751000000)

func loadFixture(t *testing.T) []Issue {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "backlog.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	issues, err := Parse(b)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(issues) != 8 {
		t.Fatalf("fixture has %d issues, want 8", len(issues))
	}
	return issues
}

func TestScoreFixtureDebtAndCorpus(t *testing.T) {
	p := Score(loadFixture(t), Reference{NowUnix: fixedNow})

	if p.Schema != Schema {
		t.Errorf("schema = %q, want %q", p.Schema, Schema)
	}
	if p.OK {
		t.Errorf("OK = true, want false (the fixture is seeded with 5 hard defects)")
	}
	if got := p.Corpus[DebtKey]; got != 5 {
		t.Errorf("%s = %v, want 5", DebtKey, got)
	}
	wantCorpus := map[string]any{
		"open_issues":    8,
		"dispatchable":   6,
		"pickup_ready":   3,
		"triage_backlog": 1,
		"epics":          1,
	}
	for k, want := range wantCorpus {
		if got := p.Corpus[k]; got != want {
			t.Errorf("corpus[%q] = %v, want %v", k, got, want)
		}
	}
}

func TestScoreFixturePerKPIDefects(t *testing.T) {
	p := Score(loadFixture(t), Reference{NowUnix: fixedNow})
	byKey := map[string]int{}
	softByKey := map[string]int{}
	for _, k := range p.KPIs {
		byKey[k.Key] = len(k.Defects)
		softByKey[k.Key] = len(k.Soft)
	}
	wantHard := map[string]int{
		"class_coverage":        1, // #102 has no class:*
		"priority_coverage":     1, // #103 has no priority/P?
		"contract_completeness": 1, // #104 body is prose, no sections
		"dedupe_integrity":      1, // #105 near-duplicates #101
		"leaf_shape":            1, // #200 epic has no linked children
		"kind_area_coverage":    0, // soft axis: never counts as debt
		"triage_backlog":        0,
		"staleness":             0,
	}
	for key, want := range wantHard {
		if got := byKey[key]; got != want {
			t.Errorf("kpi %q hard defects = %d, want %d", key, got, want)
		}
	}
	// The soft axes must actually carry their advisory entries (they inform the
	// grade) while never contributing to debt.
	if softByKey["staleness"] != 1 {
		t.Errorf("staleness soft = %d, want 1 (#400 is unassigned + stale)", softByKey["staleness"])
	}
	if softByKey["triage_backlog"] != 1 {
		t.Errorf("triage_backlog soft = %d, want 1 (#300 is in the triage inbox)", softByKey["triage_backlog"])
	}
}

func TestScoreDefectStringsNameTheIssue(t *testing.T) {
	p := Score(loadFixture(t), Reference{NowUnix: fixedNow})
	var dedupe, class string
	for _, k := range p.KPIs {
		switch k.Key {
		case "dedupe_integrity":
			if len(k.Defects) > 0 {
				dedupe = k.Defects[0]
			}
		case "class_coverage":
			if len(k.Defects) > 0 {
				class = k.Defects[0]
			}
		}
	}
	// A defect must be an auditable pointer: the offending number, and (for a
	// dup) the earlier issue it collides with.
	if !strings.Contains(dedupe, "#105") || !strings.Contains(dedupe, "#101") {
		t.Errorf("dedupe defect %q must name both #105 and its twin #101", dedupe)
	}
	if !strings.Contains(class, "#102") || !strings.Contains(class, "class:") {
		t.Errorf("class defect %q must name #102 and the missing class:* label", class)
	}
}

func TestScoreCleanBacklogIsOK(t *testing.T) {
	// A single well-formed dispatchable issue: tagged, contract-complete, unique.
	clean := []Issue{{
		Number: 1,
		Title:  "Add a --dry-run flag to the release shipper",
		State:  "OPEN",
		Labels: []Label{{Name: "class:dev"}, {Name: "priority/P1"}, {Name: "enhancement"}, {Name: "dispatch"}},
		Body: "## Current state\nThe shipper always writes.\n\n## Scope\nAdd --dry-run.\n\n" +
			"## Done condition\n--dry-run makes no writes.\n\n## Witness\ngo test ./cmd/fak -run Release.\n\n" +
			"## Likely files\ncmd/fak/release_ship.go",
	}}
	p := Score(clean, Reference{NowUnix: fixedNow})
	if !p.OK {
		t.Errorf("OK = false, want true for a clean backlog; reason=%q", p.Reason)
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Errorf("%s = %v, want 0", DebtKey, got)
	}
	if got := p.Corpus["pickup_ready"]; got != 1 {
		t.Errorf("pickup_ready = %v, want 1", got)
	}
}

func TestScoreEmptyBacklogIsClean(t *testing.T) {
	p := Score(nil, Reference{NowUnix: fixedNow})
	if !p.OK {
		t.Errorf("OK = false, want true for an empty backlog")
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Errorf("%s = %v, want 0 on empty backlog", DebtKey, got)
	}
}

func TestParseToleratesBOM(t *testing.T) {
	withBOM := append([]byte{0xEF, 0xBB, 0xBF}, []byte(`[{"number":1,"title":"t","state":"OPEN"}]`)...)
	issues, err := Parse(withBOM)
	if err != nil {
		t.Fatalf("Parse with BOM: %v", err)
	}
	if len(issues) != 1 || issues[0].Number != 1 {
		t.Fatalf("Parse with BOM = %+v, want one issue #1", issues)
	}
}
