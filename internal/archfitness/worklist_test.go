package archfitness

import (
	"strings"
	"testing"
)

func TestWorkListEmpty(t *testing.T) {
	r := Report{}
	got := WorkList(r)
	if got != "" {
		t.Fatalf("expected empty string for empty report, got: %q", got)
	}
}

func TestWorkListFormatting(t *testing.T) {
	r := Report{
		Work: []Finding{
			{
				Dimension: "dependency_dag",
				Severity:  "hard",
				File:      "internal/foo/bar.go",
				Symbol:    "Import",
				Reason:    "lateral import forbidden",
			},
			{
				Dimension: "schema_migration",
				Severity:  "soft",
				File:      "internal/schema/v1.go",
				Symbol:    "UserSchema",
				Reason:    "unversioned payload",
			},
		},
	}
	got := WorkList(r)
	expectedLines := []string{
		"hard dependency_dag internal/foo/bar.go:Import — lateral import forbidden",
		"soft schema_migration internal/schema/v1.go:UserSchema — unversioned payload",
	}
	lines := strings.Split(strings.TrimSuffix(got, "\n"), "\n")
	if len(lines) != len(expectedLines) {
		t.Fatalf("expected %d lines, got %d:\n%s", len(expectedLines), len(lines), got)
	}
	for i, exp := range expectedLines {
		if lines[i] != exp {
			t.Errorf("line %d mismatch: got %q, want %q", i, lines[i], exp)
		}
	}
}

func TestWorkListEdgeCases(t *testing.T) {
	// Edge cases: missing symbol, missing file, empty reason.
	r := Report{
		Work: []Finding{
			{
				Dimension: "hot_path_scaling",
				Severity:  "hard",
				File:      "",
				Symbol:    "",
				Reason:    "",
			},
		},
	}
	got := WorkList(r)
	want := "hard hot_path_scaling : — \n"
	if got != want {
		t.Fatalf("edge case format mismatch: got %q, want %q", got, want)
	}
}

func TestAnalyzeDeterministicSorting(t *testing.T) {
	// Invariant: architecture fitness sort order is deterministic by dimension, file, symbol.
	in := Input{
		ForbiddenImports: []Finding{
			{File: "z.go", Symbol: "B", Reason: "r1"},
			{File: "a.go", Symbol: "A", Reason: "r2"},
			{File: "a.go", Symbol: "Z", Reason: "r3"},
		},
		FrozenSeamChurn: []Finding{
			{File: "m.go", Symbol: "Func", Reason: "r4"},
		},
	}
	r := Analyze(in)
	if len(r.Work) != 4 {
		t.Fatalf("expected 4 work items, got %d", len(r.Work))
	}
	// "dependency_dag" comes before "frozen_seams" alphabetically.
	if r.Work[0].Dimension != "dependency_dag" || r.Work[0].File != "a.go" || r.Work[0].Symbol != "A" {
		t.Errorf("work[0] mismatch: %+v", r.Work[0])
	}
	if r.Work[1].Dimension != "dependency_dag" || r.Work[1].File != "a.go" || r.Work[1].Symbol != "Z" {
		t.Errorf("work[1] mismatch: %+v", r.Work[1])
	}
	if r.Work[2].Dimension != "dependency_dag" || r.Work[2].File != "z.go" || r.Work[2].Symbol != "B" {
		t.Errorf("work[2] mismatch: %+v", r.Work[2])
	}
	if r.Work[3].Dimension != "frozen_seams" {
		t.Errorf("work[3] dimension mismatch: %+v", r.Work[3])
	}
}

func TestAnalyzeFailClosedUnknownSeverity(t *testing.T) {
	// Guard: fail-closed on unknown severities by defaulting unset/unknown severities to hard debt.
	in := Input{
		DynamicHotPath: []Finding{
			{Severity: "unknown_custom", File: "hot.go", Symbol: "Exec", Reason: "custom severity"},
			{Severity: "", File: "hot2.go", Symbol: "Run", Reason: "empty severity"},
			{Severity: "soft", File: "hot3.go", Symbol: "Walk", Reason: "soft severity"},
		},
	}
	r := Analyze(in)
	if r.HardDebt != 2 {
		t.Fatalf("expected 2 hard debt items (failing closed on unknown/empty), got %d", r.HardDebt)
	}
	if r.Work[0].Severity != "hard" || r.Work[1].Severity != "hard" {
		t.Errorf("expected hard severities for unknown/empty, got %q, %q", r.Work[0].Severity, r.Work[1].Severity)
	}
	if r.Work[2].Severity != "soft" {
		t.Errorf("expected soft severity preserved, got %q", r.Work[2].Severity)
	}
}
