package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// writeVerdict drops a minimal classifier verdict payload carrying a single HARD
// scaffold with the given grade, and returns its path.
func writeVerdict(t *testing.T, dir, id, grade string) string {
	t.Helper()
	payload := map[string]any{
		"schema": "fak.harness-strength.v1",
		"scaffolds": []map[string]any{
			{"id": id, "hardness": "HARD", "grade": grade, "rationale": "graded " + grade + " against current model strength"},
		},
	}
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		t.Fatalf("marshal verdict: %v", err)
	}
	path := filepath.Join(dir, "verdict-"+grade+".json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write verdict: %v", err)
	}
	return path
}

func runHDD(t *testing.T, argv []string) map[string]any {
	t.Helper()
	var out, errb bytes.Buffer
	code := runHarnessDebtDispatch(&out, &errb, argv)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var result map[string]any
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("json output: %v\n%s", err, out.String())
	}
	if result["mode"] != "dry-run" {
		t.Fatalf("mode=%v, want dry-run", result["mode"])
	}
	return result
}

// TestHarnessDebtDispatchGrades asserts the boundary directly: a HARD scaffold
// graded REDUNDANT or HOBBLING files exactly one deduped candidate issue, while a
// LOAD_BEARING scaffold files nothing. Table-driven over the three grades.
func TestHarnessDebtDispatchGrades(t *testing.T) {
	cases := []struct {
		grade       string
		wantPlanned int
	}{
		{"REDUNDANT", 1},
		{"HOBBLING", 1},
		{"LOAD_BEARING", 0},
	}
	for _, tc := range cases {
		t.Run(tc.grade, func(t *testing.T) {
			dir := t.TempDir()
			verdict := writeVerdict(t, dir, "defer-cold-tools", tc.grade)
			cache := filepath.Join(dir, "seen", "seen.json")
			result := runHDD(t, []string{"--verdict", verdict, "--cache", cache, "--cap", "5", "--json"})

			planned, _ := result["planned"].([]any)
			if len(planned) != tc.wantPlanned {
				t.Fatalf("grade=%s planned=%d, want %d", tc.grade, len(planned), tc.wantPlanned)
			}
			// Dry-run must never write the seen-cache.
			if _, err := os.Stat(cache); !os.IsNotExist(err) {
				t.Fatalf("dry-run wrote cache, stat err=%v", err)
			}
			if tc.wantPlanned == 1 {
				row, _ := planned[0].(map[string]any)
				if row["grade"] != tc.grade {
					t.Fatalf("planned grade=%v, want %s", row["grade"], tc.grade)
				}
			}
		})
	}
}

// TestHarnessDebtDispatchDedup proves both dedup channels hold: a verdict carrying
// one REDUNDANT and one HOBBLING HARD scaffold plans exactly one issue per debt
// scaffold, and a re-run against either the seen-cache or existing issue bodies
// carrying those same marker keys plans zero.
func TestHarnessDebtDispatchDedup(t *testing.T) {
	dir := t.TempDir()
	payload := map[string]any{
		"schema": "fak.harness-strength.v1",
		"scaffolds": []map[string]any{
			{"id": "defer-cold-tools", "hardness": "HARD", "grade": "REDUNDANT"},
			{"id": "reask-on-empty-tool-result", "hardness": "HARD", "grade": "HOBBLING"},
			{"id": "structured-retry-envelope", "hardness": "HARD", "grade": "LOAD_BEARING"},
		},
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	verdict := filepath.Join(dir, "verdict.json")
	if err := os.WriteFile(verdict, b, 0o644); err != nil {
		t.Fatalf("write verdict: %v", err)
	}

	// First run: the two debt scaffolds file, the LOAD_BEARING one does not.
	first := runHDD(t, []string{"--verdict", verdict, "--cache", filepath.Join(dir, "a", "seen.json"), "--cap", "5", "--json"})
	planned, _ := first["planned"].([]any)
	if len(planned) != 2 {
		t.Fatalf("first run planned=%d, want 2", len(planned))
	}

	// The marker keys are seeded from scaffold identity alone.
	keyRed := hddStableKey(hddScaffold{ID: "defer-cold-tools", Grade: "REDUNDANT", Hard: true})
	keyHob := hddStableKey(hddScaffold{ID: "reask-on-empty-tool-result", Grade: "HOBBLING", Hard: true})

	// Dedup channel 1: existing issue bodies carrying those marker keys.
	existing := []hddIssue{
		{Number: 11, Body: fmt.Sprintf("<!-- fak-harness-debt-key: %s -->\nbody", keyRed)},
		{Number: 12, Body: fmt.Sprintf("<!-- fak-harness-debt-key: %s -->\nbody", keyHob)},
	}
	eb, _ := json.MarshalIndent(existing, "", "  ")
	existingPath := filepath.Join(dir, "existing.json")
	if err := os.WriteFile(existingPath, eb, 0o644); err != nil {
		t.Fatalf("write existing: %v", err)
	}
	viaExisting := runHDD(t, []string{"--verdict", verdict, "--cache", filepath.Join(dir, "b", "seen.json"),
		"--existing-json", existingPath, "--cap", "5", "--json"})
	if planned, _ := viaExisting["planned"].([]any); len(planned) != 0 {
		t.Fatalf("existing-body re-run planned=%d, want 0", len(planned))
	}

	// Dedup channel 2: a seen-cache already carrying both keys.
	seen := hddSeenCache{Schema: hddSeenSchema, Seen: map[string]hddSeenRecord{
		keyRed: {ID: "defer-cold-tools", Grade: "REDUNDANT"},
		keyHob: {ID: "reask-on-empty-tool-result", Grade: "HOBBLING"},
	}}
	seenPath := filepath.Join(dir, "c", "seen.json")
	if err := hddSaveSeen(seenPath, seen); err != nil {
		t.Fatalf("save seen: %v", err)
	}
	viaSeen := runHDD(t, []string{"--verdict", verdict, "--cache", seenPath, "--cap", "5", "--json"})
	if planned, _ := viaSeen["planned"].([]any); len(planned) != 0 {
		t.Fatalf("seen-cache re-run planned=%d, want 0", len(planned))
	}
}

// TestHarnessDebtDispatchCap bounds one run to --cap deletion issues even when more
// debt scaffolds are present.
func TestHarnessDebtDispatchCap(t *testing.T) {
	dir := t.TempDir()
	payload := map[string]any{
		"schema": "fak.harness-strength.v1",
		"scaffolds": []map[string]any{
			{"id": "scaffold-a", "hardness": "HARD", "grade": "REDUNDANT"},
			{"id": "scaffold-b", "hardness": "HARD", "grade": "HOBBLING"},
			{"id": "scaffold-c", "hardness": "HARD", "grade": "REDUNDANT"},
		},
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	verdict := filepath.Join(dir, "verdict.json")
	if err := os.WriteFile(verdict, b, 0o644); err != nil {
		t.Fatalf("write verdict: %v", err)
	}
	result := runHDD(t, []string{"--verdict", verdict, "--cache", filepath.Join(dir, "seen.json"), "--cap", "2", "--json"})
	planned, _ := result["planned"].([]any)
	if len(planned) != 2 {
		t.Fatalf("capped planned=%d, want 2", len(planned))
	}
}
