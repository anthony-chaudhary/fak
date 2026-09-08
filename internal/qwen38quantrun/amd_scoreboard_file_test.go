package qwen38quantrun

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildAMDScoreboardFileWritesTypedReport(t *testing.T) {
	dir := t.TempDir()
	inputPath, reportPath := filepath.Join(dir, "input.json"), filepath.Join(dir, "report.json")
	raw, err := json.Marshal(validAMDScoreboardInput())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inputPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	report, err := BuildAMDScoreboardFile(inputPath, reportPath)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Comparable {
		t.Fatalf("report=%+v", report)
	}
	written, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AMDScoreboardReport
	if err := json.Unmarshal(written, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAMDScoreboardReport(decoded); err != nil {
		t.Fatal(err)
	}
}

func TestBuildAMDScoreboardFileRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.json")
	if err := os.WriteFile(inputPath, []byte(`{"schema":"fak.qwen38.amd-scoreboard-input.v1","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildAMDScoreboardFile(inputPath, filepath.Join(dir, "report.json")); err == nil {
		t.Fatal("expected strict decode failure")
	}
}

func TestStrixHaloComparisonLedgerFailsClosedAndRendersIndex(t *testing.T) {
	ledgerPath := filepath.Join("..", "..", "docs", "benchmarks", "strix-halo-comparison-ladder.json")
	raw, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatal(err)
	}
	var ledger struct {
		Schema   string `json:"schema"`
		Status   string `json:"status"`
		Contract struct {
			Dispositions          []string `json:"dispositions"`
			RequiredForComparable []string `json:"required_for_comparable"`
			HardExclusions        []string `json:"hard_exclusions"`
			ContextOnlyBoundaries []string `json:"context_only_boundaries"`
		} `json:"contract"`
		Rows             []map[string]any `json:"rows"`
		NegativeFixtures []struct {
			ID                   string   `json:"id"`
			AttemptedDisposition string   `json:"attempted_disposition"`
			OmittedRequiredPaths []string `json:"omitted_required_paths"`
			ViolatedBoundary     string   `json:"violated_boundary"`
			Expected             struct {
				Accepted     bool   `json:"accepted"`
				RatioEmitted bool   `json:"ratio_emitted"`
				Reason       string `json:"reason"`
			} `json:"expected"`
		} `json:"negative_fixtures"`
	}
	if err := json.Unmarshal(raw, &ledger); err != nil {
		t.Fatal(err)
	}
	if ledger.Schema != "fak.benchmark.strix-halo-comparison-ladder/v1" || ledger.Status != "NO_COMPARABLE_LOCAL_RESULT" {
		t.Fatalf("ledger identity = %q/%q", ledger.Schema, ledger.Status)
	}
	if len(ledger.Rows) != 9 || len(ledger.NegativeFixtures) != 7 {
		t.Fatalf("ledger rows/negative fixtures = %d/%d, want 9/7", len(ledger.Rows), len(ledger.NegativeFixtures))
	}

	required := stringSet(ledger.Contract.RequiredForComparable)
	dispositions := stringSet(ledger.Contract.Dispositions)
	hardExclusions := stringSet(ledger.Contract.HardExclusions)
	contextOnly := stringSet(ledger.Contract.ContextOnlyBoundaries)
	fixtureIDs := make(map[string]bool, len(ledger.NegativeFixtures))
	for _, fixture := range ledger.NegativeFixtures {
		if fixture.ID == "" || fixtureIDs[fixture.ID] {
			t.Fatalf("empty or duplicate negative fixture id %q", fixture.ID)
		}
		fixtureIDs[fixture.ID] = true
		if fixture.AttemptedDisposition != "COMPARABLE" || fixture.Expected.Accepted || fixture.Expected.RatioEmitted || fixture.Expected.Reason == "" {
			t.Fatalf("negative fixture %q is not fail-closed: %+v", fixture.ID, fixture.Expected)
		}

		candidate := make(map[string]any)
		for path := range required {
			setComparisonPath(candidate, path, "fixture-value")
		}
		for _, path := range fixture.OmittedRequiredPaths {
			if !required[path] {
				t.Fatalf("negative fixture %q omits non-required path %q", fixture.ID, path)
			}
			deleteComparisonPath(candidate, path)
		}
		if comparisonCandidateAccepted(candidate, required, hardExclusions, contextOnly, fixture.ViolatedBoundary) {
			t.Fatalf("negative fixture %q was accepted", fixture.ID)
		}
	}
	for _, id := range []string{
		"reject-comparable-missing-quality",
		"reject-comparable-missing-source-revision",
		"reject-comparable-missing-artifact",
		"reject-comparable-missing-workload",
		"reject-comparable-missing-engine",
		"reject-comparable-sparse-moe-as-dense",
		"reject-comparable-corrupt-output",
	} {
		if !fixtureIDs[id] {
			t.Errorf("missing negative fixture %q", id)
		}
	}

	var table strings.Builder
	table.WriteString("| Disposition | Source row | Frozen point | Why no ratio is emitted |\n")
	table.WriteString("|---|---|---:|---|\n")
	for _, row := range ledger.Rows {
		id := comparisonString(t, row, "id")
		disposition := comparisonString(t, row, "disposition")
		if !dispositions[disposition] {
			t.Errorf("row %q has unknown disposition %q", id, disposition)
		}
		ratio, exists := row["ratio"]
		if !exists || disposition != "COMPARABLE" && ratio != nil {
			t.Errorf("row %q must carry ratio:null outside COMPARABLE", id)
		}
		if disposition == "COMPARABLE" && !comparisonCandidateAccepted(row, required, hardExclusions, contextOnly, "") {
			t.Errorf("row %q is incomplete but marked COMPARABLE", id)
		}
		fmt.Fprintf(&table, "| **%s** | `%s` | %s | %s |\n",
			disposition,
			id,
			comparisonString(t, row, "index_display.frozen_point"),
			comparisonString(t, row, "index_display.why_no_ratio"),
		)
	}
	index, err := os.ReadFile(filepath.Join("..", "..", "docs", "benchmarks", "QWEN-PERFORMANCE-INDEX.md"))
	if err != nil {
		t.Fatal(err)
	}
	normalizedIndex := strings.ReplaceAll(string(index), "\r\n", "\n")
	if !strings.Contains(normalizedIndex, table.String()) {
		t.Fatalf("Qwen performance index Strix table drifted from ledger:\n%s", table.String())
	}
}

func comparisonCandidateAccepted(candidate map[string]any, required, hardExclusions, contextOnly map[string]bool, violation string) bool {
	for path := range required {
		value, ok := comparisonPath(candidate, path)
		if !ok || value == nil || value == "" {
			return false
		}
	}
	return !hardExclusions[violation] && !contextOnly[violation]
}

func comparisonString(t *testing.T, row map[string]any, path string) string {
	t.Helper()
	value, ok := comparisonPath(row, path)
	text, stringOK := value.(string)
	if !ok || !stringOK || text == "" {
		t.Fatalf("row field %q is missing or not a string", path)
	}
	return text
}

func comparisonPath(root map[string]any, path string) (any, bool) {
	var current any = root
	for _, part := range strings.Split(path, ".") {
		object, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		current, ok = object[part]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

func setComparisonPath(root map[string]any, path string, value any) {
	parts := strings.Split(path, ".")
	current := root
	for _, part := range parts[:len(parts)-1] {
		next, ok := current[part].(map[string]any)
		if !ok {
			next = make(map[string]any)
			current[part] = next
		}
		current = next
	}
	current[parts[len(parts)-1]] = value
}

func deleteComparisonPath(root map[string]any, path string) {
	parts := strings.Split(path, ".")
	current := root
	for _, part := range parts[:len(parts)-1] {
		next, ok := current[part].(map[string]any)
		if !ok {
			return
		}
		current = next
	}
	delete(current, parts[len(parts)-1])
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
