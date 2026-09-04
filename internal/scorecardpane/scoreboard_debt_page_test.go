package scorecardpane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScoreboardDebtPageCategoryCount(t *testing.T) {
	if len(DebtCategories) < 20 {
		t.Fatalf("expected at least 20 debt categories, got %d", len(DebtCategories))
	}
}

func TestScoreboardDebtPageFreshOnTrunk(t *testing.T) {
	root := findRepoRoot()
	ok, msg, err := CheckScoreboardDebtDoc(root)
	if err != nil {
		t.Fatalf("CheckScoreboardDebtDoc error: %v", err)
	}
	if !ok {
		t.Fatalf("Scoreboard debt doc is not fresh on trunk: %s", msg)
	}
}

func TestScoreboardDebtPageStaleDetection(t *testing.T) {
	tempDir := t.TempDir()
	baselinePath := filepath.Join(tempDir, filepath.FromSlash(BaselineRel))
	if err := os.MkdirAll(filepath.Dir(baselinePath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(baselinePath, []byte(`{"schema":"fak-scorecard-control-pane.baseline/1","commit":"test","total_debt":10,"grade_debt":2,"metrics":{"code":10},"grade_weights":{"code":2}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	docPath := filepath.Join(tempDir, filepath.FromSlash(ScoreboardDebtDocRel))
	if err := os.MkdirAll(filepath.Dir(docPath), 0o755); err != nil {
		t.Fatal(err)
	}

	// Missing file should report stale
	ok, msg, err := CheckScoreboardDebtDoc(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if ok || !strings.Contains(msg, "not found") {
		t.Fatalf("expected missing doc to report stale not found, got ok=%v, msg=%s", ok, msg)
	}

	// Mutated content should report stale
	if err := os.WriteFile(docPath, []byte("mutated content"), 0o644); err != nil {
		t.Fatal(err)
	}
	ok, msg, err = CheckScoreboardDebtDoc(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if ok || !strings.Contains(msg, "drifted") {
		t.Fatalf("expected mutated doc to report drifted stale, got ok=%v, msg=%s", ok, msg)
	}

	// WriteScoreboardDebtDoc should restore freshness
	changed, err := WriteScoreboardDebtDoc(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatalf("expected WriteScoreboardDebtDoc to make changes")
	}
	ok, msg, err = CheckScoreboardDebtDoc(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("expected doc to be fresh after WriteScoreboardDebtDoc, got msg=%s", msg)
	}
}
