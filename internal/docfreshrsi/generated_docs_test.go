package docfreshrsi

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/milestonedoc"
	"github.com/anthony-chaudhary/fak/internal/supportmaturityscore"
)

func TestDefaultGeneratedDocsRegistry(t *testing.T) {
	docs := DefaultGeneratedDocs()
	if len(docs) == 0 {
		t.Fatal("expected registered generated docs, got empty")
	}
	seen := make(map[string]bool)
	for _, d := range docs {
		if d.ID == "" {
			t.Error("doc entry has empty ID")
		}
		if d.Path == "" {
			t.Errorf("doc %s has empty path", d.ID)
		}
		if d.Check == nil {
			t.Errorf("doc %s has nil Check", d.ID)
		}
		if d.Refresh == nil {
			t.Errorf("doc %s has nil Refresh", d.ID)
		}
		if seen[d.ID] {
			t.Errorf("duplicate doc ID %s", d.ID)
		}
		seen[d.ID] = true
	}
}

func TestAuditAndRefreshGeneratedDocsInTempDir(t *testing.T) {
	tmp := t.TempDir()

	// 1. Create a fake STATUS.md with drifted milestone block
	statusDir := filepath.Join(tmp, "docs", "milestones")
	if err := os.MkdirAll(statusDir, 0755); err != nil {
		t.Fatal(err)
	}
	statusFile := filepath.Join(statusDir, "STATUS.md")
	driftedContent := "# Milestone status\n\n" +
		milestonedoc.Begin + "\n\n" +
		"STALE DRIFT CONTENT\n\n" +
		milestonedoc.End + "\n"
	if err := os.WriteFile(statusFile, []byte(driftedContent), 0644); err != nil {
		t.Fatal(err)
	}

	// 2. Create a fake HARDWARE-MATRIX.md with drifted content
	matrixFile := filepath.Join(tmp, "docs", "HARDWARE-MATRIX.md")
	driftedMatrix := "# Hardware Matrix\n\n" +
		supportmaturityscore.MatrixBegin + "\n\n" +
		"STALE MATRIX CONTENT\n\n" +
		supportmaturityscore.MatrixEnd + "\n"
	if err := os.WriteFile(matrixFile, []byte(driftedMatrix), 0644); err != nil {
		t.Fatal(err)
	}

	// 3. Audit should detect staleness in STATUS.md and HARDWARE-MATRIX.md
	statuses, allFresh := AuditGeneratedDocs(tmp)
	if allFresh {
		t.Fatal("expected allFresh=false on drifted temp tree")
	}

	var statusMap = make(map[string]GeneratedDocStatus)
	for _, s := range statuses {
		statusMap[s.ID] = s
	}

	if statusMap["milestone-status"].Fresh {
		t.Errorf("expected milestone-status to be stale, got fresh")
	}
	if statusMap["hardware-matrix"].Fresh {
		t.Errorf("expected hardware-matrix to be stale, got fresh")
	}

	// 4. Run Refresh
	refreshedStatuses, err := RefreshGeneratedDocs(tmp)
	if err != nil {
		t.Fatalf("refresh failed: %v", err)
	}

	var refreshedMap = make(map[string]GeneratedDocStatus)
	for _, s := range refreshedStatuses {
		refreshedMap[s.ID] = s
	}

	if !refreshedMap["milestone-status"].Refreshed {
		t.Errorf("expected milestone-status to report refreshed=true")
	}
	if !refreshedMap["milestone-status"].Fresh {
		t.Errorf("expected milestone-status to be fresh after refresh")
	}
	if !refreshedMap["hardware-matrix"].Refreshed {
		t.Errorf("expected hardware-matrix to report refreshed=true")
	}
	if !refreshedMap["hardware-matrix"].Fresh {
		t.Errorf("expected hardware-matrix to be fresh after refresh")
	}

	// 5. Verify the files on disk now match expected
	b, err := os.ReadFile(statusFile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), milestonedoc.Block()) {
		t.Errorf("STATUS.md does not contain fresh milestonedoc.Block()")
	}

	mb, err := os.ReadFile(matrixFile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(mb), supportmaturityscore.MatrixBlock()) {
		t.Errorf("HARDWARE-MATRIX.md does not contain fresh supportmaturityscore.MatrixBlock()")
	}
}
