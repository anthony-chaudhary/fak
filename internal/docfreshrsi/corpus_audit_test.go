package docfreshrsi

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditCorpusDanglingLinksAndDefects(t *testing.T) {
	tmp := t.TempDir()

	doc1 := `# Guide One

Some intro text without an orientation signpost.
See [Guide Two](guide2.md#details).
`
	doc2 := `# Guide Two

> Orientation: Guide two orientation.

## Details

This is details.
Pinned to v0.10.0 release.

The latest version is available.

## Read next

- [guide1.md](guide1.md)
- [Broken Link](nonexistent.md#missing-anchor)
`
	subDir := filepath.Join(tmp, "docs")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "guide1.md"), []byte(doc1), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "guide2.md"), []byte(doc2), 0644); err != nil {
		t.Fatal(err)
	}
	// Code file on disk to test disk resolution
	codeFile := filepath.Join(tmp, "main.go")
	if err := os.WriteFile(codeFile, []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}
	doc3 := `# Guide Three

> Orientation: Guide three orientation.

Check the [code](../main.go) for details.

## Read next

- [guide1.md](guide1.md)
`
	if err := os.WriteFile(filepath.Join(subDir, "guide3.md"), []byte(doc3), 0644); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(tmp, "VERSION"), []byte("0.54.0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	c, err := ScanCorpus(tmp, nil)
	if err != nil {
		t.Fatalf("scan error: %v", err)
	}
	if len(c) != 3 {
		t.Fatalf("expected 3 docs, got %d", len(c))
	}

	targetVer := DefaultTargetVersion(tmp)
	if targetVer != "v0.54.0" {
		t.Errorf("target version = %q, want v0.54.0", targetVer)
	}

	opts := CorpusAuditOptions{
		Root:          tmp,
		TargetVersion: targetVer,
		CheckLinks:    true,
		CheckClaims:   true,
	}
	report := AuditCorpus(tmp, c, Target{Version: targetVer}, opts)

	if report.Fresh {
		t.Error("expected Fresh=false for corpus with defects and broken links")
	}
	if len(report.OrientationMissing) != 1 || report.OrientationMissing[0] != "docs/guide1.md" {
		t.Errorf("orientation missing = %v, want [docs/guide1.md]", report.OrientationMissing)
	}
	if len(report.ReadNextMissing) != 1 || report.ReadNextMissing[0] != "docs/guide1.md" {
		t.Errorf("read next missing = %v, want [docs/guide1.md]", report.ReadNextMissing)
	}
	if len(report.StalePins) != 1 || report.StalePins[0].Found != "v0.10.0" {
		t.Errorf("stale pins = %+v, want found v0.10.0", report.StalePins)
	}

	// Broken link must be identified
	foundDangling := false
	for _, dl := range report.DanglingLinks {
		if strings.Contains(dl, "nonexistent.md") {
			foundDangling = true
		}
	}
	if !foundDangling {
		t.Errorf("dangling links %v missing nonexistent.md", report.DanglingLinks)
	}

	// Link to main.go must not be dangling
	for _, dl := range report.DanglingLinks {
		if strings.Contains(dl, "main.go") {
			t.Errorf("main.go exists on disk but was reported as dangling: %s", dl)
		}
	}

	// Version claim detected
	if len(report.VersionClaims) == 0 {
		t.Errorf("expected version claims, got none")
	}
}

func TestRefreshCorpusAppliesKeptFixesToDisk(t *testing.T) {
	tmp := t.TempDir()

	doc1 := `# Guide One

> Orientation: Guide one orientation.

See [Guide Two](guide2.md#details).

## Read next

- [guide2.md](guide2.md)
`
	doc2 := `# Guide Two

> Orientation: Guide two orientation.

## Details

This is details.
Pinned to v0.10.0 release.

## Read next

- [guide1.md](guide1.md)
`
	subDir := filepath.Join(tmp, "docs")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "guide1.md"), []byte(doc1), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "guide2.md"), []byte(doc2), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "VERSION"), []byte("0.54.0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	c, err := ScanCorpus(tmp, nil)
	if err != nil {
		t.Fatal(err)
	}

	tgt := Target{Version: "v0.54.0"}

	// 1. Dry run
	dryReport, err := RefreshCorpus(tmp, c, tgt, true)
	if err != nil {
		t.Fatalf("dry run error: %v", err)
	}
	if len(dryReport.UpdatedDocs) == 0 {
		t.Fatal("expected updated docs in dry run, got 0")
	}

	bBefore, _ := os.ReadFile(filepath.Join(subDir, "guide2.md"))
	if !strings.Contains(string(bBefore), "v0.10.0") {
		t.Fatal("guide2.md was mutated during dry run")
	}

	// 2. Real refresh
	liveReport, err := RefreshCorpus(tmp, c, tgt, false)
	if err != nil {
		t.Fatalf("live refresh error: %v", err)
	}
	if liveReport.DebtAfter >= liveReport.DebtBefore {
		t.Errorf("debt did not decrease: before=%d after=%d", liveReport.DebtBefore, liveReport.DebtAfter)
	}

	bAfter, _ := os.ReadFile(filepath.Join(subDir, "guide2.md"))
	if !strings.Contains(string(bAfter), "v0.54.0") {
		t.Errorf("guide2.md after refresh does not contain v0.54.0:\n%s", string(bAfter))
	}
	if strings.Contains(string(bAfter), "v0.10.0") {
		t.Errorf("guide2.md still contains v0.10.0:\n%s", string(bAfter))
	}
}
