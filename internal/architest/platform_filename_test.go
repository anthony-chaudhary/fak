package architest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
)

// TestPlatformFilenamesCarryBuildLines is the whole-tree half of the 2026-10-10 floor: commit
// f9d4cca4 added internal/ctxmmu/closed_windows.go, which Go read as GOOS=windows, so `fileWindows`
// was undefined on every Linux build of internal/ctxmmu and its importers. Every .go file whose
// basename ends in a GOOS/GOARCH element must also name that platform in a //go:build line (or sit
// on buildwitness.PlatformNameAllowlist with a reviewed reason). Existing intentional files pass
// because they carry the line, not because they are on a frozen list; the synthetic half lives in
// internal/buildwitness/platformname_test.go and the commit-time half in internal/validate.
func TestPlatformFilenamesCarryBuildLines(t *testing.T) {
	root := repoRoot(t)
	rels, err := buildwitness.ScanPlatformFilenames(root)
	if err != nil {
		t.Fatalf("scan platform filenames under %s: %v", root, err)
	}
	bad, err := buildwitness.CheckPlatformFilenames(root, rels)
	if err != nil {
		t.Fatalf("check platform filenames: %v", err)
	}
	for _, v := range bad {
		t.Errorf("implicit platform constraint (closed_windows.go class): %s", v)
	}
	scanned := map[string]bool{}
	for _, rel := range rels {
		scanned[rel] = true
	}
	for rel := range buildwitness.PlatformNameAllowlist {
		if !scanned[rel] {
			t.Errorf("stale PlatformNameAllowlist entry %s: no such platform-suffixed file; remove it", rel)
			continue
		}
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err == nil && buildwitness.CheckPlatformFilename(rel, src) == nil {
			t.Errorf("stale PlatformNameAllowlist entry %s: the file now carries its build line; remove it", rel)
		}
	}
	if len(rels) == 0 {
		t.Fatal("scan found no platform-suffixed .go files; the tree has intentional ones (e.g. cmd/fak/*_windows.go), so the scanner is broken")
	}
}
