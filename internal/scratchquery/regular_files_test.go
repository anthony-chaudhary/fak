package scratchquery

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestScratchQueriesIgnoreSymlinkEntries(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const content = "blocked: local needle\n"
	writeScratch(t, root, "nested/local.md", content)
	outside := writeScratch(t, t.TempDir(), "outside.md", "blocked: external needle\n")
	if err := os.Symlink(outside, filepath.Join(root, "external.md")); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink creation unavailable: %v", err)
		}
		t.Fatal(err)
	}
	files, err := scratchFiles(root)
	if err != nil || !reflect.DeepEqual(files, []string{"nested/local.md"}) {
		t.Fatalf("scratchFiles = %#v, %v; want only nested/local.md", files, err)
	}
	got, err := Query(root, QueryOptions{Pattern: "needle"})
	want := QueryResult{
		Root: root, Pattern: "needle", Files: 1, Matched: 1,
		Hits:    []Hit{{File: "nested/local.md", Line: 1, Match: "blocked: local needle", Bytes: len(content)}},
		Skipped: []Skipped{},
	}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Query = %#v, %v; want %#v", got, err, want)
	}
	promoted, err := ScanPromotable(root, "session")
	wantPromoted := PromotableResult{
		Root: root, Scanned: 1,
		Candidates: []Promotable{{Kind: KindBlocked, Session: "session", Scratchpad: root, Path: "nested/local.md", Bytes: len(content)}},
	}
	if err != nil || !reflect.DeepEqual(promoted, wantPromoted) {
		t.Fatalf("ScanPromotable = %#v, %v; want %#v", promoted, err, wantPromoted)
	}
}
