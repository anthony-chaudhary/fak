package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRenderPagePreservesGeneratedDocumentShell(t *testing.T) {
	surface := []byte("# fak verb surface (generated)\n\nintro\n" + string(surfaceTableMarker) + "|---|---|---|---|---|---|---|\n")
	got, err := renderPage(surface)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(got, []byte(pageFrontmatter+"# fak verb surface (generated)\n")) {
		t.Fatalf("generated page lost required frontmatter:\n%s", got)
	}
	if bytes.Count(got, []byte("\n## Surface table\n")) != 1 {
		t.Fatalf("generated page surface heading count != 1:\n%s", got)
	}

	if _, err := renderPage([]byte("no table")); err == nil {
		t.Fatal("render accepted a page with no verb table marker")
	}
}

func TestPrintDoesNotTouchDiskAndCheckDetectsDrift(t *testing.T) {
	root := repoRoot(t)
	target := filepath.Join(root, filepath.FromSlash(outputPath))
	before, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if code := run([]string{"-root", root, "-print"}); code != 0 {
		t.Fatalf("print code=%d", code)
	}
	after, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if !before.ModTime().Equal(after.ModTime()) || before.Size() != after.Size() {
		t.Fatal("-print touched generated page")
	}

	scratch := t.TempDir()
	if err := os.MkdirAll(filepath.Join(scratch, "docs", "generated"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A non-repository root must fail rather than blessing an empty source tree.
	if code := run([]string{"-root", scratch}); code == 0 {
		t.Fatal("check accepted an empty source tree")
	}
}

func TestCheckedInPageIsFresh(t *testing.T) {
	if code := run([]string{"-root", repoRoot(t)}); code != 0 {
		t.Fatalf("check code=%d", code)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}
