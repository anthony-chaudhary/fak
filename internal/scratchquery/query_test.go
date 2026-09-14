package scratchquery

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeScratch(t *testing.T, root, rel, content string) string {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return full
}

func TestQueryFindsLiteralHitsSorted(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "b.txt", "alpha\nneedle here\nomega\n")
	writeScratch(t, root, "a.txt", "needle first\nsecond\n")

	res, err := Query(root, QueryOptions{Pattern: "needle"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Hits) != 2 {
		t.Fatalf("hits = %#v, want 2", res.Hits)
	}
	if res.Hits[0].File != "a.txt" || res.Hits[0].Line != 1 {
		t.Fatalf("first hit = %#v, want a.txt:1", res.Hits[0])
	}
	if res.Hits[1].File != "b.txt" || res.Hits[1].Line != 2 {
		t.Fatalf("second hit = %#v, want b.txt:2", res.Hits[1])
	}
	if !strings.Contains(res.Hits[0].Match, "needle first") {
		t.Fatalf("match = %q", res.Hits[0].Match)
	}
	if res.Matched != 2 || res.Files != 2 {
		t.Fatalf("matched=%d files=%d, want 2/2", res.Matched, res.Files)
	}
}

func TestQueryMissingRootIsCleanEmpty(t *testing.T) {
	res, err := Query(filepath.Join(t.TempDir(), "absent"), QueryOptions{Pattern: "x"})
	if err != nil {
		t.Fatalf("missing root errored: %v", err)
	}
	if len(res.Hits) != 0 || res.Files != 0 {
		t.Fatalf("res = %#v, want empty", res)
	}
}

func TestQuerySkipsBinaryAndLarge(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "bin.dat", "needle\x00binary")
	writeScratch(t, root, "ok.txt", "needle\n")
	// A large file above the cap is skipped by size, never read.
	big := strings.Repeat("needle\n", (MaxFileBytes/7)+10)
	writeScratch(t, root, "big.txt", big)

	res, err := Query(root, QueryOptions{Pattern: "needle"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Hits) != 1 || res.Hits[0].File != "ok.txt" {
		t.Fatalf("hits = %#v, want only ok.txt", res.Hits)
	}
	reasons := map[string]bool{}
	for _, s := range res.Skipped {
		reasons[s.Reason] = true
	}
	if !reasons[SkipBinary] || !reasons[SkipTooLarge] {
		t.Fatalf("skipped = %#v, want binary+too_large", res.Skipped)
	}
}

func TestQueryLimitTruncates(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "many.txt", "hit\nhit\nhit\nhit\n")
	res, err := Query(root, QueryOptions{Pattern: "hit", Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Hits) != 2 || !res.Truncated {
		t.Fatalf("res = %#v, want 2 truncated", res)
	}
}

func TestFileReadsBoundedContent(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "note.md", "line one\nline two\n")
	lines, err := File(root, "note.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 2 || lines[0] != "line one" {
		t.Fatalf("lines = %#v", lines)
	}
	missing, err := File(root, "nope.md")
	if err != nil || missing != nil {
		t.Fatalf("missing file = %#v, %v; want nil,nil", missing, err)
	}
	if _, err := File(root, "../escape"); err == nil {
		t.Fatal("expected escape refusal")
	}
}
