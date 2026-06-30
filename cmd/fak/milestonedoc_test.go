package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/milestonedoc"
)

// TestMilestoneStatusDocFresh is the #1441 CI witness: the committed milestone-climb
// block in docs/milestones/STATUS.md equals the block regenerated from the live
// covmatrix grid. It reds the trunk the moment a model/backend support rung changes and
// makes a committed cell stale, the same freshness fence the support-maturity matrix
// rides. Regenerate with `go run ./cmd/fak milestone status-doc --write-doc`.
func TestMilestoneStatusDocFresh(t *testing.T) {
	root := repoRootFromTest(t)
	var out, errb bytes.Buffer
	code := runMilestoneStatusDoc(&out, &errb, []string{"--workspace", root, "--check-doc"})
	if code != 0 {
		t.Fatalf("--check-doc on the real tree returned %d, want 0 (run `go run ./cmd/fak milestone status-doc --write-doc`)\nstdout:\n%s\nstderr:\n%s",
			code, out.String(), errb.String())
	}
	if !strings.HasPrefix(out.String(), "OK") {
		t.Fatalf("--check-doc OK output missing; got:\n%s", out.String())
	}
}

// TestMilestoneStatusDocDetectsStaleCell proves the freshness gate fires: a doc with one
// mutated ladder cell count reds --check-doc. This is the "a stale cell reds the
// freshness check" half of the witness.
func TestMilestoneStatusDocDetectsStaleCell(t *testing.T) {
	root := repoRootFromTest(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "milestones", "STATUS.md"))
	if err != nil {
		t.Fatalf("read real doc: %v", err)
	}
	stale := string(raw)
	for _, swap := range []struct{ from, to string }{
		{" | 0 |", " | 99 |"},
		{" | 13 |", " | 99 |"},
		{" | 19 |", " | 99 |"},
		{" | 24 |", " | 99 |"},
	} {
		if strings.Contains(stale, swap.from) {
			stale = strings.Replace(stale, swap.from, swap.to, 1)
			break
		}
	}
	if stale == string(raw) {
		t.Fatal("no ladder cell value found to mutate")
	}

	fake := t.TempDir()
	if err := os.MkdirAll(filepath.Join(fake, "docs", "milestones"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fake, "docs", "milestones", "STATUS.md"), []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runMilestoneStatusDoc(&out, &errb, []string{"--workspace", fake, "--check-doc"})
	if code == 0 {
		t.Fatalf("--check-doc passed on a doc with a mutated cell; want stale exit 1\n%s", out.String())
	}
	if !strings.Contains(out.String(), "STALE") {
		t.Fatalf("stale output should say STALE; got:\n%s", out.String())
	}
}

// TestMilestoneStatusDocWriteRoundTrip proves --write-doc into a fresh temp dir creates
// the scaffolded doc and a follow-up --check-doc accepts it.
func TestMilestoneStatusDocWriteRoundTrip(t *testing.T) {
	fake := t.TempDir()
	var out, errb bytes.Buffer
	if code := runMilestoneStatusDoc(&out, &errb, []string{"--workspace", fake, "--write-doc"}); code != 0 {
		t.Fatalf("--write-doc returned %d\nstderr:\n%s", code, errb.String())
	}
	doc, err := os.ReadFile(filepath.Join(fake, "docs", "milestones", "STATUS.md"))
	if err != nil {
		t.Fatalf("write-doc did not create the doc: %v", err)
	}
	if !milestonedoc.Fresh(string(doc)) {
		t.Fatalf("written doc is not Fresh:\n%s", doc)
	}
	out.Reset()
	errb.Reset()
	if code := runMilestoneStatusDoc(&out, &errb, []string{"--workspace", fake, "--check-doc"}); code != 0 {
		t.Fatalf("--check-doc after --write-doc returned %d\nstdout:\n%s", code, out.String())
	}
}

// TestMilestoneStatusDocBlock proves --block emits the bounded generated block.
func TestMilestoneStatusDocBlock(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runMilestoneStatusDoc(&out, &errb, []string{"--block"}); code != 0 {
		t.Fatalf("--block returned %d\nstderr:\n%s", code, errb.String())
	}
	md := out.String()
	if !strings.Contains(md, milestonedoc.Begin) || !strings.Contains(md, milestonedoc.End) {
		t.Fatalf("--block missing markers; got:\n%s", md)
	}
}
