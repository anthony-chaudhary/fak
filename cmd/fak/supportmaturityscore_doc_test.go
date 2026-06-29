package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/supportmaturityscore"
)

// TestSupportMaturityMatrixDocFresh is the #1255 witness: the committed support-maturity
// matrix block in docs/HARDWARE-MATRIX.md equals the block regenerated from the live
// covmatrix grid. It reds the trunk the moment a model/backend change makes a cell stale,
// the same freshness fence the other scorecards ride. Regenerate with
// `go run ./cmd/fak support-maturity-scorecard --write-doc`.
func TestSupportMaturityMatrixDocFresh(t *testing.T) {
	root := repoRootFromTest(t)
	var out, errb bytes.Buffer
	code := runSupportMaturityScorecard(&out, &errb, []string{"--workspace", root, "--check-doc"})
	if code != 0 {
		t.Fatalf("--check-doc on the real tree returned %d, want 0 (run `go run ./cmd/fak support-maturity-scorecard --write-doc`)\nstdout:\n%s\nstderr:\n%s",
			code, out.String(), errb.String())
	}
	if !strings.HasPrefix(out.String(), "OK") {
		t.Fatalf("--check-doc OK output missing; got:\n%s", out.String())
	}
}

// TestSupportMaturityMatrixDocDetectsStaleCell proves the freshness gate fires: a doc with
// one mutated cell verdict reds --check-doc. This is the "a stale cell reds the freshness
// check" half of the witness.
func TestSupportMaturityMatrixDocDetectsStaleCell(t *testing.T) {
	root := repoRootFromTest(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "HARDWARE-MATRIX.md"))
	if err != nil {
		t.Fatalf("read real doc: %v", err)
	}
	// Flip the first table cell verdict; " SUPPORTED |" only matches a table cell, never the
	// legend prose (which writes "SUPPORTED (runs ...").
	stale := strings.Replace(string(raw), " SUPPORTED |", " UNDEFINED |", 1)
	if stale == string(raw) {
		t.Fatal("no SUPPORTED table cell found to mutate")
	}

	fake := t.TempDir()
	if err := os.MkdirAll(filepath.Join(fake, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fake, "docs", "HARDWARE-MATRIX.md"), []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSupportMaturityScorecard(&out, &errb, []string{"--workspace", fake, "--check-doc"})
	if code == 0 {
		t.Fatalf("--check-doc passed on a doc with a mutated cell; want stale exit 1\n%s", out.String())
	}
	if !strings.Contains(out.String(), "STALE") {
		t.Fatalf("stale output should say STALE; got:\n%s", out.String())
	}
}

// TestSupportMaturityMatrixBlockDeterministic guards the generator: two builds at one commit
// must be byte-identical, so the committed snapshot regenerates with no diff.
func TestSupportMaturityMatrixBlockDeterministic(t *testing.T) {
	if supportmaturityscore.MatrixBlock() != supportmaturityscore.MatrixBlock() {
		t.Fatal("MatrixBlock() is not deterministic across two calls")
	}
}

// TestSupportMaturityMatrixMDRenders proves --matrix-md emits the bounded block with a row
// per model family.
func TestSupportMaturityMatrixMDRenders(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSupportMaturityScorecard(&out, &errb, []string{"--matrix-md"}); code != 0 {
		t.Fatalf("--matrix-md returned %d\nstderr:\n%s", code, errb.String())
	}
	md := out.String()
	if !strings.Contains(md, supportmaturityscore.MatrixBegin) || !strings.Contains(md, supportmaturityscore.MatrixEnd) {
		t.Fatalf("--matrix-md missing markers; got:\n%s", md)
	}
	if !strings.Contains(md, "| Model family | Topology |") {
		t.Fatalf("--matrix-md missing table header; got:\n%s", md)
	}
}
