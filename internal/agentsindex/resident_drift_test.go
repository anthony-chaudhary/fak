package agentsindex

import (
	"os"
	"path/filepath"
	"testing"
)

// resident_drift_test.go is the live-tree drift gate promised in toc.go (#3535, epic
// #3229): it re-derives the resident TOC block from the REAL repo AGENTS.md and fails
// CLOSED if the block committed into CLAUDE.md has drifted from it. This is the guarantee
// that the compact resident map a session holds never silently diverges from the source
// doc it stands in for — the same fail-closed discipline resident_budget_test.go applies
// to the block's size, applied here to its content.
//
// It SKIPS (never fails) when the repo root can't be located, when CLAUDE.md is
// unreadable, or when CLAUDE.md carries no resident markers yet: the block is written by
// `fak index agents --write-resident`, and until that writer has run there is nothing to
// drift against. Once the markers are present the gate is live and any divergence between
// CLAUDE.md's block and a freshly rendered one is a hard failure. This mirrors the
// devindex freshness tests, which skip on a packaged build with no dos.toml.

// repoClaudeMD locates root/CLAUDE.md via the same dos.toml walk the budget gate uses,
// returning its bytes. ok=false with a t.Skip already issued when it cannot be reached.
func repoClaudeMD(t *testing.T) (path string, claude []byte, ok bool) {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	root, found := FindRoot(wd)
	if !found {
		t.Skipf("no dos.toml above %s; skipping resident drift gate", wd)
		return "", nil, false
	}
	p := filepath.Join(root, "CLAUDE.md")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Skipf("CLAUDE.md not readable under %s: %v", root, err)
		return "", nil, false
	}
	return p, b, true
}

// TestResidentBlockMatchesAgents is the drift gate: the CLAUDE.md resident block must be
// byte-identical to what RenderTOC produces from the current AGENTS.md.
func TestResidentBlockMatchesAgents(t *testing.T) {
	_, claude, ok := repoClaudeMD(t)
	if !ok {
		return
	}
	cur, present := ExtractResident(claude)
	if !present {
		t.Skipf("CLAUDE.md carries no %s marker yet; resident block unwritten "+
			"(run `fak index agents --write-resident`)", ResidentBegin)
	}
	want := loadRealAgents(t).ResidentBlock()
	if cur != want {
		t.Fatalf("resident AGENTS.md TOC in CLAUDE.md has drifted from the source doc; " +
			"re-run `fak index agents --write-resident` to regenerate the marker-bounded block")
	}
}

// TestResidentDriftGateActivatesWhenWritten proves the gate is not vacuous: given a
// CLAUDE.md that DOES carry the freshly rendered block, the same comparison passes; given
// one whose block has been mutated, it would fail. This runs without touching the repo
// CLAUDE.md, so it exercises the gate's logic even while the live block is still unwritten.
func TestResidentDriftGateActivatesWhenWritten(t *testing.T) {
	d := loadRealAgents(t)
	fresh := []byte("# Repo\n\n" + d.ResidentBlock() + "\n")
	cur, present := ExtractResident(fresh)
	if !present {
		t.Fatal("ExtractResident found no block in a freshly spliced CLAUDE.md")
	}
	if cur != d.ResidentBlock() {
		t.Fatal("a freshly rendered block must match itself (gate would false-positive)")
	}
	// A one-byte mutation inside the block must be caught (gate is not vacuous).
	mutated := []byte("# Repo\n\n" + ResidentBegin + "\nTAMPERED\n" + ResidentEnd + "\n")
	if got, _ := ExtractResident(mutated); got == d.ResidentBlock() {
		t.Fatal("drift gate failed to distinguish a tampered block from the rendered one")
	}
}
