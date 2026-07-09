package agentsindex

import (
	"os"
	"testing"
)

// resident_budget_test.go is the live-tree gate: it parses the REAL repo AGENTS.md and
// proves the lever is both real and bounded — the resident TOC fits its budget while the
// full doc is multiples larger. It fails CLOSED (a broken parse yielding few sections is
// a test failure, not a skip). It skips only when the repo root can't be located (e.g. a
// packaged build with no dos.toml), mirroring the devindex freshness tests.

func loadRealAgents(t *testing.T) *Doc {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	root, ok := FindRoot(wd)
	if !ok {
		t.Skipf("no dos.toml above %s; skipping live AGENTS.md gate", wd)
	}
	d, err := Load(root)
	if err != nil {
		t.Skipf("AGENTS.md not readable under %s: %v", root, err)
	}
	return d
}

func TestRealAgentsParsesEnoughSections(t *testing.T) {
	d := loadRealAgents(t)
	if len(d.Sections) < 5 {
		t.Fatalf("real AGENTS.md parsed into %d sections; want >=5 (fail-closed on a broken parse)", len(d.Sections))
	}
}

func TestResidentTOCWithinBudget(t *testing.T) {
	d := loadRealAgents(t)
	toc := d.RenderTOC()
	est := EstTokensOf(toc)
	if est > TOCBudgetTokens {
		t.Fatalf("resident TOC is ~%d est. tokens; budget is %d — trim the rows", est, TOCBudgetTokens)
	}
	if est == 0 {
		t.Fatalf("resident TOC is empty — fail closed")
	}
}

func TestLeverIsRealDocDwarfsTOC(t *testing.T) {
	d := loadRealAgents(t)
	tocEst := EstTokensOf(d.RenderTOC())
	docEst := d.EstTokens()
	if docEst < 3*tocEst {
		t.Fatalf("full doc ~%d est. tokens is not >=3x the TOC ~%d — the lever is not worth its floor tax", docEst, tocEst)
	}
}

func TestResidentBlockSpliceRoundTrips(t *testing.T) {
	d := loadRealAgents(t)
	claude := []byte("# Repo\n\nintro\n\n" + ResidentBegin + "\nOLD\n" + ResidentEnd + "\n\ntail\n")
	out, err := SpliceResident(claude, d.ResidentBlock())
	if err != nil {
		t.Fatalf("splice: %v", err)
	}
	got, ok := ExtractResident(out)
	if !ok {
		t.Fatal("extract after splice found no block")
	}
	if got != d.ResidentBlock() {
		t.Fatalf("spliced block != rendered block")
	}
	// idempotence: splicing the same block again is a no-op.
	out2, err := SpliceResident(out, d.ResidentBlock())
	if err != nil {
		t.Fatalf("second splice: %v", err)
	}
	if string(out2) != string(out) {
		t.Fatalf("splice not idempotent")
	}
}

func TestSpliceMissingMarkersErrors(t *testing.T) {
	if _, err := SpliceResident([]byte("no markers here"), "x"); err == nil {
		t.Fatalf("expected an error when markers are absent")
	}
}
