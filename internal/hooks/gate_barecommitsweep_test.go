package hooks

import (
	"strings"
	"testing"
)

// gate_barecommitsweep_test.go — unit tests for the BARE_COMMIT_SWEEP gate (#3615). The gate
// reads only d.StagedPaths + the handshake/escape env, so these drive it over a hand-built
// StagedDiff with no git.

func TestGateBareCommitSweep_UnvettedStagedSetFires(t *testing.T) {
	d := &StagedDiff{StagedPaths: []string{"internal/b/x.go", "internal/a/y.go"}}
	findings, err := gateBareCommitSweep(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("an unvetted staged set should fire exactly one finding, got %d: %+v", len(findings), findings)
	}
	f := findings[0]
	if f.Gate != "BARE_COMMIT_SWEEP" {
		t.Fatalf("gate name = %q, want BARE_COMMIT_SWEEP", f.Gate)
	}
	// File anchors to the first path in sorted order (deterministic).
	if f.File != "internal/a/y.go" {
		t.Fatalf("finding should anchor to the first sorted path, got %q", f.File)
	}
	// Detail names both swept paths and the count.
	for _, want := range []string{"internal/a/y.go", "internal/b/x.go", "2 staged path"} {
		if !strings.Contains(f.Detail, want) {
			t.Fatalf("detail should mention %q; got %q", want, f.Detail)
		}
	}
}

func TestGateBareCommitSweep_VettedMarkerStandsDown(t *testing.T) {
	t.Setenv(bareCommitVettedEnv, "1")
	d := &StagedDiff{StagedPaths: []string{"internal/a/y.go"}}
	findings, err := gateBareCommitSweep(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("a vetted commit (marker set) must not fire, got %+v", findings)
	}
}

func TestGateBareCommitSweep_EmptyStagedSetClean(t *testing.T) {
	if findings, err := gateBareCommitSweep(&StagedDiff{}); err != nil || len(findings) != 0 {
		t.Fatalf("an empty staged set sweeps nothing; got findings=%+v err=%v", findings, err)
	}
}

func TestGateBareCommitSweep_FamilyOffDisables(t *testing.T) {
	t.Setenv(bareCommitPreStagedEnv, "off")
	d := &StagedDiff{StagedPaths: []string{"internal/a/y.go"}}
	if findings, err := gateBareCommitSweep(d); err != nil || len(findings) != 0 {
		t.Fatalf("FAK_PRESTAGED_PATH_GUARD=off must disable the gate; got findings=%+v err=%v", findings, err)
	}
}

func TestGateBareCommitSweep_ListIsCappedWithOverflow(t *testing.T) {
	paths := make([]string, bareCommitListCap+5)
	for i := range paths {
		// zero-padded so sort order is stable and the overflow count is deterministic.
		paths[i] = "internal/pkg/f" + string(rune('a'+i%26)) + "_" + itoa2(i) + ".go"
	}
	f, err := gateBareCommitSweep(&StagedDiff{StagedPaths: paths})
	if err != nil || len(f) != 1 {
		t.Fatalf("expected one finding, got %+v err=%v", f, err)
	}
	if !strings.Contains(f[0].Detail, "(+5 more)") {
		t.Fatalf("detail should cap the list and report the overflow count; got %q", f[0].Detail)
	}
	if !strings.Contains(f[0].Detail, "17 staged path") { // cap 12 + 5 overflow = 17 total
		t.Fatalf("detail should report the full staged count (17); got %q", f[0].Detail)
	}
}

// itoa2 renders i as a 2-digit zero-padded decimal (00..99) — a tiny helper so the cap test's
// synthetic paths sort lexicographically in numeric order without importing strconv here.
func itoa2(i int) string {
	return string(rune('0'+(i/10)%10)) + string(rune('0'+i%10))
}
