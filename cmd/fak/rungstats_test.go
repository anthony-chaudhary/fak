package main

import (
	"bytes"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// TestCmdRungStatsTable is the golden assertion for `fak rungstats`: folding the
// fixed probe set through the real adjudicator chain must print the exact
// rung-decision distribution table in testdata/rungstats.golden. The probe verdicts
// are stable under the built-in DefaultPolicy (three allows, then one tool per
// distinct deny reason), every cheaper rung defers, and rungobs.Observer.Snapshot
// sorts deterministically — so the table is byte-stable. We pin adjudicator.Default
// to the default floor first so a sibling test that swapped in a custom policy
// (attest_test.go / main_test.go) cannot make this order-dependent.
func TestCmdRungStatsTable(t *testing.T) {
	adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy())
	ifc.Default.Reset("")
	t.Cleanup(func() { ifc.Default.Reset("") })

	var out bytes.Buffer
	if rc := runRungStats(&out, nil); rc != 0 {
		t.Fatalf("runRungStats rc=%d, want 0; output:\n%s", rc, out.String())
	}

	// Regenerate after an intentional column/probe change with UPDATE_GOLDEN=1.
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile("testdata/rungstats.golden", out.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	want, err := os.ReadFile("testdata/rungstats.golden")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if got := out.Bytes(); !bytes.Equal(got, want) {
		t.Fatalf("rungstats table mismatch:\n--- got ---\n%s\n--- want (testdata/rungstats.golden) ---\n%s", got, want)
	}
}
