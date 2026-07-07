package issuefanout

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// The dogfood invocation ledger (#2515): every planner invocation appends one
// durable JSONL row, and a fold surfaces the outcome counts per ISO week. This
// drives the whole seam end to end — three real Build invocations (a success, a
// contract refusal, and a synthetic internal error) append rows through the
// BuildLogged/AppendRow seam, then FoldWeekly folds the very rows just written
// and the readout shows the real per-week counts. That captured readout is the
// witness the issue asks for.
func TestLedgerAppendAndWeeklyFold(t *testing.T) {
	var buf bytes.Buffer

	// Two invocations in one ISO week: a success and a deliberate refusal.
	wk1 := time.Date(2026, 6, 29, 9, 0, 0, 0, time.UTC) // Monday, ISO week 27
	if _, err := BuildLogged(spineInput(), wk1, &buf); err != nil {
		t.Fatalf("BuildLogged(valid): unexpected error %v", err)
	}
	badSpine := spineInput()
	badSpine.SpineRef = " "
	if _, err := BuildLogged(badSpine, wk1.Add(6*time.Hour), &buf); err == nil {
		t.Fatal("BuildLogged(empty spine) should refuse")
	}
	// One invocation the next ISO week: a synthetic non-Refusal internal error.
	wk2 := time.Date(2026, 7, 6, 8, 0, 0, 0, time.UTC) // Monday, ISO week 28
	if err := AppendRow(&buf, NewLedgerRow("issuefanout", wk2, errors.New("boom: unexpected internal failure"))); err != nil {
		t.Fatalf("AppendRow: %v", err)
	}

	// The fold surfaces counts per week, oldest first.
	weeks, err := FoldWeekly(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("FoldWeekly: %v", err)
	}
	if len(weeks) != 2 {
		t.Fatalf("want 2 ISO-week buckets, got %d: %+v", len(weeks), weeks)
	}
	if weeks[0].Week != "2026-W27" || weeks[0].Success != 1 || weeks[0].Refused != 1 || weeks[0].Total != 2 {
		t.Fatalf("week 1 fold wrong: %+v", weeks[0])
	}
	if weeks[1].Week != "2026-W28" || weeks[1].Error != 1 || weeks[1].Total != 1 {
		t.Fatalf("week 2 fold wrong: %+v", weeks[1])
	}

	// The readout is the witness: it shows the real per-week counts.
	readout := RenderWeekly(weeks)
	for _, want := range []string{"3 invocation(s) across 2 week(s)", "2026-W27", "2026-W28", "1 success, 1 refused", "1 error"} {
		if !strings.Contains(readout, want) {
			t.Fatalf("readout %q missing %q", readout, want)
		}
	}
	t.Logf("ledger weekly readout:\n%s", readout)

	// The committed rows must not leak a private boundary: only schema, an RFC3339
	// UTC timestamp, the lane token, and the outcome — never a path/host/title.
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if strings.Contains(line, spineInput().SpineRef) || strings.Contains(line, "internal/") {
			t.Fatalf("ledger row leaks a private boundary: %q", line)
		}
		if !strings.Contains(line, `"schema":"`+LedgerSchema+`"`) {
			t.Fatalf("ledger row missing schema: %q", line)
		}
	}
}

// The committed witness ledger folds to real per-week counts — the durable file
// the issue asks to capture (internal/issuefanout/testdata/ledger.jsonl).
func TestLedgerWitnessFileFolds(t *testing.T) {
	f, err := os.Open("testdata/ledger.jsonl")
	if err != nil {
		t.Fatalf("open witness ledger: %v", err)
	}
	defer f.Close()
	weeks, err := FoldWeekly(f)
	if err != nil {
		t.Fatalf("FoldWeekly(witness): %v", err)
	}
	var total, success, refused, failed int
	for _, w := range weeks {
		total += w.Total
		success += w.Success
		refused += w.Refused
		failed += w.Error
	}
	if total != 5 || success != 3 || refused != 1 || failed != 1 {
		t.Fatalf("witness fold wrong: total=%d success=%d refused=%d error=%d (weeks=%+v)", total, success, refused, failed, weeks)
	}
	if len(weeks) != 2 {
		t.Fatalf("witness spans 2 ISO weeks, got %d: %+v", len(weeks), weeks)
	}
	t.Logf("witness ledger fold:\n%s", RenderWeekly(weeks))
}

// A nil writer is a no-op append (drop-in on any call site), and a malformed row
// is a hard, named error rather than a silent under-count.
func TestLedgerEdgePaths(t *testing.T) {
	// nil writer: BuildLogged still returns the plan and never panics.
	if _, err := BuildLogged(spineInput(), time.Date(2026, 7, 7, 0, 0, 0, 0, time.UTC), nil); err != nil {
		t.Fatalf("BuildLogged(nil writer): %v", err)
	}
	// malformed JSONL surfaces as an error naming the bad line.
	if _, err := FoldWeekly(strings.NewReader("not-json\n")); err == nil {
		t.Fatal("FoldWeekly(malformed) should error")
	}
	// a bad timestamp is rejected too.
	bad := `{"schema":"` + LedgerSchema + `","at":"not-a-time","leaf":"issuefanout","outcome":"success"}`
	if _, err := FoldWeekly(strings.NewReader(bad + "\n")); err == nil {
		t.Fatal("FoldWeekly(bad timestamp) should error")
	}
	// a blank line is skipped, not counted.
	weeks, err := FoldWeekly(strings.NewReader("\n\n"))
	if err != nil {
		t.Fatalf("FoldWeekly(blank): %v", err)
	}
	if len(weeks) != 0 {
		t.Fatalf("blank ledger folds to 0 weeks, got %+v", weeks)
	}
}
