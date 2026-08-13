package ctxresidency_test

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// emitCapFaults commits n CAP_FAULT rows through the real journal writer, so the
// fixture is a genuine hash-chained log (and Cut has real rows to archive) rather
// than hand-written JSONL.
func emitCapFaults(j *journal.Journal, n int, name string) {
	for i := 0; i < n; i++ {
		j.Emit(abi.Event{
			Kind:   abi.EvCapFault,
			Fields: map[string]any{"cap_kind": "skill", "cap_name": name, "cap_digest": "d"},
		})
	}
}

// TestLoaderJournalTotalsSpanTheCut is the migrated-consumer half of the #6488
// witness: LoaderJournal reconciles COUNTS against the kernel's ledger, so it must
// fold the whole journal. Before the migration it read the live segment only, so a
// rotated journal under-counted every event before the cut and reported a FALSE
// discrepancy against a kernel whose counters were right. The rotated journal must
// now reconcile against exactly the same counters as the unrotated one.
func TestLoaderJournalTotalsSpanTheCut(t *testing.T) {
	const (
		before = 3 // faults committed before the cut
		after  = 2 // faults committed after it
	)

	rotated := filepath.Join(t.TempDir(), "journal.jsonl")
	rj, err := journal.Open(rotated)
	if err != nil {
		t.Fatalf("open rotated: %v", err)
	}
	emitCapFaults(rj, before, "skill-old")
	if _, err := rj.Cut(); err != nil {
		t.Fatalf("cut: %v", err)
	}
	emitCapFaults(rj, after, "skill-new")
	if err := rj.Close(); err != nil {
		t.Fatalf("close rotated: %v", err)
	}
	// The fixture really did rotate: the live file alone is a strict tail.
	if live, lerr := journal.ReadRows(rotated); lerr != nil || len(live) >= before+after {
		t.Fatalf("fixture did not rotate: live rows = %d (err %v), want < %d", len(live), lerr, before+after)
	}

	// The same history, never cut, is the control.
	flat := filepath.Join(t.TempDir(), "journal.jsonl")
	fj, err := journal.Open(flat)
	if err != nil {
		t.Fatalf("open flat: %v", err)
	}
	emitCapFaults(fj, before, "skill-old")
	emitCapFaults(fj, after, "skill-new")
	if err := fj.Close(); err != nil {
		t.Fatalf("close flat: %v", err)
	}

	got, err := ctxresidency.LoaderJournal(rotated, before+after, 0, 0)
	if err != nil {
		t.Fatalf("LoaderJournal(rotated): %v", err)
	}
	want, err := ctxresidency.LoaderJournal(flat, before+after, 0, 0)
	if err != nil {
		t.Fatalf("LoaderJournal(flat): %v", err)
	}

	if got.Faults != before+after {
		t.Fatalf("rotated Faults = %d, want %d (the total must span the cut)", got.Faults, before+after)
	}
	if got.Faults != want.Faults {
		t.Fatalf("rotated Faults = %d, unrotated = %d; a roll-up must not depend on rotation", got.Faults, want.Faults)
	}
	if len(got.Operations) != len(want.Operations) {
		t.Fatalf("rotated Operations = %d, unrotated = %d", len(got.Operations), len(want.Operations))
	}
	if !got.Reconciled {
		t.Fatalf("rotated snapshot not reconciled against honest kernel counters: %+v", got)
	}
	// The CUT anchor is rotation bookkeeping, not a capability event: it must not
	// leak into the operation list.
	for _, op := range got.Operations {
		if op.Kind == journal.KindCut {
			t.Fatalf("CUT anchor leaked into Operations: %+v", op)
		}
	}
}
