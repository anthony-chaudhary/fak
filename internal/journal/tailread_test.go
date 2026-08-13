package journal

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestTailReadDeclaresOmissionAcrossCut is the #6488 witness: after a cut, the
// segment-aware read still totals the WHOLE journal, and the tail-only read returns
// the short slice *and says so* instead of handing back a bare prefix that looks
// exactly like a complete small journal.
func TestTailReadDeclaresOmissionAcrossCut(t *testing.T) {
	const (
		before = 7 // N rows committed before the cut
		after  = 4 // M rows committed after it
	)
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")

	j, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	appendDecides(j, before)
	if _, err := j.Cut(); err != nil {
		t.Fatalf("cut: %v", err)
	}
	appendDecides(j, after)
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// (a) The segment-aware read loses nothing across the cut. Raw, it is the literal
	// chain (N + the CUT anchor + M); folded through WithoutCutAnchors it is exactly
	// the N+M the same journal would have had unrotated — the Done condition.
	all, err := ReadAllSegments(path)
	if err != nil {
		t.Fatalf("ReadAllSegments: %v", err)
	}
	if len(all) != before+1+after {
		t.Fatalf("ReadAllSegments len = %d, want %d (N + CUT anchor + M)", len(all), before+1+after)
	}
	folded := WithoutCutAnchors(all)
	if len(folded) != before+after {
		t.Fatalf("WithoutCutAnchors(ReadAllSegments) len = %d, want %d", len(folded), before+after)
	}
	for _, r := range folded {
		if r.Kind == KindCut {
			t.Fatalf("WithoutCutAnchors left a %s anchor at seq %d", KindCut, r.Seq)
		}
	}

	// (b) The tail-only read returns the live segment only — and REPORTS the omission
	// rather than returning a bare short slice.
	tail, om, err := ReadTail(path)
	if err != nil {
		t.Fatalf("ReadTail: %v", err)
	}
	if len(WithoutCutAnchors(tail)) != after {
		t.Fatalf("ReadTail rows = %d, want %d (live segment only)", len(WithoutCutAnchors(tail)), after)
	}
	if !om.Omitted() {
		t.Fatalf("ReadTail omission = %+v, want a declared omission after a cut", om)
	}
	if om.SealedSegments != 1 {
		t.Fatalf("omission.SealedSegments = %d, want 1", om.SealedSegments)
	}
	if om.RowsBeforeCut != before {
		t.Fatalf("omission.RowsBeforeCut = %d, want %d", om.RowsBeforeCut, before)
	}
	if s := om.String(); !strings.Contains(s, "omitted") {
		t.Fatalf("omission.String() = %q, want an operator-readable disclaimer", s)
	}

	// The tail read must not be silently confusable with the whole journal.
	if len(tail) == len(all) {
		t.Fatalf("tail read returned the whole chain (%d rows); the fixture did not rotate", len(tail))
	}
}

// TestTailReadOnUncutJournalDeclaresNothing keeps the disclaimer honest in the other
// direction: an un-rotated journal's tail IS the whole journal, so the omission is
// empty and renders as no text at all.
func TestTailReadOnUncutJournalDeclaresNothing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")

	j, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	appendDecides(j, 3)
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	rows, om, err := ReadTail(path)
	if err != nil {
		t.Fatalf("ReadTail: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("ReadTail len = %d, want 3", len(rows))
	}
	if om.Omitted() || om.String() != "" {
		t.Fatalf("uncut journal omission = %+v (%q), want empty", om, om.String())
	}
	all, err := ReadAllSegments(path)
	if err != nil {
		t.Fatalf("ReadAllSegments: %v", err)
	}
	if len(all) != len(rows) {
		t.Fatalf("uncut: ReadAllSegments len = %d, tail len = %d; they must agree", len(all), len(rows))
	}
}

// TestReadTailMissingJournalIsEmpty keeps ReadTail as tolerant as ReadRows: a
// not-yet-written journal is the empty "no rows yet" state, not an error.
func TestReadTailMissingJournalIsEmpty(t *testing.T) {
	rows, om, err := ReadTail(filepath.Join(t.TempDir(), "absent.jsonl"))
	if err != nil {
		t.Fatalf("ReadTail(absent) = %v, want nil error", err)
	}
	if len(rows) != 0 {
		t.Fatalf("ReadTail(absent) len = %d, want 0", len(rows))
	}
	if om.Omitted() {
		t.Fatalf("ReadTail(absent) omission = %+v, want none", om)
	}
}
