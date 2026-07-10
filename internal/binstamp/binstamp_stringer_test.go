package binstamp

import "testing"

// The String() methods are the human/diagnostic face of the freshness verdict — the package
// doc calls out that a diagnostic "warns loudly" on Unstamped while staying quiet on the
// benign causes, and it does that by rendering these strings. They had no direct coverage, so
// a silent rename ("stale" -> "outdated") or a lost default branch could drift the operator
// surface unnoticed. Pin the full contract, including the out-of-range default that guards
// against an unhandled future enum value rendering as an empty string.
func TestFreshnessStringContract(t *testing.T) {
	cases := []struct {
		f    Freshness
		want string
	}{
		{Fresh, "fresh"},
		{Stale, "stale"},
		{Unknown, "unknown"},
		{Freshness(99), "unknown"}, // any unmapped value collapses to the benign default
	}
	for _, c := range cases {
		if got := c.f.String(); got != c.want {
			t.Errorf("Freshness(%d).String() = %q, want %q", int(c.f), got, c.want)
		}
	}
}

func TestCauseStringContract(t *testing.T) {
	cases := []struct {
		c    Cause
		want string
	}{
		{CauseMatched, "matched"},
		{CauseDiverged, "diverged"},
		{CauseUnstamped, "unstamped"},
		{CauseDirty, "dirty"},
		{CauseNoHead, "no-head"},
		{Cause(99), "unknown"}, // out-of-range guard
	}
	for _, c := range cases {
		if got := c.c.String(); got != c.want {
			t.Errorf("Cause(%d).String() = %q, want %q", int(c.c), got, c.want)
		}
	}
}

// revisionsMatch accepts a match only when the shorter rev is at least 7 chars, so a weak
// partial SHA can never spoof freshness. The existing table exercises 4- and 13-char revs but
// not the exact len==7 threshold — the classic off-by-one where a `< 7` could silently become
// `<= 7`. Pin both sides of the boundary.
func TestRevisionsMatchSevenCharThreshold(t *testing.T) {
	const full = "abcdef1234567890abcdef1234567890abcdef12"
	if revisionsMatch("abcdef", full) { // 6 chars — one below the floor
		t.Fatal("a 6-char prefix must not match (below the 7-char floor)")
	}
	if !revisionsMatch("abcdef1", full) { // exactly 7 chars — the minimum accepted
		t.Fatal("a 7-char prefix is the minimum and must match")
	}
}

// revisionsMatch normalizes which operand is short vs long internally, so the verdict must be
// identical regardless of argument order. Compare/Explain feed it (running, head): a caller
// that passes a SHORT head (e.g. `git rev-parse --short HEAD`) against a full running rev must
// still read Fresh. The existing tests only cover the mirror image (short running, full head).
func TestCompareShortHeadAgainstFullRunning(t *testing.T) {
	const shortHead = "abcdef1" // 7-char short SHA a caller might pass as HEAD
	running := Stamp{Revision: "abcdef1234567890abcdef1234567890abcdef12", HasVCS: true}

	if got := Compare(running, shortHead); got != Fresh {
		t.Fatalf("Compare(full running, short head) = %v, want Fresh", got)
	}
	if gotF, gotC := Explain(running, shortHead); gotF != Fresh || gotC != CauseMatched {
		t.Fatalf("Explain(full running, short head) = (%v,%v), want (Fresh, CauseMatched)", gotF, gotC)
	}
}
