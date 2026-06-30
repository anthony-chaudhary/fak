package milestonedoc

import (
	"strings"
	"testing"
)

// TestBlockDeterministic guards the generator: two builds at one commit must be
// byte-identical, so the committed snapshot regenerates with no diff.
func TestBlockDeterministic(t *testing.T) {
	if Block() != Block() {
		t.Fatal("Block() is not deterministic across two calls")
	}
}

// TestBlockHasMarkersAndLadder proves the generated block is bounded by the markers and
// carries a row per ladder rung (M0..M7) plus the climb header.
func TestBlockHasMarkersAndLadder(t *testing.T) {
	b := Block()
	if !strings.HasPrefix(b, Begin) {
		t.Fatalf("block must start with the begin marker; got:\n%s", b)
	}
	if !strings.HasSuffix(b, End) {
		t.Fatalf("block must end with the end marker; got:\n%s", b)
	}
	if !strings.Contains(b, "| Rung | Meaning | Cells |") {
		t.Fatalf("block missing the ladder table header; got:\n%s", b)
	}
	for _, id := range []string{"M0", "M4", "M7"} {
		if !strings.Contains(b, "| "+id+" |") {
			t.Fatalf("block missing ladder row %s; got:\n%s", id, b)
		}
	}
	if !strings.Contains(b, "Climb") {
		t.Fatalf("block missing the climb header; got:\n%s", b)
	}
}

// TestBlockASCIIOnly proves the committed bytes are byte-stable cross-platform: no
// non-ASCII rune (em-dash / middot) that would mojibake on a Windows checkout.
func TestBlockASCIIOnly(t *testing.T) {
	for i, r := range Block() {
		if r > 127 {
			t.Fatalf("non-ASCII rune %q at byte offset %d; the block must be ASCII-only", r, i)
		}
	}
}

// TestWriteThenCheckRoundTripsFresh is the #1441 witness: splicing the live block into a
// scaffold yields a doc that Fresh() accepts and Extract() round-trips. This is the
// "write-then-check is fresh" half of the contract.
func TestWriteThenCheckRoundTripsFresh(t *testing.T) {
	written, err := Splice(Scaffold())
	if err != nil {
		t.Fatalf("splice into scaffold: %v", err)
	}
	if !Fresh(written) {
		t.Fatalf("a freshly written doc must pass Fresh(); got:\n%s", written)
	}
	got, ok := Extract(written)
	if !ok {
		t.Fatal("Extract found no block in a freshly written doc")
	}
	if got != Block() {
		t.Fatalf("extracted block != live Block()\nextracted:\n%s\nlive:\n%s", got, Block())
	}
}

// TestStaleDocFailsCheck proves the freshness gate fires: a doc whose committed block
// drifted from the live fold is NOT Fresh. This is the "a stale doc fails the check"
// half of the witness.
func TestStaleDocFailsCheck(t *testing.T) {
	written, err := Splice(Scaffold())
	if err != nil {
		t.Fatalf("splice into scaffold: %v", err)
	}
	if !Fresh(written) {
		t.Fatal("precondition: freshly written doc should be fresh")
	}
	// Mutate one cell count inside the generated block. " | 0 |" / " | 1 |" only matches
	// a table cell value, never the markers or prose; flipping it drifts the block.
	stale := written
	for _, swap := range []struct{ from, to string }{
		{" | 0 |", " | 99 |"},
		{" | 1 |", " | 99 |"},
		{" | 2 |", " | 99 |"},
	} {
		if strings.Contains(stale, swap.from) {
			stale = strings.Replace(stale, swap.from, swap.to, 1)
			break
		}
	}
	if stale == written {
		t.Fatal("could not find a ladder cell value to mutate")
	}
	if Fresh(stale) {
		t.Fatalf("a doc with a mutated ladder cell must NOT be fresh; got:\n%s", stale)
	}
}

// TestMissingMarkersIsStale proves Extract / Fresh treat a doc with no markers as stale
// (the freshness check reds rather than silently passing an un-generated doc).
func TestMissingMarkersIsStale(t *testing.T) {
	const bare = "# Milestone status\n\nno markers here\n"
	if _, ok := Extract(bare); ok {
		t.Fatal("Extract should not find a block in a marker-less doc")
	}
	if Fresh(bare) {
		t.Fatal("a marker-less doc must not be Fresh")
	}
}

// TestSpliceErrorsWithoutMarkers proves Splice refuses to guess a write location when
// the markers are absent, so the generator never corrupts an arbitrary doc.
func TestSpliceErrorsWithoutMarkers(t *testing.T) {
	if _, err := Splice("# no markers\n"); err == nil {
		t.Fatal("Splice should error when the begin marker is absent")
	}
}
