package session

import "testing"

// TestContinuationEpochRoundTrip proves the continuation id and the uint64 epoch are
// two encodings of one lineage value: every id ContinuationID mints decodes to an
// epoch that rebuilds the SAME id. This is the round-trip #914 requires, and it binds
// the epoch decoding to the actual mint format (continuationID) — a change to the mint
// breaks this test rather than silently desyncing the shared id space.
func TestContinuationEpochRoundTrip(t *testing.T) {
	cases := []struct {
		trace string
		rev   uint64
	}{
		{"guard", 1},
		{"gw-7", 42},
		{"win-0123456789abcdef", 1000}, // a continuation re-continuing again
		{"", 0},
	}
	for _, c := range cases {
		id := ContinuationID(c.trace, c.rev)
		e, ok := ContinuationEpoch(id)
		if !ok {
			t.Fatalf("ContinuationEpoch(%q) ok=false, want true (id minted by ContinuationID)", id)
		}
		if got := ContinuationIDForEpoch(e); got != id {
			t.Errorf("round trip %q -> %d -> %q changed the id", id, e, got)
		}
	}
}

// TestContinuationEpochRejectsNonContinuation proves a string that is not a minted
// continuation id has no epoch — a caller reads (0, false) as "generation 0", never a
// guessed epoch. Covers the wrong-prefix, wrong-length, and non-hex paths.
func TestContinuationEpochRejectsNonContinuation(t *testing.T) {
	for _, id := range []string{
		"guard",                  // original trace, no prefix
		"gw-123",                 // minted gateway trace, no prefix
		"",                       // empty
		"win-",                   // prefix only, no hex
		"win-abc",                // too short
		"win-zzzzzzzzzzzzzzzz",   // 16 chars but not hex
		"win-0123456789abcdef00", // too long
	} {
		if e, ok := ContinuationEpoch(id); ok || e != 0 {
			t.Errorf("ContinuationEpoch(%q) = (%d, %v), want (0, false)", id, e, ok)
		}
	}
}
