package cachemeta

import (
	"reflect"
	"testing"
)

// Shared fixtures — one unambiguous token per class.
const (
	fxUUID    = "550e8400-e29b-41d4-a716-446655440000"
	fxJWT     = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
	fxSHA256  = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" // 64 hex
	fxMD5     = "d41d8cd98f00b204e9800998ecf8427e"                                 // 32 hex
	fxISOTime = "2026-06-26T12:30"                                                 // sub-day
	fxISODate = "2026-06-26"                                                       // date-only, stable
)

// TestVolatileClassify is the issue #3341 done-condition witness: a system prompt carrying
// a UUID, a JWT, and a SHA-256 hash classifies as uuid=1, jwt=1, hex_hash=1, and the
// operator warning names exactly those classes.
func TestVolatileClassify(t *testing.T) {
	head := []byte(`{"system":"You are a helpful agent. session=` + fxUUID +
		` token=` + fxJWT + ` build=` + fxSHA256 + `"}`)

	rep := ClassifyVolatile(head)

	want := map[VolatileClass]int{VolatileUUID: 1, VolatileJWT: 1, VolatileHexHash: 1}
	if !reflect.DeepEqual(rep.Counts, want) {
		t.Fatalf("Counts = %v, want %v", rep.Counts, want)
	}
	if !rep.Unstable() {
		t.Fatalf("Unstable() = false, want true")
	}
	if rep.Total() != 3 {
		t.Fatalf("Total() = %d, want 3", rep.Total())
	}
	const wantWarn = "cache prefix unstable: uuid=1, jwt=1, hex_hash=1 — move dynamic values out of the system prompt"
	if got := rep.Warning(); got != wantWarn {
		t.Fatalf("Warning() =\n  %q\nwant\n  %q", got, wantWarn)
	}
}

// TestVolatileClassifyStableHead — a stable prompt (a date-ONLY token, ordinary prose, and
// a short id) carries no volatile class: no counts, no warning. Guards the fail-safe
// boundary so a cacheable "Today's date is 2026-06-26" head is not flagged.
func TestVolatileClassifyStableHead(t *testing.T) {
	head := []byte(`{"system":"You are a helpful agent. Today is ` + fxISODate + `. Ticket ABC-42."}`)
	rep := ClassifyVolatile(head)
	if rep.Unstable() {
		t.Fatalf("stable head flagged unstable: counts=%v", rep.Counts)
	}
	if len(rep.Counts) != 0 {
		t.Fatalf("Counts = %v, want empty", rep.Counts)
	}
	if w := rep.Warning(); w != "" {
		t.Fatalf("Warning() = %q, want empty", w)
	}
}

func TestVolatileClassifyEmptyHead(t *testing.T) {
	rep := ClassifyVolatile(nil)
	if rep.Unstable() || len(rep.Counts) != 0 || rep.Warning() != "" {
		t.Fatalf("empty head not stable: counts=%v warn=%q", rep.Counts, rep.Warning())
	}
}

// TestVolatileClassifyISOTimeVsDate — a sub-day timestamp is volatile; a date-only token is
// not. This is the same distinction agent.volDateTime draws, kept exactly.
func TestVolatileClassifyISOTimeVsDate(t *testing.T) {
	if rep := ClassifyVolatile([]byte("now=" + fxISOTime)); rep.Counts[VolatileISO8601] != 1 {
		t.Fatalf("sub-day ISO not classified: counts=%v", rep.Counts)
	}
	if rep := ClassifyVolatile([]byte("day=" + fxISODate)); rep.Counts[VolatileISO8601] != 0 {
		t.Fatalf("date-only wrongly classified iso8601: counts=%v", rep.Counts)
	}
}

// TestVolatileClassifyUUIDNotHexHash — a UUID's hex groups (max 12 contiguous) must NOT
// also register as a hex_hash (min 32), i.e. no cross-class double count.
func TestVolatileClassifyUUIDNotHexHash(t *testing.T) {
	rep := ClassifyVolatile([]byte("id=" + fxUUID))
	if rep.Counts[VolatileUUID] != 1 {
		t.Fatalf("uuid not classified: %v", rep.Counts)
	}
	if rep.Counts[VolatileHexHash] != 0 {
		t.Fatalf("uuid double-counted as hex_hash: %v", rep.Counts)
	}
	if rep.Total() != 1 {
		t.Fatalf("Total() = %d, want 1", rep.Total())
	}
}

// TestVolatileClassifyMultipleHashes — per-class counting is a real occurrence count, and
// distinct digest widths (MD5 + SHA-256) both count as hex_hash (the issue's jwt=1,
// hex_hash=2 warning shape).
func TestVolatileClassifyMultipleHashes(t *testing.T) {
	head := []byte("a=" + fxSHA256 + " b=" + fxMD5 + " t=" + fxJWT)
	rep := ClassifyVolatile(head)
	if rep.Counts[VolatileHexHash] != 2 {
		t.Fatalf("hex_hash count = %d, want 2 (counts=%v)", rep.Counts[VolatileHexHash], rep.Counts)
	}
	if rep.Counts[VolatileJWT] != 1 {
		t.Fatalf("jwt count = %d, want 1", rep.Counts[VolatileJWT])
	}
	const wantWarn = "cache prefix unstable: jwt=1, hex_hash=2 — move dynamic values out of the system prompt"
	if got := rep.Warning(); got != wantWarn {
		t.Fatalf("Warning() =\n  %q\nwant\n  %q", got, wantWarn)
	}
}

// TestVolatileClassifyClasses — Classes() returns the present classes in the stable render
// order regardless of the order they appear in the head.
func TestVolatileClassifyClasses(t *testing.T) {
	head := []byte("hash=" + fxSHA256 + " id=" + fxUUID) // hash appears first in the bytes
	got := ClassifyVolatile(head).Classes()
	want := []VolatileClass{VolatileUUID, VolatileHexHash} // still uuid-before-hex order
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Classes() = %v, want %v", got, want)
	}
}
