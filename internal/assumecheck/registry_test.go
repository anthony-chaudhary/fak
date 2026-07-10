package assumecheck

import (
	"regexp"
	"testing"
)

// TestRegistryInvariants proves every registered assumption is well-formed: closed
// Level/WitnessKind/WitnessStatus memberships, no blank identity field, a
// SCREAMING_SNAKE data-only refusal token, and unique ids. Deliberately NOT a
// len(registry)==N freeze — that is the CHANGE_DETECTOR_TEST anti-pattern
// (boundarylint catalog): assert the relation every row must hold, not the count.
func TestRegistryInvariants(t *testing.T) {
	rows := Registry()
	if len(rows) == 0 {
		t.Fatal("empty assumption registry")
	}
	refusal := regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)
	seen := make(map[string]bool, len(rows))
	for _, a := range rows {
		if a.ID == "" || a.Owner == "" || a.Statement == "" {
			t.Fatalf("registry row has a blank identity field: %+v", a)
		}
		if !ValidLevel(a.Level) {
			t.Fatalf("assumption %q declares level %q outside the closed vocabulary", a.ID, a.Level)
		}
		if !ValidWitnessKind(a.WitnessKind) {
			t.Fatalf("assumption %q declares witness kind %q outside the closed vocabulary", a.ID, a.WitnessKind)
		}
		if !ValidWitnessStatus(a.WitnessStatus) {
			t.Fatalf("assumption %q declares witness status %q outside the closed vocabulary", a.ID, a.WitnessStatus)
		}
		if a.RefusalReason != "" && !refusal.MatchString(a.RefusalReason) {
			t.Fatalf("assumption %q refusal reason %q is not a SCREAMING_SNAKE token", a.ID, a.RefusalReason)
		}
		if seen[a.ID] {
			t.Fatalf("duplicate assumption id %q", a.ID)
		}
		seen[a.ID] = true
	}
}

// TestRegistryRowZeroIsSeatLaunchable proves the exported C1 spine var IS registry
// row 0 (field-for-field, not a lookalike), so the shell reference and the registry
// share one source of truth.
func TestRegistryRowZeroIsSeatLaunchable(t *testing.T) {
	rows := Registry()
	if rows[0] != SeatLaunchable {
		t.Fatalf("registry row 0 is not the exported SeatLaunchable var:\n row0 =%+v\n spine=%+v", rows[0], SeatLaunchable)
	}
	if SeatLaunchable.WitnessStatus != WitnessWired {
		t.Fatalf("the spine's one wired assumption must be marked %s, got %s", WitnessWired, SeatLaunchable.WitnessStatus)
	}
}

// TestRegistryStableOrderAndCopy proves Registry() returns the same order on every
// call and hands out a COPY — a caller mutating its slice cannot corrupt the table.
func TestRegistryStableOrderAndCopy(t *testing.T) {
	first := Registry()
	second := Registry()
	if len(first) != len(second) {
		t.Fatalf("Registry() length changed between calls: %d then %d", len(first), len(second))
	}
	for i := range first {
		if first[i].ID != second[i].ID {
			t.Fatalf("Registry() order unstable at row %d: %q then %q", i, first[i].ID, second[i].ID)
		}
	}
	second[0].ID = "mutated-by-caller"
	if again := Registry(); again[0].ID != first[0].ID {
		t.Fatalf("Registry() exposes its backing array: row 0 became %q", again[0].ID)
	}
}

// TestLookupRoundTripsAndFailsClosed proves every registered id resolves to its own
// row and an unregistered id reports ok=false instead of a zero-value guess.
func TestLookupRoundTripsAndFailsClosed(t *testing.T) {
	for _, want := range Registry() {
		got, ok := Lookup(want.ID)
		if !ok {
			t.Fatalf("registered assumption %q not found by Lookup", want.ID)
		}
		if got != want {
			t.Fatalf("Lookup(%q) diverges from its registry row:\n got =%+v\n want=%+v", want.ID, got, want)
		}
	}
	if a, ok := Lookup("no-such-assumption"); ok {
		t.Fatalf("Lookup of an unregistered id returned ok=true: %+v", a)
	}
}
