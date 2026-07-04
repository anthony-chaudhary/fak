package relay

import "testing"

// TestDoNotRederiveDedupsAndSurvivesRoundTrip is the C4 (#1873) done-condition witness:
// duplicate closed-path pointers dedup, and the deduped result survives a full Baton
// codec round-trip (Marshal then Parse) unchanged.
func TestDoNotRederiveDedupsAndSurvivesRoundTrip(t *testing.T) {
	idx := NewDoNotRederiveIndex([]string{
		"memory:relay-schema-freeform-draft",
		"issue:#1852",
	})
	idx.Add("issue:#1852") // duplicate of an existing entry
	idx.Add("memory:relay-schema-freeform-draft")
	idx.Add("commit:0123456789abcdef0123456789abcdef01234567")
	idx.Add("") // empty pointer must never be recorded

	if got, want := idx.Len(), 3; got != want {
		t.Fatalf("Len() = %d, want %d (duplicates and the empty string must not count)", got, want)
	}

	want := []string{
		"memory:relay-schema-freeform-draft",
		"issue:#1852",
		"commit:0123456789abcdef0123456789abcdef01234567",
	}
	got := idx.Pointers()
	if len(got) != len(want) {
		t.Fatalf("Pointers() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Pointers()[%d] = %q, want %q (order must be first-seen)", i, got[i], want[i])
		}
	}

	b := Baton{Schema: Schema, RelayID: "RLY-donotrederive-roundtrip", DoNotRederive: idx.Pointers()}
	wire, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	parsed, err := Parse(wire)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(parsed.DoNotRederive) != len(want) {
		t.Fatalf("round-tripped DoNotRederive = %v, want %v", parsed.DoNotRederive, want)
	}
	for i := range want {
		if parsed.DoNotRederive[i] != want[i] {
			t.Fatalf("round-tripped DoNotRederive[%d] = %q, want %q", i, parsed.DoNotRederive[i], want[i])
		}
	}
}

// TestDoNotRederiveIndexZeroValueUsable pins that the zero DoNotRederiveIndex behaves
// like an empty index rather than panicking on first use — a caller building an index
// fresh (not from NewDoNotRederiveIndex) still gets correct dedup behavior.
func TestDoNotRederiveIndexZeroValueUsable(t *testing.T) {
	var idx DoNotRederiveIndex
	if idx.Len() != 0 {
		t.Fatalf("zero-value Len() = %d, want 0", idx.Len())
	}
	if got := idx.Pointers(); len(got) != 0 {
		t.Fatalf("zero-value Pointers() = %v, want empty", got)
	}
	idx.Add("file:internal/relay/donotrederive.go")
	if idx.Len() != 1 {
		t.Fatalf("Len() after one Add = %d, want 1", idx.Len())
	}
}
