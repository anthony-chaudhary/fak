package contextq

import (
	"strings"
	"testing"
)

func TestDiffWorkingSetsClassifiesHandles(t *testing.T) {
	base := Result{
		Query: "q1",
		Slices: []SliceRef{
			{Step: 1, Role: "user", Descriptor: "a.txt", Bytes: 10},
			{Step: 2, Role: "tool", Descriptor: "b.txt", Bytes: 20}, // dropped by next
		},
		Refused: []Refusal{{Step: 9, Role: "tool", Descriptor: "sealed.txt", Reason: "sealed"}},
	}
	next := Result{
		Query: "q2",
		Slices: []SliceRef{
			{Step: 1, Role: "user", Descriptor: "a.txt", Bytes: 10}, // unchanged
			{Step: 3, Role: "tool", Descriptor: "c.txt", Bytes: 30}, // added
		},
		Refused: []Refusal{{Step: 7, Role: "user", Descriptor: "poison.txt", Reason: "poisoned"}},
	}

	d := DiffWorkingSets(base, next)
	if len(d.Added) != 1 || d.Added[0].Descriptor != "c.txt" {
		t.Fatalf("Added = %+v, want [c.txt]", d.Added)
	}
	if len(d.Removed) != 1 || d.Removed[0].Descriptor != "b.txt" {
		t.Fatalf("Removed = %+v, want [b.txt]", d.Removed)
	}
	if len(d.Unchanged) != 1 || d.Unchanged[0].Descriptor != "a.txt" {
		t.Fatalf("Unchanged = %+v, want [a.txt]", d.Unchanged)
	}
	// Sealed/poisoned refusals must be carried through from BOTH sides, never dropped.
	if len(d.RefusedBase) != 1 || d.RefusedBase[0].Reason != "sealed" {
		t.Fatalf("RefusedBase = %+v, want the sealed refusal", d.RefusedBase)
	}
	if len(d.RefusedNext) != 1 || d.RefusedNext[0].Reason != "poisoned" {
		t.Fatalf("RefusedNext = %+v, want the poisoned refusal", d.RefusedNext)
	}

	md := d.Markdown()
	for _, want := range []string{"+ added", "- removed", "= unchanged", "c.txt", "b.txt", "a.txt",
		"added=1 removed=1 unchanged=1", "raw-evidence expansion blocked", "sealed", "poisoned"} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown transcript missing %q\n%s", want, md)
		}
	}
}

// TestDiffWorkingSetsIdenticalIsAllUnchanged is the degenerate witness: diffing a
// query against itself adds and removes nothing.
func TestDiffWorkingSetsIdenticalIsAllUnchanged(t *testing.T) {
	r := Result{
		Query:  "q",
		Slices: []SliceRef{{Step: 1, Role: "user", Descriptor: "a.txt", Bytes: 5}},
	}
	d := DiffWorkingSets(r, r)
	if len(d.Added) != 0 || len(d.Removed) != 0 || len(d.Unchanged) != 1 {
		t.Fatalf("self-diff = +%d/-%d/=%d, want 0/0/1", len(d.Added), len(d.Removed), len(d.Unchanged))
	}
}
