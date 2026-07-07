package recall

import "testing"

// The linear case: B supersedes A — A is withheld, B survives.
func TestResolveSupersession_linear(t *testing.T) {
	got := ResolveSupersession(map[string][]string{"b": {"a"}}, []string{"a", "b"})
	if len(got) != 1 || got["a"] != "b" {
		t.Fatalf("withheldBy = %v, want a→b only", got)
	}
}

// A chain collapses to its head: C→B→A leaves only C live — a retired note's
// own supersedes edge still retires its predecessor.
func TestResolveSupersession_chainCollapsesToHead(t *testing.T) {
	got := ResolveSupersession(
		map[string][]string{"c": {"b"}, "b": {"a"}},
		[]string{"a", "b", "c"})
	if len(got) != 2 || got["a"] != "b" || got["b"] != "c" {
		t.Fatalf("withheldBy = %v, want a→b and b→c", got)
	}
	if _, withheld := got["c"]; withheld {
		t.Fatalf("chain head c must stay live, got %v", got)
	}
}

// Mutual supersession resolves deterministically: the member latest in index
// order survives, and repeated resolution over the same inputs is identical.
func TestResolveSupersession_mutualCycleTieBreak(t *testing.T) {
	edges := map[string][]string{"a": {"b"}, "b": {"a"}}
	order := []string{"a", "b"}
	got := ResolveSupersession(edges, order)
	if len(got) != 1 || got["a"] != "b" {
		t.Fatalf("withheldBy = %v, want only a→b (b latest in index order survives)", got)
	}
	for i := 0; i < 10; i++ {
		again := ResolveSupersession(edges, order)
		if len(again) != 1 || again["a"] != "b" {
			t.Fatalf("resolution not deterministic on run %d: %v", i, again)
		}
	}
}

// A longer cycle terminates (no infinite loop) and keeps exactly the member
// latest in index order.
func TestResolveSupersession_threeCycle(t *testing.T) {
	got := ResolveSupersession(
		map[string][]string{"a": {"b"}, "b": {"c"}, "c": {"a"}},
		[]string{"a", "b", "c"})
	if len(got) != 2 {
		t.Fatalf("withheldBy = %v, want exactly a and b withheld", got)
	}
	if _, withheld := got["c"]; withheld {
		t.Fatalf("cycle member latest in index order must survive, got %v", got)
	}
}

// An edge from OUTSIDE a cycle still retires the cycle's surviving member —
// the tie-break exempts intra-cycle edges only.
func TestResolveSupersession_outsideEdgeRetiresCycleSurvivor(t *testing.T) {
	got := ResolveSupersession(
		map[string][]string{"a": {"b"}, "b": {"a"}, "d": {"b"}},
		[]string{"a", "b", "d"})
	if got["a"] != "b" || got["b"] != "d" {
		t.Fatalf("withheldBy = %v, want a→b and b→d", got)
	}
	if _, withheld := got["d"]; withheld {
		t.Fatalf("out-of-cycle superseder d must stay live, got %v", got)
	}
}

// A self-edge is not an edge; a source absent from the index still resolves
// deterministically (ranked earliest, walked lexically after indexed sources).
func TestResolveSupersession_selfEdgeAndOffIndexSource(t *testing.T) {
	got := ResolveSupersession(
		map[string][]string{"a": {"a"}, "zz-off-index": {"a"}},
		[]string{"a"})
	if len(got) != 1 || got["a"] != "zz-off-index" {
		t.Fatalf("withheldBy = %v, want only a→zz-off-index", got)
	}
}
