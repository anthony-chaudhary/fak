package deepseekv4kv

import (
	"strings"
	"testing"
)

// The load-bearing case from the study note: a 512-token request that includes an SWA
// group is NOT servable for all 512 leading tokens — the sliding window only ever
// reached its last SWAWindow tokens, so the model-wide hittable prefix is bounded by
// the window, not the request length.
func TestServablePrefixBoundedByWindow(t *testing.T) {
	const seq = 512
	got := ServablePrefixUnits(seq, []Kind{KindCSA, KindHCA, KindSWA, KindTail})
	if got != SWAWindow {
		t.Fatalf("512-token request with an SWA group must bound to the window %d, got %d", SWAWindow, got)
	}
	if got == seq {
		t.Fatalf("bound must not over-count to the full request length %d", seq)
	}
}

// All full-reach groups (no sliding window) serve the whole prefix: the bound is seq.
func TestServablePrefixFullReachAll(t *testing.T) {
	const seq = 512
	got := ServablePrefixUnits(seq, []Kind{KindCSA, KindHCA, KindTail})
	if got != seq {
		t.Fatalf("all full-reach groups must serve the whole prefix %d, got %d", seq, got)
	}
}

// The tightest group binds: adding an SWA group to any mix clamps the bound to the
// window; below the window the request length itself binds.
func TestServablePrefixTightestBinds(t *testing.T) {
	cases := []struct {
		seq   int
		kinds []Kind
		want  int
	}{
		{512, []Kind{KindSWA}, SWAWindow},              // lone window binds
		{512, []Kind{KindTail, KindSWA}, SWAWindow},    // window binds against a full group
		{64, []Kind{KindSWA, KindTail}, 64},            // below the window, seq binds
		{SWAWindow, []Kind{KindSWA}, SWAWindow},        // exactly at the window
		{SWAWindow + 1, []Kind{KindSWA}, SWAWindow},    // one past the window saturates
		{Ctx1M, []Kind{KindCSA, KindHCA, KindTail}, Ctx1M}, // no window, no clamp
	}
	for _, c := range cases {
		if got := ServablePrefixUnits(c.seq, c.kinds); got != c.want {
			t.Errorf("ServablePrefixUnits(%d, %v) = %d, want %d", c.seq, c.kinds, got, c.want)
		}
	}
}

// Fail-closed degenerates: no groups, non-positive request, and an unknown kind all
// yield 0 (nothing servable) rather than an over-counted bound.
func TestServablePrefixFailsClosed(t *testing.T) {
	if got := ServablePrefixUnits(512, nil); got != 0 {
		t.Errorf("no groups must yield 0, got %d", got)
	}
	if got := ServablePrefixUnits(512, []Kind{}); got != 0 {
		t.Errorf("empty groups must yield 0, got %d", got)
	}
	if got := ServablePrefixUnits(0, []Kind{KindTail}); got != 0 {
		t.Errorf("zero request must yield 0, got %d", got)
	}
	if got := ServablePrefixUnits(-7, []Kind{KindTail}); got != 0 {
		t.Errorf("negative request must yield 0, got %d", got)
	}
	if got := ServablePrefixUnits(512, []Kind{KindTail, Kind(99)}); got != 0 {
		t.Errorf("an unknown group must fail closed to 0, got %d", got)
	}
}

// The rendered line names each group's hittable length and the binding bound.
func TestFormatServablePrefix(t *testing.T) {
	out := FormatServablePrefix(512, []Kind{KindTail, KindSWA})
	for _, must := range []string{"512", "tail=512", "SWA=128", "bound=128"} {
		if !strings.Contains(out, must) {
			t.Errorf("rendered line missing %q\n%s", must, out)
		}
	}
}
