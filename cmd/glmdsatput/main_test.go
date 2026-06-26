package main

import (
	"testing"
	"time"
)

// TestLCGIDs pins the deterministic id generator: every id lands in [0,vocab),
// the slice is exactly n long, and the stream is reproducible (same seed path
// every call) — the property the benchmark relies on to compare reps fairly.
func TestLCGIDs(t *testing.T) {
	const n, vocab = 256, 8192
	got := lcgIDs(n, vocab)
	if len(got) != n {
		t.Fatalf("lcgIDs(%d,%d) length = %d, want %d", n, vocab, len(got), n)
	}
	for i, id := range got {
		if id < 0 || id >= vocab {
			t.Fatalf("id[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}
	// Deterministic: a second call must reproduce the first byte-for-byte.
	again := lcgIDs(n, vocab)
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("lcgIDs not deterministic at %d: %d != %d", i, got[i], again[i])
		}
	}
	// Non-degenerate: with vocab >> 1 the stream must not be a single constant.
	allSame := true
	for _, id := range got[1:] {
		if id != got[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatalf("lcgIDs produced a constant stream (%d) — LCG is dead", got[0])
	}
}

// TestMedianMS checks the median-in-milliseconds reducer and, crucially, that it
// does NOT reorder its caller's slice (it sorts a copy) — the timings slices are
// reused across reps, so an in-place sort would corrupt later measurements.
func TestMedianMS(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []time.Duration
		want float64
	}{
		{"odd", []time.Duration{3 * time.Millisecond, time.Millisecond, 2 * time.Millisecond}, 2.0},
		{"even-upper-middle", []time.Duration{time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond, 4 * time.Millisecond}, 3.0},
		{"single", []time.Duration{5 * time.Millisecond}, 5.0},
	} {
		in := append([]time.Duration(nil), tc.in...)
		first := in[0]
		if got := medianMS(in); got != tc.want {
			t.Errorf("%s: medianMS = %v, want %v", tc.name, got, tc.want)
		}
		if in[0] != first {
			t.Errorf("%s: medianMS mutated caller slice (in[0] %v -> %v)", tc.name, first, in[0])
		}
	}
}

// TestRoundHelpers pins the half-up rounding to 2 and 3 decimals on values chosen
// to avoid binary-float representation edges, so the test stays deterministic.
func TestRoundHelpers(t *testing.T) {
	if got := round2(3.14159); got != 3.14 {
		t.Errorf("round2(3.14159) = %v, want 3.14", got)
	}
	if got := round2(0); got != 0 {
		t.Errorf("round2(0) = %v, want 0", got)
	}
	if got := round3(3.14159); got != 3.142 {
		t.Errorf("round3(3.14159) = %v, want 3.142", got)
	}
	if got := round3(2.0005); got != 2.001 {
		t.Errorf("round3(2.0005) = %v, want 2.001", got)
	}
}
