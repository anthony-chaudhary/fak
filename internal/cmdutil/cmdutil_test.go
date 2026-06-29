package cmdutil

import (
	"testing"
	"time"
)

func TestArgmax(t *testing.T) {
	cases := []struct {
		name string
		in   []float32
		want int
	}{
		{"empty", nil, 0},
		{"single", []float32{3.2}, 0},
		{"last-max", []float32{1, 2, 9}, 2},
		{"first-of-ties", []float32{5, 5, 1}, 0},
		{"negatives", []float32{-9, -2, -7}, 1},
	}
	for _, c := range cases {
		if got := Argmax(c.in); got != c.want {
			t.Errorf("%s: Argmax(%v)=%d, want %d", c.name, c.in, got, c.want)
		}
	}
}

func TestMs(t *testing.T) {
	if got := Ms(2 * time.Millisecond); got != 2.0 {
		t.Errorf("Ms(2ms)=%v, want 2.0", got)
	}
	if got := Ms(1500 * time.Microsecond); got != 1.5 {
		t.Errorf("Ms(1500us)=%v, want 1.5", got)
	}
}

func TestMedianMS(t *testing.T) {
	ds := []time.Duration{3 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond}
	if got := MedianMS(ds); got != 2.0 {
		t.Errorf("MedianMS=%v, want 2.0", got)
	}
	// the input slice must be left unsorted
	if ds[0] != 3*time.Millisecond {
		t.Errorf("MedianMS mutated caller slice: %v", ds)
	}
}

func TestLCGIDs(t *testing.T) {
	if got := LCGIDs(0, 100, 7); got != nil {
		t.Errorf("LCGIDs(0,...)=%v, want nil", got)
	}
	ids := LCGIDs(64, 32, 1)
	if len(ids) != 64 {
		t.Fatalf("len=%d, want 64", len(ids))
	}
	for i, id := range ids {
		if id < 0 || id >= 32 {
			t.Fatalf("ids[%d]=%d out of [0,32)", i, id)
		}
	}
	// deterministic for a fixed seed
	if got := LCGIDs(64, 32, 1); !equalInts(got, ids) {
		t.Errorf("LCGIDs not deterministic for fixed seed")
	}
	// distinct seeds diverge
	if other := LCGIDs(64, 32, 2); equalInts(other, ids) {
		t.Errorf("LCGIDs ignored the seed")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
