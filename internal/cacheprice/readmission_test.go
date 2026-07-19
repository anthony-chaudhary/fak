package cacheprice

import (
	"fmt"
	"testing"
)

// TestFrequencySketchMonotoneBoundedOvercount locks the two estimator guarantees the
// re-admission gate leans on (#5290): within an epoch the estimate is (a) MONOTONE — more
// touches never lower a key's estimate — and (b) a BOUNDED OVERCOUNT — never below the
// key's true touch count (collisions only inflate, min-over-rows never undercounts) and
// never above the total touches the sketch has absorbed (no counter can exceed it).
func TestFrequencySketchMonotoneBoundedOvercount(t *testing.T) {
	s := NewFrequencySketch(64, 4) // decay period 256; this test stays well under it

	// Monotone: one key's estimate never decreases as its touches accumulate.
	prev := s.Estimate("mono")
	for i := 0; i < 40; i++ {
		s.Touch("mono")
		if got := s.Estimate("mono"); got < prev {
			t.Fatalf("estimate fell %d -> %d after touch %d: sketch must be monotone within an epoch", prev, got, i+1)
		} else {
			prev = got
		}
	}

	// Bounded overcount: true count ≤ estimate ≤ total touches, for every key.
	truth := map[string]int{"mono": 40}
	total := 40
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("k%d", i)
		for j := 0; j <= i; j++ {
			s.Touch(key)
		}
		truth[key] = i + 1
		total += i + 1
	}
	for key, want := range truth {
		got := s.Estimate(key)
		if got < want {
			t.Fatalf("Estimate(%q) = %d undercounts true touches %d: min-over-rows must be an upper bound", key, got, want)
		}
		if got > total {
			t.Fatalf("Estimate(%q) = %d exceeds total touches %d: overcount must be bounded by the epoch's mass", key, got, total)
		}
	}
}

// TestFrequencySketchDecayHalvesAtSaturation pins the self-decay: once total touches reach
// width×depth every counter halves and the epoch resets — Mooncake's timer-free aging. A
// single key drives the sketch so the arithmetic is exact regardless of hashing: 16 touches
// on an 8×2 sketch (period 16) land estimate 16 then immediately halve to 8; a second full
// epoch of 16 more touches climbs to 24 and halves to 12. Floor-halving also erases a
// one-hit-wonder's trace (1 → 0) after one epoch.
func TestFrequencySketchDecayHalvesAtSaturation(t *testing.T) {
	s := NewFrequencySketch(8, 2) // decay period = 16

	for i := 0; i < 16; i++ {
		s.Touch("hot")
	}
	if got := s.Estimate("hot"); got != 8 {
		t.Fatalf("after 16 touches on a period-16 sketch, want halved estimate 8, got %d", got)
	}
	for i := 0; i < 16; i++ {
		s.Touch("hot")
	}
	if got := s.Estimate("hot"); got != 12 {
		t.Fatalf("after a second epoch (8+16 halved), want estimate 12, got %d", got)
	}

	// A one-hit-wonder's count floors to zero across a decay: touched once, then an epoch
	// of unrelated traffic ages it fully out.
	w := NewFrequencySketch(8, 2)
	w.Touch("wonder")
	for i := 0; i < 15; i++ {
		w.Touch("noise")
	}
	if got := w.Estimate("wonder"); got != 0 {
		t.Fatalf("one-hit-wonder should age to 0 after a decay epoch, got %d", got)
	}
}

// TestShouldReadmitGates locks the three-gate verdict and its order-free semantics: the
// frequency bar refuses cold candidates, the headroom gate refuses displacement into a full
// tier, and the in-flight budget refuses a promotion storm — with cap<0 opting out of budget
// gating and threshold≤0 opting out of the frequency bar.
func TestShouldReadmitGates(t *testing.T) {
	cases := []struct {
		name               string
		freq, threshold    int
		headroom           bool
		inflight, capacity int
		want               bool
	}{
		{"cold candidate refused below threshold", 1, 2, true, 0, 4, false},
		{"hot candidate admitted at threshold", 2, 2, true, 0, 4, true},
		{"hot candidate admitted above threshold", 9, 2, true, 0, 4, true},
		{"no headroom refuses even a hot candidate", 9, 2, false, 0, 4, false},
		{"saturated budget refuses even a hot candidate", 9, 2, true, 4, 4, false},
		{"over-saturated budget refuses", 9, 2, true, 7, 4, false},
		{"budget with room admits", 9, 2, true, 3, 4, true},
		{"zero cap is a zero budget: refuse all", 9, 2, true, 0, 0, false},
		{"negative cap is unbounded: admit", 9, 2, true, 1000, -1, true},
		{"zero threshold disables the frequency bar", 0, 0, true, 0, 4, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ShouldReadmit(tc.freq, tc.threshold, tc.headroom, tc.inflight, tc.capacity)
			if got != tc.want {
				t.Fatalf("ShouldReadmit(%d, %d, %v, %d, %d) = %v, want %v",
					tc.freq, tc.threshold, tc.headroom, tc.inflight, tc.capacity, got, tc.want)
			}
		})
	}
}

// TestReadmissionAntiThrash proves the property the gate exists for (#5290): a hot resident
// survives a burst of one-hit-wonder candidates. The fast tier is modeled as a single slot
// holding a genuinely hot prefix; 100 distinct cold keys each get touched once (their miss)
// and challenge for re-admission. Every challenger estimates 1, below the bar of 2, so none
// displaces the resident — while the hot prefix itself, once evicted and re-touched, clears
// the same bar and is copied back. Without the frequency gate every challenger would win
// the slot on first touch and the hot entry would thrash out 100 times.
func TestReadmissionAntiThrash(t *testing.T) {
	s := NewFrequencySketch(256, 4) // decay period 1024; the burst stays inside one epoch
	const threshold = 2

	for i := 0; i < 16; i++ {
		s.Touch("hot-prefix")
	}
	resident := "hot-prefix"

	evictions := 0
	for i := 0; i < 100; i++ {
		cold := fmt.Sprintf("one-hit-%d", i)
		s.Touch(cold) // the single miss access that tempts a naive pool to promote
		if ShouldReadmit(s.Estimate(cold), threshold, true, 0, 4) {
			resident = cold // a naive admission would displace the hot resident
			evictions++
		}
	}
	if resident != "hot-prefix" {
		t.Fatalf("hot resident thrashed out by one-hit-wonders: resident = %q after %d displacements", resident, evictions)
	}

	// The dual: the hot prefix itself, evicted and then re-touched, DOES earn re-admission —
	// the gate blocks thrash, not recovery.
	s.Touch("hot-prefix")
	if !ShouldReadmit(s.Estimate("hot-prefix"), threshold, true, 0, 4) {
		t.Fatalf("re-touched hot prefix (estimate %d) must clear the re-admission bar %d", s.Estimate("hot-prefix"), threshold)
	}
}
