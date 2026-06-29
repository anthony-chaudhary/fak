package turntaxmeter

import (
	"sync"
	"testing"
)

// TestSamplerRateExact pins the 1-in-n contract: ShouldSample admits the first event
// of every n-window and no other, so over T serial calls it admits exactly ceil(T/n).
// The realized rate is an EXACT equality, not a statistical tolerance — that is what
// makes "the sampling rate is honored" a witnessed fact (issue #1156 / T4).
func TestSamplerRateExact(t *testing.T) {
	for _, n := range []int{1, 2, 4, 8, 100} {
		s := NewSampler(n)
		const total = 1000
		admitted := 0
		// The admitting positions must be exactly 1, 1+n, 1+2n, … (1-indexed call
		// number), so we check each call's verdict against that predicate too — not
		// only the final tally — to catch a phase shift that happens to total right.
		for i := 1; i <= total; i++ {
			got := s.ShouldSample()
			want := i%n == 1%n
			if got != want {
				t.Fatalf("rate %d: call %d admitted=%v, want %v — the n-window phase drifted", n, i, got, want)
			}
			if got {
				admitted++
			}
		}
		// ceil(total/n) admissions; for n that divides total exactly this is total/n.
		want := (total + n - 1) / n
		if admitted != want {
			t.Fatalf("rate %d: admitted %d of %d, want exactly %d (1-in-%d)", n, admitted, total, want, n)
		}
		if int(s.Admitted()) != admitted {
			t.Fatalf("rate %d: Admitted() = %d, want %d — the internal tally disagrees with the returns", n, s.Admitted(), admitted)
		}
	}
}

// TestSamplerRateHonoredUnderLoad is issue #1156's "the sampling rate is honored under
// load" criterion. Many goroutines hammer ONE shared sampler concurrently; because the
// gate's decision is a single atomic increment, the realized admission count must still
// be EXACTLY ceil(total/n) — no admission lost to a race, no double-count from two
// goroutines reading the same window. A lock-free gate that drifted under contention
// would fail here; an exact one cannot.
func TestSamplerRateHonoredUnderLoad(t *testing.T) {
	for _, n := range []int{1, 4, 16} {
		s := NewSampler(n)
		const (
			workers = 16
			perW    = 4096
		)
		total := workers * perW

		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < perW; i++ {
					s.ShouldSample()
				}
			}()
		}
		wg.Wait()

		if got := int(s.Seen()); got != total {
			t.Fatalf("rate %d under load: Seen() = %d, want %d — a call was lost to a race", n, got, total)
		}
		want := (total + n - 1) / n
		if got := int(s.Admitted()); got != want {
			t.Fatalf("rate %d under load: admitted %d of %d concurrent events, want exactly %d (1-in-%d) — the rate drifted under contention", n, got, total, want, n)
		}
	}
}

// TestNewSamplerClampsNonPositive pins the fail-toward-observability clamp: a
// miscomputed non-positive rate degrades to full-instrument (rate 1, sample
// everything) rather than dividing by zero or sampling nothing — a meter that records
// too much is caught by the cost cap, but one that records NOTHING would silently hide
// a regression, so the clamp must err loud, not silent.
func TestNewSamplerClampsNonPositive(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		s := NewSampler(n)
		if s.Rate() != 1 {
			t.Fatalf("NewSampler(%d).Rate() = %d, want 1 (clamp to full-instrument)", n, s.Rate())
		}
		if !s.ShouldSample() || !s.ShouldSample() {
			t.Fatalf("NewSampler(%d) must admit every event after clamp to rate 1", n)
		}
	}
}
