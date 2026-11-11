package perfledger

import (
	"math"
	"testing"
)

// Clock-offset tracker tests, authored adversarially from the pinned contract
// alone (scratch/clockoffset-contract.md): RTT = recv-send; Offset = server -
// (send+recv)/2 computed in float; bounded FIFO window; MAD gate on offsets;
// estimate = min-RTT survivor (earliest on RTT tie). All stamps are
// hand-picked integer milliseconds; no time.Now, no rand, stdlib only.
// Expected values are computed by hand in the comments at each assertion.

// coProbe builds a sample whose RTTMS()==rttMS and OffsetMS()==offsetMS.
// Requires rttMS even and offsetMS integral so ServerUnixMS stays an exact
// integer: server = offset + send + rtt/2 gives
//
//	server - (send+recv)/2 = offset + send + rtt/2 - (2*send + rtt)/2 = offset
//
// exactly in float64 (no rounding: all terms are dyadic).
func coProbe(sendMS, rttMS int64, offsetMS float64) ClockOffsetSample {
	recvMS := sendMS + rttMS
	serverMS := int64(offsetMS) + sendMS + rttMS/2
	return ClockOffsetSample{
		ClientSendUnixMS: sendMS,
		ClientRecvUnixMS: recvMS,
		ServerUnixMS:     serverMS,
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetSampleRTTAndOffset(t *testing.T) {
	tests := []struct {
		name       string
		s          ClockOffsetSample
		wantRTT    float64
		wantOffset float64
	}{
		{
			// RTT = 1001-1000 = 1. Midpoint (1000+1001)/2 = 1000.5 (float,
			// fractional because the sum is odd). Offset = 2000-1000.5 = 999.5.
			// An integer-midpoint implementation would yield 999 or 1000.
			name:       "odd sum gives fractional midpoint",
			s:          ClockOffsetSample{ClientSendUnixMS: 1000, ClientRecvUnixMS: 1001, ServerUnixMS: 2000},
			wantRTT:    1,
			wantOffset: 999.5,
		},
		{
			// RTT = 2004-2000 = 4. Midpoint (2000+2004)/2 = 2002.
			// Offset = 1000-2002 = -1002 (server behind client).
			name:       "negative offset",
			s:          ClockOffsetSample{ClientSendUnixMS: 2000, ClientRecvUnixMS: 2004, ServerUnixMS: 1000},
			wantRTT:    4,
			wantOffset: -1002,
		},
		{
			// RTT = 0 (same-clock probe). Midpoint 500. Offset = 500-500 = 0.
			name:       "zero rtt same clock",
			s:          ClockOffsetSample{ClientSendUnixMS: 500, ClientRecvUnixMS: 500, ServerUnixMS: 500},
			wantRTT:    0,
			wantOffset: 0,
		},
		{
			// RTT = 1. Midpoint (1001+1002)/2 = 1001.5.
			// Offset = 3000-1001.5 = 1998.5.
			name:       "larger odd sum",
			s:          ClockOffsetSample{ClientSendUnixMS: 1001, ClientRecvUnixMS: 1002, ServerUnixMS: 3000},
			wantRTT:    1,
			wantOffset: 1998.5,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.s.RTTMS(); got != tc.wantRTT {
				t.Fatalf("RTTMS() = %v, want %v", got, tc.wantRTT)
			}
			if got := tc.s.OffsetMS(); got != tc.wantOffset {
				t.Fatalf("OffsetMS() = %v, want %v (exact float midpoint)", got, tc.wantOffset)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetAddRefusesNegativeRTT(t *testing.T) {
	tr := NewClockOffsetTracker(8, 3.5)

	// RTT = 1990-2000 = -10 < 0 -> refused; window must stay empty.
	if tr.Add(ClockOffsetSample{ClientSendUnixMS: 2000, ClientRecvUnixMS: 1990, ServerUnixMS: 3000}) {
		t.Fatalf("Add with RTT=-10 accepted, want refused")
	}
	if got := tr.Len(); got != 0 {
		t.Fatalf("Len after refused add = %d, want 0 (refusal must not enter window)", got)
	}
	if _, ok := tr.Offset(); ok {
		t.Fatalf("Offset ok=true after only refusals, want false")
	}

	// RTT = 4999-5000 = -1 < 0 -> refused; Len unchanged.
	if tr.Add(ClockOffsetSample{ClientSendUnixMS: 5000, ClientRecvUnixMS: 4999, ServerUnixMS: 5000}) {
		t.Fatalf("Add with RTT=-1 accepted, want refused")
	}
	// Negative stamps are fine when RTT >= 0: RTT = -990-(-1000) = 10,
	// midpoint (-1000+-990)/2 = -995, offset = -990-(-995) = 5.
	if !tr.Add(coProbe(-1000, 10, 5)) {
		t.Fatalf("Add with negative stamps but RTT=10 refused, want admitted")
	}
	if got := tr.Len(); got != 1 {
		t.Fatalf("Len = %d, want 1", got)
	}
	if got, ok := tr.Offset(); !ok || got != 5 {
		t.Fatalf("Offset = (%v,%v), want (5,true)", got, ok)
	}

	// Zero RTT is admissible (same-clock probe): RTT = 1000-1000 = 0,
	// offset = 1042-1000 = 42. RTT 0 is the minimum, so it must be able
	// to win the estimate later.
	if !tr.Add(coProbe(1000, 0, 42)) {
		t.Fatalf("Add with RTT=0 refused, want admitted (same-clock probe)")
	}
	if got := tr.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2", got)
	}
	// Window offsets {5 (RTT 10), 42 (RTT 0)}: median 23.5, devs {18.5,18.5},
	// MAD 18.5, gate 3.5*18.5 = 64.75 -> both survive; min RTT 0 -> 42.
	if got, ok := tr.Offset(); !ok || got != 42 {
		t.Fatalf("Offset = (%v,%v), want (42,true): zero-RTT sample must win min-RTT estimate", got, ok)
	}

	// The all-zero literal is RTT 0 / offset 0 and must be admissible too.
	if !tr.Add(ClockOffsetSample{}) {
		t.Fatalf("Add(ClockOffsetSample{}) refused, want admitted (RTT 0)")
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetConstructorWindowCapFallback(t *testing.T) {
	// windowCap <= 0 must fall back to DefaultClockWindowCap = 32, observed
	// through eviction: Len() == min(admitted, 32) after every add, and the
	// 33rd add is still admitted (a cap=0-means-refuse-all bug would return
	// false; an unbounded-window bug would push Len to 33).
	for _, cap0 := range []int{0, -7} {
		tr := NewClockOffsetTracker(cap0, 3.5)
		// Sample 0 has offset 500; every later sample offset 0. With any
		// zero present the median is 0 and MAD is 0, so the 500 sample is
		// gate-excluded and the estimate stays 0 — Len is the witness here.
		for i := 0; i < 33; i++ {
			var s ClockOffsetSample
			if i == 0 {
				// RTT 20, offset 500: server = 500+1000+10 = 1510,
				// midpoint (1000+1020)/2 = 1010, offset = 500.
				s = coProbe(1000, 20, 500)
			} else {
				// RTT 20, offset 0: server = send+10 = midpoint.
				s = coProbe(int64(2000+10*i), 20, 0)
			}
			if !tr.Add(s) {
				t.Fatalf("cap0=%d: add %d refused, want admitted", cap0, i)
			}
			want := i + 1
			if want > 32 {
				want = 32
			}
			if got := tr.Len(); got != want {
				t.Fatalf("cap0=%d: Len after add %d = %d, want %d (default cap 32)", cap0, i, got, want)
			}
		}
		// Window is now 32 zero-offset samples: median 0, MAD 0, all
		// survive, min-RTT tie -> earliest -> offset 0.
		if got, ok := tr.Offset(); !ok || got != 0 {
			t.Fatalf("cap0=%d: Offset = (%v,%v), want (0,true)", cap0, got, ok)
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetMADKFallbackAndGate(t *testing.T) {
	// Window (add order): offsets 0,1,2,3,100 with RTTs 20,16,30,26,2 —
	// the 100 outlier carries the MINIMUM RTT, so any gate wide enough to
	// admit it would make it the estimate.
	// median{0,1,2,3,100} = 2; devs {2,1,0,1,98}; MAD = 1; gate = 3.5.
	// Survivors: devs 2,1,0,1 (offsets 0,1,2,3); 98 > 3.5 excludes 100.
	// Min-RTT survivor: RTT 16 -> offset 1. Every bad madK must behave
	// exactly like 3.5 and yield (1,true): +Inf would admit the outlier
	// (estimate 100); NaN fails every comparison (no survivor, ok=false);
	// 0 keeps only the exact median (offset 2, RTT 30 -> estimate 2);
	// -1/-Inf admit nothing (ok=false).
	offsets := []float64{0, 1, 2, 3, 100}
	rtts := []int64{20, 16, 30, 26, 2}
	madKs := []float64{3.5, 0, -1, math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, k := range madKs {
		tr := NewClockOffsetTracker(16, k)
		for i, off := range offsets {
			// server = off + send + rtt/2 keeps OffsetMS == off exactly.
			if !tr.Add(coProbe(int64(1000+1000*i), rtts[i], off)) {
				t.Fatalf("madK=%v: add %d refused, want admitted", k, i)
			}
		}
		if got := tr.Len(); got != 5 {
			t.Fatalf("madK=%v: Len = %d, want 5", k, got)
		}
		got, ok := tr.Offset()
		if !ok || got != 1 {
			t.Fatalf("madK=%v: Offset = (%v,%v), want (1,true): bad madK must fall back to DefaultClockMADK=3.5", k, got, ok)
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetFIFOEvictionOldestFirst(t *testing.T) {
	tr := NewClockOffsetTracker(3, 3.5)
	// A wins the estimate while present (RTT 4 is the global minimum and
	// offset 100 survives the gate), so the post-overflow estimate flip to
	// 0 proves A was evicted — not the newest sample, not nothing.
	adds := []ClockOffsetSample{
		// offset 100, RTT 4: server = 100+1000+2 = 1102, midpoint 1002.
		coProbe(1000, 4, 100),
		// offset 0, RTT 20: server = 2010 = midpoint.
		coProbe(2000, 20, 0),
		// offset 200, RTT 20: server = 200+3000+10 = 3210, midpoint 3010.
		coProbe(3000, 20, 200),
	}
	for i, s := range adds {
		if !tr.Add(s) {
			t.Fatalf("add %d refused, want admitted", i)
		}
		if got := tr.Len(); got != i+1 {
			t.Fatalf("Len after add %d = %d, want %d", i, got, i+1)
		}
	}
	// Offsets {100,0,200}: median 100, devs {100,0,100}, MAD 100,
	// gate 350 -> all survive; min RTT 4 -> A -> 100.
	if got, ok := tr.Offset(); !ok || got != 100 {
		t.Fatalf("pre-overflow Offset = (%v,%v), want (100,true)", got, ok)
	}
	// D overflows the cap: FIFO must evict A (oldest). Window {B,C,D}:
	// offsets {0,200,0}, median 0, devs {0,200,0}, MAD 0 -> only exact
	// median (0) survives -> C excluded; B and D tie on RTT 20 -> earliest
	// B -> 0. (Evict-newest would keep A and estimate 100; an unbounded
	// window would also estimate 100 via A.)
	if !tr.Add(coProbe(4000, 20, 0)) {
		t.Fatalf("overflow add refused, want admitted")
	}
	if got := tr.Len(); got != 3 {
		t.Fatalf("Len after overflow = %d, want 3 (== cap)", got)
	}
	if got, ok := tr.Offset(); !ok || got != 0 {
		t.Fatalf("post-overflow Offset = (%v,%v), want (0,true): oldest sample must be evicted FIFO", got, ok)
	}
	// E evicts B (now oldest). Window {C,D,E}: offsets {200,0,0}, median 0,
	// MAD 0 -> survivors D,E -> earliest D -> 0; Len stays at cap.
	if !tr.Add(coProbe(5000, 20, 0)) {
		t.Fatalf("second overflow add refused, want admitted")
	}
	if got := tr.Len(); got != 3 {
		t.Fatalf("Len after second overflow = %d, want 3", got)
	}
	if got, ok := tr.Offset(); !ok || got != 0 {
		t.Fatalf("Offset = (%v,%v), want (0,true)", got, ok)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetMADGateKeepsMedian(t *testing.T) {
	// Same offsets as the fallback window but the MEDIAN sample (offset 2)
	// carries the minimum RTT: median{0,1,2,3,100} = 2, devs {2,1,0,1,98},
	// MAD 1, gate 3.5 -> survivors {0,1,2,3}; min RTT 2 -> offset 2.
	// A gate that dropped the median sample (e.g. strict <) would leave
	// {0,1,3} and estimate 1 instead.
	tr := NewClockOffsetTracker(16, 3.5)
	offsets := []float64{0, 1, 2, 3, 100}
	rtts := []int64{20, 16, 2, 26, 40}
	for i, off := range offsets {
		if !tr.Add(coProbe(int64(1000+1000*i), rtts[i], off)) {
			t.Fatalf("add %d refused, want admitted", i)
		}
	}
	if got, ok := tr.Offset(); !ok || got != 2 {
		t.Fatalf("Offset = (%v,%v), want (2,true): median sample must always survive the gate", got, ok)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetMADZeroKeepsExactMedianOnly(t *testing.T) {
	// MAD == 0 must keep ONLY offsets exactly equal to the median — a near
	// miss (dev 2) is excluded even though it is small in absolute terms.
	// Odd count {7,7,9}: median 7, devs {0,0,2}, MAD 0 -> 9 excluded; the
	// 9 sample has the min RTT (2), so admitting it would flip 7 -> 9.
	tr := NewClockOffsetTracker(16, 3.5)
	// (1000,10,7): server 1012, midpoint 1005... offset = 1012-1005 = 7.
	tr.Add(coProbe(1000, 10, 7))
	// (2000,12,7): server = 7+2000+6 = 2013, midpoint 2006, offset 7.
	tr.Add(coProbe(2000, 12, 7))
	// (3000,2,9): server = 9+3000+1 = 3010, midpoint 3001, offset 9.
	tr.Add(coProbe(3000, 2, 9))
	if got, ok := tr.Offset(); !ok || got != 7 {
		t.Fatalf("odd-count Offset = (%v,%v), want (7,true): MAD=0 must keep only exact-median offsets", got, ok)
	}
	// Even count {0,0,0,6}: median (0+0)/2 = 0, devs {0,0,0,6}, MAD 0 ->
	// only the 0s survive; the 6 sample has min RTT 2.
	tr2 := NewClockOffsetTracker(16, 3.5)
	tr2.Add(coProbe(4000, 10, 0))
	tr2.Add(coProbe(5000, 12, 0))
	tr2.Add(coProbe(6000, 14, 0))
	// (7000,2,6): server = 6+7000+1 = 7007, midpoint 7001, offset 6.
	tr2.Add(coProbe(7000, 2, 6))
	if got, ok := tr2.Offset(); !ok || got != 0 {
		t.Fatalf("even-count Offset = (%v,%v), want (0,true): MAD=0 must keep only exact-median offsets", got, ok)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetEstimateMinRTTTieEarliest(t *testing.T) {
	// Offsets {0,10,20,40} (add order): sorted median (10+20)/2 = 15,
	// devs {15,5,5,25}, MAD (5+15)/2 = 10, gate 35 -> all survive.
	// Min RTT 6 ties between offset 20 (3rd) and 40 (4th); the
	// earliest-added survivor must win -> 20. Preferring the later or the
	// larger offset would yield 40; preferring earliest overall would
	// yield 0.
	tr := NewClockOffsetTracker(16, 3.5)
	adds := []ClockOffsetSample{
		// (1000,50,0): server 1025, midpoint 1025, offset 0.
		coProbe(1000, 50, 0),
		// (2000,50,10): server = 10+2000+25 = 2035, midpoint 2025.
		coProbe(2000, 50, 10),
		// (3000,6,20): server = 20+3000+3 = 3023, midpoint 3003.
		coProbe(3000, 6, 20),
		// (4000,6,40): server = 40+4000+3 = 4043, midpoint 4003.
		coProbe(4000, 6, 40),
	}
	for i, s := range adds {
		if !tr.Add(s) {
			t.Fatalf("add %d refused, want admitted", i)
		}
	}
	for call := 0; call < 2; call++ {
		got, ok := tr.Offset()
		if !ok || got != 20 {
			t.Fatalf("Offset call %d = (%v,%v), want (20,true): RTT tie must resolve to earliest-added survivor, unchanged window must be deterministic", call, got, ok)
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestClockOffsetEmptyAndZeroValue(t *testing.T) {
	// Empty constructed tracker: (0,false), Len 0.
	tr := NewClockOffsetTracker(4, 3.5)
	if got, ok := tr.Offset(); ok || got != 0 {
		t.Fatalf("empty Offset = (%v,%v), want (0,false)", got, ok)
	}
	if got := tr.Len(); got != 0 {
		t.Fatalf("empty Len = %d, want 0", got)
	}

	// Zero value must be safe for Offset/Len (treated as empty) and Add
	// (defaults: cap 32, madK 3.5).
	var zv ClockOffsetTracker
	if got, ok := zv.Offset(); ok || got != 0 {
		t.Fatalf("zero-value Offset = (%v,%v), want (0,false)", got, ok)
	}
	if got := zv.Len(); got != 0 {
		t.Fatalf("zero-value Len = %d, want 0", got)
	}
	// (1000,10,5): server = 5+1000+5 = 1010, midpoint 1005, offset 5.
	if !zv.Add(coProbe(1000, 10, 5)) {
		t.Fatalf("zero-value Add refused, want admitted (defaults apply)")
	}
	if got := zv.Len(); got != 1 {
		t.Fatalf("zero-value Len = %d, want 1", got)
	}
	// 32 more zero-offset samples (RTT 10): 33 adds total must leave Len
	// at the default cap 32; the offset-5 sample is oldest and is evicted
	// first, so the final estimate is 0.
	for i := 0; i < 32; i++ {
		if !zv.Add(coProbe(int64(2000+10*i), 10, 0)) {
			t.Fatalf("zero-value add %d refused, want admitted", i)
		}
	}
	if got := zv.Len(); got != 32 {
		t.Fatalf("zero-value Len after 33 adds = %d, want 32 (default cap)", got)
	}
	if got, ok := zv.Offset(); !ok || got != 0 {
		t.Fatalf("zero-value Offset = (%v,%v), want (0,true)", got, ok)
	}
}
