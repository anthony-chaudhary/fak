package compute

import (
	"reflect"
	"testing"
)

// hotSpan is a proven-reused resident span (many hits, compact) — expensive to lose.
// coldScan is a one-shot pollutant (no reuse) at the same per-token byte cost — the span a
// value-aware cache must refuse to admit so it cannot evict the hot span.
var (
	hotSpan  = KVSpanStats{Tokens: 100, Bytes: 100, Hits: 20, LastUsed: 5}
	coldScan = KVSpanStats{Tokens: 100, Bytes: 100, Hits: 0, LastUsed: 9}
)

// TestDecideKVAdmissionScanResistance pins the load-bearing claim (#2672): a one-shot scan
// candidate is BYPASSED when it would displace a proven-hot resident span, while a candidate
// genuinely hotter than the victim is ADMITTED and correctly replaces it. This is the hole
// eviction alone cannot close — by the time the hot span is the victim, the pollutant is
// already admitted.
func TestDecideKVAdmissionScanResistance(t *testing.T) {
	// A cold one-shot scan candidate against a hot resident victim: refuse it.
	got := DecideKVAdmission(coldScan, hotSpan, 0)
	if got.Verdict != AdmitBypass || got.Reason != ReasonColderThanVictim {
		t.Fatalf("cold scan vs hot victim: got %v/%q, want bypass/colder_than_victim", got.Verdict, got.Reason)
	}
	// A candidate hotter than the victim it displaces: admit and let it replace the cold span.
	hotCand := KVSpanStats{Tokens: 100, Bytes: 100, Hits: 50, LastUsed: 9}
	coldVictim := KVSpanStats{Tokens: 100, Bytes: 100, Hits: 1, LastUsed: 3}
	got = DecideKVAdmission(hotCand, coldVictim, 0)
	if got.Verdict != AdmitCache || got.Reason != ReasonOutranksVictim {
		t.Fatalf("hot cand vs cold victim: got %v/%q, want cache/outranks_victim", got.Verdict, got.Reason)
	}
}

// TestDecideKVAdmissionFailsOpen pins the fail-open fences: an unpriced candidate and an
// empty resident set (a zero-value victim) both ALWAYS admit — the gate never refuses to
// cache a span it cannot prove is a pollutant. This is what makes wiring the gate additive
// and safe (the discard_admit fail-open contract).
func TestDecideKVAdmissionFailsOpen(t *testing.T) {
	// Unpriced candidate (Bytes <= 0): value undefined -> admit.
	if got := DecideKVAdmission(KVSpanStats{Tokens: 100, Bytes: 0, Hits: 0}, hotSpan, 3.0); got.Verdict != AdmitCache || got.Reason != ReasonUnpricedCandidate {
		t.Fatalf("unpriced candidate: got %v/%q, want cache/candidate_unpriced", got.Verdict, got.Reason)
	}
	// Empty resident set (zero-value victim, Bytes == 0): nothing hot displaced -> admit.
	if got := DecideKVAdmission(coldScan, KVSpanStats{}, 3.0); got.Verdict != AdmitCache || got.Reason != ReasonNoResidentVictim {
		t.Fatalf("empty resident set: got %v/%q, want cache/no_resident_victim", got.Verdict, got.Reason)
	}
	// The zero-value verdict is AdmitCache: a default-constructed decision reproduces the
	// insert-always status quo byte-identically.
	if (KVAdmitDecision{}).Verdict != AdmitCache {
		t.Fatalf("zero verdict: got %v, want cache (insert-always reduction)", (KVAdmitDecision{}).Verdict)
	}
}

// TestDecideKVAdmissionAgingClock pins the GDSF admission variant (#2668): a candidate whose
// value clears the current aging clock L is admitted even against a momentarily-cheaper
// victim — AND that path is INERT when the pool does not age (clock <= 0), so the no-clock
// decision reduces to the pure W-TinyLFU victim comparison.
func TestDecideKVAdmissionAgingClock(t *testing.T) {
	// A candidate no hotter than the victim, but whose value (KVEvictionCost) clears the clock.
	// cand cost = 100*(2)/100 = 2.0; victim cost = 100*(3)/100 = 3.0; cand does NOT outrank the
	// victim, so with no clock it bypasses...
	cand := KVSpanStats{Tokens: 100, Bytes: 100, Hits: 1}   // cost 2.0
	victim := KVSpanStats{Tokens: 100, Bytes: 100, Hits: 2} // cost 3.0
	if got := DecideKVAdmission(cand, victim, 0); got.Verdict != AdmitBypass || got.Reason != ReasonColderThanVictim {
		t.Fatalf("no clock, cand colder than victim: got %v/%q, want bypass/colder_than_victim", got.Verdict, got.Reason)
	}
	// ...but a clock the candidate's value clears (L = 1.5 <= 2.0) admits it (GDSF admission).
	if got := DecideKVAdmission(cand, victim, 1.5); got.Verdict != AdmitCache || got.Reason != ReasonClearsAgingClock {
		t.Fatalf("clock cleared: got %v/%q, want cache/clears_aging_clock", got.Verdict, got.Reason)
	}
	// A clock the candidate's value does NOT clear (L = 2.5 > 2.0) still bypasses.
	if got := DecideKVAdmission(cand, victim, 2.5); got.Verdict != AdmitBypass || got.Reason != ReasonColderThanVictim {
		t.Fatalf("clock not cleared: got %v/%q, want bypass/colder_than_victim", got.Verdict, got.Reason)
	}
}

// TestPlanKVAdmissionStats checks the aggregate scan-resistance numbers over a mixed batch of
// misses: proven-hot victims protected by a bypass are counted as HotSpansProtected, admits
// count separately, and the plan is index-aligned 1:1 with the input.
func TestPlanKVAdmissionStats(t *testing.T) {
	cands := []KVAdmissionCandidate{
		{Cand: coldScan, Victim: hotSpan},                                       // bypass — protects a hot span (Hits 20)
		{Cand: coldScan, Victim: hotSpan},                                       // bypass — protects a hot span
		{Cand: KVSpanStats{Tokens: 100, Bytes: 100, Hits: 50}, Victim: hotSpan}, // admit — hotter than the victim
		{Cand: coldScan, Victim: KVSpanStats{}},                                 // admit — empty resident set (fail-open)
		{Cand: coldScan, Victim: KVSpanStats{Tokens: 100, Bytes: 100, Hits: 0}}, // bypass — victim cold (Hits 0), not counted hot
	}
	items, stats := PlanKVAdmission(cands)
	if len(items) != len(cands) {
		t.Fatalf("plan length: got %d, want %d", len(items), len(cands))
	}
	for i, it := range items {
		if it.Index != i {
			t.Fatalf("item %d index: got %d, want %d", i, it.Index, i)
		}
	}
	want := KVAdmissionStats{Candidates: 5, Admitted: 2, Bypassed: 3, HotSpansProtected: 2}
	if stats != want {
		t.Fatalf("stats: got %+v, want %+v", stats, want)
	}
}

// TestPlanKVAdmissionEmpty pins the nil/empty contract: no candidates -> nil items, zero stats.
func TestPlanKVAdmissionEmpty(t *testing.T) {
	items, stats := PlanKVAdmission(nil)
	if items != nil {
		t.Fatalf("nil input items: got %v, want nil", items)
	}
	if !reflect.DeepEqual(stats, KVAdmissionStats{}) {
		t.Fatalf("nil input stats: got %+v, want zero", stats)
	}
}

// TestKVAdmitVerdictString pins the log/observability strings.
func TestKVAdmitVerdictString(t *testing.T) {
	if AdmitCache.String() != "cache" {
		t.Fatalf("AdmitCache string: got %q, want cache", AdmitCache.String())
	}
	if AdmitBypass.String() != "bypass" {
		t.Fatalf("AdmitBypass string: got %q, want bypass", AdmitBypass.String())
	}
}

// TestKVAdmissionScanTraceProtectsHot is the host-free witness of the concrete scan-resistance
// VALUE (not just the decision): a burst of one-shot cold scan candidates, each decided against
// the same hot resident victim, is bypassed in full — the hot span is never displaced. This is
// the case insert-always (today) gets wrong (it would evict the hot span for the first
// pollutant); a proper #2244 replay arm is the R3 rung, this pins the primitive drives it.
func TestKVAdmissionScanTraceProtectsHot(t *testing.T) {
	scan := make([]KVAdmissionCandidate, 0, 32)
	for i := 0; i < 32; i++ {
		// Distinct one-shot spans (varying LastUsed) but all cold (Hits 0) at uniform byte cost.
		scan = append(scan, KVAdmissionCandidate{
			Cand:   KVSpanStats{Tokens: 100, Bytes: 100, Hits: 0, LastUsed: uint64(100 + i)},
			Victim: hotSpan,
		})
	}
	_, stats := PlanKVAdmission(scan)
	if stats.Admitted != 0 {
		t.Fatalf("scan trace: %d pollutants admitted, want 0 (hot span must never be displaced)", stats.Admitted)
	}
	if stats.Bypassed != 32 || stats.HotSpansProtected != 32 {
		t.Fatalf("scan trace: got bypassed=%d protected=%d, want 32/32", stats.Bypassed, stats.HotSpansProtected)
	}
}
