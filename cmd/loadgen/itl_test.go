package main

// Adversarial tests for the aiperf-style ITL first-chunk divisor port
// (cmd/loadgen/itl.go), derived from the upstream spec
// (ai-dynamo/aiperf src/aiperf/metrics/types/inter_token_latency_metric.py @
// ab610add, Apache-2.0; ADAPT disposition) rather than from implementation
// internals:
//
//	ITL = (Request Latency - TTFT) / (OSL - First Content Chunk Tokens)
//
// Divisor rules (upstream _decode_token_count):
//   - first-chunk token count not reported -> legacy osl-1 fallback (default path)
//   - reported but non-positive, OR osl-firstChunk < 1 (lagging usage, or the
//     whole response in one chunk) -> osl-1 fallback
//   - otherwise -> osl-firstChunk, sourced from server per-chunk usage
//   - osl < 2 -> no ITL computable (upstream raises NoMetricValue)
//
// Upstream's explicit-empty-first-chunk rule (divisor = osl) has no analog in
// loadgen: TTFT is set only on non-empty content chunks, so the first content
// chunk always carries content by construction.

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"
	"time"
)

// itlIntPtr returns a pointer to n, mirroring the per-chunk usage shape
// (reqResult.firstChunkTokens is *int: nil = not reported).
func itlIntPtr(n int) *int { return &n }

// fcStr renders a *int for test failure messages.
func fcStr(fc *int) string {
	if fc == nil {
		return "nil"
	}
	return fmt.Sprintf("%d", *fc)
}

// TestITLDecodeTokenDivisor_NilFirstChunkFallback pins the expected default path:
// when the first content chunk reports no per-chunk usage, the divisor is the
// legacy osl-1 fallback and is NOT flagged as per-chunk-sourced.
func TestITLDecodeTokenDivisor_NilFirstChunkFallback(t *testing.T) {
	for _, osl := range []int{2, 3, 10, 11, 1000} {
		d, ok := DecodeTokenDivisor(osl, nil)
		if !ok {
			t.Fatalf("osl=%d: ok=false, want legacy osl-1 fallback ok=true", osl)
		}
		if d.Tokens != osl-1 {
			t.Fatalf("osl=%d: Tokens=%d, want %d (osl-1)", osl, d.Tokens, osl-1)
		}
		if d.PerChunk {
			t.Fatalf("osl=%d: PerChunk=true, want false (fallback is not per-chunk-sourced)", osl)
		}
	}
}

// TestITLDecodeTokenDivisor_NonPositiveFirstChunkFallback: a reported but
// non-positive first-chunk token count (explicit 0, or a negative lag) cannot
// describe a real first chunk, so the divisor falls back to osl-1.
func TestITLDecodeTokenDivisor_NonPositiveFirstChunkFallback(t *testing.T) {
	for _, osl := range []int{2, 7, 50} {
		for _, fc := range []int{0, -1, -100} {
			d, ok := DecodeTokenDivisor(osl, itlIntPtr(fc))
			if !ok {
				t.Fatalf("osl=%d fc=%d: ok=false, want osl-1 fallback ok=true", osl, fc)
			}
			if d.Tokens != osl-1 || d.PerChunk {
				t.Fatalf("osl=%d fc=%d: got (%+v,%v), want ({Tokens:%d PerChunk:false},true)", osl, fc, d, ok, osl-1)
			}
		}
	}
}

// TestITLDecodeTokenDivisor_WholeResponseOneChunkFallback: when the first content
// chunk already carries the whole response (firstChunk == osl), osl-firstChunk is
// 0, so the divisor falls back to osl-1 rather than dividing by zero.
func TestITLDecodeTokenDivisor_WholeResponseOneChunkFallback(t *testing.T) {
	for _, osl := range []int{2, 9, 64} {
		d, ok := DecodeTokenDivisor(osl, itlIntPtr(osl))
		if !ok || d.Tokens != osl-1 || d.PerChunk {
			t.Fatalf("osl=%d fc=osl: got (%+v,%v), want ({Tokens:%d PerChunk:false},true)", osl, d, ok, osl-1)
		}
	}
}

// TestITLDecodeTokenDivisor_FirstChunkAboveOSLFallback: per-chunk usage lagging
// behind the final count (firstChunk > osl) makes osl-firstChunk negative, so the
// divisor falls back to osl-1 rather than dividing by a negative token count.
func TestITLDecodeTokenDivisor_FirstChunkAboveOSLFallback(t *testing.T) {
	for _, tc := range []struct{ osl, fc int }{{2, 3}, {10, 11}, {10, 100}} {
		d, ok := DecodeTokenDivisor(tc.osl, itlIntPtr(tc.fc))
		if !ok || d.Tokens != tc.osl-1 || d.PerChunk {
			t.Fatalf("osl=%d fc=%d: got (%+v,%v), want ({Tokens:%d PerChunk:false},true)", tc.osl, tc.fc, d, ok, tc.osl-1)
		}
	}
}

// TestITLDecodeTokenDivisor_PerChunkSourcedDivisor: when the first content chunk
// reports a positive count strictly below osl, the divisor is osl-firstChunk,
// sourced from server per-chunk usage (PerChunk=true). firstChunk == osl-1 is the
// minimal case: exactly one decode token remains. firstChunk == 1 yields the same
// numeric divisor as the fallback (osl-1) but must still be flagged PerChunk —
// the flag records provenance, not value.
func TestITLDecodeTokenDivisor_PerChunkSourcedDivisor(t *testing.T) {
	// firstChunk == osl-1 -> decode = 1
	if d, ok := DecodeTokenDivisor(10, itlIntPtr(9)); !ok || d.Tokens != 1 || !d.PerChunk {
		t.Fatalf("osl=10 fc=9: got (%+v,%v), want ({Tokens:1 PerChunk:true},true)", d, ok)
	}
	// general case: osl=10, first=3 -> decode = 7
	if d, ok := DecodeTokenDivisor(10, itlIntPtr(3)); !ok || d.Tokens != 7 || !d.PerChunk {
		t.Fatalf("osl=10 fc=3: got (%+v,%v), want ({Tokens:7 PerChunk:true},true)", d, ok)
	}
	// first=1 -> decode = 9 (numerically osl-1) but per-chunk-sourced
	if d, ok := DecodeTokenDivisor(10, itlIntPtr(1)); !ok || d.Tokens != 9 || !d.PerChunk {
		t.Fatalf("osl=10 fc=1: got (%+v,%v), want ({Tokens:9 PerChunk:true},true)", d, ok)
	}
}

// TestITLDecodeTokenDivisor_MinimalComputableOSL2: osl=2 is the smallest computable
// output; every first-chunk shape must resolve to divisor 1 without dividing by
// zero or going negative.
func TestITLDecodeTokenDivisor_MinimalComputableOSL2(t *testing.T) {
	// nil -> legacy fallback divisor 1
	if d, ok := DecodeTokenDivisor(2, nil); !ok || d.Tokens != 1 || d.PerChunk {
		t.Fatalf("osl=2 fc=nil: got (%+v,%v), want ({Tokens:1 PerChunk:false},true)", d, ok)
	}
	// fc=0 -> non-positive rule -> fallback divisor 1
	if d, ok := DecodeTokenDivisor(2, itlIntPtr(0)); !ok || d.Tokens != 1 || d.PerChunk {
		t.Fatalf("osl=2 fc=0: got (%+v,%v), want ({Tokens:1 PerChunk:false},true)", d, ok)
	}
	// fc=1 -> decode = 1, per-chunk-sourced
	if d, ok := DecodeTokenDivisor(2, itlIntPtr(1)); !ok || d.Tokens != 1 || !d.PerChunk {
		t.Fatalf("osl=2 fc=1: got (%+v,%v), want ({Tokens:1 PerChunk:true},true)", d, ok)
	}
	// fc=2 (whole response in one chunk) -> decode 0 -> fallback divisor 1
	if d, ok := DecodeTokenDivisor(2, itlIntPtr(2)); !ok || d.Tokens != 1 || d.PerChunk {
		t.Fatalf("osl=2 fc=2: got (%+v,%v), want ({Tokens:1 PerChunk:false},true)", d, ok)
	}
	// fc=5 (lagging usage) -> negative decode -> fallback divisor 1
	if d, ok := DecodeTokenDivisor(2, itlIntPtr(5)); !ok || d.Tokens != 1 || d.PerChunk {
		t.Fatalf("osl=2 fc=5: got (%+v,%v), want ({Tokens:1 PerChunk:false},true)", d, ok)
	}
}

// TestITLDecodeTokenDivisor_TooFewTokensNotComputable: with fewer than two output
// tokens there is no inter-token interval at all (upstream raises NoMetricValue);
// the port reports ok=false with the zero divisor, regardless of first-chunk shape.
func TestITLDecodeTokenDivisor_TooFewTokensNotComputable(t *testing.T) {
	for _, osl := range []int{1, 0, -5} {
		for _, fc := range []*int{nil, itlIntPtr(0), itlIntPtr(1), itlIntPtr(osl)} {
			d, ok := DecodeTokenDivisor(osl, fc)
			if ok {
				t.Fatalf("osl=%d fc=%s: ok=true, want false (no ITL computable)", osl, fcStr(fc))
			}
			if d != (ITLDivisor{}) {
				t.Fatalf("osl=%d fc=%s: divisor=%+v, want zero value on ok=false", osl, fcStr(fc), d)
			}
		}
	}
}

// TestITLDecodeTokenDivisor_PerChunkFlagExactBranch sweeps the (osl, firstChunk)
// matrix against a spec-derived oracle: PerChunk is true exactly when the divisor
// came from server per-chunk usage (first chunk reported, positive, and
// osl-firstChunk >= 1); ok is exactly osl >= 2; ok=false carries the zero divisor;
// ok=true always carries a positive token count (never divides by zero).
func TestITLDecodeTokenDivisor_PerChunkFlagExactBranch(t *testing.T) {
	for osl := -5; osl <= 30; osl++ {
		fcs := []*int{nil, itlIntPtr(-3), itlIntPtr(0), itlIntPtr(1), itlIntPtr(2)}
		if osl-1 >= 1 {
			fcs = append(fcs, itlIntPtr(osl-1))
		}
		fcs = append(fcs, itlIntPtr(osl), itlIntPtr(osl+5))
		for _, fc := range fcs {
			d, ok := DecodeTokenDivisor(osl, fc)
			wantOK := osl >= 2
			// Spec oracle: per-chunk-sourced exactly when the first chunk reported a
			// positive count and at least one decode token remains.
			wantPerChunk := fc != nil && *fc > 0 && osl-*fc >= 1
			if ok != wantOK {
				t.Fatalf("osl=%d fc=%s: ok=%v, want %v", osl, fcStr(fc), ok, wantOK)
			}
			if d.PerChunk != wantPerChunk {
				t.Fatalf("osl=%d fc=%s: PerChunk=%v, want %v", osl, fcStr(fc), d.PerChunk, wantPerChunk)
			}
			if !ok {
				if d != (ITLDivisor{}) {
					t.Fatalf("osl=%d fc=%s: ok=false must carry the zero divisor, got %+v", osl, fcStr(fc), d)
				}
				continue
			}
			if d.Tokens < 1 {
				t.Fatalf("osl=%d fc=%s: ok=true with Tokens=%d, want >= 1 (no zero division)", osl, fcStr(fc), d.Tokens)
			}
			wantTokens := osl - 1
			if wantPerChunk {
				wantTokens = osl - *fc
			}
			if d.Tokens != wantTokens {
				t.Fatalf("osl=%d fc=%s: Tokens=%d, want %d", osl, fcStr(fc), d.Tokens, wantTokens)
			}
		}
	}
}

// TestITLMS_ExactMSConversion pins the arithmetic: (latency - ttft) in
// milliseconds divided by the divisor's token count. A 1.5s decode window over 3
// tokens is exactly 500ms per token.
func TestITLMS_ExactMSConversion(t *testing.T) {
	if got := ITLMS(2*time.Second, 500*time.Millisecond, ITLDivisor{Tokens: 3}); got != 500 {
		t.Fatalf("ITLMS(2s,500ms,3 tokens)=%v, want exactly 500", got)
	}
	// Sub-millisecond precision must survive the ns->ms conversion: 1ms over 2
	// tokens = 0.5ms.
	if got := ITLMS(1*time.Millisecond, 0, ITLDivisor{Tokens: 2}); got != 0.5 {
		t.Fatalf("ITLMS(1ms,0,2 tokens)=%v, want exactly 0.5", got)
	}
	// A zero decode window is a measured zero, not an error.
	if got := ITLMS(300*time.Millisecond, 300*time.Millisecond, ITLDivisor{Tokens: 4}); got != 0 {
		t.Fatalf("ITLMS(300ms,300ms,4 tokens)=%v, want exactly 0", got)
	}
	// The divisor passed in is authoritative: the same window over the fallback
	// (osl-1 = 9) vs per-chunk (6) divisor yields different per-token latencies,
	// and the PerChunk flag itself does not enter the arithmetic.
	window := 1800 * time.Millisecond
	fallback := ITLMS(window, 0, ITLDivisor{Tokens: 9})
	perChunk := ITLMS(window, 0, ITLDivisor{Tokens: 6, PerChunk: true})
	if fallback != 200 || perChunk != 300 {
		t.Fatalf("divisor not honored: fallback=%v (want 200), perChunk=%v (want 300)", fallback, perChunk)
	}
}

// TestITLAggregate_PopulatesFromStreamedSuccesses: aggregate() is the fold point —
// mean/p50/p95 ITL populate from streamed successes with usable usage, and
// per_chunk_usage_divisor_requests counts only per-chunk-sourced divisors.
func TestITLAggregate_PopulatesFromStreamedSuccesses(t *testing.T) {
	results := []reqResult{
		// fallback divisor: osl=10, no per-chunk usage -> 9 tokens; window 1000ms -> 1000/9 ms
		{latency: 1500 * time.Millisecond, ttft: 500 * time.Millisecond, completionTokens: 10, streamed: true},
		// per-chunk divisor: osl=10, first chunk carried 4 -> 6 tokens; window 1000ms -> 1000/6 ms
		{latency: 1200 * time.Millisecond, ttft: 200 * time.Millisecond, completionTokens: 10, streamed: true, firstChunkTokens: itlIntPtr(4)},
	}
	pr := aggregate(1, 2, 2*time.Second, results)
	wantMean := (1000.0/9.0 + 1000.0/6.0) / 2
	if math.Abs(pr.MeanITLMS-wantMean) > 1e-9 {
		t.Fatalf("MeanITLMS=%v, want %v", pr.MeanITLMS, wantMean)
	}
	// nearest-rank over sorted [1000/9, 1000/6]: p50 -> 1000/9, p95 -> 1000/6
	if math.Abs(pr.ITLP50MS-1000.0/9.0) > 1e-9 {
		t.Fatalf("ITLP50MS=%v, want %v", pr.ITLP50MS, 1000.0/9.0)
	}
	if math.Abs(pr.ITLP95MS-1000.0/6.0) > 1e-9 {
		t.Fatalf("ITLP95MS=%v, want %v", pr.ITLP95MS, 1000.0/6.0)
	}
	if pr.PerChunkUsageDivisorRequests != 1 {
		t.Fatalf("PerChunkUsageDivisorRequests=%d, want 1 (only the second request)", pr.PerChunkUsageDivisorRequests)
	}
}

// TestITLAggregate_SkipsUnusableRequests: requests that failed, did not stream,
// lack usable usage, lack a measured TTFT, or carry osl<2 still count toward
// latency/usage accounting but must not contaminate the ITL fields.
func TestITLAggregate_SkipsUnusableRequests(t *testing.T) {
	results := []reqResult{
		{err: fmt.Errorf("boom")},                               // failed: no ITL
		{latency: 400 * time.Millisecond, completionTokens: 10}, // plain-JSON fallback: no ITL
		// streamed but usage missing: no ITL
		{latency: 400 * time.Millisecond, completionTokens: 10, streamed: true, tokenUsageMissing: true, ttft: 100 * time.Millisecond},
		// streamed, usage present, but no first content chunk (ttft=0): no ITL
		{latency: 400 * time.Millisecond, completionTokens: 10, streamed: true},
		// streamed, ttft measured, but osl=1 < 2: no ITL
		{latency: 400 * time.Millisecond, completionTokens: 1, streamed: true, ttft: 100 * time.Millisecond},
		// streamed, ttft measured, usage reported as 0 (osl<2): no ITL
		{latency: 400 * time.Millisecond, completionTokens: 0, streamed: true, ttft: 100 * time.Millisecond},
		// the only ITL contributor: osl=5, no per-chunk usage -> divisor 4; window 500ms -> 125ms
		{latency: 600 * time.Millisecond, completionTokens: 5, streamed: true, ttft: 100 * time.Millisecond},
	}
	pr := aggregate(1, 7, time.Second, results)
	if pr.MeanITLMS != 125 || pr.ITLP50MS != 125 || pr.ITLP95MS != 125 {
		t.Fatalf("itl mean/p50/p95=%v/%v/%v, want 125/125/125 (only the last request contributes)",
			pr.MeanITLMS, pr.ITLP50MS, pr.ITLP95MS)
	}
	if pr.PerChunkUsageDivisorRequests != 0 {
		t.Fatalf("PerChunkUsageDivisorRequests=%d, want 0", pr.PerChunkUsageDivisorRequests)
	}
}

// TestITLAggregate_AllUnusableLeavesITLZero: with no ITL contributor the fields
// stay at their zero values (omitempty then keeps them off the wire).
func TestITLAggregate_AllUnusableLeavesITLZero(t *testing.T) {
	results := []reqResult{
		{err: fmt.Errorf("boom")},
		{latency: 400 * time.Millisecond, completionTokens: 10, streamed: true, tokenUsageMissing: true, ttft: 100 * time.Millisecond},
		{latency: 400 * time.Millisecond, completionTokens: 10, streamed: true},
	}
	pr := aggregate(1, 3, time.Second, results)
	if pr.MeanITLMS != 0 || pr.ITLP50MS != 0 || pr.ITLP95MS != 0 || pr.PerChunkUsageDivisorRequests != 0 {
		t.Fatalf("ITL fields nonzero with no contributor: %+v", pr)
	}
}

// TestITLAggregate_PerChunkDivisorCount: per_chunk_usage_divisor_requests counts
// only divisors sourced from server per-chunk usage — fallback divisors (not
// reported, non-positive, or osl-firstChunk < 1) contribute ITL values but never
// the count.
func TestITLAggregate_PerChunkDivisorCount(t *testing.T) {
	results := []reqResult{
		// per-chunk: osl=10, first=2 -> 8 tokens; window 800ms -> 100ms
		{latency: 1000 * time.Millisecond, ttft: 200 * time.Millisecond, completionTokens: 10, streamed: true, firstChunkTokens: itlIntPtr(2)},
		// fallback (not reported): osl=10 -> 9 tokens; window 1800ms -> 200ms
		{latency: 1900 * time.Millisecond, ttft: 100 * time.Millisecond, completionTokens: 10, streamed: true},
		// per-chunk: osl=8, first=3 -> 5 tokens; window 1000ms -> 200ms
		{latency: 1500 * time.Millisecond, ttft: 500 * time.Millisecond, completionTokens: 8, streamed: true, firstChunkTokens: itlIntPtr(3)},
		// fallback (whole response in one chunk): osl=6, first=6 -> 5 tokens; window 500ms -> 100ms
		{latency: 1000 * time.Millisecond, ttft: 500 * time.Millisecond, completionTokens: 6, streamed: true, firstChunkTokens: itlIntPtr(6)},
	}
	pr := aggregate(1, 4, 2*time.Second, results)
	if pr.PerChunkUsageDivisorRequests != 2 {
		t.Fatalf("PerChunkUsageDivisorRequests=%d, want 2", pr.PerChunkUsageDivisorRequests)
	}
	// itls in fold order: 100, 200, 200, 100 -> mean exactly 150
	if pr.MeanITLMS != 150 {
		t.Fatalf("MeanITLMS=%v, want 150", pr.MeanITLMS)
	}
	// sorted [100,100,200,200]: nearest-rank p50 -> 100, p95 -> 200
	if pr.ITLP50MS != 100 || pr.ITLP95MS != 200 {
		t.Fatalf("itl p50/p95=%v/%v, want 100/200", pr.ITLP50MS, pr.ITLP95MS)
	}
}

// TestITLAggregate_PercentilesKnownInput: five known ITL values (100..500ms) pin
// the nearest-rank percentiles and the mean exactly on a known input set.
func TestITLAggregate_PercentilesKnownInput(t *testing.T) {
	var results []reqResult
	// osl=11 with no per-chunk usage -> fallback divisor 10; window ms / 10 = itl.
	for _, windowMS := range []int{1000, 2000, 3000, 4000, 5000} {
		results = append(results, reqResult{
			latency:          time.Duration(windowMS)*time.Millisecond + 100*time.Millisecond,
			ttft:             100 * time.Millisecond,
			completionTokens: 11,
			streamed:         true,
		})
	}
	pr := aggregate(1, 5, 3*time.Second, results)
	if pr.MeanITLMS != 300 {
		t.Fatalf("MeanITLMS=%v, want 300", pr.MeanITLMS)
	}
	// sorted [100..500]: nearest-rank p50 -> 300, p95 -> 500
	if pr.ITLP50MS != 300 || pr.ITLP95MS != 500 {
		t.Fatalf("itl p50/p95=%v/%v, want 300/500", pr.ITLP50MS, pr.ITLP95MS)
	}
	if pr.PerChunkUsageDivisorRequests != 0 {
		t.Fatalf("PerChunkUsageDivisorRequests=%d, want 0 (all fallback)", pr.PerChunkUsageDivisorRequests)
	}
}

// TestITLPointResultJSONShape pins the wire names of the ITL fields and their
// omitempty behavior: a point with ITL contributors carries all four keys; a
// point with none carries none.
func TestITLPointResultJSONShape(t *testing.T) {
	itlKeys := []string{"mean_itl_ms", "itl_p50_ms", "itl_p95_ms", "per_chunk_usage_divisor_requests"}

	populated, err := json.Marshal(PointResult{MeanITLMS: 123.5, ITLP50MS: 100, ITLP95MS: 150, PerChunkUsageDivisorRequests: 2})
	if err != nil {
		t.Fatalf("marshal populated: %v", err)
	}
	var fields map[string]any
	if err := json.Unmarshal(populated, &fields); err != nil {
		t.Fatalf("unmarshal populated: %v", err)
	}
	for _, key := range itlKeys {
		if _, ok := fields[key]; !ok {
			t.Fatalf("populated point missing JSON key %q in %s", key, populated)
		}
	}

	empty, err := json.Marshal(PointResult{})
	if err != nil {
		t.Fatalf("marshal empty: %v", err)
	}
	fields = nil // fresh map: json.Unmarshal merges into a non-nil map; stale keys would defeat the omission assertion
	if err := json.Unmarshal(empty, &fields); err != nil {
		t.Fatalf("unmarshal empty: %v", err)
	}
	for _, key := range itlKeys {
		if _, ok := fields[key]; ok {
			t.Fatalf("empty point must omit JSON key %q (omitempty), got %s", key, empty)
		}
	}
}
