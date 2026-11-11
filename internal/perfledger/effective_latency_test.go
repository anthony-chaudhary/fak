// fak-test:runtime fast est=10ms
package perfledger

// Adversarial spec tests for the effective-latency pair (direct port of
// NVIDIA aiperf derived_latency.py). Every expected number is hand-computed
// in comments so a reviewer can verify each vector without running numpy.

import (
	"math"
	"testing"
)

// effSample builds one EffectiveSample whose latency is deltaMS milliseconds
// measured from a credit instant creditMS milliseconds after the epoch, so
// EndNS - CreditIssuedNS == deltaMS exactly in whole milliseconds.
func effSample(deltaMS, creditMS int64, errored bool) EffectiveSample {
	creditNS := creditMS * 1_000_000
	return EffectiveSample{
		EndNS:          creditNS + deltaMS*1_000_000,
		CreditIssuedNS: creditNS,
		Errored:        errored,
	}
}

// effRec builds a minimal ledger row. synthetic=true marks a readiness canary
// (IsProbe via Synthetic); prompt/cached token counts drive the other IsProbe
// path (cached>0 && prompt+cached<=32).
func effRec(requestID string, unixMS int64, errClass string, synthetic bool, prompt, cached int) Record {
	return Record{
		RequestID:    requestID,
		UnixMS:       unixMS,
		Error:        errClass,
		Synthetic:    synthetic,
		PromptTokens: prompt,
		CachedTokens: cached,
	}
}

// effNoCredit refuses every row: the credit instant was never observed.
func effNoCredit(Record) (int64, bool) { return 0, false }

// effWant is the expected EffectiveLatency field by field. Inf/NaN wants are
// matched with math.IsInf/math.IsNaN; finite wants with a 1e-9 epsilon.
type effWant struct {
	count int
	sum   float64
	avg   float64
	std   float64
	min   float64
	max   float64
	p1    float64
	p5    float64
	p10   float64
	p25   float64
	p50   float64
	p75   float64
	p90   float64
	p95   float64
	p99   float64
}

func effSameFloat(t *testing.T, label string, got, want float64) {
	t.Helper()
	switch {
	case math.IsNaN(want):
		if !math.IsNaN(got) {
			t.Errorf("%s = %v, want NaN", label, got)
		}
	case math.IsInf(want, 1):
		if !math.IsInf(got, 1) {
			t.Errorf("%s = %v, want +Inf", label, got)
		}
	case math.Abs(got-want) > 1e-9:
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

func effCheck(t *testing.T, got *EffectiveLatency, want effWant) {
	t.Helper()
	if got == nil {
		t.Fatalf("got nil, want %+v", want)
	}
	if got.Count != want.count {
		t.Errorf("Count = %d, want %d", got.Count, want.count)
	}
	effSameFloat(t, "SumMS", got.SumMS, want.sum)
	effSameFloat(t, "AvgMS", got.AvgMS, want.avg)
	effSameFloat(t, "StdMS", got.StdMS, want.std)
	effSameFloat(t, "MinMS", got.MinMS, want.min)
	effSameFloat(t, "MaxMS", got.MaxMS, want.max)
	effSameFloat(t, "P1MS", got.P1MS, want.p1)
	effSameFloat(t, "P5MS", got.P5MS, want.p5)
	effSameFloat(t, "P10MS", got.P10MS, want.p10)
	effSameFloat(t, "P25MS", got.P25MS, want.p25)
	effSameFloat(t, "P50MS", got.P50MS, want.p50)
	effSameFloat(t, "P75MS", got.P75MS, want.p75)
	effSameFloat(t, "P90MS", got.P90MS, want.p90)
	effSameFloat(t, "P95MS", got.P95MS, want.p95)
	effSameFloat(t, "P99MS", got.P99MS, want.p99)
}

func effCheckNil(t *testing.T, got *EffectiveLatency) {
	t.Helper()
	if got != nil {
		t.Errorf("got %+v, want nil", *got)
	}
}

// TestEffectiveLatency is the core happy path: five successes at 10/20/30/40/50
// ms off a shared credit instant at 1e12 ns. Sorted population [10,20,30,40,50],
// n=5, linear-interpolation rank r = q*(n-1) = q*4:
//
//	P1  r=0.04 -> 10 + 0.04*10 = 10.4      P90 r=3.60 -> 40 + 0.6*10 = 46
//	P5  r=0.20 -> 12                       P95 r=3.80 -> 48
//	P10 r=0.40 -> 14                       P99 r=3.96 -> 49.6
//	P25 r=1.00 -> 20 (exact element)
//	P50 r=2.00 -> 30   P75 r=3.00 -> 40
//
// Sum=150, Avg=30, Min=10, Max=50. Population std (ddof=0): mean 30, squared
// devs 400+100+0+100+400 = 1000, /5 = 200, sqrt = 14.142135623730951.
func TestEffectiveLatency(t *testing.T) {
	samples := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(20, 1_000_000, false),
		effSample(30, 1_000_000, false),
		effSample(40, 1_000_000, false),
		effSample(50, 1_000_000, false),
	}
	effCheck(t, ComputeEffectiveLatency(samples), effWant{
		count: 5, sum: 150, avg: 30, std: math.Sqrt(200), min: 10, max: 50,
		p1: 10.4, p5: 12, p10: 14, p25: 20, p50: 30, p75: 40, p90: 46, p95: 48, p99: 49.6,
	})
}

// TestEffectiveLatencyClampNegative: EndNS 5ms BEFORE the credit instant
// (epoch-ns ULP/skew) gives delta = -5ms, clamped to 0. Paired with a clean
// 25ms so the clamped zero participates in real statistics. Sorted [0, 25],
// n=2, rank r = q*1:
//
//	P1  0.01 -> 0.25    P50 0.50 -> 12.5    P90 0.90 -> 22.5
//	P5  0.05 -> 1.25    P75 0.75 -> 18.75   P95 0.95 -> 23.75
//	P10 0.10 -> 2.5     P25 0.25 -> 6.25    P99 0.99 -> 24.75
//
// Sum=25, Avg=12.5, Min=0, Max=25. Population std: devs -12.5/+12.5 ->
// (156.25+156.25)/2 = 156.25, sqrt = 12.5.
func TestEffectiveLatencyClampNegative(t *testing.T) {
	samples := []EffectiveSample{
		effSample(-5, 1_000_000, false), // EndNS = 1e12 - 5e6: delta -5ms -> 0
		effSample(25, 2_000_000, false),
	}
	effCheck(t, ComputeEffectiveLatency(samples), effWant{
		count: 2, sum: 25, avg: 12.5, std: 12.5, min: 0, max: 25,
		p1: 0.25, p5: 1.25, p10: 2.5, p25: 6.25, p50: 12.5, p75: 18.75, p90: 22.5, p95: 23.75, p99: 24.75,
	})
}

// TestEffectiveLatencyCreditGate: CreditIssuedNS==0 means "not issued" — such
// samples contribute NOTHING to either metric.
func TestEffectiveLatencyCreditGate(t *testing.T) {
	// A: success 10ms with credit. B: success 10ms, credit==0. C: errored,
	// credit==0. B and C are invisible to BOTH metrics: base sees only A.
	mixed := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(10, 0, false), // credit==0: not issued -> contributes nothing
		effSample(10, 0, true),  // credit==0 errored: contributes nothing either
	}
	// Base over [10], n=1: every rank collapses to the only element, std=0.
	effCheck(t, ComputeEffectiveLatency(mixed), effWant{
		count: 1, sum: 10, avg: 10, std: 0, min: 10, max: 10,
		p1: 10, p5: 10, p10: 10, p25: 10, p50: 10, p75: 10, p90: 10, p95: 10, p99: 10,
	})
	// Adjusted: the only credited sample is a success -> zero credited errors
	// -> nil (S5/S6).
	effCheckNil(t, ComputeAdjustedEffectiveLatency(mixed))

	// No sample carries a credit instant -> both nil (S3).
	allMissing := []EffectiveSample{
		effSample(10, 0, false),
		effSample(20, 0, true),
	}
	effCheckNil(t, ComputeEffectiveLatency(allMissing))
	effCheckNil(t, ComputeAdjustedEffectiveLatency(allMissing))

	// Empty and nil input -> nil (S3).
	effCheckNil(t, ComputeEffectiveLatency(nil))
	effCheckNil(t, ComputeEffectiveLatency([]EffectiveSample{}))
	effCheckNil(t, ComputeAdjustedEffectiveLatency(nil))
	effCheckNil(t, ComputeAdjustedEffectiveLatency([]EffectiveSample{}))
}

// TestEffectiveLatencySuccessOnly: errored samples are excluded from the base
// population entirely (S2). Successes 10/20/30; an errored sample WITH credit
// (999ms) and one WITHOUT credit (888ms) must both be invisible to the base.
func TestEffectiveLatencySuccessOnly(t *testing.T) {
	samples := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(20, 1_000_000, false),
		effSample(30, 1_000_000, false),
		effSample(999, 1_000_000, true), // errored, credited: excluded from base
		effSample(888, 0, true),         // errored, uncredited: excluded everywhere
	}
	// Sorted successes [10,20,30], n=3, rank r = q*2:
	//
	//	P1  0.02 -> 10.2   P25 0.5 -> 15    P75 1.5 -> 25
	//	P5  0.10 -> 11     P50 1.0 -> 20   P90 1.8 -> 28
	//	P10 0.20 -> 12     P95 1.9 -> 29   P99 1.98 -> 29.8
	//
	// Sum=60, Avg=20, Min=10, Max=30. Population std: devs -10/0/+10 ->
	// (100+0+100)/3 = 200/3, sqrt = 8.164965809277263.
	effCheck(t, ComputeEffectiveLatency(samples), effWant{
		count: 3, sum: 60, avg: 20, std: math.Sqrt(200.0 / 3.0), min: 10, max: 30,
		p1: 10.2, p5: 11, p10: 12, p25: 15, p50: 20, p75: 25, p90: 28, p95: 29, p99: 29.8,
	})

	// Exclusion taken to its limit: ONLY credited errors -> the success-only
	// population is empty -> no base metric exists.
	effCheckNil(t, ComputeEffectiveLatency([]EffectiveSample{
		effSample(10, 1_000_000, true),
		effSample(20, 1_000_000, true),
	}))
}

// TestEffectiveLatencyAdjusted: +Inf inflation. Clean successes 10/30ms; two
// errored samples WITH credit (each inflates the adjusted distribution by one
// +Inf) and one errored WITHOUT credit (invisible everywhere). Inflated
// multiset sorted [10, 30, +Inf, +Inf], n=4, nearest-rank
// r = round-half-to-even(q*3) clamped to [0,3]:
//
//	P1  0.03 -> 0 -> 10      P50 1.50 -> 2 (half-to-even) -> +Inf
//	P5  0.15 -> 0 -> 10      P75 2.25 -> 2 -> +Inf
//	P10 0.30 -> 0 -> 10      P90 2.70 -> 3 -> +Inf
//	P25 0.75 -> 1 -> 30      P95 2.85 -> 3 -> +Inf   P99 2.97 -> 3 -> +Inf
//
// Count = 2 successes + 2 credited errors = 4. Sum/Avg/Max = +Inf (any error
// present), Std = NaN (undefined with Inf), Min = clean min 10.
func TestEffectiveLatencyAdjusted(t *testing.T) {
	samples := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(30, 1_000_000, false),
		effSample(70, 1_000_000, true), // own latency irrelevant: reported as +Inf
		effSample(80, 1_000_000, true),
		effSample(90, 0, true), // uncredited error: invisible
	}
	effCheck(t, ComputeAdjustedEffectiveLatency(samples), effWant{
		count: 4, sum: math.Inf(1), avg: math.Inf(1), std: math.NaN(), min: 10, max: math.Inf(1),
		p1: 10, p5: 10, p10: 10, p25: 30, p50: math.Inf(1), p75: math.Inf(1),
		p90: math.Inf(1), p95: math.Inf(1), p99: math.Inf(1),
	})

	// Zero credited errors -> adjusted is nil even with a healthy base (S5/S6).
	effCheckNil(t, ComputeAdjustedEffectiveLatency([]EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(20, 1_000_000, false),
	}))

	// Credited errors but NO successes -> nil (S5).
	effCheckNil(t, ComputeAdjustedEffectiveLatency([]EffectiveSample{
		effSample(10, 1_000_000, true),
		effSample(20, 1_000_000, true),
	}))
}

// TestEffectiveLatencyPercentileInterpolation: dedicated vectors for both
// percentile methods.
func TestEffectiveLatencyPercentileInterpolation(t *testing.T) {
	// Linear interpolation, vector [10,20,30,40], n=4, r = q*3:
	//
	//	P1  0.03 -> 10.3    P25 0.75 -> 17.5   P75 2.25 -> 32.5
	//	P5  0.15 -> 11.5    P50 1.50 -> 25 (lands BETWEEN elements 20 and 30)
	//	P10 0.30 -> 13      P90 2.70 -> 37
	//	P95 2.85 -> 38.5    P99 2.97 -> 39.7
	//
	// Sum=100, Avg=25, Min=10, Max=40. Population std: devs -15/-5/5/15 ->
	// (225+25+25+225)/4 = 125, sqrt = 11.180339887498949.
	four := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(20, 1_000_000, false),
		effSample(30, 1_000_000, false),
		effSample(40, 1_000_000, false),
	}
	effCheck(t, ComputeEffectiveLatency(four), effWant{
		count: 4, sum: 100, avg: 25, std: math.Sqrt(125), min: 10, max: 40,
		p1: 10.3, p5: 11.5, p10: 13, p25: 17.5, p50: 25, p75: 32.5, p90: 37, p95: 38.5, p99: 39.7,
	})

	// Exact-element ranks: [100,200,300,400,500], n=5, r = q*4 lands on whole
	// elements for P25 (r=1 -> 200), P50 (r=2 -> 300), P75 (r=3 -> 400).
	//
	//	P1 0.04 -> 104   P5 0.2 -> 120   P10 0.4 -> 140
	//	P90 3.6 -> 460   P95 3.8 -> 480  P99 3.96 -> 496
	//
	// Sum=1500, Avg=300, Min=100, Max=500. Population std: devs
	// -200/-100/0/100/200 -> 100000/5 = 20000, sqrt = 141.42135623730951.
	five := []EffectiveSample{
		effSample(100, 1_000_000, false),
		effSample(200, 1_000_000, false),
		effSample(300, 1_000_000, false),
		effSample(400, 1_000_000, false),
		effSample(500, 1_000_000, false),
	}
	effCheck(t, ComputeEffectiveLatency(five), effWant{
		count: 5, sum: 1500, avg: 300, std: math.Sqrt(20000), min: 100, max: 500,
		p1: 104, p5: 120, p10: 140, p25: 200, p50: 300, p75: 400, p90: 460, p95: 480, p99: 496,
	})

	// Single sample: every rank collapses to the only element; std = 0.
	one := []EffectiveSample{effSample(42, 1_000_000, false)}
	effCheck(t, ComputeEffectiveLatency(one), effWant{
		count: 1, sum: 42, avg: 42, std: 0, min: 42, max: 42,
		p1: 42, p5: 42, p10: 42, p25: 42, p50: 42, p75: 42, p90: 42, p95: 42, p99: 42,
	})

	// Nearest-rank half-to-even discriminator for the ADJUSTED view: clean
	// [10, 30] + one credited error -> inflated [10, 30, +Inf], n=3,
	// r = round-half-to-even(q*2):
	//
	//	P25 r=0.5 -> 0 (0 is EVEN; round-half-up would give 1 -> 30) -> 10
	//	P50 r=1.0 -> 30
	//	P75 r=1.5 -> 2 (2 is even) -> +Inf
	//	P1/P5/P10 r=0.02/0.1/0.2 -> 0 -> 10;  P90/P95/P99 -> 2 -> +Inf
	//
	// Count=3, Sum/Avg/Max=+Inf, Std=NaN, Min=10.
	adj := []EffectiveSample{
		effSample(10, 1_000_000, false),
		effSample(30, 1_000_000, false),
		effSample(70, 1_000_000, true),
	}
	effCheck(t, ComputeAdjustedEffectiveLatency(adj), effWant{
		count: 3, sum: math.Inf(1), avg: math.Inf(1), std: math.NaN(), min: 10, max: math.Inf(1),
		p1: 10, p5: 10, p10: 10, p25: 10, p50: 30, p75: math.Inf(1),
		p90: math.Inf(1), p95: math.Inf(1), p99: math.Inf(1),
	})
}

// TestEffectiveLatencyRecordsAdapter: ledger rows -> samples. EndNS =
// UnixMS*1e6, Errored = Error != "", probe rows skipped (both IsProbe paths:
// Synthetic canary and the cached>0 && prompt+cached<=32 token shape), rows
// whose accessor reports no credit instant skipped, order preserved.
func TestEffectiveLatencyRecordsAdapter(t *testing.T) {
	recs := []Record{
		effRec("r1", 1000, "", false, 100, 0),                   // served: 100 uncached tokens, not a probe
		effRec("r2", 2000, "client_write_timeout", false, 0, 0), // errored turn
		effRec("r3", 3000, "", true, 0, 0),                      // Synthetic canary -> IsProbe -> skipped
		effRec("r4", 3500, "", false, 10, 20),                   // cached>0, 10+20<=32 -> IsProbe -> skipped
		effRec("r5", 4000, "", false, 0, 0),                     // accessor reports no credit -> skipped
		effRec("r6", 4000, "", false, 0, 0),                     // served, credited
	}
	credit := func(r Record) (int64, bool) {
		switch r.RequestID {
		case "r1":
			return 500_000_000, true
		case "r2":
			return 1_500_000_000, true
		case "r3", "r4":
			// Probe rows are skipped regardless of what the accessor says.
			return 3_000_000_000, true
		case "r6":
			return 3_900_000_000, true
		}
		return 0, false
	}
	got := EffectiveSamplesFromRecords(recs, credit)
	want := []EffectiveSample{
		{EndNS: 1_000_000_000, CreditIssuedNS: 500_000_000, Errored: false},   // 500ms
		{EndNS: 2_000_000_000, CreditIssuedNS: 1_500_000_000, Errored: true},  // 500ms
		{EndNS: 4_000_000_000, CreditIssuedNS: 3_900_000_000, Errored: false}, // 100ms
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d (%+v), want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	// Nil accessor -> nil (S7).
	if g := EffectiveSamplesFromRecords(recs, nil); g != nil {
		t.Errorf("nil accessor: got %+v, want nil", g)
	}
	// Empty input -> nil (S7).
	if g := EffectiveSamplesFromRecords(nil, credit); g != nil {
		t.Errorf("nil recs: got %+v, want nil", g)
	}
	if g := EffectiveSamplesFromRecords([]Record{}, credit); g != nil {
		t.Errorf("empty recs: got %+v, want nil", g)
	}
	// An accessor that refuses every row yields no samples (S7).
	if g := EffectiveSamplesFromRecords(recs, effNoCredit); g != nil {
		t.Errorf("all-refused accessor: got %+v, want nil", g)
	}
}
