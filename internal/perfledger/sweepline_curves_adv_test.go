// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"testing"
)

// Chunk-2 adversarial coverage, hand-derived from the upstream Python
// (aiperf sweepline.py / sweepline_kv_cache.py), not from the Go port.
// Timestamps are small integer nanoseconds so per-record rates are O(1) and
// the 1e-7 tolerance is meaningful; NaN marks a missing record exactly as the
// numpy masks treat it.

// Semantics 6: decode rate = (output-1)/gen_dur per record, valid only when
// gen_dur > 0, output >= 1, and both non-NaN (Python :543-551 gn_valid mask);
// invalid records contribute NO events. output == 1 is valid but contributes
// a zero-rate event pair (delta 0 sorts as an end-type event).
func TestSweepLineThroughputSweepLineRatesAndValidity(t *testing.T) {
	genStart := []float64{10, 20, 15, 25, math.NaN(), 50}
	end := []float64{30, 40, 15, 25, 70, 60}
	out := []float64{5, 3, 0, 2, 2, 1}
	// R0: rate 4/20=0.2; R1: 2/20=0.1; R2 out=0 invalid; R3 dur=0 invalid;
	// R4 NaN gen invalid; R5 out=1 -> rate 0, valid.
	gotTS, gotVals := ThroughputSweepLine(genStart, end, out)
	wantTS := []float64{10, 20, 30, 40, 50, 60}
	wantVals := []float64{0.2, 0.3, 0.1, 0, 0, 0}
	sweepCheck(t, "ThroughputSweepLine rates+validity", gotTS, gotVals, wantTS, wantVals, 1e-7)
}

// Semantics 7: prefill rate = input/(gen_start-start), valid only when the
// prefill duration is > 0 and start/gen_start/input are non-NaN (Python
// :535-540 pf_valid mask). Includes an end/start tie at t=30: the -2 end
// event sorts before the +0.5 start event.
func TestSweepLinePrefillThroughputSweepLineRatesAndValidity(t *testing.T) {
	start := []float64{10, 20, 25, 30, math.NaN()}
	gen := []float64{30, 50, 25, 70, 40}
	in := []float64{40, 30, 5, 20, 5}
	// R0: 40/20=2; R1: 30/30=1; R2 dur=0 invalid; R3: 20/40=0.5; R4 NaN start.
	gotTS, gotVals := PrefillThroughputSweepLine(start, gen, in)
	// Sorted: +2@10, +1@20, -2@30 (end-type), +0.5@30, -1@50, -0.5@70.
	wantTS := []float64{10, 20, 30, 30, 50, 70}
	wantVals := []float64{2, 3, 1, 1.5, 0.5, 0}
	sweepCheck(t, "PrefillThroughputSweepLine rates+validity", gotTS, gotVals, wantTS, wantVals, 1e-7)
}

// Semantics 8: total throughput is ONE combined sweep of prefill events
// [start,gen_start) and generation events [gen_start,end) (Python :556-569).
// Ties at 30 and 40 put each record's prefill -rate end event before its own
// generation +rate start event, so the curve dips between phases.
func TestSweepLineTotalThroughputSweepLineCombinedSweep(t *testing.T) {
	start := []float64{10, 20, math.NaN()}
	gen := []float64{30, 40, 50}
	end := []float64{60, 50, 70}
	in := []float64{40, 20, 10}
	out := []float64{7, 3, 5}
	// R0: pf 40/20=2 over [10,30), gn 6/30=0.2 over [30,60).
	// R1: pf 20/20=1 over [20,40), gn 2/10=0.2 over [40,50).
	// R2: NaN start -> prefill invalid, but the gen phase is VALID because
	// gn_valid ignores start_ns (Python :545-550): (5-1)/20=0.2 over [50,70).
	gotTS, gotVals := TotalThroughputSweepLine(start, gen, end, in, out)
	// Sorted: +2@10, +1@20, -2@30, +0.2@30, -1@40, +0.2@40, -0.2@50, +0.2@50,
	// -0.2@60, -0.2@70.
	wantTS := []float64{10, 20, 30, 30, 40, 40, 50, 50, 60, 70}
	wantVals := []float64{2, 3, 1, 1.2, 0.2, 0.4, 0.2, 0.4, 0.2, 0}
	sweepCheck(t, "TotalThroughputSweepLine combined", gotTS, gotVals, wantTS, wantVals, 1e-7)
}

// Semantics 9: KV-cache events per Python _kv_cache_events (:27-65):
// +input at start only when gen_start > start; +output at gen_start when
// gen_dur > 0; free at end as -(in+out) / -in / -out for the
// both/input_only/gen_only branches. A record with NaN gen_start contributes
// nothing even with valid start/input/end; a record with NaN end never frees
// its input (curve stays elevated forever).
func TestSweepLineTokensInFlightBranchDecomposition(t *testing.T) {
	start := []float64{10, 15, math.NaN(), 12, math.NaN()}
	gen := []float64{20, 25, 30, 22, 40}
	end := []float64{50, 40, 45, math.NaN(), 55}
	in := []float64{100, 50, math.NaN(), 10, math.NaN()}
	out := []float64{30, math.NaN(), 70, 5, 20}
	// R0 both: +100@10, +30@20, -130@50.
	// R1 input_only (out NaN kills has_gen): +50@15, -50@40.
	// R2 gen_only (NaN start kills pf_valid): +70@30, -70@45.
	// R3: +10@12, never freed (NaN end).
	// R4 gen_only: +20@40 — ties with R1's -50@40 end event, which sorts first.
	gotTS, gotVals := TokensInFlightSweepLine(start, gen, end, in, out)
	wantTS := []float64{10, 12, 15, 20, 30, 40, 40, 45, 50, 55}
	wantVals := []float64{100, 110, 160, 190, 260, 210, 230, 160, 30, 10}
	sweepCheck(t, "TokensInFlightSweepLine branches", gotTS, gotVals, wantTS, wantVals, 1e-7)
}

// Semantics 12: per-user curves divide the aggregate rate curve by the phase
// concurrency curve on the merged grid (Python add/divide :302-337); idle
// gaps where concurrency is 0 force the quotient to 0 via safe division.
func TestSweepLinePerUserDivideByConcurrency(t *testing.T) {
	// Decode: two disjoint generations [10,30) and [50,60) with an idle gap.
	gen := []float64{10, 50}
	end := []float64{30, 60}
	out := []float64{11, 3}
	tputTS, tputVals := ThroughputSweepLine(gen, end, out)
	// Rates 10/20=0.5 and 2/10=0.2 -> tput [0.5,0,0.2,0]; conc [1,0,1,0].
	puTS, puVals := ThroughputPerUserSweepLine(gen, end, tputTS, tputVals)
	wantTS := []float64{10, 30, 50, 60}
	wantVals := []float64{0.5, 0, 0.2, 0}
	sweepCheck(t, "ThroughputPerUser idle gap", puTS, puVals, wantTS, wantVals, 1e-7)

	// Prefill: overlapping prefills [10,30) and [20,40) so concurrency hits 2.
	pStart := []float64{10, 20}
	pGen := []float64{30, 40}
	pIn := []float64{40, 30}
	ptputTS, ptputVals := PrefillThroughputSweepLine(pStart, pGen, pIn)
	// Rates 2 and 1.5 -> ptput [2,3.5,1.5,0]; conc over [start,gen) [1,2,1,0].
	ppuTS, ppuVals := PrefillThroughputPerUserSweepLine(pStart, pGen, ptputTS, ptputVals)
	wantPT := []float64{10, 20, 30, 40}
	wantPV := []float64{2, 1.75, 1.5, 0}
	sweepCheck(t, "PrefillThroughputPerUser overlap", ppuTS, ppuVals, wantPT, wantPV, 1e-7)
}

// sweepCurve legally absorbs a multi-value sweep call: Go forbids a direct
// f(g()) multi-value call inside a composite literal, so each table entry
// routes the (ts, vals) pair through this two-parameter helper instead.
type sweepCurve struct {
	ts   []float64
	vals []float64
}

func sweepCurveOf(ts, vals []float64) sweepCurve { return sweepCurve{ts: ts, vals: vals} }

// Semantics 11: empty inputs and all-invalid records yield empty curves
// (Python returns np.zeros(0) pairs when no event survives the masks).
func TestSweepLineEmptyAndAllInvalidInputs(t *testing.T) {
	nan := math.NaN()
	cases := []struct {
		name  string
		curve sweepCurve
	}{
		{"concurrency empty", sweepCurveOf(ConcurrencySweepLine([]float64{}, []float64{}))},
		{"concurrency all NaN", sweepCurveOf(ConcurrencySweepLine([]float64{nan, 10}, []float64{20, nan}))},
		{"weighted NaN weight", sweepCurveOf(WeightedConcurrencySweepLine([]float64{10}, []float64{20}, []float64{nan}))},
		{"throughput out<1", sweepCurveOf(ThroughputSweepLine([]float64{10}, []float64{20}, []float64{0}))},
		{"throughput NaN gen", sweepCurveOf(ThroughputSweepLine([]float64{nan}, []float64{20}, []float64{5}))},
		{"throughput mixed invalid", sweepCurveOf(ThroughputSweepLine([]float64{nan, 10, 10}, []float64{20, 10, 10}, []float64{5, 0, 1}))},
		{"prefill dur=0", sweepCurveOf(PrefillThroughputSweepLine([]float64{10}, []float64{10}, []float64{5}))},
		{"total all NaN", sweepCurveOf(TotalThroughputSweepLine([]float64{nan}, []float64{nan}, []float64{nan}, []float64{nan}, []float64{nan}))},
		{"total zero durations", sweepCurveOf(TotalThroughputSweepLine([]float64{10}, []float64{10}, []float64{10}, []float64{5}, []float64{5}))},
		{"tokens in flight NaN gen", sweepCurveOf(TokensInFlightSweepLine([]float64{10}, []float64{nan}, []float64{20}, []float64{5}, []float64{1}))},
	}
	for _, c := range cases {
		if len(c.curve.ts) != 0 || len(c.curve.vals) != 0 {
			t.Errorf("%s: got (%v, %v) want empty curves", c.name, c.curve.ts, c.curve.vals)
		}
	}
}
