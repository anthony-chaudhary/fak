package dojo

import (
	"math"
	"strings"
	"testing"
)

func pred(metric string, claimed float64) Prediction {
	return Prediction{Lever: "lvr", Metric: metric, Claimed: claimed, Unit: "fraction", Basis: "test"}
}

func obs(realized float64, measured bool) Outcome {
	return Outcome{Realized: realized, Provenance: Observed, Source: "test", Measured: measured, Sample: 10}
}

func TestScoreCalibrated(t *testing.T) {
	e := Score("s", pred("posture_accuracy", 1.0), obs(0.977, true), DefaultCalibBand())
	if e.Verdict != VerdictCalibrated {
		t.Fatalf("want CALIBRATED, got %s (calib_err %.4f)", e.Verdict, e.CalibErr)
	}
	if e.Grade != "A" {
		t.Fatalf("want grade A, got %s", e.Grade)
	}
	if e.Residual >= 0 {
		t.Fatalf("realized below claim should give negative residual, got %.4f", e.Residual)
	}
}

func TestScoreOverClaim(t *testing.T) {
	// the projection claims ~85% of the resident is rewritten (0.85); reality is 0.68.
	e := Score("s", pred("cold_write_share", 0.85), obs(0.68, true), DefaultCalibBand())
	if e.Verdict != VerdictOverClaim {
		t.Fatalf("want OVER_CLAIM, got %s", e.Verdict)
	}
	if e.CalibErr < 0.19 || e.CalibErr > 0.21 {
		t.Fatalf("calib_err should be ~0.20, got %.4f", e.CalibErr)
	}
	if e.Grade != "B" {
		t.Fatalf("0.20 calib-err should grade B, got %s", e.Grade)
	}
}

func TestScoreUnderClaim(t *testing.T) {
	// the within-session model claims ~17% cross-session reuse (0.17); reality has it.
	e := Score("s", pred("cross_session_warm_hit_rate", 0.17), obs(0.30, true), DefaultCalibBand())
	if e.Verdict != VerdictUnderClaim {
		t.Fatalf("want UNDER_CLAIM, got %s", e.Verdict)
	}
	// calib_err = |0.30-0.17|/0.17 = 0.13/0.17 = ~0.76
	if e.CalibErr < 0.75 || e.CalibErr > 0.77 {
		t.Fatalf("calib_err should be ~0.76, got %.4f", e.CalibErr)
	}
	if e.Grade != "F" {
		t.Fatalf("calib-err ~0.76 should grade F, got %s", e.Grade)
	}
}

func TestScoreUnmeasured(t *testing.T) {
	e := Score("s", pred("posture_accuracy", 1.0), obs(0, false), DefaultCalibBand())
	if e.Verdict != VerdictUnmeasured {
		t.Fatalf("want UNMEASURED, got %s", e.Verdict)
	}
	if e.Grade != gradeNA {
		t.Fatalf("unmeasured should grade %q, got %q", gradeNA, e.Grade)
	}
	if e.CalibErr != 0 {
		t.Fatalf("unmeasured should not score a calib_err, got %.4f", e.CalibErr)
	}
}

func TestScoreClaimZeroRealizedZero(t *testing.T) {
	// a claim of "nothing" that reality confirms is perfectly calibrated, not Inf.
	e := Score("s", pred("noop", 0.0), obs(0.0, true), DefaultCalibBand())
	if e.CalibErr != 0 {
		t.Fatalf("0 vs 0 should be perfectly calibrated, got %.4f", e.CalibErr)
	}
	if e.Verdict != VerdictCalibrated {
		t.Fatalf("want CALIBRATED, got %s", e.Verdict)
	}
}

func TestCalibErrCapped(t *testing.T) {
	// a wildly wrong near-zero claim must not produce an unbounded ratio.
	e := Score("s", pred("tiny", 0.0001), obs(100.0, true), DefaultCalibBand())
	if e.CalibErr != MaxCalibErr {
		t.Fatalf("calib_err should cap at %.1f, got %.4f", MaxCalibErr, e.CalibErr)
	}
}

func TestFoldEmpty(t *testing.T) {
	r := Fold(nil, FoldOpts{Date: "2026-06-26"})
	if r.OK || r.Finding != "dojo_empty" {
		t.Fatalf("empty fold should be ACTION/dojo_empty, got ok=%v finding=%s", r.OK, r.Finding)
	}
	if r.Grade != gradeNA {
		t.Fatalf("an empty fold measured nothing - grade should be %q not %q (vacuous A contradicts ok:false)", gradeNA, r.Grade)
	}
	if code, _ := CheckGate(r); code != 1 {
		t.Fatalf("empty fold gate should fail, got %d", code)
	}
}

func TestFoldAllUnmeasured(t *testing.T) {
	eps := []Episode{
		Score("s", pred("a", 1), obs(0, false), DefaultCalibBand()),
		Score("s", pred("b", 1), obs(0, false), DefaultCalibBand()),
	}
	r := Fold(eps, FoldOpts{})
	if r.Finding != "dojo_unmeasured" || r.OK {
		t.Fatalf("all-unmeasured should be ACTION/dojo_unmeasured, got ok=%v finding=%s", r.OK, r.Finding)
	}
	if r.Grade != gradeNA {
		t.Fatalf("an all-unmeasured fold should grade %q not %q", gradeNA, r.Grade)
	}
	if r.Unmeasured != 2 || r.Measured != 0 {
		t.Fatalf("counts wrong: measured=%d unmeasured=%d", r.Measured, r.Unmeasured)
	}
}

func TestFoldRecordedWithOverClaimAdvisory(t *testing.T) {
	eps := []Episode{
		Score("s", pred("posture_accuracy", 1.0), obs(0.977, true), DefaultCalibBand()), // calibrated
		Score("s", pred("cold_write_share", 0.85), obs(0.68, true), DefaultCalibBand()), // over-claim
		Score("s", pred("warm_hit", 0.17), obs(0.30, true), DefaultCalibBand()),         // under-claim
	}
	r := Fold(eps, FoldOpts{Date: "2026-06-26", Commit: "abc123def456"})
	if !r.OK || r.Finding != "dojo_recorded" {
		t.Fatalf("recorded fold should be OK/dojo_recorded, got ok=%v finding=%s", r.OK, r.Finding)
	}
	if r.Measured != 3 || r.Calibrated != 1 {
		t.Fatalf("counts wrong: measured=%d calibrated=%d", r.Measured, r.Calibrated)
	}
	if r.LeverCount != 1 {
		t.Fatalf("all share lever 'lvr', want LeverCount 1, got %d", r.LeverCount)
	}
	if !strings.Contains(r.Reason, "over-claim") {
		t.Fatalf("recorded reason should carry the over-claim advisory, got %q", r.Reason)
	}
	if !strings.Contains(r.NextAction, "recalibrate") {
		t.Fatalf("over-claim present should steer to recalibrate, got %q", r.NextAction)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("recorded fold gate should pass even with over-claim, got %d", code)
	}
}

func TestFoldNextActionHarvestsUnderClaim(t *testing.T) {
	eps := []Episode{
		Score("s", pred("posture_accuracy", 1.0), obs(0.977, true), DefaultCalibBand()), // calibrated
		Score("s", pred("warm_hit", 0.0), obs(0.30, true), DefaultCalibBand()),          // under-claim only
	}
	r := Fold(eps, FoldOpts{})
	if !strings.Contains(r.NextAction, "harvest") {
		t.Fatalf("under-claim without over-claim should steer to harvest, got %q", r.NextAction)
	}
}

func TestRenderContainsEpisodeAndTrend(t *testing.T) {
	eps := []Episode{Score("resume-posture", pred("cold_write_share", 0.85), obs(0.68, true), DefaultCalibBand())}
	r := Fold(eps, FoldOpts{Date: "2026-06-26", Commit: "abc123def456789"})
	r.Trend = &Trend{Direction: "improved", Summary: "calibration improved -0.050 (0.300->0.250) vs 2026-06-25; 2/3 episodes calibrated"}
	out := Render(r)
	for _, want := range []string{"dojo report", "cold_write_share", "OVER_CLAIM", "trend:", "abc123def456"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
}

type fakeLever struct {
	name string
	ins  []ScoredInput
	err  error
}

func (f fakeLever) Name() string { return f.name }
func (f fakeLever) Episodes(Scenario) ([]ScoredInput, error) {
	return f.ins, f.err
}

func TestRunScoresAndCollectsErrors(t *testing.T) {
	good := fakeLever{name: "good", ins: []ScoredInput{
		{Prediction: pred("m1", 1.0), Outcome: obs(0.9, true)},
		{Prediction: pred("m2", 1.0), Outcome: obs(0.5, true)},
	}}
	broken := fakeLever{name: "broken", err: errString("boom")}
	scen := []Scenario{{Name: "corpus-a", Mode: "offline", Corpus: "/tmp/x"}}

	eps, errs := Run(scen, []Lever{good, broken}, DefaultCalibBand())
	if len(eps) != 2 {
		t.Fatalf("want 2 episodes from the good lever, got %d", len(eps))
	}
	if eps[0].Scenario != "corpus-a" {
		t.Fatalf("episode should carry the scenario name, got %q", eps[0].Scenario)
	}
	if len(errs) != 1 || errs[0].Lever != "broken" {
		t.Fatalf("broken lever should yield one RunError, got %+v", errs)
	}
}

type errString string

func (e errString) Error() string { return string(e) }

func TestLedgerRoundTripAndTrend(t *testing.T) {
	r0 := Fold([]Episode{Score("s", pred("a", 1.0), obs(0.6, true), DefaultCalibBand())}, FoldOpts{Date: "2026-06-25", Commit: "c0", GeneratedAt: "2026-06-25T00:00:00Z"})
	row0 := RowFromReport(r0)
	line0, err := AppendLedgerLine(row0)
	if err != nil {
		t.Fatalf("append line: %v", err)
	}

	r1 := Fold([]Episode{Score("s", pred("a", 1.0), obs(0.95, true), DefaultCalibBand())}, FoldOpts{Date: "2026-06-26", Commit: "c1", GeneratedAt: "2026-06-26T00:00:00Z"})
	row1 := RowFromReport(r1)

	prior := ParseLedger(line0 + "\n\ngarbage-not-json\n")
	if len(prior) != 1 {
		t.Fatalf("tolerant parse should keep 1 row and skip junk, got %d", len(prior))
	}
	tr := TrendVsLast(row1, prior)
	if tr.Direction != "improved" {
		t.Fatalf("calib-err fell from %.3f to %.3f — want improved, got %s", row0.MeanCalibErr, row1.MeanCalibErr, tr.Direction)
	}
	if tr.CalibErrDelta >= 0 {
		t.Fatalf("improved trend should have negative delta, got %.4f", tr.CalibErrDelta)
	}
}

func TestParseLedgerRejectsForeignSchema(t *testing.T) {
	// A foreign JSONL line with a "date" field unmarshals cleanly into LedgerRow
	// (extra keys dropped, Date survives) and would pollute the trend. ParseLedger
	// must keep only rows stamped with our LedgerSchema. A line with no schema at
	// all (a hand-edit, or a pre-schema row) is likewise rejected.
	good, err := AppendLedgerLine(RowFromReport(Fold(
		[]Episode{Score("s", pred("a", 1.0), obs(0.6, true), DefaultCalibBand())},
		FoldOpts{Date: "2026-06-27", Commit: "c0", GeneratedAt: "2026-06-27T00:00:00Z"},
	)))
	if err != nil {
		t.Fatalf("append line: %v", err)
	}
	foreign := `{"schema":"some-other-tool/3","date":"2026-06-27","mean_calib_err":99.0}`
	noSchema := `{"date":"2026-06-27","mean_calib_err":42.0}`

	rows := ParseLedger(strings.Join([]string{foreign, good, noSchema}, "\n") + "\n")
	if len(rows) != 1 {
		t.Fatalf("foreign-schema and no-schema lines must be rejected; want 1 row, got %d (%+v)", len(rows), rows)
	}
	if rows[0].Schema != LedgerSchema {
		t.Fatalf("the surviving row must be our schema, got %q", rows[0].Schema)
	}
	if rows[0].Commit != "c0" {
		t.Fatalf("the surviving row must be the committed one, got commit %q", rows[0].Commit)
	}
}

func TestTrendSubDisplayDeltaIsFlat(t *testing.T) {
	// Dogfood regression: a corpus drift of +0.00045 mean-calib-err once printed
	// "calibration regressed +0.000 (0.341->0.341)" — a direction the operator
	// cannot see in the rendered numbers. A delta finer than the %.3f display
	// precision must read "flat", and the summary must not contradict itself.
	prev := LedgerRow{Date: "2026-06-27", Commit: "6a2a325e", GeneratedAt: "2026-06-27T05:57:56Z", MeanCalibErr: 0.3409325648392343, Calibrated: 2, Measured: 3}
	cur := LedgerRow{Date: "2026-06-27", Commit: "d6188182", GeneratedAt: "2026-06-27T06:10:00Z", MeanCalibErr: 0.3413873609606471, Calibrated: 2, Measured: 3}
	tr := TrendVsLast(cur, []LedgerRow{prev})
	if tr.Direction != "flat" {
		t.Fatalf("delta %.5f rounds to +0.000 at display precision — want flat, got %q (%s)", tr.CalibErrDelta, tr.Direction, tr.Summary)
	}
	if strings.Contains(tr.Summary, "regressed") || strings.Contains(tr.Summary, "improved") {
		t.Fatalf("a sub-display delta summary must not claim a direction it cannot show: %q", tr.Summary)
	}
}

func TestTrendDisplayableRegressionStillReported(t *testing.T) {
	// The fix must not blunt a real, visible regression: a +0.05 rise still reads
	// "regressed" (it renders as +0.050, not +0.000).
	prev := LedgerRow{Date: "2026-06-26", MeanCalibErr: 0.300, Calibrated: 2, Measured: 3}
	cur := LedgerRow{Date: "2026-06-27", MeanCalibErr: 0.350, Calibrated: 2, Measured: 3}
	tr := TrendVsLast(cur, []LedgerRow{prev})
	if tr.Direction != "regressed" {
		t.Fatalf("a visible +0.050 rise must stay regressed, got %q (%s)", tr.Direction, tr.Summary)
	}
}

func TestTrendNewWhenNoPrior(t *testing.T) {
	row := RowFromReport(Fold([]Episode{Score("s", pred("a", 1.0), obs(0.9, true), DefaultCalibBand())}, FoldOpts{Date: "2026-06-26"}))
	tr := TrendVsLast(row, nil)
	if tr.Direction != "new" {
		t.Fatalf("no prior should be 'new', got %s", tr.Direction)
	}
}

func TestTrendIdempotentReappend(t *testing.T) {
	row := RowFromReport(Fold([]Episode{Score("s", pred("a", 1.0), obs(0.9, true), DefaultCalibBand())}, FoldOpts{Date: "2026-06-26", GeneratedAt: "2026-06-26T00:00:00Z"}))
	// the same row already on the ledger (same generated_at) is excluded.
	tr := TrendVsLast(row, []LedgerRow{row})
	if tr.Direction != "new" {
		t.Fatalf("re-appending the same tick should not trend against itself, got %s", tr.Direction)
	}
}

func TestTrendBackfillIgnoresFutureRow(t *testing.T) {
	// Bug #4: a --date backfill must trend against the prior row, never a
	// chronologically-later (future) row already on the ledger. A ledger holding a
	// 2026-06-27 row plus a 2026-06-20 row: a freshly-built 2026-06-25 backfill tick
	// must trend against 06-20, not the 06-27 future row.
	future := LedgerRow{Date: "2026-06-27", Commit: "cFut", GeneratedAt: "2026-06-27T00:00:00Z", MeanCalibErr: 0.10, Calibrated: 3, Measured: 3}
	past := LedgerRow{Date: "2026-06-20", Commit: "cPast", GeneratedAt: "2026-06-20T00:00:00Z", MeanCalibErr: 0.50, Calibrated: 1, Measured: 3}
	cur := LedgerRow{Date: "2026-06-25", Commit: "cCur", GeneratedAt: "2026-06-25T00:00:00Z", MeanCalibErr: 0.40, Calibrated: 2, Measured: 3}
	tr := TrendVsLast(cur, []LedgerRow{future, past})
	if tr.PrevDate != "2026-06-20" {
		t.Fatalf("backfill must trend against the prior 06-20 row, not a future row; got prev=%s (%s)", tr.PrevDate, tr.Summary)
	}
	if tr.CalibErrFrom != 0.50 {
		t.Fatalf("from-value should be the 06-20 row's 0.50, got %.3f (trended against the wrong row)", tr.CalibErrFrom)
	}
	// 0.50 -> 0.40 is a fall, so the direction is improved (against the PAST row).
	if tr.Direction != "improved" {
		t.Fatalf("0.50->0.40 vs the prior row is improved, got %q (%s)", tr.Direction, tr.Summary)
	}
}

func TestTrendNewWhenOnlyFutureRows(t *testing.T) {
	// Bug #4: if every other row is dated AFTER `row`, there is no valid prior, so
	// the tick is "new" rather than trended against the future. Previously the max
	// row (a future row) leaked in and produced a from-the-future direction.
	future := LedgerRow{Date: "2026-06-30", Commit: "cF", GeneratedAt: "2026-06-30T00:00:00Z", MeanCalibErr: 0.10, Calibrated: 3, Measured: 3}
	cur := LedgerRow{Date: "2026-06-25", Commit: "cC", GeneratedAt: "2026-06-25T00:00:00Z", MeanCalibErr: 0.40, Calibrated: 2, Measured: 3}
	tr := TrendVsLast(cur, []LedgerRow{future})
	if tr.Direction != "new" {
		t.Fatalf("no row precedes the backfill tick, so it must be 'new', got %q (prev=%s)", tr.Direction, tr.PrevDate)
	}
}

func TestTrendSameDateEarlierGeneratedAtStillPrior(t *testing.T) {
	// Bug #4 must not blunt the same-date case: a same-date row with an EARLIER
	// generated_at is still a valid prior (the within-day re-run case).
	earlier := LedgerRow{Date: "2026-06-27", Commit: "cE", GeneratedAt: "2026-06-27T05:00:00Z", MeanCalibErr: 0.30, Calibrated: 2, Measured: 3}
	cur := LedgerRow{Date: "2026-06-27", Commit: "cC", GeneratedAt: "2026-06-27T06:00:00Z", MeanCalibErr: 0.20, Calibrated: 3, Measured: 3}
	tr := TrendVsLast(cur, []LedgerRow{earlier})
	if tr.PrevCommit != "cE" {
		t.Fatalf("a same-date, earlier generated_at row is a valid prior; got prev=%q (%s)", tr.PrevCommit, tr.Summary)
	}
	if tr.Direction != "improved" {
		t.Fatalf("0.30->0.20 within the same day is improved, got %q", tr.Direction)
	}
}

func TestWithGate(t *testing.T) {
	r := Fold(nil, FoldOpts{})
	code, msg := CheckGate(r)
	g := r.WithGate(code, msg)
	if g.GateExit == nil || *g.GateExit != 1 || g.OK {
		t.Fatalf("WithGate should reconcile to the failing gate, got exit=%v ok=%v", g.GateExit, g.OK)
	}
}

// predDir builds a Prediction with an explicit metric direction (bug #1).
func predDir(metric string, claimed float64, lowerIsBetter bool) Prediction {
	return Prediction{Lever: "lvr", Metric: metric, Claimed: claimed, Unit: "fraction", Basis: "test", LowerIsBetter: lowerIsBetter}
}

func TestScoreLowerIsBetterRealizedWorseIsOverClaim(t *testing.T) {
	// Bug #1: false_warm_rate is LOWER-is-better. The theory claims a 10% false-warm
	// rate (0.10); billed reality came in at 18% (0.18) — WORSE than promised.
	// Realized ABOVE the claim is the over-claim side for this direction, even
	// though the raw residual is positive (the old sign-only rule mislabeled it
	// UNDER_CLAIM and steered the operator to "harvest" a regression).
	e := Score("s", predDir("false_warm_rate", 0.10, true), obs(0.18, true), DefaultCalibBand())
	if e.Verdict != VerdictOverClaim {
		t.Fatalf("lower-is-better realized above claim should be OVER_CLAIM, got %s (resid %.4f)", e.Verdict, e.Residual)
	}
	if e.Residual <= 0 {
		t.Fatalf("realized 0.18 vs claim 0.10 should give a positive residual, got %.4f", e.Residual)
	}
	// calib_err = |0.18-0.10|/0.10 = 0.80 -> grade F (direction does not change magnitude)
	if e.CalibErr < 0.79 || e.CalibErr > 0.81 {
		t.Fatalf("calib_err should be ~0.80, got %.4f", e.CalibErr)
	}
	if e.Grade != "F" {
		t.Fatalf("0.80 calib-err should grade F, got %s", e.Grade)
	}
}

func TestScoreLowerIsBetterRealizedBetterIsUnderClaim(t *testing.T) {
	// Bug #1: cold_write_share scored as LOWER-is-better. The theory claims 50%
	// (0.50); reality came in at 20% (0.20) — BETTER than promised. Realized below
	// the claim is the under-claim (free saving) side for this direction.
	e := Score("s", predDir("cold_write_share", 0.50, true), obs(0.20, true), DefaultCalibBand())
	if e.Verdict != VerdictUnderClaim {
		t.Fatalf("lower-is-better realized below claim should be UNDER_CLAIM, got %s (resid %.4f)", e.Verdict, e.Residual)
	}
	if e.Residual >= 0 {
		t.Fatalf("realized 0.20 vs claim 0.50 should give a negative residual, got %.4f", e.Residual)
	}
	// calib_err = |0.20-0.50|/0.50 = 0.60 -> grade D
	if e.CalibErr < 0.59 || e.CalibErr > 0.61 {
		t.Fatalf("calib_err should be ~0.60, got %.4f", e.CalibErr)
	}
	if e.Grade != "D" {
		t.Fatalf("0.60 calib-err should grade D, got %s", e.Grade)
	}
}

func TestScoreDirectionFlipsVerdictNotMagnitude(t *testing.T) {
	// Bug #1: the same claim and realized value, scored once higher-is-better and
	// once lower-is-better, must produce OPPOSITE verdicts but the SAME calibration
	// error — LowerIsBetter is the only lever that moves the verdict and it does
	// not touch the magnitude. claim 0.20, realized 0.50: residual +0.30.
	hi := Score("s", predDir("m", 0.20, false), obs(0.50, true), DefaultCalibBand())
	lo := Score("s", predDir("m", 0.20, true), obs(0.50, true), DefaultCalibBand())
	if hi.Verdict != VerdictUnderClaim {
		t.Fatalf("higher-is-better realized above claim should be UNDER_CLAIM, got %s", hi.Verdict)
	}
	if lo.Verdict != VerdictOverClaim {
		t.Fatalf("lower-is-better realized above claim should be OVER_CLAIM, got %s", lo.Verdict)
	}
	if math.Abs(hi.CalibErr-lo.CalibErr) > 1e-9 {
		t.Fatalf("direction must not change the magnitude: hi %.6f vs lo %.6f", hi.CalibErr, lo.CalibErr)
	}
}

func TestCalibErrZeroClaimUsesAbsoluteResidual(t *testing.T) {
	// Bug #2: a claim of "nothing" must score the ABSOLUTE residual, not a
	// magnitude-blind 1.0. Two zero-claim metrics whose realities differ must score
	// DIFFERENTLY (the old divide-by-realized form gave 1.0 for both). realized > 0
	// above the calibration band so reality clearly exceeds a zero claim ->
	// UNDER_CLAIM; calib_err is the absolute residual (== realized here).
	for _, realized := range []float64{0.30, 0.68, 0.99} {
		e := Score("s", pred("zero_claim", 0.0), obs(realized, true), DefaultCalibBand())
		if e.CalibErr < realized-1e-9 || e.CalibErr > realized+1e-9 {
			t.Fatalf("claim 0.0 realized %.2f: calib_err should be the absolute residual %.4f, got %.4f",
				realized, realized, e.CalibErr)
		}
		if e.Verdict != VerdictUnderClaim {
			t.Fatalf("claim 0.0 realized %.2f exceeds the claim, want UNDER_CLAIM, got %s", realized, e.Verdict)
		}
	}
}

func TestCalibErrZeroClaimNotConstantOne(t *testing.T) {
	// Bug #2 core defect: the old form scored EVERY zero-claim metric exactly 1.0.
	lo := Score("s", pred("z", 0.0), obs(0.10, true), DefaultCalibBand())
	hi := Score("s", pred("z", 0.0), obs(0.90, true), DefaultCalibBand())
	if lo.CalibErr == hi.CalibErr {
		t.Fatalf("zero-claim calib_err must track realized magnitude, got identical %.4f for 0.10 and 0.90", lo.CalibErr)
	}
	if lo.CalibErr >= hi.CalibErr {
		t.Fatalf("a larger zero-claim miss must score worse: 0.10->%.4f should be < 0.90->%.4f", lo.CalibErr, hi.CalibErr)
	}
}

func TestCalibErrExactZeroNotBetterThanNearZero(t *testing.T) {
	// Bug #2: the backwards discontinuity the fix removes — previously claim==0
	// scored 1.0 while a refuted near-zero claim took the ratio path and capped at
	// 2.0, so the EXACT-zero claim scored BETTER than the near-zero one. Exact-zero
	// must now be the smallest (best) score in its neighborhood, never the largest.
	exact := calibErr(0.0, 0.30)
	near := calibErr(1e-8, 0.30) // just above zeroClaimEps -> ratio path, refuted, caps
	if exact > near {
		t.Fatalf("exact-zero (%.4f) must not score worse than a refuted near-zero claim (%.4f)", exact, near)
	}
}

func TestCalibErrZeroNeighborhoodContinuous(t *testing.T) {
	// Bug #2: across claimed==0 and the sub-zeroClaimEps neighborhood the error is
	// the absolute residual, flat in claimed and continuous across zero (no jump
	// from the old divide-by-realized step at exactly claimed==0).
	base := calibErr(0.0, 0.30)
	for _, c := range []float64{1e-12, 1e-10, zeroClaimEps} {
		got := calibErr(c, 0.30)
		if math.Abs(got-base) > 1e-6 {
			t.Fatalf("zero-neighborhood not continuous: claim %.0e gave %.6f, claim 0 gave %.6f", c, got, base)
		}
	}
}

func TestCalibErrZeroClaimStillCaps(t *testing.T) {
	// Bug #2: a zero claim refuted by an out-of-[0,1] reality still caps at
	// MaxCalibErr (the absolute residual is bounded by the same cap as the ratio).
	if e := calibErr(0.0, 100.0); e != MaxCalibErr {
		t.Fatalf("zero claim vs realized 100.0 should cap at %.1f, got %.4f", MaxCalibErr, e)
	}
}
