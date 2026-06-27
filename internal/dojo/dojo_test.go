package dojo

import (
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

func TestWithGate(t *testing.T) {
	r := Fold(nil, FoldOpts{})
	code, msg := CheckGate(r)
	g := r.WithGate(code, msg)
	if g.GateExit == nil || *g.GateExit != 1 || g.OK {
		t.Fatalf("WithGate should reconcile to the failing gate, got exit=%v ok=%v", g.GateExit, g.OK)
	}
}
