package frontierswe

import (
	"strings"
	"testing"
)

func traceOf(wallSec float64, turns int) TTSTrace {
	return TTSTrace{Schema: "fak.frontierswe.tts-trace.v1", Turns: turns, TotalWallSec: wallSec}
}

func gradedTrial(id string, correctness, score, wallSec float64, turns int, mocked bool) GradedTrial {
	return GradedTrial{
		Score:  TrialScore{ID: id, Task: "git-to-zig", Correctness: correctness, Score: score},
		Trace:  traceOf(wallSec, turns),
		Mocked: mocked,
	}
}

// C14: a solved measured trial is timed at its full trajectory; provenance follows
// the mocked flag; a censored trial reports no time-to-solution.
func TestTTSMetricReachedAndProvenance(t *testing.T) {
	solvedMeasured := TTSMetricOf("git-to-zig", "t1", 1.0, traceOf(1200, 30), false)
	if !solvedMeasured.Reached {
		t.Fatalf("correctness 1.0 must be Reached")
	}
	if !floatsEqual(solvedMeasured.WallSecToCorrect, 1200) || solvedMeasured.TurnsToCorrect != 30 {
		t.Fatalf("solved trial timed at full trajectory, got wall=%v turns=%d", solvedMeasured.WallSecToCorrect, solvedMeasured.TurnsToCorrect)
	}
	if solvedMeasured.Provenance != ProvenanceMeasured {
		t.Fatalf("non-mocked trial must be measured, got %s", solvedMeasured.Provenance)
	}

	solvedMocked := TTSMetricOf("git-to-zig", "t2", 1.0, traceOf(60, 5), true)
	if solvedMocked.Provenance != ProvenanceProjected {
		t.Fatalf("mocked trial must be projected, got %s", solvedMocked.Provenance)
	}

	censored := TTSMetricOf("git-to-zig", "t3", 0.5, traceOf(999, 40), false)
	if censored.Reached {
		t.Fatalf("correctness 0.5 must be censored, not Reached")
	}
	if censored.WallSecToCorrect != 0 || censored.TurnsToCorrect != 0 {
		t.Fatalf("censored trial contributes no time, got wall=%v turns=%d", censored.WallSecToCorrect, censored.TurnsToCorrect)
	}
}

// C12: parity is evaluated first — a fak arm that regressed raw's score distribution
// stops the compare at PARITY_FAILED with no ratio, even though fak was "faster".
func TestCompareParityFailsFirst(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 1000, 40, false)}
	fak := []GradedTrial{gradedTrial("f1", 0.0, 0.0, 10, 1, false)} // faster but scored nothing
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictParityFailed {
		t.Fatalf("expected PARITY_FAILED, got %s", rep.Verdict)
	}
	if rep.TTSRatio != nil {
		t.Fatalf("a failed parity gate must emit no TTS ratio, got %v", *rep.TTSRatio)
	}
}

// C12: parity can pass while neither arm solved a trial (both partial-correct and
// equal). There is no time-to-solution to compare, so the verdict is GATED.
func TestCompareGatedWhenNoSolvedTrial(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 0.5, 0.3, 1000, 40, false)}
	fak := []GradedTrial{gradedTrial("f1", 0.5, 0.3, 200, 8, false)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if !rep.Parity.Passed {
		t.Fatalf("equal partial scores should pass parity; failures=%v", rep.Parity.Failures)
	}
	if rep.Verdict != VerdictGated {
		t.Fatalf("expected GATED, got %s", rep.Verdict)
	}
	if rep.TTSRatio != nil {
		t.Fatalf("GATED must emit no ratio")
	}
}

// C12: parity green, both arms solved on real trajectories, fak faster ⇒ a MEASURED
// win with a ratio below 1.
func TestCompareMeasuredWin(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 1000, 40, false)}
	fak := []GradedTrial{gradedTrial("f1", 1.0, 1.0, 250, 10, false)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictMeasuredWin {
		t.Fatalf("expected MEASURED_WIN, got %s (headline=%q)", rep.Verdict, rep.Headline)
	}
	if rep.TTSRatio == nil || !floatsEqual(*rep.TTSRatio, 0.25) {
		t.Fatalf("expected ratio 0.25, got %v", rep.TTSRatio)
	}
	if rep.Provenance != ProvenanceMeasured {
		t.Fatalf("expected measured provenance, got %s", rep.Provenance)
	}
}

// C12: identical arithmetic on a MOCKED run must never read as a measured win — the
// verdict and provenance both carry the projected label.
func TestCompareProjectedWinIsNotMeasured(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 1000, 40, true)}
	fak := []GradedTrial{gradedTrial("f1", 1.0, 1.0, 250, 10, true)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictProjectedWin {
		t.Fatalf("expected PROJECTED_WIN, got %s", rep.Verdict)
	}
	if rep.Provenance != ProvenanceProjected {
		t.Fatalf("a mocked comparison must be projected, got %s", rep.Provenance)
	}
	if !strings.Contains(rep.Headline, "PROJECTED") {
		t.Fatalf("projected headline must say so, got %q", rep.Headline)
	}
}

// C12: one measured arm and one mocked arm ⇒ the ratio is conservatively projected,
// never a measured win.
func TestCompareMixedProvenanceIsProjected(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 1000, 40, false)}
	fak := []GradedTrial{gradedTrial("f1", 1.0, 1.0, 250, 10, true)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Provenance != ProvenanceProjected {
		t.Fatalf("mixed measured+mocked must fold to projected, got %s", rep.Provenance)
	}
	if rep.Verdict != VerdictProjectedWin {
		t.Fatalf("expected PROJECTED_WIN under mixed provenance, got %s", rep.Verdict)
	}
}

// C12: parity green but fak no faster ⇒ no win.
func TestCompareNoWin(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 500, 20, false)}
	fak := []GradedTrial{gradedTrial("f1", 1.0, 1.0, 700, 28, false)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictMeasuredNoWin {
		t.Fatalf("expected MEASURED_NO_WIN, got %s", rep.Verdict)
	}
	if rep.TTSRatio == nil || *rep.TTSRatio <= 1.0 {
		t.Fatalf("expected ratio > 1, got %v", rep.TTSRatio)
	}
}

// traceFull carries the C8 realized reuse rate and the [agent] budget the C4 floor
// is projected from, so the reuse/floor surfacing (#1718) can be exercised offline.
func traceFull(wallSec float64, turns int, budgetSec int64, reuse float64) TTSTrace {
	return TTSTrace{
		Schema: "fak.frontierswe.tts-trace.v1", Turns: turns, TotalWallSec: wallSec, BudgetSec: budgetSec,
		CacheSeries: CacheWitnessSeries{RealizedReuseRate: reuse},
	}
}

func gradedTrialFull(id string, correctness, score, wallSec float64, turns int, budgetSec int64, reuse float64, mocked bool) GradedTrial {
	return GradedTrial{
		Score:  TrialScore{ID: id, Task: "git-to-zig", Correctness: correctness, Score: score},
		Trace:  traceFull(wallSec, turns, budgetSec, reuse),
		Mocked: mocked,
	}
}

// #1718: the compare surfaces the C8 realized reuse rate per arm and the C4
// deterministic floor the measured ratio is checked against. A measured ratio that
// sits ABOVE the floor is honest — no over-claim.
func TestCompareSurfacesReuseAndFloor(t *testing.T) {
	raw := []GradedTrial{gradedTrialFull("r1", 1.0, 1.0, 1000, 40, 3600, 0.80, false)}
	fak := []GradedTrial{gradedTrialFull("f1", 1.0, 1.0, 250, 10, 3600, 0.90, false)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if !floatsEqual(rep.Raw.MeanReuseRate, 0.80) || !floatsEqual(rep.Fak.MeanReuseRate, 0.90) {
		t.Fatalf("realized reuse (C8) must surface per arm, got raw=%v fak=%v", rep.Raw.MeanReuseRate, rep.Fak.MeanReuseRate)
	}
	if rep.FloorRatio == nil {
		t.Fatalf("a budgeted arm must project a C4 floor")
	}
	if *rep.FloorRatio <= 0 || *rep.FloorRatio >= 1 {
		t.Fatalf("C4 floor must be a fraction in (0,1), got %v", *rep.FloorRatio)
	}
	if rep.OverClaim {
		t.Fatalf("measured ratio 0.25 sits above the floor %v; over-claim must be false", *rep.FloorRatio)
	}
}

// #1718: a MEASURED ratio that dips below the deterministic C4 floor is physically
// suspicious and must be flagged as an over-claim, not banked as a bigger win.
func TestCompareOverClaimBelowFloor(t *testing.T) {
	raw := []GradedTrial{gradedTrialFull("r1", 1.0, 1.0, 1000, 40, 3600, 0.90, false)}
	fak := []GradedTrial{gradedTrialFull("f1", 1.0, 1.0, 5, 1, 3600, 0.90, false)} // absurdly fast: below the floor
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictMeasuredWin {
		t.Fatalf("expected MEASURED_WIN, got %s", rep.Verdict)
	}
	if rep.FloorRatio == nil || !rep.OverClaim {
		t.Fatalf("a measured ratio below the C4 floor must flag over-claim; floor=%v ratio=%v", rep.FloorRatio, rep.TTSRatio)
	}
}

// #1718: a PROJECTED (mocked) ratio IS the floor's own family, so it can never
// over-claim against the projection even when the arithmetic dips below it.
func TestCompareProjectedNeverOverClaim(t *testing.T) {
	raw := []GradedTrial{gradedTrialFull("r1", 1.0, 1.0, 1000, 40, 3600, 0.90, true)}
	fak := []GradedTrial{gradedTrialFull("f1", 1.0, 1.0, 5, 1, 3600, 0.90, true)}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Provenance != ProvenanceProjected {
		t.Fatalf("mocked ⇒ projected, got %s", rep.Provenance)
	}
	if rep.OverClaim {
		t.Fatalf("a projected floor is not a measurement; over-claim must not fire")
	}
}

// #1718: the markdown surfaces the per-arm score, the realized reuse rate, and the
// C4 floor beside the measured ratio.
func TestRenderCompareMarkdownSurfacesReuseScoreFloor(t *testing.T) {
	raw := []GradedTrial{gradedTrialFull("r1", 1.0, 1.0, 1000, 40, 3600, 0.80, false)}
	fak := []GradedTrial{gradedTrialFull("f1", 1.0, 1.0, 250, 10, 3600, 0.90, false)}
	md := RenderCompareMarkdown(CompareArms("git-to-zig", raw, fak, 0))
	for _, want := range []string{"avg score", "realized reuse", "C4 floor"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown must surface %q; got:\n%s", want, md)
		}
	}
}

// The markdown renders in the honest order: verdict, parity, then the ratio.
func TestRenderCompareMarkdownOrder(t *testing.T) {
	raw := []GradedTrial{gradedTrial("r1", 1.0, 1.0, 1000, 40, false)}
	fak := []GradedTrial{gradedTrial("f1", 1.0, 1.0, 250, 10, false)}
	md := RenderCompareMarkdown(CompareArms("git-to-zig", raw, fak, 0))
	parityIdx := strings.Index(md, "Parity gate")
	ratioIdx := strings.Index(md, "TTS ratio")
	if parityIdx < 0 || ratioIdx < 0 || parityIdx > ratioIdx {
		t.Fatalf("markdown must present parity before the ratio; parityIdx=%d ratioIdx=%d", parityIdx, ratioIdx)
	}
}
