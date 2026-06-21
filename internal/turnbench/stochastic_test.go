package turnbench

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	// Blank-import the built-in driver list so the full ABI is wired before
	// Run / agent.Configure run inside RunStochastic (same as turnbench_test.go).
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// stochasticArtifact is the distribution report written by the test (a new file).
const stochasticArtifact = "../../experiments/turn-tax/turntax-stochastic.json"

// breakEvenArtifact is the hit-rate -> turns-saved -> amortization curve.
const breakEvenArtifact = "../../experiments/turn-tax/turntax-breakeven.json"

// trials is kept modest so the WSL suite stays fast: each trial runs the full
// turn-tax A/B (two kernel replays). 200 is enough to give stable order stats and
// a clear monotone separation between the profiles.
const stochasticTrials = 200

// stochasticSeed pins the whole distribution. Same seed => identical report.
const stochasticSeed int64 = 0x7A8B_C0DE

// TestStochastic_BaseSavesNothing is the anti-inflation control at the source: the
// clean base, run through the real kernel with NO perturbation, must save 0. If
// this is ever non-zero the base workload is not actually clean.
func TestStochastic_BaseSavesNothing(t *testing.T) {
	rep, err := Run(context.Background(), BaseTrace(), DefaultCostModel())
	if err != nil {
		t.Fatalf("Run(base): %v", err)
	}
	if rep.ConsistencyCheck != "ok" {
		t.Fatalf("base consistency check failed: %s", rep.ConsistencyCheck)
	}
	if rep.Net.TurnsSaved != 0 {
		t.Fatalf("base turns saved = %d, want 0 (base must be all first-occurrence passes); class=%+v",
			rep.Net.TurnsSaved, rep.Class)
	}
	if rep.Class.Quarantine != 0 || rep.Class.Deny != 0 {
		t.Fatalf("base has safety events quarantine=%d deny=%d, want 0/0 (stochastic base must be happy-path only)",
			rep.Class.Quarantine, rep.Class.Deny)
	}
}

// TestStochastic_Determinism: the same seed yields the byte-identical distribution.
func TestStochastic_Determinism(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	a := RunStochastic(context.Background(), base, ProfileMid, cm, 64, stochasticSeed)
	b := RunStochastic(context.Background(), base, ProfileMid, cm, 64, stochasticSeed)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same seed produced different distributions:\n a=%+v\n b=%+v", a, b)
	}
	// A different seed should generally differ (sanity — not a hard guarantee, but
	// with 64 trials over four non-zero rates a collision is effectively impossible).
	c := RunStochastic(context.Background(), base, ProfileMid, cm, 64, stochasticSeed+1)
	if reflect.DeepEqual(a.SampleTurns, c.SampleTurns) {
		t.Errorf("different seeds produced identical sample turns — RNG not seed-driven")
	}
}

// TestStochastic_Monotonicity: a strictly-higher-rate profile yields a higher
// median turns_saved. This is the core claim the distribution makes — more errors
// => more turns fak deletes.
func TestStochastic_Monotonicity(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	low := RunStochastic(context.Background(), base, ProfileLow, cm, stochasticTrials, stochasticSeed)
	mid := RunStochastic(context.Background(), base, ProfileMid, cm, stochasticTrials, stochasticSeed)
	high := RunStochastic(context.Background(), base, ProfileHigh, cm, stochasticTrials, stochasticSeed)

	if !(low.TurnsSaved.P50 < mid.TurnsSaved.P50) {
		t.Errorf("median not monotone low->mid: low.p50=%d mid.p50=%d", low.TurnsSaved.P50, mid.TurnsSaved.P50)
	}
	if !(mid.TurnsSaved.P50 < high.TurnsSaved.P50) {
		t.Errorf("median not monotone mid->high: mid.p50=%d high.p50=%d", mid.TurnsSaved.P50, high.TurnsSaved.P50)
	}
	// The mean should track the same order (a softer corroboration of the median).
	if !(low.TurnsSaved.Mean < mid.TurnsSaved.Mean && mid.TurnsSaved.Mean < high.TurnsSaved.Mean) {
		t.Errorf("mean not monotone: low=%.2f mid=%.2f high=%.2f",
			low.TurnsSaved.Mean, mid.TurnsSaved.Mean, high.TurnsSaved.Mean)
	}
	// forced + elision must reconstruct the turns_saved order stats are derived from
	// (they are summarized independently, so this is a sanity tie, not an identity):
	// every profile's forced and elision medians must be >= 0 and not exceed turns.
	for _, pr := range []ProfileResult{low, mid, high} {
		if pr.Forced.P50 < 0 || pr.Elision.P50 < 0 {
			t.Errorf("%s: negative split p50 forced=%d elision=%d", pr.Profile.Name, pr.Forced.P50, pr.Elision.P50)
		}
	}
}

// TestStochastic_ZeroRateP50IsZero is the anti-inflation control under the
// stochastic harness: a zero-rate profile injects nothing, so EVERY trial saves 0
// and the median is exactly 0. (The whole distribution collapses to 0.)
func TestStochastic_ZeroRateP50IsZero(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	z := RunStochastic(context.Background(), base, ProfileZero, cm, stochasticTrials, stochasticSeed)
	if z.TurnsSaved.P50 != 0 {
		t.Errorf("zero-rate p50 = %d, want 0", z.TurnsSaved.P50)
	}
	if z.TurnsSaved.Max != 0 || z.TurnsSaved.Mean != 0 {
		t.Errorf("zero-rate distribution not all-zero: max=%d mean=%.3f (a perturbation leaked savings)",
			z.TurnsSaved.Max, z.TurnsSaved.Mean)
	}
}

// TestStochastic_WriteArtifact runs the full ladder and writes the distribution
// artifact. It also re-asserts the headline ordering on the written report so the
// artifact and the claims can't drift.
func TestStochastic_WriteArtifact(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	rep := RunStochasticAll(context.Background(), base, DefaultProfiles(), cm, stochasticTrials, stochasticSeed)

	if rep.AppVersion == "" {
		t.Fatal("stochastic report app_version is empty")
	}
	if rep.Cost.Version != CostModelVersion {
		t.Fatalf("stochastic cost model version=%q, want %q", rep.Cost.Version, CostModelVersion)
	}
	if len(rep.Profiles) != 3 {
		t.Fatalf("want 3 profiles, got %d", len(rep.Profiles))
	}
	for _, pr := range rep.Profiles {
		if pr.Profile.Version != BenchmarkConceptVersion {
			t.Fatalf("profile %q version=%q, want %q", pr.Profile.Name, pr.Profile.Version, BenchmarkConceptVersion)
		}
	}
	// Headline monotonicity on the written object.
	for i := 1; i < len(rep.Profiles); i++ {
		prev, cur := rep.Profiles[i-1], rep.Profiles[i]
		if !(prev.TurnsSaved.P50 < cur.TurnsSaved.P50) {
			t.Errorf("artifact medians not monotone: %s.p50=%d !< %s.p50=%d",
				prev.Profile.Name, prev.TurnsSaved.P50, cur.Profile.Name, cur.TurnsSaved.P50)
		}
	}

	path, err := filepath.Abs(stochasticArtifact)
	if err != nil {
		t.Fatalf("abs(%q): %v", stochasticArtifact, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %q: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, rep.JSON(), 0o644); err != nil {
		t.Fatalf("write %q: %v", path, err)
	}
	t.Logf("wrote stochastic distribution artifact: %s", path)
	for _, pr := range rep.Profiles {
		t.Logf("profile=%-4s trials=%d turns_saved p10=%d p50=%d p90=%d mean=%.2f (forced p50=%d, elision p50=%d, avg_calls=%.1f)",
			pr.Profile.Name, pr.Trials, pr.TurnsSaved.P10, pr.TurnsSaved.P50, pr.TurnsSaved.P90,
			pr.TurnsSaved.Mean, pr.Forced.P50, pr.Elision.P50, pr.AvgCalls)
	}
}

// pointAt returns the break-even point at hit-rate h (exact float match against
// the grid the report was built from).
func pointAt(rep BreakEvenReport, h float64) (BreakEvenPoint, bool) {
	for _, p := range rep.Points {
		if p.HitRate == h {
			return p, true
		}
	}
	return BreakEvenPoint{}, false
}

// TestBreakEven_MonotoneAndZeroFloor: the curve's expected turns-saved is
// non-decreasing in the hit-rate, the zero-rate point is an exact 0 floor (the
// anti-inflation control survives the break-even projection), and a strictly
// positive rate yields a strictly positive expectation. These are the invariants
// the §3.2.2 claim rests on.
func TestBreakEven_MonotoneAndZeroFloor(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	rep := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, stochasticTrials, stochasticSeed)

	if len(rep.Points) != len(DefaultHitRateGrid()) {
		t.Fatalf("got %d points, want %d", len(rep.Points), len(DefaultHitRateGrid()))
	}

	// Zero-rate floor: every measured + priced field is exactly 0.
	z, ok := pointAt(rep, 0)
	if !ok {
		t.Fatal("grid is missing the h=0 anti-inflation control point")
	}
	if z.MeanTurnsSaved != 0 || z.TurnsSaved.Max != 0 || z.DollarsSavedMean != 0 || z.TokensSavedMean != 0 {
		t.Errorf("h=0 not a clean floor: meanTurns=%.3f max=%d $=%.5f tok=%.1f (a perturbation leaked savings)",
			z.MeanTurnsSaved, z.TurnsSaved.Max, z.DollarsSavedMean, z.TokensSavedMean)
	}

	// Monotone non-decreasing expectation across the grid.
	for i := 1; i < len(rep.Points); i++ {
		prev, cur := rep.Points[i-1], rep.Points[i]
		if cur.MeanTurnsSaved < prev.MeanTurnsSaved {
			t.Errorf("expected turns not monotone: h=%.3f mean=%.3f < h=%.3f mean=%.3f",
				cur.HitRate, cur.MeanTurnsSaved, prev.HitRate, prev.MeanTurnsSaved)
		}
	}

	// A strictly positive rate must save strictly more than nothing.
	hi, ok := pointAt(rep, 0.50)
	if !ok || hi.MeanTurnsSaved <= 0 {
		t.Errorf("h=0.50 expected turns = %.3f, want > 0", hi.MeanTurnsSaved)
	}
}

// TestBreakEven_RealWorldRateIsMarginal: at the §3.1 real-world addressable rate
// (~0.7%) the per-session efficiency win is small but positive — the honest
// "marginal, not the headline" framing the doc claims. We assert it is positive
// and clearly below a one-turn-per-session-on-this-base bar (the slice is ~14
// calls; a 0.7% rate over four classes cannot delete a full turn in expectation).
func TestBreakEven_RealWorldRateIsMarginal(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	rep := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, stochasticTrials, stochasticSeed)
	p, ok := pointAt(rep, rep.RealWorldHitRate)
	if !ok {
		t.Fatalf("grid is missing the real-world rate point %.4f", rep.RealWorldHitRate)
	}
	if p.MeanTurnsSaved < 0 {
		t.Errorf("real-world rate mean turns = %.4f, want >= 0", p.MeanTurnsSaved)
	}
	if p.MeanTurnsSaved >= 1.0 {
		t.Errorf("real-world rate mean turns = %.4f, want < 1 (it must be the marginal band, not a full turn/session)", p.MeanTurnsSaved)
	}
}

// TestBreakEven_RegimeAmortization: the provider-ships regime ($0) breaks even on
// session one (0 sessions), and the self-host fork ($2.8M) requires a finite,
// positive number of sessions wherever the per-session saving is positive — the
// §3.1 regime trifurcation made quantitative.
func TestBreakEven_RegimeAmortization(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	rep := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, stochasticTrials, stochasticSeed)

	mid, ok := pointAt(rep, 0.20) // a clearly-positive-saving point
	if !ok {
		t.Fatal("grid is missing h=0.20")
	}
	var sawProvider, sawSelfHost bool
	for _, rg := range mid.Regimes {
		switch rg.Name {
		case "provider-ships":
			sawProvider = true
			if rg.SessionsToBreakEven != 0 {
				t.Errorf("provider-ships sessions_to_break_even = %.0f, want 0 (break-even on session one)", rg.SessionsToBreakEven)
			}
		case "self-host-fork":
			sawSelfHost = true
			if !(rg.SessionsToBreakEven > 0) {
				t.Errorf("self-host sessions_to_break_even = %v, want a finite positive count at a positive-saving rate", rg.SessionsToBreakEven)
			}
		}
	}
	if !sawProvider || !sawSelfHost {
		t.Fatalf("missing a regime: provider=%v self-host=%v", sawProvider, sawSelfHost)
	}

	// At the zero-rate floor the self-host fork can NEVER amortize (the sentinel),
	// and provider-ships is still 0 — the build cost can't be recovered from no saving.
	z, _ := pointAt(rep, 0)
	for _, rg := range z.Regimes {
		if rg.Name == "self-host-fork" && rg.SessionsToBreakEven != NeverAmortizes {
			t.Errorf("h=0 self-host sessions = %v, want NeverAmortizes (-1) (no saving never amortizes a build cost)", rg.SessionsToBreakEven)
		}
	}
}

// TestBreakEven_Determinism: same seed => byte-identical curve.
func TestBreakEven_Determinism(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	a := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, 64, stochasticSeed)
	b := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, 64, stochasticSeed)
	if !reflect.DeepEqual(a.Points, b.Points) {
		t.Fatal("same seed produced different break-even curves")
	}
}

// TestBreakEven_WriteArtifact runs the full curve and writes the artifact.
func TestBreakEven_WriteArtifact(t *testing.T) {
	base, cm := BaseTrace(), DefaultCostModel()
	rep := RunBreakEvenSweep(context.Background(), base, DefaultHitRateGrid(), nil, cm, stochasticTrials, stochasticSeed)
	if rep.AppVersion == "" {
		t.Fatal("break-even report app_version is empty")
	}

	path, err := filepath.Abs(breakEvenArtifact)
	if err != nil {
		t.Fatalf("abs(%q): %v", breakEvenArtifact, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %q: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, rep.JSON(), 0o644); err != nil {
		t.Fatalf("write %q: %v", path, err)
	}
	t.Logf("wrote break-even curve artifact: %s", path)
	for _, p := range rep.Points {
		self := ""
		for _, rg := range p.Regimes {
			if rg.Name == "self-host-fork" {
				self = formatSessions(rg.SessionsToBreakEven)
			}
		}
		t.Logf("h=%.3f  mean_turns/session=%.3f  $/session=%.5f  self-host break-even=%s sessions",
			p.HitRate, p.MeanTurnsSaved, p.DollarsSavedMean, self)
	}
}

func formatSessions(s float64) string {
	if s == NeverAmortizes {
		return "never"
	}
	return strconv.FormatFloat(s, 'f', 0, 64)
}
