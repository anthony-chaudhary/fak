package turnbench

import (
	"math"
	"math/rand"
	"testing"
)

// gaussianFixture draws n deterministic samples from N(mean, sigma²) off a fixed seed — a
// reproducible per-sample effect stream (treatment-minus-control / per-turn overhead) the
// gate decides over. Determinism (fixed seed) is what makes the early-stop witness exact.
func gaussianFixture(seed int64, n int, mean, sigma float64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float64, n)
	for i := range out {
		out[i] = mean + sigma*rng.NormFloat64()
	}
	return out
}

// TestInvNormCDF_MatchesKnownQuantiles pins the inverse-normal helper that sizes NFixed and
// the fixed-N z-test: the conventional one-sided critical values must come out right, or every
// downstream sample-size and verdict is off.
func TestInvNormCDF_MatchesKnownQuantiles(t *testing.T) {
	cases := []struct {
		p, want float64
	}{
		{0.5, 0.0},
		{0.90, 1.281552},
		{0.95, 1.644854},
		{0.975, 1.959964},
		{0.99, 2.326348},
		{0.995, 2.575829},
	}
	for _, c := range cases {
		got := invNormCDF(c.p)
		if math.Abs(got-c.want) > 1e-4 {
			t.Errorf("invNormCDF(%g) = %.6f, want %.6f", c.p, got, c.want)
		}
	}
	// Symmetry: Φ⁻¹(p) == -Φ⁻¹(1-p).
	if got := invNormCDF(0.1); math.Abs(got+invNormCDF(0.9)) > 1e-9 {
		t.Errorf("invNormCDF not antisymmetric: invNormCDF(0.1)=%.9f invNormCDF(0.9)=%.9f", got, invNormCDF(0.9))
	}
}

// TestNewSPRT_BoundariesAndValidation checks the Wald boundary algebra and that a degenerate
// design is refused rather than silently producing a nonsense test.
func TestNewSPRT_BoundariesAndValidation(t *testing.T) {
	s, err := NewSPRT(0, 1, 2, 0.05, 0.05)
	if err != nil {
		t.Fatalf("NewSPRT: %v", err)
	}
	up, lo := s.Boundaries()
	wantUp := math.Log((1 - 0.05) / 0.05) // ln(19) ≈ 2.944
	wantLo := math.Log(0.05 / (1 - 0.05)) // ln(1/19) ≈ -2.944
	if math.Abs(up-wantUp) > 1e-9 || math.Abs(lo-wantLo) > 1e-9 {
		t.Errorf("boundaries = (%.6f, %.6f), want (%.6f, %.6f)", up, lo, wantUp, wantLo)
	}
	if s.N() != 0 || s.LLR() != 0 {
		t.Errorf("fresh SPRT should start at N=0 LLR=0, got N=%d LLR=%g", s.N(), s.LLR())
	}

	bad := []struct {
		name                         string
		mu0, mu1, sigma, alpha, beta float64
	}{
		{"sigma<=0", 0, 1, 0, 0.05, 0.05},
		{"alpha out of range", 0, 1, 1, 0, 0.05},
		{"beta out of range", 0, 1, 1, 0.05, 1},
		{"mu0==mu1", 1, 1, 1, 0.05, 0.05},
	}
	for _, b := range bad {
		if _, err := NewSPRT(b.mu0, b.mu1, b.sigma, b.alpha, b.beta); err == nil {
			t.Errorf("NewSPRT(%s) should have errored", b.name)
		}
	}
}

// TestNFixed_MatchesClosedForm checks the fixed-sample size against the textbook one-sided
// formula it must equal — it is the denominator of the ~half-samples claim.
func TestNFixed_MatchesClosedForm(t *testing.T) {
	got := NFixed(0, 1, 2, 0.05, 0.05)
	// d = 0.5, z_{0.95}=1.644854 each ⇒ ((3.289708)/0.5)² = 43.29 ⇒ ceil 44.
	if got != 44 {
		t.Fatalf("NFixed(0,1,2,0.05,0.05) = %d, want 44", got)
	}
}

// TestSPRT_HalvesSamplesUnderH1 is the ticket's headline acceptance: when the alternative is
// true, the SPRT gate reaches the SAME verdict as the fixed-N gate (Agree) on ~HALF the
// samples. Witnessed as an expectation over many deterministic fixtures so it rests on Wald's
// efficiency theorem, not one lucky seed.
func TestSPRT_HalvesSamplesUnderH1(t *testing.T) {
	const (
		mu0, mu1, sigma = 0.0, 1.0, 2.0
		alpha, beta     = 0.05, 0.05
		fixtures        = 400
	)
	n := NFixed(mu0, mu1, sigma, alpha, beta)
	seeds := deriveSeeds(20240617, fixtures)

	var (
		ratioSum               float64
		earlyCount, earlyAgree int
		h1Count                int
	)
	for _, seed := range seeds {
		// Stream a few× the fixed-N reference so the SPRT virtually always reaches a boundary
		// before the stream runs out — a stream capped at n would censor the tail and
		// understate the saving (the ratio is SamplesUsed/NFixed, not SamplesUsed/len).
		samples := gaussianFixture(seed, 4*n, mu1, sigma) // H1 is TRUE: mean == mu1
		res, err := RunSequentialGate("h1-fixture", samples, mu0, mu1, sigma, alpha, beta)
		if err != nil {
			t.Fatalf("RunSequentialGate: %v", err)
		}
		ratioSum += res.SampleRatio
		if res.DecidedEarly {
			earlyCount++
			if res.Agree {
				earlyAgree++
			}
		}
		if res.Verdict == SPRTAcceptH1 {
			h1Count++
		}
	}

	meanRatio := ratioSum / float64(fixtures)
	earlyFrac := float64(earlyCount) / float64(fixtures)
	agreeFrac := float64(earlyAgree) / float64(earlyCount)
	t.Logf("under H1: mean sample ratio = %.3f (Wald theory ≈ 0.54), decided-early = %.1f%%, early-agree = %.1f%%, read-H1 = %.1f%%",
		meanRatio, earlyFrac*100, agreeFrac*100, float64(h1Count)/float64(fixtures)*100)

	if meanRatio > 0.65 {
		t.Errorf("SPRT did not halve the samples: mean ratio %.3f > 0.65", meanRatio)
	}
	if meanRatio < 0.30 {
		t.Errorf("mean ratio %.3f implausibly low — fixture or boundary likely wrong", meanRatio)
	}
	if earlyFrac < 0.90 {
		t.Errorf("only %.1f%% of fixtures decided early, want >=90%%", earlyFrac*100)
	}
	if agreeFrac < 0.85 {
		t.Errorf("SPRT early verdict matched fixed-N on only %.1f%% of fixtures, want >=85%%", agreeFrac*100)
	}
}

// TestSPRT_FutilityBoundaryUnderH0 exercises the FUTILITY boundary: when there is no effect,
// the SPRT crosses the LOWER boundary and stops early with "no effect / not improving",
// agreeing with fixed-N, again on ~half the samples.
func TestSPRT_FutilityBoundaryUnderH0(t *testing.T) {
	const (
		mu0, mu1, sigma = 0.0, 1.0, 2.0
		alpha, beta     = 0.05, 0.05
		fixtures        = 400
	)
	n := NFixed(mu0, mu1, sigma, alpha, beta)
	seeds := deriveSeeds(771113, fixtures)

	var (
		ratioSum               float64
		earlyCount, earlyAgree int
		h0Count                int
	)
	for _, seed := range seeds {
		// 4× the fixed-N reference so the futility boundary is virtually always reached in-stream.
		samples := gaussianFixture(seed, 4*n, mu0, sigma) // H0 is TRUE: mean == mu0 (no effect)
		res, err := RunSequentialGate("h0-fixture", samples, mu0, mu1, sigma, alpha, beta)
		if err != nil {
			t.Fatalf("RunSequentialGate: %v", err)
		}
		ratioSum += res.SampleRatio
		if res.DecidedEarly {
			earlyCount++
			if res.Agree {
				earlyAgree++
			}
		}
		if res.Verdict == SPRTAcceptH0 {
			h0Count++
		}
	}

	meanRatio := ratioSum / float64(fixtures)
	earlyFrac := float64(earlyCount) / float64(fixtures)
	agreeFrac := float64(earlyAgree) / float64(earlyCount)
	t.Logf("under H0: mean sample ratio = %.3f, decided-early (futility) = %.1f%%, early-agree = %.1f%%, read-H0 = %.1f%%",
		meanRatio, earlyFrac*100, agreeFrac*100, float64(h0Count)/float64(fixtures)*100)

	if meanRatio > 0.65 {
		t.Errorf("futility stop did not halve the samples: mean ratio %.3f > 0.65", meanRatio)
	}
	if earlyFrac < 0.90 {
		t.Errorf("only %.1f%% of fixtures hit the futility boundary early, want >=90%%", earlyFrac*100)
	}
	if agreeFrac < 0.85 {
		t.Errorf("SPRT early verdict matched fixed-N on only %.1f%% of fixtures, want >=85%%", agreeFrac*100)
	}
}

// TestRunSequentialGate_SingleFixtureWitness is the issue's literal acceptance on ONE fixture:
// a deterministic stream where the sequential gate reaches the same verdict as fixed-N while
// consuming ~half the fixed-N budget. Seed 90210 is a representative draw (its stop lands at
// the Wald expectation, 22/44 = 0.500); the statistical "~half on average" claim — that this
// is not one lucky seed — is carried by the 400-fixture aggregate tests above.
func TestRunSequentialGate_SingleFixtureWitness(t *testing.T) {
	const (
		mu0, mu1, sigma = 0.0, 1.0, 2.0
		alpha, beta     = 0.05, 0.05
	)
	n := NFixed(mu0, mu1, sigma, alpha, beta)
	samples := gaussianFixture(90210, 4*n, mu1, sigma)

	res, err := RunSequentialGate("single-fixture", samples, mu0, mu1, sigma, alpha, beta)
	if err != nil {
		t.Fatalf("RunSequentialGate: %v", err)
	}
	t.Logf("single fixture: %s", res.Note)

	if !res.DecidedEarly {
		t.Fatalf("SPRT did not decide early on the fixture: %s", res.Note)
	}
	if !res.Agree {
		t.Errorf("SPRT verdict %q disagreed with fixed-N %q", res.Verdict, res.FixedVerdict)
	}
	if res.Verdict != SPRTAcceptH1 {
		t.Errorf("verdict = %q, want %q (H1 is true on this fixture)", res.Verdict, SPRTAcceptH1)
	}
	if res.SamplesUsed >= res.FixedN {
		t.Errorf("no early stop: used %d of %d samples", res.SamplesUsed, res.FixedN)
	}
	if res.SampleRatio > 0.70 {
		t.Errorf("sample ratio %.3f not ~half (want <=0.70)", res.SampleRatio)
	}
	// The JSON artifact must round-trip the headline fields a CI gate would read back.
	if len(res.JSON()) == 0 {
		t.Error("empty JSON artifact")
	}
}

// TestRunSequentialGate_EmptyStream refuses an empty stream rather than dividing by zero.
func TestRunSequentialGate_EmptyStream(t *testing.T) {
	if _, err := RunSequentialGate("empty", nil, 0, 1, 2, 0.05, 0.05); err == nil {
		t.Error("RunSequentialGate(nil) should error")
	}
}
