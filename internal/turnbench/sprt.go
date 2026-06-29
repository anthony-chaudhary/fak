// sprt.go — SPRT SEQUENTIAL EARLY-STOP for the ablation A/B gate (issue #1164, T8 of the
// self-tax assurance epic #1147). The ablation gate compares two arms — fak-ON vs fak-OFF
// (or a candidate lever vs baseline) — over a stream of per-sample effects (a per-turn
// overhead delta, a per-trace tax difference) and must reach a verdict: did the treatment
// REGRESS (over budget) or not? A FIXED-N gate draws a pre-computed sample size N and decides
// once. SPRT (Wald's Sequential Probability Ratio Test) instead accumulates the log-likelihood
// ratio sample-by-sample and STOPS THE MOMENT the evidence crosses a decision boundary — on
// average using only ~HALF the fixed-N samples for the SAME error guarantees (Wald's
// efficiency theorem). That halving is this ticket's acceptance: the sequential gate reaches
// the SAME verdict as fixed-N on a fixture while consuming ~50% of the samples.
//
// THE TWO BOUNDARIES (what "stop early" means here). Over a stream of observations X_i, the
// gate tests two simple hypotheses about the mean effect:
//
//	H0: μ = mu0  — NO effect (no regression / the candidate is "not improving")
//	H1: μ = mu1  — the minimum effect worth detecting (a regression of size mu1-mu0)
//
// The cumulative log-likelihood ratio Λ_n = Σ log[f1(X_i)/f0(X_i)] is compared to Wald's two
// boundaries derived from the target error rates (α false-positive, β false-negative):
//
//	upper A = ln((1-β)/α)   — cross ABOVE ⇒ accept H1 (the EFFECT / REGRESSION is real)
//	lower B = ln(β/(1-α))   — cross BELOW ⇒ accept H0 (the FUTILITY boundary: "not improving")
//	B < Λ_n < A             — CONTINUE: draw another sample, the evidence is not yet decisive
//
// The lower boundary IS the design note's "futility boundary for 'not improving'": when the
// stream of effects keeps failing to favor H1, Λ_n drifts down and crosses B, so the gate
// stops EARLY and declares no-effect rather than burning the whole fixed-N budget to confirm
// what is already clear.
//
// THE MODEL (stated honestly — see also ope.go's honesty fence). Λ_n's increment is the
// known-variance Gaussian log-likelihood ratio: each X_i ~ N(μ, σ²) with σ a known/estimated
// per-sample scale. For that family the per-sample increment closes to
//
//	l_i = (mu1-mu0)/σ² · ( X_i − (mu0+mu1)/2 )
//
// so the test carries no per-sample distribution state beyond the running sum — it is the
// textbook Wald SPRT, not a novel estimator. The ~half-samples win is Wald's, not measured
// magic: under H1 the expected sample size is ≈ 2·ln((1-β)/α)/d² against a fixed-N of
// ((z_{1-α}+z_{1-β})/d)² (d = (mu1-mu0)/σ), a ratio that sits near 0.5 for the conventional
// α=β=0.05 design. sprt_test.go witnesses that ratio on a deterministic fixture.
package turnbench

import (
	"fmt"
	"math"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// SPRTVerdict is the decision the sequential test reports after each observation: still
// undecided (Continue), or a terminal accept of one hypothesis. The two terminal values map
// directly onto the ablation gate's outcome — H1 = the regression the gate watches for is
// real; H0 = the futility boundary, no regression / "not improving".
type SPRTVerdict string

const (
	// SPRTContinue: neither boundary crossed — the evidence is not yet decisive, draw another
	// sample. A gate that runs out of samples while still Continue falls back to the fixed-N
	// decision (RunSequentialGate handles that).
	SPRTContinue SPRTVerdict = "continue"

	// SPRTAcceptH1: the upper boundary was crossed — accept the alternative. In gate terms the
	// EFFECT (regression / the treatment arm is over budget) is real.
	SPRTAcceptH1 SPRTVerdict = "accept-h1"

	// SPRTAcceptH0: the lower (FUTILITY) boundary was crossed — accept the null. In gate terms
	// there is NO effect ("not improving" / the candidate did not regress).
	SPRTAcceptH0 SPRTVerdict = "accept-h0"
)

// SPRT is a running Wald Sequential Probability Ratio Test for the mean of a known-variance
// Gaussian stream. It holds only the cumulative log-likelihood ratio and the two precomputed
// boundaries, so Observe is O(1) and the test carries no per-sample history. One SPRT decides
// ONE A/B comparison; reuse across comparisons is by constructing a fresh test (the boundaries
// depend only on α/β, but the running LLR must reset).
type SPRT struct {
	mu0, mu1 float64 // null and alternative means; mu1 != mu0 (mu1 > mu0 for an upper one-sided gate)
	sigma    float64 // known/assumed per-sample standard deviation (> 0)
	alpha    float64 // target false-positive rate (accept H1 when H0 is true)
	beta     float64 // target false-negative rate (accept H0 when H1 is true)

	upper float64 // Wald upper boundary ln((1-β)/α): cross above ⇒ accept H1
	lower float64 // Wald lower boundary ln(β/(1-α)): cross below ⇒ accept H0 (futility)

	llr float64 // cumulative log-likelihood ratio Λ_n
	n   int     // observations consumed so far
}

// NewSPRT builds a Wald SPRT for H0:μ=mu0 vs H1:μ=mu1 over a Gaussian stream of known scale
// sigma, with target error rates alpha (false-positive) and beta (false-negative). It
// validates the parameters and precomputes the two boundaries; the running LLR starts at 0
// (no evidence). It errors on a degenerate design (non-positive sigma, an error rate outside
// (0,1), or mu0==mu1 — which would make the two hypotheses indistinguishable).
func NewSPRT(mu0, mu1, sigma, alpha, beta float64) (*SPRT, error) {
	switch {
	case sigma <= 0:
		return nil, fmt.Errorf("turnbench: SPRT needs sigma > 0, got %g", sigma)
	case alpha <= 0 || alpha >= 1:
		return nil, fmt.Errorf("turnbench: SPRT needs 0 < alpha < 1, got %g", alpha)
	case beta <= 0 || beta >= 1:
		return nil, fmt.Errorf("turnbench: SPRT needs 0 < beta < 1, got %g", beta)
	case mu0 == mu1:
		return nil, fmt.Errorf("turnbench: SPRT needs mu0 != mu1 (the hypotheses must differ), got %g", mu0)
	}
	return &SPRT{
		mu0:   mu0,
		mu1:   mu1,
		sigma: sigma,
		alpha: alpha,
		beta:  beta,
		upper: math.Log((1 - beta) / alpha),
		lower: math.Log(beta / (1 - alpha)),
	}, nil
}

// Observe folds one sample into the running log-likelihood ratio and returns the test's
// verdict AFTER this sample: SPRTContinue while B < Λ_n < A, or a terminal accept once a
// boundary is crossed. Calling Observe again after a terminal verdict keeps accumulating (the
// caller decides when to stop); the verdict is always a pure function of the current Λ_n.
func (s *SPRT) Observe(x float64) SPRTVerdict {
	s.n++
	// Known-variance Gaussian log-likelihood-ratio increment (see the file header derivation).
	s.llr += (s.mu1 - s.mu0) / (s.sigma * s.sigma) * (x - (s.mu0+s.mu1)/2)
	return s.Verdict()
}

// Verdict reports the current decision from the accumulated LLR without consuming a sample.
func (s *SPRT) Verdict() SPRTVerdict {
	switch {
	case s.llr >= s.upper:
		return SPRTAcceptH1
	case s.llr <= s.lower:
		return SPRTAcceptH0
	default:
		return SPRTContinue
	}
}

// N is how many observations the test has consumed.
func (s *SPRT) N() int { return s.n }

// LLR is the current cumulative log-likelihood ratio Λ_n.
func (s *SPRT) LLR() float64 { return s.llr }

// Boundaries returns the precomputed Wald (upper, lower) decision boundaries.
func (s *SPRT) Boundaries() (upper, lower float64) { return s.upper, s.lower }

// NFixed is the classic fixed-sample size a one-sided test of H0:μ=mu0 vs H1:μ=mu1 would need
// for the SAME (alpha, beta) guarantees: N = ((z_{1-α} + z_{1-β})·σ / (mu1-mu0))². It is the
// denominator of the ~half-samples claim — the sequential gate's expected stop is ≈ N/2 — and
// the principled size for a fixture that compares the two gates apples-to-apples.
func NFixed(mu0, mu1, sigma, alpha, beta float64) int {
	d := math.Abs(mu1-mu0) / sigma
	za := invNormCDF(1 - alpha)
	zb := invNormCDF(1 - beta)
	n := math.Pow((za+zb)/d, 2)
	return int(math.Ceil(n))
}

// fixedNVerdict is the FIXED-N reference gate: pool all samples, form the one-sided z-statistic
// of the sample mean against mu0, and accept H1 iff it clears the level-alpha critical value
// (else accept H0). This is the verdict the sequential gate must MATCH on the fixture while
// using fewer samples. An empty stream cannot reject the null, so it reads H0.
func fixedNVerdict(samples []float64, mu0, sigma, alpha float64) SPRTVerdict {
	n := len(samples)
	if n == 0 {
		return SPRTAcceptH0
	}
	sum := 0.0
	for _, x := range samples {
		sum += x
	}
	mean := sum / float64(n)
	z := (mean - mu0) / (sigma / math.Sqrt(float64(n)))
	if z >= invNormCDF(1-alpha) {
		return SPRTAcceptH1
	}
	return SPRTAcceptH0
}

// SequentialGateResult is the ablation gate's sequential artifact: the SPRT verdict and how
// few samples it took, alongside the fixed-N verdict over the SAME stream, so a reader can
// check the two agree and see the sample saving. It mirrors the other turnbench reports
// (Provenance + JSON()).
type SequentialGateResult struct {
	Provenance Provenance `json:"provenance"`

	// Verdict is the gate's final decision. When the SPRT crossed a boundary before the stream
	// was exhausted it is the SPRT's terminal verdict; otherwise the gate falls back to the
	// fixed-N decision (SamplesUsed == the stream length — no saving on this stream).
	Verdict SPRTVerdict `json:"verdict"`

	// DecidedEarly is true iff the SPRT crossed a boundary strictly before consuming the whole
	// stream — i.e. the sequential test actually stopped early.
	DecidedEarly bool `json:"decided_early"`

	SamplesUsed int `json:"samples_used"` // SPRT observations consumed before crossing (== stream length if it never crossed)
	FixedN      int `json:"fixed_n"`      // the fixed-N reference size NFixed(mu0,mu1,sigma,alpha,beta) — the saving's denominator

	FixedVerdict SPRTVerdict `json:"fixed_verdict"` // the fixed-N gate's verdict over the whole stream
	Agree        bool        `json:"agree"`         // DecidedEarly AND the SPRT verdict == the fixed-N verdict

	SampleRatio float64 `json:"sample_ratio"` // SamplesUsed / FixedN — the headline (~0.5 = halved)

	LLR   float64 `json:"final_llr"`      // cumulative LLR at the stopping point
	Upper float64 `json:"upper_boundary"` // Wald upper boundary (accept H1)
	Lower float64 `json:"lower_boundary"` // Wald lower (futility) boundary (accept H0)

	Note string `json:"note"`
}

// JSON renders the report.
func (r *SequentialGateResult) JSON() []byte { return marshalArtifact(r) }

// RunSequentialGate runs the SPRT over the sample stream until it crosses a boundary or the
// stream is exhausted, and pairs it with the fixed-N verdict over the SAME stream. The result
// witnesses this ticket's acceptance: on a fixture the sequential gate reaches the same
// verdict as fixed-N (Agree) while consuming ~half the samples (SampleRatio). label is stamped
// as the artifact's SliceID; samples is the per-sample effect stream (treatment-minus-control,
// per-turn overhead, …) the gate decides over.
func RunSequentialGate(label string, samples []float64, mu0, mu1, sigma, alpha, beta float64) (*SequentialGateResult, error) {
	s, err := NewSPRT(mu0, mu1, sigma, alpha, beta)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("turnbench: RunSequentialGate needs a non-empty sample stream")
	}
	// FixedN is the PRINCIPLED reference, not the stream length: the sample size a fixed-N
	// test of the SAME (alpha, beta) design must draw for its error guarantees. The headline
	// ratio SamplesUsed/FixedN is therefore "what fraction of the fixed-N budget the
	// sequential gate actually spent". The caller should supply a stream at least FixedN long
	// — ideally a few× — so the SPRT virtually always reaches a boundary before the stream
	// runs out; a stream capped at FixedN would censor the test's tail and understate the
	// saving. If the stream is exhausted undecided the gate falls back to fixed-N (no saving).
	fixedN := NFixed(mu0, mu1, sigma, alpha, beta)

	seqVerdict := SPRTContinue
	used := len(samples)
	for i, x := range samples {
		if v := s.Observe(x); v != SPRTContinue {
			seqVerdict, used = v, i+1
			break
		}
	}
	decidedEarly := seqVerdict != SPRTContinue

	// The fixed-N gate decides over its OWN budget — the first fixedN samples of the stream
	// (or the whole stream when it is shorter than the reference size) — not the long tail the
	// sequential gate never needed to look at.
	fixedSamples := samples
	if len(fixedSamples) > fixedN {
		fixedSamples = fixedSamples[:fixedN]
	}
	fixed := fixedNVerdict(fixedSamples, mu0, sigma, alpha)
	final := seqVerdict
	if !decidedEarly {
		// The SPRT never reached a boundary within the supplied stream — fall back to the
		// fixed-N decision (no early stop, no saving) so the gate still returns a verdict.
		final = fixed
	}
	agree := decidedEarly && seqVerdict == fixed
	ratio := float64(used) / float64(fixedN)

	note := fmt.Sprintf(
		"SPRT used %d samples vs fixed-N reference %d (%.0f%%) and read %q; fixed-N read %q (agree=%t). "+
			"upper=%.3f lower=%.3f final_llr=%.3f.",
		used, fixedN, ratio*100, final, fixed, agree, s.upper, s.lower, s.llr)
	if !decidedEarly {
		note = fmt.Sprintf(
			"SPRT did NOT cross a boundary within the %d-sample stream; gate fell back to the "+
				"fixed-N verdict %q (no early stop). upper=%.3f lower=%.3f final_llr=%.3f.",
			len(samples), fixed, s.upper, s.lower, s.llr)
	}

	return &SequentialGateResult{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunSequentialGate",
			SliceID:     label,
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (SPRT sequential ablation gate)",
		},
		Verdict:      final,
		DecidedEarly: decidedEarly,
		SamplesUsed:  used,
		FixedN:       fixedN,
		FixedVerdict: fixed,
		Agree:        agree,
		SampleRatio:  ratio,
		LLR:          s.llr,
		Upper:        s.upper,
		Lower:        s.lower,
		Note:         note,
	}, nil
}

// invNormCDF is the inverse standard-normal CDF Φ⁻¹(p) (the p-quantile) via Acklam's rational
// approximation, |abs error| < 1.15e-9 over p ∈ (0,1). Unlike the tabulated zFor (ope.go), it
// is exact for ANY level, so NFixed and the fixed-N z-test are correct for any (alpha, beta)
// design, not just the conventional ones. p outside (0,1) returns ±Inf.
func invNormCDF(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	a := [...]float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	b := [...]float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := [...]float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := [...]float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
	const plow = 0.02425
	const phigh = 1 - plow
	switch {
	case p < plow:
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= phigh:
		q := p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
}
