package cvregress

// power.go — the STOCHASTIC counterpart to the pinned-baseline fold. Where Fold compares a
// realized cache-efficiency number against a deterministic pin, a stochastic quality gate
// compares two NOISY distributions (this build vs a baseline) and must answer a prior
// question first: is the sample big enough to tell a real shift from sampling noise? A run
// with too few samples is UNDERPOWERED — it cannot detect the effect it claims to guard, so
// a clean underpowered run is INCONCLUSIVE, never a verified pass. This mirrors the
// power-analysis discipline of lm-evaluation-harness / HELM (#4509): version the sample-size
// decision, not just the score.
//
// The seam is PURE and deterministic. RequiredN and Assess are O(1) closed-form; Simulate is
// a seeded Monte-Carlo (math/rand with an explicit Seed) that replays identically on every
// platform, so "the simulation met its error/power targets" is reproducible provenance, not a
// one-off observation. Runtime/resource cost: Assess is free (a few flops); Simulate is
// O(Trials*N) draws — a PR-tier case keeps Trials modest (a few thousand), a nightly/release
// tier can afford more. Assign the tier on the PowerSpec.

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// Power verdicts — the stochastic vocabulary that folds into the package's OK | REGRESSED |
// INSUFFICIENT verdicts via StochasticVerdict. INVALID is a malformed spec (bad statistics or
// provenance); UNDERPOWERED is a real, load-bearing "cannot conclude".
const (
	VerdictAdequate     = "ADEQUATELY_POWERED"
	VerdictUnderpowered = "UNDERPOWERED"
	VerdictInvalid      = "INVALID"
)

// Tier is the run cadence a stochastic case is assigned to — an explicit contract the epic
// (#4509) requires so a resource cost is attached to a schedule. A heavy simulation belongs
// on nightly/release, not on every PR.
const (
	TierPR      = "pr"
	TierNightly = "nightly"
	TierRelease = "release"
)

// PowerSpec is one stochastic quality case: the statistics that size its sample plus the
// provenance that makes the decision replayable. EffectSize (delta) and StdDev (sigma) are in
// the SAME units as the metric under test (e.g. hit-rate percentage points); the sample-size
// math depends only on their ratio, the standardized effect delta/sigma.
type PowerSpec struct {
	// The statistics that determine the sample size.
	EffectSize float64 `json:"effect_size"` // delta: the smallest mean shift worth detecting (> 0)
	StdDev     float64 `json:"std_dev"`     // sigma: the metric's per-sample standard deviation (> 0)
	Alpha      float64 `json:"alpha"`       // type-I (false-positive) rate, in (0,1), e.g. 0.05
	Power      float64 `json:"power"`       // target power 1-beta (true-positive rate), in (0,1), e.g. 0.80
	Tails      int     `json:"tails"`       // 1 (one-sided) or 2 (two-sided)

	// Provenance the case records so a verdict is auditable and replayable (#4509 shared
	// contract). Seed makes the Monte-Carlo deterministic; the rest name what was under test.
	Seed     int64  `json:"seed,omitempty"`     // deterministic-simulation seed
	Oracle   string `json:"oracle,omitempty"`   // deterministic oracle / metric name the case scores
	Revision string `json:"revision,omitempty"` // code/module revision under test
	Baseline string `json:"baseline,omitempty"` // tolerance/baseline provenance the effect is measured against
	Tier     string `json:"tier,omitempty"`     // pr | nightly | release — the run cadence this case is assigned to
}

// Validate refuses a malformed spec: the sample-size math is undefined for a non-positive
// effect or variance, an alpha/power outside (0,1), or a tail count that is not 1 or 2. A Tier,
// if set, must be one of the three explicit cadences. A malformed spec is INVALID, never a pass.
func (s PowerSpec) Validate() error {
	if !(s.EffectSize > 0) || math.IsNaN(s.EffectSize) || math.IsInf(s.EffectSize, 0) {
		return fmt.Errorf("cvregress: effect_size must be > 0 (finite), got %v", s.EffectSize)
	}
	if !(s.StdDev > 0) || math.IsNaN(s.StdDev) || math.IsInf(s.StdDev, 0) {
		return fmt.Errorf("cvregress: std_dev must be > 0 (finite), got %v", s.StdDev)
	}
	if !(s.Alpha > 0 && s.Alpha < 1) {
		return fmt.Errorf("cvregress: alpha must be in (0,1), got %v", s.Alpha)
	}
	if !(s.Power > 0 && s.Power < 1) {
		return fmt.Errorf("cvregress: power must be in (0,1), got %v", s.Power)
	}
	if s.Tails != 1 && s.Tails != 2 {
		return fmt.Errorf("cvregress: tails must be 1 or 2, got %d", s.Tails)
	}
	switch s.Tier {
	case "", TierPR, TierNightly, TierRelease:
	default:
		return fmt.Errorf("cvregress: tier must be one of pr|nightly|release, got %q", s.Tier)
	}
	return nil
}

// RequiredN is the minimum number of (paired) samples needed to detect EffectSize at the
// spec's alpha and power, from the textbook normal-approximation formula
//
//	n = ceil( ((z_{1-alpha/tails} + z_power) * sigma / delta)^2 ).
//
// It is the answer to the issue's scope line — "choose sample count from effect size,
// variance, alpha, and power". A malformed spec returns an error, never a silent n.
func RequiredN(s PowerSpec) (int, error) {
	if err := s.Validate(); err != nil {
		return 0, err
	}
	za := mathx.NormalQuantile(1 - s.Alpha/float64(s.Tails))
	zb := mathx.NormalQuantile(s.Power)
	n := math.Pow((za+zb)*s.StdDev/s.EffectSize, 2)
	return int(math.Ceil(n)), nil
}

// PowerReport is Assess's verdict: whether the sample on hand is large enough to trust a null
// (no-regression) result. Conclusive is the load-bearing bit — false whenever the run is
// underpowered or the spec is invalid, so a caller can never read an underpowered "clean" run
// as a verified pass. Finding names the first actionable divergence.
type PowerReport struct {
	Spec       PowerSpec `json:"spec"`
	Verdict    string    `json:"verdict"` // ADEQUATELY_POWERED | UNDERPOWERED | INVALID
	Adequate   bool      `json:"adequate"`
	Conclusive bool      `json:"conclusive"` // false when underpowered/invalid — inconclusive is never a pass
	RequiredN  int       `json:"required_n"`
	ActualN    int       `json:"actual_n"`
	Finding    string    `json:"finding"`
}

// Assess grades an actual sample count against the spec's required sample size. actualN below
// the requirement is UNDERPOWERED and NOT conclusive; a malformed spec is INVALID and not
// conclusive. Only a sample at or above the requirement is ADEQUATELY_POWERED and conclusive.
func Assess(s PowerSpec, actualN int) PowerReport {
	rep := PowerReport{Spec: s, ActualN: actualN}
	req, err := RequiredN(s)
	if err != nil {
		rep.Verdict = VerdictInvalid
		rep.Finding = "INVALID — " + err.Error()
		return rep
	}
	rep.RequiredN = req
	switch {
	case actualN < req:
		rep.Verdict = VerdictUnderpowered
		rep.Finding = fmt.Sprintf(
			"UNDERPOWERED — n=%d < required %d to detect delta=%.4g at sigma=%.4g, alpha=%.4g (%d-tailed), power=%.4g; result is inconclusive, not a pass",
			actualN, req, s.EffectSize, s.StdDev, s.Alpha, s.Tails, s.Power)
	default:
		rep.Adequate = true
		rep.Conclusive = true
		rep.Verdict = VerdictAdequate
		rep.Finding = fmt.Sprintf(
			"ADEQUATELY_POWERED — n=%d >= required %d to detect delta=%.4g at sigma=%.4g, alpha=%.4g (%d-tailed), power=%.4g",
			actualN, req, s.EffectSize, s.StdDev, s.Alpha, s.Tails, s.Power)
	}
	return rep
}

// SimResult is the empirical outcome of a seeded Monte-Carlo validation of a sample size:
// EmpiricalPower is the fraction of true-effect trials that correctly rejected the null,
// EmpiricalAlpha the fraction of no-effect trials that falsely rejected it. MeetsTargets is
// true only when both clear the spec's targets within tolerance — the "simulations meet
// error/power targets" acceptance made checkable.
type SimResult struct {
	Trials         int     `json:"trials"`
	N              int     `json:"n"`
	Seed           int64   `json:"seed"`
	EmpiricalPower float64 `json:"empirical_power"`
	EmpiricalAlpha float64 `json:"empirical_alpha"`
	Tolerance      float64 `json:"tolerance"`
	MeetsTargets   bool    `json:"meets_targets"`
	Finding        string  `json:"finding"`
}

// Simulate validates a chosen sample size n by replaying the exact z-test the sample-size math
// assumes: for each of Trials iterations it draws n samples from N(delta, sigma) and counts how
// often the known-sigma z-test rejects the null (empirical power), then draws n samples from
// N(0, sigma) and counts false rejections (empirical alpha). The draw stream is seeded from the
// spec, so the result is identical on every replay. It reports MeetsTargets = empirical power is
// within tol of the target AND empirical alpha does not exceed alpha by more than tol.
func Simulate(s PowerSpec, n, trials int, tol float64) SimResult {
	res := SimResult{Trials: trials, N: n, Seed: s.Seed, Tolerance: tol}
	if err := s.Validate(); err != nil {
		res.Finding = "INVALID — " + err.Error()
		return res
	}
	if n <= 0 || trials <= 0 {
		res.Finding = fmt.Sprintf("INVALID — n=%d and trials=%d must both be > 0", n, trials)
		return res
	}
	// Reject-region critical value, matching RequiredN's z_{1-alpha/tails}.
	zc := mathx.NormalQuantile(1 - s.Alpha/float64(s.Tails))
	rng := rand.New(rand.NewSource(s.Seed))
	se := s.StdDev / math.Sqrt(float64(n)) // standard error of the mean of n draws

	reject := func(mean float64) bool {
		var sum float64
		for i := 0; i < n; i++ {
			sum += mean + s.StdDev*rng.NormFloat64()
		}
		z := (sum / float64(n)) / se
		if s.Tails == 2 {
			return math.Abs(z) > zc
		}
		return z > zc // one-sided upper tail; EffectSize is > 0 by Validate
	}

	var powerHits, alphaHits int
	for i := 0; i < trials; i++ {
		if reject(s.EffectSize) {
			powerHits++
		}
		if reject(0) {
			alphaHits++
		}
	}
	res.EmpiricalPower = float64(powerHits) / float64(trials)
	res.EmpiricalAlpha = float64(alphaHits) / float64(trials)
	res.MeetsTargets = res.EmpiricalPower >= s.Power-tol && res.EmpiricalAlpha <= s.Alpha+tol
	if res.MeetsTargets {
		res.Finding = fmt.Sprintf("OK — empirical power %.3f >= %.3f-%.3f and empirical alpha %.3f <= %.3f+%.3f over %d trials at n=%d",
			res.EmpiricalPower, s.Power, tol, res.EmpiricalAlpha, s.Alpha, tol, trials, n)
	} else {
		res.Finding = fmt.Sprintf("MISSED — empirical power %.3f (target %.3f) / empirical alpha %.3f (target %.3f) over %d trials at n=%d misses within tol %.3f",
			res.EmpiricalPower, s.Power, res.EmpiricalAlpha, s.Alpha, trials, n, tol)
	}
	return res
}

// StochasticVerdict folds a power assessment and an observed regression flag into the package's
// OK | REGRESSED | INSUFFICIENT vocabulary. The load-bearing rule: an UNDERPOWERED (or INVALID)
// run is INSUFFICIENT and NOT conclusive no matter what it observed — a clean underpowered run
// is never a verified pass. ok follows the package's fall-open convention (INSUFFICIENT does not
// red CI), but conclusive=false records that no positive claim was earned. Only an adequately
// powered run yields a real OK (clean) or REGRESSED (dirty).
func StochasticVerdict(rep PowerReport, regressed bool) (verdict string, ok, conclusive bool) {
	if !rep.Adequate {
		return "INSUFFICIENT", true, false
	}
	if regressed {
		return "REGRESSED", false, true
	}
	return "OK", true, true
}
