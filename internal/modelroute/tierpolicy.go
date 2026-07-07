package modelroute

import (
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelscore"
)

// ---------------------------------------------------------------------------
// THE TIER POLICY — turn a work class + a model's RAW capability evidence into a
// tier decision, WITHOUT ever letting a benchmark score buy past a risk floor.
// ---------------------------------------------------------------------------
//
// C1 (internal/modelscore) stores raw, unbounded capability evidence and refuses
// to interpret it. This file is the interpretation: the score-vector POLICY that
// maps a model onto the tiers of WORK it may take (#3040). It is the missing
// middle of the working path — score evidence -> THIS -> dispatch choice (C5).
//
// TWO ORTHOGONAL AXES, deliberately kept apart:
//
//   - THE WORK CLASS sets a RISK FLOOR. An ultra-hard task, and any task that can
//     push / delete / release / touch a policy or security surface, requires a
//     high tier NO MATTER how small it looks or how well a cheap model benchmarks.
//     The floor is a property of the WORK, and a model's score never lowers it.
//
//   - THE MODEL'S CAPABILITY sets a CEILING. Raw evidence (FrontierSWE, SWE-bench,
//     Terminal-Bench) folds into the most demanding tier the model can serve. A
//     model may take work at or below its capability; it is refused work above it.
//
// A choice is admitted only when the model's capability MEETS the work's required
// floor. Over-tier (a frontier model on routine work) is WASTE — allowed, flagged.
// Under-tier (a cheap model on high-risk work) is RISK — refused, with a closed
// reason string. This asymmetry is the whole point.
//
// THE NUMBERING TRAP (load-bearing): T0 is the MOST demanding tier but the LOWEST
// number, so capability order runs OPPOSITE to the label numbers. Every tier
// comparison goes through WorkTier.MeetsRequirement — never a raw `<` — so the
// inversion can never leak into a silent mis-route. This is exactly the confusion
// #3040 warns about.

// WorkTier is the difficulty/risk tier of a unit of WORK — not a model tier. T0
// is the most demanding (ultra-hard, high-ambiguity, high-risk, or score-
// sensitive), T1 is normal implementation, T2 is routine/bounded/low-trust.
type WorkTier int

const (
	TierT0 WorkTier = iota // ultra-hard / high-risk — the MOST demanding
	TierT1                 // normal implementation
	TierT2                 // routine / bounded / low-trust
)

// String renders the canonical T0/T1/T2 label.
func (t WorkTier) String() string {
	switch t {
	case TierT0:
		return "T0"
	case TierT1:
		return "T1"
	case TierT2:
		return "T2"
	default:
		return "T?"
	}
}

// Valid reports whether t is one of the three defined tiers.
func (t WorkTier) Valid() bool { return t >= TierT0 && t <= TierT2 }

// workTierTokenRE extracts a T<N> tier token from a value like "T1",
// "`tier/T1-required`", or "tier/t2-optimal". The leading 't' is MANDATORY, so a
// stray "P1" (a priority, not a tier) never parses as a tier — the same
// disambiguation the issue-label grammar bakes in.
var workTierTokenRE = regexp.MustCompile(`t([0-9]+)`)

// ParseWorkTier parses a canonical tier token into a WorkTier, tolerating
// surrounding whitespace/backticks and a "tier/T1-required"-style wrapper (the
// T<N> token is extracted). It is the inverse of String() and the one home for the
// string form of the tier vocabulary. ok=false for an absent or OUT-OF-RANGE token
// (e.g. "T3", "P1", "") — the caller decides whether that is a missing or an
// invalid tier.
func ParseWorkTier(s string) (WorkTier, bool) {
	m := workTierTokenRE.FindStringSubmatch(strings.ToLower(s))
	if m == nil {
		return 0, false
	}
	switch m[1] {
	case "0":
		return TierT0, true
	case "1":
		return TierT1, true
	case "2":
		return TierT2, true
	default: // a real T<N> token, but not a defined work tier (T3+)
		return 0, false
	}
}

// MeetsRequirement reports whether a model whose capability tops out at tier
// `capability` can serve work whose required floor is `required`. Capability must
// be at least as DEMANDING as the floor. Because T0<T1<T2 numerically but T0 is
// the MOST demanding, "at least as demanding" is `capability <= required`. Every
// tier comparison in this package routes through here so the label/number
// inversion can never leak into a raw `<`.
func (capability WorkTier) MeetsRequirement(required WorkTier) bool {
	return capability <= required
}

// MoreDemandingThan reports whether tier a is strictly more demanding than b
// (a is T0 where b is T1, say) — i.e. using a for b's work is over-tier waste.
func (a WorkTier) MoreDemandingThan(b WorkTier) bool { return a < b }

// WorkClass is the KIND of work, which fixes its risk floor independent of how
// any model scores. It is a closed vocabulary; an unrecognized class is treated
// conservatively (the highest floor) rather than silently down-tiered.
type WorkClass string

const (
	// ClassUltraHard is FrontierSWE-like ultra-long-horizon / high-ambiguity work.
	ClassUltraHard WorkClass = "ultra-hard"
	// ClassNormalImpl is ordinary feature/bugfix implementation.
	ClassNormalImpl WorkClass = "normal-impl"
	// ClassRoutine is bounded, low-trust-impact work (watchdog, status, scoped edits).
	ClassRoutine WorkClass = "routine"
	// ClassSecurityRelease is anything that can push, delete, release, or modify a
	// policy/security surface. Its floor never drops to T2 however small it looks.
	ClassSecurityRelease WorkClass = "security-release-destructive"
)

// Closed-vocabulary reason strings. A status surface renders these verbatim so a
// refusal or a waste flag is explainable without free text.
const (
	ReasonOptimalMatch     = "optimal-tier-match"      // capability equals the optimal tier
	ReasonMeetsRequired    = "meets-required-tier"     // capability meets the floor (at/above)
	ReasonOverTierWaste    = "over-tier-waste"         // capability exceeds optimal — allowed, wasteful
	ReasonUnderTierRefused = "under-tier-refused"      // capability below the floor — refused
	ReasonSecurityFloor    = "security-floor-enforced" // floor raised because work is security/release/destructive
	ReasonUnknownClass     = "unknown-class-conservative"
)

// TierPolicy is the tier decision for a work class: the required floor, the
// best-fit optimal tier, the fallbacks allowed above optimal, and the closed
// reason strings that explain the floor. It carries NO model — it is the policy
// for the WORK, which a model is then admitted against.
type TierPolicy struct {
	Class            WorkClass  `json:"class"`
	RequiredTier     WorkTier   `json:"required_tier"`
	OptimalTier      WorkTier   `json:"optimal_tier"`
	AllowedFallbacks []WorkTier `json:"allowed_fallbacks"`
	Reasons          []string   `json:"reasons"`
}

// PolicyFor returns the tier policy for a work class. The mapping is the risk
// floor, fixed by the class alone:
//
//   - ultra-hard                     -> required T0, optimal T0
//   - normal-impl                    -> required T1, optimal T1
//   - routine                        -> required T2, optimal T2
//   - security-release-destructive   -> required T1 (floor never T2), optimal T0
//
// AllowedFallbacks are the tiers MORE demanding than optimal (a more capable
// model may always stand in for a less demanding target — over-tier waste, never
// risk). No fallback is ever less demanding than RequiredTier.
func PolicyFor(class WorkClass) TierPolicy {
	switch class {
	case ClassUltraHard:
		return newPolicy(class, TierT0, TierT0, nil)
	case ClassNormalImpl:
		return newPolicy(class, TierT1, TierT1, nil)
	case ClassRoutine:
		return newPolicy(class, TierT2, TierT2, nil)
	case ClassSecurityRelease:
		// Optimal is the frontier (T0), but the FLOOR is T1: a security/release/
		// destructive task must never fall to routine T2 even when it looks small.
		return newPolicy(class, TierT1, TierT0, []string{ReasonSecurityFloor})
	default:
		// Unknown class: stay conservative at the highest floor rather than infer
		// a cheap tier (the C5 "missing tier stays conservative" assumption).
		return newPolicy(class, TierT0, TierT0, []string{ReasonUnknownClass})
	}
}

// newPolicy builds a policy and derives AllowedFallbacks as every tier strictly
// more demanding than optimal (and thus at or above the required floor).
func newPolicy(class WorkClass, required, optimal WorkTier, extraReasons []string) TierPolicy {
	var fallbacks []WorkTier
	for t := TierT0; t.Valid(); t++ {
		if t.MoreDemandingThan(optimal) && t.MeetsRequirement(required) {
			fallbacks = append(fallbacks, t)
		}
	}
	sort.Slice(fallbacks, func(i, j int) bool { return fallbacks[i] < fallbacks[j] })
	return TierPolicy{
		Class:            class,
		RequiredTier:     required,
		OptimalTier:      optimal,
		AllowedFallbacks: fallbacks,
		Reasons:          append([]string(nil), extraReasons...),
	}
}

// TierChoice is the decision for ONE model against a policy: whether it is
// admitted, why (closed reason strings), and the policy tiers it was judged
// against. It NEVER admits below the required floor.
type TierChoice struct {
	Model        string    `json:"model"`
	Capability   WorkTier  `json:"capability"`
	Class        WorkClass `json:"class"`
	RequiredTier WorkTier  `json:"required_tier"`
	OptimalTier  WorkTier  `json:"optimal_tier"`
	Admitted     bool      `json:"admitted"`
	OverTier     bool      `json:"over_tier"`
	Reasons      []string  `json:"reasons"`
}

// Admit judges a model of the given capability against the policy. It returns
// admitted=false with ReasonUnderTierRefused when the capability does not meet
// the required floor — a high score cannot rescue an under-tier choice, because
// capability, not score, is compared here and the floor is fixed by the class.
// An over-capable model is admitted with ReasonOverTierWaste (waste, not risk).
func (p TierPolicy) Admit(model string, capability WorkTier) TierChoice {
	c := TierChoice{
		Model:        model,
		Capability:   capability,
		Class:        p.Class,
		RequiredTier: p.RequiredTier,
		OptimalTier:  p.OptimalTier,
	}
	c.Reasons = append(c.Reasons, p.Reasons...)
	if !capability.MeetsRequirement(p.RequiredTier) {
		c.Admitted = false
		c.Reasons = append(c.Reasons, ReasonUnderTierRefused)
		return c
	}
	c.Admitted = true
	switch {
	case capability == p.OptimalTier:
		c.Reasons = append(c.Reasons, ReasonOptimalMatch)
	case capability.MoreDemandingThan(p.OptimalTier):
		c.OverTier = true
		c.Reasons = append(c.Reasons, ReasonOverTierWaste)
	default:
		c.Reasons = append(c.Reasons, ReasonMeetsRequired)
	}
	return c
}

// ---------------------------------------------------------------------------
// SCORE-VECTOR -> CAPABILITY. The one blend this file owns: fold a model's raw
// capability evidence (from C1) into the most demanding work tier it can serve.
// ---------------------------------------------------------------------------
//
// This is a ROUGH, transparent, overridable ladder, not a truth. It reads the
// long-horizon and implementation benchmarks a work tier is sensitive to and
// picks the highest tier the model clears. Thresholds are documented constants,
// and an ILLUSTRATIVE-only score (a fixture placeholder, never a measurement) is
// NOT allowed to lift a model above the routine floor — fak never promotes a
// model to hard work on an invented number.

// CapabilityLadder is the rough score->tier ladder. A model reaches a tier when
// it clears that tier's threshold on the named benchmark (native units). The
// ladder is data so a caller can override it; DefaultCapabilityLadder is the
// built-in.
type CapabilityLadder struct {
	// T0Benchmark / T0Min: the long-horizon benchmark and minimum native score a
	// model must clear to be T0-capable (ultra-hard work).
	T0Benchmark string
	T0Min       float64
	// T1Benchmark / T1Min: the implementation benchmark and minimum score for
	// T1-capable (normal implementation).
	T1Benchmark string
	T1Min       float64
}

// DefaultCapabilityLadder is the built-in rough ladder, anchored to the
// benchmarks C1's fixture carries. The thresholds are deliberately illustrative
// order-of-magnitude cut points, overridable, never a published bar.
func DefaultCapabilityLadder() CapabilityLadder {
	return CapabilityLadder{
		T0Benchmark: "frontier-swe", T0Min: 10.0,
		T1Benchmark: "swe-bench-verified", T1Min: 40.0,
	}
}

// CapabilityFromProfile folds a model's raw capability profile into the most
// demanding WorkTier it can serve, using the ladder. It returns the tier and the
// closed reason strings that justify it. A model with no clearing evidence, or
// whose only clearing evidence is ILLUSTRATIVE, floors at T2 (routine) — an
// invented score never promotes a model to hard work.
func CapabilityFromProfile(p modelscore.Profile, ladder CapabilityLadder) (WorkTier, []string) {
	if clearsMeasured(p, ladder.T0Benchmark, ladder.T0Min) {
		return TierT0, []string{"cleared:" + ladder.T0Benchmark}
	}
	if clearsMeasured(p, ladder.T1Benchmark, ladder.T1Min) {
		return TierT1, []string{"cleared:" + ladder.T1Benchmark}
	}
	return TierT2, []string{"below-implementation-floor-or-illustrative-only"}
}

// clearsMeasured reports whether the profile has a MEASURED (non-illustrative)
// row on the named benchmark at or above min. An illustrative fixture row does
// not count toward a promotion.
func clearsMeasured(p modelscore.Profile, benchmark string, min float64) bool {
	b, ok := p.Benchmark(benchmark)
	if !ok || b.Provenance.Illustrative {
		return false
	}
	return b.Score >= min
}
