package ablate

// The MODEL-STRENGTH axis of the self-ablation sweep (#4412, epic #4396, under
// self-ablation #607 / open ablation registry #2828).
//
// THE QUESTION. `fak ablate` measures each fak-feature's marginal delta at ONE
// implicit model strength. A scaffold the model still NEEDS and a scaffold the model
// has OUTGROWN both read as "a positive delta" there, so the report cannot tell them
// apart — and a feature that HELPS a weak model while HURTING a strong one averages
// out to "fine". This file adds the missing dimension: replay every arm across a
// weak -> mid -> strong ladder, record the marginal delta at each rung, and fold the
// trajectory into ONE closed verdict per feature.
//
// THE SEAM. TierScorer mirrors the armRunner/ExecArmRunner swap that already makes
// the rung-2 subprocess path testable: production binds StubTierScorer, a test injects
// a fake with fixed per-tier outcomes, so the classifier is deterministic and $0 in CI.
//
// THE HONESTY FENCE. Live per-tier measurement (a real model call per rung) is a LATER
// rung and is deliberately NOT wired here. StubTierScorer measures nothing and SAYS so,
// and a trajectory with no measured rung classifies UNMEASURED — never a fabricated
// LOAD_BEARING. The verdict is only as strong as the scorer behind it.
//
// THE CONSUMER. `fak harness-debt-dispatch` (#4414) already ships as the deletion half
// of this pair: it reads a verdict payload's rows and files one retirement issue per
// HARD scaffold graded REDUNDANT or HOBBLING (a LOAD_BEARING scaffold files nothing).
// So the report also writes flat verdict rows under the top-level "verdicts" key that
// reader accepts, using the underscore grade spelling its own constants carry — the
// producer matches the already-shipped consumer's contract rather than making it adapt.

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// The model-strength ladder, WEAKEST FIRST. The order is the load-bearing part: the
// classifier reads the STRONGEST rung, so a caller passing `--models strong,weak` must
// still be classified on `strong`. ParseModelTiers therefore re-sorts any spec into
// this canonical order rather than trusting the order it was typed in.
const (
	TierWeak   = "weak"
	TierMid    = "mid"
	TierStrong = "strong"
)

// modelTierLadder pins the weak -> strong ordering the classifier depends on.
var modelTierLadder = []string{TierWeak, TierMid, TierStrong}

// KnownModelTiers returns the ladder for `--models` help text and validation, weakest
// first. It returns a COPY: the ladder is the classifier's ordering invariant, and a
// caller mutating the help-text slice must not be able to reorder what "strongest"
// means.
func KnownModelTiers() []string {
	return append([]string(nil), modelTierLadder...)
}

// The CLOSED verdict vocabulary. The underscore spelling is the one the shipped
// #4414 router compares against (it upper-cases and maps "-" to "_" on read, so the
// hyphenated "LOAD-BEARING" prose spelling normalizes to the same token). Kept
// general on purpose: the per-skill strength sweep (#2873) reuses this enum and the
// TierScorer seam rather than forking its own.
const (
	// VerdictLoadBearing — the feature still pulls weight at the strongest tier.
	VerdictLoadBearing = "LOAD_BEARING"
	// VerdictRedundant — the strongest model has erased the feature's contribution.
	VerdictRedundant = "REDUNDANT"
	// VerdictHobbling — the strongest model is measurably BETTER without the feature.
	VerdictHobbling = "HOBBLING"
	// VerdictUnmeasured — no tier reported a measured outcome, so there is nothing to
	// classify. It is a REFUSAL, not a fourth grade: the debt router ignores it, which
	// is exactly right, because a stub scorer must never file a deletion issue.
	VerdictUnmeasured = "UNMEASURED"
)

// The documented default cut points. Scores are scorer-defined but expected to be
// task-outcome rates in [0,1], so these read as "one point of outcome is noise, five
// points is real". Live-calibrated values are a later concern (#4412 ships the seam,
// not the tuning policy).
const (
	// DefaultStrengthEpsilon is the REDUNDANCY FLOOR: a strongest-tier delta below it
	// is indistinguishable from no contribution at all.
	DefaultStrengthEpsilon = 0.01
	// DefaultStrengthThreshold is the SUSTAINED BAR: the delta a feature must still
	// carry at EVERY measured tier to be reported as sustained across the ladder.
	DefaultStrengthThreshold = 0.05
)

// StrengthParams carries the classifier's two cut points. A zero value takes the
// documented defaults, so callers that do not tune anything pass StrengthParams{}.
type StrengthParams struct {
	Epsilon   float64 `json:"epsilon"`
	Threshold float64 `json:"threshold"`
}

func (p StrengthParams) withDefaults() StrengthParams {
	if p.Epsilon <= 0 {
		p.Epsilon = DefaultStrengthEpsilon
	}
	if p.Threshold <= 0 {
		p.Threshold = DefaultStrengthThreshold
	}
	return p
}

// TierOutcome is what a TierScorer reports for ONE arm at ONE tier. Measured is the
// honesty bit: a scorer that cannot actually measure this rung returns Measured=false
// with a Detail naming why, and the classifier refuses to grade off it instead of
// treating an absent score as a zero delta.
type TierOutcome struct {
	Score    float64 `json:"score"`
	Measured bool    `json:"measured"`
	Detail   string  `json:"detail,omitempty"`
}

// TierScorer is the pluggable, replayable per-tier outcome seam — the model-strength
// twin of the armRunner injection point. It takes the arm's id and the feature
// descriptor it ran under (both already on AblationRun), so it also works over a
// report LOADED from disk, not just one assembled in this process.
type TierScorer interface {
	ScoreArm(ctx context.Context, tier, armID string, features map[string]string) (TierOutcome, error)
}

// StubTierScorer is the DEFAULT TierScorer and it measures NOTHING. Live per-tier
// measurement means a real model call per rung, which is a later leaf; until that
// lands, production must classify UNMEASURED rather than invent a trajectory. This is
// the whole reason TierOutcome carries Measured: the stub is honest by construction
// and cannot be mistaken for a $0 measurement.
type StubTierScorer struct{}

// ScoreArm reports an explicitly unmeasured outcome for every (tier, arm).
func (StubTierScorer) ScoreArm(context.Context, string, string, map[string]string) (TierOutcome, error) {
	return TierOutcome{
		Detail: "no live per-tier measurement is wired: #4412 ships the classifier seam and the fake-scorer witness; the live rung is a later leaf",
	}, nil
}

// compile-time assertion: the stub satisfies the seam production binds it to.
var _ TierScorer = StubTierScorer{}

// TierDelta is one rung of an arm's strength trajectory: the arm's score at that tier,
// the baseline arm's score at the SAME tier, and the marginal delta between them.
type TierDelta struct {
	Tier          string  `json:"tier"`
	Score         float64 `json:"score"`
	BaselineScore float64 `json:"baseline_score"`
	Delta         float64 `json:"delta"`
	Measured      bool    `json:"measured"`
	Detail        string  `json:"detail,omitempty"`
}

// ModelStrength is the per-arm strength-axis card written into the AblationReport. It
// carries the verdict AND the trajectory it was computed from, so a reader can always
// re-derive the grade instead of taking it on faith.
type ModelStrength struct {
	Verdict        string      `json:"verdict"`
	StrongestTier  string      `json:"strongest_tier,omitempty"`
	StrongestDelta float64     `json:"strongest_delta"`
	Sustained      bool        `json:"sustained"`
	Epsilon        float64     `json:"epsilon"`
	Threshold      float64     `json:"threshold"`
	Reason         string      `json:"reason"`
	Tiers          []TierDelta `json:"tiers,omitempty"`
}

// StrengthVerdict is the FLAT per-feature row the report publishes alongside runs[] —
// the exact shape `fak harness-debt-dispatch` (#4414) reads: a stable id, the grade,
// the HARD/soft scaffold class it filters on, and the rationale it quotes into the
// deletion issue.
type StrengthVerdict struct {
	ID             string  `json:"id"`
	Grade          string  `json:"grade"`
	Hardness       string  `json:"hardness,omitempty"`
	StrongestTier  string  `json:"strongest_tier,omitempty"`
	StrongestDelta float64 `json:"strongest_delta"`
	Rationale      string  `json:"rationale,omitempty"`
}

// ParseModelTiers turns a `--models weak,mid,strong` spec into the canonical
// weak -> strong ladder. An EMPTY spec returns nil, which is what preserves the legacy
// single-tier output byte-for-byte: no tiers means the axis never runs. An unknown
// tier fails loud — a typo must not silently sweep a shorter ladder and hand back a
// verdict classified on the wrong "strongest" rung. Duplicates collapse.
func ParseModelTiers(spec string) ([]string, error) {
	seen := map[string]bool{}
	for _, raw := range strings.Split(spec, ",") {
		token := strings.ToLower(strings.TrimSpace(raw))
		if token == "" {
			continue
		}
		if !validModelTier(token) {
			return nil, fmt.Errorf("ablate: unknown model tier %q (known: %s)", token, strings.Join(modelTierLadder, ", "))
		}
		seen[token] = true
	}
	if len(seen) == 0 {
		return nil, nil
	}
	return canonicalTiers(seen), nil
}

// canonicalTiers collapses a SET of tier tokens into the weak -> strong ladder order the
// classifier depends on. Both readers of a tier set need it — the `--models` spec parser
// and the replay table's own ladder — and both need the SAME answer: whatever order the
// tiers arrived in, ClassifyStrength reads the last rung, so "strongest" has to mean the
// same thing on every path. Sharing the ordering keeps that invariant in one place.
func canonicalTiers(seen map[string]bool) []string {
	out := make([]string, 0, len(seen))
	for _, tier := range modelTierLadder {
		if seen[tier] {
			out = append(out, tier)
		}
	}
	return out
}

func validModelTier(token string) bool {
	for _, tier := range modelTierLadder {
		if tier == token {
			return true
		}
	}
	return false
}

// ClassifyStrength folds one arm's per-tier trajectory into a verdict.
//
// It classifies on the STRONGEST measured rung, never on the mean — averaging is the
// exact failure this axis exists to fix (a feature that carries a weak model and drags
// a strong one nets out to a healthy-looking positive). The three grades partition the
// measured space on that one delta:
//
//	HOBBLING      strongest delta <  0        the stronger model is BETTER without it
//	REDUNDANT     strongest delta <  Epsilon  the stronger model has erased it
//	LOAD_BEARING  strongest delta >= Epsilon  it still pulls weight at the top tier
//
// Threshold is the second, SEPARATE reading: Sustained is true when the delta clears
// it at EVERY measured rung, distinguishing a feature that holds up across the whole
// ladder from one that only survives at the top. It is reported as a field rather than
// a fourth grade, so the closed 3-verdict vocabulary the #4414 router and the #2873
// per-skill sibling share stays closed.
//
// A trajectory with no measured rung returns UNMEASURED with the reason — a refusal,
// not a grade.
func ClassifyStrength(tiers []TierDelta, p StrengthParams) ModelStrength {
	p = p.withDefaults()
	ms := ModelStrength{
		Verdict:   VerdictUnmeasured,
		Epsilon:   p.Epsilon,
		Threshold: p.Threshold,
		Tiers:     tiers,
	}
	strongest := -1
	for i := range tiers {
		if tiers[i].Measured {
			strongest = i
		}
	}
	if strongest < 0 {
		ms.Reason = "no tier reported a measured outcome, so there is no trajectory to classify (the default TierScorer is a stub; wire a measuring TierScorer for a real grade)"
		return ms
	}

	top := tiers[strongest]
	ms.StrongestTier = top.Tier
	ms.StrongestDelta = top.Delta
	ms.Sustained = true
	for _, td := range tiers {
		if td.Measured && td.Delta < p.Threshold {
			ms.Sustained = false
			break
		}
	}

	switch {
	case top.Delta < 0:
		ms.Verdict = VerdictHobbling
		ms.Reason = fmt.Sprintf("marginal delta at the strongest tier %q is %+.4f (< 0): the stronger model performs BETTER without this feature, so the scaffold now costs capability",
			top.Tier, top.Delta)
	case top.Delta < p.Epsilon:
		ms.Verdict = VerdictRedundant
		ms.Reason = fmt.Sprintf("marginal delta at the strongest tier %q is %+.4f, below the %.4f redundancy floor: the stronger model has erased this feature's contribution",
			top.Tier, top.Delta, p.Epsilon)
	default:
		ms.Verdict = VerdictLoadBearing
		ms.Reason = fmt.Sprintf("marginal delta at the strongest tier %q is %+.4f, at or above the %.4f redundancy floor (sustained >= %.4f across every measured tier: %t)",
			top.Tier, top.Delta, p.Epsilon, p.Threshold, ms.Sustained)
	}
	return ms
}

// AnnotateModelStrength runs the model-strength axis over an ALREADY-ASSEMBLED report:
// it scores every arm at every requested tier through scorer, computes each arm's
// marginal delta against the baseline arm at the SAME tier, classifies the trajectory,
// and writes the verdict onto the run plus a flat row into Report.Verdicts.
//
// An empty tier list is a no-op, which is what keeps `fak ablate` without `--models`
// byte-identical to its legacy single-tier output. A nil scorer takes the stub.
//
// The BASELINE arm is deliberately left ungraded: its delta against itself is 0 by
// construction, so grading it would mint a fabricated REDUNDANT for the reference arm
// the whole table is measured against. Only feature arms carry a verdict.
func AnnotateModelStrength(ctx context.Context, rep *Report, tiers []string, scorer TierScorer, p StrengthParams) error {
	if rep == nil {
		return fmt.Errorf("ablate: nil report")
	}
	if len(tiers) == 0 {
		return nil
	}
	if scorer == nil {
		scorer = StubTierScorer{}
	}
	baseline := rep.ArmByID(rep.Baseline)
	if baseline == nil {
		return fmt.Errorf("ablate: the model-strength axis needs a baseline arm to take each tier's marginal delta against (report baseline %q is not among the %d arms)",
			rep.Baseline, len(rep.Runs))
	}

	// Score the baseline ONCE per tier: every arm's delta at that tier is measured
	// against the same reference, the same way the single-tier table is.
	baseOutcome := make(map[string]TierOutcome, len(tiers))
	for _, tier := range tiers {
		out, err := scorer.ScoreArm(ctx, tier, baseline.ArmID, baseline.Features)
		if err != nil {
			return fmt.Errorf("ablate: score baseline arm %q at tier %q: %w", baseline.ArmID, tier, err)
		}
		baseOutcome[tier] = out
	}

	rep.Verdicts = nil
	for i := range rep.Runs {
		run := &rep.Runs[i]
		if run.ArmID == baseline.ArmID {
			continue
		}
		trajectory := make([]TierDelta, 0, len(tiers))
		for _, tier := range tiers {
			out, err := scorer.ScoreArm(ctx, tier, run.ArmID, run.Features)
			if err != nil {
				return fmt.Errorf("ablate: score arm %q at tier %q: %w", run.ArmID, tier, err)
			}
			base := baseOutcome[tier]
			// A rung counts as measured only when BOTH sides of the subtraction were
			// measured — a delta against an unmeasured baseline is not a measurement.
			measured := out.Measured && base.Measured
			td := TierDelta{
				Tier:          tier,
				Score:         out.Score,
				BaselineScore: base.Score,
				Measured:      measured,
				Detail:        strings.TrimSpace(out.Detail),
			}
			if measured {
				td.Delta = out.Score - base.Score
			}
			trajectory = append(trajectory, td)
		}
		ms := ClassifyStrength(trajectory, p)
		run.ModelStrength = &ms
		rep.Verdicts = append(rep.Verdicts, StrengthVerdict{
			ID:             run.ArmID,
			Grade:          ms.Verdict,
			Hardness:       strengthHardness(run.Features),
			StrongestTier:  ms.StrongestTier,
			StrongestDelta: ms.StrongestDelta,
			Rationale:      ms.Reason,
		})
	}
	sort.SliceStable(rep.Verdicts, func(a, b int) bool { return rep.Verdicts[a].ID < rep.Verdicts[b].ID })
	return nil
}

// strengthHardness grades an arm's scaffold class for the #4414 debt router. Every
// registered ablate concept is a CODE-LEVEL lever — an env var the kernel reads at
// process start, or an in-process setter (Register refuses a concept carrying
// neither) — so an arm that turns one on is a HARD scaffold, and retiring it is the
// code deletion #4414 files an issue for. An arm with no concept switched on is not a
// scaffold at all and grades empty, which that router reads as "not debt".
func strengthHardness(features map[string]string) string {
	for token, value := range features {
		if value != "on" {
			continue
		}
		if _, ok := registeredConcept(token); ok {
			return "HARD"
		}
	}
	return ""
}
