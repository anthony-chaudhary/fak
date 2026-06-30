package rsiloop

// rulesynth.go is rung 2 of #537's loop made REAL against this engine: it wires
// internal/rulesynth's refusal-log rule SYNTHESIS into the propose -> measure ->
// keep/revert cycle this package already runs. internal/rsiloop's only other shipped
// Proposer (worktree.go) rewrites the DefaultCacheSize integer literal; this one's
// Candidate.Payload is a CANDIDATE STRUCTURAL RULE clustered from the near-miss corpus
// — turning #503's fixed-genome Deny-by-name hill-climb into a GENERATIVE one whose new
// alleles come from the kernel's own refusal log, and #172's manual hole-patching into
// a generated, gated diff.
//
// WHY THE WIRING IS THE INCREMENT. rulesynth already PROPOSES (Propose) and PROVES
// (Validate) standalone, but the issue asks specifically for "a Proposer whose
// Candidate.Payload is a candidate structural rule" — i.e. the synthesis driven by THIS
// engine's keep-bit, not a parallel one. This file is that Proposer: Candidates() are
// the clustered candidates, and Measure() replays each through the REAL adjudicator
// (model-free, zero model calls) and hands the engine the three raw witness fields. The
// keep-bit stays where it belongs — shipgate.Evaluate, folded by Run — so the loop, not
// the proposer, decides what lands. A candidate KEEPs only when it newly CATCHES a
// near-miss (Metric gain over a zero baseline) without REGRESSING a benign call
// (SuiteGreen) and catches its WHOLE cluster (TruthClean): exactly rulesynth.Validate's
// honesty gate, re-folded by the engine that owns the keep decision.
//
// IN-PROCESS, NOT A WORKTREE. Unlike NewWorktreeHarness, this harness needs no git
// fork: the corpus is frozen and the replay is a pure adjudicator call, so every seam
// is an in-memory closure. The baseline is the floor with NO synthesized rule, which by
// construction catches ZERO near-misses (a near-miss is, definitionally, a call the
// baseline ADMITTED) — so the baseline metric is 0 and any kept rule is a strict gain.
//
// LANDING IS STILL GATED. A KEEP here advances only the in-memory baseline and the
// journal, exactly as Run documents; it does NOT mutate the live policy. The reviewable
// artifact an operator lands is the candidate's rulesynth.Candidate.ManifestDiff(), and
// a candidate whose guarded trees intersect the harness (Candidate.SelfModify) must
// still route through the require-witness rung (#386/#387/#388) — the engine's keep-bit
// is a measurement, never landing authority.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/rulesynth"
)

// RuleSynthMetricName labels the KPI this harness drives in the journal: the count of
// refusal-log near-misses a synthesized rule newly catches.
const RuleSynthMetricName = "near_misses_caught"

// NewRuleSynthHarness wires a Harness that drives internal/rulesynth's rule synthesis
// through this engine. corpus is the frozen near-miss corpus mined from the refusal log
// (rulesynth.Detect / the stream Harvester); benign is the corpus of benign calls a
// candidate must not regress. Candidates() yields one structural-rule candidate per
// unrecognized-verb cluster (rulesynth.Propose), and Measure() proves each via the real
// model-free adjudicator replay (rulesynth.Validate), reporting the raw witness fields
// the engine folds into the non-forgeable keep-bit.
func NewRuleSynthHarness(corpus []rulesynth.NearMiss, benign []rulesynth.Call) Harness {
	return Harness{
		MetricName:      RuleSynthMetricName,
		LowerBetter:     false, // more near-misses caught is better
		BaselineRefName: "frozen-near-miss-corpus",
		BaselineMetric: func() (float64, string, error) {
			// The floor with no synthesized rule catches zero near-misses by
			// construction: a near-miss is a call the baseline ADMITTED. The ref is the
			// corpus identity, not a git SHA — this replay never forks a worktree.
			return 0, fmt.Sprintf("corpus@%d", len(corpus)), nil
		},
		Candidates: func() []Candidate {
			props := rulesynth.Propose(corpus)
			cs := make([]Candidate, 0, len(props))
			for _, p := range props {
				cs = append(cs, Candidate{
					Label:   "rule:" + p.Verb,
					Payload: p,
				})
			}
			return cs
		},
		Measure: func(c Candidate) (Measurement, error) {
			cand, ok := c.Payload.(rulesynth.Candidate)
			if !ok {
				return Measurement{}, fmt.Errorf("candidate payload is %T, want rulesynth.Candidate", c.Payload)
			}
			v, err := rulesynth.Validate(cand, corpus, benign)
			if err != nil {
				return Measurement{}, err
			}
			// Hand the engine the RAW measured fields; Run folds them into
			// shipgate.Evaluate (the same keep-bit rulesynth.Validate uses), so the
			// keep decision is computed once, by the engine that owns it.
			return Measurement{
				Metric:     float64(v.Caught),
				SuiteGreen: v.Regressed == 0,
				TruthClean: v.CatchesCluster,
				Score:      ruleSynthScorecard(cand, v),
				Note: fmt.Sprintf("verb=%q caught=%d regressed=%d catches_cluster=%v self_modify=%v",
					cand.Verb, v.Caught, v.Regressed, v.CatchesCluster, cand.SelfModify),
			}, nil
		},
	}
}

func ruleSynthScorecard(cand rulesynth.Candidate, v rulesynth.Verdict) *Scorecard {
	catchesCluster := 0.0
	if v.CatchesCluster {
		catchesCluster = 1
	}
	selfModify := 0.0
	if cand.SelfModify {
		selfModify = 1
	}
	grade := "clean"
	switch {
	case v.Regressed > 0:
		grade = "regressing"
	case !v.CatchesCluster:
		grade = "partial"
	case v.Caught == 0:
		grade = "no-catch"
	}
	return &Scorecard{
		Name:  RuleSynthMetricName,
		Value: float64(v.Caught),
		Grade: grade,
		Components: []ScoreComponent{
			{Name: "caught", Value: float64(v.Caught), Unit: "calls"},
			{Name: "regressed", Value: float64(v.Regressed), Unit: "calls"},
			{Name: "support", Value: float64(cand.Support), Unit: "calls"},
			{Name: "guarded_globs", Value: float64(len(cand.Globs)), Unit: "globs"},
			{Name: "catches_cluster", Value: catchesCluster, Unit: "bool"},
			{Name: "self_modify", Value: selfModify, Unit: "bool"},
		},
	}
}
