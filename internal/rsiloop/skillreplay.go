package rsiloop

// skillreplay.go — #2835 (part of Track A #2834): witness a proposed skill on a
// REPLAY OF THE SITUATION THAT PRODUCED IT, not only on a generic held corpus.
//
// #2872's skillcandidate.go already gates a proposed skill through the RSI keep/revert
// keep-bit against a FROZEN fixture corpus (DefaultSkillFixtureCorpus) — a generic held
// benchmark. #2835's ask is narrower and stronger: the Hermes background-review forks a
// skill out of a SPECIFIC session, so the honest witness is a replay of THAT situation —
// the skill must make the very case that motivated it succeed where the un-skilled
// baseline failed, or reach the same success with fewer turns/tokens. A skill that pays
// for itself on a generic corpus but not on the situation it was extracted from has not
// earned its keep on the evidence #2835 names.
//
// This file adds that situation-replay witness. It changes only the BENCHMARK a skill is
// scored against; it REUSES the exact same non-forgeable keep-bit (shipgate.Evaluate
// under ClassFull), the SkillPromotion verdict, and the curator revert vocabulary
// (ReasonUnwitnessed + SkillPromotion.Journal) that PromoteSkill already ships — so a
// replay-witnessed KEEP and a corpus-witnessed KEEP fold through one gate, and a
// replay-reverted candidate is journaled and revertable on the same read-path (#2841).
// It is the "what the review produces once spawned" complement to reviewgate.go (#2837),
// which gates only WHEN the review fires.

import "github.com/anthony-chaudhary/fak/internal/shipgate"

// SkillReplayMetric is the metric label the keep-bit records for a situation-replay
// witness, distinct from the fixture-corpus "task_success_rate" so a reader of the
// witness can tell WHICH harness (the producing-situation replay, not the generic held
// corpus) witnessed a promotion.
const SkillReplayMetric = "situation_replay_outcome"

// SkillReplayOutcome is one run of the producing situation: whether the task SUCCEEDED
// and the cost it took (Turns primary, Tokens a bounded tie-break). Errored marks a run
// that BROKE the harness — a suite-red signal — as distinct from a clean run that simply
// did not solve the task (Succeeded == false, Errored == false).
type SkillReplayOutcome struct {
	Succeeded bool   `json:"succeeded"`
	Turns     int    `json:"turns,omitempty"`
	Tokens    uint64 `json:"tokens,omitempty"`
	Errored   bool   `json:"errored,omitempty"`
}

// outcomeScore folds a replay run into the scalar, higher-is-better KPI the keep-bit
// compares. CORRECTNESS dominates COST: any success scores in (1.0, 2.0] and any failure
// scores 0.0, so a task that SUCCEEDS WHERE THE UN-SKILLED BASELINE FAILED is always a
// strict gain. Among two runs of equal success, LOWER cost scores strictly higher — so
// "fewer turns/tokens" is a strict gain too — with Turns primary and Tokens a bounded
// tie-break (a one-turn difference always outweighs any token difference, because the
// token term is confined to [0,1)). A failed run scores 0 regardless of its cost: a
// failure's cost is not a witness of anything.
func (o SkillReplayOutcome) outcomeScore() float64 {
	if !o.Succeeded {
		return 0
	}
	cost := float64(o.Turns) + boundedTokenFraction(o.Tokens)
	return 1.0 + 1.0/(1.0+cost)
}

// boundedTokenFraction maps a token count into [0,1) so the token cost can only ever
// break a tie between equal-Turn runs, never outweigh a whole turn. 0 tokens -> 0; the
// value approaches (but never reaches) 1 as the count grows.
func boundedTokenFraction(n uint64) float64 {
	return float64(n) / (float64(n) + 1)
}

// SkillSituationReplay is the situation that PRODUCED a candidate skill, replayed twice:
// once un-skilled (Baseline) and once with the candidate skill applied (Candidate), plus
// the truth-clean verdict from the isolated apply (a witness the candidate did not
// author). Situation is an id/label for the producing case so a journaled revert names
// WHICH situation failed to witness the skill.
type SkillSituationReplay struct {
	Skill      string             `json:"skill"`
	Situation  string             `json:"situation,omitempty"`
	Baseline   SkillReplayOutcome `json:"baseline"`
	Candidate  SkillReplayOutcome `json:"candidate"`
	TruthClean bool               `json:"truth_clean"`
}

// WitnessSkillReplay runs a candidate skill through the RSI keep/revert gate on a REPLAY
// of its producing situation. It folds the baseline/candidate outcome scores
// (correctness-dominant, cost tie-broken), the suite-green signal (the candidate did not
// break the harness) and the truth-clean signal through shipgate.Evaluate under
// ClassFull — so a skill is promoted ONLY on a STRICT improvement on the situation that
// produced it (a correctness flip, or fewer turns/tokens) that is also suite-green and
// truth-clean. Every other outcome is REVERTED with the structured ReasonUnwitnessed,
// never silently kept:
//
//   - no improvement (the candidate did not beat the baseline outcome),
//   - a regression (the baseline succeeded and the candidate failed — a negative witness,
//     which drops the score and so can never clear the strict-gain bit),
//   - no witness at all (baseline and candidate both fail — nothing to keep),
//   - a dirty witness (the candidate broke the harness, or the apply was truth-dirty).
//
// It reuses the same keep-bit, SkillPromotion verdict and curator vocabulary as
// PromoteSkill (#2872); it changes only the benchmark the skill is witnessed against.
func WitnessSkillReplay(r SkillSituationReplay) SkillPromotion {
	w := shipgate.Witness{
		Class:       shipgate.ClassFull, // a skill entering the active set needs the full witness
		Metric:      SkillReplayMetric,
		Before:      r.Baseline.outcomeScore(),
		After:       r.Candidate.outcomeScore(),
		LowerBetter: false, // outcomeScore is higher-is-better
		SuiteGreen:  !r.Candidate.Errored,
		TruthClean:  r.TruthClean,
	}
	decision, witness := shipgate.Evaluate(w)
	p := SkillPromotion{
		Skill:    r.Skill,
		Decision: decision,
		Promoted: witness.Kept(), // the non-forgeable keep-bit, never the candidate's say-so
		Witness:  witness,
	}
	if !p.Promoted {
		p.Reason = CuratorReason{Kind: ReasonUnwitnessed}
	}
	return p
}
