package rsiloop

// skillcandidate.go — #2872 (Hermes-inspiration epic #2871): witnessed skill
// creation. Hermes' background_review.py auto-creates skills on an LLM's say-so
// ("be ACTIVE — most sessions produce at least one skill update") with NO
// measurement that they help — a slop bias where a skill enters the active set on
// an unmeasured claim. fak already runs propose -> witness -> keep/revert
// (shipgate.Evaluate, the non-forgeable keep-bit). This file wires SKILL creation
// into that gate: a proposed skill is a SkillCandidate that must WITNESS a strict
// improvement on a held fixture corpus before it is promoted; a candidate that does
// not beat baseline is auto-REVERTED with a structured curator reason
// (ReasonUnwitnessed), never silently promoted. The honesty property — a
// net-negative skill cannot be promoted no matter how clean its suite/truth witness
// — is exactly what shipgate.Evaluate's STRICT-gain requirement enforces, and
// skillcandidate_test.go is its witness.
//
// This is the FIRST SLICE. It gates a proposed skill against a FROZEN, deterministic
// fixture corpus (the same discipline rulesynth_corpus.go uses so a KEEP reproduces
// bit-for-bit on any box). The live per-skill usage-vs-value ledger over real
// sessions is deliberately deferred to #2873; this promotion gate is the floor it
// builds on. It is distinct from skillvalue.go's #2842 detector: that flags a
// self-fulfilling skill ALREADY in the set from live use_count telemetry, while this
// decides whether a proposed skill ENTERS the set at all, against a held benchmark.

import "github.com/anthony-chaudhary/fak/internal/shipgate"

// SkillFixtureTask is one held benchmark task in the fixture corpus: the outcome
// score measured on that task WITHOUT the candidate skill (BaselineScore) and WITH
// it (CandidateScore), plus whether the candidate broke the task (Errored). The
// scores are measurements the candidate did NOT author — the witness the keep-bit
// folds — so a skill cannot promote itself by asserting its own value.
type SkillFixtureTask struct {
	Name           string  `json:"name"`
	BaselineScore  float64 `json:"baseline_score"`
	CandidateScore float64 `json:"candidate_score"`
	Errored        bool    `json:"errored,omitempty"`
}

// SkillFixtureCorpus is the held task/benchmark a SkillCandidate is witnessed
// against: a metric label, its direction, and the frozen task set. The corpus
// aggregate (mean task score) is the before/after the RSI keep/revert gate compares.
type SkillFixtureCorpus struct {
	Metric      string             `json:"metric"`
	LowerBetter bool               `json:"lower_better,omitempty"`
	Tasks       []SkillFixtureTask `json:"tasks"`
}

// baselineMetric and candidateMetric fold the corpus into the scalar KPI the keep-bit
// compares: the mean task score under baseline and under the candidate. An empty
// corpus yields 0 for both — which cannot show a strict gain, so an unbenchmarked
// skill is never promoted (the honest default: no corpus, no witness, no promotion).
func (c SkillFixtureCorpus) baselineMetric() float64  { return meanTaskScore(c.Tasks, false) }
func (c SkillFixtureCorpus) candidateMetric() float64 { return meanTaskScore(c.Tasks, true) }

// suiteGreen reports that the candidate broke NO held task — the suite-green signal
// the keep-bit requires. A candidate that errors any fixture task is suite-red, and
// an EMPTY corpus is suite-red too: there is no held task to witness a gain against.
func (c SkillFixtureCorpus) suiteGreen() bool {
	if len(c.Tasks) == 0 {
		return false
	}
	for _, t := range c.Tasks {
		if t.Errored {
			return false
		}
	}
	return true
}

func meanTaskScore(tasks []SkillFixtureTask, candidate bool) float64 {
	if len(tasks) == 0 {
		return 0
	}
	var sum float64
	for _, t := range tasks {
		if candidate {
			sum += t.CandidateScore
		} else {
			sum += t.BaselineScore
		}
	}
	return sum / float64(len(tasks))
}

// SkillCandidate is one proposed skill awaiting promotion: the skill's name, the
// held fixture corpus it must beat, and the truth-clean verdict derived from the
// isolated apply (dos verify) — a witness the candidate does not author. The
// keep/revert decision is computed from these by PromoteSkill; the candidate never
// gets to declare itself promoted.
type SkillCandidate struct {
	Skill      string             `json:"skill"`
	Corpus     SkillFixtureCorpus `json:"corpus"`
	TruthClean bool               `json:"truth_clean"`
}

// SkillPromotion is the keep/revert verdict for one candidate: the non-forgeable
// decision, whether it was promoted, the full witness the keep-bit folded, and — on
// a revert — the structured curator reason. Reason is the zero CuratorReason when
// Promoted is true (a promoted skill has nothing to revert).
type SkillPromotion struct {
	Skill    string            `json:"skill"`
	Decision shipgate.Decision `json:"-"`
	Promoted bool              `json:"promoted"`
	Witness  shipgate.Witness  `json:"witness"`
	Reason   CuratorReason     `json:"reason,omitzero"`
}

// PromoteSkill runs a SkillCandidate through the RSI keep/revert gate against its
// held fixture corpus. It folds the corpus baseline/candidate aggregate, the
// suite-green (no broken task) and the truth-clean signal through shipgate.Evaluate
// under ClassFull — so a skill is promoted ONLY on a STRICT measured gain that is
// also suite-green and truth-clean. A candidate that does not beat baseline (a
// net-negative or flat skill), or whose witness is dirty, is REVERTED with the
// structured ReasonUnwitnessed — never silently promoted. This is the honesty
// property #2872 ships: no skill enters the active set on an unmeasured claim.
func PromoteSkill(c SkillCandidate) SkillPromotion {
	w := shipgate.Witness{
		Class:       shipgate.ClassFull, // a skill entering the active set needs the full witness
		Metric:      c.Corpus.Metric,
		Before:      c.Corpus.baselineMetric(),
		After:       c.Corpus.candidateMetric(),
		LowerBetter: c.Corpus.LowerBetter,
		SuiteGreen:  c.Corpus.suiteGreen(),
		TruthClean:  c.TruthClean,
	}
	decision, witness := shipgate.Evaluate(w)
	p := SkillPromotion{
		Skill:    c.Skill,
		Decision: decision,
		Promoted: witness.Kept(), // the non-forgeable keep-bit, never the candidate's say-so
		Witness:  witness,
	}
	if !p.Promoted {
		p.Reason = CuratorReason{Kind: ReasonUnwitnessed}
	}
	return p
}

// Journal records a REVERTED candidate into the curator ledger so the per-decision
// revert read-path (Why / Log / Revert, #2841) can answer "why was this skill not
// promoted?" from the journal alone — the same structured, queryable substitute for
// Hermes' unmeasured whole-snapshot restore that the curator uses for archived
// skills. A promoted candidate is not journaled (it has nothing to revert): Journal
// returns seq 0 and no error. The reverted candidate is archived under
// ReasonUnwitnessed, distinct from the stale/superseded/slop/self-fulfilling tokens
// so an operator can tell "never beat baseline" from "decayed after promotion."
func (p SkillPromotion) Journal(l *CuratorLedger) (int, error) {
	if p.Promoted {
		return 0, nil
	}
	return l.Archive(p.Skill, p.Reason)
}
