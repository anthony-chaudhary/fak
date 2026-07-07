package rsiloop

// skillcandidate_corpus.go — the FROZEN, deterministic fixture corpus #2872's
// promotion gate is witnessed against, committed as a reproducible held benchmark
// (the same discipline rulesynth_corpus.go uses so a KEEP reproduces bit-for-bit on
// any box) rather than living only as inline test data. A proposed skill must beat
// this held task set through PromoteSkill before it is promoted; because the corpus
// is frozen in source, the keep/revert verdict computed against it is identical on
// every host — the determinism the RSI keep-bit requires.

// DefaultSkillFixtureCorpus returns the frozen held benchmark a proposed skill is
// witnessed against: a small, fixed set of scored tasks with a stated baseline. It
// is a pure function of source text (never a live, box-dependent stream), so a
// candidate that beats it on one machine beats it on all of them. The returned
// corpus carries baseline-only scores; a caller overlays the measured candidate
// column with WithCandidateScores after running the skill over these tasks.
func DefaultSkillFixtureCorpus() SkillFixtureCorpus {
	return SkillFixtureCorpus{
		Metric: "task_success_rate",
		Tasks: []SkillFixtureTask{
			{Name: "grep-scoped-symbol", BaselineScore: 0.72},
			{Name: "refactor-preserves-tests", BaselineScore: 0.80},
			{Name: "commit-by-explicit-path", BaselineScore: 0.65},
			{Name: "witness-before-claim", BaselineScore: 0.78},
		},
	}
}

// WithCandidateScores returns a copy of the corpus with each task's CandidateScore
// set from scores keyed by task Name (an absent task keeps CandidateScore 0 — an
// unmeasured task is no gain, so an un-run candidate cannot be promoted) and Errored
// set for any task the skill run broke. This is how a caller turns the frozen
// baseline held benchmark into a witnessed candidate measurement without
// hand-authoring a whole corpus: the baseline column stays frozen, only the measured
// candidate column moves.
func (c SkillFixtureCorpus) WithCandidateScores(scores map[string]float64, errored map[string]bool) SkillFixtureCorpus {
	out := c
	out.Tasks = make([]SkillFixtureTask, len(c.Tasks))
	for i, t := range c.Tasks {
		t.CandidateScore = scores[t.Name]
		if errored[t.Name] {
			t.Errored = true
		}
		out.Tasks[i] = t
	}
	return out
}
