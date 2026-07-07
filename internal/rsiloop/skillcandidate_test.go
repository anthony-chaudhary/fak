package rsiloop

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// TestSkillCandidateNetNegativeCannotPromote is the #2872 honesty witness: a
// proposed skill that does not beat baseline on the held fixture corpus CANNOT be
// promoted, no matter how clean its suite/truth witness — and is reverted with the
// structured ReasonUnwitnessed. A skill that DOES show a strict measured gain
// (suite-green, truth-clean) is promoted. This is propose -> witness -> keep/revert
// applied to skill creation: no skill enters the active set on an unmeasured claim.
func TestSkillCandidateNetNegativeCannotPromote(t *testing.T) {
	// A frozen fixture corpus scored higher = better (task success rate). The
	// candidate REGRESSES two of three held tasks: its mean is below baseline.
	netNegative := SkillCandidate{
		Skill:      "over-eager-refactor",
		TruthClean: true, // a clean witness — the gate must STILL refuse it
		Corpus: SkillFixtureCorpus{
			Metric: "task_success_rate",
			Tasks: []SkillFixtureTask{
				{Name: "t1", BaselineScore: 0.80, CandidateScore: 0.55},
				{Name: "t2", BaselineScore: 0.70, CandidateScore: 0.60},
				{Name: "t3", BaselineScore: 0.90, CandidateScore: 0.90},
			},
		},
	}
	got := PromoteSkill(netNegative)
	if got.Promoted {
		t.Fatalf("net-negative skill was promoted; the keep-bit must refuse it (witness before=%v after=%v)", got.Witness.Before, got.Witness.After)
	}
	if got.Decision != shipgate.REVERT {
		t.Fatalf("net-negative skill decision = %v, want REVERT", got.Decision)
	}
	if got.Reason.Kind != ReasonUnwitnessed || !got.Reason.Valid() {
		t.Fatalf("net-negative revert reason = %+v, want a valid ReasonUnwitnessed", got.Reason)
	}

	// The honesty edge: a FLAT skill (candidate == baseline) shows no STRICT gain,
	// so it is not promoted either — "no worse" is not "better."
	flat := netNegative
	flat.Skill = "no-op-skill"
	flat.Corpus.Tasks = []SkillFixtureTask{
		{Name: "t1", BaselineScore: 0.8, CandidateScore: 0.8},
		{Name: "t2", BaselineScore: 0.7, CandidateScore: 0.7},
	}
	if f := PromoteSkill(flat); f.Promoted {
		t.Fatalf("a flat (no strict gain) skill was promoted; strict gain is required")
	}

	// A genuinely improving skill — strict gain, suite-green, truth-clean — IS
	// promoted, and carries no revert reason.
	improving := SkillCandidate{
		Skill:      "sharper-grep",
		TruthClean: true,
		Corpus: SkillFixtureCorpus{
			Metric: "task_success_rate",
			Tasks: []SkillFixtureTask{
				{Name: "t1", BaselineScore: 0.60, CandidateScore: 0.85},
				{Name: "t2", BaselineScore: 0.70, CandidateScore: 0.80},
				{Name: "t3", BaselineScore: 0.75, CandidateScore: 0.90},
			},
		},
	}
	w := PromoteSkill(improving)
	if !w.Promoted || w.Decision != shipgate.KEEP {
		t.Fatalf("improving skill: promoted=%v decision=%v, want promoted/KEEP", w.Promoted, w.Decision)
	}
	if w.Reason.Kind != "" {
		t.Fatalf("a promoted skill carried a revert reason %+v; it must be zero", w.Reason)
	}
}

// TestSkillCandidateDirtyWitnessCannotPromote proves the other two keep-bit floors:
// even a skill that beats baseline is NOT promoted if its witness is dirty — a
// suite-red run (it broke a held task) or a truth-dirty apply. A strict gain alone
// is not enough to enter the active set. An empty corpus (no held task) is likewise
// unpromotable: there is nothing to witness a gain against.
func TestSkillCandidateDirtyWitnessCannotPromote(t *testing.T) {
	base := SkillFixtureCorpus{
		Metric: "task_success_rate",
		Tasks: []SkillFixtureTask{
			{Name: "t1", BaselineScore: 0.6, CandidateScore: 0.9},
			{Name: "t2", BaselineScore: 0.6, CandidateScore: 0.9},
		},
	}

	// Strict gain but the candidate ERRORED a held task -> suite-red -> not promoted.
	suiteRed := base
	suiteRed.Tasks = append([]SkillFixtureTask(nil), base.Tasks...)
	suiteRed.Tasks[1].Errored = true
	if got := PromoteSkill(SkillCandidate{Skill: "breaks-a-task", Corpus: suiteRed, TruthClean: true}); got.Promoted {
		t.Fatalf("a suite-red skill (broke a held task) was promoted; suite-green is required")
	}

	// Strict gain, suite-green, but a truth-dirty apply -> not promoted.
	if got := PromoteSkill(SkillCandidate{Skill: "truth-dirty", Corpus: base, TruthClean: false}); got.Promoted {
		t.Fatalf("a truth-dirty skill was promoted; truth-clean is required")
	}

	// No held corpus at all -> no witness -> not promoted, reverted as unwitnessed.
	empty := PromoteSkill(SkillCandidate{Skill: "no-corpus", TruthClean: true})
	if empty.Promoted || empty.Reason.Kind != ReasonUnwitnessed {
		t.Fatalf("empty-corpus skill: promoted=%v reason=%+v, want reverted/ReasonUnwitnessed", empty.Promoted, empty.Reason)
	}
}

// TestSkillPromotionJournalsRevertWithStructuredReason wires the skill-candidate
// gate into the per-decision curator ledger (#2841): a reverted candidate is
// journaled under ReasonUnwitnessed so the read-path answers "why was this skill not
// promoted?" from the journal alone, distinct from slop_scored; a promoted candidate
// journals nothing; and a per-decision revert restores exactly the one skill.
func TestSkillPromotionJournalsRevertWithStructuredReason(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curator.jsonl")
	l, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	reverted := PromoteSkill(SkillCandidate{
		Skill:      "unmeasured-skill",
		TruthClean: true,
		Corpus: SkillFixtureCorpus{Metric: "task_success_rate", Tasks: []SkillFixtureTask{
			{Name: "t1", BaselineScore: 0.9, CandidateScore: 0.4},
		}},
	})
	seq, err := reverted.Journal(l)
	if err != nil {
		t.Fatalf("journal reverted skill: %v", err)
	}
	if seq == 0 {
		t.Fatalf("a reverted skill was not journaled (seq 0)")
	}
	r, gone := l.Why("unmeasured-skill")
	if !gone || r.Kind != ReasonUnwitnessed {
		t.Fatalf("Why(unmeasured-skill) = %+v gone=%v, want ReasonUnwitnessed/gone", r, gone)
	}

	// The unwitnessed token must not collapse into slop_scored — distinct reasons an
	// operator acts on differently (never-promoted vs low-quality once promoted).
	if _, err := l.Archive("sloppy", CuratorReason{Kind: ReasonSlopScored, SlopScore: 0.8}); err != nil {
		t.Fatalf("archive sloppy: %v", err)
	}
	if r.String() == mustWhy(t, l, "sloppy") {
		t.Fatalf("unwitnessed and slop_scored rendered identically (%q) — reasons collapsed", r.String())
	}

	// A promoted skill journals nothing — there is no revert to record.
	promoted := PromoteSkill(SkillCandidate{
		Skill:      "witnessed-skill",
		TruthClean: true,
		Corpus: SkillFixtureCorpus{Metric: "task_success_rate", Tasks: []SkillFixtureTask{
			{Name: "t1", BaselineScore: 0.5, CandidateScore: 0.9},
		}},
	})
	if s, err := promoted.Journal(l); err != nil || s != 0 {
		t.Fatalf("promoted skill journaled seq=%d err=%v, want no-op (0, nil)", s, err)
	}

	// Per-decision revert restores exactly the unmeasured skill; sibling stays gone.
	if err := l.Revert(seq); err != nil {
		t.Fatalf("revert: %v", err)
	}
	if _, gone := l.Why("unmeasured-skill"); gone {
		t.Fatalf("unmeasured-skill should be restored after per-decision revert")
	}
	if _, gone := l.Why("sloppy"); !gone {
		t.Fatalf("sibling sloppy was rolled back by the revert")
	}
}
