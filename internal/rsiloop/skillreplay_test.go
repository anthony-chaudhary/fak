package rsiloop

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// TestWitnessSkillReplayCorrectnessFlipPromotes is the #2835 primary witness: a skill
// that makes the producing situation SUCCEED WHERE THE UN-SKILLED BASELINE FAILED is a
// strict gain on the very case it was extracted from, so — suite-green and truth-clean —
// it is promoted through the same keep-bit PromoteSkill uses.
func TestWitnessSkillReplayCorrectnessFlipPromotes(t *testing.T) {
	flip := SkillSituationReplay{
		Skill:      "commit-by-explicit-path",
		Situation:  "session-that-swept-a-sibling",
		Baseline:   SkillReplayOutcome{Succeeded: false, Turns: 9},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 6},
		TruthClean: true,
	}
	got := WitnessSkillReplay(flip)
	if !got.Promoted || got.Decision != shipgate.KEEP {
		t.Fatalf("correctness-flip skill: promoted=%v decision=%v, want promoted/KEEP (before=%v after=%v)",
			got.Promoted, got.Decision, got.Witness.Before, got.Witness.After)
	}
	if got.Reason.Kind != "" {
		t.Fatalf("a promoted skill carried a revert reason %+v; it must be zero", got.Reason)
	}
}

// TestWitnessSkillReplayEfficiencyWinPromotes proves the second #2835 witness mode:
// among runs of EQUAL success, FEWER turns (and, at equal turns, fewer tokens) is a
// strict gain. A candidate that reaches the same outcome cheaper pays for itself.
func TestWitnessSkillReplayEfficiencyWinPromotes(t *testing.T) {
	fewerTurns := SkillSituationReplay{
		Skill:      "sharper-grep",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 8, Tokens: 4000},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 4000},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(fewerTurns); !g.Promoted || g.Decision != shipgate.KEEP {
		t.Fatalf("fewer-turns skill: promoted=%v decision=%v, want promoted/KEEP", g.Promoted, g.Decision)
	}

	fewerTokens := SkillSituationReplay{
		Skill:      "tighter-context",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 9000},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 3000},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(fewerTokens); !g.Promoted || g.Decision != shipgate.KEEP {
		t.Fatalf("fewer-tokens (equal turns) skill: promoted=%v decision=%v, want promoted/KEEP", g.Promoted, g.Decision)
	}
}

// TestWitnessSkillReplayTurnsDominateTokens proves the cost ordering #2835 names ("turns
// OR tokens"): a whole-turn saving outranks any token count, and a token saving that
// costs an extra turn is NOT a gain. Turns are primary; tokens only break a tie.
func TestWitnessSkillReplayTurnsDominateTokens(t *testing.T) {
	// One fewer turn but many MORE tokens -> still a strict gain (turns dominate).
	turnWinsDespiteTokens := SkillSituationReplay{
		Skill:      "one-fewer-turn",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 6, Tokens: 1000},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 50000},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(turnWinsDespiteTokens); !g.Promoted {
		t.Fatalf("a one-fewer-turn skill must promote even at higher token cost; got %+v", g.Witness)
	}

	// Fewer tokens but an EXTRA turn -> not a gain (turns dominate) -> reverted.
	tokenSavingCostsATurn := SkillSituationReplay{
		Skill:      "token-shaver-costs-a-turn",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 50000},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 6, Tokens: 1000},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(tokenSavingCostsATurn); g.Promoted {
		t.Fatalf("a token saving that costs a whole extra turn must NOT promote; got %+v", g.Witness)
	}
}

// TestWitnessSkillReplayRegressionAndNoWitnessRevert is the #2835 honesty witness: a
// skill that REGRESSES the situation (baseline succeeded, candidate failed) is reverted
// no matter how clean its suite/truth witness, and a situation neither run solves yields
// NO witness — both revert with the structured ReasonUnwitnessed. This distinguishes the
// three cases the contract's confusion-risk names: positive delta (keep) vs no-witness
// (revert) vs negative-witness (revert).
func TestWitnessSkillReplayRegressionAndNoWitnessRevert(t *testing.T) {
	regression := SkillSituationReplay{
		Skill:      "breaks-what-worked",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 5},
		Candidate:  SkillReplayOutcome{Succeeded: false, Turns: 3}, // failed even though cheaper
		TruthClean: true,
	}
	got := WitnessSkillReplay(regression)
	if got.Promoted || got.Decision != shipgate.REVERT {
		t.Fatalf("a regressing skill was not reverted: promoted=%v decision=%v", got.Promoted, got.Decision)
	}
	if got.Reason.Kind != ReasonUnwitnessed || !got.Reason.Valid() {
		t.Fatalf("regression revert reason = %+v, want a valid ReasonUnwitnessed", got.Reason)
	}

	// Neither run solves the situation -> no witness -> reverted.
	noWitness := SkillSituationReplay{
		Skill:      "helps-nothing",
		Baseline:   SkillReplayOutcome{Succeeded: false, Turns: 7},
		Candidate:  SkillReplayOutcome{Succeeded: false, Turns: 4},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(noWitness); g.Promoted {
		t.Fatalf("a skill that solves nothing was promoted on a cheaper-but-still-failing run; %+v", g.Witness)
	}

	// A FLAT replay (identical outcome, no cost change) shows no strict gain -> reverted.
	flat := SkillSituationReplay{
		Skill:      "no-op",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 2000},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 5, Tokens: 2000},
		TruthClean: true,
	}
	if g := WitnessSkillReplay(flat); g.Promoted {
		t.Fatalf("a flat replay (no strict gain) was promoted; strict gain is required")
	}
}

// TestWitnessSkillReplayDirtyWitnessCannotPromote proves the two remaining keep-bit
// floors carry over unchanged to the replay harness: even a real correctness flip is NOT
// promoted if the candidate BROKE the harness (suite-red) or the apply was truth-dirty.
func TestWitnessSkillReplayDirtyWitnessCannotPromote(t *testing.T) {
	base := SkillSituationReplay{
		Skill:      "real-gain-dirty-witness",
		Baseline:   SkillReplayOutcome{Succeeded: false, Turns: 8},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 5},
		TruthClean: true,
	}

	// Correctness flip but the candidate ERRORED the harness -> suite-red -> not promoted.
	suiteRed := base
	suiteRed.Candidate.Errored = true
	if g := WitnessSkillReplay(suiteRed); g.Promoted {
		t.Fatalf("a suite-red replay (candidate broke the harness) was promoted; suite-green is required")
	}

	// Correctness flip, suite-green, but a truth-dirty apply -> not promoted.
	truthDirty := base
	truthDirty.TruthClean = false
	if g := WitnessSkillReplay(truthDirty); g.Promoted {
		t.Fatalf("a truth-dirty replay was promoted; truth-clean is required")
	}
}

// TestWitnessSkillReplayJournalsRevertOnCuratorLedger proves the replay verdict reuses
// the #2841 curator read-path: a replay-reverted skill journals under ReasonUnwitnessed
// (so "why was this skill not promoted?" is answerable from the journal alone) and a
// replay-promoted skill journals nothing. This is the shared substrate #2835 reuses
// rather than rebuilding.
func TestWitnessSkillReplayJournalsRevertOnCuratorLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curator.jsonl")
	l, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	reverted := WitnessSkillReplay(SkillSituationReplay{
		Skill:      "unwitnessed-on-its-own-situation",
		Baseline:   SkillReplayOutcome{Succeeded: true, Turns: 5},
		Candidate:  SkillReplayOutcome{Succeeded: false, Turns: 3},
		TruthClean: true,
	})
	seq, err := reverted.Journal(l)
	if err != nil {
		t.Fatalf("journal reverted replay: %v", err)
	}
	if seq == 0 {
		t.Fatalf("a reverted replay skill was not journaled (seq 0)")
	}
	if r, gone := l.Why("unwitnessed-on-its-own-situation"); !gone || r.Kind != ReasonUnwitnessed {
		t.Fatalf("Why = %+v gone=%v, want ReasonUnwitnessed/gone", r, gone)
	}

	promoted := WitnessSkillReplay(SkillSituationReplay{
		Skill:      "witnessed-on-its-own-situation",
		Baseline:   SkillReplayOutcome{Succeeded: false, Turns: 7},
		Candidate:  SkillReplayOutcome{Succeeded: true, Turns: 4},
		TruthClean: true,
	})
	if s, err := promoted.Journal(l); err != nil || s != 0 {
		t.Fatalf("promoted replay journaled seq=%d err=%v, want no-op (0, nil)", s, err)
	}
}
