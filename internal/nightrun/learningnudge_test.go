package nightrun

import (
	"math"
	"testing"
)

// TestNudgeNoveltyMeasuresShape checks the two ends of the novelty signal: a session
// against an empty history is fully novel, the same shape a second time is not, and a
// genuinely-new ORDER of already-seen tools still reads as partly novel (the bigram).
func TestNudgeNoveltyMeasuresShape(t *testing.T) {
	seen := map[string]bool{}

	first := NudgeSession{ToolSeq: []string{"Read", "Grep", "Bash"}}
	if got := NudgeNovelty(first, seen); got != 1.0 {
		t.Fatalf("first-ever session novelty = %v, want 1.0", got)
	}
	nudgeFold(seen, first)

	if got := NudgeNovelty(first, seen); got != 0.0 {
		t.Fatalf("repeat session novelty = %v, want 0.0", got)
	}

	// Same tool SET, new order: unigrams all seen, but the "Bash->Read" bigram is new.
	reorder := NudgeSession{ToolSeq: []string{"Bash", "Read"}}
	got := NudgeNovelty(reorder, seen)
	if got <= 0 || got >= 1 {
		t.Fatalf("reordered session novelty = %v, want strictly between 0 and 1", got)
	}

	// Empty session has no shape to learn from.
	if got := NudgeNovelty(NudgeSession{}, seen); got != 0 {
		t.Fatalf("empty session novelty = %v, want 0", got)
	}
}

// TestNudgeFrictionAloneTriggers proves the confusion-risk resolution: a session with
// ZERO novelty but saturating unresolved friction still crosses the gate — friction is
// an independent trigger, not folded away by novelty.
func TestNudgeFrictionAloneTriggers(t *testing.T) {
	cfg := DefaultNudgeConfig()
	seen := map[string]bool{}
	// Seed the shape so it is not novel.
	repeat := NudgeSession{ToolSeq: []string{"Read", "Grep"}}
	nudgeFold(seen, repeat)

	if NudgeNovelty(repeat, seen) != 0 {
		t.Fatalf("precondition: repeated shape should have 0 novelty")
	}
	frictional := NudgeSession{ToolSeq: []string{"Read", "Grep"}, ToolErrors: 10}
	if !cfg.ShouldReview(frictional, seen) {
		t.Fatalf("a fully-frictional, zero-novelty session should be reviewed; score=%v thr=%v",
			cfg.NudgeScore(frictional, seen), cfg.Threshold)
	}
	// A quiet, already-seen session with no friction must NOT be reviewed.
	if cfg.ShouldReview(repeat, seen) {
		t.Fatalf("a low-novelty, friction-free session should be skipped; score=%v",
			cfg.NudgeScore(repeat, seen))
	}
}

// ablationStream builds a deterministic labeled corpus: a handful of novel/frictional
// sessions that yield kept skills, plus a long tail of repetitive filler that learned
// nothing. It is shaped so the every-10 clock lands mostly on filler while the gate
// concentrates on the yield.
func ablationStream() []NudgeSession {
	const tokens = 800
	seed := []string{"Read", "Grep"} // the common, learned-nothing filler shape
	stream := make([]NudgeSession, 30)
	for i := range stream {
		// Default: repetitive filler — known shape, no friction, keeps no skill.
		stream[i] = NudgeSession{
			ID:           id(i),
			ToolSeq:      seed,
			ReviewTokens: tokens,
		}
	}
	// Novel, high-yield sessions. Session 0 also contains the filler shape so every
	// later filler is genuinely non-novel.
	stream[0] = NudgeSession{ID: id(0), ToolSeq: []string{"Read", "Grep", "U0a", "U0b", "Bash"}, ReviewTokens: tokens, KeptSkills: 4}
	stream[1] = NudgeSession{ID: id(1), ToolSeq: []string{"U1a", "U1b", "Bash"}, ReviewTokens: tokens, KeptSkills: 4}
	stream[2] = NudgeSession{ID: id(2), ToolSeq: []string{"U2a", "U2b", "Bash"}, ReviewTokens: tokens, KeptSkills: 4}
	stream[3] = NudgeSession{ID: id(3), ToolSeq: []string{"U3a", "U3b", "Bash"}, ReviewTokens: tokens, KeptSkills: 4}
	// A novel yield session that ALSO lands on the fixed clock (index 19 -> the 20th).
	stream[19] = NudgeSession{ID: id(19), ToolSeq: []string{"U19a", "U19b", "Bash"}, ReviewTokens: tokens, KeptSkills: 4}
	// A friction-only yield session: repeated shape, but unresolved failures the gate
	// catches and the clock (index 25) misses.
	stream[25] = NudgeSession{ID: id(25), ToolSeq: seed, ToolErrors: 10, ReviewTokens: tokens, KeptSkills: 3}
	return stream
}

func id(i int) string { return "sess-" + string(rune('a'+i%26)) + string(rune('0'+i/26)) }

// TestNudgeAblationBeatsFixedCadence is the issue's witness: over the same session
// stream, the novelty/friction-gated trigger spends fewer tokens per kept skill than
// the fixed every-10 cadence, because it fires the costly review where learning value
// is highest instead of on a clock.
func TestNudgeAblationBeatsFixedCadence(t *testing.T) {
	cfg := DefaultNudgeConfig()
	gated, fixed := cfg.NudgeAblate(ablationStream())

	if gated.KeptSkills == 0 || fixed.KeptSkills == 0 {
		t.Fatalf("both policies must keep at least one skill for a finite ratio: gated=%+v fixed=%+v", gated, fixed)
	}
	if math.IsInf(gated.TokensPerKeptSkill, 0) || math.IsInf(fixed.TokensPerKeptSkill, 0) {
		t.Fatalf("tokens-per-kept-skill must be finite: gated=%v fixed=%v", gated.TokensPerKeptSkill, fixed.TokensPerKeptSkill)
	}
	if !(gated.TokensPerKeptSkill < fixed.TokensPerKeptSkill) {
		t.Fatalf("ablation FAILED: novelty-gated tokens-per-kept-skill %.1f is not below fixed-cadence %.1f\n  gated=%+v\n  fixed=%+v",
			gated.TokensPerKeptSkill, fixed.TokensPerKeptSkill, gated, fixed)
	}
	t.Logf("ablation: novelty-gated %.1f tokens/kept-skill (%d reviews, %d kept) < fixed-cadence %.1f (%d reviews, %d kept)",
		gated.TokensPerKeptSkill, gated.Reviewed, gated.KeptSkills,
		fixed.TokensPerKeptSkill, fixed.Reviewed, fixed.KeptSkills)
}
