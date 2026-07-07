package nightrun

// learningnudge.go — the value-gated RSI learning-nudge trigger (#2910).
//
// THE PROBLEM. A background "review this session for extractable memory/skills"
// fork costs real tokens. Firing it on a FIXED clock — Hermes fires its memory
// nudge every user turn and its skill nudge every tool iteration, both on a default
// every-10 counter — spends that cost blind to whether the session produced anything
// worth learning. A dense, novel session is under-reviewed; a repetitive one that
// learned nothing still pays the fork.
//
// WHAT THIS DOES. fak already MEASURES sessions (the #2822 session-analytics epic).
// This gate replaces the every-10-turns counter with a MEASURED trigger: score each
// session over its tool/sequence SHAPE (novelty, reusing the #2823/#2825 shape idea)
// and its UNRESOLVED FRICTION (repeated tool errors / guard refusals / interrupts),
// and fire the review only when that blended score crosses a threshold — so the review
// budget lands where expected learning value is highest.
//
// NOVELTY vs FRICTION (the named confusion risk). The two are DISTINCT signals and are
// combined DELIBERATELY, not conflated: novelty is "this session did something new"
// (new tool unigrams / consecutive-pair bigrams never seen before), friction is "this
// session kept failing at something" (a saturating count of unresolved errors). Either
// one alone can be worth a review — a novel success teaches a new skill; a repeated
// failure teaches a guardrail — so the gate is a weighted SUM with a single threshold:
// a maximally-frictional session crosses on friction alone, a maximally-novel one on
// novelty alone, and a session that is both crosses easily.
//
// Tier: foundation (1). Pure and deterministic — no clock, no RNG, no I/O: the same
// session stream scores identically every time, so the ablation below is a fixed
// witness. The impure shell (a `fak sessions` reader that projects a transcript onto a
// NudgeSession and then forks the real background review) is the follow-on; this file
// owns only the WHEN decision and the ablation that proves it beats the fixed cadence.
// It changes only WHEN a review is triggered, never how the review itself works.

import "math"

// NudgeSession is a #2822 session-analytics record projected onto exactly the fields
// the nudge gate reads. The impure shell builds it from a transcript; a test supplies
// fixtures. It carries structured signal only — a tool-name sequence and friction
// counts — never raw prompt/result prose, so a corpus of these stays committable
// (the same discipline internal/sessionobs holds).
type NudgeSession struct {
	// ID is the transcript's opaque session id (not content).
	ID string
	// ToolSeq is the ordered sequence of tool NAMES the session issued — its shape.
	// Novelty is scored over the unigrams and consecutive-pair bigrams of this slice.
	ToolSeq []string
	// ToolErrors, GuardRefusals, Interrupts are the unresolved-friction counts (the
	// #2822 Signals fields): is_error tool_results, kernel DENIES, interrupted turns.
	ToolErrors    int
	GuardRefusals int
	Interrupts    int
	// ReviewTokens is what a background review of THIS session would spend — the cost
	// the gate is deciding whether to pay. The ablation folds it over reviewed sessions.
	ReviewTokens int64
	// KeptSkills is an ABLATION-ONLY ground-truth label: how many skills a review of
	// this session would keep. It is NEVER read by the gate (the gate cannot see the
	// future); it exists so the offline ablation can measure tokens-per-kept-skill of a
	// review policy against a labeled corpus.
	KeptSkills int
}

// NudgeConfig is the gate's tunables. The weights sum to 1.0 so a fully-novel or a
// fully-frictional session lands on a comparable [0,1] scale against Threshold.
type NudgeConfig struct {
	WNovelty     float64 // weight on the novelty signal
	WFriction    float64 // weight on the unresolved-friction signal
	FrictionNorm float64 // friction count that saturates to 1.0
	Threshold    float64 // blended score at/above which the review fires
	FixedCadence int     // the every-N-sessions clock the ablation compares against
}

// DefaultNudgeConfig is the shipped gate: novelty-leaning but friction alone can still
// trigger (WFriction == Threshold, so friction==1.0 lands exactly on the line), a
// friction count of 5 saturates, and the fixed-cadence baseline is Hermes' every-10.
func DefaultNudgeConfig() NudgeConfig {
	return NudgeConfig{WNovelty: 0.6, WFriction: 0.4, FrictionNorm: 5, Threshold: 0.4, FixedCadence: 10}
}

// nudgeShingles returns the deterministic shape tokens of a tool sequence: every tool
// unigram and every consecutive-pair bigram ("A\x1fB"). Bigrams are what make novelty
// a SEQUENCE signal and not just a tool-set signal — a session that runs a known set of
// tools in a genuinely new ORDER still reads as novel.
func nudgeShingles(seq []string) []string {
	if len(seq) == 0 {
		return nil
	}
	out := make([]string, 0, len(seq)*2)
	for i, t := range seq {
		out = append(out, "1\x1f"+t)
		if i > 0 {
			out = append(out, "2\x1f"+seq[i-1]+"\x1f"+t)
		}
	}
	return out
}

// NudgeNovelty is the fraction of a session's SHAPE that has never been seen before —
// the share of its shingles absent from seen, in [0,1]. An empty session has no shape,
// so its novelty is 0 (nothing new to learn from a session that did nothing).
func NudgeNovelty(s NudgeSession, seen map[string]bool) float64 {
	sh := nudgeShingles(s.ToolSeq)
	if len(sh) == 0 {
		return 0
	}
	// Dedup within the session first, so a shape repeated inside one session is not
	// double-counted for/against its own novelty.
	uniq := make(map[string]bool, len(sh))
	for _, k := range sh {
		uniq[k] = true
	}
	novel := 0
	for k := range uniq {
		if !seen[k] {
			novel++
		}
	}
	return float64(novel) / float64(len(uniq))
}

// friction is the saturating unresolved-friction signal: the total error/refusal/
// interrupt count over FrictionNorm, clamped to [0,1].
func (c NudgeConfig) friction(s NudgeSession) float64 {
	norm := c.FrictionNorm
	if norm <= 0 {
		norm = 1
	}
	total := s.ToolErrors + s.GuardRefusals + s.Interrupts
	return clamp01(float64(total) / norm)
}

// NudgeScore blends novelty and friction into the single value the gate thresholds.
func (c NudgeConfig) NudgeScore(s NudgeSession, seen map[string]bool) float64 {
	return c.WNovelty*NudgeNovelty(s, seen) + c.WFriction*c.friction(s)
}

// ShouldReview is the gate: does this session's measured novelty/friction score cross
// the threshold — i.e. is a background review worth its token cost here? This is the
// literal replacement for the every-10-turns counter.
func (c NudgeConfig) ShouldReview(s NudgeSession, seen map[string]bool) bool {
	return c.NudgeScore(s, seen) >= c.Threshold
}

// nudgeFold records a session's shape into the seen set, so the next session's novelty
// is measured against everything before it (online history).
func nudgeFold(seen map[string]bool, s NudgeSession) {
	for _, k := range nudgeShingles(s.ToolSeq) {
		seen[k] = true
	}
}

// NudgeAblation is one review policy's outcome over a session stream: how much review
// budget it spent and how many kept skills that bought — folded into the headline
// tokens-per-kept-skill (lower is better: fewer tokens spent per skill actually kept).
type NudgeAblation struct {
	Policy             string  `json:"policy"`
	Reviewed           int     `json:"reviewed"`
	ReviewTokens       int64   `json:"review_tokens"`
	KeptSkills         int     `json:"kept_skills"`
	TokensPerKeptSkill float64 `json:"tokens_per_kept_skill"` // +Inf when KeptSkills==0
}

func newNudgeAblation(policy string, reviewTokens int64, keptSkills, reviewed int) NudgeAblation {
	tpks := math.Inf(1)
	if keptSkills > 0 {
		tpks = float64(reviewTokens) / float64(keptSkills)
	}
	return NudgeAblation{
		Policy:             policy,
		Reviewed:           reviewed,
		ReviewTokens:       reviewTokens,
		KeptSkills:         keptSkills,
		TokensPerKeptSkill: tpks,
	}
}

// NudgeAblate runs BOTH review policies over the same session stream, in order, and
// returns their tokens-per-kept-skill so a caller can prove the gate wins:
//
//   - gated:  fire the review when ShouldReview crosses the threshold. Novelty is
//     scored against the shapes seen BEFORE each session (every session folds into the
//     history afterward), so it is measured exactly as a live online loop would.
//   - fixed:  fire the review every FixedCadence sessions (the content-blind clock —
//     the every-10-turns baseline), regardless of what the session produced.
//
// The witness the issue asks for is gated.TokensPerKeptSkill < fixed.TokensPerKeptSkill:
// the measured gate spends fewer tokens per kept skill than the fixed cadence because it
// concentrates the review budget on the sessions that actually produced something.
func (c NudgeConfig) NudgeAblate(stream []NudgeSession) (gated, fixed NudgeAblation) {
	seen := map[string]bool{}
	var gTokens, fTokens int64
	var gKept, fKept, gReviewed, fReviewed int
	cadence := c.FixedCadence
	if cadence <= 0 {
		cadence = 1
	}
	for i, s := range stream {
		if c.ShouldReview(s, seen) {
			gReviewed++
			gTokens += s.ReviewTokens
			gKept += s.KeptSkills
		}
		if (i+1)%cadence == 0 {
			fReviewed++
			fTokens += s.ReviewTokens
			fKept += s.KeptSkills
		}
		nudgeFold(seen, s)
	}
	return newNudgeAblation("novelty-gated", gTokens, gKept, gReviewed),
		newNudgeAblation("fixed-cadence", fTokens, fKept, fReviewed)
}
