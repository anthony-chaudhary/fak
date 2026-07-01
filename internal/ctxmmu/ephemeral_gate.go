// Package ctxmmu — the ephemeral-observation write-time gate (issue #1592).
//
// CONTEXT-IS-NOT-MEMORY.md's rung 1 (classifyDurability/ClassifyText, mmu.go) already
// answers "how long is THIS span true" with a cheap lexical/tense prior, defaulting an
// unclassified body to DurabilityTurn. That prior is necessary but, on its own, silent:
// a caller who never consults it can still hand a raw situational remark straight to
// internal/memq's durable store, because nothing stops the crossing itself. #1598's
// disposition.go closed the analogous gap one level up (generalizing ONE observation
// into a standing trait); this file closes it at the base: promoting ONE raw
// observation straight into durable memory with no evidence it should outlive the turn.
//
// This is a narrow, additive companion to classifyDurability, not a new classifier: it
// reuses ClassifyText for the underlying durability verdict and adds only what's
// missing — (a) named sub-classes for the three fixture shapes the issue calls out
// (timestamp, current-step, mood/state), so a caller/test can see WHY an observation
// was refused, not just THAT it was, and (b) a closed Reclassification vocabulary an
// explicit override can supply, mirroring memq's ConsentExplicit and disposition.go's
// EvidenceUserConfirmed/EvidenceEstablishedPattern. The default is refuse — the same
// fail-closed posture as classifyDurability and GateDisposition: a false-negative
// (an un-promoted-but-true fact) is a recoverable re-ask; a false-positive (a promoted
// timestamp/step/mood) is a silent, standing, wrong belief.
package ctxmmu

import "regexp"

// SituationalKind names the closed set of situational (non-durable-by-default) shapes
// this gate recognizes, so a refusal can say WHICH kind of ephemeral fact tripped it —
// useful for audit/telemetry and for tests that pin the issue's own fixtures. It is
// deliberately a refinement OF ClassifyText's "turn"/"session" verdict, never a parallel
// classifier: SituationalOther covers any non-durable body that doesn't match one of the
// three named shapes (e.g. a bare unmatched tool result), so the gate still has
// somewhere to put it without inventing an "unknown situational" bucket.
type SituationalKind int

const (
	// SituationalNone: the observation classified DurabilityDurable on its own tense —
	// it is not a situational fact at all, so the gate has nothing to refuse.
	SituationalNone SituationalKind = iota
	// SituationalTimestamp: a clock-time / date / "now" deictic ("it's 3pm", "today is
	// a holiday") — CONTEXT-IS-NOT-MEMORY.md's own headline example.
	SituationalTimestamp
	// SituationalCurrentStep: a checkout-flow/wizard/task-progress marker ("I'm on step
	// 4", "we're on step 3 of the wizard") — true only for the current task instance.
	SituationalCurrentStep
	// SituationalMood: a transient emotional/physical state ("I am tired today", "I'm
	// frustrated with this bug") — true only for the current mood, not a standing trait.
	SituationalMood
	// SituationalOther: classified non-durable (turn or session) but does not match one
	// of the three named shapes above — still refused by default, just not one of the
	// issue's named fixture categories.
	SituationalOther
)

// String renders the kind for audit/log lines and test failure messages.
func (k SituationalKind) String() string {
	switch k {
	case SituationalNone:
		return "none"
	case SituationalTimestamp:
		return "timestamp"
	case SituationalCurrentStep:
		return "current-step"
	case SituationalMood:
		return "mood"
	case SituationalOther:
		return "other"
	default:
		return "other"
	}
}

// currentStepFrame: task/wizard/checkout progress markers ("I'm on step 4", "we're on
// step 3 of the wizard", "current step: 4"). Distinct from turnFrame's bare deictics
// (mmu.go) because a step marker doesn't always say "now"/"today" in words — it names a
// position in a task that is by construction over once the task ends.
var currentStepFrame = regexp.MustCompile(`(?i)\b(i'?m on step|we'?re on step|on step \d+|current step|step \d+ of)\b`)

// moodFrame: transient emotional/physical state ("I am tired", "I'm frustrated", "I'm
// stressed", "I'm exhausted", "I'm anxious", "I'm in a hurry"). Narrow by design (a
// closed adjective list, not a sentiment model) — the same "cheap lexical prior, fail
// closed" posture as durableFrame/turnFrame in mmu.go. Deliberately excludes "busy",
// which durability_test.go already pins to the transient "i-am-a" turn case via the
// generic fail-closed default, so this list only needs the CLEARLY mood/state words the
// issue itself names ("I am tired today").
var moodFrame = regexp.MustCompile(`(?i)\bi'?m (?:feeling )?(tired|exhausted|frustrated|stressed|anxious|overwhelmed|in a hurry|irritated|annoyed)\b|\bi am (?:feeling )?(tired|exhausted|frustrated|stressed|anxious|overwhelmed|irritated|annoyed)\b`)

// ClassifySituational names WHICH situational shape an observation's text matches, for
// audit/test purposes. It never overrides ClassifyText's durability verdict: a body
// that ClassifyText already calls DurabilityDurable is SituationalNone regardless of
// whether it happens to also match one of the lexical frames below (a durable-shaped
// remark that also names a step number is still durable — see classifyDurability's
// most-durable-first precedence, mmu.go). Only once ClassifyText says non-durable does
// this function refine WHICH kind of situational fact it is.
func ClassifySituational(text string) SituationalKind {
	if ClassifyText("", text) == DurabilityDurable {
		return SituationalNone
	}
	switch {
	case currentStepFrame.MatchString(text):
		return SituationalCurrentStep
	case moodFrame.MatchString(text):
		return SituationalMood
	case turnFrame.MatchString(text):
		return SituationalTimestamp
	default:
		return SituationalOther
	}
}

// Reclassification is the closed, explicit-override vocabulary this gate accepts —
// mirroring memq's ConsentExplicit and disposition.go's
// EvidenceUserConfirmed/EvidenceEstablishedPattern so a caller reuses one shape across
// all three "may this cross the promotion boundary" gates in the repo rather than
// inventing a fourth. The zero value, ReclassifyNone, is the default: no override
// offered, so a situational observation is refused.
type Reclassification int

const (
	// ReclassifyNone: no explicit override offered — the default. A situational
	// observation is refused for durable memory.
	ReclassifyNone Reclassification = iota
	// ReclassifyExplicitConsent: the user/operator explicitly asked for this exact fact
	// to be remembered despite its situational shape (mirrors memq.ConsentExplicit).
	ReclassifyExplicitConsent
	// ReclassifyUserConfirmed: the user explicitly confirmed the durable generalization
	// (mirrors disposition.go's EvidenceUserConfirmed) — e.g. "yes, remember that I'm
	// generally a tired/evening person."
	ReclassifyUserConfirmed
	// ReclassifyEstablishedPattern: an already-established pattern/prior durable record
	// backs this promotion (mirrors disposition.go's EvidenceEstablishedPattern) — e.g.
	// a consolidation step corroborated by multiple independent observations.
	ReclassifyEstablishedPattern
)

// EphemeralGateOutcome is the typed result of GateEphemeral — never a bare bool, so a
// caller/test can branch on WHY, the same closed-outcome discipline as
// DispositionOutcome above.
type EphemeralGateOutcome struct {
	// Allowed reports whether the observation may cross into durable memory.
	Allowed bool
	// Situational names which shape (if any) the raw text matched.
	Situational SituationalKind
	// Reclassification echoes the override the caller supplied (ReclassifyNone if
	// none), so an audit trail can show WHY a situational fact was let through.
	Reclassification Reclassification
	// Reason is a human-readable justification for audit/logging.
	Reason string
}

// GateEphemeral is the write-time gate #1592 asks for: given the raw observation text
// and an optional explicit Reclassification, it decides whether the fact may cross into
// durable memory (internal/memq's store). It is pure and deterministic — no model call,
// no I/O — the same posture as classifyDurability/GateDisposition.
//
// Decision:
//  1. If the text is not situational at all (ClassifyText already says
//     DurabilityDurable), the gate allows it — there is nothing to refuse; this is the
//     ordinary durable-write path, unaffected by this gate.
//  2. If it IS situational (timestamp-bound, current-step-bound, or mood/state-bound —
//     or any other non-durable shape) and no explicit reclassification is supplied
//     (ReclassifyNone), the gate refuses: default is EXPIRE, not promote.
//  3. If it IS situational but the caller supplies ANY non-zero Reclassification
//     (explicit consent, user confirmation, or an established-pattern override), the
//     gate allows it — "unless explicitly reclassified" per the issue's done condition.
//     The override is caller-asserted (this gate does not itself verify the evidence
//     behind EvidenceUserConfirmed/EvidenceEstablishedPattern the way GateDisposition
//     does — a caller that wants that stronger check should run GateDisposition first
//     and pass ReclassifyEstablishedPattern/ReclassifyUserConfirmed only once GateDisposition
//     minted).
func GateEphemeral(text string, reclass Reclassification) EphemeralGateOutcome {
	kind := ClassifySituational(text)
	if kind == SituationalNone {
		return EphemeralGateOutcome{
			Allowed:          true,
			Situational:      kind,
			Reclassification: reclass,
			Reason:           "not situational: classified durable on its own tense",
		}
	}
	if reclass != ReclassifyNone {
		return EphemeralGateOutcome{
			Allowed:          true,
			Situational:      kind,
			Reclassification: reclass,
			Reason:           kind.String() + "-class observation allowed via explicit reclassification (" + reclass.String() + ")",
		}
	}
	return EphemeralGateOutcome{
		Allowed:          false,
		Situational:      kind,
		Reclassification: reclass,
		Reason:           kind.String() + "-class observation refused for durable memory: expire by default, promotion is the earned exception",
	}
}

// String renders the Reclassification for audit/log lines.
func (r Reclassification) String() string {
	switch r {
	case ReclassifyNone:
		return "none"
	case ReclassifyExplicitConsent:
		return "explicit-consent"
	case ReclassifyUserConfirmed:
		return "user-confirmed"
	case ReclassifyEstablishedPattern:
		return "established-pattern"
	default:
		return "none"
	}
}
