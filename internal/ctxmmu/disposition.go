// Package ctxmmu — the disposition-minting gate (issue #1598).
//
// classifyDurability (mmu.go) answers "how long is THIS span true" — a per-write
// TTL tag. It does NOT answer a different, downstream question: "may this ONE
// situational observation be GENERALIZED into a standing trait about the user?"
// That second question is a mint decision, not a classification, and today
// nothing in the repo gates it — a consolidation step could read a single "I am
// tired today" and free-associate "user prefers short answers" with no evidence
// the trait is real. That is exactly the "ephemeral promoted" failure
// CONTEXT-IS-NOT-MEMORY.md §2 names, one level up: not a raw fact surviving past
// its turn, but a raw fact laundered into a FABRICATED durable belief it never
// supported in the first place.
//
// This file adds that missing gate. It is deliberately NOT a new parallel
// durability mechanism: an Observation reuses classifyDurability/ClassifyText for
// its own situational-vs-durable shape, and a minted Disposition rides the same
// {turn,session,durable} vocabulary as its Durability. What's new is narrow and
// closed:
//
//   - Observation  — a raw, single, timestamped remark (the write-boundary input).
//   - Disposition  — a standing trait/preference a caller wants to mint FROM one
//     or more observations.
//   - Evidence     — what backs the generalization: corroborating observations,
//     an explicit user confirmation, or an already-established pattern.
//   - GateDisposition — the pure decision function. No model call, no I/O,
//     deterministic on its inputs — the same "cheap prior, fail closed" posture
//     as classifyDurability.
//
// The gate's default is refuse, mirroring the repo's existing durability
// default (classifyDurability's `default: turn`) and the promotion gate's
// default (recall.PromotionWarn / PromotionEnforce refuses non-durable): a
// false-negative (an unminted-but-true disposition) is recoverable — the user
// restates it, or a later observation corroborates it. A false-positive (a
// minted-but-false disposition) is a silent, standing, wrong belief that colors
// every future turn — "strictly worse than absence" (CONTEXT-IS-NOT-MEMORY.md
// §4). GateDisposition never silently drops and never silently promotes: it
// always returns a typed Outcome the caller must branch on.
package ctxmmu

import "strings"

// ---------------------------------------------------------------------------
// Observation / Disposition — the closed vocabulary (#1598).
// ---------------------------------------------------------------------------

// Observation is a single raw, situational remark as it arrived at the write
// boundary — the input a consolidation/summarization step might be tempted to
// generalize. It carries its own text and the durability class its own tense
// implies (via ClassifyText), so the gate can tell a punctual/progressive
// remark ("I am tired today") from one that is already habitual/stative on its
// face ("I always prefer short answers") without a second classifier.
type Observation struct {
	// Text is the raw remark, verbatim (e.g. "I am tired today").
	Text string
	// Source names who/what produced it (e.g. "user", a tool id); informational
	// only — the gate does not branch on it today, reserved for a future
	// principal-weighted prior (mirrors classifyDurability's reserved *ToolCall
	// argument in mmu.go).
	Source string
}

// situationalClass is the ClassifyText verdict for this observation's own text —
// the SAME rung-1 lexical/tense prior the write-time gate already runs, reused
// rather than reinvented. A durable-shaped remark ("I always...") is already
// evidence of a standing pattern on its own; a turn/session-shaped remark
// ("today", "right now") is exactly the situational case the issue is about.
func (o Observation) situationalClass() string {
	return ClassifyText(o.Source, o.Text)
}

// Disposition is a standing trait/preference a caller wants to mint — the
// durable generalization a consolidation step derives FROM one or more
// Observations (e.g. "user prefers short answers"). Minting one is exactly the
// promotion this gate exists to guard: once minted it is meant to be believed
// across sessions, so it must not be founded on a single situational remark.
type Disposition struct {
	// Trait is the standing claim text (e.g. "user prefers short answers").
	Trait string
	// Durability is the class the minted disposition would carry if allowed —
	// always DurabilityDurable for a gate call (see GateDisposition), carried
	// on the type so a Minted outcome is a self-describing record consistent
	// with the rest of the package's {turn,session,durable} vocabulary.
	Durability string
}

// EvidenceKind is the closed set of explicit-support shapes GateDisposition
// accepts. A bare Observation with no Evidence (the zero value, EvidenceNone)
// never mints — that is the fixture the issue pins: "I am tired today" alone.
type EvidenceKind int

const (
	// EvidenceNone: no explicit support offered — the default for a single,
	// un-corroborated observation. Never sufficient to mint.
	EvidenceNone EvidenceKind = iota
	// EvidenceCorroboration: multiple independent observations pointing at the
	// same disposition (see Evidence.Corroborating). Sufficient once the count
	// reaches MinCorroboration.
	EvidenceCorroboration
	// EvidenceUserConfirmed: the user explicitly confirmed the generalization
	// ("yes, in general I prefer short answers") — the strongest signal,
	// sufficient on its own regardless of corroboration count.
	EvidenceUserConfirmed
	// EvidenceEstablishedPattern: an already-established, previously-minted
	// disposition or externally-recorded pattern backs this one (e.g. a prior
	// session's durable store already holds the trait) — sufficient on its own.
	EvidenceEstablishedPattern
)

// Evidence is what a caller offers to justify generalizing an Observation into
// a Disposition. The zero value (EvidenceNone, no corroborating observations)
// represents "nothing but the one observation" — the exact unsupported case the
// gate must refuse.
type Evidence struct {
	Kind EvidenceKind
	// Corroborating holds the OTHER observations offered as independent
	// corroboration (the triggering Observation passed to GateDisposition is
	// never counted twice, even if a caller accidentally includes it here — see
	// GateDisposition). Only meaningful when Kind == EvidenceCorroboration, but
	// harmless (ignored) otherwise.
	Corroborating []Observation
	// Confirmed is the user's own confirming statement, kept for audit (e.g.
	// "yes, always keep it brief"). Only meaningful when
	// Kind == EvidenceUserConfirmed.
	Confirmed string
	// Pattern names the already-established disposition/source this claim
	// rests on (e.g. a prior durable Disposition.Trait, or an external record
	// id). Only meaningful when Kind == EvidenceEstablishedPattern.
	Pattern string
}

// MinCorroboration is the minimum number of INDEPENDENT corroborating
// observations (beyond the single triggering one) required before
// EvidenceCorroboration is sufficient to mint. Two independent situational
// remarks pointing at the same trait is the cheapest bar that still isn't "one
// remark generalized" — a single corroborator is still just two anecdotes, which
// is why the bar is 2, not 1. Exported so a caller can reason about / test the
// threshold rather than it being a silent magic number.
const MinCorroboration = 2

// OutcomeKind is the closed, typed result of a gate decision. GateDisposition
// never returns a bare bool and never silently drops or silently promotes —
// every call resolves to exactly one of these, so a caller (and a test) can
// branch on the SHAPE of the decision, not parse a reason string.
type OutcomeKind int

const (
	// OutcomeRefusedUnsupported: a single situational observation with no
	// evidence — refuse to mint. The issue's own fixture case: "I am tired
	// today" alone must land here, not silently vanish and not silently become
	// "user prefers short answers".
	OutcomeRefusedUnsupported OutcomeKind = iota
	// OutcomeRefusedInsufficientCorroboration: EvidenceCorroboration was
	// offered but fell short of MinCorroboration — still a refusal, but
	// distinguishable from "no evidence at all" for audit/telemetry.
	OutcomeRefusedInsufficientCorroboration
	// OutcomeMinted: the generalization is allowed — sufficient explicit
	// support was present. Outcome.Disposition is populated.
	OutcomeMinted
)

// DispositionOutcome is the typed result of GateDisposition: either a minted
// Disposition (Kind == OutcomeMinted, Disposition populated) or a refusal
// (Kind is one of the OutcomeRefused* values, Disposition is the zero value)
// carrying a human-readable Reason for audit/logging. A caller must switch on
// Kind — there is no ambiguous "ok bool" to ignore.
type DispositionOutcome struct {
	Kind        OutcomeKind
	Disposition Disposition
	Reason      string
}

// Minted reports whether the gate allowed the generalization. Convenience only
// — equivalent to Kind == OutcomeMinted — kept because most callers only need
// the yes/no and the populated Disposition, not the refined refusal reason.
func (o DispositionOutcome) Minted() bool { return o.Kind == OutcomeMinted }

// GateDisposition is the pure decision function (#1598): given a single
// situational Observation, the Disposition trait a caller wants to derive from
// it, and the Evidence offered to support that generalization, it decides
// whether minting is allowed.
//
// It is pure and deterministic: no model call, no I/O, no shared state — the
// same posture as classifyDurability/ScreenBytes elsewhere in this package.
// The decision:
//
//  1. EvidenceUserConfirmed with a non-empty Confirmed string mints
//     unconditionally — an explicit "yes, that's a general preference" is the
//     strongest and most direct support there is.
//  2. EvidenceEstablishedPattern with a non-empty Pattern mints unconditionally
//     — the trait is not being founded on this one observation at all, it is
//     being corroborated by an already-established source.
//  3. EvidenceCorroboration mints only once len(Evidence.Corroborating) (after
//     de-duplicating the triggering observation and blank text out of it) is
//     >= MinCorroboration. Short of that, it is
//     OutcomeRefusedInsufficientCorroboration, not OutcomeRefusedUnsupported —
//     the caller offered support, just not enough.
//  4. EvidenceNone (or an Evidence zero value) with a single Observation always
//     refuses as OutcomeRefusedUnsupported, regardless of how the observation's
//     own text classifies — even a durable-*shaped* remark ("I always...") said
//     exactly once is still one utterance, and one utterance minting a
//     cross-session belief is the false-positive-promotion failure this gate
//     exists to close off. (A durable-shaped remark is free to pass through
//     classifyDurability/the recall promotion gate as high-durability CONTEXT;
//     this gate is strictly about MINTING A NEW DISPOSITION, a stronger claim.)
//
// The fixture from the issue is exactly case 4: Observation{Text: "I am tired
// today"} with no Evidence must not mint Disposition{Trait: "user prefers short
// answers"}.
func GateDisposition(obs Observation, want Disposition, ev Evidence) DispositionOutcome {
	switch ev.Kind {
	case EvidenceUserConfirmed:
		if strings.TrimSpace(ev.Confirmed) != "" {
			return mint(want)
		}
		// Claimed user-confirmed but carries no actual confirmation text — that
		// is indistinguishable from no evidence, so it fails closed rather than
		// trusting an empty confirmation.
		return refuseUnsupported(obs)

	case EvidenceEstablishedPattern:
		if strings.TrimSpace(ev.Pattern) != "" {
			return mint(want)
		}
		return refuseUnsupported(obs)

	case EvidenceCorroboration:
		n := independentCorroborators(obs, ev.Corroborating)
		if n >= MinCorroboration {
			return mint(want)
		}
		return DispositionOutcome{
			Kind:   OutcomeRefusedInsufficientCorroboration,
			Reason: "only " + itoa(n) + " independent corroborating observation(s); need " + itoa(MinCorroboration),
		}

	default: // EvidenceNone or any unrecognized/zero-value Kind — fail closed.
		return refuseUnsupported(obs)
	}
}

// mint builds the OutcomeMinted result, stamping the durable class onto the
// returned Disposition so it is self-describing and consistent with the
// package's {turn,session,durable} vocabulary — a minted disposition is by
// definition the durable class (that is what "standing trait" means).
func mint(want Disposition) DispositionOutcome {
	d := want
	d.Durability = DurabilityDurable
	return DispositionOutcome{Kind: OutcomeMinted, Disposition: d, Reason: "explicit support present"}
}

// refuseUnsupported builds the OutcomeRefusedUnsupported result for a bare
// situational observation with no (or no usable) evidence.
func refuseUnsupported(obs Observation) DispositionOutcome {
	return DispositionOutcome{
		Kind:   OutcomeRefusedUnsupported,
		Reason: "single situational observation (" + obs.situationalClass() + "-class) with no corroboration, confirmation, or established pattern",
	}
}

// independentCorroborators counts observations in candidates that are
// independent of the triggering observation obs: non-blank text, and not a
// literal repeat of obs.Text (a caller accidentally re-passing the same
// observation twice must not be able to double-count it into corroboration).
// Duplicate candidate texts among themselves also collapse to one — three
// copies of the identical remark is still one data point, not three.
func independentCorroborators(obs Observation, candidates []Observation) int {
	seen := map[string]bool{strings.TrimSpace(obs.Text): true}
	n := 0
	for _, c := range candidates {
		t := strings.TrimSpace(c.Text)
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		n++
	}
	return n
}

// itoa is a tiny dependency-free int->string helper (avoids importing
// strconv for one call site's error-message formatting).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
