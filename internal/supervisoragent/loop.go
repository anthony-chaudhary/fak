package supervisoragent

import (
	"errors"
	"time"
)

// This file is build-plan step S3 of the Chief-of-Staff runtime note
// (fak-private docs/notes/CONCEPT-CHIEF-OF-STAFF-AGENT-RUNTIME-2026-09-21.md,
// section 7; issue #13464): the bounded seat-loop driver — the
// wake -> assemble -> decide -> lower -> journal cycle over the shipped
// contracts. It is deterministic-first by construction: every stage lowers to
// a pure function or an existing admission verb, and NO stage reaches a model.
// A case a fixed rung can decide never leaves this package's Decide ladder;
// the model-backed advisor is a separate gated follow-on, not a stage here.
//
// One Seat drives one wake at a time (the doctrine's single-writer rule: no
// unshared parallel writers on the journal). RunWake appends EXACTLY one
// DecisionRecord per call — the addressed, payload-free record S1 defined —
// so a restart is a replay from the journal cursor, never a re-decision.
//
// Dry-run is structural, not conventional: with dryRun == true the lower
// stage does not touch the AdmissionVerbs seam at all — zero arbitration
// calls, zero admits, zero packet emits — and the record carries
// OutcomeDryRun with the verdict a live lower stage would have received.
// That is the zero-side-effect proof the build plan asks for.

// ErrEmptyTopic refuses a wake without its address: the journal is addressed
// (one record per wake topic), and an unaddressed record cannot be replayed
// back to the wake that produced it.
var ErrEmptyTopic = errors.New("supervisoragent: wake requires a non-empty topic")

// ErrNoAdmissionVerbs refuses a LIVE lower stage with no bound verbs: lowering
// through nothing would silently turn an admitted action into a no-op. The
// dry-run path never needs verbs and is unaffected.
var ErrNoAdmissionVerbs = errors.New("supervisoragent: live lowering requires bound admission verbs")

// Seat is one bounded instance of the seat loop: the durable journal it
// appends to, the closed admission verbs a live lower stage may reach, the
// earned Envelope, and the soak Mode. All four are passed in — the Seat owns
// no clock, no environment read, and no I/O of its own beyond the journal
// append (the wake-time caller binds the real surfaces; tests bind doubles).
type Seat struct {
	Journal  Journal
	Verbs    AdmissionVerbs // nil permitted only for dry-run wakes
	Envelope Envelope
	Mode     EnvelopeMode
}

// RunWake drives exactly one wake through the five stages and appends exactly
// one DecisionRecord:
//
//  1. assemble — Assemble projects the witness sources into the closed
//     SupervisorInput; an unreadable surface stays absent.
//  2. decide — the pure Decide ladder maps input + envelope + mode to exactly
//     one proposed action and its closed reason token. No I/O, no clock, no
//     model.
//  3. gate — Authorize renders the envelope verdict on the proposal; the
//     verdict rides the record either way.
//  4. lower — live wakes lower ONLY through LowerInEnvelope (the one path that
//     knows both the verdict and the closed admission verbs); dry-run wakes
//     make ZERO admission calls and record OutcomeDryRun instead.
//  5. journal — one addressed DecisionRecord is appended; the returned record
//     is the journal-stamped copy (Seq assigned by the owning Journal).
//
// Outcomes: lowered (an admission verb ran and left its artifact), refused
// (enforce held the verb below earned width — nothing ran), escalated (the
// fail-closed path emitted the escalation packet), hold (the deliberate
// no-op), dry_run (preview — nothing ran), verb_refused (the admission call
// itself refused — recorded, never retried).
func (s *Seat) RunWake(topic string, src Sources, at time.Time, dryRun bool) (DecisionRecord, error) {
	if topic == "" {
		return DecisionRecord{}, ErrEmptyTopic
	}
	if !dryRun && s.Verbs == nil {
		return DecisionRecord{}, ErrNoAdmissionVerbs
	}
	if s.Journal == nil {
		return DecisionRecord{}, errors.New("supervisoragent: the seat requires a bound journal")
	}

	// Stage 1: assemble (pure projection; absence preserved).
	in := Assemble(src)

	// Stage 2: decide (pure ladder; deterministic-first — no model reachable).
	action, reason := Decide(in, s.Envelope, s.Mode)

	// Stage 3: gate (pure verdict; rides the record either way).
	verdict := Authorize(in, s.Envelope, s.Mode, action)

	var outcome Outcome
	var artifact string
	if dryRun {
		// Stage 4 (dry-run): the preview runs the full cycle but makes ZERO
		// admission calls — the Verbs seam is never touched, so the proof of
		// zero side effects is structural, not asserted after the fact.
		outcome = OutcomeDryRun
	} else {
		// Stage 4 (live): the ONLY execution path is LowerInEnvelope.
		_, eff, err := LowerInEnvelope(action, s.Verbs, in, s.Envelope, s.Mode)
		switch {
		case err == nil:
			switch eff.Verb {
			case VerbNone:
				// hold lowers to NO verb at all: nothing ran, nothing was
				// left — the deliberate no-op, not a lowering.
				outcome = OutcomeHold
			case VerbEmitEscalation:
				// the fail-closed path emitted its packet: that IS the
				// escalation outcome S1 names, with the packet head as witness.
				outcome, artifact = OutcomeEscalated, artifactRef(eff)
			default:
				outcome, artifact = OutcomeLowered, artifactRef(eff)
			}
		case errors.Is(err, ErrConfirmRequired):
			outcome = OutcomeRefused
		case errors.Is(err, ErrEnvelopeFailClosed):
			outcome = OutcomeEscalated
		default:
			// An admitted verb whose admission call refused: record it, do
			// NOT retry, force, or widen around the refusal.
			outcome = OutcomeVerbRefused
		}
	}

	// Stage 5: journal — exactly one record per wake.
	rec := DecisionRecord{
		Topic:           topic,
		InputDigest:     DigestInput(in),
		ProposedVerb:    action.Kind(),
		ReasonToken:     reason,
		EnvelopeVerdict: verdict,
		Outcome:         outcome,
		ArtifactRef:     artifact,
		DecidedAt:       at,
	}
	if err := s.Journal.Append(rec); err != nil {
		return DecisionRecord{}, err
	}
	// The Journal owns the sequence: fetch back the stamped copy. Single
	// writer per journal (the doctrine's no-unshared-writers rule), so Last
	// is this wake's record.
	stamped, ok := s.Journal.Last()
	if !ok {
		return DecisionRecord{}, errors.New("supervisoragent: journal lost the just-appended record")
	}
	return stamped, nil
}

// artifactRef names the witnessed artifact a lowered verb left, in the closed
// "verb:identity" shape the journal examples use ("admit:RID-2#4480"). Hold
// leaves none, and neither does any refused path — those records carry an
// empty ArtifactRef by construction.
func artifactRef(eff ActionEffect) string {
	switch eff.Verb {
	case VerbArbitrate:
		if eff.Lease != nil {
			return "arbitrate:" + eff.Lease.Lane
		}
	case VerbAdmit:
		if eff.Admit != nil {
			return "admit:" + eff.Admit.RunID + "#" + eff.Admit.Issue
		}
	case VerbEmitEscalation:
		if eff.Packet != nil {
			return "escalation:" + eff.Packet.ID
		}
	}
	return ""
}
