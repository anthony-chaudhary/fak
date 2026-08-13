// Package assumecheck is the pure assumption-audit kernel (#3819, epic #3818 C1):
// "an assumption an agent (or operator) is relying on" as a first-class, checkable
// value instead of an unexamined belief baked into a prompt or a loop.
//
// This package is deliberately the SAME SHAPE as internal/recall's stale-fact gate
// (recall/stalefact.go, #1594) and ctxplan's page-fault decision: a pure function of
// (assumption, evidence) -> EXACTLY ONE closed-vocabulary outcome, never a silent
// no-decision, replayable because it is a pure function of its inputs; plus a
// fail-closed hard gate that returns a typed fault instead of ever letting a caller
// proceed on an assumption that did not positively hold. No I/O, no clock reads, no
// hidden state live here — evidence is GATHERED by an impure shell (cmd/fak/assume.go
// for the CLI spine) and handed in as data.
//
// The C1 spine freezes the three closed vocabularies (Level, WitnessKind, Outcome)
// and ships exactly one real assumption end-to-end (SeatLaunchable, witnessed by the
// `fak accounts next` rotation authority). The registry of many assumptions (C2),
// witness-driver plurality (C3), refusal-reason token binding (C4), the re-check loop
// (C5) and the scorecard (C6) are follow-ons on epic #3818 and are OUT of this file
// by design.
package assumecheck

import (
	"errors"
	"fmt"
)

// Level is the CLOSED scope vocabulary for how far an assumption reaches — the blast
// radius of being wrong about it. Mirrors the closed-string-type + membership-set +
// fail-closed-String() shape of recall.StaleFactOutcome.
type Level string

const (
	// LevelSession — holds only within one agent session (e.g. "my transcript is the
	// one I resumed").
	LevelSession Level = "session"
	// LevelSubsystem — holds about one subsystem's state (e.g. "the gateway is serving").
	LevelSubsystem Level = "subsystem"
	// LevelLoop — holds about a dispatch/monitor loop's premises (e.g. "the picker's
	// issue list is current").
	LevelLoop Level = "loop"
	// LevelInfra — holds about host/fleet infrastructure (e.g. "this seat is launchable").
	LevelInfra Level = "infra"
	// LevelOperator — holds about operator-supplied intent (e.g. "the operator still
	// wants this standing order").
	LevelOperator Level = "operator"
)

// validLevels is the membership set every Level must belong to — used by tests and
// any (de)serializing caller to fail closed on a corrupt or foreign value.
var validLevels = map[Level]bool{
	LevelSession:   true,
	LevelSubsystem: true,
	LevelLoop:      true,
	LevelInfra:     true,
	LevelOperator:  true,
}

// ValidLevel reports whether l is a member of the closed vocabulary.
func ValidLevel(l Level) bool { return validLevels[l] }

func (l Level) String() string {
	if ValidLevel(l) {
		return string(l)
	}
	if l == "" {
		return "(unset)"
	}
	return "unknown(" + string(l) + ")"
}

// WitnessKind is the CLOSED vocabulary of evidence sources an assumption can declare.
// The kinds are ordered by forgeability in spirit (git ancestry is non-forgeable;
// a session report is a self-report — the least trustworthy signal in the stack, per
// the dos_commit_audit "subject-only" rung), and Check enforces that ordering where
// it matters: a session-report can refute an assumption (an admission against
// interest is credible) but can never positively confirm one.
type WitnessKind string

const (
	// WitnessGitAncestry — git merge-base / log evidence (non-forgeable: git recorded it).
	WitnessGitAncestry WitnessKind = "git-ancestry"
	// WitnessWorktreeGrep — a comment-aware working-tree content probe.
	WitnessWorktreeGrep WitnessKind = "worktree-grep"
	// WitnessCommandProbe — a live command's observed output/exit code.
	WitnessCommandProbe WitnessKind = "command-probe"
	// WitnessConfigFlag — a declared config/flag value read from its source of truth.
	WitnessConfigFlag WitnessKind = "config-flag"
	// WitnessLedgerRead — a read of a durable ledger/registry/roster the fleet
	// already trusts (e.g. the accounts registry + rotation plan behind
	// `fak accounts next`).
	WitnessLedgerRead WitnessKind = "ledger-read"
	// WitnessSessionReport — an agent's own narration. Emittable, but Check never
	// lets it positively confirm an assumption (see Check rule 4).
	WitnessSessionReport WitnessKind = "session-report"
)

// validWitnessKinds is the membership set every WitnessKind must belong to.
var validWitnessKinds = map[WitnessKind]bool{
	WitnessGitAncestry:   true,
	WitnessWorktreeGrep:  true,
	WitnessCommandProbe:  true,
	WitnessConfigFlag:    true,
	WitnessLedgerRead:    true,
	WitnessSessionReport: true,
}

// ValidWitnessKind reports whether k is a member of the closed vocabulary.
func ValidWitnessKind(k WitnessKind) bool { return validWitnessKinds[k] }

func (k WitnessKind) String() string {
	if ValidWitnessKind(k) {
		return string(k)
	}
	if k == "" {
		return "(unset)"
	}
	return "unknown(" + string(k) + ")"
}

// Outcome is the CLOSED disposition vocabulary for one assumption check. Exactly one
// outcome is produced per Check; there is no "no decision" state once Check runs —
// UNVERIFIABLE is the explicit "cannot witness" branch, never a silent fallthrough.
type Outcome string

const (
	// OutcomeHolds — the declared witness ran and positively confirmed the assumption.
	// The only outcome GuardAssumption lets a caller proceed on.
	OutcomeHolds Outcome = "HOLDS"
	// OutcomeViolated — the declared witness ran and the assumed condition is FALSE.
	OutcomeViolated Outcome = "VIOLATED"
	// OutcomeUnverifiable — no decision could be witnessed: the witness could not run,
	// the evidence came from the wrong witness kind, or the only evidence is a
	// self-report claiming success. Fail-closed: unverifiable is NOT holds.
	OutcomeUnverifiable Outcome = "UNVERIFIABLE"
	// OutcomeStale — the witness once produced a decision but the evidence has aged
	// past its declared freshness bound; re-witness before relying on it.
	OutcomeStale Outcome = "STALE"
)

// validOutcomes is the membership set every Verdict.Outcome belongs to.
var validOutcomes = map[Outcome]bool{
	OutcomeHolds:        true,
	OutcomeViolated:     true,
	OutcomeUnverifiable: true,
	OutcomeStale:        true,
}

// ValidOutcome reports whether o is a member of the closed vocabulary.
func ValidOutcome(o Outcome) bool { return validOutcomes[o] }

func (o Outcome) String() string {
	if ValidOutcome(o) {
		return string(o)
	}
	if o == "" {
		return "(unset)"
	}
	return "unknown(" + string(o) + ")"
}

// blocksReliance reports whether this outcome must stop a caller from relying on the
// assumption — i.e. every outcome except OutcomeHolds (fail-closed: an unverifiable
// or stale assumption is exactly as unsafe to act on as a violated one).
func (o Outcome) blocksReliance() bool { return o != OutcomeHolds }

// RefusalReason maps an outcome onto the closed, dos-registered refusal token a
// refusing caller emits (#3822, epic #3818 C4): the OUTCOME CLASS — not the
// per-assumption label — is what `dos man wedge <TOKEN> --explain` resolves against the workspace
// dos.toml [reasons] vocabulary. Total over the blocking set; OutcomeHolds (and any
// foreign value) maps to "" because a holding assumption refuses nothing.
func (o Outcome) RefusalReason() string {
	switch o {
	case OutcomeViolated:
		return "ASSUMPTION_VIOLATED"
	case OutcomeUnverifiable:
		return "ASSUMPTION_UNVERIFIABLE"
	case OutcomeStale:
		return "ASSUMPTION_STALE"
	}
	return ""
}

// KnownAssumptionRefusalReasons enumerates the closed refusal-reason vocabulary in a
// stable order — one token per blocking outcome — mirroring toon.KnownSkipReasons:
// the dosreasons binding test iterates this set against the dos.toml [reasons]
// registration, so the Go floor and the workspace vocabulary cannot drift apart.
func KnownAssumptionRefusalReasons() []string {
	return []string{
		OutcomeViolated.RefusalReason(),
		OutcomeUnverifiable.RefusalReason(),
		OutcomeStale.RefusalReason(),
	}
}

// Assumption is one declared, checkable belief: who relies on it, what it claims,
// how far it reaches, and which witness kind is allowed to adjudicate it. It is pure
// data — the C2 registry will hold many of these; the C1 spine ships exactly one
// (SeatLaunchable).
type Assumption struct {
	// ID is the stable slug the assumption is addressed by (e.g. "seat-launchable").
	ID string `json:"id"`
	// Owner names the subsystem/loop that relies on the assumption.
	Owner string `json:"owner"`
	// Statement is the operator-readable claim being assumed.
	Statement string `json:"statement"`
	// Level is the closed scope class (see Level).
	Level Level `json:"level"`
	// WitnessKind is the ONLY evidence kind allowed to adjudicate this assumption.
	// Evidence of any other kind yields OutcomeUnverifiable, never a cross-witness
	// guess.
	WitnessKind WitnessKind `json:"witness_kind"`
	// RefusalReason is the per-assumption label naming WHICH assumption blocked
	// (e.g. SEAT_NOT_LAUNCHABLE). It is NOT the dos-resolvable refusal reason: the
	// token a refusing caller emits is the OUTCOME-CLASS Outcome.RefusalReason()
	// (#3822 C4); GuardAssumption folds this label into the verdict's reason text
	// as detail so it is not lost.
	RefusalReason string `json:"refusal_reason,omitempty"`
	// ConfidenceClass labels how the assumption's holder rates it (informational;
	// e.g. "witnessed", "declared", "folk"). Not part of the decision.
	ConfidenceClass string `json:"confidence_class,omitempty"`
	// WitnessStatus is the per-row wiring marker (registry.go): "wired" when an
	// evidence gatherer exists for this assumption, "declared-only" when the row is
	// registered but has no driver until witness plurality (#3818 C3). Data, not
	// decision — Check never reads it; the shell uses it to EXPLAIN an
	// UNVERIFIABLE on a declared-only row instead of returning one silently.
	WitnessStatus WitnessStatus `json:"witness_status,omitempty"`
}

// Evidence is what an impure witness gatherer hands the pure kernel: which witness
// produced it, whether that witness could produce a decision at all, the decision,
// and how fresh it is. The kernel never gathers — it only judges what it is handed,
// so the same (Assumption, Evidence) always reproduces the same Verdict.
type Evidence struct {
	// Kind is the witness kind that produced this evidence. Must equal the
	// assumption's declared WitnessKind or the check is UNVERIFIABLE.
	Kind WitnessKind `json:"kind"`
	// Witnessed reports whether the witness RAN and produced a decision (true even
	// when the decision is "does not hold"). False means the witness could not
	// adjudicate — source unreadable, premise absent, probe failed.
	Witnessed bool `json:"witnessed"`
	// Holds is the witnessed decision: does the assumed condition hold right now?
	// Meaningful only when Witnessed is true.
	Holds bool `json:"holds"`
	// AgeSeconds is how old the evidence is at check time (0 = gathered now). The
	// caller computes it; the kernel reads no clock.
	AgeSeconds int64 `json:"age_seconds,omitempty"`
	// MaxAgeSeconds is the freshness bound the evidence must satisfy (0 = no bound).
	// Evidence older than this is OutcomeStale regardless of what it says.
	MaxAgeSeconds int64 `json:"max_age_seconds,omitempty"`
	// Detail is the operator-readable trace of what the witness actually saw.
	Detail string `json:"detail,omitempty"`
}

// Verdict is the typed decision Check produces: the outcome plus the reason and the
// identity it was computed about, so a decision is self-describing (mirrors
// recall.StaleFactDecision).
type Verdict struct {
	AssumptionID string      `json:"assumption_id"`
	Level        Level       `json:"level"`
	Witness      WitnessKind `json:"witness"`
	Outcome      Outcome     `json:"outcome"`
	Reason       string      `json:"reason"`
}

// Check is the PURE decision function: given an Assumption and the Evidence a
// witness gathered about it, it returns EXACTLY ONE closed-vocabulary outcome. No
// clock read, no I/O, no hidden state.
//
// Decision order (first match wins, most conservative first — mirrors
// recall.DetectStaleFact's structure):
//
//  1. Evidence from a witness kind the assumption did not declare (including an
//     invalid/unset kind on either side) -> OutcomeUnverifiable. A mismatched
//     witness cannot confirm OR refute the assumption — judging it anyway would be
//     the cross-witness guess the closed vocabulary exists to prevent.
//  2. The witness could not produce a decision (Witnessed=false) -> OutcomeUnverifiable,
//     the explicit "cannot witness" branch.
//  3. The evidence has aged past its declared freshness bound (MaxAgeSeconds>0 and
//     AgeSeconds>MaxAgeSeconds) -> OutcomeStale, whatever the decision was: a lapsed
//     observation is not a current fact (the recall/stalefact posture).
//  4. A session-report claiming the assumption HOLDS -> OutcomeUnverifiable. A
//     self-report is forgeable, so it can never positively confirm (the
//     dos_commit_audit "subject-only" rung); a session-report ADMITTING violation
//     still falls through to rule 5 and refutes — an admission against interest is
//     credible.
//  5. Holds -> OutcomeHolds; otherwise OutcomeViolated.
func Check(a Assumption, ev Evidence) Verdict {
	v := Verdict{AssumptionID: a.ID, Level: a.Level, Witness: ev.Kind}

	if !ValidWitnessKind(a.WitnessKind) || !ValidWitnessKind(ev.Kind) || ev.Kind != a.WitnessKind {
		v.Outcome = OutcomeUnverifiable
		v.Reason = fmt.Sprintf("evidence kind %s cannot adjudicate an assumption declaring witness %s: refusing the cross-witness guess", ev.Kind, a.WitnessKind)
		return v
	}

	if !ev.Witnessed {
		v.Outcome = OutcomeUnverifiable
		v.Reason = "the declared witness could not produce a decision" + detailSuffix(ev.Detail)
		return v
	}

	if ev.MaxAgeSeconds > 0 && ev.AgeSeconds > ev.MaxAgeSeconds {
		v.Outcome = OutcomeStale
		v.Reason = fmt.Sprintf("evidence is %ds old, past its %ds freshness bound: re-witness before relying on it", ev.AgeSeconds, ev.MaxAgeSeconds)
		return v
	}

	if ev.Kind == WitnessSessionReport && ev.Holds {
		v.Outcome = OutcomeUnverifiable
		v.Reason = "a session self-report claims the assumption holds, but a self-report cannot positively confirm: unverifiable until a stronger witness corroborates" + detailSuffix(ev.Detail)
		return v
	}

	if ev.Holds {
		v.Outcome = OutcomeHolds
		v.Reason = "the declared witness confirmed the assumption" + detailSuffix(ev.Detail)
		return v
	}
	v.Outcome = OutcomeViolated
	v.Reason = "the declared witness refuted the assumption" + detailSuffix(ev.Detail)
	return v
}

// detailSuffix renders an evidence detail into a reason tail (": <detail>"), or
// nothing when the witness left none — so reasons stay one self-contained sentence.
func detailSuffix(detail string) string {
	if detail == "" {
		return ""
	}
	return ": " + detail
}

// ErrAssumptionViolated is the sentinel GuardAssumption's returned error wraps, so a
// caller can branch with errors.Is without parsing AssumptionViolationError's message
// — the same errors.Is contract recall.ErrStaleFactAsCurrent gives its callers.
var ErrAssumptionViolated = errors.New("assumecheck: assumption refused as unsafe to rely on")

// AssumptionViolationError is the TYPED FAULT GuardAssumption returns in place of
// ever letting a caller proceed on a non-holding assumption. The closed Verdict
// travels with it, so a caller can branch on Outcome (violated/unverifiable/stale)
// instead of parsing Error()'s text, and RefusalReason carries the OUTCOME-CLASS
// token from the closed DOS refusal vocabulary (Outcome.RefusalReason, #3822 C4)
// for the caller that must refuse loudly; the per-assumption label travels folded
// into Verdict.Reason.
type AssumptionViolationError struct {
	Verdict       Verdict
	RefusalReason string
}

func (e *AssumptionViolationError) Error() string {
	return fmt.Sprintf("%s: %s (%s): %s", ErrAssumptionViolated, e.Verdict.AssumptionID, e.Verdict.Outcome, e.Verdict.Reason)
}

func (e *AssumptionViolationError) Unwrap() error { return ErrAssumptionViolated }

// GuardAssumption is the HARD GATE at the reliance boundary: it either returns the
// holding Verdict with a nil error (safe to proceed on the assumption), or a non-nil
// *AssumptionViolationError wrapping ErrAssumptionViolated for EVERY other outcome.
// Fail-closed by construction — unlike recall.GuardAgainstStaleFact there is no
// Required knob to soften it: an assumption is by definition load-bearing for whoever
// declared it, so violated, unverifiable, and stale all refuse. A caller cannot
// accidentally ignore the decision and proceed anyway, because the success path only
// exists when the outcome is OutcomeHolds.
func GuardAssumption(a Assumption, ev Evidence) (Verdict, error) {
	v := Check(a, ev)
	if v.Outcome.blocksReliance() {
		if a.RefusalReason != "" {
			// Fold the per-assumption label into the verdict's reason as detail, so
			// naming WHICH assumption blocked survives the outcome-class token
			// replacing it as the emitted refusal reason (#3822 C4).
			v.Reason += " [assumption label: " + a.RefusalReason + "]"
		}
		return v, &AssumptionViolationError{Verdict: v, RefusalReason: v.Outcome.RefusalReason()}
	}
	return v, nil
}

// SeatLaunchable is the ONE real assumption the C1 spine wires end-to-end: a seat the
// config plane (`fak accounts doctor` / dispatch preflight) names launch-clean is
// actually launchable per the rotation authority behind `fak accounts next`
// (Registry.RotationPlanWithHeadroom + the runtime headroom/cooldown overlay). The
// witness gathering lives in cmd/fak/assume.go (impure shell); this is the pure
// declaration. The C2 registry (registry.go) holds this exact var as row 0, so the
// shell reference and the registry share one source of truth.
var SeatLaunchable = Assumption{
	ID:              "seat-launchable",
	Owner:           "accounts",
	Statement:       "a seat named launch-clean by `fak accounts doctor`/preflight is actually launchable per the `fak accounts next` rotation authority",
	Level:           LevelInfra,
	WitnessKind:     WitnessLedgerRead,
	RefusalReason:   "SEAT_NOT_LAUNCHABLE",
	ConfidenceClass: "witnessed",
	WitnessStatus:   WitnessWired,
}
