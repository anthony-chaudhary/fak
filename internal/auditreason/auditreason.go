// Package auditreason holds closed vocabularies for audit-facing failure
// surfaces: commit-audit verification failures, and non-guard tool failures
// such as hangs, timeouts, shell mismatches, and partial applies.
//
// # The transient-vs-permanent contract
//
// When a fleet worker claims it closed an issue, the commit-audit step tries to
// bind a resolving commit to that claim. When the audit CANNOT verify the
// commit, the failure falls into exactly one of two classes:
//
//   - TRANSIENT (retry-eligible): the audit was defeated by a passing condition
//     of the shared tree or the network — a held git lock, a lost push race
//     against a peer, an unreachable remote, a rebase in flight. The claim may
//     well be witnessed; the audit just could not run to completion right now.
//     Retrying later is expected to succeed. These are NOT drift.
//
//   - PERMANENT (not retryable): the audit ran to completion and found the claim
//     genuinely unwitnessed — no commit references the issue, the subject claims
//     work the diff does not witness, or no witness artifact exists. This is
//     real drift. Retrying will never turn an unwitnessed claim into a witnessed
//     one; the claim itself must be fixed. Reporting these as "retry later"
//     would HIDE the miss, so they must stay non-retryable.
//
// The mapping is deliberately conservative in one direction: an audit-failure
// message that matches no known transient signature is classified as
// ReasonUnknown, which is PERMANENT (non-retryable). A genuine miss must never
// be masked as a transient "try again later"; the cost of a spurious
// non-retryable is only a re-examination, while the cost of a spurious
// retryable is a silently-dropped drift signal.
//
// The package is deterministic and stdlib-only: it maps a raw failure message
// to a closed Reason by best-effort substring match, classifies the Reason, and
// renders a Report. It performs no I/O and imports nothing internal.
package auditreason

import "strings"

// Reason is a member of the closed vocabulary of commit-audit failure reasons.
// The zero value is ReasonUnknown, which is conservatively non-retryable.
type Reason string

const (
	// ReasonUnknown is the conservative default for any failure message that
	// matches no known signature. It is PERMANENT (non-retryable) so a genuine
	// unwitnessed claim is never masked as "retry later".
	ReasonUnknown Reason = "unknown"

	// --- TRANSIENT (retry-eligible) reasons ---

	// ReasonLockBusy: a git lock (index.lock, packed-refs.lock, the fak commit
	// lane lock) was held by a concurrent process, so the audit could not read
	// or write refs. Expected to clear.
	ReasonLockBusy Reason = "lock_busy"

	// ReasonPushRejectedRace: a concurrent peer push advanced the remote between
	// fetch and push, so this push was rejected (non-fast-forward). A re-pull +
	// retry is expected to succeed.
	ReasonPushRejectedRace Reason = "push_rejected_race"

	// ReasonRemoteUnreachable: the remote could not be reached (network error,
	// DNS, timeout, connection refused). Expected to be transient.
	ReasonRemoteUnreachable Reason = "remote_unreachable"

	// ReasonRebaseInFlight: a rebase / merge was in progress in the shared tree,
	// so the audit could not resolve a stable HEAD. Expected to clear once the
	// operation completes.
	ReasonRebaseInFlight Reason = "rebase_in_flight"

	// --- PERMANENT (not retryable) reasons — genuine unwitnessed claim ---

	// ReasonNoCommitFound: no commit in the audited range references the issue.
	// The claim is unwitnessed; retrying will not create the commit.
	ReasonNoCommitFound Reason = "no_commit_found"

	// ReasonSubjectDiffMismatch: a commit references the issue but its subject
	// claims work the diff does not witness (subject says X, the diff does not
	// show X). Real drift; the commit must be fixed.
	ReasonSubjectDiffMismatch Reason = "subject_diff_mismatch"

	// ReasonNoWitnessArtifact: the claim requires a witness artifact (a test, a
	// fixture, a recorded output) and none exists. Real drift.
	ReasonNoWitnessArtifact Reason = "no_witness_artifact"
)

// Class is the retry disposition of a Reason.
type Class string

const (
	// Transient failures are retry-eligible: the audit was defeated by a
	// passing tree/network condition, not by a genuine unwitnessed claim.
	Transient Class = "transient"

	// Permanent failures are not retryable: the claim is genuinely unwitnessed
	// (real drift), so retrying cannot help.
	Permanent Class = "permanent"
)

// transientReasons is the closed set of retry-eligible reasons. Every Reason NOT
// in this set — including ReasonUnknown — is Permanent by construction, so a new
// or unrecognized failure defaults to non-retryable.
var transientReasons = map[Reason]bool{
	ReasonLockBusy:          true,
	ReasonPushRejectedRace:  true,
	ReasonRemoteUnreachable: true,
	ReasonRebaseInFlight:    true,
}

// Classify returns the retry disposition of a reason. Any reason not in the
// closed transient set is Permanent — this is what makes ReasonUnknown, and any
// future permanent reason, conservatively non-retryable.
func Classify(reason Reason) Class {
	if transientReasons[reason] {
		return Transient
	}
	return Permanent
}

// RetryEligible reports whether a reason is worth retrying. It is true exactly
// for the transient class.
func RetryEligible(reason Reason) bool {
	return Classify(reason) == Transient
}

// signature pairs a lowercased substring found in raw audit-failure messages
// with the closed Reason it maps to. Transient signatures are matched first so a
// message that mentions both a lock and a missing commit is treated as the
// transient (retryable) condition rather than prematurely declared drift.
type signature struct {
	needle string
	reason Reason
}

// transientSignatures map raw failure text to the retry-eligible reasons. These
// are matched BEFORE permanent signatures.
var transientSignatures = []signature{
	{"unable to create", ReasonLockBusy},
	{".lock': file exists", ReasonLockBusy},
	{"lock exists", ReasonLockBusy},
	{"index.lock", ReasonLockBusy},
	{"packed-refs.lock", ReasonLockBusy},
	{"another git process", ReasonLockBusy},
	{"lock_busy", ReasonLockBusy},
	{"lock busy", ReasonLockBusy},

	{"non-fast-forward", ReasonPushRejectedRace},
	{"fetch first", ReasonPushRejectedRace},
	{"tip of your current branch is behind", ReasonPushRejectedRace},
	{"failed to push some refs", ReasonPushRejectedRace},
	{"updates were rejected", ReasonPushRejectedRace},
	{"push_rejected_race", ReasonPushRejectedRace},

	{"could not resolve host", ReasonRemoteUnreachable},
	{"connection timed out", ReasonRemoteUnreachable},
	{"connection refused", ReasonRemoteUnreachable},
	{"unable to access", ReasonRemoteUnreachable},
	{"network is unreachable", ReasonRemoteUnreachable},
	{"remote_unreachable", ReasonRemoteUnreachable},

	{"rebase in progress", ReasonRebaseInFlight},
	{"you are in the middle of a rebase", ReasonRebaseInFlight},
	{"merge in progress", ReasonRebaseInFlight},
	{"rebase_head", ReasonRebaseInFlight},
	{"rebase_in_flight", ReasonRebaseInFlight},
}

// permanentSignatures map raw failure text to the genuine-drift reasons. These
// are matched AFTER the transient signatures.
var permanentSignatures = []signature{
	{"no commit found", ReasonNoCommitFound},
	{"no commit references", ReasonNoCommitFound},
	{"not_shipped", ReasonNoCommitFound},
	{"no resolving commit", ReasonNoCommitFound},
	{"no_commit_found", ReasonNoCommitFound},

	{"subject claims", ReasonSubjectDiffMismatch},
	{"diff does not witness", ReasonSubjectDiffMismatch},
	{"subject/diff mismatch", ReasonSubjectDiffMismatch},
	{"subject does not match diff", ReasonSubjectDiffMismatch},
	{"subject_diff_mismatch", ReasonSubjectDiffMismatch},

	{"no witness artifact", ReasonNoWitnessArtifact},
	{"missing witness", ReasonNoWitnessArtifact},
	{"unwitnessed", ReasonNoWitnessArtifact},
	{"no_witness_artifact", ReasonNoWitnessArtifact},
}

// FromMessage maps a raw audit-failure message to the closed Reason by
// best-effort, case-insensitive substring match. Transient signatures are
// checked before permanent ones, so a message that mentions both a passing
// condition and a miss is treated as retryable rather than prematurely declared
// drift. A message that matches no known signature returns ReasonUnknown, which
// Classify reports as Permanent — a genuine miss is never masked as retryable.
func FromMessage(msg string) Reason {
	low := strings.ToLower(msg)
	for _, s := range transientSignatures {
		if strings.Contains(low, s.needle) {
			return s.reason
		}
	}
	for _, s := range permanentSignatures {
		if strings.Contains(low, s.needle) {
			return s.reason
		}
	}
	return ReasonUnknown
}

// FromError is the error-typed convenience wrapper over FromMessage. A nil error
// maps to ReasonUnknown (there is no failure to classify).
func FromError(err error) Reason {
	if err == nil {
		return ReasonUnknown
	}
	return FromMessage(err.Error())
}

// Report is the closed-schema result of examining one commit-audit failure: the
// mapped Reason, its Class, whether it is retry-eligible, and a free-form Detail
// carrying the raw message for the operator.
type Report struct {
	Reason        Reason
	Class         Class
	RetryEligible bool
	Detail        string
}

// Examine maps a raw audit-failure message to a fully-classified Report.
func Examine(msg string) Report {
	r := FromMessage(msg)
	return Report{
		Reason:        r,
		Class:         Classify(r),
		RetryEligible: RetryEligible(r),
		Detail:        msg,
	}
}

// ExamineError is the error-typed convenience wrapper over Examine.
func ExamineError(err error) Report {
	if err == nil {
		return Examine("")
	}
	return Examine(err.Error())
}

// String renders the Report as a single deterministic line naming the reason,
// its class, and its retry eligibility, with the detail appended when present.
// The retry token is spelled RETRYABLE / NOT-RETRYABLE so the closure surface
// can be scanned by eye or grep.
func (r Report) String() string {
	retry := "NOT-RETRYABLE"
	if r.RetryEligible {
		retry = "RETRYABLE"
	}
	base := "audit-failure reason=" + string(r.Reason) +
		" class=" + string(r.Class) +
		" " + retry
	if r.Detail != "" {
		base += " detail=" + r.Detail
	}
	return base
}
