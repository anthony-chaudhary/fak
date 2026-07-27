package dispatchtick

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// The input `modelroute.AfterAttempt` was missing (epic #5416 track D).
//
// AfterAttempt decides whether a finished attempt stands, needs checking, or has earned a
// retry one rung up — but it takes a `FailureKind` and never infers one, because only the
// caller can tell a model that tried and could not from a socket that never opened. Nothing
// produced that classification, so the rule had no live input. This is that producer, over
// the one failure record this repo already computes without asking the worker: the witness
// sweep.
//
// The success half is NOT decided here. `witnessOutcome` already grades a slot into
// (success, provenance) for capability evidence, and this reuses it verbatim. A second
// opinion about what counts as success is the specific defect worth avoiding: the same slot
// would be a failure for grading and a success for spending, and a fleet would bank a model's
// result in one ledger while buying a bigger model over it in the other. One grader, two
// readers.
//
// What this file adds is the failure kind, and it is mostly a set of refusals to escalate:
//
//	guard refusal (self-modify / policy block / off-trunk)  ->  FailRefused   (stop)
//	auth wall, capacity wall, unknown model, banner no-op   ->  FailTransport (retry in place)
//	a test that RAN and FAILED                              ->  FailUnderpowered (earns a rung)
//	anything else                                           ->  FailUnclassified (stop)
//
// Three of those are load-bearing in the direction of NOT spending.
//
// The refusal check runs FIRST, before any other field is read, so no combination of a
// guard refusal with anything else can produce an escalation. AfterAttempt's rule 3 checks
// refusals before every bound; a classifier that let a refused slot in as underpowered would
// route around that rule while leaving it visibly intact.
//
// The model-switchable trio (usage cap / rate limit / unknown model) is transport, not
// capability, and that is the most tempting wrong mapping in the file. A full local pool
// FEELS like the moment to reach for a frontier model, but a capacity wall says nothing
// about whether the work was too hard — it says the rung was busy. Filing it as underpowered
// would convert every weekly-bucket rollover into vendor spend, and would do it precisely
// when the fleet is busiest, which is when the bill is already highest. The existing Layer-2
// downgrade already handles these by switching to the next chain model on the SAME rung.
// An auth wall and a banner no-op are the same shape: no model ever saw the work.
//
// A test that ran and failed is the only thing that earns a rung, because it is the only
// evidence in the sweep that a model actually attempted the work and actually did not
// deliver it — independently observed, not self-reported. The assumption in that mapping is
// worth stating plainly rather than burying: on a shared tree a package can be red before
// the worker touched it, and nothing in the record separates "this model wrote failing code"
// from "this package was already failing". That ambiguity is not resolved here and cannot
// be; it is why escalation is bounded by an operator's explicitly declared ceiling and
// attempt budget instead of by this function's confidence.
//
// Two things this deliberately does not emit:
//
// CLAIM_UNWITNESSED without a red test is unclassified, not underpowered. It looks like the
// clearest capability failure in the vocabulary — the model committed something that does
// not match its own claim — but `dos commit-audit` also abstains on a vague commit subject,
// so the bucket holds "the model could not do the work" and "the model described the work
// badly" together. Escalating it buys a frontier model because a worker wrote a poor commit
// subject.
//
// FailWorkItem is never emitted. It is a real state — work no rung can complete — but
// nothing in a witness record distinguishes it from an underpowered attempt, and inventing
// the distinction would produce a confident stop on work a bigger model could have done.
// It stays a kind the vocabulary can express and this producer cannot claim, which an
// operator sees as an item that failed on every rung until its budget ran out.

// AttemptResultFor grades one finished worker slot into the attempt result
// `modelroute.AfterAttempt` decides on.
//
// Pure: it reads only the record. A slot this cannot classify becomes an unclassified
// FAILURE rather than an absent one, since AfterAttempt stops on unclassified and the
// alternative — reporting no failure at all — would read as a success.
func AttemptResultFor(r WitnessRecord) modelroute.AttemptResult {
	ok, verify := witnessOutcome(r)
	if ok {
		return modelroute.AttemptResult{Succeeded: true, Verify: verify}
	}
	return modelroute.AttemptResult{Verify: verify, Fail: failKindFor(r)}
}

// failKindFor names why a graded slot did not succeed. Order is the contract: refusals are
// read before any evidence that could earn a rung.
func failKindFor(r WitnessRecord) modelroute.FailureKind {
	if kind, named := failKindForNoCommitReason(r.Reason); named {
		return kind
	}
	if strings.TrimSpace(r.TestClaim) == ClaimTestRed {
		return modelroute.FailUnderpowered
	}
	return modelroute.FailUnclassified
}

// failKindForNoCommitReason maps the structured no-commit reason. named is false only for an
// empty reason (a slot that committed, so the reason field was never set); every other
// value — including one this build does not recognise — is named, so a reason string from a
// newer producer stops the ladder instead of falling through to the rung-earning branch.
func failKindForNoCommitReason(reason string) (modelroute.FailureKind, bool) {
	switch strings.TrimSpace(reason) {
	case "":
		return modelroute.FailNone, false
	case NoCommitSelfModify, NoCommitPolicyBlock, NoCommitOffTrunk:
		// A guard said no. Re-issuing the same work at a rung that guard may not cover is a
		// bypass by retry, and re-issuing it at the SAME rung hits the identical refusal, so
		// neither direction is a retry worth making.
		return modelroute.FailRefused, true
	case NoCommitAuthWall, NoCommitUsageCap, NoCommitModelUnknown, NoCommitRateLimit, NoCommitBannerNoop:
		// Nothing reached a model that could try the work: a login/entitlement wall, a
		// capacity or weekly-bucket wall, an id the account is not entitled to, or a worker
		// that printed its banner and stopped. None of them is evidence about capability.
		return modelroute.FailTransport, true
	default:
		return modelroute.FailUnclassified, true
	}
}
