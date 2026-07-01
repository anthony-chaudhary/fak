package recall

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// stalefact.go — issue #1594: managed-context needs a runtime, not just a vocabulary,
// gate for "an old recalled fact is being treated as current state." Rung 1 (#82,
// CONTEXT-IS-NOT-MEMORY.md) already classifies write-time truth-duration
// (turn/session/durable, ctxmmu.DurabilityKey) and stamps it on Page.Durability; the
// bounded-validity half of rung 2 (#81) already ships ValidTo + ErrExpired on
// Session.Resolve/Recall. What was missing is the DECISION a caller makes the instant
// a recalled Page is about to be handed into ACTION CONTEXT: is this fact still safe
// to treat as current, or would admitting it launder a stale observation into a
// present-tense belief (the "it's 3pm" failure CONTEXT-IS-NOT-MEMORY.md §4 names)?
//
// This file is deliberately the SAME SHAPE as ctxplan.PageFaultOutcome
// (internal/ctxplan/pagefault.go, #1587): a pure function of (fact, policy) -> exactly
// one closed-vocabulary outcome, never a silent no-decision, replayable because it is a
// pure function of its inputs. It reuses recall's OWN vocabulary rather than inventing a
// parallel one: Page.Durability (rung 1's turn/session/durable classes, ctxmmu constants),
// Page.ValidTo (rung 2's bounded-validity tick, recall.go), and the asOf tick idiom
// Session.Resolve/Recall already thread through every read path. No new fact type, no
// new time representation.
//
// THE AXIS THIS CLOSES. validityGate (recall.go) already refuses a page read after its
// ValidTo tick — but only for the "bounded" class, and only as a binary allow/ErrExpired
// at the CAS-fetch boundary. It has no notion of "this is a turn-scoped or session-scoped
// fact being recalled in a LATER turn/session" (a fact that never carried an explicit
// ValidTo can still be stale-as-current the moment asOf has moved past the turn/session
// it was recorded in), and it returns a single bare error rather than a closed decision a
// caller can branch on (refresh vs ask vs refuse). DetectStaleFact generalizes the check
// across all durability classes and returns a typed StaleFactDecision; GuardAgainstStaleFact
// is the HARD GATE a caller uses at the action-context boundary: it returns a typed
// StaleFactError instead of ever handing back the recalled value when the decision is not
// StaleFactFresh and the fact is Required for the action about to run.

// StaleFactOutcome is the CLOSED vocabulary of dispositions for a recalled fact that
// might be stale-as-current — mirroring ctxplan.PageFaultOutcome's shape (closed string
// type, membership set, fail-closed String()). Exactly one outcome is produced per
// check; there is no "no decision" state once DetectStaleFact runs.
type StaleFactOutcome string

const (
	// StaleFactFresh: the fact's truth-duration window still covers asOf (a durable
	// fact, or a bounded/session/turn fact whose validity has not yet lapsed relative
	// to the caller's current tick). Safe to admit into action context unchanged.
	StaleFactFresh StaleFactOutcome = "fresh"
	// StaleFactExpiredNeedsRefresh: the fact's window has lapsed but the action does
	// not strictly require it to proceed (Required=false) — the safe move is to
	// refresh from the source before next relying on it, not to block now. Mirrors
	// StaleRecallRefreshSource (stale_recall.go): re-query the source of truth.
	StaleFactExpiredNeedsRefresh StaleFactOutcome = "expired_needs_refresh"
	// StaleFactExpiredMustQuery: the fact's window has lapsed AND the action requires
	// it, but the fact is refreshable (not a policy-carrying identity claim) — ask
	// before acting on it, exactly as ctxplan.PageFaultQueryUser asks rather than
	// assumes a reload is faithful. Chosen for a Required turn/session-scoped fact
	// recalled across a turn/session boundary with no hard ValidTo to enforce a deny.
	StaleFactExpiredMustQuery StaleFactOutcome = "expired_must_query"
	// StaleFactExpiredDeny: the fact carries an explicit, lapsed validity boundary
	// (bounded class + ValidTo) AND the action requires it — the same posture as
	// ErrExpired, but surfaced as a closed decision instead of a bare error so a
	// caller can distinguish "refuse outright" (a stated expiry was crossed) from
	// "ask first" (no stated expiry, just aged out of scope).
	StaleFactExpiredDeny StaleFactOutcome = "expired_deny"
)

// validStaleFactOutcomes is the membership set every StaleFactDecision.Outcome belongs
// to — used by tests and any (de)serializing caller to fail closed on a corrupt or
// foreign value.
var validStaleFactOutcomes = map[StaleFactOutcome]bool{
	StaleFactFresh:               true,
	StaleFactExpiredNeedsRefresh: true,
	StaleFactExpiredMustQuery:    true,
	StaleFactExpiredDeny:         true,
}

// ValidStaleFactOutcome reports whether o is a member of the closed vocabulary.
func ValidStaleFactOutcome(o StaleFactOutcome) bool { return validStaleFactOutcomes[o] }

func (o StaleFactOutcome) String() string {
	if ValidStaleFactOutcome(o) {
		return string(o)
	}
	if o == "" {
		return "(unset)"
	}
	return "unknown(" + string(o) + ")"
}

// blocksAction reports whether this outcome must stop a Required fact from reaching
// action context unrefreshed — i.e. every outcome except StaleFactFresh.
func (o StaleFactOutcome) blocksAction() bool { return o != StaleFactFresh }

// ErrStaleFactAsCurrent is the sentinel GuardAgainstStaleFact's returned error wraps,
// so a caller can branch with errors.Is without parsing StaleFactError's message —
// the same errors.Is contract ErrSealed/ErrExpired/ErrStale already give recall callers.
var ErrStaleFactAsCurrent = errors.New("recall: stale fact refused as current context")

// StaleFactCheck is the input DetectStaleFact decides over: the recalled page plus the
// two facts only the CALLER can supply, exactly mirroring ctxplan.PageFaultEvent's
// split between store-known state and caller-known intent.
type StaleFactCheck struct {
	// AsOf is the caller's current tick (the same idiom Session.Resolve/Recall
	// already accept as asOf ...int64) — "now," in whatever monotonic tick space the
	// caller's turn/session counter uses. 0 means "no current tick known," which
	// never expires a fact (mirrors validityGate's existing asOf<=0 no-op) — a
	// caller that never threads a clock gets the pre-existing behavior, not a new
	// false-positive.
	AsOf int64
	// Required marks that the action about to run treats this fact as load-bearing
	// current state (not a backgrounded aside) — the same Required flag
	// ctxplan.PageFaultEvent uses to force a refusal instead of a silent guess.
	Required bool
}

// StaleFactDecision is the typed transition DetectStaleFact produces: the outcome, the
// reason (operator-readable, mirroring ctxplan.PageFaultDecision.Reason), and the page
// step it was computed about, so a decision is self-describing.
type StaleFactDecision struct {
	Step       int              `json:"step"`
	Durability string           `json:"durability"`
	Outcome    StaleFactOutcome `json:"outcome"`
	Reason     string           `json:"reason"`
}

// DetectStaleFact is the PURE detection function: given a recalled Page and a
// StaleFactCheck (the caller's current tick + whether the action requires the fact),
// it returns EXACTLY ONE closed-vocabulary outcome. No clock read, no I/O, no hidden
// state — the same (Page, StaleFactCheck) always reproduces the same decision.
//
// Decision order (first match wins, most safety-critical first — mirrors
// ctxplan.DecidePageFault's structure):
//
//  1. No current tick known (AsOf<=0) -> StaleFactFresh. Mirrors validityGate's
//     existing asOf<=0 no-op: a caller that supplies no clock gets today's behavior,
//     never a new false-positive refusal.
//  2. bounded class with a stated ValidTo that AsOf has crossed:
//     - Required -> StaleFactExpiredDeny (a stated expiry was crossed; refuse
//     outright, the same posture ErrExpired already takes at the CAS boundary).
//     - not Required -> StaleFactExpiredNeedsRefresh (not blocking, but earn a
//     refresh before this is relied on again).
//  3. turn or session class fact where AsOf is beyond the page's own Step (i.e. the
//     fact is being recalled in a LATER turn than the one that produced it — the
//     exact "it's 3pm, recalled tomorrow" shape): a turn/session-scoped fact has NO
//     durability past its own turn/session by construction (CONTEXT-IS-NOT-MEMORY.md
//     §4), so recalling it as current in a later turn is stale-as-current even
//     though no explicit ValidTo was ever stamped.
//     - Required -> StaleFactExpiredMustQuery (ask before acting on it — the fact is
//     refreshable, unlike a policy-carrying identity claim, so a hard deny would be
//     overcautious; mirrors ctxplan.PageFaultQueryUser).
//     - not Required -> StaleFactExpiredNeedsRefresh.
//  4. durable class, or a bounded/turn/session fact still inside its window ->
//     StaleFactFresh.
func DetectStaleFact(p Page, chk StaleFactCheck) StaleFactDecision {
	class := promotionClassForStaleCheck(p.Durability)
	out := StaleFactDecision{Step: p.Step, Durability: class}

	if chk.AsOf <= 0 {
		out.Outcome = StaleFactFresh
		out.Reason = "no current tick supplied: cannot judge staleness, defaulting to fresh (mirrors the existing asOf<=0 no-op)"
		return out
	}

	if class == durabilityBounded && p.ValidTo > 0 && chk.AsOf > p.ValidTo {
		if chk.Required {
			out.Outcome = StaleFactExpiredDeny
			out.Reason = fmt.Sprintf("bounded fact's stated validity (valid_to=%d) is behind the current tick (as_of=%d) and the action requires it: refusing rather than acting on an expired boundary", p.ValidTo, chk.AsOf)
		} else {
			out.Outcome = StaleFactExpiredNeedsRefresh
			out.Reason = fmt.Sprintf("bounded fact's stated validity (valid_to=%d) is behind the current tick (as_of=%d) but the action does not require it: refresh before next reliance", p.ValidTo, chk.AsOf)
		}
		return out
	}

	if (class == ctxmmu.DurabilityTurn || class == ctxmmu.DurabilitySession) && chk.AsOf > int64(p.Step) {
		if chk.Required {
			out.Outcome = StaleFactExpiredMustQuery
			out.Reason = fmt.Sprintf("%s-scoped fact recorded at step %d is being recalled at a later tick (as_of=%d) and the action requires it: querying rather than assuming it still holds", class, p.Step, chk.AsOf)
		} else {
			out.Outcome = StaleFactExpiredNeedsRefresh
			out.Reason = fmt.Sprintf("%s-scoped fact recorded at step %d is being recalled at a later tick (as_of=%d) but the action does not require it: refresh before next reliance", class, p.Step, chk.AsOf)
		}
		return out
	}

	out.Outcome = StaleFactFresh
	out.Reason = "fact's truth-duration window covers the current tick: safe to admit as current"
	return out
}

// promotionClassForStaleCheck normalizes a raw Page.Durability value for the staleness
// check, failing closed to turn for anything unrecognized (mirrors promotionClass's
// posture) but ADDITIONALLY recognizing the local "bounded" class validityGate already
// uses, which promotionClass's rung-1 vocabulary does not carry.
func promotionClassForStaleCheck(raw string) string {
	switch raw {
	case ctxmmu.DurabilityDurable, ctxmmu.DurabilitySession, ctxmmu.DurabilityTurn, durabilityBounded:
		return raw
	default:
		return ctxmmu.DurabilityTurn
	}
}

// StaleFactError is the TYPED FAULT GuardAgainstStaleFact returns in place of ever
// handing back a stale-as-current recalled value. It is not a bare error string: the
// closed StaleFactDecision travels with it, so a caller can branch on Outcome
// (refresh/query/deny) instead of parsing Error()'s text, exactly as recall's other
// read-time guards (ErrSealed, ErrExpired, ErrStale) are branchable via errors.Is while
// still carrying a human-readable Error().
type StaleFactError struct {
	Decision StaleFactDecision
}

func (e *StaleFactError) Error() string {
	return fmt.Sprintf("%s: page %d (%s): %s", ErrStaleFactAsCurrent, e.Decision.Step, e.Decision.Outcome, e.Decision.Reason)
}

func (e *StaleFactError) Unwrap() error { return ErrStaleFactAsCurrent }

// GuardAgainstStaleFact is the HARD GATE at the action-context boundary: given a
// recalled Page and a StaleFactCheck, it either returns the fresh decision with a nil
// error (safe to admit the fact into action context), or a non-nil *StaleFactError
// wrapping ErrStaleFactAsCurrent whenever the fact is Required and DetectStaleFact
// produced anything other than StaleFactFresh. This is what makes the gate a REFUSAL,
// not a warning: a caller cannot accidentally ignore the decision and use the value
// anyway, because the value was never on the success path in the first place — the
// same "typed fault instead of allowing the recalled value into action context" shape
// fault_syndrome.go's ClassifyFault and rehydrate's Refuse already use.
//
// A non-Required stale fact still yields its decision (so a caller can log/refresh)
// but with a nil error: the done condition is about a fact "being used as current
// state," which by definition means the caller intends to rely on it now — a
// non-required, merely-observed fact is not being smuggled into action context, so it
// is not hard-gated. This mirrors DecidePageFault's own asymmetry: only the
// Required-and-unrecoverable path forces PageFaultDeny.
func GuardAgainstStaleFact(p Page, chk StaleFactCheck) (StaleFactDecision, error) {
	d := DetectStaleFact(p, chk)
	if chk.Required && d.Outcome.blocksAction() {
		return d, &StaleFactError{Decision: d}
	}
	return d, nil
}
