// Package attemptbudget is a pure fold over one issue's attempt history: given a
// bounded budget and the recorded attempts (each carrying the failure class it
// ended in), it decides whether the issue is still dispatchable, COOLING_DOWN
// under a failure-class-aware backoff window, or HELD for human triage -- so a
// repeatedly failing issue stops burning workers once it crosses the budget,
// instead of being re-offered forever (#1777), and so different kinds of
// failure cool down at different rates instead of all sharing one window
// (#1778). A held issue additionally carries a structured, queryable
// BlockReason -- same-error-repeated / distinct-errors / precondition-unmet /
// transient-exhausted -- and the Route that reason drives (retry, escalate,
// known-bad), so an operator can tell a genuinely stuck issue from a flaky one
// instead of reading a bare count (#2860). Crossing the budget is adjudicated,
// not blunt (#2892): a history whose LAST failure is transient (rate-limit,
// network flake) keeps retrying under its measured class backoff up to an
// extended transient ceiling, and only a structural or exhausted verdict --
// recorded on the Decision -- actually holds the issue. It never decides WHY
// an attempt failed; it only counts,
// classifies, and thresholds facts the caller already gathered. Pure: same Input
// in, same Decision out; zero I/O, zero clock reads -- the caller supplies "now"
// as data (Input.NowUnix), the same discipline internal/dispatchorder and
// internal/skipledger already use for clock-dependent folds.
package attemptbudget

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Status is the closed dispatchability verdict for one issue.
type Status string

const (
	StatusDispatchable Status = "dispatchable"
	// StatusCoolingDown means the issue is still under its attempt budget but
	// its last failure's class-specific backoff window has not yet elapsed --
	// distinct from StatusHeld, which is a hard stop on attempt count.
	StatusCoolingDown Status = "cooling_down"
	StatusHeld        Status = "held"
)

// FailureClass is the closed, caller-facing backoff vocabulary this package
// assigns a window to. A caller's raw Attempt.FailureClass string is mapped
// onto one of these (see classify); an unrecognized string maps to
// FailureClassOther, so an unknown failure class never crashes or gets
// silently treated as the least cautious window.
type FailureClass string

const (
	// FailureClassAuth: the attempt failed on an authentication/authorization
	// problem (bad/expired credentials, permission denied). These often need a
	// human to rotate a secret or grant access, so they get the LONGEST
	// default backoff -- retrying fast just burns another worker on the same
	// wall.
	FailureClassAuth FailureClass = "auth"
	// FailureClassMerge: the attempt failed on a merge/rebase conflict against
	// trunk. A moderate backoff gives concurrent peers time to land, after
	// which trunk has likely moved and a retry may no longer conflict.
	FailureClassMerge FailureClass = "merge"
	// FailureClassTest: the attempt failed a test run. A short default
	// backoff -- test failures are the most likely to be flaky or fixed by a
	// quick follow-up, so the issue should come back around soon.
	FailureClassTest FailureClass = "test"
	// FailureClassRateLimit: the attempt failed on provider/forge throttling
	// (a 429/529, "rate limit exceeded", "overloaded", quota exhaustion). The
	// SHORTEST backoff of all: nothing about the issue itself failed -- the
	// fleet just hit a shared capacity window that reopens on its own, usually
	// within minutes. Before this class existed these fell to
	// FailureClassOther's 1h window, holding an overload-throttled issue ~6x
	// longer than a flaky test (#1778's distinct-window rationale, applied to
	// the transient class high-concurrency fleets actually hit most).
	FailureClassRateLimit FailureClass = "rate_limit"
	// FailureClassNetwork: the attempt failed on a network flake (connection
	// reset/refused mid-run, DNS blip, unreachable host, broken socket) -- the
	// other self-clearing transient family (#2892), a short window just above
	// rate-limit's. A bare "timeout" deliberately does NOT classify here: a
	// timed-out attempt is as often a wedged worker as a slow wire, and reading
	// it as transient would defer a block the history has actually earned.
	FailureClassNetwork FailureClass = "network"
	// FailureClassAmbiguousScope: the attempt failed because the issue's scope
	// was unclear or contested (e.g. it collided with a concurrent peer's
	// area, or the worker could not determine the target package). A long
	// backoff, close to auth's, since re-dispatching immediately just repeats
	// the same ambiguity.
	FailureClassAmbiguousScope FailureClass = "ambiguous_scope"
	// FailureClassOther: any failure class the caller supplies that this
	// package does not recognize. Gets the default/moderate window rather
	// than being coerced into one of the named classes above.
	FailureClassOther FailureClass = "other"
)

// classify maps a caller-supplied raw failure-class string onto the closed
// FailureClass vocabulary via case-insensitive substring checks, so callers
// already using descriptive tags (internal/timeoutphase-style, e.g.
// "test_failure", "auth_error", "merge_conflict") don't have to pre-normalize
// onto attemptbudget's exact tokens.
func classify(raw string) FailureClass {
	low := strings.ToLower(raw)
	switch {
	// Rate-limit is checked BEFORE auth: throttling prose routinely mentions
	// authentication (GitHub's "API rate limit exceeded ... authenticated
	// requests get a higher rate limit" contains "auth"), and misreading a
	// reopening capacity window as a needs-a-human auth failure would cool the
	// issue 4h instead of minutes.
	case strmatch.ContainsAny(low, "rate limit", "rate_limit", "ratelimit", "429", "529", "overloaded", "too many requests", "quota"):
		return FailureClassRateLimit
	case strmatch.ContainsAny(low, "auth", "credential", "permission", "unauthorized", "forbidden"):
		return FailureClassAuth
	// Network is checked AFTER auth so a certificate/permission failure whose
	// prose also mentions the wire ("certificate signed by unknown authority",
	// "ssh connection denied: permission") keeps its needs-a-human reading --
	// misreading a config wall as a self-clearing flake would defer a block
	// forever -- but BEFORE merge/test, so "connection reset during test run"
	// reads as the infra flake it is.
	case strmatch.ContainsAny(low, "network", "connection", "dns", "unreachable", "socket", "broken pipe"):
		return FailureClassNetwork
	case strmatch.ContainsAny(low, "merge", "conflict", "rebase"):
		return FailureClassMerge
	case strmatch.ContainsAny(low, "test", "assert"):
		return FailureClassTest
	case strmatch.ContainsAny(low, "ambiguous", "scope"):
		return FailureClassAmbiguousScope
	default:
		return FailureClassOther
	}
}

// DefaultBackoffSeconds is the closed, total default policy: how long an
// issue cools down after its LAST recorded attempt, keyed by that attempt's
// classified FailureClass. Auth and ambiguous-scope failures usually need a
// human (rotate a credential, resolve a scope collision) so they cool down
// the longest; merge conflicts cool down long enough for a concurrent peer to
// land; test failures cool down briefly, since they are cheap to retry and
// often flaky; rate-limit/overload failures cool down the shortest of all,
// since the throttling window reopens on its own. Every FailureClass has an entry
// -- callers needing a different policy pass their own via Input.Backoff.
var DefaultBackoffSeconds = map[FailureClass]int64{
	FailureClassAuth:           4 * 3600, // 4h: needs a human to rotate/grant
	FailureClassMerge:          30 * 60,  // 30m: give trunk time to move
	FailureClassTest:           10 * 60,  // 10m: cheap to retry, often flaky
	FailureClassRateLimit:      5 * 60,   // 5m: a shared capacity window reopening on its own
	FailureClassNetwork:        6 * 60,   // 6m: a wire flake settling; above rate-limit (the documented shortest), below test
	FailureClassAmbiguousScope: 2 * 3600, // 2h: needs a human to resolve scope
	FailureClassOther:          60 * 60,  // 1h: moderate default
}

// transientClass reports whether a classified FailureClass is self-clearing
// (#2892): the failure says nothing about the issue itself -- the fleet hit a
// shared capacity window or a wire flake that reopens on its own. These are
// the classes whose budget crossing is deferred to the transient ceiling
// instead of holding on the raw count.
func transientClass(c FailureClass) bool {
	return c == FailureClassRateLimit || c == FailureClassNetwork
}

// defaultTransientBudgetMultiplier sizes the extended attempt ceiling a
// transient-last history earns when Input.TransientBudget is unset: Budget x
// this. Wide enough that a routine rate-limit window (a handful of throttled
// attempts under minutes-scale backoff) never exhausts it, small enough that a
// throttle which NEVER clears still blocks the issue in bounded attempts.
const defaultTransientBudgetMultiplier = 4

// transientCeiling resolves the extended attempt ceiling applied when the last
// recorded failure is transient: Input.TransientBudget when positive, else
// Budget x defaultTransientBudgetMultiplier. Setting TransientBudget equal to
// Budget deliberately restores the blunt hard stop for transient failures --
// except the block then carries the exhausted verdict, never a known-bad
// signature.
func transientCeiling(in Input) int {
	if in.TransientBudget > 0 {
		return in.TransientBudget
	}
	return in.Budget * defaultTransientBudgetMultiplier
}

// BlockReason is the closed, structured vocabulary explaining WHY an issue was
// auto-blocked (StatusHeld) once its failed attempts crossed the budget. A bare
// count says an issue spun; it does not say whether it is genuinely stuck or
// merely flaky, so an operator cannot tell the two apart (#2860). The reason is
// derived from the whole attempt history, not just the last attempt, and it is
// what Route dispatches on.
type BlockReason string

const (
	// BlockReasonSameErrorRepeated: every recorded attempt failed in the SAME
	// classified FailureClass, at least twice -- a stable failure signature. The
	// issue is genuinely stuck: another worker will hit the same wall, so the
	// signature belongs in the known-bad ledger (internal/knownbad) rather than
	// back in the dispatch queue.
	BlockReasonSameErrorRepeated BlockReason = "SAME_ERROR_REPEATED"
	// BlockReasonDistinctErrors: the attempts do NOT share one stable classified
	// FailureClass -- including a history too short to establish repetition at
	// all (a single attempt under a Budget of 1). There is no signature to
	// record, so the issue reads as flaky rather than stuck. This is the
	// deliberate fail-safe complement of BlockReasonSameErrorRepeated: promoting
	// a fleet-wide known-bad signature off a history that never repeated is the
	// exact misclassification a noisy signature would cause, so anything short
	// of proven repetition lands here and routes to a plain retry.
	BlockReasonDistinctErrors BlockReason = "DISTINCT_ERRORS"
	// BlockReasonPreconditionUnmet: the LAST attempt failed on a class that
	// retrying the issue cannot clear -- auth (rotate a credential) or
	// ambiguous scope (a human resolves the collision). The wall is outside the
	// issue, so neither a retry nor a known-bad signature is the right move:
	// escalate. Checked FIRST, before repetition, because three identical auth
	// failures are a precondition problem, not a known-bad code signature.
	BlockReasonPreconditionUnmet BlockReason = "PRECONDITION_UNMET"
	// BlockReasonTransientExhausted: the LAST attempt failed on a transient
	// class (rate-limit, network flake) and the issue is being blocked anyway --
	// which can only mean the extended transient ceiling is spent (#2892).
	// Nothing about the issue itself failed, so a known-bad signature would
	// poison the fleet ledger with a self-clearing wall; but a capacity window
	// that never reopened across the whole extended budget is an account/infra
	// condition a human should look at, so it escalates rather than re-offering
	// yet another worker onto the same wall.
	BlockReasonTransientExhausted BlockReason = "TRANSIENT_EXHAUSTED"
)

// Verdict is the closed adjudication of a budget crossing (#2892): what Decide
// decided once AttemptCount reached Budget, recorded on the Decision so an
// adjudicator can later verify the call -- witnessed, not just logged. The
// whole verdict is a pure fold over the recorded per-attempt reasons
// (Attempt.FailureClass/AtUnix), so re-running Decide over the same history
// reproduces it exactly. Empty until the budget is actually crossed: there is
// no adjudication to witness before then.
type Verdict string

const (
	// VerdictTransientRetry: the budget was crossed but the last failure is
	// transient and the extended transient ceiling still has room -- the blunt
	// hold is deferred and the issue keeps retrying under its measured
	// class-specific backoff (Hermes would have blocked it here).
	VerdictTransientRetry Verdict = "transient_retry"
	// VerdictTransientExhausted: the transient headroom is spent -- held, with
	// BlockReasonTransientExhausted recorded.
	VerdictTransientExhausted Verdict = "transient_exhausted"
	// VerdictStructuralBlock: the last failure is structural (not transient) at
	// or past the budget -- held, with the #2860 BlockReason recorded.
	VerdictStructuralBlock Verdict = "structural_block"
)

// Route is the closed routing verdict a BlockReason drives -- the point of the
// structured reason. It is what turns "blocked, count=2" into an action.
type Route string

const (
	// RouteRetry: no stable signature, no unmet precondition -- re-offer the
	// issue after its cooldown.
	RouteRetry Route = "retry"
	// RouteEscalate: a human must clear the precondition before any retry can
	// succeed.
	RouteEscalate Route = "escalate"
	// RouteKnownBad: a stable, repeated failure signature -- record it so peers
	// read it instead of rediscovering it (internal/knownbad).
	RouteKnownBad Route = "known_bad"
)

// preconditionClass reports whether a classified FailureClass is one a retry of
// the issue itself can never clear -- the two classes DefaultBackoffSeconds
// already documents as "needs a human" (rotate a credential, resolve a scope
// collision).
func preconditionClass(c FailureClass) bool {
	return c == FailureClassAuth || c == FailureClassAmbiguousScope
}

// ClassifyBlock is the failure-signature classifier over one issue's attempt
// history: it folds the recorded failures into the structured BlockReason the
// auto-block carries and the Route that reason drives. It is exported so a
// dispatcher (or the known-bad ledger wiring) can ask for the verdict directly
// rather than re-deriving it from a Decision.
//
// Signature stability is why this classifies onto the closed FailureClass
// vocabulary rather than comparing the caller's raw strings: "test_failure" and
// "assertion failed" are the same wall, and a signature that split them would
// read a genuinely stuck issue as flaky. An empty history yields empty values --
// there is nothing to explain.
//
// Order matters. Transient is checked first (#2892): blocking a history whose
// last failure is self-clearing can only mean the extended transient budget is
// exhausted, and letting it fall through to repetition would promote a
// rate-limit storm into a known-bad signature -- the exact ledger poisoning
// the transient adjudication exists to prevent. Then precondition before
// repetition (an unmet precondition repeated N times is still a
// precondition), and repetition requires at least two attempts, so a
// known-bad signature is never promoted off a single sample.
func ClassifyBlock(attempts []Attempt) (BlockReason, Route) {
	if len(attempts) == 0 {
		return "", ""
	}
	last := classify(attempts[len(attempts)-1].FailureClass)
	if transientClass(last) {
		return BlockReasonTransientExhausted, RouteEscalate
	}
	if preconditionClass(last) {
		return BlockReasonPreconditionUnmet, RouteEscalate
	}
	if len(attempts) >= 2 && sameClass(attempts) {
		return BlockReasonSameErrorRepeated, RouteKnownBad
	}
	return BlockReasonDistinctErrors, RouteRetry
}

// sameClass reports whether every attempt classifies onto one FailureClass --
// the stable failure signature BlockReasonSameErrorRepeated stands for. The
// class itself is already carried on Decision.BackoffClass (the last attempt's
// class, which for a same-class history IS the repeated class), so a caller
// building a known-bad signature reads it from there rather than a duplicate
// field.
func sameClass(attempts []Attempt) bool {
	first := classify(attempts[0].FailureClass)
	for _, a := range attempts[1:] {
		if classify(a.FailureClass) != first {
			return false
		}
	}
	return true
}

// Attempt is one recorded try at an issue: the failure class it ended in (the
// caller's vocabulary -- e.g. "test_failure", "timeout", "merge_conflict") and
// when it happened. An attempt that SUCCEEDED should simply not be recorded
// here; this package only ever sees the failed history.
type Attempt struct {
	FailureClass string `json:"failure_class"`
	AtUnix       int64  `json:"at_unix"`
}

// Input is one issue's attempt-budget facts.
type Input struct {
	IssueID  string    `json:"issue_id"`
	Attempts []Attempt `json:"attempts"`
	// Budget is the maximum number of recorded (failed) attempts allowed
	// before the issue is held for triage. A Budget <= 0 means unlimited --
	// the issue is never held on attempt count alone.
	Budget int `json:"budget"`
	// TransientBudget is the extended attempt ceiling applied instead of
	// Budget when the LAST recorded failure is transient (#2892) -- rate-limit
	// or network -- so a self-clearing wall keeps retrying under its measured
	// backoff instead of being blunt-blocked at Budget. <= 0 means the
	// default, Budget x defaultTransientBudgetMultiplier. Ignored while the
	// last failure is structural.
	TransientBudget int `json:"transient_budget,omitempty"`
	// NowUnix is the caller-supplied clock reading used for backoff math (is
	// the last attempt's class-specific cooldown window still open?). Zero
	// means the caller does not care about cooldown timing -- the Decision
	// still reports the backoff window that WOULD apply, but Status never
	// becomes StatusCoolingDown on a zero clock (there is no "now" to compare
	// against). This package never reads a clock itself.
	NowUnix int64 `json:"now_unix,omitempty"`
	// Backoff optionally overrides DefaultBackoffSeconds for this issue only
	// (e.g. an operator tuning one noisy issue's windows). A nil map uses
	// DefaultBackoffSeconds.
	Backoff map[FailureClass]int64 `json:"backoff,omitempty"`
}

// Decision is the verdict for one issue.
type Decision struct {
	IssueID          string `json:"issue_id"`
	Status           Status `json:"status"`
	AttemptCount     int    `json:"attempt_count"`
	Budget           int    `json:"budget"`
	LastFailureClass string `json:"last_failure_class,omitempty"`
	// BackoffClass is the closed FailureClass the LastFailureClass was
	// classified into (empty when there is no recorded attempt yet).
	BackoffClass FailureClass `json:"backoff_class,omitempty"`
	// BackoffSeconds is the cooldown window that failure class carries under
	// the effective policy (Input.Backoff, or DefaultBackoffSeconds).
	BackoffSeconds int64 `json:"backoff_seconds,omitempty"`
	// CooldownUntilUnix is the last attempt's AtUnix plus BackoffSeconds --
	// the earliest time this issue should be re-offered. Zero when there is
	// no recorded attempt.
	CooldownUntilUnix int64 `json:"cooldown_until_unix,omitempty"`
	// BlockReason is the structured, queryable reason this issue was
	// auto-blocked, distinguishing a genuinely-stuck issue (a repeated failure
	// signature) from a flaky one (distinct errors) from one walled off by an
	// unmet precondition (#2860). Set ONLY when Status is StatusHeld -- it
	// explains a block, and an issue that is not blocked has none. Empty
	// otherwise.
	BlockReason BlockReason `json:"block_reason,omitempty"`
	// Route is the action BlockReason drives -- retry, escalate, or record a
	// known-bad signature. Set exactly when BlockReason is.
	Route Route `json:"route,omitempty"`
	// Verdict is the witnessed adjudication of the budget crossing (#2892):
	// transient_retry when the blunt hold was deferred for a transient last
	// failure, transient_exhausted / structural_block when the issue was
	// actually held. Empty until AttemptCount reaches Budget -- before that
	// there is no adjudication to witness.
	Verdict Verdict `json:"verdict,omitempty"`
}

// Decide folds one issue's Input into a Decision, in this order: once
// AttemptCount reaches Budget (Budget > 0) the crossing is ADJUDICATED, not
// blunt (#2892) -- a transient last failure (rate-limit, network) with room
// under the transient ceiling keeps its cooldown/dispatchable status and
// records VerdictTransientRetry, while a structural last failure or spent
// transient headroom is HELD (overriding cooldown) with the verdict and
// BlockReason recorded; otherwise COOLING_DOWN when NowUnix is positive and
// still before the last attempt's class-specific CooldownUntilUnix; otherwise
// DISPATCHABLE. The Decision always carries the LAST recorded attempt's
// classified BackoffClass/BackoffSeconds/CooldownUntilUnix (when there is a
// recorded attempt) regardless of Status, so a report can show the policy
// even for a HELD issue.
func Decide(in Input) Decision {
	d := Decision{
		IssueID:      in.IssueID,
		Status:       StatusDispatchable,
		AttemptCount: len(in.Attempts),
		Budget:       in.Budget,
	}
	if len(in.Attempts) > 0 {
		last := in.Attempts[len(in.Attempts)-1]
		d.LastFailureClass = last.FailureClass
		d.BackoffClass = classify(last.FailureClass)
		d.BackoffSeconds = backoffSeconds(d.BackoffClass, in.Backoff)
		d.CooldownUntilUnix = last.AtUnix + d.BackoffSeconds
		if in.NowUnix > 0 && in.NowUnix < d.CooldownUntilUnix {
			d.Status = StatusCoolingDown
		}
	}
	if in.Budget > 0 && d.AttemptCount >= in.Budget {
		if transientClass(d.BackoffClass) && d.AttemptCount < transientCeiling(in) {
			// The Hermes-blunt hold is deferred: the last failure says nothing
			// about the issue itself, so the measured class backoff keeps
			// pacing retries. The verdict is the witness that the crossing was
			// adjudicated rather than the budget check having been skipped.
			d.Verdict = VerdictTransientRetry
			return d
		}
		d.Status = StatusHeld
		// The block is the only thing that needs explaining, so the structured
		// reason (and the route it drives) is stamped here rather than on every
		// dispatchable issue.
		d.BlockReason, d.Route = ClassifyBlock(in.Attempts)
		if d.BlockReason == BlockReasonTransientExhausted {
			d.Verdict = VerdictTransientExhausted
		} else {
			d.Verdict = VerdictStructuralBlock
		}
	}
	return d
}

// backoffSeconds resolves the effective window for class under an optional
// per-issue override map, falling back to DefaultBackoffSeconds and finally
// to FailureClassOther's default if the class is somehow missing from both
// (never zero, so a caller can't be left with "no cooldown at all" by an
// incomplete override map).
func backoffSeconds(class FailureClass, override map[FailureClass]int64) int64 {
	if override != nil {
		if s, ok := override[class]; ok {
			return s
		}
	}
	if s, ok := DefaultBackoffSeconds[class]; ok {
		return s
	}
	return DefaultBackoffSeconds[FailureClassOther]
}

// Report is the batch verdict over many issues.
type Report struct {
	Decisions         []Decision `json:"decisions"`
	DispatchableCount int        `json:"dispatchable_count"`
	CoolingDownCount  int        `json:"cooling_down_count"`
	HeldCount         int        `json:"held_count"`
}

// DecideAll folds a batch of issues, in the order given, into a Report.
func DecideAll(inputs []Input) Report {
	rep := Report{Decisions: make([]Decision, 0, len(inputs))}
	for _, in := range inputs {
		d := Decide(in)
		switch d.Status {
		case StatusHeld:
			rep.HeldCount++
		case StatusCoolingDown:
			rep.CoolingDownCount++
		default:
			rep.DispatchableCount++
		}
		rep.Decisions = append(rep.Decisions, d)
	}
	return rep
}

// RepeatedFailureTracker reports when the same failure key occurs three times
// in a row. A success or a changed key starts a new sequence.
type RepeatedFailureTracker struct {
	key      string
	failures int
}

// Record adds one result and reports whether the identical-failure budget is
// exhausted.
func (t *RepeatedFailureTracker) Record(key string, success bool) bool {
	if success {
		t.key = ""
		t.failures = 0
		return false
	}
	if key != t.key {
		t.key = key
		t.failures = 0
	}
	t.failures++
	return t.failures >= 3
}
