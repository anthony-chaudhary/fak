// retry_interrupt.go carries the rate-limit truth across a cancelled retry wait (#2257).
// The retry loops classify an upstream 429 the moment it lands (classifyLimit429, #1362),
// but until now that classification lived only in loop-local state: when the CALLER's
// context died mid-sleep — the flagship case being the wrapped Claude Code client timing
// out its own request ~300s into an honored multi-minute cap wait — the loop returned the
// bare context error, and every downstream readout (the FAILED debug line, the
// upstream-error metric, the post-mortem log) collapsed a KNOWN usage-cap park candidate
// into the catch-all "error". RetryInterruptedError is the envelope that keeps the two
// truths together: WHAT the upstream said (the classified status error) and WHY the wait
// ended early (the caller's context), so the observability layer never has to choose.

package agent

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// RetryInterruptedError reports a retry wait that was cut short by the CALLER's context
// while the loop was honoring a classified upstream pushback (a 429 rate limit / account
// cap, a 503/529 overload). It wraps BOTH causes so errors.Is/As reach each: Cause (the
// last real upstream status error, with any LimitReason/LimitResetHint classification and
// Retry-After) and Err (the context error that ended the wait — context.Canceled when the
// client hung up, context.DeadlineExceeded when the caller's own deadline fired). The
// gateway's kind classifier and Retry-After relay therefore see the true 429/5xx, not an
// opaque cancellation, and the FAILED debug line can carry the cap kind, the announced
// wait, and the client-disconnect marker without re-deriving any of them.
type RetryInterruptedError struct {
	// Cause is the classified upstream pushback the interrupted wait was honoring —
	// never nil (an interruption with no prior status error surfaces the raw context
	// error instead of this type, so a genuinely-unclassified failure still reads
	// "error" downstream).
	Cause *UpstreamStatusError
	// Err is the context error that cut the wait short, verbatim.
	Err error
	// AnnouncedWait is the FULL wait the retry loop announced (RetryNotify) and began
	// sleeping — e.g. the ~1h10m toward a usage-cap reset — of which only a fraction
	// may have elapsed before the caller vanished. It is what lets an operator (or a
	// supervisor) tell "died 300s into a known 1h wait" apart from "died after 300s".
	AnnouncedWait time.Duration
}

// Error names both halves: the upstream truth and the interruption. The upstream BODY is
// already truncated/sanitized by Cause; no new upstream text is introduced here.
func (e *RetryInterruptedError) Error() string {
	return fmt.Sprintf("planner: retry wait interrupted %v into the announced %s (upstream said: %v)",
		e.Err, e.AnnouncedWait.Round(time.Second), e.Cause)
}

// Unwrap exposes both causes to errors.Is/As: the classified *UpstreamStatusError (so the
// gateway's status/kind/Retry-After ladders see the true 429/5xx) and the context error
// (so callers that branch on context.Canceled still can).
func (e *RetryInterruptedError) Unwrap() []error { return []error{e.Cause, e.Err} }

// ClientGone reports whether the wait ended because the caller HUNG UP (context.Canceled
// — on the proxy path, the wrapped client closing its request) rather than a deadline
// elapsing. It is the client-disconnect marker the FAILED debug line renders.
func (e *RetryInterruptedError) ClientGone() bool { return errors.Is(e.Err, context.Canceled) }
