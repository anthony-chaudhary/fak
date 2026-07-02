// retry_ceiling.go — never sleep past the client (#2258). The retry loops honor a
// provider-named wait of up to an hour per sleep (maxHonoredRetryAfter), which is right
// when the caller can actually outlast it — but on the proxy path the wrapped client has
// its OWN request timeout (~300s for Claude Code), so an in-handler sleep toward a
// multi-minute cap reset is structurally uncompletable: the client times out first, burns
// its own retries against the same wall (the 2026-07-01 evidence logs show twelve
// consecutive 300s cycles against an announced ~1h10m wait), and finally dies with an
// opaque timeout instead of the rate-limit truth the gateway KNEW. The ceiling here stops
// the loop BEFORE such a sleep and surfaces the classified upstream response downstream —
// status, cap kind, and the real Retry-After (or the classified reset delta) — so the
// client backs off or fails fast with the true cause, and a supervisor (#2256) can park on
// it. Waits at or under the ceiling keep the absorb-in-handler behavior byte-for-byte:
// transient throttle/5xx backoff (≤30s schedule) never reaches it.

package agent

import (
	"fmt"
	"os"
	"time"
)

// defaultInHandlerWaitCeiling is the largest single retry wait the loop will absorb
// in-handler before it instead surfaces the truthful upstream status downstream. 90s sits
// safely under the ~300s request timeout observed from wrapped clients (Claude Code)
// while still absorbing every transient backoff (the exponential schedule caps at 30s)
// and short provider-named waits, where silent in-handler absorption is the right UX.
const defaultInHandlerWaitCeiling = 90 * time.Second

// maxInHandlerWaitCeiling clamps FAK_INHANDLER_WAIT_CEILING: past the per-sleep
// Retry-After cap (1h) a larger ceiling could never matter, and a fat-fingered huge value
// must not silently restore the sleep-past-the-client behavior this exists to end.
const maxInHandlerWaitCeiling = maxHonoredRetryAfter

// inHandlerWaitCeiling resolves the client-survivable per-wait ceiling:
// FAK_INHANDLER_WAIT_CEILING (any Go duration) overrides the 90s default, clamped to
// [0, 1h]. 0 disables the ceiling and restores the historical absorb-everything behavior
// (the wait is then bounded only by maxHonoredRetryAfter and the retry budget).
func inHandlerWaitCeiling() time.Duration {
	if v := os.Getenv("FAK_INHANDLER_WAIT_CEILING"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxInHandlerWaitCeiling {
				return maxInHandlerWaitCeiling
			}
			return d
		}
	}
	return defaultInHandlerWaitCeiling
}

// RetryCeilingError reports a retry the loop REFUSED to wait out in-handler: the next
// honored wait exceeded the client-survivable ceiling, so instead of sleeping past the
// caller fak surfaces the upstream's truth immediately. Cause is the classified upstream
// status error — with RetryAfter guaranteed non-empty when a wait was derivable (the
// provider's own header verbatim, else the classified cap-reset delta in the same
// delta-seconds form) — so the gateway's existing errors.As ladders relay the true
// 429/5xx + Retry-After downstream instead of an opaque 502 or a client-side timeout.
type RetryCeilingError struct {
	Cause   *UpstreamStatusError
	Wait    time.Duration // the computed wait the loop declined to sleep
	Ceiling time.Duration // the ceiling it exceeded (inHandlerWaitCeiling at decision time)
}

// Error names the refused wait and the truth being surfaced instead. The upstream BODY is
// already truncated/sanitized by Cause; no new upstream text is introduced here.
func (e *RetryCeilingError) Error() string {
	return fmt.Sprintf("planner: honored wait %s exceeds the in-handler ceiling %s — surfacing the upstream status downstream instead of sleeping past the client (upstream said: %v)",
		e.Wait.Round(time.Second), e.Ceiling.Round(time.Second), e.Cause)
}

// Unwrap exposes the classified *UpstreamStatusError so the gateway's status, kind, and
// Retry-After ladders see the true upstream condition.
func (e *RetryCeilingError) Unwrap() error { return e.Cause }
