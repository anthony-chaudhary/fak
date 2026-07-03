package agent

import (
	"fmt"
	"net/http"
	"time"
)

// retryState accumulates the between-attempt truth every planner retry loop tracks.
// Complete, streamConnect, and the Anthropic streaming passthrough all run the same
// loop shape; this is the state those three copies used to duplicate (#776, #1419).
//
// lastStatusErr is kept SEPARATE from lastErr and is NEVER cleared by a subsequent
// transient transport error, so a network glitch on a later attempt cannot shadow the
// 429/503/529 that actually drove the failure: on exhaustion the loop surfaces the
// real status (and Retry-After), not an opaque 502 (#1358).
type retryState struct {
	lastErr        error
	lastStatusErr  *UpstreamStatusError // last real-HTTP-status error; a glitch can't shadow it (#1358)
	lastStatus     int                  // status that triggered the pending retry (0 = a transient transport error)
	lastRetryAfter string               // the triggering response's Retry-After header, honored as the next wait
	lastCapWait    string               // classified account-cap wait (#1362): toward the named reset when Retry-After is absent
}

// noteTransportGlitch records a transient transport error: no HTTP status, no
// Retry-After to honor, and — a glitch is not a cap — no cap wait to stretch the
// next retry toward. lastStatusErr is deliberately left intact (#1358).
func (s *retryState) noteTransportGlitch(err error) {
	s.lastErr = err
	s.lastStatus = 0
	s.lastRetryAfter = ""
	s.lastCapWait = ""
}

// noteRetryableStatus records a retryable non-200 (429 rate-limit, 503/529 overload,
// 408/5xx transient): build the status error, classify a 429 LIVE against the closed
// rate-limit vocabulary (#1362) — so a session/weekly/usage cap waits toward its
// named reset while a plain throttle keeps the transient schedule — and honor the
// response's Retry-After as the next wait. bodyCap bounds the echoed error body.
func (s *retryState) noteRetryableStatus(status int, raw []byte, hdr http.Header, bodyCap int) {
	ra := hdr.Get("Retry-After")
	se := &UpstreamStatusError{Status: status, Body: truncate(raw, bodyCap), RetryAfter: ra}
	cls, capWait := classifyLimit429(status, raw, hdr, time.Now())
	se.LimitReason, se.LimitResetHint = cls.Reason, cls.ResetHint
	s.lastErr = se
	s.lastStatusErr = se
	s.lastStatus = status
	s.lastRetryAfter = ra
	s.lastCapWait = capWait
}

// exhausted folds the accumulated state into the terminal error once the attempt
// budget is spent, preferring the last error that carried a real upstream status
// (and Retry-After) over a later transient transport glitch (#1358).
func (s *retryState) exhausted(prefix string) error {
	if s.lastStatusErr != nil {
		return fmt.Errorf("%s: %w", prefix, s.lastStatusErr)
	}
	return fmt.Errorf("%s: %w", prefix, s.lastErr)
}
