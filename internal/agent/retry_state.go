package agent

import (
	"context"
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

// waitBeforeAttempt applies the common between-attempt wait. Attempt zero is a
// no-op so callers can keep checkpoint work conditional without duplicating the
// retry-loop control flow.
func (p *HTTPPlanner) waitBeforeAttempt(ctx context.Context, attempt int, s *retryState, deadline time.Time, budgetOn bool) (bool, error) {
	if attempt == 0 {
		return false, nil
	}
	return p.retryBackoffWait(ctx, attempt, s.lastStatus, s.lastRetryAfter, s.lastCapWait, s.lastStatusErr, deadline, budgetOn)
}

func providerResponseStatus(resp *http.Response) int {
	if resp == nil {
		return 0
	}
	return resp.StatusCode
}

func newUpstreamStatusError(status int, raw []byte, hdr http.Header, bodyCap int) *UpstreamStatusError {
	return &UpstreamStatusError{
		Status:     status,
		Body:       truncate(raw, bodyCap),
		RetryAfter: hdr.Get("Retry-After"),
	}
}

// noteImmediateRetry records an uncounted credential refresh or account
// failover resend. Unlike noteRetryableStatus, this path deliberately carries
// no backoff or cap classification because the next attempt runs immediately.
func (s *retryState) noteImmediateRetry(status int, raw []byte, hdr http.Header, bodyCap int) {
	s.lastErr = newUpstreamStatusError(status, raw, hdr, bodyCap)
	s.lastStatus = status
	s.lastRetryAfter = ""
}

type rejectedResponseRetry struct {
	triedAuthRefresh     *bool
	forbidden            *forbiddenRetryState
	triedRehome          *bool
	rehomePending        *bool
	triedFailover        *bool
	failoverPending      *bool
	triedTransientRetry  *bool
	triedTransientTarget *bool
	recordRefreshState   bool
	bodyCap              int
}

// handleRejectedResponse is the shared pre-first-byte recovery policy for the
// buffered, OpenAI-stream, and Anthropic-stream paths. It returns retry=true
// for a counted transient retry and rewind=true for an immediate, uncounted
// credential/account resend. Every other response becomes the terminal status
// error returned in err.
func (c *upstreamCall) handleRejectedResponse(ctx context.Context, p *HTTPPlanner, s *retryState, resp *http.Response, raw []byte, attempt int, ctl rejectedResponseRetry) (retry, rewind bool, err error) {
	status := resp.StatusCode
	if p != nil && p.TransientTargetFunc != nil && transientTargetStatus(status) && ctl.triedTransientRetry != nil && ctl.triedTransientTarget != nil {
		if !*ctl.triedTransientRetry {
			*ctl.triedTransientRetry = true
			s.noteImmediateRetry(status, raw, resp.Header, ctl.bodyCap)
			notifyImmediateStatusRetry(p, attempt, status)
			return true, true, nil
		}
		if !*ctl.triedTransientTarget {
			*ctl.triedTransientTarget = true
			if c.failoverTransientTarget(p, status) {
				s.noteImmediateRetry(status, raw, resp.Header, ctl.bodyCap)
				notifyImmediateStatusRetry(p, attempt, status)
				return true, true, nil
			}
		}
	}
	if retryableStatus(status) {
		action := c.noteRetryableCapMaybeRehome(p, s, status, raw, resp.Header, ctl.bodyCap, false, ctl.triedRehome, ctl.rehomePending, attempt)
		return true, action == capRehomeResend, nil
	}
	if status == http.StatusUnauthorized && !*ctl.triedAuthRefresh && c.authRefreshable {
		if c.refreshAPIKeyWait(ctx, p) {
			*ctl.triedAuthRefresh = true
			notifyAuthRefresh(p, AuthRefreshRecovered, attempt)
			if ctl.recordRefreshState {
				s.noteImmediateRetry(status, raw, resp.Header, ctl.bodyCap)
			}
			return true, true, nil
		}
		notifyAuthRefresh(p, AuthRefreshExhausted, attempt)
	}
	if (status == http.StatusForbidden || status == http.StatusPaymentRequired) && usageOrOverageRejected(resp.Header) {
		action := c.noteRetryableCapMaybeRehome(p, s, status, raw, resp.Header, ctl.bodyCap, true, ctl.triedRehome, ctl.rehomePending, attempt)
		return true, action == capRehomeResend, nil
	}
	if status == http.StatusForbidden && ctl.forbidden.step403(ctx, p, raw, attempt) {
		return true, true, nil
	}
	if ctl.triedFailover != nil && classifyUpstream(status, raw, resp.Header) == RemedyFailoverAccount && !*ctl.triedFailover && p.AccountFailoverFunc != nil {
		if c.failoverAccountCred(p, RemedyFailoverAccount.String()) {
			*ctl.triedFailover = true
			*ctl.failoverPending = true
			s.noteImmediateRetry(status, raw, resp.Header, ctl.bodyCap)
			return true, true, nil
		}
		*ctl.triedFailover = true
		notifyAccountFailover(p, AccountFailoverExhausted, attempt)
	}
	return false, false, newUpstreamStatusError(status, raw, resp.Header, 400)
}

func notifyImmediateStatusRetry(p *HTTPPlanner, attempt, status int) {
	if p != nil && p.RetryNotify != nil {
		p.RetryNotify(attempt, status, 0)
	}
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
	se := newUpstreamStatusError(status, raw, hdr, bodyCap)
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
