package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"
)

// ErrSerialRetryBudgetExhausted is terminal across nested replica dispatchers,
// even when the error also retains a retryable upstream status.
var ErrSerialRetryBudgetExhausted = errors.New("planner: serial retry budget exhausted")

type serialRetryBudgetKey struct{}
type replicaFailoverKey struct{}

type serialRetryBudget struct {
	mu            sync.Mutex
	remaining     int
	deadline      time.Time
	budgetOn      bool
	lastStatusErr *UpstreamStatusError
}

// WithSerialRetryBudget shares the configured attempt count and deadline across
// serial replica calls, including credential resends and buffered-to-stream
// recovery. Nested dispatchers inherit the existing budget without replenishing
// it. Callers without this opt-in retain their ordinary planner retry behavior.
func WithSerialRetryBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Value(serialRetryBudgetKey{}).(*serialRetryBudget); ok {
		return ctx, func() {}
	}
	count, deadline, budgetOn := retryBounds(time.Now())
	budget := &serialRetryBudget{remaining: count, deadline: deadline, budgetOn: budgetOn}
	ctx = context.WithValue(ctx, serialRetryBudgetKey{}, budget)
	if budgetOn {
		return context.WithDeadline(ctx, deadline)
	}
	return ctx, func() {}
}

// WithReplicaFailover lets an ordinary transient retry yield to an untried
// replica. An outer dispatcher's remaining siblings still count when a nested
// dispatcher reaches its own last replica.
func WithReplicaFailover(ctx context.Context, available bool) context.Context {
	if available {
		return context.WithValue(ctx, replicaFailoverKey{}, true)
	}
	return ctx
}

// RecordSerialReplicaError publishes the planner's final truth before retargeting.
// In particular, an accepted SSE response may fail outside its connection retry
// loop: that read error supersedes an older status even without another wait.
func RecordSerialReplicaError(ctx context.Context, err error) {
	if budget, ok := ctx.Value(serialRetryBudgetKey{}).(*serialRetryBudget); ok {
		var status *UpstreamStatusError
		errors.As(err, &status)
		budget.mu.Lock()
		budget.lastStatusErr = status
		budget.mu.Unlock()
	}
}

// WithSerialStreamBoundary observes the existing held-activity callback without
// replacing it. Reasoning/tool fragments can commit the gateway's role frame
// before any content reaches the sink, making replica replay irreversible too.
func WithSerialStreamBoundary(ctx context.Context, emitted func()) context.Context {
	held := streamHeldActivity(ctx)
	if held == nil {
		return ctx
	}
	return WithStreamHeldActivity(ctx, func() {
		emitted()
		held()
	})
}

func (b *serialRetryBudget) syncStatus(s *retryState) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !s.serialInitialized {
		// Carry a prior replica's status into the ordinary retry-state policy:
		// transport glitches retain it, whereas an HTTP 200 body-read failure
		// explicitly clears it at the existing buffered/streaming call sites.
		s.lastStatusErr = b.lastStatusErr
		s.serialInitialized = true
		return
	}
	b.lastStatusErr = s.lastStatusErr
}

func (b *serialRetryBudget) available() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.remaining > 0
}

func (b *serialRetryBudget) admit() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.remaining == 0 {
		return false
	}
	b.remaining--
	return true
}

func (s *retryState) serialExhausted() error {
	if s.lastStatusErr != nil {
		return errors.Join(ErrSerialRetryBudgetExhausted, s.lastStatusErr)
	}
	if s.lastErr != nil {
		return errors.Join(ErrSerialRetryBudgetExhausted, s.lastErr)
	}
	return ErrSerialRetryBudgetExhausted
}

func (s *retryState) serialInterrupted(ctx context.Context, budget *serialRetryBudget) error {
	err := ctx.Err()
	if err == nil && budget.budgetOn && !time.Now().Before(budget.deadline) {
		err = context.DeadlineExceeded
	}
	if err != nil && s.lastStatusErr != nil {
		return &RetryInterruptedError{Cause: s.lastStatusErr, Err: err}
	}
	return err
}

// SerialReplicaFallbackAllowed limits new static failover to transient failures.
// A terminal wait, request-shape refusal, or deterministic configuration error
// must not become retryable merely because it wraps a 5xx or a network error.
func SerialReplicaFallbackAllowed(err error) bool {
	var ceiling *RetryCeilingError
	var interrupted *RetryInterruptedError
	if errors.Is(err, ErrSerialRetryBudgetExhausted) || errors.As(err, &ceiling) || errors.As(err, &interrupted) {
		return false
	}
	var status *UpstreamStatusError
	if errors.As(err, &status) {
		return transientTargetStatus(status.Status) && status.RetryAfter == "" &&
			classifyUpstream(status.Status, []byte(status.Body), nil) == RemedyBackoff
	}
	if deterministicTransportError(err) {
		return false
	}
	var network net.Error
	return errors.As(err, &network) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

// retryState accumulates the between-attempt truth every planner retry loop tracks.
// Complete, streamConnect, and the Anthropic streaming passthrough all run the same
// loop shape; this is the state those three copies used to duplicate (#776, #1419).
//
// lastStatusErr is kept SEPARATE from lastErr and is NEVER cleared by a subsequent
// transient transport error, so a network glitch on a later attempt cannot shadow the
// 429/503/529 that actually drove the failure: on exhaustion the loop surfaces the
// real status (and Retry-After), not an opaque 502 (#1358).
type retryState struct {
	lastErr           error
	lastStatusErr     *UpstreamStatusError // last real-HTTP-status error; a glitch can't shadow it (#1358)
	lastStatus        int                  // status that triggered the pending retry (0 = a transient transport error)
	lastRetryAfter    string               // the triggering response's Retry-After header, honored as the next wait
	lastCapWait       string               // classified account-cap wait (#1362): toward the named reset when Retry-After is absent
	serialInitialized bool                 // prior replica truth is seeded once, never after a newer response clears it
}

// waitBeforeAttempt applies the common between-attempt wait. Under serial
// dispatch, every pass also consumes an admitted attempt opportunity, including
// immediate rewinds to attempt zero. Admission/request construction may still
// refuse afterward, so this conservatively bounds actual HTTP sends.
func (p *HTTPPlanner) waitBeforeAttempt(ctx context.Context, attempt int, s *retryState, deadline time.Time, budgetOn bool) (bool, error) {
	budget, shared := ctx.Value(serialRetryBudgetKey{}).(*serialRetryBudget)
	if shared {
		budget.syncStatus(s)
		if err := s.serialInterrupted(ctx, budget); err != nil {
			return false, err
		}
		if !budget.available() {
			return false, s.serialExhausted()
		}
		deadline, budgetOn = budget.deadline, budget.budgetOn
	}
	if attempt > 0 {
		stop, err := p.retryBackoffWait(ctx, attempt, s.lastStatus, s.lastRetryAfter, s.lastCapWait, s.lastStatusErr, deadline, budgetOn)
		if stop || err != nil {
			if shared && stop {
				if interrupted := s.serialInterrupted(ctx, budget); interrupted != nil {
					return false, interrupted
				}
				return false, s.serialExhausted()
			}
			return stop, err
		}
		if shared {
			if err := s.serialInterrupted(ctx, budget); err != nil {
				return false, err
			}
		}
		// Provider-directed waits and cap/auth recovery keep their existing local
		// policy. Only an ordinary transient failure can hand off to a sibling.
		if failover, _ := ctx.Value(replicaFailoverKey{}).(bool); shared && failover && s.lastRetryAfter == "" && s.lastCapWait == "" &&
			(transientTargetStatus(s.lastStatus) || (s.lastStatus == 0 && s.lastErr != nil)) {
			return true, nil
		}
	}
	if shared && !budget.admit() {
		return false, s.serialExhausted()
	}
	return false, nil
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
	if chatTemplateRejected(raw) {
		return false, false, newUpstreamStatusError(status, raw, resp.Header, 400)
	}
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
		// A stream-required refusal is NOT a transient overload even when it arrives as a 500:
		// backoff cannot make a buffered request acceptable. The caller's streaming-capable path
		// (Complete) reissues once as a stream BEFORE reaching here; if we are here, the wire
		// cannot stream (or the reissue already happened), so surface the refusal terminally
		// with its real cause instead of draining the retry budget on an unfixable status.
		if classifyUpstream(status, raw, resp.Header) == RemedyStreamRequired {
			return false, false, newUpstreamStatusError(status, raw, resp.Header, 400)
		}
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
