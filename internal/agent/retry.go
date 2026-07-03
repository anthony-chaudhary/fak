// retry.go is the upstream-transport retry/backoff cluster, split out of chat.go to
// keep that file under the architecture scorecard's 1500-line gate. Same package
// (agent): these helpers classify which transport errors a retry cannot fix, parse a
// Retry-After header, compute capped jittered backoff, and sleep cancellably — the
// host-side planner's upstream client wraps its attempts in them. See chat.go for the
// planner seam itself and doc.go for the package's trust framing.

package agent

import (
	"context"
	"crypto/tls"
	"errors"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// deterministicTransportError reports whether a transport error from Client.Do is
// a configuration error a retry cannot fix: a refused connection (nothing
// listening on the port — the canonical "wrong port / server not started"
// misconfiguration), a DNS name that does not resolve (NXDOMAIN — a wrong host),
// or a TLS handshake failure (a wrong scheme / untrusted cert). A plain timeout
// or a reset mid-flight is NOT deterministic — it may be transient packet loss —
// so it stays on the retry path.
func deterministicTransportError(err error) bool {
	if err == nil {
		return false
	}
	// DNS name does not resolve (NXDOMAIN) — a wrong host. A *temporary* DNS
	// failure (IsNotFound false) may clear, so it stays on the retry path.
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return dnsErr.IsNotFound
	}
	// TLS handshake failures — a wrong scheme (https to a plaintext port) or an
	// untrusted certificate; neither is transient.
	var recErr tls.RecordHeaderError
	if errors.As(err, &recErr) {
		return true
	}
	var certErr *tls.CertificateVerificationError
	if errors.As(err, &certErr) {
		return true
	}
	// Connection refused is the canonical "wrong port / server not started"
	// misconfiguration. errors.Is(syscall.ECONNREFUSED) catches it on Linux/macOS;
	// on Windows the OS errno (WSAECONNREFUSED) does NOT equal the BSD constant, so
	// fall back to a dial-time, non-timeout *net.OpError — which also covers "no
	// route to host" / "network unreachable", equally deterministic. A dial that
	// TIMED OUT may be transient packet loss, so it is left to retry.
	if errors.Is(err, syscall.ECONNREFUSED) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" && !opErr.Timeout() {
		return true
	}
	return false
}

// statusOverloaded is Anthropic's non-standard HTTP 529 "Overloaded" — the upstream is
// momentarily over capacity. net/http has no constant for it, and it is exactly as
// transient as a 503, so it belongs in retryableStatus. fak most often fronts Claude, so
// a 529 from the real Anthropic API was the single most common retryable status the
// original 429/5xx set silently dropped onto the non-retried path.
const statusOverloaded = 529

// retryableStatus reports whether an HTTP status warrants a backoff retry: the
// transient/overload family. 408 (the upstream timed out RECEIVING the request) and 429
// (rate limited) are the retryable 4xx; 500/502/503/504 are the 5xx overload/transient
// family; 529 is Anthropic's "Overloaded". Every OTHER 4xx is a request error a retry
// cannot fix and is NOT retried.
func retryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, // 408
		http.StatusTooManyRequests,     // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout,      // 504
		statusOverloaded:               // 529
		return true
	}
	return false
}

// plannerMaxAttempts is the TOTAL number of upstream tries (first attempt + retries)
// Complete makes on a transient failure. The default of 8 (raised from 4) trades a
// longer worst-case stall for far better resilience to the long rate-limit/overload
// windows a fleet sharing one account actually hits. FAK_PLANNER_MAX_ATTEMPTS overrides
// it, clamped to [1, 16] so a typo can neither disable retries (0/negative) nor wedge a
// turn for hours (huge). 1 means a single attempt with no retries.
func plannerMaxAttempts() int {
	if v := os.Getenv("FAK_PLANNER_MAX_ATTEMPTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= 16 {
			return n
		}
	}
	return 8
}

// defaultRetryBudget is how long the retry loop will keep trying a transient upstream
// failure when the attempt count is NOT explicitly pinned — long enough to ride out the
// multi-hour rate-limit/overload windows a session actually hits (a 5-hour-cap reset, a
// sustained 429/529 storm) rather than dropping the turn after a couple of minutes.
const defaultRetryBudget = 4 * time.Hour

// maxRetryBudget bounds FAK_PLANNER_RETRY_BUDGET so a fat-fingered value cannot wedge a
// turn effectively forever; the caller's context is still the real ceiling under it.
const maxRetryBudget = 24 * time.Hour

// retryAttemptHardCap is a spin guard for the time-bounded path: even with a huge budget
// and near-zero waits the loop can never exceed this many upstream tries. 4h of ~300ms
// minimum jittered waits is well under this, so it only catches a pathological zero-wait
// loop, never a legitimate long backoff.
const retryAttemptHardCap = 100000

// retryBounds resolves the two independent limits the retry loop runs under. When the
// operator PINS the attempt count (FAK_PLANNER_MAX_ATTEMPTS set in range), that count is
// authoritative and exact — the historical behavior, relied on by callers that want a
// fast, bounded give-up. When it is NOT pinned, the TIME budget is the primary limiter
// (default 4h, FAK_PLANNER_RETRY_BUDGET override) and the attempt cap rises to the hard
// spin guard so the full window is actually reachable. The loop stops at whichever bound
// trips first; the caller's context cancels under both. budgetOn is false only when the
// resolved budget is non-positive (FAK_PLANNER_RETRY_BUDGET=0 disables the time bound and
// restores pure attempt-count behavior).
func retryBounds(now time.Time) (maxAttempts int, deadline time.Time, budgetOn bool) {
	pinned := false
	if v := os.Getenv("FAK_PLANNER_MAX_ATTEMPTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= 16 {
			pinned = true
		}
	}
	budget := plannerRetryBudget()
	budgetOn = budget > 0
	if pinned || !budgetOn {
		// Attempt count is the bound (explicitly pinned, or the time bound is disabled).
		return plannerMaxAttempts(), now.Add(budget), budgetOn
	}
	// Time budget is the bound; attempts only backstop a pathological spin.
	return retryAttemptHardCap, now.Add(budget), true
}

// plannerRetryBudget is the TOTAL wall-clock window across all retries of one upstream
// call. It defaults to defaultRetryBudget (4h) so a long rate-limit/overload window is
// ridden out instead of dropping the turn; FAK_PLANNER_RETRY_BUDGET overrides it (any Go
// duration, e.g. "30m", "4h"), clamped to [0, maxRetryBudget]. A value of 0 disables the
// time bound, restoring pure attempt-count behavior. On the proxy path the full window is
// deliberately NOT reachable in-handler: any single wait past the client-survivable
// ceiling (inHandlerWaitCeiling, #2258) stops the loop and relays the truthful 429/5xx +
// Retry-After downstream instead of sleeping past the wrapped client's own request
// timeout — riding out a longer window is the supervisor's job (`fak guard` park, #2256),
// not an in-handler sleep's. The caller's context is still the real bound under both.
func plannerRetryBudget() time.Duration {
	if v := os.Getenv("FAK_PLANNER_RETRY_BUDGET"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxRetryBudget {
				return maxRetryBudget
			}
			return d
		}
	}
	return defaultRetryBudget
}

// defaultAuthRefreshWindow is how long a 401 on the rotating-subscription path will keep
// polling the on-disk credential for a FRESH, different token before giving up. It exists
// for the re-login race: when the OAuth token expires the upstream 401s the instant it
// dies, but Claude Code (or a user running `claude` / `claude /login` in another terminal)
// needs a beat to refresh and rewrite .credentials.json. A single boot-time-style read at
// the 401 instant usually still sees the SAME stale token, so a one-shot refresh gives up
// and the 401 surfaces to the wrapped agent — which then drops into its OWN /login and the
// live guarded session is lost. Polling for a few seconds lets the re-login land and the
// session self-heal in place. The common case (token already rotated on disk) returns on
// the first poll with zero added latency, and a genuinely-dead credential with no re-login
// coming still fails within this bounded window rather than looping forever.
//
// Raised from 3s to 10s (#1834): 3s assumed an INTERACTIVE Claude Code process was always
// concurrently rewriting .credentials.json, which a headless `fak accounts launch` never has
// — every headless 401 was timing out this window by construction. cmd/fak/guard.go now runs
// a PROACTIVE freshness check (accounts.NewRehydrateCredRung, #1183) before the first request
// so a headless launch should rarely reach this reactive path at all; this default is now
// purely a backstop for a refresh that lands slightly after the proactive check gave up (or
// for callers that bypass guard's launch path entirely), so it is widened rather than left at
// the too-tight value the fleet was measurably blocked on. FAK_AUTH_REFRESH_WINDOW still
// overrides it for an operator who needs to tune either window without a rebuild.
const defaultAuthRefreshWindow = 10 * time.Second

// maxAuthRefreshWindow clamps FAK_AUTH_REFRESH_WINDOW so a fat-fingered value cannot wedge
// a turn waiting on a re-login that is never coming; the caller's context is the real
// ceiling under it.
const maxAuthRefreshWindow = 30 * time.Second

// authRefreshPollInterval is how often the 401 wait re-reads the credential within the
// window. Short enough that a freshly-written token is adopted promptly, long enough not to
// hammer the disk (the read is a small JSON parse with its own torn-read retries).
const authRefreshPollInterval = 150 * time.Millisecond

// authRefreshWindow resolves the total wait the 401 auth-recovery polls disk for a fresh
// token, defaulting to defaultAuthRefreshWindow and honoring FAK_AUTH_REFRESH_WINDOW (any
// Go duration), clamped to [0, maxAuthRefreshWindow]. A value of 0 restores the historical
// single-read behavior (one refresh attempt, no wait).
func authRefreshWindow() time.Duration {
	if v := os.Getenv("FAK_AUTH_REFRESH_WINDOW"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxAuthRefreshWindow {
				return maxAuthRefreshWindow
			}
			return d
		}
	}
	return defaultAuthRefreshWindow
}

// A 403 has TWO populations that a bare status cannot tell apart. The PERMANENT one — the
// credential genuinely lacks entitlement for this model/org/region — is terminal: retrying
// only stalls the turn before the same "run /login" answer. The TRANSIENT one — a
// server-side abuse/capacity gate that trips under a burst and clears in seconds (the
// 2026-07-03 gem8 storm: five sessions 403'd for ~9 minutes, then the SAME pinned token
// succeeded on the next request with no local change) — is exactly as recoverable as a 529,
// and today dies the same terminal death, dropping the live session into a spurious /login.
//
// So a 403 gets its OWN bounded retry arm, deliberately NOT folded into retryableStatus:
// that family inherits the multi-HOUR budget (plannerRetryBudget), which is right for an
// overload but catastrophic for a permanent denial — it would wedge a truly-unentitled turn
// for hours. Instead a 403 retries a FEW times across a SHORT window: long enough to ride
// out a transient abuse-gate flap, short enough that a permanent denial surfaces promptly
// with the real answer. Bounded by BOTH a max attempt count and a total window, whichever
// trips first; a permanent 403 exhausts the small budget in seconds and surfaces terminally.

// defaultForbiddenRetryWindow is the total wall-clock a transient-403 recovery will keep
// retrying before surfacing the 403 terminally. Sized to the observed transient-403 flap
// (seconds-to-low-minutes), NOT the hours a rate-limit window can run: a permanent
// entitlement 403 must surface fast with the actionable "run /login", so this stays short.
// FAK_FORBIDDEN_RETRY_WINDOW overrides it; 0 disables the 403 retry arm entirely (restore
// the historical terminal-on-first-403 behavior).
const defaultForbiddenRetryWindow = 30 * time.Second

// maxForbiddenRetryWindow clamps FAK_FORBIDDEN_RETRY_WINDOW so a fat-fingered value cannot
// turn a permanent 403 into a multi-minute stall before the real "run /login" answer. The
// caller's context is still the real ceiling under it.
const maxForbiddenRetryWindow = 2 * time.Minute

// forbiddenRetryMaxAttempts caps the number of 403 retries independently of the window, so
// even a zero-Retry-After 403 answered instantly cannot spin: at most this many re-sends,
// then surface. Combined with the window (whichever trips first), a transient flap gets a
// handful of paced probes and a permanent denial gives up in well under the window.
const forbiddenRetryMaxAttempts = 5

// forbiddenRetryWindow resolves the transient-403 recovery window, defaulting to
// defaultForbiddenRetryWindow and honoring FAK_FORBIDDEN_RETRY_WINDOW (any Go duration),
// clamped to [0, maxForbiddenRetryWindow]. A value of 0 disables the 403 retry arm.
func forbiddenRetryWindow() time.Duration {
	if v := os.Getenv("FAK_FORBIDDEN_RETRY_WINDOW"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxForbiddenRetryWindow {
				return maxForbiddenRetryWindow
			}
			return d
		}
	}
	return defaultForbiddenRetryWindow
}

// forbiddenRetryWait is the paced backoff between 403 re-sends: the same jittered exponential
// schedule the transient-status family uses, so a fleet of sessions hitting the same abuse
// gate does not stampede it the instant it might clear. Reuses backoffDuration's floor
// (600ms) so the small attempt budget spans the window rather than firing all at once.
func forbiddenRetryWait(attempt int) time.Duration {
	return jitter(backoffDuration(attempt))
}

// forbiddenBodyIsPermanent reports whether a 403 body carries an upstream signature that marks
// the denial as PERMANENT — a hard entitlement/permission refusal a retry cannot fix — so the
// transient recovery arm skips it and surfaces the actionable answer immediately. It matches
// only conservative, unambiguous signatures (the provider naming permission/entitlement/region
// as the cause); anything it does not recognize is treated as POSSIBLY-transient and gets the
// bounded retry, because the storm's whole lesson is that an unlabeled 403 may well clear. The
// match is lowercase-substring over the truncated body, so it is robust to surrounding JSON.
func forbiddenBodyIsPermanent(body []byte) bool {
	b := strings.ToLower(string(body))
	for _, sig := range []string{
		"permission_error",     // provider typed a hard permission denial
		"not have access",      // "...does not have access to model..."
		"does not have access", // same, explicit
		"not entitled",         // entitlement refusal
		"not allowed to use",   // model/feature not on the plan
		"unsupported_region",   // geographic entitlement — a retry never clears it
		"region is not",        // "...region is not supported/allowed..."
	} {
		if strings.Contains(b, sig) {
			return true
		}
	}
	return false
}

// forbiddenRetryDecision is the closed verdict of one 403 recovery step.
type forbiddenRetryDecision int

const (
	// forbiddenRetryGiveUp: the denial is permanent (body signature) or the bounded budget is
	// spent — surface the 403 terminally with the actionable answer.
	forbiddenRetryGiveUp forbiddenRetryDecision = iota
	// forbiddenRetryGo: a transient 403 within budget — the arm has already slept its paced wait
	// and the caller should re-send.
	forbiddenRetryGo
)

// forbiddenRetryState bounds a single upstream call's 403 recovery arm — the transient-403
// analogue of the 401 auth-refresh window. It is SELF-CONTAINED: step() decides whether to
// retry, and when it does it sleeps its OWN short paced wait (never the 429/5xx hour-budget
// backoff), so a permanent entitlement 403 exhausts a seconds-scale budget promptly instead of
// inheriting the multi-hour transient window. Zero value is ready to use; the window/attempt
// bounds are resolved lazily on first use so a test env override is picked up per call.
type forbiddenRetryState struct {
	tries    int       // 403 retries already spent by this arm
	deadline time.Time // total-window bound; zero until the first step arms it
	max      int       // attempt bound; zero until the first step arms it
	window   time.Duration
	fired    bool // whether this arm ever decided to retry (drives the exhausted notify)
}

// step is called on each 403 for one upstream call. It returns forbiddenRetryGo after sleeping a
// paced wait when the denial looks transient and the bounded budget (window AND attempts) has
// room, or forbiddenRetryGiveUp when the body marks a permanent denial, the budget is spent, or
// the context is cancelled. The ctx cancel path returns give-up promptly so an abandoned turn
// never blocks here.
func (s *forbiddenRetryState) step(ctx context.Context, body []byte) forbiddenRetryDecision {
	if s.deadline.IsZero() {
		s.window = forbiddenRetryWindow()
		s.deadline = time.Now().Add(s.window)
		s.max = forbiddenRetryMaxAttempts
	}
	// A disabled arm (FAK_FORBIDDEN_RETRY_WINDOW=0) or a permanent-signature body never retries.
	if s.window <= 0 || forbiddenBodyIsPermanent(body) {
		return forbiddenRetryGiveUp
	}
	if s.tries >= s.max || !time.Now().Before(s.deadline) {
		return forbiddenRetryGiveUp
	}
	wait := forbiddenRetryWait(s.tries + 1)
	if rem := time.Until(s.deadline); rem < wait {
		wait = rem // never sleep past the window
	}
	if wait <= 0 {
		return forbiddenRetryGiveUp
	}
	if err := sleepCtx(ctx, wait); err != nil {
		return forbiddenRetryGiveUp // context cancelled/expired: do not keep an abandoned turn waiting
	}
	s.tries++
	s.fired = true
	return forbiddenRetryGo
}

// attempted reports whether this arm ever decided to retry, so the caller fires the "exhausted"
// notify only when a recovery was genuinely attempted-and-spent — not on a first-403 give-up
// (permanent signature, or the arm disabled), which is a plain terminal denial, not a self-heal
// that ran out of budget.
func (s *forbiddenRetryState) attempted() bool { return s.fired }

// maxBackoff caps a single exponential backoff wait. The attempt²×600ms schedule would
// otherwise grow without bound as the attempt budget rises; the cap keeps any ONE wait
// reasonable while still letting the OVERALL retry window stretch across many attempts.
const maxBackoff = 30 * time.Second

// maxHonoredRetryAfter caps how long an upstream-supplied Retry-After can make us wait in
// a SINGLE sleep. A rate-limited/overloaded upstream names when to come back and is
// usually right, so we honor it — up to this per-wait ceiling. It is now 1h (was 60s): the
// real bound on the total wait is the retry BUDGET (plannerRetryBudget, default 4h) and
// the remaining-budget clamp in retryWaitWithin, so a genuine multi-minute window can be
// honored in one sleep without a fat-fingered or hostile header running away — the total
// can never exceed the budget regardless. Beyond this per-wait cap we wait the cap, then
// re-read a fresh Retry-After on the next try. Note the tighter client-facing bound in
// retryBackoffWait: a wait past inHandlerWaitCeiling (default 90s, #2258) is not slept at
// all — the classified status is surfaced downstream instead, because a sleep the CLIENT
// cannot survive helps no one.
const maxHonoredRetryAfter = time.Hour

// retryWait returns how long to sleep before `attempt`. A server-directed Retry-After
// (delta-seconds OR HTTP-date) wins — the upstream knows when it will be ready better than
// any local schedule — capped at the per-wait ceiling, and nudged with a little UPWARD
// jitter so a fleet honoring the SAME value does not stampede the instant it expires.
// Otherwise it falls back to the exponential schedule with equal jitter, which both
// desynchronizes lockstep retries and keeps the reported wait strictly positive.
func retryWait(attempt int, retryAfter string) time.Duration {
	if d, ok := parseRetryAfter(retryAfter, time.Now()); ok {
		if d > maxHonoredRetryAfter {
			d = maxHonoredRetryAfter
		}
		return jitterUp(d)
	}
	return jitter(backoffDuration(attempt))
}

// minRetryWait floors the per-attempt wait on the TIME-BUDGET path so a long budget can
// never degrade into a hammering loop. The exponential-backoff path is already floored
// (backoffDuration(1) = 600ms), but a server-directed Retry-After of "0" (or any tiny
// value) bypasses backoff entirely via the honored-Retry-After branch — so an overloaded
// upstream answering "503 Retry-After: 0" against a multi-hour budget would otherwise spin
// up to retryAttemptHardCap near-instant requests at the very server that just said it is
// over capacity. Flooring each wait turns "retry for 4h" into a patient retry, not a 4h
// flood. It applies only to the budgeted path (retryWaitWithin); the pinned attempt-count
// path keeps retryWait's exact historical behavior. 250ms × the hard cap is still well
// within the budget, so the floor never shortens the reachable window.
const minRetryWait = 250 * time.Millisecond

// retryWaitWithin is retryWait bounded by a deadline: it never returns a wait that would
// sleep PAST `deadline`, so the total retry window cannot exceed the budget. It floors the
// wait at minRetryWait so a tiny/zero Retry-After cannot turn the budget into a tight spin,
// and returns a negative duration when there is no budget left to even wait (the caller
// treats that as exhaustion). With a zero deadline (time bound disabled) it is exactly
// retryWait.
func retryWaitWithin(attempt int, retryAfter string, deadline time.Time, now time.Time) time.Duration {
	w := retryWait(attempt, retryAfter)
	if deadline.IsZero() {
		return w
	}
	if w < minRetryWait {
		w = minRetryWait // anti-flood floor: never hammer an overloaded upstream, even on Retry-After: 0
	}
	rem := deadline.Sub(now)
	if rem <= 0 {
		return -1
	}
	if w > rem {
		// Budget nearly spent: take the remaining sliver, then the next iteration sees
		// rem<=0 and stops. The floor above cannot reintroduce a spin here because this
		// branch sleeps the whole remainder and then exhausts.
		return rem
	}
	return w
}

// backoffDuration is the exponential backoff base for retry `attempt`: attempt²×600ms
// (600ms, 2.4s, 5.4s, 9.6s, …), capped at maxBackoff. retryWait applies jitter on top, so
// this is the pre-jitter schedule, not the literal sleep.
func backoffDuration(attempt int) time.Duration {
	d := time.Duration(attempt*attempt) * 600 * time.Millisecond
	if d > maxBackoff {
		return maxBackoff
	}
	return d
}

// jitter applies equal-jitter to base: a uniformly random wait in [base/2, base]. A fleet
// that hit the same rate-limit window at the same instant then retries spread across the
// window instead of in lockstep, so it does not immediately re-trigger the limit. base<=0
// (the no-wait first attempt) stays 0.
func jitter(base time.Duration) time.Duration {
	if base <= 0 {
		return 0
	}
	half := int64(base / 2)
	return time.Duration(half) + time.Duration(rand.Int63n(half+1))
}

// jitterUp returns base plus a uniformly random extra in [0, base/4]. Used for a honored
// Retry-After: the upstream asked us not to come back BEFORE its named instant, so the
// wait is never reduced — only nudged slightly past it so a fleet sharing the value fans
// out instead of stampeding the upstream the moment it expires.
func jitterUp(base time.Duration) time.Duration {
	if base <= 0 {
		return 0
	}
	q := int64(base / 4)
	if q <= 0 {
		return base
	}
	return base + time.Duration(rand.Int63n(q+1))
}

// parseRetryAfterSeconds parses an RFC 7231 Retry-After in its delta-seconds form ("120")
// into a duration. The HTTP-date form is intentionally NOT handled — honoring an absolute
// date would mean trusting the upstream's clock and could imply an arbitrarily long wait —
// so on that form (or any non-numeric/negative value) it returns ok=false and the caller
// falls back to local exponential backoff.
func parseRetryAfterSeconds(v string) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0, false
	}
	return time.Duration(n) * time.Second, true
}

// parseRetryAfter parses an RFC 7231 Retry-After in EITHER supported form into a wait
// relative to `now`: the delta-seconds form ("120") OR the HTTP-date form
// ("Wed, 21 Oct 2025 07:28:00 GMT"). The date form is resolved against `now` (passed in so
// the parse is testable and deterministic) and yields the remaining duration until that
// instant; a date already in the past yields ok=false (nothing to wait for). A non-numeric
// non-date value yields ok=false and the caller falls back to local exponential backoff.
// The total wait is bounded elsewhere (the per-wait cap in retryWait, the remaining-budget
// clamp in retryWaitWithin), so honoring an absolute date can never imply an unbounded wait.
func parseRetryAfter(v string, now time.Time) (time.Duration, bool) {
	if d, ok := parseRetryAfterSeconds(v); ok {
		return d, true
	}
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d, true
		}
	}
	return 0, false
}

// sleepCtx waits for d, returning ctx.Err() early if the context is cancelled. A
// non-positive d does not sleep but still surfaces an already-cancelled context, so a
// cancelled turn never sneaks one more upstream attempt past the check.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// retryBackoffWait performs the pre-attempt backoff shared by every upstream retry loop
// (Complete, streamConnect, StreamAnthropicRaw) on a retry iteration (attempt > 0). It computes
// the next wait — honoring a named Retry-After, else the classified cap wait (lastCapWait, the
// delta-seconds toward a session/weekly/usage-cap reset from classifyLimit429; "" on the
// transient path), else the jittered exponential schedule, clamped to the remaining time budget
// when budgetOn — surfaces it via RetryNotify BEFORE the otherwise-invisible sleep (so the
// operator's `retry` line and counter fire with the exact wait slept), then sleeps it out
// cancellably under ctx. An upstream-supplied Retry-After outranks the derived cap wait: the
// provider's own instruction is always the better signal. It returns stop=true when the time
// budget is spent (the caller breaks the loop and surfaces the last error) and a non-nil err
// when ctx was cancelled during the wait (the caller returns it). The wait is computed ONCE and
// shared with the hook, so the reported wait carries the same jitter and honored value as the sleep.
//
// lastStatusErr is the loop's #1358 keepsake — the last error that carried a REAL upstream
// HTTP status, with its #1362 classification. When the caller's context dies DURING the
// sleep and that keepsake exists, the returned error is a RetryInterruptedError wrapping
// BOTH (#2257): the turn was killed mid-wait toward a KNOWN 429/5xx condition, and
// surfacing only the bare context error would collapse a classified rate-limit park
// candidate into the catch-all "error" on every downstream readout. With no prior status
// error (a pure transport-glitch loop) the context error returns unchanged, so a
// genuinely-unclassified failure still reads exactly as before.
func (p *HTTPPlanner) retryBackoffWait(ctx context.Context, attempt, lastStatus int, lastRetryAfter, lastCapWait string, lastStatusErr *UpstreamStatusError, deadline time.Time, budgetOn bool) (stop bool, err error) {
	if lastRetryAfter == "" {
		lastRetryAfter = lastCapWait
	}
	var wait time.Duration
	if budgetOn {
		wait = retryWaitWithin(attempt, lastRetryAfter, deadline, time.Now())
		if wait < 0 {
			return true, nil // budget spent — nothing left to even wait
		}
	} else {
		wait = retryWait(attempt, lastRetryAfter)
	}
	// Never sleep past the client (#2258): a wait beyond the client-survivable ceiling is
	// structurally uncompletable on the proxy path (the wrapped client's own request
	// timeout fires first, burning futile retries against a wall the gateway can name).
	// Stop retrying and surface the classified upstream truth downstream — with the real
	// Retry-After, or the classified cap-reset delta when the header was absent (both in
	// RFC 7231 delta-seconds/date form; lastRetryAfter already carries the merge). Only a
	// wait with a REAL classified status behind it takes this path: the transient backoff
	// schedule (≤30s) never exceeds the default 90s ceiling, so absorb behavior under the
	// ceiling is byte-for-byte unchanged, and FAK_INHANDLER_WAIT_CEILING=0 disables it.
	if ceiling := inHandlerWaitCeiling(); ceiling > 0 && wait > ceiling && lastStatusErr != nil {
		se := *lastStatusErr
		if se.RetryAfter == "" {
			se.RetryAfter = lastRetryAfter
		}
		return false, &RetryCeilingError{Cause: &se, Wait: wait, Ceiling: ceiling}
	}
	if p.RetryNotify != nil {
		p.RetryNotify(attempt, lastStatus, wait)
	}
	if err := sleepCtx(ctx, wait); err != nil {
		if lastStatusErr != nil {
			return false, &RetryInterruptedError{Cause: lastStatusErr, Err: err, AnnouncedWait: wait}
		}
		return false, err
	}
	return false, nil
}
