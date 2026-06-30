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
// time bound, restoring pure attempt-count behavior. The caller's context is still the
// real bound: on the synchronous serve path the http.Server WriteTimeout
// (FAK_HTTP_WRITE_TIMEOUT_S) caps the in-handler wait far below this — there the durable
// session park (#1363) is what extends the wait across a process boundary.
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
// live guarded session is lost. Polling for ~a few seconds lets the re-login land and the
// session self-heal in place. The common case (token already rotated on disk) returns on
// the first poll with zero added latency, and a genuinely-dead credential with no re-login
// coming still fails within this bounded window rather than looping forever.
const defaultAuthRefreshWindow = 3 * time.Second

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
// re-read a fresh Retry-After on the next try.
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
