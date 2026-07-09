package agent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// anthropicLimitBody wraps a provider refusal message in the Anthropic error wire shape
// the live 429 path actually receives.
func anthropicLimitBody(msg string) []byte {
	return []byte(fmt.Sprintf(`{"type":"error","error":{"type":"rate_limit_error","message":%q}}`, msg))
}

// A live 429 whose body names the 5-hour session cap and whose unified headers relay the
// window's reset must classify as session_limit AND derive the cap wait from that reset —
// not from the transient exponential schedule (#1362).
func TestClassifyLimit429_SessionCapWaitsTowardHeaderReset(t *testing.T) {
	now := time.Now()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(now.Add(90*time.Minute).Unix(), 10))
	cls, capWait := classifyLimit429(429, anthropicLimitBody("You've hit your session limit · resets 8pm (America/Los_Angeles)."), h, now)
	if cls.Reason != resume.LimitSession {
		t.Fatalf("reason = %q, want %q", cls.Reason, resume.LimitSession)
	}
	if cls.ResetHint == "" {
		t.Fatalf("reset hint empty, want the banner's 'resets …' tail")
	}
	secs, err := strconv.Atoi(capWait)
	if err != nil {
		t.Fatalf("capWait %q not delta-seconds: %v", capWait, err)
	}
	if secs < 85*60 || secs > 95*60 {
		t.Fatalf("capWait = %ds, want ~%ds (the relayed 5h-window reset)", secs, 90*60)
	}
}

// A weekly cap must prefer the 7d window's reset over an EARLIER 5h reset: the 5h window
// clearing does not un-cap a weekly-limited account.
func TestClassifyLimit429_WeeklyPrefersSevenDayWindow(t *testing.T) {
	now := time.Now()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(now.Add(10*time.Minute).Unix(), 10))
	h.Set("Anthropic-Ratelimit-Unified-7d-Reset", strconv.FormatInt(now.Add(20*time.Minute).Unix(), 10))
	cls, capWait := classifyLimit429(429, anthropicLimitBody("You've hit your weekly limit · resets Oct 14."), h, now)
	if cls.Reason != resume.LimitWeekly {
		t.Fatalf("reason = %q, want %q", cls.Reason, resume.LimitWeekly)
	}
	secs, _ := strconv.Atoi(capWait)
	if secs < 19*60 || secs > 21*60 {
		t.Fatalf("capWait = %ds, want ~%ds (the 7d reset, not the earlier 5h one)", secs, 20*60)
	}
}

// A classified cap with NO machine-readable reset (no unified headers, or only a stale
// one) must fall back to the slow cap-probe interval — never the 600ms..30s transient
// hammer — and FAK_CAP_PROBE_INTERVAL must tune it.
func TestClassifyLimit429_CapWithoutResetUsesProbeFloor(t *testing.T) {
	now := time.Now()
	stale := http.Header{}
	stale.Set("Anthropic-Ratelimit-Unified-Reset", strconv.FormatInt(now.Add(-10*time.Second).Unix(), 10))
	cls, capWait := classifyLimit429(429, anthropicLimitBody("usage limit reached"), stale, now)
	if cls.Reason != resume.LimitUsage {
		t.Fatalf("reason = %q, want %q", cls.Reason, resume.LimitUsage)
	}
	if want := strconv.Itoa(int(defaultCapProbeInterval.Seconds())); capWait != want {
		t.Fatalf("capWait = %q, want the default probe floor %q (stale reset must not be honored)", capWait, want)
	}
	t.Setenv("FAK_CAP_PROBE_INTERVAL", "90s")
	if _, capWait := classifyLimit429(429, anthropicLimitBody("usage limit reached"), nil, now); capWait != "90" {
		t.Fatalf("capWait = %q with FAK_CAP_PROBE_INTERVAL=90s, want \"90\"", capWait)
	}
}

// The "a 4-hour wait shouldn't be considered temporary" fix: a GENERIC-body 429 (no
// session/weekly/usage keyword — it would classify LimitRate on the body alone) whose ONLY cap
// signal is a far-future machine-readable wait must promote to a rehome-worthy account cap, so
// isAccountCap429 fires and the loop rehomes instead of sleeping toward a reset no wrapped client
// can outlast. The promotion keys on the WAIT MAGNITUDE (past the in-handler ceiling), not the
// body taxonomy — from a unified reset header OR a large Retry-After.
func TestClassifyLimit429_LongWaitPromotesGenericThrottle(t *testing.T) {
	now := time.Now()
	longReset := http.Header{}
	longReset.Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(now.Add(4*time.Hour).Unix(), 10))
	cls, capWait := classifyLimit429(429, []byte("Too many requests"), longReset, now)
	if cls.Reason != resume.LimitUsage {
		t.Fatalf("reason = %q, want %q (a 4h-wait generic 429 is a de-facto cap, not a throttle)", cls.Reason, resume.LimitUsage)
	}
	secs, err := strconv.Atoi(capWait)
	if err != nil || secs < int((4*time.Hour-time.Minute).Seconds()) {
		t.Fatalf("capWait = %q, want the ~4h reset delta in seconds", capWait)
	}
	if !isAccountCap429(429, []byte("Too many requests"), longReset, now) {
		t.Fatal("a generic 429 with a 4h reset must be a rehome-worthy account cap")
	}

	// Same promotion from a large Retry-After alone (no unified header).
	ra := http.Header{}
	ra.Set("Retry-After", "14400") // 4h
	if !isAccountCap429(429, []byte("Too many requests"), ra, now) {
		t.Fatal("a generic 429 with Retry-After: 14400 must promote to a rehome-worthy cap")
	}

	// The negative fence: a SHORT wait (under the in-handler ceiling) stays a transient throttle —
	// it keeps its seat and rides the exponential backoff, no rehome. 30s is well under the 90s
	// default ceiling.
	shortReset := http.Header{}
	shortReset.Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(now.Add(30*time.Second).Unix(), 10))
	if cls, capWait := classifyLimit429(429, []byte("Too many requests"), shortReset, now); cls.Reason != resume.LimitRate || capWait != "" {
		t.Fatalf("short-wait throttle = (%q, %q), want (rate_limited, \"\") — a brief wait keeps its seat", cls.Reason, capWait)
	}
	if isAccountCap429(429, []byte("Too many requests"), shortReset, now) {
		t.Fatal("a 30s-wait generic 429 must NOT be an account cap — it is a real transient throttle")
	}
}

// A plain server-side throttle (rate_limited) and a non-429 must leave the wait decision
// untouched: capWait stays empty so Retry-After / exponential behavior is unchanged.
func TestClassifyLimit429_ThrottleAndNon429Unchanged(t *testing.T) {
	cls, capWait := classifyLimit429(429, []byte("Too many requests"), nil, time.Now())
	if cls.Reason != resume.LimitRate || capWait != "" {
		t.Fatalf("throttle = (%q, %q), want (rate_limited, \"\")", cls.Reason, capWait)
	}
	cls, capWait = classifyLimit429(statusOverloaded, []byte("overloaded"), nil, time.Now())
	if cls.Reason != "" || capWait != "" {
		t.Fatalf("529 = (%q, %q), want no classification (only a 429 is a rate-limit verdict)", cls.Reason, capWait)
	}
}

// The acceptance fence of #1362: the LIVE classification and the POST-MORTEM one
// (resume.ClassifyLimitText, what `fak resume scan` runs on a dead transcript) must agree
// on the same refusal text — one vocabulary, two call sites, no divergence. The table walks
// EVERY member of the closed vocabulary (session/weekly/usage/rate), including the bare
// "Fable 5 limit"/usage-credits phrasing an earlier fix (#2369) had to teach the shared
// classifier, so a future edit to either call site that drops a member reds this fence.
func TestClassifyLimit429_AgreesWithPostMortemClassifier(t *testing.T) {
	cases := []struct {
		msg  string
		want string
	}{
		{"You've hit your session limit · resets 8pm", resume.LimitSession},
		{"You've hit your weekly limit · resets Oct 14", resume.LimitWeekly},
		{"usage limit reached", resume.LimitUsage},
		{"You've reached your Fable 5 limit. Run /usage-credits to continue.", resume.LimitUsage},
		{"Rate limited", resume.LimitRate},
	}
	for _, tc := range cases {
		// Post-mortem: the exact classifier `fak resume scan` runs on a dead transcript.
		post, ok := resume.ClassifyLimitText(tc.msg)
		if !ok {
			t.Fatalf("post-mortem classifier rejected %q", tc.msg)
		}
		if post != tc.want {
			t.Fatalf("post-mortem=%q for %q, want %q", post, tc.msg, tc.want)
		}
		// Live: the same refusal wrapped in the wire shape the 429 path receives.
		cls, _ := classifyLimit429(429, anthropicLimitBody(tc.msg), nil, time.Now())
		if cls.Reason != post {
			t.Fatalf("live=%q post-mortem=%q for %q — the two call sites diverged", cls.Reason, post, tc.msg)
		}
	}
}

// End-to-end through Complete: a 429 session cap whose unified header names a reset ~3s
// out must drive a retry wait anchored on that reset (well past the 600ms first-retry
// exponential base), and the turn must still succeed on the post-reset probe.
func TestComplete_SessionCap429_WaitsTowardReset(t *testing.T) {
	resetIn := 3 * time.Second
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(time.Now().Add(resetIn).Unix(), 10))
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write(anthropicLimitBody("You've hit your session limit · resets 8pm"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	var waits []time.Duration
	p.RetryNotify = func(_, _ int, wait time.Duration) { waits = append(waits, wait) }
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete after capped 429: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	if len(waits) != 1 {
		t.Fatalf("RetryNotify fired %d times, want 1", len(waits))
	}
	// jitterUp bounds the honored wait to [reset, reset*1.25]; the exponential first-retry
	// base is 600ms, so anything past ~2s proves the wait was reset-anchored.
	if waits[0] < 2*time.Second || waits[0] > resetIn+resetIn/4+time.Second {
		t.Fatalf("retry wait = %s, want ~%s (anchored on the relayed reset, not the 600ms schedule)", waits[0], resetIn)
	}
}

// The error Complete finally surfaces must NAME the classification the recovery acted on
// (#1362's no-divergence rule): a capped account's exhausted turn reports session_limit +
// the banner's reset hint, not an anonymous 429.
func TestComplete_ExhaustedCap429_ErrorNamesClassification(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1") // single attempt: exhaust without sleeping
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write(anthropicLimitBody("You've hit your session limit · resets 8pm"))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Complete succeeded against a permanently capped upstream")
	}
	var se *UpstreamStatusError
	if !errors.As(err, &se) {
		t.Fatalf("error %T does not unwrap to *UpstreamStatusError: %v", err, err)
	}
	if se.LimitReason != resume.LimitSession {
		t.Fatalf("LimitReason = %q, want %q", se.LimitReason, resume.LimitSession)
	}
	if se.LimitResetHint == "" {
		t.Fatal("LimitResetHint empty, want the banner's reset hint carried on the surfaced error")
	}
}

// The AC2 acceptance fence of #1362 — the end-to-end mirror of
// TestComplete_SessionCap429_WaitsTowardReset. Where that test proves a 5-hour cap PARKS
// toward its named reset, this proves the OTHER half of the live vocabulary: a transient
// rate_limited 429 (no session/weekly/usage cap, no unified-reset header) that carries a
// Retry-After is HONORED as a short retry and the turn succeeds on the retry — it is NOT
// promoted to a cap park toward a far-off reset. So the live classification leaves the
// throttle path "retries as today", closing the acceptance on both sides of the taxonomy.
func TestComplete_Throttle429_HonorsRetryAfterAndRetries(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Retry-After", "1") // a short server-side throttle, honored as ~1s
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write(anthropicLimitBody("Rate limited"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	var waits []time.Duration
	p.RetryNotify = func(_, _ int, wait time.Duration) { waits = append(waits, wait) }
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete after transient 429: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok (the turn must succeed on the post-throttle retry)", comp.Message.Content)
	}
	if len(waits) != 1 {
		t.Fatalf("RetryNotify fired %d times, want 1", len(waits))
	}
	// Honored: the wait is at least the server's 1s Retry-After (jitterUp bounds it to
	// [1s, 1.25s]) — the provider's own instruction, not the bare exponential base.
	if waits[0] < time.Second {
		t.Fatalf("retry wait = %s, want >= 1s (the server's Retry-After honored)", waits[0])
	}
	// Transient, not capped: a session/weekly/usage cap WITHOUT a machine-readable reset waits
	// the slow cap-probe floor (defaultCapProbeInterval); a real throttle must stay nowhere near
	// it — reaching it would mean a transient 429 was wrongly parked as an account cap.
	if waits[0] >= defaultCapProbeInterval {
		t.Fatalf("retry wait = %s reached the cap-probe floor (%s) — a transient throttle was wrongly parked as a cap", waits[0], defaultCapProbeInterval)
	}
}
