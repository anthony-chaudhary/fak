// retry_limit.go wires the closed rate-limit vocabulary (internal/resume, #1362) into
// the LIVE upstream retry decision. Until now the vocabulary was computed only
// post-mortem — `fak resume scan` classifying a dead transcript — while the live loops
// in chat.go/stream.go/anthropic_stream.go saw only "HTTP 429 + Retry-After" and treated
// a 5-hour session cap exactly like a transient throttle: burn the exponential schedule
// (600ms..30s per wait) against a wall that cannot clear for hours, then die. Here the
// SAME classifier (resume.ClassifyLimitResponse — one vocabulary, two call sites) runs
// at decision time, so a session/weekly/usage cap waits toward its named reset instead,
// and the error fak finally surfaces names the classification the recovery acted on.
//
// The reset instant comes from the provider's own relayed headers
// (anthropic-ratelimit-unified-*-reset, parsed by the internal/accountobs leaf — fak
// never invents a reset time), else the wait falls back to a slow cap-probe interval.
// An upstream-supplied Retry-After always outranks the derived wait (retryBackoffWait),
// and the per-wait ceiling (maxHonoredRetryAfter) + retry budget still bound every
// sleep, so a far-off reset degrades to an hourly re-probe, never an unbounded wait.

package agent

import (
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// defaultCapProbeInterval is the wait between re-probes of an upstream that answered
// with an ACCOUNT CAP (session/weekly/usage limit) that named no machine-readable reset
// — neither a Retry-After nor a unified-reset header. Hammering a capped account on the
// 600ms..30s transient schedule cannot help (the cap holds for hours) and burns the
// fleet's shared request budget; a capped upstream is probed slowly instead. A plain
// server-side throttle (rate_limited) keeps the transient schedule untouched.
const defaultCapProbeInterval = 5 * time.Minute

// maxCapProbeInterval clamps FAK_CAP_PROBE_INTERVAL so a fat-fingered value cannot
// stretch the probe past the hourly re-check the per-wait Retry-After ceiling already
// enforces for known resets.
const maxCapProbeInterval = time.Hour

// capProbeInterval resolves the unknown-reset cap probe wait: FAK_CAP_PROBE_INTERVAL
// (any Go duration ≥ 1s) overrides the 5m default, clamped to [1s, 1h].
func capProbeInterval() time.Duration {
	if v := os.Getenv("FAK_CAP_PROBE_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= time.Second {
			if d > maxCapProbeInterval {
				return maxCapProbeInterval
			}
			return d
		}
	}
	return defaultCapProbeInterval
}

// isAccountCap429 reports whether a retryable upstream response is an ACCOUNT-SCOPED 429
// rate-limit cap — a session/weekly/usage limit — as opposed to a transient server-side
// throttle (rate_limited) or a non-429 overload (5xx/529). It is the honest discriminator
// that decides whether a 429 is worth REHOMING the session to an available sibling seat:
// an account cap can hold for the full 5h/7d window (and is frequently a multi-account or
// billing condition that is "longer than it looks"), so sleeping on the capped seat toward
// its named reset is expensive when a permitted sibling seat could serve the turn now. A
// plain throttle clears in seconds, so it keeps the same seat and rides the short backoff.
//
// It reuses classifyLimit429's own signal: capWait is non-empty ONLY for an account cap
// (session/weekly/usage), empty for a rate_limited throttle and for every non-429 status.
// So there is ONE taxonomy — the same classification that drives the cap-aware wait also
// decides the rehome — not a second predicate that can drift from it.
func isAccountCap429(status int, body []byte, h http.Header, now time.Time) bool {
	_, capWait := classifyLimit429(status, body, h, now)
	return capWait != ""
}

// classifyLimit429 classifies one upstream rate/usage refusal into the closed rate-limit
// vocabulary shared with the post-mortem transcript scan, and derives the cap-aware wait the
// retry loop should use when the response carried no Retry-After.
//
// It is HEADER-AWARE (ClassifyLimitResponseWithHeader), so despite the name it now catches the
// population a 429-only gate missed: a Claude subscription USAGE/OVERAGE cap that surfaces as a
// 403 (overage disabled, a rolling window at its cap) whose org-flavored body text is
// indistinguishable from a permanent org wall — only the anthropic-ratelimit-unified-* /
// -overage-status headers separate the two. Such a 403/402 cap now classifies as
// session/weekly/usage_limit and gets a real capWait toward its reset, exactly like a 429 cap,
// instead of dying in the transient-403 arm. A 429 with no overage header is byte-for-byte
// unchanged.
//
// cls is the zero value when the response is neither a 429 nor a header-signalled cap. capWait is
// a delta-seconds string (the same form an upstream Retry-After takes, consumed by
// retryBackoffWait) and is NON-EMPTY only for an account cap — session_limit / weekly_limit /
// usage_limit: the seconds until the provider-relayed unified reset for the matching window when
// one is named, else the slow cap-probe interval. A rate_limited throttle returns capWait "" so
// the transient Retry-After/exponential behavior is byte-for-byte unchanged.
func classifyLimit429(status int, body []byte, h http.Header, now time.Time) (cls resume.LimitClassification, capWait string) {
	cls, ok := resume.ClassifyLimitResponseWithHeader(status, body, h)
	if !ok || cls.Reason == resume.LimitRate {
		return cls, ""
	}
	if reset, ok := unifiedResetFor(cls.Reason, h, now); ok {
		secs := int(math.Ceil(reset.Sub(now).Seconds()))
		if secs < 1 {
			secs = 1
		}
		return cls, strconv.Itoa(secs)
	}
	return cls, strconv.Itoa(int(capProbeInterval().Seconds()))
}

// unifiedResetFor resolves the reset instant for a classified cap from the provider's
// relayed anthropic-ratelimit-unified-* headers (via the accountobs leaf — OBSERVED
// values only, fak never fabricates one). A session_limit prefers the 5h window, a
// weekly_limit the 7d family, then either falls back to the top-level unified scope,
// then to the earliest reset still in the future; a reset at/behind `now` is stale and
// is not used. ok is false when no future reset was relayed at all.
func unifiedResetFor(reason string, h http.Header, now time.Time) (time.Time, bool) {
	t := accountobs.New()
	t.Observe(http.StatusTooManyRequests, h)
	windows := t.Snapshot().Unified()
	future := windows[:0]
	for _, w := range windows {
		if w.HaveReset && w.Reset.After(now) {
			future = append(future, w)
		}
	}
	var prefix string
	switch reason {
	case resume.LimitSession:
		prefix = "5h"
	case resume.LimitWeekly:
		prefix = "7d"
	}
	if prefix != "" {
		for _, w := range future {
			if strings.HasPrefix(w.Name, prefix) {
				return w.Reset, true
			}
		}
	}
	for _, w := range future {
		if w.Name == "" {
			return w.Reset, true
		}
	}
	var best time.Time
	for _, w := range future {
		if best.IsZero() || w.Reset.Before(best) {
			best = w.Reset
		}
	}
	return best, !best.IsZero()
}
