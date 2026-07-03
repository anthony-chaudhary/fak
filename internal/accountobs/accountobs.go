// Package accountobs observes the ACCOUNT side of a guarded session's economy: the
// rate-limit / usage headers the upstream provider relays on every response — the
// subscription "unified" windows Claude Pro/Max accounts are governed by
// (anthropic-ratelimit-unified-*: per-window utilization, status, reset) and the
// API-key token/request families (anthropic-ratelimit-<family>-limit/-remaining/
// -reset, plus the x-ratelimit-* OpenAI-compatible spelling) — so `fak guard` can
// answer "how loaded is the account this session is spending?" the same way it
// already answers "how loaded is the node?" (internal/harnessres).
//
// PROVENANCE FENCE (the conflation-score law). Every value this package captures is
// OBSERVED / provider-relayed: the provider computed it, fak only carried it. fak
// neither sets nor controls these numbers, and a missing header is rendered as
// absent — never as a fabricated 0. The only WITNESSED numbers here are fak's own
// response tallies (how many upstream responses it saw, how many were 429s).
//
// Stdlib-only leaf, off the hot path (one map fold per upstream response header
// set); imports nothing internal. Wired by the host: the gateway exposes an
// UpstreamResponseObserver seam and `fak guard` points it at a Tracker.
package accountobs

import (
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// observedPrefixes are the response-header families the tracker captures. Everything
// else is ignored, so no credential- or content-bearing header can ever land in a
// snapshot.
var observedPrefixes = []string{"anthropic-ratelimit-", "x-ratelimit-"}

// Tracker folds upstream response headers into the latest-value account view. The
// zero value is not usable; construct with New. Safe for concurrent Observe calls
// (upstream turns can overlap under a replica router).
type Tracker struct {
	mu          sync.Mutex
	responses   int
	rateLimited int
	lastStatus  int
	headers     map[string]string // lowercased header name -> latest value
	nowFn       func() time.Time
	lastAt      time.Time
}

// New returns an empty Tracker reading the wall clock.
func New() *Tracker {
	return &Tracker{headers: map[string]string{}, nowFn: time.Now}
}

// Observe folds one upstream response's status + headers. Later responses overwrite
// earlier values header-by-header (the account view is "latest known"), so a window
// the provider stops relaying keeps its last observed value rather than vanishing
// mid-session. Nil-safe.
func (t *Tracker) Observe(status int, h http.Header) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.responses++
	t.lastStatus = status
	if status == http.StatusTooManyRequests {
		t.rateLimited++
	}
	t.lastAt = t.nowFn()
	for k, vs := range h {
		lk := strings.ToLower(k)
		captured := false
		for _, p := range observedPrefixes {
			if strings.HasPrefix(lk, p) {
				captured = true
				break
			}
		}
		if !captured || len(vs) == 0 {
			continue
		}
		t.headers[lk] = strings.TrimSpace(vs[0])
	}
}

// Snapshot is one folded reading of the account view: fak's WITNESSED response
// tallies plus the latest OBSERVED provider-relayed rate-limit headers.
type Snapshot struct {
	Responses   int // WITNESSED: upstream responses fak observed this session
	RateLimited int // WITNESSED: how many of those were HTTP 429
	LastStatus  int
	LastAt      time.Time
	Headers     map[string]string // OBSERVED: latest value per relayed rate-limit header
}

// Snapshot returns a copy of the current account view. Nil-safe.
func (t *Tracker) Snapshot() Snapshot {
	if t == nil {
		return Snapshot{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	h := make(map[string]string, len(t.headers))
	for k, v := range t.headers {
		h[k] = v
	}
	return Snapshot{
		Responses:   t.responses,
		RateLimited: t.rateLimited,
		LastStatus:  t.lastStatus,
		LastAt:      t.lastAt,
		Headers:     h,
	}
}

// UnifiedWindow is one subscription usage window the provider relayed (e.g. the 5h
// and 7d windows a Claude Pro/Max account is governed by). Name is the window token
// from the header ("5h", "7d", "7d_opus", …); "" is the top-level unified scope
// (anthropic-ratelimit-unified-status / -reset with no window segment).
type UnifiedWindow struct {
	Name            string
	UtilizationPct  float64 // 0..100+; only meaningful when HaveUtilization
	HaveUtilization bool
	Status          string // e.g. allowed | allowed_warning | rejected ("" when not relayed)
	Reset           time.Time
	HaveReset       bool
}

// unifiedPrefix is the subscription-window header family.
const unifiedPrefix = "anthropic-ratelimit-unified-"

// Unified parses the subscription windows out of the snapshot's headers, sorted by
// window name (the top-level ""-named scope first, then "5h" before "7d" — plain
// lexicographic order happens to be chronological for the known windows).
func (s Snapshot) Unified() []UnifiedWindow {
	byName := map[string]*UnifiedWindow{}
	get := func(name string) *UnifiedWindow {
		w := byName[name]
		if w == nil {
			w = &UnifiedWindow{Name: name}
			byName[name] = w
		}
		return w
	}
	for k, v := range s.Headers {
		rest, ok := strings.CutPrefix(k, unifiedPrefix)
		if !ok {
			continue
		}
		switch {
		case rest == "status":
			get("").Status = v
		case rest == "reset":
			if ts, ok := parseReset(v); ok {
				w := get("")
				w.Reset, w.HaveReset = ts, true
			}
		case strings.HasSuffix(rest, "-utilization"):
			if pct, ok := parseUtilizationPct(v); ok {
				w := get(strings.TrimSuffix(rest, "-utilization"))
				w.UtilizationPct, w.HaveUtilization = pct, true
			}
		case strings.HasSuffix(rest, "-status"):
			get(strings.TrimSuffix(rest, "-status")).Status = v
		case strings.HasSuffix(rest, "-reset"):
			if ts, ok := parseReset(v); ok {
				w := get(strings.TrimSuffix(rest, "-reset"))
				w.Reset, w.HaveReset = ts, true
			}
		}
	}
	out := make([]UnifiedWindow, 0, len(byName))
	for _, w := range byName {
		out = append(out, *w)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// UsageRejection is the folded verdict of whether a response's subscription rate-limit
// headers show a self-recovering USAGE/OVERAGE cap — a rolling-window rejection the account
// clears on its own at Reset, NOT a permanent wall. It is the ONE header taxonomy shared by
// the live retry classifier (internal/agent) and the post-mortem/resume classifier
// (internal/resume), so both sides draw the cap-vs-wall line the same way from the same bytes.
type UsageRejection struct {
	// Rejected is true when the provider relayed a usage/overage rejection: either
	// anthropic-ratelimit-unified-overage-status: rejected, or any unified window
	// (top-level, 5h, 7d, …) whose status is "rejected".
	Rejected bool
	// Window names the rejected unified window ("" top-level, "5h", "7d", …), or "overage"
	// when only the overage-status header signalled it. It lets the caller pick the matching
	// reset window (session vs weekly).
	Window string
	// Reset is the rejected window's reset instant when one was relayed; HaveReset is false
	// when the header named no future reset (the caller then falls back to a slow cap probe).
	Reset     time.Time
	HaveReset bool
}

// overageStatusKey is the header the provider sets to "rejected" when an account has overage
// disabled and this request would tip a rolling window past its cap — the exact live day30
// signal. It is NOT a per-window field the Unified() parser surfaces, so it is read directly.
const overageStatusKey = "anthropic-ratelimit-unified-overage-status"

// UsageOverageRejection folds a response's headers into the shared usage/overage verdict. It is
// true when the overage-status header is "rejected" OR any unified window reports status
// "rejected" — a self-recovering cap, never a standing wall. When a window is rejected its reset
// is returned so the caller can wait toward it (the same reset Unified() already parsed). A
// response with no such signal (a genuine org/permission wall carries none) returns Rejected
// false. Nil/empty headers are safe (Rejected false). This is the single source of truth for the
// "is this a recoverable cap?" question across the live and post-mortem paths.
func UsageOverageRejection(h http.Header) UsageRejection {
	if len(h) == 0 {
		return UsageRejection{}
	}
	t := New()
	t.Observe(http.StatusForbidden, h)
	windows := t.Snapshot().Unified()

	// Locate the top-level ("") window's reset up front: it is where the overage-status cap's
	// reset lives (the "overage" pseudo-window Unified() derives from the overage-status header
	// carries none of its own). Named windows (5h/7d) carry their own reset directly.
	var topReset time.Time
	var topHaveReset bool
	for _, w := range windows {
		if w.Name == "" && w.HaveReset {
			topReset, topHaveReset = w.Reset, true
		}
	}

	// A rejected REAL usage window (5h/7d, or the top-level "") is the richest signal — it carries
	// its own reset and names the window so the caller can pick session vs weekly. The synthetic
	// "overage" pseudo-window (from the overage-status header) is handled separately below so its
	// missing reset can borrow the top-level one. Prefer a NAMED window over the top-level scope.
	var best *UnifiedWindow
	for i := range windows {
		w := windows[i]
		if w.Name == "overage" { // synthetic; handled by the overage-status branch below
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(w.Status), "rejected") {
			continue
		}
		if best == nil || (best.Name == "" && w.Name != "") {
			ww := w
			best = &ww
		}
	}
	if best != nil {
		reset, have := best.Reset, best.HaveReset
		if !have { // a rejected window that relayed no reset of its own borrows the top-level reset
			reset, have = topReset, topHaveReset
		}
		return UsageRejection{Rejected: true, Window: best.Name, Reset: reset, HaveReset: have}
	}
	// No rejected real window, but the overage-status header alone can mark the cap (the day30
	// case: overage disabled, request would exceed the cap). Its reset rides the top-level
	// unified-reset, so surface that when present.
	if strings.EqualFold(strings.TrimSpace(h.Get(http.CanonicalHeaderKey(overageStatusKey))), "rejected") {
		return UsageRejection{Rejected: true, Window: "overage", Reset: topReset, HaveReset: topHaveReset}
	}
	return UsageRejection{}
}

// Family is one API-key rate-limit family the provider relayed (requests,
// input-tokens, output-tokens, tokens — the anthropic-ratelimit-<family>-limit/
// -remaining/-reset triple, or its x-ratelimit-* spelling).
type Family struct {
	Name          string
	Limit         int64
	HaveLimit     bool
	Remaining     int64
	HaveRemaining bool
	Reset         time.Time
	HaveReset     bool
}

// Families parses the API-key families out of the snapshot's headers, sorted by
// name. Unified subscription headers are excluded (they carry no -limit/-remaining
// pair and are parsed by Unified).
func (s Snapshot) Families() []Family {
	byName := map[string]*Family{}
	get := func(name string) *Family {
		f := byName[name]
		if f == nil {
			f = &Family{Name: name}
			byName[name] = f
		}
		return f
	}
	for k, v := range s.Headers {
		if strings.HasPrefix(k, unifiedPrefix) {
			continue
		}
		rest := ""
		if r, ok := strings.CutPrefix(k, "anthropic-ratelimit-"); ok {
			rest = r
		} else if r, ok := strings.CutPrefix(k, "x-ratelimit-"); ok {
			rest = r
		} else {
			continue
		}
		switch {
		case strings.HasSuffix(rest, "-limit"):
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				f := get(strings.TrimSuffix(rest, "-limit"))
				f.Limit, f.HaveLimit = n, true
			}
		case strings.HasSuffix(rest, "-remaining"):
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				f := get(strings.TrimSuffix(rest, "-remaining"))
				f.Remaining, f.HaveRemaining = n, true
			}
		case strings.HasSuffix(rest, "-reset"):
			if ts, ok := parseReset(v); ok {
				f := get(strings.TrimSuffix(rest, "-reset"))
				f.Reset, f.HaveReset = ts, true
			}
		}
	}
	out := make([]Family, 0, len(byName))
	for _, f := range byName {
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// parseUtilizationPct normalizes a relayed utilization value to percent. The wire
// has carried both spellings (a 0..1 fraction and a 0..100 percent), so: a value in
// [0,1] is read as a fraction (1.0 → 100%), anything above 1 as an already-percent
// number. Negative or unparseable values are dropped (absent, never a fake 0).
func parseUtilizationPct(v string) (float64, bool) {
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil || f < 0 {
		return 0, false
	}
	if f <= 1 {
		return f * 100, true
	}
	return f, true
}

// parseReset accepts both reset spellings the provider uses: unix epoch seconds
// (the unified family) and RFC3339 (the API-key families).
func parseReset(v string) (time.Time, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return time.Time{}, false
	}
	if n, err := strconv.ParseInt(v, 10, 64); err == nil {
		if n <= 0 {
			return time.Time{}, false
		}
		return time.Unix(n, 0), true
	}
	if ts, err := time.Parse(time.RFC3339, v); err == nil {
		return ts, true
	}
	return time.Time{}, false
}

// Report renders the one-line human account-usage summary for the guard exit
// summary. Empty when no upstream response was observed (a session that never
// reached the provider has no account economy to report). now anchors the
// "resets in Xm" rendering so the line is unit-testable.
func (s Snapshot) Report(now time.Time) string {
	if s.Responses == 0 {
		return ""
	}
	var parts []string
	windows := s.Unified()
	for _, w := range windows {
		if w.Name == "" {
			continue // top-level status/reset rendered after the windows
		}
		seg := w.Name + " window"
		if w.HaveUtilization {
			seg += fmt.Sprintf(" %.0f%% used", w.UtilizationPct)
		} else if w.Status != "" {
			seg += " " + w.Status
		} else {
			continue
		}
		if w.HaveReset {
			seg += " (resets " + formatReset(w.Reset, now) + ")"
		}
		parts = append(parts, seg)
	}
	for _, w := range windows {
		if w.Name != "" {
			continue
		}
		if w.Status != "" {
			seg := "status " + w.Status
			if w.HaveReset {
				seg += " (resets " + formatReset(w.Reset, now) + ")"
			}
			parts = append(parts, seg)
		} else if w.HaveReset {
			parts = append(parts, "resets "+formatReset(w.Reset, now))
		}
	}
	for _, f := range s.Families() {
		seg := f.Name
		switch {
		case f.HaveRemaining && f.HaveLimit:
			seg += fmt.Sprintf(" %d/%d remaining", f.Remaining, f.Limit)
		case f.HaveRemaining:
			seg += fmt.Sprintf(" %d remaining", f.Remaining)
		case f.HaveLimit:
			seg += fmt.Sprintf(" limit %d", f.Limit)
		default:
			continue
		}
		if f.HaveReset {
			seg += " (resets " + formatReset(f.Reset, now) + ")"
		}
		parts = append(parts, seg)
	}
	var b strings.Builder
	if len(parts) > 0 {
		b.WriteString("rate-limit [OBSERVED provider-relayed] ")
		b.WriteString(strings.Join(parts, ", "))
	} else {
		b.WriteString("provider relayed no rate-limit headers")
	}
	fmt.Fprintf(&b, "; %d upstream response(s)", s.Responses)
	if s.RateLimited > 0 {
		fmt.Fprintf(&b, ", %d rate-limited (429)", s.RateLimited)
	}
	return b.String()
}

// formatReset renders a reset instant as local wall-clock plus a relative delta —
// "15:04 (in 42m)" — the two forms an operator actually acts on. A reset already in
// the past (a stale header from earlier in the session) shows only the clock time.
func formatReset(ts, now time.Time) string {
	clock := ts.Local().Format("15:04")
	d := ts.Sub(now)
	if d <= 0 {
		return clock
	}
	return clock + ", in " + humanDur(d)
}

func humanDur(d time.Duration) string {
	switch {
	case d >= 24*time.Hour:
		return fmt.Sprintf("%.1fd", d.Hours()/24)
	case d >= time.Hour:
		return fmt.Sprintf("%.1fh", d.Hours())
	case d >= time.Minute:
		return fmt.Sprintf("%dm", int(d.Round(time.Minute).Minutes()))
	default:
		return fmt.Sprintf("%ds", int(d.Round(time.Second).Seconds()))
	}
}

// PrometheusText renders the fak_account_* gauge family for the /metrics harness
// block. Only parsed values emit sample lines (a header the provider did not relay
// is absent, never a fabricated 0); HELP text carries the OBSERVED provenance label
// per the conflation-score law.
func (s Snapshot) PrometheusText() string {
	if s.Responses == 0 {
		return ""
	}
	var b strings.Builder
	windows := s.Unified()
	families := s.Families()

	wroteUtil := false
	for _, w := range windows {
		if !w.HaveUtilization {
			continue
		}
		if !wroteUtil {
			writeHelp(&b, "fak_account_ratelimit_utilization_pct", "OBSERVED provider-relayed account rate-limit window utilization percent (anthropic-ratelimit-unified-*-utilization); the provider computes this, fak only relays it.", "gauge")
			wroteUtil = true
		}
		fmt.Fprintf(&b, "fak_account_ratelimit_utilization_pct{window=%q} %s\n", w.Name, promFloat(w.UtilizationPct))
	}
	wroteReset := false
	for _, w := range windows {
		if !w.HaveReset {
			continue
		}
		if !wroteReset {
			writeHelp(&b, "fak_account_ratelimit_reset_unix_seconds", "OBSERVED provider-relayed account rate-limit window reset time (unix seconds); the provider computes this, fak only relays it.", "gauge")
			wroteReset = true
		}
		name := w.Name
		if name == "" {
			name = "unified"
		}
		fmt.Fprintf(&b, "fak_account_ratelimit_reset_unix_seconds{window=%q} %d\n", name, w.Reset.Unix())
	}
	wroteRemaining := false
	for _, f := range families {
		if !f.HaveRemaining {
			continue
		}
		if !wroteRemaining {
			writeHelp(&b, "fak_account_ratelimit_remaining", "OBSERVED provider-relayed remaining budget per rate-limit family (anthropic-ratelimit-<family>-remaining / x-ratelimit-<family>-remaining); the provider computes this, fak only relays it.", "gauge")
			wroteRemaining = true
		}
		fmt.Fprintf(&b, "fak_account_ratelimit_remaining{family=%q} %d\n", f.Name, f.Remaining)
	}
	wroteLimit := false
	for _, f := range families {
		if !f.HaveLimit {
			continue
		}
		if !wroteLimit {
			writeHelp(&b, "fak_account_ratelimit_limit", "OBSERVED provider-relayed budget ceiling per rate-limit family; the provider computes this, fak only relays it.", "gauge")
			wroteLimit = true
		}
		fmt.Fprintf(&b, "fak_account_ratelimit_limit{family=%q} %d\n", f.Name, f.Limit)
	}
	writeHelp(&b, "fak_account_upstream_responses_total", "WITNESSED count of upstream provider responses fak observed this session.", "gauge")
	fmt.Fprintf(&b, "fak_account_upstream_responses_total %d\n", s.Responses)
	writeHelp(&b, "fak_account_rate_limited_responses_total", "WITNESSED count of upstream HTTP 429 (rate-limited) responses fak observed this session.", "gauge")
	fmt.Fprintf(&b, "fak_account_rate_limited_responses_total %d\n", s.RateLimited)
	return b.String()
}

func writeHelp(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func promFloat(v float64) string {
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.6f", v), "0"), ".")
}
