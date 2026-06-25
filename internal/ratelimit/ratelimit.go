// Package ratelimit is the throughput/cost governor: the adjudicator that turns
// the already-plumbed RATE_LIMITED reason into an actual enforcer.
//
// THE GAP THIS CLOSES (issue #13). RATE_LIMITED is in the closed refusal
// vocabulary (abi.ReasonRateLimited) AND the deny-loopback disposition map maps it
// to WAIT (kernel.Disposition) — but until now NO adjudicator ever emitted it.
// The reason code was plumbed and the disposition was steered; the enforcer was
// missing. So the one regime that most needs a cap — async-batch at volume, an
// escalation loop, a runaway agent — had nothing to throttle it. This leaf is that
// enforcer: a per-key call-count quota and/or cumulative-cost budget that emits
// Deny(RATE_LIMITED) when a key exceeds its cap, so the loop receives a WAIT (a
// back-off) instead of burning another model turn on a call that cannot proceed.
//
// WHY AN ADJUDICATOR, NOT A CORE EDIT. Like every other rung, the limiter attaches
// through the frozen registry (abi.RegisterAdjudicator) from its own init(); the
// kernel folds it in restrictiveness order with no edit to the kernel or the
// authoritative monitor. Enabling it is one blank-import line in
// internal/registrations; configuring it is environment (below) — never a fork.
//
// FAIL-OPEN BY DEFAULT (the "no spontaneous refusal" contract, dos.toml SAFETY).
// A limiter with no cap configured Defers on every call — registering this leaf
// changes no behavior until an operator sets a cap. Two ways to set one:
//
//   - Declarative: the fak-policy/v1 `rate_limit:` block (issue #699, Epic 8),
//     applied to Default at boot and on --policy hot-reload from cmd/fak's
//     applyRuntime — the same surface SafeSinks/Authorize reach through ifc. A
//     policy load is AUTHORITATIVE over the cap: present installs it, absent resets
//     the limiter to inert. Env is the fallback only when no --policy is given.
//   - Environment, read once at process start (the same stance internal/modelengine
//     takes with FAK_MODEL_DIR) — the keyless / single-process deploy surface:
//     FAK_RATELIMIT_MAX_CALLS=<n>   per-key admitted-call quota
//     FAK_RATELIMIT_MAX_COST=<n>    per-key cumulative cost budget (arg bytes ~ tokens)
//     FAK_RATELIMIT_KEY=trace|tool|global   which dimension to bucket by (default trace)
//     FAK_RATELIMIT_RETRY_AFTER_MS=<n>  advisory back-off (ms) surfaced on the WAIT
//   - In-process / tests: (*Limiter).SetLimit.
//
// RETRY-AFTER ON WAIT (issue #699). An over-cap deny climbs from a bare WAIT token
// to a WAIT carrying an advisory `retry_after` in its deny meta — the recoverable
// back-off the loop pairs with WAIT the way errno pairs EAGAIN with a retry window.
// The value is the operator-declared RetryAfter when one is set (manifest
// `retry_after_ms` / env), else a small default advisory constant: this limiter is a
// fixed-ceiling quota with NO time window, so a duration is an estimate, not a
// guaranteed admission time — advisory back-off like HTTP Retry-After, not a
// reservation (windowed/decaying budgets stay the lifecycle leaf's job).
//
// BOUNDED STATE. The per-key counters live in a map capped at maxKeys entries.
// Past the ceiling a NEW key is not tracked (fail-open) rather than evicting a live
// budget — eviction would let a caller cycle keys to reset its own cap. Existing
// keys keep enforcing. (Eviction/TTL for very-long-lived processes is the
// lifecycle leaf's job, issue #12; Reset/ResetAll below is its hook.)
package ratelimit

import (
	"context"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// by labels every verdict this leaf emits (forensics; never dispatch).
const by = "ratelimit"

// capRateLimit is the negotiable feature token the leaf advertises.
const capRateLimit abi.Capability = "ratelimit.v1"

// defaultMaxKeys bounds the per-key counter map so a high-cardinality key
// dimension (e.g. per-trace under churn) cannot grow memory without limit.
const defaultMaxKeys = 8192

// defaultRetryAfter is the advisory back-off surfaced on an over-cap WAIT when no
// explicit RetryAfter is configured. This limiter is a fixed-ceiling quota with no
// time window (no calls/sec or tokens/sec rate is tracked), so there is no duration
// it can honestly DERIVE from the cap — any magnitude-based formula would be
// fabricated precision. A small documented constant is the honest default: it tells
// the loop "this is a hard quota, back off and retry" without promising a specific
// admission time. An operator who knows better declares `retry_after_ms`.
const defaultRetryAfter = 1 * time.Second

// KeyMode selects the dimension a budget buckets by.
type KeyMode uint8

const (
	// KeyPerTrace buckets by ToolCall.TraceID — the per-run / per-agent identity.
	// This is the default: it caps one agent/run without throttling its peers.
	KeyPerTrace KeyMode = iota
	// KeyPerTool buckets by tool name — cap a hot tool across all callers.
	KeyPerTool
	// KeyGlobal is one shared budget for the whole process (a global concurrency/
	// volume ceiling).
	KeyGlobal
)

// Limit is the per-key budget. A zero Limit (both cap fields 0) is UNLIMITED: the
// limiter Defers on every call, so the leaf is inert until a cap is set. RetryAfter
// alone (no cap) is meaningless and still inert — a back-off only matters once a
// cap can fire.
type Limit struct {
	MaxCalls   int           // max ADMITTED calls per key (a quota). 0 = unlimited.
	MaxCost    int64         // max cumulative cost per key (arg bytes ~ tokens; a budget). 0 = unlimited.
	RetryAfter time.Duration // advisory back-off surfaced on the over-cap WAIT; 0 = defaultRetryAfter.
}

// unlimited reports whether no cap is configured (the inert state).
func (l Limit) unlimited() bool { return l.MaxCalls == 0 && l.MaxCost == 0 }

// counter is the running consumption for one key.
type counter struct {
	calls int
	cost  int64
}

// Limiter is the throughput/cost-governor adjudicator. It holds per-key counters
// guarded by one mutex; config (lim/mode) is read under the same lock so SetLimit
// is safe against a concurrent decide.
type Limiter struct {
	mu      sync.Mutex
	lim     Limit
	mode    KeyMode
	byKey   map[string]*counter
	maxKeys int

	// forensics / KPI (read via Stats).
	admits  int64
	denies  int64
	dropped int64 // calls left untracked because the key ceiling was hit (fail-open)
}

// New returns an inert limiter (no cap; KeyPerTrace). It Defers on every call
// until SetLimit (or env, for Default) configures a cap.
func New() *Limiter {
	return &Limiter{mode: KeyPerTrace, byKey: map[string]*counter{}, maxKeys: defaultMaxKeys}
}

// SetLimit installs a new cap and key dimension. It does NOT clear existing
// counters (config and consumed-state are separate; use ResetAll for that), so an
// operator can tighten a cap mid-flight without wiping accrued budgets.
func (r *Limiter) SetLimit(l Limit, mode KeyMode) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lim, r.mode = l, mode
}

// Reset clears one key's accrued consumption (the per-trace ledger reset the
// lifecycle leaf drives when a trace ends — issue #12).
func (r *Limiter) Reset(key string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.byKey, key)
}

// ResetAll clears every key's consumption (a fresh window).
func (r *Limiter) ResetAll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byKey = map[string]*counter{}
}

// Caps satisfies abi.Adjudicator. The negotiable token is registered in init();
// the adjudicator itself advertises none on the call path.
func (r *Limiter) Caps() []abi.Capability { return nil }

// Adjudicate enforces the cap. Under the cap it Defers (abstains, like every other
// rung that has nothing to prove); over the cap it emits Deny(RATE_LIMITED), whose
// reason the kernel maps to a WAIT disposition — the loop backs off rather than
// retrying a call that cannot succeed.
func (r *Limiter) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Inert until configured: no cap => abstain, never refuse.
	if r.lim.unlimited() {
		return defer_()
	}
	if c == nil {
		return defer_()
	}

	key := r.keyLocked(c)
	st := r.byKey[key]
	if st == nil {
		// Bounded state: past the key ceiling, do not track a new key (fail-open)
		// rather than evict a live budget (which would let a caller reset its cap by
		// cycling keys). Existing keys still enforce.
		if len(r.byKey) >= r.maxKeys {
			r.dropped++
			return defer_()
		}
		st = &counter{}
		r.byKey[key] = st
	}

	cost := costOf(c)

	// Check BEFORE consuming: a refused call must not consume budget. So a caller
	// that keeps probing an over-cap key gets a deterministic, idempotent WAIT
	// instead of digging the hole deeper, and a denied call never advances the
	// counter past the cap.
	if r.lim.MaxCalls > 0 && st.calls+1 > r.lim.MaxCalls {
		r.denies++
		return denyVerdict(key, "max_calls", int64(r.lim.MaxCalls), r.retryAfterLocked())
	}
	if r.lim.MaxCost > 0 && st.cost+cost > r.lim.MaxCost {
		r.denies++
		return denyVerdict(key, "max_cost", r.lim.MaxCost, r.retryAfterLocked())
	}

	st.calls++
	st.cost += cost
	r.admits++
	return defer_()
}

// keyLocked resolves the bucket key for a call under the configured dimension.
// Caller holds r.mu.
func (r *Limiter) keyLocked(c *abi.ToolCall) string {
	switch r.mode {
	case KeyPerTool:
		return "tool:" + c.Tool
	case KeyGlobal:
		return "*"
	default: // KeyPerTrace
		return "trace:" + c.TraceID
	}
}

// Stats reports admits / denies / dropped (KPI + forensics).
func (r *Limiter) Stats() (admits, denies, dropped int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.admits, r.denies, r.dropped
}

// defer_ is the abstain verdict (named with a trailing underscore — `defer` is a
// keyword). Every "nothing to refuse" path returns this.
func defer_() abi.Verdict { return abi.Verdict{Kind: abi.VerdictDefer, By: by} }

// retryAfterLocked is the advisory back-off surfaced on an over-cap deny. The
// operator-declared RetryAfter wins; otherwise the default constant stands in
// (there is no rate window to derive a real duration from). Caller holds r.mu.
func (r *Limiter) retryAfterLocked() time.Duration {
	if r.lim.RetryAfter > 0 {
		return r.lim.RetryAfter
	}
	return defaultRetryAfter
}

// denyVerdict is the bounded-disclosure RATE_LIMITED refusal: it names the cap
// that fired, its limit, the key, and the advisory retry-after — never the call's
// arguments. The kernel derives the WAIT disposition from the reason and surfaces
// the retry-after on the deny-as-value (issue #699).
func denyVerdict(key, cap string, limit int64, retryAfter time.Duration) abi.Verdict {
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonRateLimited,
		By:     by,
		Meta: map[string]string{
			"cap":            cap,
			"limit":          strconv.FormatInt(limit, 10),
			"key":            key,
			"retry_after":    retryAfter.String(),
			"retry_after_ms": strconv.FormatInt(retryAfter.Milliseconds(), 10),
		},
	}
}

// costOf is a call's cost against the MaxCost budget. A caller may supply a real
// token count via Meta["fak.ratelimit.cost"]; otherwise the cheap, resolver-free
// proxy is the argument byte length (inline length when present, else the Ref's
// declared Len). There is no NL tokenizer on the hot path, so bytes is the honest
// stand-in for tokens — the same stance internal/modelengine takes.
func costOf(c *abi.ToolCall) int64 {
	if c.Meta != nil {
		if s, ok := c.Meta["fak.ratelimit.cost"]; ok {
			if n, err := strconv.ParseInt(s, 10, 64); err == nil && n >= 0 {
				return n
			}
		}
	}
	if c.Args.Kind == abi.RefInline {
		return int64(len(c.Args.Inline))
	}
	if c.Args.Len > 0 {
		return c.Args.Len
	}
	return 0
}

// Default is the registered governor. It is inert unless the environment sets a
// cap (configureFromEnv), so the defconfig blank-import never refuses anything on
// its own.
var Default = New()

func init() {
	Default.configureFromEnv()
	// Rank 8: an early load-shed, after the free well-formedness rung (grammar=5)
	// and before the expensive trust rungs (preflight=10, plancfi=25, ifc=30,
	// shipgate=40, the rank-100 monitor) — so an over-cap call is shed cheaply,
	// before the heavy checks run.
	abi.RegisterAdjudicator(8, Default)
	abi.RegisterCapability(capRateLimit)
}

// configureFromEnv reads the cap from the environment once at process start. With
// no FAK_RATELIMIT_* set the limiter stays inert.
func (r *Limiter) configureFromEnv() {
	mc := envInt("FAK_RATELIMIT_MAX_CALLS")
	cost := envInt("FAK_RATELIMIT_MAX_COST")
	if mc <= 0 && cost <= 0 {
		return
	}
	mode := KeyPerTrace
	switch os.Getenv("FAK_RATELIMIT_KEY") {
	case "tool":
		mode = KeyPerTool
	case "global":
		mode = KeyGlobal
	}
	ra := envInt("FAK_RATELIMIT_RETRY_AFTER_MS")
	r.SetLimit(Limit{MaxCalls: mc, MaxCost: int64(cost), RetryAfter: time.Duration(ra) * time.Millisecond}, mode)
}

// envInt reads a non-negative int from env, returning 0 (unset/unlimited) on an
// absent or malformed value.
func envInt(name string) int {
	s := os.Getenv(name)
	if s == "" {
		return 0
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return 0
	}
	return n
}
