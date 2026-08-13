package dispatchtick

// preflight_ratelimit.go -- the rate_budget admission cap term: fold a MEASURED burst
// of genuine CONCURRENCY rate-limit walls UP into the spawn preflight so a fleet that
// is storming a throttled seat backs off instead of re-storming it.
//
// This is the fifth term in the documented effective-cap formula
// (docs/safe-to-raise-cap-checklist.md, docs/notes/SAFE-CONCURRENCY-HEADROOM-2026-07-01.md):
//
//	effective cap = min(configured_max, dos_target>0?, host_cap, seats, rate_budget)
//
// EvaluatePreflight folds the first four; ApplyGateBackpressure adds the gate-health
// term; this adds rate_budget. The static max-workers ceiling was deliberately raised
// to 20 "so host/lease/RATE gates, not the old static cap, decide when to back off" --
// this is the rate gate that decision assumed.
//
// The signal: a finished worker whose slot the witness classifier graded
// CLAIM_NO_COMMIT with reason=rate_limit (a transient 429/529 concurrency overload --
// sessionsignals.IsAPIError over the log tail, NOT a self-report). Under a shared-seat
// concurrency burst each extra spawn 429s on its first request, exits banner-only, and
// lands 0 commits, so admitting more of them just wastes admission slots. The impure
// shell counts those exits within a recent window (per backend); >= a threshold arms
// the backoff, which freezes the backend at its live count and refuses with the closed
// RATE_LIMIT_BACKOFF token so the sweep terminates on it and the loop routes to a
// DIFFERENT provider/backend (GLM/opencode, codex, another Claude seat) instead.
//
// DISAMBIGUATION (load-bearing -- most "rate-limit-looking" exits are NOT concurrency
// bursts). Only reason=rate_limit is counted. A weekly/usage cap (usage_cap), a
// model-specific cap (model_unknown), and a login/credit wall (auth_wall) are each a
// DISTINCT classifier reason and are DELIBERATELY EXCLUDED here: backing off concurrency
// does not clear any of them (they need a seat reset / a model downgrade / a re-login),
// and they are owned by the seat gate, the Layer-2 downgrade ladder, and the auth flow
// respectively. Freezing the whole fleet on a weekly-cap "429" would starve the
// multi-account goal exactly when the other providers are healthy, so the taxonomy
// filter is what keeps this term honest. The classifier's precedence already routes a
// banner-carrying cap/auth/model wall to its own reason BEFORE rate_limit
// (ClassifyNoCommitReason), so a rate_limit slot is the residual transient-overload
// class.
//
// Like the gate term this is a COMPOSABLE fold over the PreflightResult (state in,
// decision out): it can only LOWER the effective cap, abstains below the threshold, and
// holds a cold-start floor so a burst throttles GROWTH, not liveness -- a cold fleet
// that just lost every worker to a burst is still allowed ONE probe to learn whether the
// overload cleared, rather than freezing at a zero cap forever.

import (
	"fmt"
	"time"
)

// PreflightRefuseRateLimit is the verdict a spawn preflight returns when a concurrency
// rate-limit burst is the sole binding admission term. It is not safe-to-spawn, so the
// sweep stops on it exactly as it does on REFUSE_AT_CAP / REFUSE_NO_SEAT / REFUSE_GATE.
const PreflightRefuseRateLimit = "REFUSE_RATE_LIMIT"

// RateLimitBackoff is the closed-vocabulary refusal token PreflightRefuseRateLimit
// carries in its reason. It MUST stay byte-identical to the dos.toml
// [reasons.RATE_LIMIT_BACKOFF] declaration so the token this fold emits is the one
// `dos man wedge <TOKEN> --explain` verifies and the loop routes on.
const RateLimitBackoff = "RATE_LIMIT_BACKOFF"

// DefaultRateLimitMin429 is the burst threshold: the fewest recent rate_limit-classified
// worker exits (within the window, on this backend) that arm the backoff. A single 429
// is noise -- a one-off transient a retry clears -- so the default requires a genuine
// cluster before it throttles a backend. The impure shell overlays FAK_RATELIMIT_MIN_429.
const DefaultRateLimitMin429 = 3

// DefaultRateLimitMinWorkers is the cold-start floor: the fewest workers the rate term
// admits even under a live burst. Backoff throttles GROWTH, never liveness -- a floor of
// 0 would let a burst that killed every worker freeze the backend at a zero cap, and the
// window that gates it could never age out because no probe would ever run to witness the
// recovery. Holding a minimal presence keeps one probe live. The shell overlays
// FAK_RATELIMIT_MIN_WORKERS on top of this.
const DefaultRateLimitMinWorkers = 1

// RateLimitCheck carries the MEASURED concurrency-rate-limit burst the backoff term
// folds. Recent is the count of finished worker slots the witness classifier graded
// CLAIM_NO_COMMIT/reason=rate_limit within Window on the current backend -- computed by
// the impure shell from the .witness sidecars, never a worker self-report. The zero
// value means "no burst signal" (Recent 0 < any positive threshold) and never lowers the
// cap, so a caller that wires nothing keeps the existing behavior.
type RateLimitCheck struct {
	// Recent is the windowed, backend-scoped count of rate_limit-classified worker
	// exits. ONLY reason=rate_limit is counted -- usage_cap / model_unknown / auth_wall
	// are excluded by construction (see the file header disambiguation note).
	Recent int
	// Window is the lookback the count was taken over; used only to cite the burst in
	// the refusal reason. Zero means the shell disabled the term (Recent will be 0).
	Window time.Duration
	// Threshold is the burst floor that arms the backoff; <= 0 means DefaultRateLimitMin429.
	Threshold int
	// MinWorkers is the cold-start floor the backoff holds to even under a burst; <= 0
	// means DefaultRateLimitMinWorkers. The shell sets it from FAK_RATELIMIT_MIN_WORKERS.
	MinWorkers int
}

// threshold resolves the burst floor, defaulting a zero/negative Threshold to the
// built-in DefaultRateLimitMin429 so the zero-value check stays hermetic.
func (r RateLimitCheck) threshold() int {
	if r.Threshold <= 0 {
		return DefaultRateLimitMin429
	}
	return r.Threshold
}

// floor resolves the cold-start allowance, defaulting a zero/negative MinWorkers to
// DefaultRateLimitMinWorkers so the zero-value keeps the liveness-preserving default.
func (r RateLimitCheck) floor() int {
	if r.MinWorkers <= 0 {
		return DefaultRateLimitMinWorkers
	}
	return r.MinWorkers
}

// pressured reports whether the measured burst count clears the arming threshold. Pure:
// state in, decision out; a sub-threshold count abstains (no-op).
func (r RateLimitCheck) pressured() bool {
	return r.Recent >= r.threshold()
}

// ApplyRateLimitBackpressure folds a concurrency-rate-limit burst into an
// already-evaluated preflight as the rate_budget cap term. A live burst holds the
// backend at max(live, floor) -- admit no NEW concurrent worker onto a throttled seat
// beyond a minimal cold-start probe -- which can only LOWER the effective cap.
//
// A WARM backend (live at/above the floor) freezes at its live count and refuses with
// PreflightRefuseRateLimit so the sweep stops on it and the loop routes to a different
// provider. A COLD backend (live below the floor, e.g. a burst just killed every worker)
// is lowered to the floor and kept SPAWN_OK, so one probe still runs to witness whether
// the overload cleared rather than deadlocking at a zero cap.
//
// The fold is a no-op when the preflight ALREADY refused for a higher-precedence reason
// (host / seat / at-cap / account / gate): the backend is then already not growing, so
// the rate burst is not the sole binding term and the existing verdict stands. It is also
// a no-op below the arming threshold and when the floor meets/exceeds the existing cap
// (the term never manufactures capacity).
func ApplyRateLimitBackpressure(res PreflightResult, r RateLimitCheck) PreflightResult {
	// Bottom-up backpressure on a SAFE-to-spawn preflight only.
	if !res.OK {
		return res
	}
	if !r.pressured() || res.Live >= res.Cap {
		return res
	}
	// Hold at max(live, floor). res.Live < res.Cap holds here (a SPAWN_OK preflight has
	// headroom), so a floor at or above the cap means the term cannot bind -- it never
	// RAISES the cap, so leave the preflight untouched.
	hold := res.Live
	if floor := r.floor(); hold < floor {
		hold = floor
	}
	if hold >= res.Cap {
		return res
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = "rate"
	if res.Headroom > 0 {
		// Cold backend: the term lowered the cap to the floor (throttling growth) but
		// left a minimal cold-start probe, so the verdict stays SPAWN_OK.
		return res
	}
	// Warm backend (live at/above the floor): no headroom above the hold -- freeze and
	// refuse with the closed RATE_LIMIT_BACKOFF token so the sweep terminates on it.
	res.OK = false
	res.Verdict = PreflightRefuseRateLimit
	res.Reason = rateLimitBackoffReason(r, res.Live)
	return res
}

// rateLimitBackoffReason names the closed RATE_LIMIT_BACKOFF refusal token and cites the
// measured burst (count within the window) plus the load-bearing disambiguation, so a
// reader -- and `dos man wedge <TOKEN> --explain` -- can bind both the refusal class and its evidence.
func rateLimitBackoffReason(r RateLimitCheck, live int) string {
	return fmt.Sprintf("%s: %d recent concurrency rate-limit worker exit(s) (reason=rate_limit, a transient 429/529 overload) within %s on this backend -- holding it at %d live worker(s); admit no new concurrent load onto a throttled seat until the burst ages out. Weekly/usage caps, model caps, and login walls are excluded (they need a seat reset / model downgrade / re-login, not concurrency backoff) -- route new work to a different provider/backend instead.",
		RateLimitBackoff, r.Recent, r.Window, live)
}
