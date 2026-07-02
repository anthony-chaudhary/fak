package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// guard_park.go — the STALE_CRED pre-spawn park (#2260): when a HEADLESS launch finds the
// pinned Claude subscription credential expired and the #1834 proactive rehydrate window
// (≤30s, guard_child.go) could not catch a rotation, the session used to die inside half a
// minute against a HUMAN-paced recovery — a re-login lands minutes-to-hours later, and the
// fleet self-heals the moment `claude` runs once anywhere on the host. Instead of refusing
// immediately, guard now PARKS: it polls the credential file on a few-minute cadence until
// a live token lands (the launch proceeds) or the park budget exhausts (today's fail-loud
// STALE_CRED exit, now carrying the elapsed park time — a logged give-up, never silent).
// Field witness: nine dispatch sessions refused at spawn across 2026-07-01/02
// (.dispatch-runs/dispatch-docs-20260702-103022.log et al.) while the dispatch loop ticked
// all morning with zero resolved issues (.dispatch-runs/progress.jsonl).

// defaultGuardParkBudget is how long the STALE_CRED park waits for a re-login before giving
// up when FAK_GUARD_PARK_BUDGET does not override it. 24h — the "someone re-auths within a
// working day" budget the fleet actually needs (the observed refusal window spanned a whole
// morning) — deliberately far past the rotation-in-progress windows (10s reactive 401 poll,
// 30s proactive rehydrate): those ride out a refresh ALREADY happening; this rides out a
// human noticing.
const defaultGuardParkBudget = 24 * time.Hour

// maxGuardParkBudget clamps FAK_GUARD_PARK_BUDGET. It equals the default on purpose: the
// knob tunes the park DOWN (or off, at 0) — a fat-fingered value can never wedge a spawn
// past the day the default already grants.
const maxGuardParkBudget = 24 * time.Hour

// guardParkBudget resolves the park ceiling: FAK_GUARD_PARK_BUDGET (any Go duration)
// clamped to [0, maxGuardParkBudget]. 0 disables the park entirely, restoring the
// immediate fail-loud refusal byte-for-byte.
func guardParkBudget() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_GUARD_PARK_BUDGET")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxGuardParkBudget {
				return maxGuardParkBudget
			}
			return d
		}
	}
	return defaultGuardParkBudget
}

// defaultGuardParkPoll is the cadence the park re-reads the credential file at. A re-login
// is human-paced, so minute-scale polling recovers promptly without putting meaningful
// pressure on the disk (the read is a small JSON parse); it also matches the "every few
// minutes" cadence the cap-probe path (internal/agent/retry_limit.go) already uses for
// slow re-checks of a condition that clears out-of-band.
const defaultGuardParkPoll = 2 * time.Minute

// minGuardParkPoll / maxGuardParkPoll clamp FAK_GUARD_PARK_POLL so a typo can neither spin
// the poll into a disk-hammering loop nor stretch it past the hourly re-check any human
// recovery deserves.
const (
	minGuardParkPoll = time.Second
	maxGuardParkPoll = time.Hour
)

// guardParkPoll resolves the poll cadence: FAK_GUARD_PARK_POLL (any Go duration) clamped
// to [minGuardParkPoll, maxGuardParkPoll], defaulting to defaultGuardParkPoll.
func guardParkPoll() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_GUARD_PARK_POLL")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			if d < minGuardParkPoll {
				return minGuardParkPoll
			}
			if d > maxGuardParkPoll {
				return maxGuardParkPoll
			}
			return d
		}
	}
	return defaultGuardParkPoll
}

// guardParkResult reports what the park did, so the caller can tell "did not run"
// (Attempted false — budget 0 or nothing to watch) from "ran and recovered" from "ran the
// full budget and gave up".
type guardParkResult struct {
	Attempted bool          // the park ran (budget > 0 and a credential path to watch)
	Recovered bool          // a live credential landed within the budget
	Elapsed   time.Duration // how long the park waited before its outcome
}

// guardParkCredLive reports whether the credential on disk is live at t, per
// credExpiresAt's contract: ok=false (missing/torn/no token) is not live; a zero expiry
// with ok=true means "no expiry recorded — never-expiring" and IS live (Claude Code's own
// convention for a token that does not rotate); otherwise live means the expiry is still
// ahead of t.
func guardParkCredLive(credPath string, t time.Time) bool {
	expiresAt, ok := credExpiresAt(credPath)
	if !ok {
		return false
	}
	return expiresAt.IsZero() || expiresAt.After(t)
}

// guardParkForRelogin blocks until a live credential lands at credPath or the budget is
// spent, polling every poll. It prints ONE park line up front (naming the path, cadence,
// ceiling, and how to disable) and one outcome line, so an operator tailing the log sees a
// parked session as parked — never a silent hang. now/sleep default to the real clock and
// are injectable so tests never sleep wall-clock time. A budget ≤ 0 or an empty credPath
// returns the zero result without printing (the park is off; the caller keeps today's
// immediate-refusal path).
func guardParkForRelogin(credPath string, budget, poll time.Duration, now func() time.Time, sleep func(time.Duration), stderr io.Writer) guardParkResult {
	if budget <= 0 || strings.TrimSpace(credPath) == "" {
		return guardParkResult{}
	}
	if now == nil {
		now = time.Now
	}
	if sleep == nil {
		sleep = time.Sleep
	}
	start := now()
	deadline := start.Add(budget)
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: parked — STALE_CRED: waiting for a re-login to land at %s (poll every %s, ceiling %s). Run `claude` once to refresh, `claude setup-token` for a long-lived token, or export CLAUDE_CODE_OAUTH_TOKEN; FAK_GUARD_PARK_BUDGET=0 disables this park.\n", credPath, poll, budget)
	}
	for {
		if !now().Before(deadline) {
			elapsed := now().Sub(start)
			if stderr != nil {
				fmt.Fprintf(stderr, "fak guard: park gave up — no re-login landed within %s (parked %s); failing loud with STALE_CRED.\n", budget, elapsed.Round(time.Second))
			}
			return guardParkResult{Attempted: true, Elapsed: elapsed}
		}
		sleep(poll)
		if guardParkCredLive(credPath, now()) {
			elapsed := now().Sub(start)
			if stderr != nil {
				fmt.Fprintf(stderr, "fak guard: re-login landed after %s — resuming the launch.\n", elapsed.Round(time.Second))
			}
			return guardParkResult{Attempted: true, Recovered: true, Elapsed: elapsed}
		}
	}
}
