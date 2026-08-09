// Package guardrotate is the pure decision core for `fak guard`'s cooldown-aware seat
// selection. A bare `fak guard -- claude` (no --rotate) resolves its account purely from
// the environment (CLAUDE_CONFIG_DIR else ~/.claude) and, unlike `fak accounts launch
// --rotate`, never consults the fleet-shared cooldown store — so it would launch against
// an account the launcher just watched bounce off its own weekly/usage cap, burning a turn
// on a walled seat. Plan closes that gap: given a refreshed registry, the loaded cooldown
// store, the headroom signal, and the currently-resolved config dir, it decides whether to
// rotate onto a live alternate bucket and returns that seat's dir.
//
// The core lives in its own package (imported by cmd/fak's thin I/O wrapper) so the
// decision is buildable and unit-testable independently of package main — which mixes many
// unrelated command surfaces — and depends only on internal/accounts.
package guardrotate

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// Note carries the human-facing facts about a rotation Plan resolved, so the caller can
// render one explanation line without re-deriving anything.
type Note struct {
	From     string    // seat name we rotated OFF (the cooled one)
	To       string    // seat name we rotated ONTO
	ResetAt  time.Time // when the cooled account's window elapses (zero if unknown)
	Headroom *float64  // the target seat's headroom score (nil in stable-by-name mode)
}

// Explain renders the one-line operator message for a resolved rotation — the exact line the
// guard prints when it rotates off a cooling seat: the base "cooling down — rotating to" fact,
// then an optional reset instant (when ResetAt is known) and an optional headroom word (when the
// target carried a score, via the shared accounts.HeadroomLabel so the sign→word mapping is not
// re-implemented here). It lives beside Note so the message contract is unit-testable without a
// live registry / cooldown store / env — the I/O wrapper (cmd/fak guardRotateOffCooldown) only
// prints what this returns.
func (n Note) Explain() string {
	msg := fmt.Sprintf("fak guard: account %q is cooling down — rotating to %q", n.From, n.To)
	if !n.ResetAt.IsZero() {
		msg += fmt.Sprintf(" (resets %s)", n.ResetAt.UTC().Format(time.RFC3339))
	}
	if n.Headroom != nil {
		msg += fmt.Sprintf(" (headroom=%s)", accounts.HeadroomLabel(*n.Headroom))
	}
	return msg
}

// Plan is the pure core: given a refreshed registry, a loaded cooldown store, the injected
// headroom signal, the currently-resolved config dir, and now, it decides whether to rotate
// and onto which seat's dir. No I/O — every input is passed in — so it is unit-tested
// directly. It returns (newDir, note, true) when the current dir maps to an enrolled account
// that is actively cooled AND a live non-walled alternate bucket exists — with ONE additional
// suppression: if the only alternate has UNKNOWN headroom (nil/0 per hasRoom) AND the cooled
// seat's own reset is imminent (within WaitResetHorizon; see resetImminent), it keeps the
// current seat, because an unmeasured hop buys nothing when the cool is about to elapse. An
// OFFERABLE (strictly-positive-headroom) alternate always wins regardless of imminence. Every
// other case returns (cur, zero, false) and the caller keeps the original dir.
//
// This imminence tie-break is the ONE thing this launch-time path shares in SPIRIT with
// rehome.Resolve's WAIT_RESET — but NOT in rule (see resetImminent's note on the deliberate
// boundary divergence). They share only the 15m horizon VALUE (WaitResetHorizon), pinned equal
// by a test; the imminence-vs-alternate-quality decision is intentionally different because
// launch rotation only swaps a config dir (free) while a re-home copies a transcript.
//
// FAIL-OPEN is the invariant: every no-decision branch returns the original cur unchanged,
// so cooldown bookkeeping can never block or error a launch.
func Plan(reg accounts.Registry, store *accounts.CooldownStore, hr accounts.RotationHeadroom, cur string, now time.Time) (string, Note, bool) {
	if store == nil || strings.TrimSpace(cur) == "" {
		return cur, Note{}, false
	}
	// Which enrolled seat is the ambient CLAUDE_CONFIG_DIR? Match by normalized dir path;
	// an ambient dir that is no enrolled seat is not ours to rotate.
	h, ok := HomeForDir(reg, cur)
	if !ok {
		return cur, Note{}, false
	}
	key := h.Identity.AccountKey()
	if key == "" {
		return cur, Note{}, false
	}
	entry, cooled := store.CooledDown(key, now)
	if !cooled {
		return cur, Note{}, false
	}
	// Cooled: rotate off this seat's bucket onto the best non-anchor, non-walled bucket.
	// The injected headroom already folds the cooldown overlay, so the cooled bucket
	// (and any other cooled bucket) is walled out of the pool. A no-candidate result (only
	// bucket, all others walled, empty pool) means there is nowhere live to go — fail open.
	dec := reg.NextRotationDecision(h.Name, hr)
	if !dec.OK {
		return cur, Note{}, false
	}
	// NextRotationDecision only guarantees the chosen seat is NOT known-walled — it can still
	// return a seat whose headroom is UNKNOWN (nil/0: no runtime telemetry, the 0 band from
	// accounts_headroom.go). An OFFERABLE alternate (strictly positive headroom) is always worth
	// rotating onto. For an UNKNOWN alternate the call is a gamble: it MIGHT also be at its cap,
	// we just can't see it. But that gamble is still strictly better than staying on a PROVABLY
	// cooled seat — the current seat is a guaranteed wasted turn, while the unmeasured one is at
	// worst no worse and at best live. The ONE case where staying is not worse is when the current
	// seat's cool is about to elapse anyway: within WaitResetHorizon an unmeasured hop buys
	// nothing (the seat is nearly usable again). So reject an UNKNOWN alternate ONLY when the
	// current seat's reset is imminent; otherwise rotate off the walled seat. (This corrects the
	// earlier unconditional gate, which kept a long-capped seat — the july/day26 pile-up shape —
	// over a healthy-but-unmeasured alternate.)
	if !hasRoom(dec.Seat.Headroom) && resetImminent(entry.ResetAt, now) {
		return cur, Note{}, false
	}
	// Resolve the chosen seat through the cooldown-aware fall-forward (#4675). The blind
	// Serve stopped ON the decision seat even when the fleet-shared store had already
	// cooled its account (the headroom row lags the wall), trading one walled seat for
	// another. ServeAt walks past cooled pool-mates; a non-nil cdEntry is its all-cooled
	// terminal — even the fall-forward landed on a walled seat — where rotating buys
	// nothing over staying, so fail open.
	home, _, cdEntry, err := reg.ServeAt(dec.Seat.Name, store, now)
	if err != nil || cdEntry != nil || strings.TrimSpace(home.Dir) == "" {
		return cur, Note{}, false
	}
	note := Note{
		From:     h.Name,
		To:       home.Name,
		ResetAt:  entry.ResetAt,
		Headroom: dec.Seat.Headroom,
	}
	if home.Name != dec.Seat.Name {
		// The fall-forward walked past the decision seat: its headroom score does not
		// describe the seat we actually landed on.
		note.Headroom = nil
	}
	return home.Dir, note, true
}

// PersistCooldownForRehome records a self-recovering usage-limit cooldown for the account a
// LIVE 429 account-cap rehome just walled, so the cap outlives the guard process and the next
// launch (via Plan / the login overlay) avoids the just-capped account instead of re-selecting
// it. It is the durable half of the automatic-on-429 loop: the in-process seat swap already
// hops the running session off the cap, but only an in-memory mark; this makes the wall
// fleet-visible and launch-visible.
//
// It only records when isAccountCap is true — i.e. the wall is a timed 429 usage/session/weekly
// cap that genuinely self-recovers. A 403 org/region/billing wall (isAccountCap false) must NOT
// be recorded: it is not a timed cap, and a default cooldown window would wrongly re-admit a
// durably-blocked org after it elapses. An explicit reset parsed from resetSource wins over the
// kind's default window. account is the bucket key (uuid:…/tok:…). Returns the entry and whether
// it wrote; a false with no write is the correct outcome for both an empty account and a non-cap
// wall. The store is mutated in place; the caller persists it (Save) — kept separate so this
// stays pure over the in-memory store and unit-testable without disk.
func PersistCooldownForRehome(store *accounts.CooldownStore, account, reason, resetSource string, isAccountCap bool, now time.Time) (accounts.CooldownEntry, bool) {
	if store == nil || !isAccountCap || strings.TrimSpace(account) == "" {
		return accounts.CooldownEntry{}, false
	}
	// The reset is resolved by the shared accounts.ResolveCooldownReset so this live-429 writer
	// and the launch-exit writer (cmd/fak recordLaunchCooldown) can never hold one account to two
	// different reset times — an explicit RFC3339 reset wins, else a weekly cap's announced
	// relative wait (announced_wait≈1h7m) is honored (#2610), else an UNANNOUNCED weekly cap
	// takes the weekly floor rather than the rolling cap's 1-hour default (#5890), else the
	// kind's default window.
	entry := store.Cool(account, accounts.CooldownUsageLimit, cooldownReasonLine(reason), now, accounts.ResolveCooldownReset(resetSource, now))
	return entry, true
}

// cooldownReasonLine renders a short, credential-safe reason line for a persisted cooldown from
// an arbitrary reason string (a remedy label like "rehomed_seat", or a limit message). It never
// carries anything token-like: it trims and truncates. An empty reason yields a stable default
// label so a persisted entry always names why it exists.
func cooldownReasonLine(reason string) string {
	r := strings.TrimSpace(reason)
	if r == "" {
		return "live usage cap (429)"
	}
	const max = 160
	if len(r) > max {
		r = r[:max] + "…"
	}
	return r
}

// hasRoom reports whether a seat's headroom score proves it is OFFERABLE right now, reading the
// shared tier contract (accounts.Classify): a TierOfferable score is the (1,2] positive band
// accounts_headroom.go assigns an available, non-throttled bucket. A nil score (no runtime
// signal / stable-by-name mode) is NOT proof of room, so the nil-guard stays HERE — Classify
// takes a concrete value and a nil pointer means "no signal", a distinct state from any tier. A
// TierUnknown (0) or TierWalled (<0) score likewise returns false, though NextRotationDecision
// already excludes walled ones. An OFFERABLE seat is rotated onto unconditionally; an UNKNOWN one
// only when the current seat's cool is not imminent (see resetImminent) — a provably-walled seat
// is worse than an unmeasured one unless it is about to free up anyway.
func hasRoom(headroom *float64) bool {
	return headroom != nil && accounts.Classify(*headroom) == accounts.TierOfferable
}

// WaitResetHorizon is how close a cooled seat's reset must be for Plan to keep it rather than
// rotate onto an UNKNOWN-headroom alternate. Its VALUE mirrors rehome.WaitResetHorizonSeconds
// (both 15m), pinned equal by a cross-package test, so launch-time rotation and resume-time
// re-home agree on what "about to elapse" MEANS. Only the horizon value is shared: the two
// paths deliberately apply it differently (launch lets an OFFERABLE alternate beat an imminent
// cool; rehome's WAIT_RESET fires before any target is even selected) — see Plan's doc and
// resetImminent below.
const WaitResetHorizon = 15 * time.Minute

// resetImminent reports whether a cooled seat's reset is known AND within WaitResetHorizon of
// now. A zero (unknown) reset is NOT imminent — we cannot prove the wall is about to lift, so we
// prefer rotating off it. A reset already at/behind now counts as imminent (the window is
// elapsing now, so the seat is about to be usable again).
//
// Boundary note — this deliberately DIFFERS from rehome.waitResetDecision at the past/zero-reset
// edge, and the two predicates must NOT be unified: rehome requires wait>=0, so an expired reset
// re-homes (pinned by rehome's TestResolveWaitHorizon "expired reset rehomes") because an
// expired-but-still-blocked estimate is stale and copying is safer than waiting on a guess. Here
// the "past = imminent" arm is a never-taken safety default: Plan only calls resetImminent after
// store.CooledDown has already gated on now < ResetAt, and Cool always writes a concrete
// ResetAt, so in production resetAt is always a real future instant. Pure and time-injected.
func resetImminent(resetAt, now time.Time) bool {
	if resetAt.IsZero() {
		return false
	}
	return !resetAt.After(now.Add(WaitResetHorizon))
}

// HomeForDir finds the registry Home whose config dir is the same on-disk path as dir. The
// comparison is normalized (Clean + platform case-fold via NormalizeDir) so an ambient
// CLAUDE_CONFIG_DIR that differs only in trailing slash or separator/case style still
// matches its seat. Returns the first match; a dir that names no enrolled seat reports
// ok=false.
func HomeForDir(reg accounts.Registry, dir string) (accounts.Home, bool) {
	want := NormalizeDir(dir)
	if want == "" {
		return accounts.Home{}, false
	}
	for _, h := range reg.Homes {
		if h.Dir == "" {
			continue
		}
		if NormalizeDir(h.Dir) == want {
			return h, true
		}
	}
	return accounts.Home{}, false
}

// NormalizeDir canonicalizes a config-dir path for equality: absolute where possible,
// cleaned, and lower-cased on the case-insensitive filesystems (Windows) the registry and
// the ambient env var can disagree about. An empty input stays empty.
func NormalizeDir(dir string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return ""
	}
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	dir = filepath.Clean(dir)
	if filepath.Separator == '\\' {
		// Windows paths compare case-insensitively; the registry stores one casing and the
		// ambient env another, but they name the same directory.
		dir = strings.ToLower(dir)
	}
	return dir
}
