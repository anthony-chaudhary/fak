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
	"path/filepath"
	"regexp"
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

// Plan is the pure core: given a refreshed registry, a loaded cooldown store, the injected
// headroom signal, the currently-resolved config dir, and now, it decides whether to rotate
// and onto which seat's dir. No I/O — every input is passed in — so it is unit-tested
// directly. It returns (newDir, note, true) only when the current dir maps to an enrolled
// account that is actively cooled AND a live non-walled alternate bucket exists; otherwise
// (cur, zero, false) so the caller keeps the original dir.
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
	// NextRotationDecision only guarantees the chosen seat is NOT known-walled — it will still
	// return a seat whose headroom is UNKNOWN (nil/0: no runtime telemetry). Rotating a live seat
	// onto a bucket we have no evidence has room would just move the problem: we'd swap a
	// provably-cooled account for one that might ALSO be at its cap, we just can't see it. So
	// require the target to be provably OFFERABLE (strictly positive headroom, the (1,2] band from
	// accounts_headroom.go) before rotating. If the best non-walled candidate is only UNKNOWN,
	// fail open and keep the current seat — a known-cooled seat whose window is about to elapse is
	// no worse than an unmeasured gamble, and the launch still proceeds. This is the "rotate only
	// to a seat that actually has room" guarantee.
	if !hasRoom(dec.Seat.Headroom) {
		return cur, Note{}, false
	}
	home, _, err := reg.Serve(dec.Seat.Name)
	if err != nil || strings.TrimSpace(home.Dir) == "" {
		return cur, Note{}, false
	}
	return home.Dir, Note{
		From:     h.Name,
		To:       dec.Seat.Name,
		ResetAt:  entry.ResetAt,
		Headroom: dec.Seat.Headroom,
	}, true
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
	entry := store.Cool(account, accounts.CooldownUsageLimit, cooldownReasonLine(reason), now, parseRehomeReset(resetSource))
	return entry, true
}

// parseRehomeReset pulls an explicit absolute reset time (RFC3339) out of a limit reason, so a
// long weekly cap is held to its real reset rather than the 1h default. Only the unambiguous
// RFC3339 forms are trusted; a looser phrasing yields the zero time and the caller's default
// window stands. Mirrors accounts_cooldown.go's parseCooldownReset so the live-429 and
// launch-exit writers parse a reset identically.
func parseRehomeReset(reason string) time.Time {
	m := rehomeResetRE.FindStringSubmatch(reason)
	if m == nil {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05"} {
		if ts, err := time.Parse(layout, m[1]); err == nil {
			return ts.UTC()
		}
	}
	return time.Time{}
}

var rehomeResetRE = regexp.MustCompile(`(?i)resets?\s+at\s+(\d{4}-\d{2}-\d{2}[tT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)`)

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

// hasRoom reports whether a seat's headroom score proves it is OFFERABLE right now — the
// (1,2] positive band accounts_headroom.go assigns an available, non-throttled bucket. A nil
// score (no runtime signal / stable-by-name mode) or a 0 (the UNKNOWN band) is NOT proof of
// room, so it returns false: we only rotate onto a seat we can positively see has capacity,
// never onto an unmeasured one. A negative score (WALLED) is likewise false, though
// NextRotationDecision already excludes those.
func hasRoom(headroom *float64) bool {
	return headroom != nil && *headroom > 0
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
