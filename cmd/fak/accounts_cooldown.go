package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/guardrotate"
)

// accounts_cooldown.go is the WRITE + resolve side of the account usage-limit
// cooldown gate (the pure store lives in internal/accounts/cooldown.go). When a
// launch bounces off a usage/weekly cap or a transient 429, the seat's upstream
// account is recorded with a reset time in the fleet-shared cooldown store; the
// login overlay (LoginReportAt) then drops every seat sharing that account from
// the servable pool until the window elapses — so the dispatcher stops handing
// the same wall to fresh workers.

// defaultCooldownStorePath resolves the fleet-shared cooldown file. It mirrors
// defaultFleetRegistryDir's FLEET_STATE_DIR-first chain, but lands the file at the
// ROOT of the state dir (parallel to registry/), because a cooldown is account
// state every checkout and watchdog must share, not registry state. A checkout
// with no fleet state dir falls back to the repo's tools/_registry so a solo dev
// box still persists cooldowns between launches.
func defaultCooldownStorePath() string {
	if v := strings.TrimSpace(os.Getenv("FLEET_STATE_DIR")); v != "" {
		return filepath.Join(v, "account-cooldown.json")
	}
	if v := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); v != "" {
		candidate := filepath.Join(v, "Fleet")
		if fi, err := os.Stat(candidate); err == nil && fi.IsDir() {
			return filepath.Join(candidate, "account-cooldown.json")
		}
	}
	if runtime.GOOS == "windows" {
		candidate := filepath.Join(os.TempDir(), "Fleet")
		if fi, err := os.Stat(candidate); err == nil && fi.IsDir() {
			return filepath.Join(candidate, "account-cooldown.json")
		}
	}
	cwd, _ := os.Getwd()
	return filepath.Join(findRepoRoot(cwd), "tools", "_registry", "account-cooldown.json")
}

// loadCooldownStoreFailOpen loads the fleet-shared cooldown store for a cooldown-aware
// resolution (Registry.ServeAt) and FAILS OPEN: an absent or unreadable store yields nil —
// the cooldown-blind fold — so bad cooldown state can never block a resolve, launch, or
// dispatch preflight (#4675). Callers that must EXPLAIN unreadability (the status and
// cooldown verbs) load the store themselves and surface the error instead.
func loadCooldownStoreFailOpen() *accounts.CooldownStore {
	store, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		return nil
	}
	return store
}

// accountsCooldown is the `fak accounts cooldown` surface: with no --clear, it
// lists the accounts currently within an active cooldown window (the seats the
// login overlay is dropping from the pool) and when each returns; with --clear
// <account> it removes that account's cooldown so its seats re-enter the pool at
// once. Reads/writes the same fleet-shared store the launcher records into.
func accountsCooldown(stdout, stderr io.Writer, clear string, asJSON bool) int {
	path := defaultCooldownStorePath()
	store, err := accounts.LoadCooldownStore(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts cooldown: cooldown store unreadable (%s): %v\n", path, err)
		return 1
	}
	now := time.Now()

	if clear != "" {
		removed := store.Clear(clear)
		if err := store.Save(); err != nil {
			fmt.Fprintf(stderr, "fak accounts cooldown: cooldown store unwritable (%s): %v\n", path, err)
			return 1
		}
		if asJSON {
			json.NewEncoder(stdout).Encode(map[string]any{"cleared": removed, "account": clear})
			return 0
		}
		if removed {
			fmt.Fprintf(stdout, "cleared cooldown for %q — its seats re-enter the servable pool\n", clear)
		} else {
			fmt.Fprintf(stdout, "no active cooldown for %q (nothing to clear)\n", clear)
		}
		return 0
	}

	store.Prune(now)
	active := store.Active(now)
	if asJSON {
		json.NewEncoder(stdout).Encode(map[string]any{
			"schema":  accounts.CooldownStoreSchema,
			"path":    path,
			"now":     now.UTC().Format(time.RFC3339),
			"entries": active,
		})
		return 0
	}
	if len(active) == 0 {
		fmt.Fprintf(stdout, "no accounts are cooled down (store: %s)\n", path)
		return 0
	}
	fmt.Fprintf(stdout, "cooled accounts (%d) — dropped from the servable pool until reset:\n", len(active))
	for _, e := range active {
		remaining := e.ResetAt.Sub(now).Round(time.Second)
		reason := e.Reason
		if reason != "" {
			reason = " — " + reason
		}
		fmt.Fprintf(stdout, "  %-28s  %-11s  resets %s (in %s)%s\n",
			e.Account, e.Kind, e.ResetAt.UTC().Format(time.RFC3339), remaining, reason)
	}
	return 0
}

// launchKindToCooldownKind maps the launch-layer classification to the store's
// cooldown vocabulary. Only usage- and rate-limit kinds cool an account; an
// unknown-model refusal or an available launch never does (they are not
// account-scoped walls).
func launchKindToCooldownKind(k launchModelUnavailKind) (accounts.CooldownKind, bool) {
	switch k {
	case launchModelUsageLimit:
		return accounts.CooldownUsageLimit, true
	case launchModelRateLimit:
		return accounts.CooldownRateLimit, true
	default:
		return "", false
	}
}

// recordLaunchCooldown persists a cooldown for account when the launch stderr is a
// usage/rate limit. It is best-effort and fail-open: a store it cannot read or
// write is logged to stderr and skipped, never fatal to the launch path. Returns
// the recorded entry and true when a cooldown was written. now is injected so the
// caller (and tests) control the clock.
func recordLaunchCooldown(stderr io.Writer, account, launchStderr string, kind launchModelUnavailKind, now time.Time) (accounts.CooldownEntry, bool) {
	ck, ok := launchKindToCooldownKind(kind)
	if !ok || strings.TrimSpace(account) == "" {
		return accounts.CooldownEntry{}, false
	}
	path := defaultCooldownStorePath()
	store, err := accounts.LoadCooldownStore(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: cooldown store unreadable (%s): %v — not gating this account\n", path, err)
		return accounts.CooldownEntry{}, false
	}
	reset := accounts.ResolveCooldownReset(launchStderr, now)
	reason := cooldownReasonFromStderr(launchStderr)
	entry := store.Cool(account, ck, reason, now, reset)
	if err := store.Save(); err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: cooldown store unwritable (%s): %v — not gating this account\n", path, err)
		return accounts.CooldownEntry{}, false
	}
	fmt.Fprintf(stderr, "fak accounts launch: cooled account %q (%s) until %s — it drops from the servable pool until then\n",
		account, ck, entry.ResetAt.Format(time.RFC3339))
	return entry, true
}

// recordRehomeCooldown persists a usage-limit cooldown for an account that a LIVE
// mid-session 429 account-cap rehome just walled — closing the gap where the guard's
// in-process failover swapped seats but left the cap invisible to the fleet-shared store
// (so the next `fak guard`/`fak accounts launch` re-selected the just-capped account). The
// live rehome already proved the cap by classifying the 429 as an account cap
// (isAccountCap429), so the caller passes isAccountCap=true; the pure write decision (kind =
// usage-limit, reset parsed from the reason, else default window) lives in
// guardrotate.PersistCooldownForRehome so it is unit-tested independently of this package.
// Keyed on the account bucket, so every seat sharing the cap cools together — exactly like the
// launch-exit writer recordLaunchCooldown, but fired from the running session instead of a
// bounced child's exit.
//
// Best-effort and fail-open: an empty account, a non-cap wall, or an unreadable/unwritable
// store is logged and skipped, never fatal to the session. now is injected so the caller (and
// tests) control the clock. Returns the recorded entry and true when a cooldown was written.
func recordRehomeCooldown(stderr io.Writer, account, reason string, isAccountCap bool, now time.Time) (accounts.CooldownEntry, bool) {
	if !isAccountCap || strings.TrimSpace(account) == "" {
		return accounts.CooldownEntry{}, false
	}
	path := defaultCooldownStorePath()
	store, err := accounts.LoadCooldownStore(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard: cooldown store unreadable (%s): %v — not persisting the 429 wall\n", path, err)
		return accounts.CooldownEntry{}, false
	}
	entry, wrote := guardrotate.PersistCooldownForRehome(store, account, reason, reason, isAccountCap, now)
	if !wrote {
		return accounts.CooldownEntry{}, false
	}
	if err := store.Save(); err != nil {
		fmt.Fprintf(stderr, "fak guard: cooldown store unwritable (%s): %v — not persisting the 429 wall\n", path, err)
		return accounts.CooldownEntry{}, false
	}
	fmt.Fprintf(stderr, "fak guard: account cooled by a live usage cap until %s — it drops from the servable pool (and the next launch avoids it) until then\n",
		entry.ResetAt.Format(time.RFC3339))
	return entry, true
}

// cooldownReasonFromStderr extracts a short, credential-safe reason line from a
// launch stderr for the cooldown record. It returns the first non-empty line that
// carries a limit signal, truncated, so the persisted reason is human-readable
// without dumping a whole traceback (or anything token-like) into fleet state.
func cooldownReasonFromStderr(launchStderr string) string {
	for _, line := range strings.Split(launchStderr, "\n") {
		low := strings.ToLower(line)
		hit := false
		for _, sig := range launchModelUsageLimitSignals {
			if strings.Contains(low, sig) {
				hit = true
				break
			}
		}
		if !hit {
			for _, sig := range launchModelRateLimitSignals {
				if strings.Contains(low, sig) {
					hit = true
					break
				}
			}
		}
		if !hit {
			continue
		}
		trimmed := strings.TrimSpace(line)
		const max = 160
		if len(trimmed) > max {
			trimmed = trimmed[:max] + "…"
		}
		return trimmed
	}
	return ""
}
