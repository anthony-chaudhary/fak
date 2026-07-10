package main

import (
	"fmt"
	"os"
	"strings"
)

// The fleet-wide managed-cache posture knobs (epic #1844 C6). Every fak launcher that fronts
// an agent with `fak guard` — `fak accounts launch`, `fak codex`, the dispatch worker, and the
// resume watchdog — builds its own guard argv. The shared posture policy lives here so all of
// them name the managed-cache mode identically instead of each growing its own flag.
//
// DEFAULT (operator policy, 2026-07-10): best-effort managed cache everywhere. An UNSET
// FAK_MANAGED_CACHE now resolves to `on`, so an unconfigured launcher splices --managed-cache
// on and forces the stable-prefix 1h-TTL upgrade regardless of billing. Previously an unset
// knob deferred to guard's OWN auto default, which stays PASSIVE on a subscription-OAuth seat
// (resolveGuardManagedCache's never-speculate rule refuses to manage a wire whose billing it
// cannot see) — that passive-by-default posture is what this flip supersedes.
//
// FAK_MANAGED_CACHE selects the mode (auto|on|off): unset|on force the 1h-TTL upgrade; an
// EXPLICIT `auto` restores guard's billing-gated auto (emits nothing, guard decides); `off` is
// the express opt-out for a seat where on self-blocks. FAK_GUARD_API_KEY_ENV names the
// --api-key-env var: on the Anthropic wire that flag is the explicit opt-IN to API billing which
// lets guard's AUTO resolve ACTIVE (guard bills the key, not the seat's subscription OAuth);
// with the new on-default it is no longer required to reach the ACTIVE 1h-TTL upgrade.
const (
	fleetManagedCacheEnv   = "FAK_MANAGED_CACHE"
	fleetGuardAPIKeyEnvEnv = "FAK_GUARD_API_KEY_ENV"
)

// normalizeManagedCacheMode validates a managed-cache token against guard's own closed set
// (auto|on|off). Empty/whitespace resolves to ON — the operator policy is best-effort managed
// cache everywhere, so an unconfigured launcher now fronts guard with --managed-cache on rather
// than deferring to guard's billing-gated auto (which stays passive on a subscription-OAuth
// seat). An EXPLICIT "auto" still selects that billing-gated auto, and "off" is the express
// opt-out for a seat where on self-blocks. An unknown token is an error so a fleet
// misconfiguration fails loud at the launcher instead of silently launching the wrong
// posture — the same fail-loud contract resolveGuardManagedCache enforces inside guard.
//
// Safety: on forces the stable-prefix 1h-TTL upgrade regardless of billing. The upgrade's
// old 0-turn HTTP 400 (a 1h request that omitted the extended-cache-ttl beta) is fixed at
// internal/gateway/messages_transform.go (the ttl1hUpgraded beta union), so a current binary
// sends a well-formed request; a stale fleet binary predating that fix would still 400 under
// this default (keep the hot-swapped copies current).
func normalizeManagedCacheMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return guardManagedCacheOn, nil
	case guardManagedCacheAuto:
		return guardManagedCacheAuto, nil
	case guardManagedCacheOn:
		return guardManagedCacheOn, nil
	case guardManagedCacheOff:
		return guardManagedCacheOff, nil
	default:
		return "", fmt.Errorf("%s=%q: unknown managed-cache mode (auto|on|off)", fleetManagedCacheEnv, raw)
	}
}

// guardCachePostureArgs shapes the guard flags a launcher splices into its `fak guard ... --`
// argv from an already-resolved posture. mode is a normalized managed-cache mode; apiKeyEnv is
// the --api-key-env var (empty => omit). It is pure (no I/O) so every launcher's wiring is
// unit-tested without touching the process environment. The order is stable — --api-key-env
// then --managed-cache. Only an EXPLICIT `auto` mode emits NO --managed-cache (guard keeps its
// own billing-gated auto); on|off are spliced through. Because normalizeManagedCacheMode now
// maps unset → on, an unconfigured launcher resolves to on and DOES carry --managed-cache on.
func guardCachePostureArgs(mode, apiKeyEnv string) []string {
	var args []string
	if v := strings.TrimSpace(apiKeyEnv); v != "" {
		args = append(args, "--api-key-env", v)
	}
	if m := strings.TrimSpace(mode); m != "" && m != guardManagedCacheAuto {
		args = append(args, "--managed-cache", m)
	}
	return args
}

// accountsLaunchManagedCacheWord renders the one-line posture note for `fak accounts launch`'s
// startup summary, echoing the operator's INTENT (the guard child prints the RESOLVED
// ACTIVE/passive line itself once the upstream wire is known). It names the same lever and the
// activation path so the summary is actionable, not just a flag echo.
func accountsLaunchManagedCacheWord(mode, apiKeyEnv string) string {
	switch mode {
	case guardManagedCacheOff:
		return "off (--managed-cache off; guard stays passive)"
	case guardManagedCacheOn:
		return "on (forces the stable-prefix 1h-TTL cache upgrade regardless of billing)"
	default: // auto
		if strings.TrimSpace(apiKeyEnv) != "" {
			return fmt.Sprintf("auto (ACTIVE when --api-key-env %s bills a key on the Anthropic wire; subscription OAuth stays passive)", strings.TrimSpace(apiKeyEnv))
		}
		return fmt.Sprintf("auto (passive on a subscription-OAuth seat; set $%s or --managed-cache on to activate)", fleetGuardAPIKeyEnvEnv)
	}
}

// fleetGuardCachePostureArgs resolves the managed-cache posture from the fleet env knobs and
// shapes the guard flags. It is the single place the two env vars are read, so every launcher
// that fronts guard shares exactly one posture policy. A malformed FAK_MANAGED_CACHE fails
// loud (returns the error) rather than defaulting silently — the caller decides whether to
// abort (an interactive front-end) or warn-and-continue (a headless worker).
func fleetGuardCachePostureArgs() ([]string, error) {
	mode, err := normalizeManagedCacheMode(os.Getenv(fleetManagedCacheEnv))
	if err != nil {
		return nil, err
	}
	return guardCachePostureArgs(mode, strings.TrimSpace(os.Getenv(fleetGuardAPIKeyEnvEnv))), nil
}
