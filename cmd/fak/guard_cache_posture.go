package main

import (
	"fmt"
	"os"
	"strings"
)

// The fleet-wide managed-cache posture knobs (epic #1844 C6). Every fak launcher that fronts
// an agent with `fak guard` — `fak accounts launch`, `fak codex`, and the dispatch worker —
// builds its own guard argv, and until now none of them carried a --managed-cache posture.
// So a launched/dispatched session inherited guard's OWN auto default, which stays PASSIVE on
// a subscription-OAuth seat by construction: resolveGuardManagedCache's never-speculate rule
// refuses to manage a wire whose billing it cannot see. The consequence is that a fleet billed
// on an API key could not reach the ACTIVE 1h-TTL cache upgrade without hand-editing launch
// code. These two knobs are that missing surface, shared here so all three launchers name the
// posture identically instead of each growing its own flag.
//
// FAK_MANAGED_CACHE selects the mode (auto|on|off). auto is guard's own default, so it emits
// NOTHING and the guard argv stays byte-identical to before this knob existed; only a
// non-default on|off is spliced in. FAK_GUARD_API_KEY_ENV names the --api-key-env var: on the
// Anthropic wire that flag is the explicit opt-IN to API billing which lets guard's AUTO
// resolve ACTIVE (guard bills the key, not the seat's subscription OAuth). Empty keeps the
// subscription-OAuth default, so an unconfigured fleet's posture is unchanged (passive).
const (
	fleetManagedCacheEnv   = "FAK_MANAGED_CACHE"
	fleetGuardAPIKeyEnvEnv = "FAK_GUARD_API_KEY_ENV"
)

// normalizeManagedCacheMode validates a managed-cache token against guard's own closed set
// (auto|on|off). Empty/whitespace resolves to auto. An unknown token is an error so a fleet
// misconfiguration fails loud at the launcher instead of silently launching the wrong
// posture — the same fail-loud contract resolveGuardManagedCache enforces inside guard.
func normalizeManagedCacheMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", guardManagedCacheAuto:
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
// then --managed-cache — and an auto/empty mode emits NO --managed-cache, so an unconfigured
// launcher's argv is byte-identical to before this knob existed and guard keeps its own auto.
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
