package main

import (
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

// compressActivates reports whether the --compress flag should turn the native
// context-compressor on for this guard process. The flag only fills an UNSET
// FAK_COMPRESSOR, so an explicit env value (including `noop` to opt out) always
// wins — the same flag-defers-to-explicit-env rule as --landlock-hooks.
func compressActivates(flag bool, env string) bool {
	return flag && strings.TrimSpace(env) == ""
}

// guardHeadlessExposeTools is the curated in-kernel fak_* allowlist a single-issue dispatch worker
// actually uses — the SessionStart affordance set (pull ranked work, gate/execute a tool call,
// durable memory) plus fak_tools_search so every PRUNED tool stays reachable on demand (the search
// view sees the full exposed surface). It trims the ~9.9k-token full-registry schema floor a
// headless worker otherwise pays on every turn (#3607). Each name must be a real registered tool —
// compileToolExposeAllow fails loud on a zero-match glob, so a typo reds the guard at startup rather
// than silently hiding the surface (pinned by TestGuardHeadlessExposeProfileNamesAreReal).
var guardHeadlessExposeTools = []string{
	"fak_index_work", "fak_admit", "fak_adjudicate", "fak_memory_run", "fak_tools_search",
}

// resolveGuardExposeTools maps the --expose-profile value (with the FAK_GUARD_EXPOSE_PROFILE env
// taking precedence as the fleet opt-out) to a gateway ExposeTools allowlist. ONLY "headless" prunes;
// every other value — "", "full", "off", anything unrecognized — returns nil, i.e. the full registry,
// so the interactive default and an operator opt-out are byte-for-byte the pre-#3607 surface.
func resolveGuardExposeTools(flagValue string) []string {
	if strings.EqualFold(effectiveGuardExposeProfile(flagValue), "headless") {
		return append([]string(nil), guardHeadlessExposeTools...)
	}
	return nil
}

// resolveGuardCompactBudget picks the compaction budget for a guard launch. An explicit
// operator --compact-history-budget always wins (explicit==true → the flag value verbatim,
// including 0=off). Otherwise a headless dispatch worker gets the floor-aware
// gateway.HeadlessCompactHistoryBudget in place of the interactive default, because its fixed
// tool+system floor would otherwise sit permanently past the 48k default (see that constant);
// every non-headless launch keeps flagValue (which carries DefaultCompactHistoryBudget when the
// operator left the flag alone). Keyed off effectiveGuardExposeProfile so the FAK_GUARD_EXPOSE_PROFILE
// full/off opt-out restores the interactive budget in the same move it restores the full registry.
func resolveGuardCompactBudget(flagValue int, explicit bool, exposeProfileFlag string) int {
	if explicit {
		return flagValue
	}
	if strings.EqualFold(effectiveGuardExposeProfile(exposeProfileFlag), "headless") {
		return gateway.HeadlessCompactHistoryBudget
	}
	return flagValue
}

// effectiveGuardExposeProfile resolves the operative expose-profile: the --expose-profile
// launch flag with FAK_GUARD_EXPOSE_PROFILE taking precedence (the fleet-wide opt-out kill
// switch). It is the single source of truth for "is this a headless dispatch worker?", so
// the tool-surface prune (resolveGuardExposeTools) and the floor-aware compaction budget
// key off the SAME determination — an operator who flips the env to full/off restores both
// the full registry AND the interactive budget default in one move.
func effectiveGuardExposeProfile(flagValue string) string {
	profile := strings.TrimSpace(flagValue)
	if env := strings.TrimSpace(os.Getenv("FAK_GUARD_EXPOSE_PROFILE")); env != "" {
		profile = env // env overrides the launch flag — the fleet-wide opt-out kill switch
	}
	return profile
}

func recordInteractiveSessionRows(row guardsessions.Row) error {
	if err := guardsessions.Record(resolveSweepRegDir(""), row); err != nil {
		return err
	}
	// Mirror into the machine control-plane registry when SCM is installed. A user
	// Guard can write only because installation grants Authenticated Users append;
	// SCM remains the sole policy/actuator owner.
	if machine := machineGuardRegistryDir(); machine != "" {
		if err := guardsessions.Record(machine, row); err != nil {
			return err
		}
	}
	return nil
}

func guardOwnsInteractiveTerminal() bool {
	// Dispatcher-owned sessions already have a restart owner and must never be
	// double-launched by the terminal-host actuator.
	if strings.TrimSpace(os.Getenv("FAK_DISPATCH_ID")) != "" || strings.TrimSpace(os.Getenv("FAK_HEADLESS")) != "" || strings.EqualFold(strings.TrimSpace(os.Getenv("CI")), "true") {
		return false
	}
	switch runtime.GOOS {
	case "windows":
		return strings.TrimSpace(os.Getenv("WT_SESSION")) != "" || strings.TrimSpace(os.Getenv("WT_WINDOW")) != ""
	case "darwin":
		return strings.TrimSpace(os.Getenv("TERM_PROGRAM")) != ""
	default:
		return strings.TrimSpace(os.Getenv("TERM")) != "" || strings.TrimSpace(os.Getenv("SSH_TTY")) != ""
	}
}
