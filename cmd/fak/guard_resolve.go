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
// actually uses — the SessionStart affordance set (discover the task-scoped toolbelt, gate/execute
// a tool call, durable memory) plus fak_tools_search so every PRUNED tool stays reachable on demand
// (the search view sees the full exposed surface). It trims the ~9.9k-token full-registry schema
// floor a headless worker otherwise pays on every turn (#3607). Each name must be a real registered
// tool — compileToolExposeAllow fails loud on a zero-match glob, so a typo reds the guard at startup
// rather than silently hiding the surface (pinned by TestGuardHeadlessExposeProfileNamesAreReal).
var guardHeadlessExposeTools = []string{
	"fak_capabilities", "fak_admit", "fak_adjudicate", "fak_memory_run", "fak_tools_search",
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
// including 0=off). Every other `fak guard` launch gets the floor-aware
// gateway.HeadlessCompactHistoryBudget in place of the lean gateway.DefaultCompactHistoryBudget,
// because EVERY guard launch fronts Claude Code and therefore carries its large fixed
// system+tools floor — the 48k default assumes a LEAN prompt that this path never has.
//
// The budget is DECOUPLED from the expose profile (#4888). It used to key off
// effectiveGuardExposeProfile, which inverted the sizing: the headless profile PRUNES the tool
// registry to guardHeadlessExposeTools (a ~9.9k-token-smaller floor) yet took the 96k budget,
// while an interactive session carrying the FULL 76-tool registry (floor_tokens 42292 observed,
// resident 86205 / peak 93010) was left on 48k — a budget BELOW its own immutable floor. A budget
// the floor alone already exceeds has no resting point: with head-anchoring engaged the whole
// message array is compactible, so the cut re-fires every turn, sheds only the incremental
// overflow, and still emits the user-visible `[fak] compacted N earlier turn(s)` stub (10
// context_events over 12 turns observed; avg_turns_between_events 1.2). Sizing the budget is a
// property of the FLOOR the launch carries, not of how many tools it exposes — so the profile no
// longer moves it, and the FAK_GUARD_EXPOSE_PROFILE full/off opt-out now restores only the full
// registry (which makes the floor BIGGER, never smaller). An operator who genuinely wants the lean
// line passes --compact-history-budget explicitly.
func resolveGuardCompactBudget(flagValue int, explicit bool) int {
	if explicit {
		return flagValue
	}
	return gateway.HeadlessCompactHistoryBudget
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
