package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestResolveGuardExposeTools pins the #3607 profile → ExposeTools resolution and the
// FAK_GUARD_EXPOSE_PROFILE env opt-out precedence.
func TestResolveGuardExposeTools(t *testing.T) {
	t.Setenv("FAK_GUARD_EXPOSE_PROFILE", "") // isolate from an operator env

	// The curated headless profile resolves to exactly the allowlist, INCLUDING fak_tools_search
	// (crit-3: the pruned tools stay reachable through it).
	got := resolveGuardExposeTools("headless")
	if !reflect.DeepEqual(got, guardHeadlessExposeTools) {
		t.Fatalf("headless profile = %v, want %v", got, guardHeadlessExposeTools)
	}
	if !containsStr(got, "fak_tools_search") {
		t.Fatalf("fak_tools_search must stay exposed so pruned tools page in: %v", got)
	}

	// Every non-headless value keeps the full registry (nil = no allowlist) — the interactive
	// default and the opt-out are byte-for-byte the pre-#3607 surface.
	for _, v := range []string{"", "full", "off", "FULL", "anything"} {
		if r := resolveGuardExposeTools(v); r != nil {
			t.Fatalf("profile %q must resolve to the full registry (nil), got %v", v, r)
		}
	}

	// The env OVERRIDES the flag (the fleet opt-out kill switch): a dispatch launch flag of
	// "headless" is overridden to full by FAK_GUARD_EXPOSE_PROFILE=full.
	t.Setenv("FAK_GUARD_EXPOSE_PROFILE", "full")
	if r := resolveGuardExposeTools("headless"); r != nil {
		t.Fatalf("env=full must override the headless flag to the full registry, got %v", r)
	}
	// And the env can force headless even when the flag is empty.
	t.Setenv("FAK_GUARD_EXPOSE_PROFILE", "headless")
	if r := resolveGuardExposeTools(""); !reflect.DeepEqual(r, guardHeadlessExposeTools) {
		t.Fatalf("env=headless must force the curated set, got %v", r)
	}
}

// TestGuardHeadlessExposeProfileNamesAreReal proves every name in the curated allowlist matches a
// real registered tool: gateway.New runs compileToolExposeAllow, which fails LOUD on a zero-match
// glob. A green build here means a typo would red the guard at startup (not silently hide the
// surface); the bogus-name arm proves that guard is actually load-bearing (the green arm isn't
// vacuous).
func TestGuardHeadlessExposeProfileNamesAreReal(t *testing.T) {
	base := gateway.Config{
		EngineID:     "inkernel",
		Model:        "expose-profile-test",
		Invalidation: "global",
		Logf:         func(string, ...any) {},
	}

	cfg := base
	cfg.ExposeTools = guardHeadlessExposeTools
	if _, err := gateway.New(cfg); err != nil {
		t.Fatalf("curated headless allowlist has an unknown tool name (compileToolExposeAllow rejected it): %v", err)
	}

	bogus := base
	bogus.ExposeTools = []string{"fak_this_tool_does_not_exist"}
	if _, err := gateway.New(bogus); err == nil {
		t.Fatalf("expected gateway.New to FAIL LOUD on an --expose pattern matching no tool; got nil (the name-validity guard is not load-bearing)")
	}
}

// TestResolveGuardCompactBudget pins the floor-aware budget for EVERY guard launch: the
// resolved default is gateway.HeadlessCompactHistoryBudget, so a launch's fixed Claude-Code
// tool+system floor never sits permanently past its own budget. Only an explicit operator
// --compact-history-budget moves it.
func TestResolveGuardCompactBudget(t *testing.T) {
	t.Setenv("FAK_GUARD_EXPOSE_PROFILE", "") // isolate from an operator env

	// Operator left the flag alone (explicit=false): floor-aware default.
	if got := resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, false); got != gateway.HeadlessCompactHistoryBudget {
		t.Fatalf("guard default = %d, want %d", got, gateway.HeadlessCompactHistoryBudget)
	}

	// An explicit --compact-history-budget ALWAYS wins — including an explicit 0 (compaction
	// OFF), which the floor-aware default must never resurrect.
	if got := resolveGuardCompactBudget(120000, true); got != 120000 {
		t.Fatalf("explicit budget must win: got %d, want 120000", got)
	}
	if got := resolveGuardCompactBudget(0, true); got != 0 {
		t.Fatalf("explicit 0 (off) must win: got %d, want 0", got)
	}
}

// TestInteractiveGuardBudgetClearsItsOwnFloor is the #4888 regression: the INTERACTIVE
// (non-headless) guard path must not be left on a budget its own immutable floor already
// exceeds. Before the fix, resolveGuardCompactBudget keyed off the expose profile, so every
// non-headless launch kept the lean 48000 default while carrying the full 76-tool Claude-Code
// registry — a budget BELOW the observed 42292-token floor. With head-anchoring engaged the
// whole message array is compactible, so that budget has no under_budget resting point: the cut
// re-fires every turn, sheds only the incremental overflow, and still emits the user-visible
// `[fak] compacted N earlier turn(s)` stub (10 context_events over 12 turns observed).
//
// The failure class is "resolved budget <= observed floor" — assert against the real numbers the
// issue captured, not against a constant, so this fails loudly if the budget is ever re-coupled
// to the tool surface. Under the old profile-keyed logic every case below resolves to 48000 and
// fails; a peak-clearing budget passes.
func TestInteractiveGuardBudgetClearsItsOwnFloor(t *testing.T) {
	// Live interactive `fak guard -- claude` trace from #4888 (fak_context_value, trace `guard`).
	const (
		observedFloorTokens    = 42292 // system+tools, immutable (38 builtin + 38 MCP schemas)
		observedPeakResident   = 93010 // floor + the conversation window it actually held
		observedResidentTokens = 86205
	)

	// The profile no longer moves the budget: an interactive launch, an explicit full/off
	// opt-out, and an unrecognized profile all resolve to the same floor-aware line. Note
	// full/off make the floor BIGGER (they restore the full registry), so restoring the LEAN
	// budget there — the pre-#4888 behavior — was exactly backwards.
	t.Setenv("FAK_GUARD_EXPOSE_PROFILE", "")
	base := resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, false)
	for _, env := range []string{"", "full", "off", "anything"} {
		t.Setenv("FAK_GUARD_EXPOSE_PROFILE", env)
		got := resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, false)
		if got != base {
			t.Errorf("FAK_GUARD_EXPOSE_PROFILE=%q moved the compaction budget to %d (want %d): "+
				"budget must key off the floor the launch carries, not the tool surface", env, got, base)
		}
		if got <= observedFloorTokens {
			t.Errorf("FAK_GUARD_EXPOSE_PROFILE=%q resolved budget %d <= observed floor %d: the floor "+
				"alone exceeds the budget, so the session is structurally past-compact from turn one",
				env, got, observedFloorTokens)
		}
		if got < observedPeakResident {
			t.Errorf("FAK_GUARD_EXPOSE_PROFILE=%q resolved budget %d < observed peak resident %d: the cut "+
				"has no resting point, so it re-fires every turn and emits a `[fak] compacted` stub each time "+
				"(observed resident %d)", env, got, observedPeakResident, observedResidentTokens)
		}
	}

	// The escape hatch is still real: an operator who genuinely wants the lean line asks for it.
	if got := resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, true); got != gateway.DefaultCompactHistoryBudget {
		t.Fatalf("an explicit lean budget must still win: got %d, want %d", got, gateway.DefaultCompactHistoryBudget)
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
