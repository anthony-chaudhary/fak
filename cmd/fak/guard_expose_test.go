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

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
