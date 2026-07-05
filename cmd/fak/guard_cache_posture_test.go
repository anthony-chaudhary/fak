package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestNormalizeManagedCacheMode(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", "auto", false},
		{"  ", "auto", false},
		{"auto", "auto", false},
		{"AUTO", "auto", false},
		{"on", "on", false},
		{" On ", "on", false},
		{"off", "off", false},
		{"OFF", "off", false},
		{"active", "", true},
		{"1h", "", true},
		{"true", "", true},
	}
	for _, tc := range cases {
		got, err := normalizeManagedCacheMode(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("normalizeManagedCacheMode(%q) = %q, want error", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("normalizeManagedCacheMode(%q) unexpected error: %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("normalizeManagedCacheMode(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestGuardCachePostureArgs(t *testing.T) {
	cases := []struct {
		name      string
		mode      string
		apiKeyEnv string
		want      []string
	}{
		{
			// The unconfigured fleet: auto + no api-key-env emits NOTHING, so the guard argv
			// stays byte-identical and guard keeps its own auto (passive on subscription OAuth).
			name: "auto no key emits nothing",
			mode: "auto",
			want: nil,
		},
		{
			name: "empty mode no key emits nothing",
			mode: "",
			want: nil,
		},
		{
			// An API-key-billed fleet: api-key-env makes guard's AUTO resolve ACTIVE on the
			// Anthropic wire without forcing the mode.
			name:      "api-key-env alone lets auto activate",
			mode:      "auto",
			apiKeyEnv: "ANTHROPIC_API_KEY",
			want:      []string{"--api-key-env", "ANTHROPIC_API_KEY"},
		},
		{
			name: "on forces the upgrade even without a key",
			mode: "on",
			want: []string{"--managed-cache", "on"},
		},
		{
			name: "off is emitted explicitly",
			mode: "off",
			want: []string{"--managed-cache", "off"},
		},
		{
			// Both knobs together, stable order: --api-key-env then --managed-cache.
			name:      "key and mode together, stable order",
			mode:      "on",
			apiKeyEnv: "ANTHROPIC_API_KEY",
			want:      []string{"--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"},
		},
		{
			name:      "whitespace api-key-env is treated as unset",
			mode:      "auto",
			apiKeyEnv: "   ",
			want:      nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardCachePostureArgs(tc.mode, tc.apiKeyEnv)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("guardCachePostureArgs(%q, %q) = %#v, want %#v", tc.mode, tc.apiKeyEnv, got, tc.want)
			}
		})
	}
}

func TestFleetGuardCachePostureArgs(t *testing.T) {
	// auto default (env unset) => no flags. t.Setenv restores the prior value on cleanup.
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")
	got, err := fleetGuardCachePostureArgs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("unconfigured fleet should emit no posture flags, got %#v", got)
	}

	// Configured API-key fleet => --api-key-env + --managed-cache on.
	t.Setenv(fleetManagedCacheEnv, "on")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "ANTHROPIC_API_KEY")
	got, err = fleetGuardCachePostureArgs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("configured fleet = %#v, want %#v", got, want)
	}

	// A malformed mode fails loud rather than defaulting silently.
	t.Setenv(fleetManagedCacheEnv, "bogus")
	if _, err := fleetGuardCachePostureArgs(); err == nil {
		t.Fatalf("malformed FAK_MANAGED_CACHE should return an error")
	}
}

func TestSpliceGuardPostureArgs(t *testing.T) {
	// The dispatch worker's guard argv, as GuardedLaunchCommand builds it.
	base := []string{"fak", "guard", "--provider", "anthropic", "--audit", "a.jsonl", "--", "claude", "-p", "x"}

	// Posture flags land immediately before `--`, so guard parses them and the agent (after `--`)
	// never sees them.
	got := spliceGuardPostureArgs(base, []string{"--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"})
	want := []string{"fak", "guard", "--provider", "anthropic", "--audit", "a.jsonl", "--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on", "--", "claude", "-p", "x"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("spliced = %#v, want %#v", got, want)
	}

	// Empty posture returns the argv unchanged (unconfigured fleet is byte-identical).
	if got := spliceGuardPostureArgs(base, nil); !reflect.DeepEqual(got, base) {
		t.Fatalf("nil posture must be a no-op, got %#v", got)
	}

	// An argv with no `--` (an unguarded command) is returned unchanged even with a posture.
	noSep := []string{"claude", "-p", "x"}
	if got := spliceGuardPostureArgs(noSep, []string{"--managed-cache", "on"}); !reflect.DeepEqual(got, noSep) {
		t.Fatalf("no-`--` argv must be a no-op, got %#v", got)
	}
}

func TestAccountsLaunchManagedCacheWord(t *testing.T) {
	// Each posture names its lever and, for auto, the activation path — so the launch summary
	// is actionable, not just a flag echo.
	if w := accountsLaunchManagedCacheWord("auto", ""); !strings.Contains(w, "passive") || !strings.Contains(w, fleetGuardAPIKeyEnvEnv) {
		t.Errorf("auto/no-key word should mention passive + the api-key-env knob, got %q", w)
	}
	if w := accountsLaunchManagedCacheWord("auto", "ANTHROPIC_API_KEY"); !strings.Contains(w, "ACTIVE") || !strings.Contains(w, "ANTHROPIC_API_KEY") {
		t.Errorf("auto/with-key word should mention ACTIVE + the var, got %q", w)
	}
	if w := accountsLaunchManagedCacheWord("on", ""); !strings.Contains(w, "forces") {
		t.Errorf("on word should say it forces the upgrade, got %q", w)
	}
	if w := accountsLaunchManagedCacheWord("off", ""); !strings.Contains(w, "passive") {
		t.Errorf("off word should say passive, got %q", w)
	}
}
