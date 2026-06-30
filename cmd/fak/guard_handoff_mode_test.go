package main

import "testing"

// TestGuardTaskHandoffEffectiveMode pins the attended-interactive default for the Stop-hook
// task-handoff gate: the per-stop handoff demand must auto-OFF for an interactive agent the
// operator did not explicitly gate, so a plain `fak guard -- claude` never spams the TUI or
// blocks a clean hand-back — while a headless `-p` fleet worker keeps the enforce default, and an
// explicit --task-handoff value always wins.
func TestGuardTaskHandoffEffectiveMode(t *testing.T) {
	cases := []struct {
		name             string
		configured       string
		explicitlySet    bool
		childInteractive bool
		want             string
	}{
		// The headline fix: the enforce default on an attended interactive child auto-OFFs.
		{"default interactive -> off", guardPreCompactModeEnforce, false, true, guardPreCompactModeOff},
		// Headless fleet worker (`claude -p …`): childInteractive=false keeps the configured gate.
		{"default headless -> keep enforce", guardPreCompactModeEnforce, false, false, guardPreCompactModeEnforce},
		// An explicit --task-handoff is a knowing opt-in: honor it even on an interactive child.
		{"explicit enforce interactive -> keep", guardPreCompactModeEnforce, true, true, guardPreCompactModeEnforce},
		{"explicit shadow interactive -> keep", guardPreCompactModeShadow, true, true, guardPreCompactModeShadow},
		// Explicit off, headless: unchanged (off stays off).
		{"explicit off headless -> off", guardPreCompactModeOff, true, false, guardPreCompactModeOff},
		// Defensive: an explicit off on an interactive child is still off (no accidental re-arm).
		{"explicit off interactive -> off", guardPreCompactModeOff, true, true, guardPreCompactModeOff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardTaskHandoffEffectiveMode(tc.configured, tc.explicitlySet, tc.childInteractive)
			if got != tc.want {
				t.Errorf("guardTaskHandoffEffectiveMode(%q, set=%v, child=%v) = %q, want %q",
					tc.configured, tc.explicitlySet, tc.childInteractive, got, tc.want)
			}
		})
	}
}
