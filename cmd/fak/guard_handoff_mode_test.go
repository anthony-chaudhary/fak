package main

import "testing"

// TestGuardTaskHandoffEffectiveMode pins the attended-interactive default for the Stop-hook
// task-handoff gate: the per-stop handoff demand must auto-OFF for an interactive agent the
// operator did not explicitly gate, so a plain `fak guard -- claude` never spams the TUI or
// blocks a clean hand-back. A local one-shot `fak guard --probe -- claude -p …` also auto-OFFs
// the gate so the smoke proves the guarded wire instead of fleet continuity. A plain headless `-p`
// fleet worker keeps the enforce default, and an explicit --task-handoff value always wins.
func TestGuardTaskHandoffEffectiveMode(t *testing.T) {
	cases := []struct {
		name             string
		configured       string
		explicitlySet    bool
		childInteractive bool
		probeMode        bool
		want             string
	}{
		// The headline fix: the enforce default on an attended interactive child auto-OFFs.
		{"default interactive -> off", guardPreCompactModeEnforce, false, true, false, guardPreCompactModeOff},
		// Headless fleet worker (`claude -p …`): childInteractive=false keeps the configured gate.
		{"default headless -> keep enforce", guardPreCompactModeEnforce, false, false, false, guardPreCompactModeEnforce},
		// Local one-shot smoke: explicit --probe disables only the implicit handoff gate.
		{"probe headless -> off", guardPreCompactModeEnforce, false, false, true, guardPreCompactModeOff},
		// An explicit --task-handoff is a knowing opt-in: honor it even on an interactive child.
		{"explicit enforce interactive -> keep", guardPreCompactModeEnforce, true, true, false, guardPreCompactModeEnforce},
		{"explicit shadow interactive -> keep", guardPreCompactModeShadow, true, true, false, guardPreCompactModeShadow},
		{"explicit enforce probe -> keep", guardPreCompactModeEnforce, true, false, true, guardPreCompactModeEnforce},
		// Explicit off, headless: unchanged (off stays off).
		{"explicit off headless -> off", guardPreCompactModeOff, true, false, false, guardPreCompactModeOff},
		// Defensive: an explicit off on an interactive child is still off (no accidental re-arm).
		{"explicit off interactive -> off", guardPreCompactModeOff, true, true, false, guardPreCompactModeOff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardTaskHandoffEffectiveMode(tc.configured, tc.explicitlySet, tc.childInteractive, tc.probeMode)
			if got != tc.want {
				t.Errorf("guardTaskHandoffEffectiveMode(%q, set=%v, child=%v, probe=%v) = %q, want %q",
					tc.configured, tc.explicitlySet, tc.childInteractive, tc.probeMode, got, tc.want)
			}
		})
	}
}
