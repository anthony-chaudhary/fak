package policy

import (
	"strings"
	"testing"
)

// The in-flight-repair knob is opt-in, session-scoped, and loud about a typo. These
// tests pin all three: the default is OFF, only the sanctioned mode turns it on, and
// an unrecognized value REFUSES rather than silently leaving the guard held.

func TestAutoRepairSidestepDefaultsOff(t *testing.T) {
	// Unset and every spelling of "no" must leave the preview-confirm hold in place.
	// This is the property every already-deployed guard depends on: adding the knob
	// changed nothing for an operator who never sets it.
	for _, v := range []string{"", "   ", "off", "OFF", "0", "false", "none"} {
		got, err := autoRepairSidestepFromEnv(v)
		if err != nil {
			t.Fatalf("autoRepairSidestepFromEnv(%q) errored: %v", v, err)
		}
		if got {
			t.Fatalf("autoRepairSidestepFromEnv(%q) = true, want false (hold is the default)", v)
		}
	}
}

func TestAutoRepairSidestepModeEnables(t *testing.T) {
	// Case and surrounding whitespace are operator noise, not meaning -- an env var
	// set from a shell script routinely carries both.
	for _, v := range []string{"sidestep", "SIDESTEP", " SideStep "} {
		got, err := autoRepairSidestepFromEnv(v)
		if err != nil {
			t.Fatalf("autoRepairSidestepFromEnv(%q) errored: %v", v, err)
		}
		if !got {
			t.Fatalf("autoRepairSidestepFromEnv(%q) = false, want true", v)
		}
	}
}

func TestAutoRepairSidestepUnknownModeRefusesLoudly(t *testing.T) {
	// The typo case, and the reason this parser returns an error at all. A near-miss
	// must not read as "off": the operator would believe repair is active and
	// misread every subsequent hold as a safe-subset bug.
	for _, v := range []string{"sidesteps", "1", "true", "on", "yes", "repair"} {
		got, err := autoRepairSidestepFromEnv(v)
		if err == nil {
			t.Fatalf("autoRepairSidestepFromEnv(%q) = (%v, nil), want an error", v, got)
		}
		if got {
			t.Fatalf("autoRepairSidestepFromEnv(%q) returned true alongside its error", v)
		}
		// The message has to name the offending value AND the valid one, or it sends
		// the operator to the docs to fix a one-character mistake.
		if !strings.Contains(err.Error(), v) || !strings.Contains(err.Error(), autoRepairSidestepMode) {
			t.Fatalf("error %q does not name both the bad value %q and the valid mode", err, v)
		}
		if !strings.Contains(err.Error(), AutoRepairEnv) {
			t.Fatalf("error %q does not name the env var that carried it", err)
		}
	}
}

// TestAutoRepairSidestepReachesRuntimePolicy is the wiring witness: the knob is
// useless unless ToRuntime actually lands it on the adjudicator.Policy the monitor
// installs. Without this, the parser above could be perfect and the guard would still
// never repair anything.
func TestAutoRepairSidestepReachesRuntimePolicy(t *testing.T) {
	t.Setenv(AutoRepairEnv, autoRepairSidestepMode)
	rt, err := Manifest{}.ToRuntime()
	if err != nil {
		t.Fatalf("ToRuntime: %v", err)
	}
	if !rt.Adjudicator.AutoRepairSidestep {
		t.Fatal("AutoRepairSidestep = false on the runtime policy, want true (env not wired through ToRuntime)")
	}
}

func TestAutoRepairSidestepAbsentFromRuntimePolicyByDefault(t *testing.T) {
	t.Setenv(AutoRepairEnv, "")
	rt, err := Manifest{}.ToRuntime()
	if err != nil {
		t.Fatalf("ToRuntime: %v", err)
	}
	if rt.Adjudicator.AutoRepairSidestep {
		t.Fatal("AutoRepairSidestep = true with the env unset, want false")
	}
}

// A bad mode must fail the POLICY LOAD, not just the parser in isolation -- that is
// what makes the typo visible at startup instead of never.
func TestAutoRepairSidestepBadModeFailsPolicyLoad(t *testing.T) {
	t.Setenv(AutoRepairEnv, "sidesteps")
	if _, err := (Manifest{}).ToRuntime(); err == nil {
		t.Fatal("ToRuntime succeeded with an unknown FAK_GUARD_AUTOREPAIR mode, want a loud refusal")
	}
}
