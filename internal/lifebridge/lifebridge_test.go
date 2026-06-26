package lifebridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestSharedTokensNotDefinedIndependently is the keystone of #913: it proves the
// served session and the loop supervisor no longer spell the four shared lifecycle
// tokens independently. For each shared state the session's wire token, the
// supervisor's LoopState constant, and the canonical lifecycle token must be the
// SAME string. The expected literal is pinned too, so changing the shared token in
// internal/lifecycle breaks this test rather than silently diverging the wire
// contract.
func TestSharedTokensNotDefinedIndependently(t *testing.T) {
	cases := []struct {
		name  string
		run   session.RunState
		loop  loopmgr.LoopState
		phase lifecycle.Phase
		wire  string
	}{
		{"running", session.Running, loopmgr.StateRunning, lifecycle.Running, "running"},
		{"paused", session.Paused, loopmgr.StatePaused, lifecycle.Paused, "paused"},
		{"draining", session.Draining, loopmgr.StateDraining, lifecycle.Draining, "draining"},
		{"stopped", session.Stopped, loopmgr.StateStopped, lifecycle.Stopped, "stopped"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.run.String(); got != c.wire {
				t.Errorf("session.RunState wire = %q, want %q", got, c.wire)
			}
			if got := string(c.loop); got != c.wire {
				t.Errorf("loopmgr.LoopState const = %q, want %q", got, c.wire)
			}
			if got := c.phase.String(); got != c.wire {
				t.Errorf("lifecycle token = %q, want %q", got, c.wire)
			}
			// The three surfaces agree because they are one definition, not three.
			if c.run.String() != string(c.loop) || string(c.loop) != c.phase.String() {
				t.Errorf("token divergence: run=%q loop=%q phase=%q",
					c.run.String(), string(c.loop), c.phase.String())
			}
		})
	}
}

// TestConverterRoundTrips proves the LoopState<->RunState converter is an inverse
// pair over the four shared states.
func TestConverterRoundTrips(t *testing.T) {
	runs := []session.RunState{session.Running, session.Paused, session.Draining, session.Stopped}
	for _, rs := range runs {
		ls, ok := RunToLoop(rs)
		if !ok {
			t.Fatalf("RunToLoop(%v) ok = false, want true", rs)
		}
		back, ok := LoopToRun(ls)
		if !ok {
			t.Fatalf("LoopToRun(%v) ok = false, want true", ls)
		}
		if back != rs {
			t.Errorf("round trip %v -> %q -> %v changed the state", rs, ls, back)
		}
	}
}

// TestExtrasHaveNoPeer proves the converter is honest about the layer-specific
// extras: a state with no peer at the other altitude converts to (zero, false),
// never a silent default.
func TestExtrasHaveNoPeer(t *testing.T) {
	if ls, ok := RunToLoop(session.Throttled); ok {
		t.Errorf("RunToLoop(Throttled) = (%q, true), want (\"\", false) — Throttled has no supervisor peer", ls)
	}
	for _, ls := range []loopmgr.LoopState{loopmgr.StateArmed, loopmgr.StateDisabled} {
		if rs, ok := LoopToRun(ls); ok {
			t.Errorf("LoopToRun(%q) = (%v, true), want (0, false) — %q has no served-session peer", ls, rs, ls)
		}
	}
	if rs, ok := LoopToRun("bogus"); ok {
		t.Errorf("LoopToRun(bogus) = (%v, true), want (0, false)", rs)
	}
}
