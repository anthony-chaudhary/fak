package lifecycle

import "testing"

// TestStringParseRoundTrip pins the canonical wire tokens and proves String/Parse
// are inverses over the shared core. The literal wire strings are asserted here so
// a change to a Token* breaks this test — the four tokens are a wire contract both
// session.RunState and loopmgr.LoopState now depend on.
func TestStringParseRoundTrip(t *testing.T) {
	cases := []struct {
		phase Phase
		token string
	}{
		{Running, "running"},
		{Paused, "paused"},
		{Draining, "draining"},
		{Stopped, "stopped"},
	}
	for _, c := range cases {
		if got := c.phase.String(); got != c.token {
			t.Errorf("Phase(%d).String() = %q, want %q", c.phase, got, c.token)
		}
		got, ok := Parse(c.token)
		if !ok {
			t.Errorf("Parse(%q) ok = false, want true", c.token)
		}
		if got != c.phase {
			t.Errorf("Parse(%q) = %d, want %d", c.token, got, c.phase)
		}
	}
}

// TestParseFailsClosed proves Parse rejects everything outside the shared core —
// including the layer-specific extras, which are real tokens in their own packages
// but not shared Phases. A rejected token must yield the zero value, never a silent
// Running.
func TestParseFailsClosed(t *testing.T) {
	for _, tok := range []string{"throttled", "armed", "disabled", "", "RUNNING", "unknown", "fired"} {
		if got, ok := Parse(tok); ok || got != 0 {
			t.Errorf("Parse(%q) = (%d, %v), want (0, false)", tok, got, ok)
		}
	}
}

// TestStringOutOfRange proves an out-of-range Phase renders "unknown" rather than
// panicking — a wire-derived value is never trusted to be in range.
func TestStringOutOfRange(t *testing.T) {
	if got := Phase(200).String(); got != "unknown" {
		t.Errorf("Phase(200).String() = %q, want %q", got, "unknown")
	}
}
