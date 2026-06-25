package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dropin"
)

// The gallery must be a faithful reading of dropin: one card per known agent, each
// carrying the exact invocation, the resolved provider/wire, and the injected env the
// agent would receive. A card that disagreed with dropin.PlanFor would be a brochure,
// not a demo — this pins it to the real resolution.
func TestGalleryMatchesDropin(t *testing.T) {
	const gw = "http://127.0.0.1:8080"
	cards := gallery(gw)
	agents := dropin.KnownAgents()
	if len(cards) != len(agents) {
		t.Fatalf("gallery has %d cards, want %d (one per known agent)", len(cards), len(agents))
	}
	for i, c := range cards {
		a := agents[i]
		if c.Command != a.Command || c.Display != a.Display {
			t.Errorf("card %d = (%q,%q), want (%q,%q)", i, c.Command, c.Display, a.Command, a.Display)
		}
		if c.Invocation != "fak guard -- "+a.Command {
			t.Errorf("card %q invocation = %q", c.Command, c.Invocation)
		}
		plan := dropin.PlanFor(a.Command, "", "", gw)
		if c.Provider != plan.Provider || c.Autodetected != plan.Autodetected || c.BaseURL != plan.BaseURL {
			t.Errorf("card %q drifted from dropin.PlanFor: card=(%q,%v,%q) plan=(%q,%v,%q)",
				c.Command, c.Provider, c.Autodetected, c.BaseURL, plan.Provider, plan.Autodetected, plan.BaseURL)
		}
		if len(c.EnvVars) != len(plan.EnvVars) {
			t.Errorf("card %q env vars = %v, want %v", c.Command, c.EnvVars, plan.EnvVars)
		}
		// Wire label must name the route the client actually hits (explains the /v1 split).
		if c.Provider == "anthropic" && !strings.Contains(c.Wire, "/v1/messages") {
			t.Errorf("anthropic card %q wire = %q, want it to name /v1/messages", c.Command, c.Wire)
		}
		if c.Provider == "openai" && !strings.Contains(c.Wire, "/v1/chat/completions") {
			t.Errorf("openai card %q wire = %q, want it to name /v1/chat/completions", c.Command, c.Wire)
		}
	}
}

func TestWireLabel(t *testing.T) {
	if got := wireLabel("anthropic"); !strings.Contains(got, "Anthropic") || !strings.Contains(got, "/v1/messages") {
		t.Errorf("anthropic wire label = %q", got)
	}
	if got := wireLabel("openai"); !strings.Contains(got, "OpenAI") || !strings.Contains(got, "/v1/chat/completions") {
		t.Errorf("openai wire label = %q", got)
	}
	// An unknown wire falls back to the bare provider name (never panics / drops it).
	if got := wireLabel("gemini"); got != "gemini" {
		t.Errorf("unknown wire label = %q, want bare %q", got, "gemini")
	}
}

// runSelfcheck must return 0 for the shipped dropin table — it is the CI / cross-platform
// dog-food of the gallery's data path, so a green selfcheck IS the regression gate.
func TestRunSelfcheckGreen(t *testing.T) {
	if code := runSelfcheck("http://127.0.0.1:8080"); code != 0 {
		t.Fatalf("runSelfcheck returned %d, want 0 (the shipped drop-in invariants must hold)", code)
	}
}
