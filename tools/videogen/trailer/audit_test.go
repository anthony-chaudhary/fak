package main

import (
	"strings"
	"testing"
)

func goodConfig() Config {
	return Config{Width: 1280, Height: 720, FPS: 30, Scenes: []Scene{{Kind: "hook", Secs: 4, Title: "Your agent can act."}, {Kind: "checkpoint", Secs: 4, Title: "Before the tool runs."}, {Kind: "proof", Secs: 3, Title: "Dangerous call stopped."}, {Kind: "cta", Secs: 8, Title: "Try one guarded session.", Command: "fak guard -- claude"}}}
}
func TestAuditAcceptsFocusedReadableTrailer(t *testing.T) {
	a, e := audit(goodConfig())
	if e != nil {
		t.Fatal(e)
	}
	if a.Duration != 19 || a.CTAStart != 11 || a.CTAHold != 8 {
		t.Fatalf("audit=%+v", a)
	}
}
func TestAuditRefusesLateCTA(t *testing.T) {
	c := goodConfig()
	c.Scenes[0].Secs = 7
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "CTA starts") {
		t.Fatalf("error=%v", e)
	}
}
func TestAuditRefusesDenseCopy(t *testing.T) {
	c := goodConfig()
	c.Scenes[0].Title = "one two three four five six seven eight nine"
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "max 8") {
		t.Fatalf("error=%v", e)
	}
}
func TestAuditRefusesUnreadablyBriefScene(t *testing.T) {
	c := goodConfig()
	c.Scenes[1].Secs = 1
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "too brief") {
		t.Fatalf("error=%v", e)
	}
}

func TestAuditAcceptsTokenSavingsValueModule(t *testing.T) {
	c := Config{Width: 1920, Height: 1080, FPS: 60, Scenes: []Scene{
		{Kind: "token-hook", Secs: 4, Title: "Stop paying the turn tax."},
		{Kind: "token-grid", Secs: 4, Title: "Six ways to spend less.", Items: []string{"Reuse stable prefixes", "Serve repeats locally", "Route each call", "Shed stale turns", "Skip known work", "Reuse live KV"}},
		{Kind: "token-flow", Secs: 4, Title: "Save this turn and every next turn.", Items: []string{"Skip a model turn", "Shrink context", "Run longer"}},
		{Kind: "cta", Secs: 8, Title: "Put fak at the boundary.", Command: "fak guard -- claude"},
	}}
	if _, err := audit(c); err != nil {
		t.Fatal(err)
	}
}

func TestAuditRefusesIncompleteTokenSavingsGrid(t *testing.T) {
	c := goodConfig()
	c.Width, c.Height, c.FPS = 1920, 1080, 60
	c.Scenes[0] = Scene{Kind: "token-grid", Secs: 4, Title: "Ways to spend less.", Items: []string{"Reuse prefixes"}}
	if _, err := audit(c); err == nil || !strings.Contains(err.Error(), "exactly 6") {
		t.Fatalf("error=%v", err)
	}
}

func TestAuditRefusesLowResolutionTokenVisual(t *testing.T) {
	c := goodConfig()
	c.Scenes[0].Kind = "token-hook"
	if _, err := audit(c); err == nil || !strings.Contains(err.Error(), "below 1920x1080") {
		t.Fatalf("error=%v", err)
	}
}

func TestAuditRefusesLowFrameRateTokenVisual(t *testing.T) {
	c := goodConfig()
	c.Width, c.Height, c.FPS = 1920, 1080, 30
	c.Scenes[0].Kind = "token-hook"
	if _, err := audit(c); err == nil || !strings.Contains(err.Error(), "floor 60") {
		t.Fatalf("error=%v", err)
	}
}

func TestValidateLayoutSamplesEverySceneAtThreeTimes(t *testing.T) {
	c := goodConfig()
	n, _, _, err := validateLayout(c)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(c.Scenes)*3 {
		t.Fatalf("samples=%d", n)
	}
}

func TestTokenHeroKeepsWideSafeArea(t *testing.T) {
	c := Config{Width: 1920, Height: 1080, FPS: 60, Scenes: []Scene{
		{Kind: "token-hook", Secs: 4, Title: "Stop paying for the same work."},
		{Kind: "token-grid", Secs: 4, Title: "Six ways tokens stop leaking.", Items: []string{"Stable prefix", "Local repeats", "Right-size model", "Stale turns shed", "Known work skipped", "Live KV reused"}},
		{Kind: "token-flow", Secs: 4, Title: "One avoided turn shrinks every next turn.", Items: []string{"Skip a model turn", "Shrink context", "Run longer"}},
		{Kind: "cta", Secs: 8, Title: "Put fak at the boundary.", Command: "fak guard -- claude"},
	}}
	a, err := audit(c)
	if err != nil {
		t.Fatal(err)
	}
	_, right, _, err := validateLayout(c)
	if err != nil {
		t.Fatal(err)
	}
	if a.SafeMarginPx < 210 {
		t.Fatalf("safe margin=%d, want at least 210", a.SafeMarginPx)
	}
	if right > c.Width-a.SafeMarginPx {
		t.Fatalf("text edge=%d crosses safe boundary=%d", right, c.Width-a.SafeMarginPx)
	}
}

func TestHarnessTrailerKindsAuditAndRender(t *testing.T) {
	cfg := Config{Width: 1920, Height: 1080, FPS: 30, Scenes: []Scene{
		{Kind: "harness-hook", Secs: 3.5, Eyebrow: "YOUR AGENT", Title: "Build your harness", Subtitle: "Connect the parts"},
		{Kind: "harness-blueprint", Secs: 4.5, Eyebrow: "PICK PARTS", Title: "One blueprint", Detail: "owned core", Items: []string{"BRAND", "MODEL", "TOOLS", "MEMORY", "POLICY", "UI"}},
		{Kind: "harness-run", Secs: 3.5, Eyebrow: "RUN IT", Title: "Checked action", Items: []string{"YOUR UI", "fak kernel", "YOUR TOOLS"}, Verdict: "SELF-CHECK PASS"},
		{Kind: "cta", Secs: 7, Eyebrow: "START", Title: "Make it yours", Command: "fak harness init my-agent"},
	}}
	if _, err := audit(cfg); err != nil {
		t.Fatalf("audit harness trailer: %v", err)
	}
	p, err := newPainter()
	if err != nil {
		t.Fatal(err)
	}
	for _, scene := range cfg.Scenes[:3] {
		im := sceneFrame(cfg, scene, scene.Secs/2, p)
		if got := im.Bounds().Size(); got.X != cfg.Width || got.Y != cfg.Height {
			t.Fatalf("%s frame size = %v", scene.Kind, got)
		}
	}
}

func TestHarnessBlueprintRejectsMissingParts(t *testing.T) {
	cfg := Config{Width: 1280, Height: 720, FPS: 30, Scenes: []Scene{
		{Kind: "harness-blueprint", Secs: 5, Items: []string{"MODEL"}},
		{Kind: "proof", Secs: 5},
		{Kind: "cta", Secs: 8, Command: "fak harness init my-agent"},
	}}
	if _, err := audit(cfg); err == nil || !strings.Contains(err.Error(), "needs exactly 6") {
		t.Fatalf("expected six-part blueprint error, got %v", err)
	}
}
