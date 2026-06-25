package dropin

import "testing"

func TestDetectProvider(t *testing.T) {
	cases := []struct {
		command        string
		wantProvider   string
		wantRecognized bool
	}{
		{"claude", "anthropic", true},
		{"claude-code", "anthropic", true},
		{"/usr/local/bin/claude", "anthropic", true},              // absolute path
		{`C:\Program Files\claude\claude.exe`, "anthropic", true}, // Windows launcher
		{"Claude", "anthropic", true},                             // case-insensitive
		{"codex", "openai", true},
		{"opencode", "openai", true},
		{"opencode.cmd", "openai", true}, // the Windows .cmd worker
		{"aider", "openai", true},        // reads OPENAI_API_BASE, which guard injects alongside OPENAI_BASE_URL
		{"vim", "", false},
		{"", "", false},
	}
	for _, tc := range cases {
		p, ok := DetectProvider(tc.command)
		if p != tc.wantProvider || ok != tc.wantRecognized {
			t.Errorf("DetectProvider(%q) = (%q,%v), want (%q,%v)", tc.command, p, ok, tc.wantProvider, tc.wantRecognized)
		}
	}
}

func TestResolveProvider(t *testing.T) {
	cases := []struct {
		flagValue      string
		command        string
		wantProvider   string
		wantAutodetect bool
	}{
		{"openai", "claude", "openai", false},          // explicit flag wins over the name
		{"  Anthropic ", "codex", "anthropic", false},  // explicit flag is normalized, still wins
		{"", "codex", "openai", true},                  // empty flag -> inferred
		{"", "claude", "anthropic", true},              // empty flag -> inferred (the common case)
		{"", "some-unknown-agent", "anthropic", false}, // unrecognized -> anthropic fallback, NOT flagged as detected
	}
	for _, tc := range cases {
		p, auto := ResolveProvider(tc.flagValue, tc.command)
		if p != tc.wantProvider || auto != tc.wantAutodetect {
			t.Errorf("ResolveProvider(%q,%q) = (%q,%v), want (%q,%v)", tc.flagValue, tc.command, p, auto, tc.wantProvider, tc.wantAutodetect)
		}
	}
}

func TestDefaultBaseURL(t *testing.T) {
	if got := DefaultBaseURL("anthropic"); got != "https://api.anthropic.com" {
		t.Errorf("anthropic default = %q", got)
	}
	if got := DefaultBaseURL("openai"); got != "https://api.openai.com/v1" {
		t.Errorf("openai default = %q", got)
	}
	if got := DefaultBaseURL("groq"); got != "" {
		t.Errorf("unknown provider should have no default, got %q", got)
	}
}

func TestEnvVar(t *testing.T) {
	cases := []struct {
		provider string
		override string
		want     string
	}{
		{"anthropic", "", "ANTHROPIC_BASE_URL"},
		{"openai", "", "OPENAI_BASE_URL"},
		{"gemini", "", "OPENAI_BASE_URL"},
		{"xai", "", "OPENAI_BASE_URL"},
		{"anthropic", "MY_BASE", "MY_BASE"},        // override always wins
		{"openai", "  CUSTOM_URL  ", "CUSTOM_URL"}, // trimmed
	}
	for _, tc := range cases {
		if got := EnvVar(tc.provider, tc.override); got != tc.want {
			t.Errorf("EnvVar(%q,%q) = %q, want %q", tc.provider, tc.override, got, tc.want)
		}
	}
}

func TestEnvValue(t *testing.T) {
	gw := "http://127.0.0.1:8137"
	// Anthropic clients append "/v1/messages" — the value must be the bare host.
	if got := EnvValue("anthropic", gw); got != gw {
		t.Errorf("anthropic value = %q, want bare host %q", got, gw)
	}
	// OpenAI-compatible clients treat the value as ending in /v1 and append
	// "/chat/completions" — so it MUST carry /v1 or the gateway 404s.
	for _, p := range []string{"openai", "gemini", "xai", "other"} {
		if got := EnvValue(p, gw); got != gw+"/v1" {
			t.Errorf("%s value = %q, want %s/v1", p, got, gw)
		}
	}
	// A trailing slash on the host does not double up.
	if got := EnvValue("openai", gw+"/"); got != gw+"/v1" {
		t.Errorf("trailing-slash host = %q, want %s/v1", got, gw)
	}
}

func TestInjectedEnv(t *testing.T) {
	const gw = "http://127.0.0.1:8137"

	// Anthropic: exactly one var, the bare host (the client appends /v1/messages).
	if got := InjectedEnv("anthropic", "", gw); len(got) != 1 || got[0] != [2]string{"ANTHROPIC_BASE_URL", gw} {
		t.Errorf("anthropic injected = %v, want one ANTHROPIC_BASE_URL=%s", got, gw)
	}

	// OpenAI wire with no override: BOTH conventional base-URL vars, each carrying /v1,
	// so a client reading OPENAI_API_BASE (Aider, LiteLLM) connects as well as one
	// reading OPENAI_BASE_URL (Codex, OpenCode, the OpenAI SDK).
	want := [][2]string{{"OPENAI_BASE_URL", gw + "/v1"}, {"OPENAI_API_BASE", gw + "/v1"}}
	for _, p := range []string{"openai", "gemini", "xai", "other"} {
		got := InjectedEnv(p, "", gw)
		if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Errorf("%s injected = %v, want %v", p, got, want)
		}
	}

	// An explicit override yields exactly that one var (no OPENAI_API_BASE alias),
	// still carrying the /v1 the OpenAI wire needs.
	if got := InjectedEnv("openai", "MY_BASE", gw); len(got) != 1 || got[0] != [2]string{"MY_BASE", gw + "/v1"} {
		t.Errorf("override injected = %v, want one MY_BASE=%s/v1", got, gw)
	}
}

// PlanFor must compose the resolution steps faithfully: the common drop-ins resolve
// to the wire, base URL, and injected env a real `fak guard -- <agent>` would wire.
func TestPlanFor(t *testing.T) {
	const gw = "http://127.0.0.1:9000"

	claude := PlanFor("claude", "", "", gw)
	if claude.Provider != "anthropic" || !claude.Autodetected || !claude.Recognized {
		t.Errorf("claude plan provider/autodetect/recognized = %q/%v/%v", claude.Provider, claude.Autodetected, claude.Recognized)
	}
	if claude.BaseURL != "https://api.anthropic.com" {
		t.Errorf("claude base URL = %q", claude.BaseURL)
	}
	if len(claude.EnvVars) != 1 || claude.EnvVars[0] != [2]string{"ANTHROPIC_BASE_URL", gw} {
		t.Errorf("claude env = %v, want one ANTHROPIC_BASE_URL=%s (bare host)", claude.EnvVars, gw)
	}

	codex := PlanFor("codex", "", "", gw)
	if codex.Provider != "openai" || !codex.Autodetected || !codex.Recognized {
		t.Errorf("codex plan provider/autodetect/recognized = %q/%v/%v", codex.Provider, codex.Autodetected, codex.Recognized)
	}
	if len(codex.EnvVars) != 2 || codex.EnvVars[0] != [2]string{"OPENAI_BASE_URL", gw + "/v1"} {
		t.Errorf("codex env = %v, want OPENAI_BASE_URL + OPENAI_API_BASE carrying /v1", codex.EnvVars)
	}

	// An explicit --provider on an unknown agent: NOT autodetected, NOT recognized,
	// but still fully wired (the universal `fak guard --provider openai -- <tool>`).
	custom := PlanFor("my-cli", "openai", "", gw)
	if custom.Provider != "openai" || custom.Autodetected || custom.Recognized {
		t.Errorf("custom plan provider/autodetect/recognized = %q/%v/%v", custom.Provider, custom.Autodetected, custom.Recognized)
	}

	// An unknown agent with NO --provider falls back to anthropic passthrough (not
	// flagged as autodetected) — exactly the historical default.
	fallback := PlanFor("some-tool", "", "", gw)
	if fallback.Provider != "anthropic" || fallback.Autodetected || fallback.Recognized {
		t.Errorf("fallback plan = %q/%v/%v, want anthropic/false/false", fallback.Provider, fallback.Autodetected, fallback.Recognized)
	}
}

// TestKnownAgentsAreRecognized is the load-bearing invariant of the gallery: every
// agent the demo advertises as a `fak guard -- <name>` drop-in MUST actually be
// autodetected by DetectProvider, or the card would promise an autodetect that does
// not exist. It also pins the gallery to non-empty, well-formed rows.
func TestKnownAgentsAreRecognized(t *testing.T) {
	agents := KnownAgents()
	if len(agents) == 0 {
		t.Fatal("KnownAgents() is empty — the gallery has nothing to show")
	}
	seen := map[string]bool{}
	for _, a := range agents {
		if a.Command == "" || a.Display == "" || a.Note == "" {
			t.Errorf("agent %q has an empty Command/Display/Note: %+v", a.Command, a)
		}
		if seen[a.Command] {
			t.Errorf("agent command %q is listed twice", a.Command)
		}
		seen[a.Command] = true
		provider, recognized := DetectProvider(a.Command)
		if !recognized {
			t.Errorf("KnownAgents lists %q but DetectProvider does not recognize it — the card would advertise an autodetect that does not exist", a.Command)
		}
		// And PlanFor for a known agent is always autodetected (no flag needed).
		if p := PlanFor(a.Command, "", "", "http://127.0.0.1:0"); !p.Autodetected || p.Provider != provider {
			t.Errorf("known agent %q: PlanFor autodetected=%v provider=%q, want true/%q", a.Command, p.Autodetected, p.Provider, provider)
		}
	}
}
