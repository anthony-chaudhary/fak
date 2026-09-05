package harnessprofile

import "testing"

// TestLookupReproducesGuardDetectTable is the acceptance witness for C1: the registry
// must reproduce guardDetectProvider's wire for EVERY name the current switch
// recognizes, and the unrecognized case must be a miss (the caller supplies the
// anthropic fallback). The expected wires below are guardDetectProvider's returns
// (cmd/fak/guard_provider.go) copied by hand — C2 makes guardDetectProvider delegate
// here, at which point the existing guard suite cross-checks this against the live
// switch. Keeping the expectations explicit here means this leaf proves the table
// without importing cmd/fak (which a foundation leaf may not do).
func TestLookupReproducesGuardDetectTable(t *testing.T) {
	cases := []struct {
		command  string
		wantWire Wire
		wantOK   bool
	}{
		{"claude", WireAnthropic, true},
		{"claude-code", WireAnthropic, true},
		{"/usr/local/bin/claude", WireAnthropic, true},              // absolute POSIX path
		{`C:\Program Files\claude\claude.exe`, WireAnthropic, true}, // Windows launcher
		{"Claude", WireAnthropic, true},                             // case-insensitive
		{"codex", WireOpenAIResponses, true},
		{"codex.cmd", WireOpenAIResponses, true},         // Windows .cmd worker
		{"/opt/openai/codex", WireOpenAIResponses, true}, // absolute path still matches
		{"gemini", WireGemini, true},
		{"gemini.ps1", WireGemini, true},
		{"fak", WireOpenAI, true},
		{"fak-agent", WireOpenAI, true},
		{"/usr/local/bin/fak", WireOpenAI, true},
		{`C:\Program Files\fak\fak.exe`, WireOpenAI, true},
		{"fak.cmd", WireOpenAI, true},
		{"opencode", WireOpenAI, true},
		{"opencode.cmd", WireOpenAI, true},
		{"aider", WireOpenAI, true},
		{"hermes", WireOpenAI, true},
		{"pi", WireAnthropic, true},                // Pi (earendil-works) speaks Anthropic Messages
		{"pi.exe", WireAnthropic, true},            // Windows launcher
		{"/usr/local/bin/pi", WireAnthropic, true}, // absolute POSIX path
		{"vim", "", false},                         // unrecognized -> miss (caller falls back to anthropic)
		{"some-unknown-agent", "", false},          // unrecognized -> miss
		{"", "", false},                            // empty -> miss
	}
	for _, tc := range cases {
		p, ok := Lookup(tc.command)
		if ok != tc.wantOK {
			t.Errorf("Lookup(%q) recognized = %v, want %v", tc.command, ok, tc.wantOK)
			continue
		}
		if p.Wire != tc.wantWire {
			t.Errorf("Lookup(%q).Wire = %q, want %q", tc.command, p.Wire, tc.wantWire)
		}
		if p.Recognized() != tc.wantOK {
			t.Errorf("Lookup(%q).Recognized() = %v, want %v", tc.command, p.Recognized(), tc.wantOK)
		}
	}
}

// TestDefaultBaseURLMatchesGuardTable pins the wire→URL mapping (BaseURLForWire) and the
// per-profile DefaultBaseURL against guardDefaultBaseURL's values: anthropic has NO /v1
// (its client appends the Messages path), both OpenAI wires share the /v1 root, and an
// unknown wire yields "" so the caller can require an explicit --base-url.
func TestDefaultBaseURLMatchesGuardTable(t *testing.T) {
	cases := []struct {
		wire Wire
		want string
	}{
		{WireAnthropic, "https://api.anthropic.com"},
		{WireOpenAI, "https://api.openai.com/v1"},
		{WireOpenAIResponses, "https://api.openai.com/v1"},
		{WireGemini, "https://generativelanguage.googleapis.com/v1beta"},
		{Wire("groq"), ""}, // unknown wire -> no default
	}
	for _, tc := range cases {
		if got := BaseURLForWire(tc.wire); got != tc.want {
			t.Errorf("BaseURLForWire(%q) = %q, want %q", tc.wire, got, tc.want)
		}
	}
	// Every built-in profile's DefaultBaseURL must agree with the wire→URL table — the
	// two must never drift, since C2 uses BaseURLForWire while the profile carries its own.
	for _, p := range Profiles() {
		if got := BaseURLForWire(p.Wire); got != p.DefaultBaseURL {
			t.Errorf("profile %q: DefaultBaseURL %q != BaseURLForWire(%q) %q", p.Name, p.DefaultBaseURL, p.Wire, got)
		}
	}
}

// TestRepointEncodesTodaysWiring pins each built-in's declared Repoint set to what guard
// actually fires today: env is always-on (guardInjectedEnv runs for every provider),
// Claude additionally takes settings-file, Codex additionally takes cli-config, and the
// plain OpenAI agents take env only. C3 dispatches off exactly these declarations.
func TestRepointEncodesTodaysWiring(t *testing.T) {
	want := map[string][]RepointMechanism{
		"claude":         {RepointEnv, RepointSettingsFile},
		"codex":          {RepointEnv, RepointCLIConfig},
		"fak":            {RepointEnv, RepointSettingsFile},
		"gemini":         {RepointEnv, RepointSystemSettingsEnv},
		"openai-generic": {RepointEnv},
		"pi":             {RepointEnv, RepointExtension},
	}
	for _, p := range Profiles() {
		got := p.Repoint
		exp := want[p.Name]
		if len(got) != len(exp) {
			t.Errorf("profile %q Repoint = %v, want %v", p.Name, got, exp)
			continue
		}
		for i := range exp {
			if got[i] != exp[i] {
				t.Errorf("profile %q Repoint[%d] = %q, want %q", p.Name, i, got[i], exp[i])
			}
		}
		// Every profile must declare the always-on env repoint.
		if !p.HasRepoint(RepointEnv) {
			t.Errorf("profile %q must declare RepointEnv (guardInjectedEnv fires for every provider)", p.Name)
		}
	}
}

// TestClosedVocabulariesValidate guards the C6 validation contract: the built-in wires
// and mechanisms are Valid, and a made-up value is not (so config load rejects it).
func TestClosedVocabulariesValidate(t *testing.T) {
	for _, w := range []Wire{WireAnthropic, WireOpenAI, WireOpenAIResponses, WireGemini} {
		if !w.Valid() {
			t.Errorf("built-in wire %q should be Valid", w)
		}
	}
	if Wire("groq").Valid() {
		t.Errorf("unknown wire should not be Valid")
	}
	for _, m := range []RepointMechanism{RepointEnv, RepointCLIConfig, RepointSettingsFile, RepointExtension, RepointSystemSettingsEnv} {
		if !m.Valid() {
			t.Errorf("built-in mechanism %q should be Valid", m)
		}
	}
	if RepointMechanism("smoke-signal").Valid() {
		t.Errorf("unknown mechanism should not be Valid")
	}
}

// TestProfilesIsACopy confirms the dump surface hands out a copy, so a caller mutating
// the returned slice cannot corrupt the shared registry.
func TestProfilesIsACopy(t *testing.T) {
	a := Profiles()
	if len(a) == 0 {
		t.Fatal("Profiles() returned nothing")
	}
	a[0].Name = "mutated"
	if b := Profiles(); b[0].Name == "mutated" {
		t.Errorf("Profiles() must return a fresh copy; registry was mutated through it")
	}
}

func TestLookupFakHarnessProfile(t *testing.T) {
	for _, tc := range []struct {
		cmd      string
		wantName string
	}{
		{"fak", "fak"},
		{"fak-agent", "fak"},
		{"FAK", "fak"},
		{"FAK-AGENT", "fak"},
		{"/usr/local/bin/fak", "fak"},
		{`C:\Program Files\fak\fak.exe`, "fak"},
		{"fak-agent.exe", "fak"},
	} {
		p, ok := Lookup(tc.cmd)
		if !ok {
			t.Fatalf("Lookup(%q) ok = false, want true", tc.cmd)
		}
		if p.Name != tc.wantName {
			t.Errorf("Lookup(%q).Name = %q, want %q", tc.cmd, p.Name, tc.wantName)
		}
		if p.Wire != WireOpenAI {
			t.Errorf("Lookup(%q).Wire = %q, want %q", tc.cmd, p.Wire, WireOpenAI)
		}
		if p.AdapterVersion != "1.0.0" {
			t.Errorf("Lookup(%q).AdapterVersion = %q, want 1.0.0", tc.cmd, p.AdapterVersion)
		}
		if p.DefaultBaseURL != "https://api.openai.com/v1" {
			t.Errorf("Lookup(%q).DefaultBaseURL = %q, want https://api.openai.com/v1", tc.cmd, p.DefaultBaseURL)
		}
		if !p.HasRepoint(RepointEnv) || !p.HasRepoint(RepointSettingsFile) {
			t.Errorf("Lookup(%q).Repoint = %v, want env and settings-file", tc.cmd, p.Repoint)
		}
		if p.Credential.Kind != CredentialEnvKey || p.Credential.EnvKey != "OPENAI_API_KEY" {
			t.Errorf("Lookup(%q).Credential = %+v, want OPENAI_API_KEY", tc.cmd, p.Credential)
		}
		if p.ConfigHomeGlob != ".fak*" {
			t.Errorf("Lookup(%q).ConfigHomeGlob = %q, want .fak*", tc.cmd, p.ConfigHomeGlob)
		}
		if p.Identity != IdentityEnvKey {
			t.Errorf("Lookup(%q).Identity = %q, want env-key", tc.cmd, p.Identity)
		}
	}
}
