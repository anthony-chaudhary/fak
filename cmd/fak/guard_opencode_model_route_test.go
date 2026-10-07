package main

import "testing"

// fak-test:runtime fast est=1s
func TestGuardOpenCodeModelPrefixSelectsUpstream(t *testing.T) {
	for _, tc := range []struct {
		name             string
		command          []string
		explicit         string
		model            string
		wantProvider     string
		wantModel        string
		wantAutodetected bool
	}{
		{"google prefix routes to gemini", []string{"opencode", "run", "--format", "json", "--model", "google/gemini-3.8-flash", "fix it"}, "", "", "gemini", "gemini-3.8-flash", true},
		{"short flag", []string{"opencode", "run", "-m", "anthropic/claude-sonnet-5"}, "", "", "anthropic", "claude-sonnet-5", true},
		{"equals form", []string{"opencode", "run", "--model=openai/gpt-5.5"}, "", "", "openai", "gpt-5.5", true},
		{"xai prefix", []string{"opencode", "run", "--model", "xai/grok-5"}, "", "", "xai", "grok-5", true},
		{"windows exe path", []string{`C:\Users\x\.opencode\bin\opencode.exe`, "run", "--model", "google/gemini-3.8-flash"}, "", "", "gemini", "gemini-3.8-flash", true},
		{"unknown prefix keeps default wire", []string{"opencode", "run", "--model", "hive-ai/deepseek-v4"}, "", "", "openai", "", true},
		{"guard fak prefix keeps default wire", []string{"opencode", "run", "--model", "fak/qwen38"}, "", "", "openai", "", true},
		{"no model keeps default wire", []string{"opencode", "run", "hello"}, "", "", "openai", "", true},
		{"explicit matching provider strips prefix", []string{"opencode", "run", "--model", "google/gemini-3.8-flash"}, "gemini", "", "gemini", "gemini-3.8-flash", false},
		{"explicit other provider keeps model", []string{"opencode", "run", "--model", "google/gemini-3.8-flash"}, "openai", "", "openai", "", false},
		{"operator model wins", []string{"opencode", "run", "--model", "google/gemini-3.8-flash"}, "", "gemini-3.8-pro", "gemini", "gemini-3.8-pro", true},
		{"non-opencode child untouched", []string{"codex", "exec", "--model", "google/gemini-3.8-flash"}, "", "", "openai-responses", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := newGuardLaunchPlan(tc.command)
			provider, autodetected := plan.resolveProvider(tc.explicit)
			provider, model, autodetected := plan.applyOpenCodeModelRoute(tc.explicit, provider, tc.model, autodetected)
			if provider != tc.wantProvider || model != tc.wantModel || autodetected != tc.wantAutodetected {
				t.Fatalf("got provider=%q model=%q autodetected=%v, want %q %q %v",
					provider, model, autodetected, tc.wantProvider, tc.wantModel, tc.wantAutodetected)
			}
		})
	}
}
