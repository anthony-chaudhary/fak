package main

import (
	"strings"
	"testing"
)

// TestClaudeGLMZaiBashLauncherPreset locks the hosted Z.ai coding-plan GLM-5.2 preset
// (`claude-glm-zai` / FAK_DOGFOOD_PRESET=glm-zai) into the bash launcher: the openai backend
// fronts Z.ai's MANAGED endpoint (api.z.ai/api/coding/paas/v4) with the provider-scoped
// zai-coding-plan/glm-5.2 model and a ZAI_API_KEY bearer, mirroring tools/claude_agent_chat.py
// --glm so both z.ai front doors agree. Opt-in in the #3034 graduation manifest (external route,
// key/availability not yet classified).
func TestClaudeGLMZaiBashLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.sh")
	for _, want := range []string{
		"glm-zai)",
		"claude-glm-zai)    PRESET=\"glm-zai\" ;;",
		`PRESET="glm-zai"`,
		`DEFAULT_BACKEND="openai"`,
		`DEFAULT_OPENAI_BASE_URL="${FAK_ZAI_BASE_URL:-https://api.z.ai/api/coding/paas/v4}"`,
		`DEFAULT_MODEL="${FAK_ZAI_MODEL:-zai-coding-plan/glm-5.2}"`,
		`DEFAULT_UPSTREAM_API_KEY_ENV="${FAK_ZAI_API_KEY_ENV:-ZAI_API_KEY}"`,
		// The openai backend already raises the planner/write timeout floors GLM's prefill needs.
		"ensure_timeout_floor FAK_PLANNER_TIMEOUT_S",
		"ensure_timeout_floor FAK_HTTP_WRITE_TIMEOUT_S",
		// Post-#3034 the launcher->preset mapping and its opt-in verdict live in the manifest.
		"claude-glm-zai|glm-zai|no|ZAI_API_KEY|",
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
	// The hosted preset must NOT append /v1 to the coding-plan root (fak's openai backend
	// appends /chat/completions to the base itself).
	if strings.Contains(sh, "https://api.z.ai/api/coding/paas/v4/v1") {
		t.Fatalf("glm-zai base URL must not be suffixed with /v1")
	}
}

// TestClaudeGLMZaiPowerShellLauncherPreset locks the same preset in the PowerShell launcher so
// the primary Windows platform gets a first-class claude-glm-zai launcher at parity with the .sh.
func TestClaudeGLMZaiPowerShellLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.ps1")
	for _, want := range []string{
		"'glm-zai' {",
		"$PresetBackend   = 'openai'",
		"if ($env:FAK_ZAI_BASE_URL) { $env:FAK_ZAI_BASE_URL } else { 'https://api.z.ai/api/coding/paas/v4' }",
		"if ($env:FAK_ZAI_MODEL)    { $env:FAK_ZAI_MODEL }    else { 'zai-coding-plan/glm-5.2' }",
		"if ($env:FAK_ZAI_API_KEY_ENV) { $env:FAK_ZAI_API_KEY_ENV } else { 'ZAI_API_KEY' }",
		"Launcher='claude-glm-zai';       Preset='glm-zai';",
		"KeyEnv='ZAI_API_KEY';",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}
