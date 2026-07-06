package main

import "testing"

// TestClaudeGeminiGCPBashLauncherPreset locks the `gemini-gcp` preset wiring in the bash
// launcher (scripts/dogfood-claude.sh): the claude-gemini-gcp launcher fronts GCP Vertex AI
// Gemini 3.5 Flash through fak's openai backend, so every tool call still crosses the kernel
// floor. It is the claude-glm-gcp wire pointed at a Google-MANAGED model (no VM to stand up),
// authed like the mac preset (a bearer token env var). Proven on any host from the script text.
func TestClaudeGeminiGCPBashLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.sh")
	for _, want := range []string{
		// invoked-name -> preset mapping (the installed claude-gemini-gcp launcher)
		`claude-gemini-gcp) PRESET="gemini-gcp" ;;`,
		// the preset case: openai backend, Vertex OpenAI-compat upstream, tier-2 Flash id
		"gemini-gcp)",
		`PRESET="gemini-gcp"`,
		`DEFAULT_BACKEND="openai"`,
		`DEFAULT_MODEL="${FAK_GEMINI_GCP_MODEL:-google/gemini-3.5-flash}"`,
		// authenticated remote: the GCP access-token bearer env var, like the mac preset
		`DEFAULT_UPSTREAM_API_KEY_ENV="FAK_GEMINI_GCP_KEY"`,
		`DEFAULT_OPENAI_TOOL_MESSAGES_AS_TEXT="1"`,
		// the Vertex OpenAI-compat endpoint the base URL is built from
		"aiplatform.googleapis.com",
		"endpoints/openapi",
		// project/location knobs + the fail-loud posture when neither base URL nor project is set
		"FAK_GEMINI_GCP_PROJECT",
		"FAK_GEMINI_GCP_LOCATION",
		`_gem_loc="${FAK_GEMINI_GCP_LOCATION:-global}"`,
		`_gem_host="aiplatform.googleapis.com"`,
		"/models probe skipped",
		"FAK_DOGFOOD_PRESET=gemini-gcp needs FAK_GEMINI_GCP_BASE_URL",
		// installed launcher symlink
		`gemini_name="claude-gemini-gcp"`,
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
}

// TestClaudeGeminiGCPPowerShellLauncherPreset is the Windows twin: the same preset wiring in
// scripts/dogfood-claude.ps1, including the claude-gemini-gcp.cmd shim the installer writes.
func TestClaudeGeminiGCPPowerShellLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.ps1")
	for _, want := range []string{
		"'gemini-gcp'",
		"FAK_GEMINI_GCP_BASE_URL",
		"FAK_GEMINI_GCP_PROJECT",
		"$PresetApiKeyEnv = 'FAK_GEMINI_GCP_KEY'",
		"$PresetOpenAIToolMessagesAsText = '1'",
		"else { 'google/gemini-3.5-flash' }",
		"else { 'global' }",
		"$gemHost = if ($gemLoc -eq 'global')",
		"/models probe skipped",
		"aiplatform.googleapis.com",
		"endpoints/openapi",
		"claude-gemini-gcp.cmd",
		"FAK_DOGFOOD_PRESET=gemini-gcp",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}
