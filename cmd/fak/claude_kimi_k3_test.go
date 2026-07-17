package main

import "testing"

func TestClaudeKimiK3WrapperPowerShell(t *testing.T) {
	root := repoRootFromTest(t)
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "claude-kimi-k3.ps1")
	for _, want := range []string{
		"dogfood-claude.ps1", "$env:FAK_DOGFOOD_BACKEND = 'openai'",
		"https://api.moonshot.ai/v1", "kimi-k3", "MOONSHOT_API_KEY",
		"FAK_KIMI_K3_BASE_URL", "FAK_KIMI_K3_MODEL", "FAK_KIMI_K3_API_KEY_ENV",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}

func TestClaudeKimiK3WrapperBash(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "claude-kimi-k3.sh")
	for _, want := range []string{
		"dogfood-claude.sh", `FAK_DOGFOOD_BACKEND="${FAK_DOGFOOD_BACKEND:-openai}"`,
		"https://api.moonshot.ai/v1", "kimi-k3", "MOONSHOT_API_KEY",
		"FAK_KIMI_K3_BASE_URL", "FAK_KIMI_K3_MODEL", "FAK_KIMI_K3_API_KEY_ENV",
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
}
