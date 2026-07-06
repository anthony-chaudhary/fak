package main

import "testing"

func TestClaudeNIMKimiWrapperPowerShell(t *testing.T) {
	root := repoRootFromTest(t)
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "claude-nim-kimi.ps1")
	for _, want := range []string{
		"dogfood-claude.ps1",
		"$env:FAK_DOGFOOD_BACKEND = 'openai'",
		"https://integrate.api.nvidia.com/v1",
		"moonshotai/kimi-k2.6",
		"$env:FAK_DOGFOOD_API_KEY_ENV",
		"NVIDIA_API_KEY",
		"FAK_NIM_KIMI_BASE_URL",
		"FAK_NIM_KIMI_MODEL",
		"FAK_NIM_KIMI_API_KEY_ENV",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}

func TestClaudeNIMKimiWrapperBash(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "claude-nim-kimi.sh")
	for _, want := range []string{
		"dogfood-claude.sh",
		`FAK_DOGFOOD_BACKEND="${FAK_DOGFOOD_BACKEND:-openai}"`,
		"https://integrate.api.nvidia.com/v1",
		"moonshotai/kimi-k2.6",
		`FAK_DOGFOOD_API_KEY_ENV="${FAK_DOGFOOD_API_KEY_ENV:-${FAK_NIM_KIMI_API_KEY_ENV:-NVIDIA_API_KEY}}"`,
		"FAK_NIM_KIMI_BASE_URL",
		"FAK_NIM_KIMI_MODEL",
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
}
