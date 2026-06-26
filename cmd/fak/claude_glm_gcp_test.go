package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestClaudeGLMGCPBashLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.sh")
	for _, want := range []string{
		"glm-gcp)",
		`DEFAULT_BACKEND="openai"`,
		`DEFAULT_OPENAI_BASE_URL="${FAK_GLM_GCP_BASE_URL:-http://127.0.0.1:8200/v1}"`,
		`DEFAULT_MODEL="${FAK_GLM_GCP_MODEL:-glm-5.2}"`,
		"claude-glm-gcp)",
		`PRESET="glm-gcp"`,
		`glm_name="claude-glm-gcp"`,
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
}

func TestClaudeGLMGCPPowerShellLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.ps1")
	for _, want := range []string{
		"'glm-gcp'",
		"FAK_GLM_GCP_BASE_URL",
		"$OpenaiBackend = ($Backend -eq 'openai')",
		"Resolve-OpenAiBaseUrl",
		"Get-FirstOpenAiModel",
		"claude-glm-gcp.cmd",
		"FAK_DOGFOOD_PRESET=glm-gcp",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}

func TestClaudeGLMGCPBringupPlanWiring(t *testing.T) {
	root := repoRootFromTest(t)
	gcp := readRepoTextForClaudeGLMGCP(t, root, "scripts", "gcp-glm-serve.sh")
	for _, want := range []string{
		"glm52_sglang_vllm_serve.sh",
		"gcp_accel.py",
		"--emit-shell",
		"claude-glm-gcp",
		"FAK_GLM_GCP_BASE_URL",
		`MODE="plan"`,
	} {
		requireContainsForClaudeGLMGCP(t, gcp, want)
	}
}

func TestClaudeGLMGCPBringupPlanRendersWithoutCreds(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash plan rendering is covered under WSL/Unix CI")
	}
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash not on PATH")
	}
	root := repoRootFromTest(t)
	cmd := exec.Command(bash, filepath.Join(root, "scripts", "gcp-glm-serve.sh"))
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gcp-glm-serve plan failed: %v\n%s", err, out)
	}
	text := string(out)
	for _, want := range []string{
		"gcloud compute instances create",
		"glm52_sglang_vllm_serve.sh",
		"claude-glm-gcp",
		"a3-ultragpu-8g",
	} {
		requireContainsForClaudeGLMGCP(t, text, want)
	}
}

func TestClaudeGLMGCPDefaultTierClearsDSAFloor(t *testing.T) {
	root := repoRootFromTest(t)
	accel := readRepoTextForClaudeGLMGCP(t, root, "tools", "gcp_accel.py")
	idx := strings.Index(accel, `slug="a3-ultra-h200"`)
	if idx < 0 {
		t.Fatalf("a3-ultra-h200 tier missing from tools/gcp_accel.py")
	}
	tier := accel[idx:]
	if end := strings.Index(tier, "),"); end >= 0 {
		tier = tier[:end]
	}
	for _, want := range []string{
		`machine_type="a3-ultragpu-8g"`,
		`accelerator_type="nvidia-h200-141gb"`,
		`compute_capability="90"`,
	} {
		requireContainsForClaudeGLMGCP(t, tier, want)
	}
}

func readRepoTextForClaudeGLMGCP(t *testing.T, root string, elems ...string) string {
	t.Helper()
	path := filepath.Join(append([]string{root}, elems...)...)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

func requireContainsForClaudeGLMGCP(t *testing.T, text, want string) {
	t.Helper()
	if !strings.Contains(text, want) {
		t.Fatalf("missing %q", want)
	}
}
