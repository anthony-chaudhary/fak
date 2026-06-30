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
		"$OpenaiBackend    = ($Backend -eq 'openai')",
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

func TestClaudeGLMGCPA100TiersInRegistry(t *testing.T) {
	root := repoRootFromTest(t)
	accel := readRepoTextForClaudeGLMGCP(t, root, "tools", "gcp_accel.py")
	// The 80GB A100 tier (a2-ultragpu-8g) is the same 640 GB-VRAM shape as the private GPU-server
	// example; sm_80 is below the DSA floor, so the bring-up serves it via the pure fak
	// kernel / llama.cpp, never the stock SGLang/vLLM DSA path.
	idx := strings.Index(accel, `slug="a2-ultra-a100-80gb"`)
	if idx < 0 {
		t.Fatalf("a2-ultra-a100-80gb tier missing from tools/gcp_accel.py")
	}
	tier := accel[idx:]
	if end := strings.Index(tier, "),"); end >= 0 {
		tier = tier[:end]
	}
	for _, want := range []string{
		`machine_type="a2-ultragpu-8g"`,
		`accelerator_type="nvidia-a100-80gb"`,
		`compute_capability="80"`,
	} {
		requireContainsForClaudeGLMGCP(t, tier, want)
	}
	// The 40GB A100 keeps the legacy "Tesla" accelerator string.
	requireContainsForClaudeGLMGCP(t, accel, `accelerator_type="nvidia-tesla-a100"`)
}

func TestClaudeGLMGCPFakNativeServeWiring(t *testing.T) {
	root := repoRootFromTest(t)
	serve := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_fak_native_serve.sh")
	for _, want := range []string{
		"--backend cuda",                 // prefill+decode on the GPU HAL
		"--cpu-offload-experts",          // the ~424 GB MoE experts stay on host RAM
		"--context-budget-tokens",        // cap the KV plan (default 1M context => FitTooBig)
		"build_cuda.sh binary ./cmd/fak", // the canonical -tags cuda fak binary build
		"GLM52_FAK_NATIVE_SERVE_READY",   // the real-chat-completion health gate
	} {
		requireContainsForClaudeGLMGCP(t, serve, want)
	}
	bc := readRepoTextForClaudeGLMGCP(t, root, "internal", "compute", "build_cuda.sh")
	requireContainsForClaudeGLMGCP(t, bc, "binary)") // the DRY cuda-binary subcommand
}

func TestClaudeGLMGCPA100PlanWiresPureFakKernel(t *testing.T) {
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
	cmd.Env = append(os.Environ(), "GCP_TIER=a2-ultra-a100-80gb")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gcp-glm-serve A100 plan failed: %v\n%s", err, out)
	}
	text := string(out)
	// The Ampere default is the PURE FAK KERNEL native serve, not the stock DSA engine.
	for _, want := range []string{
		"a2-ultragpu-8g",
		"type=nvidia-a100-80gb,count=8",
		"glm52_fak_native_serve.sh",
		"PURE FAK KERNEL",
		"claude-glm-gcp",
	} {
		requireContainsForClaudeGLMGCP(t, text, want)
	}
	if strings.Contains(text, "glm52_sglang_vllm_serve.sh") {
		t.Fatalf("A100 default plan wired the sm_90 SGLang/vLLM serve; expected the pure fak kernel path")
	}
}

func TestClaudeGLMGCPA100LlamacppBenchmarkPlan(t *testing.T) {
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
	cmd.Env = append(os.Environ(), "GCP_TIER=a2-ultra-a100-80gb", "SERVE=llamacpp")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gcp-glm-serve A100 llamacpp plan failed: %v\n%s", err, out)
	}
	text := string(out)
	// SERVE=llamacpp stands up the private GPU-server example (llama.cpp MLA) as the benchmark baseline.
	for _, want := range []string{
		"a2-ultragpu-8g",
		"glm52_stage_serve_dgx3.sh",
		"BENCHMARK",
	} {
		requireContainsForClaudeGLMGCP(t, text, want)
	}
}

func TestClaudeGLMGCPA100StockEngineFailsClosed(t *testing.T) {
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
	// sm_80 is below the DSA kernel floor: the stock engines MUST fail closed, never render a
	// serve. This locks the central A100 invariant (the script gate, not just the registry cap).
	cmd.Env = append(os.Environ(), "GCP_TIER=a2-ultra-a100-80gb", "SERVE=sglang")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit for SERVE=sglang on an A100 (sm_80) tier; got success\n%s", out)
	}
	requireContainsForClaudeGLMGCP(t, string(out), "needs sm_90+")
	if strings.Contains(string(out), "glm52_sglang_vllm_serve.sh") {
		t.Fatalf("A100 + SERVE=sglang rendered a serve instead of failing closed:\n%s", out)
	}
}

// TestClaudeGLMGCPDemoPlanWiring locks the one-command demo orchestrator's contract from the
// script text (runs on every OS, no bash needed): it defaults to the 8x H100 tier, forces the
// PURE FAK KERNEL serve (so the cache-value metric exists at all), composes the canonical
// bring-up rather than re-implementing it, and renders the probe -> cache-value -> teardown
// steps. This is the host-witnessable half of epic #1476 C1 (#1477).
func TestClaudeGLMGCPDemoPlanWiring(t *testing.T) {
	root := repoRootFromTest(t)
	demo := readRepoTextForClaudeGLMGCP(t, root, "scripts", "gcp-glm-demo.sh")
	for _, want := range []string{
		`GCP_TIER="${GCP_TIER:-a3-high-h100}"`,      // the 8x H100 demo tier (GLM-5.2 needs 640 GB)
		`SERVE="${SERVE:-fak}"`,                     // the PURE FAK KERNEL — the goal, and the metric's precondition
		"scripts/gcp-glm-serve.sh",                  // composes the canonical bring-up, never re-implements it
		"claude-glm-gcp --probe",                    // step 2: the cache-warming probe turns
		"fak_gateway_kv_prefix_reused_tokens_total", // step 3: the WITNESSED cache-value datum (#1010)
		"gcloud compute instances delete",           // step 4: teardown — the demo leaves zero cost
		`MODE="plan"`,                               // plan-by-default
	} {
		requireContainsForClaudeGLMGCP(t, demo, want)
	}
}

// TestClaudeGLMGCPDemoPlanRendersWithoutCreds runs the demo orchestrator with no creds and
// asserts the rendered plan resolves the 8x H100 shape, forces the pure fak-kernel native
// serve even on sm_90 (where the serve script would otherwise pick stock SGLang), and prints
// the cache-value scrape + teardown. The live turn stays hardware-gated; the plan is proven here.
func TestClaudeGLMGCPDemoPlanRendersWithoutCreds(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash plan rendering is covered under WSL/Unix CI")
	}
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash not on PATH")
	}
	root := repoRootFromTest(t)
	cmd := exec.Command(bash, filepath.Join(root, "scripts", "gcp-glm-demo.sh"))
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gcp-glm-demo plan failed: %v\n%s", err, out)
	}
	text := string(out)
	for _, want := range []string{
		"gcloud compute instances create",           // step 1: provision (from the composed serve plan)
		"a3-highgpu-8g",                             // the 8x H100 machine type — the default tier resolved
		"glm52_fak_native_serve.sh",                 // SERVE=fak forced the pure kernel even on sm_90
		"fak_gateway_kv_prefix_reused_tokens_total", // step 3: the cache-value witness
		"gcloud compute instances delete",           // step 4: teardown
	} {
		requireContainsForClaudeGLMGCP(t, text, want)
	}
	// The cache value only exists because fak itself serves: the stock DSA serve must NOT appear.
	if strings.Contains(text, "glm52_sglang_vllm_serve.sh") {
		t.Fatalf("demo rendered the stock SGLang serve instead of the pure fak kernel:\n%s", text)
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
