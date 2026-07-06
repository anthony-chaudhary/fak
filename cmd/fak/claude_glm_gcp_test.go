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
		`DEFAULT_PROVIDER_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'`,
		"ensure_timeout_floor FAK_PLANNER_TIMEOUT_S",
		"ensure_timeout_floor FAK_HTTP_WRITE_TIMEOUT_S",
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
		`$PresetExtraBody = '{"chat_template_kwargs":{"enable_thinking":false}}'`,
		"Ensure-TimeoutFloor 'FAK_PLANNER_TIMEOUT_S'",
		"Ensure-TimeoutFloor 'FAK_HTTP_WRITE_TIMEOUT_S'",
		"claude-glm-gcp.cmd",
		"FAK_DOGFOOD_PRESET=glm-gcp",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}

func TestClaudeQwen36DogfoodLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.sh")
	ps1 := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.ps1")

	for _, want := range []string{
		"fak-qwen36-claude)",
		`PRESET="qwen36-local"`,
		`DEFAULT_OPENAI_BASE_URL="http://127.0.0.1:8131/v1"`,
		`DEFAULT_MODEL="lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"`,
		`DEFAULT_PROVIDER_EXTRA_BODY='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'`,
		`qwen_name="fak-qwen36-claude"`,
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
	}
	for _, want := range []string{
		"'qwen36-local'",
		"$PresetBaseUrl   = 'http://127.0.0.1:8131/v1'",
		"$PresetModel     = 'lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M'",
		`$PresetExtraBody = '{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'`,
		"fak-qwen36-claude.cmd",
		"FAK_DOGFOOD_PRESET=qwen36-local",
		"FAK_PROVIDER_EXTRA_BODY_JSON",
	} {
		requireContainsForClaudeGLMGCP(t, ps1, want)
	}
}

func TestClaudeMacDogfoodBashLauncherPreset(t *testing.T) {
	root := repoRootFromTest(t)
	sh := readRepoTextForClaudeGLMGCP(t, root, "scripts", "dogfood-claude.sh")
	for _, want := range []string{
		"claude-mac)",
		`PRESET="mac"`,
		`FAK_DOGFOOD_PRESET=mac requires FAK_MAC_GATEWAY=http://<macbook-ip>:8080`,
		`DEFAULT_OPENAI_BASE_URL="$FAK_MAC_GATEWAY"`,
		`DEFAULT_MODEL="${FAK_MAC_MODEL:-lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M}"`,
		`DEFAULT_UPSTREAM_API_KEY_ENV="FAK_GATEWAY_KEY"`,
		`UPSTREAM_API_KEY_ENV="${FAK_DOGFOOD_API_KEY_ENV:-$DEFAULT_UPSTREAM_API_KEY_ENV}"`,
		`mac_name="claude-mac"`,
		`ln -sf "$target" "$bindir/$mac_name"`,
	} {
		requireContainsForClaudeGLMGCP(t, sh, want)
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
		"MAX_DAILY_USD",
		"fak-budget-reaper",
		"PROVISIONING_MODEL",
		"REQUEST_VALID_FOR_DURATION",
		"--request-valid-for-duration",
		`CTX="${CTX:-65536}"`,
		`--setenv=CTX="${CTX}"`,
		`EP_RANKS="${EP_RANKS:-1}"`,
		`--setenv=EP_RANKS="${EP_RANKS}"`,
		`FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG:-}"`,
		`--setenv=FAK_EP_REQUIRE_DEVICE_PG="${FAK_EP_REQUIRE_DEVICE_PG}"`,
		`EP_RANKS=$EP_RANKS exceeds tier`,
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

func TestClaudeGLMGCPEPRanksMustFitTier(t *testing.T) {
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
	cmd.Env = append(os.Environ(), "GCP_TIER=a3-high-h100-1g", "SERVE=fak", "EP_RANKS=8")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected EP_RANKS > GPU count to fail closed; got success\n%s", out)
	}
	requireContainsForClaudeGLMGCP(t, string(out), "EP_RANKS=8 exceeds tier")
	requireContainsForClaudeGLMGCP(t, string(out), "GPU count (1)")
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

func TestClaudeGLMGCPH100MegaTierInRegistry(t *testing.T) {
	root := repoRootFromTest(t)
	accel := readRepoTextForClaudeGLMGCP(t, root, "tools", "gcp_accel.py")
	idx := strings.Index(accel, `slug="a3-mega-h100"`)
	if idx < 0 {
		t.Fatalf("a3-mega-h100 tier missing from tools/gcp_accel.py")
	}
	tier := accel[idx:]
	if end := strings.Index(tier, "),"); end >= 0 {
		tier = tier[:end]
	}
	for _, want := range []string{
		`machine_type="a3-megagpu-8g"`,
		`accelerator_type="nvidia-h100-mega-80gb"`,
		`compute_capability="90"`,
	} {
		requireContainsForClaudeGLMGCP(t, tier, want)
	}

	probe := readRepoTextForClaudeGLMGCP(t, root, "tools", "gcp_gpu_probe.py")
	requireContainsForClaudeGLMGCP(t, probe, `"nvidia-h100-mega-80gb": "NVIDIA_H100_MEGA"`)
}

func TestClaudeGLMGCPFakNativeServeWiring(t *testing.T) {
	root := repoRootFromTest(t)
	serve := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_fak_native_serve.sh")
	for _, want := range []string{
		"--backend cuda",                 // prefill+decode on the GPU HAL
		"--cpu-offload-experts",          // the ~424 GB MoE experts stay on host RAM
		"EP_RANKS",                       // >1 launches resident expert-parallel ranks instead
		"FAK_EP_RANK",                    // sharded rank identity for the EP serve
		"FAK_EP_COORD_ADDR",              // rank rendezvous for resident EP
		"FAK_EP_REQUIRE_DEVICE_PG",       // perf-grade EP refuses host DistComm fallback by default
		"FAK_EP_FANOUT_ADDRS",            // rank 0 mirrors a client request to follower ranks
		"FAK_KQ_INT8",                    // mixed Q5_K/Q6_K experts use the production int8 fallback
		"GLM_SMOKE_MAX_TOKENS",           // live readiness proves first-token decode, not an 8-token soak
		"SMOKE_EP",                       // EP readiness fans one request to every rank before the NCCL reduce
		"--expert-parallel",              // no-cpu-offload resident-expert topology
		"--context-budget-tokens",        // cap the KV plan (default 1M context => FitTooBig)
		"build_cuda.sh binary ./cmd/fak", // the canonical -tags cuda fak binary build
		"GLM52_FAK_NATIVE_SERVE_READY",   // the real-chat-completion health gate
	} {
		requireContainsForClaudeGLMGCP(t, serve, want)
	}
	bc := readRepoTextForClaudeGLMGCP(t, root, "internal", "compute", "build_cuda.sh")
	requireContainsForClaudeGLMGCP(t, bc, "binary)") // the DRY cuda-binary subcommand
}

func TestClaudeGLMGCPFakNativeServeStaysAttachedAfterReady(t *testing.T) {
	root := repoRootFromTest(t)
	serve := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_fak_native_serve.sh")
	idx := strings.Index(serve, "GLM52_FAK_NATIVE_SERVE_READY")
	if idx < 0 {
		t.Fatalf("ready marker missing from tools/glm52_fak_native_serve.sh")
	}
	afterReady := serve[idx:]
	for _, want := range []string{
		`wait "$SRV"`,
		`SERVER_EXITED rc=$rc`,
	} {
		requireContainsForClaudeGLMGCP(t, afterReady, want)
	}
	if strings.Contains(afterReady, `GLM52_FAK_NATIVE_SERVE_READY port=$PORT model=$MODEL_ID"; exit 0`) {
		t.Fatalf("ready path exits immediately instead of staying attached to the server")
	}
}

func TestClaudeGLMGCPResidentEPWitnessLaunchesShardedRanks(t *testing.T) {
	root := repoRootFromTest(t)
	witness := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_ep_witness.sh")
	for _, want := range []string{
		"LAUNCH_RANK rank=$r",
		`CUDA_VISIBLE_DEVICES="$gpu"`,
		`FAK_EP_RANK="$r"`,
		`FAK_EP_COORD_ADDR="$COORD_ADDR"`,
		`FAK_Q4K=1`,
		`--expert-parallel "$RANKS"`,
		"serve_ep_rank*.log",
	} {
		requireContainsForClaudeGLMGCP(t, witness, want)
	}
	if strings.Contains(witness, `CUDA_VISIBLE_DEVICES="$VIS" "$ROOT/fakbin_nccl" serve`) {
		t.Fatalf("EP witness still launches one monolithic process across all GPUs; it must launch sharded ranks")
	}
}

func TestClaudeGLMGCPSGLangServeUsesPython3Preflight(t *testing.T) {
	root := repoRootFromTest(t)
	serve := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_sglang_vllm_serve.sh")
	for _, want := range []string{
		`PYTHON_BIN="${PYTHON_BIN:-python3}"`,
		`SPECULATIVE="${SPECULATIVE:-1}"`,
		`if [ "${SPECULATIVE}" = "1" ]; then`,
		`log "speculative decode disabled (SPECULATIVE=${SPECULATIVE})"`,
		`if ! "${PYTHON_BIN}" "${HERE}/tools/glm52_serve_preflight.py"`,
		`dependency check: jinja2>=3.1.0`,
	} {
		requireContainsForClaudeGLMGCP(t, serve, want)
	}
	if strings.Contains(serve, `if ! python "${HERE}/tools/glm52_serve_preflight.py"`) {
		t.Fatalf("SGLang serve wrapper uses bare python; GCP images only provide python3")
	}
}

func TestClaudeGLMGCPSGLangServeStaysAttachedAfterReady(t *testing.T) {
	root := repoRootFromTest(t)
	serve := readRepoTextForClaudeGLMGCP(t, root, "tools", "glm52_sglang_vllm_serve.sh")
	idx := strings.Index(serve, `GLM52_${ENGINE}_READY`)
	if idx < 0 {
		t.Fatalf("ready marker missing from tools/glm52_sglang_vllm_serve.sh")
	}
	afterReady := serve[idx:]
	for _, want := range []string{
		`SMOKE="$(curl -sf -m 120`,
		`\"chat_template_kwargs\":{\"enable_thinking\":false}`,
		`\"max_tokens\":32`,
		`smoke completion failed; refusing READY`,
	} {
		requireContainsForClaudeGLMGCP(t, serve, want)
	}
	for _, want := range []string{
		`wait "${SRV_PID}"`,
		`SERVER_EXITED rc=${rc}`,
	} {
		requireContainsForClaudeGLMGCP(t, afterReady, want)
	}
	if strings.Contains(afterReady, `exit 0`) {
		t.Fatalf("SGLang ready path exits immediately instead of staying attached to the server")
	}
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
// script text (runs on every OS, no bash needed): it defaults to the 8x H100 Mega tier, forces the
// PURE FAK KERNEL serve (so the cache-value metric exists at all), composes the canonical
// bring-up rather than re-implementing it, and renders the probe -> cache-value -> teardown
// steps. This is the host-witnessable half of epic #1476 C1 (#1477).
func TestClaudeGLMGCPDemoPlanWiring(t *testing.T) {
	root := repoRootFromTest(t)
	demo := readRepoTextForClaudeGLMGCP(t, root, "scripts", "gcp-glm-demo.sh")
	for _, want := range []string{
		`GCP_TIER="${GCP_TIER:-a3-mega-h100}"`,      // the 8x H100 Mega demo tier (GLM-5.2 needs 640 GB)
		`SERVE="${SERVE:-fak}"`,                     // the PURE FAK KERNEL — the goal, and the metric's precondition
		`EP_RANKS="${EP_RANKS:-8}"`,                 // resident expert-parallel by default for the H100 demo
		`wait_for_remote_ready`,                     // apply waits for the VM-side service instead of deleting immediately
		`collect_remote_witness`,                    // apply preserves remote serve logs for the pure-kernel/device-PG claim
		`run_probe_turns`,                           // apply drives the headless turns itself
		"dogfood-claude.ps1",                        // WSL/Windows apply can use the native Claude Code runner
		`summarize_probe_perf`,                      // apply gates the "performant" part with probe duration evidence
		`scrape_cache_value`,                        // apply records the cache-value witness before teardown
		"scripts/gcp-glm-serve.sh",                  // composes the canonical bring-up, never re-implements it
		"claude-glm-gcp --probe",                    // step 2: the cache-warming probe turns
		"performance-summary.json",                  // step 3: the duration/throughput witness
		"fak_gateway_kv_prefix_reused_tokens_total", // step 3: the WITNESSED cache-value datum (#1010)
		"gcloud compute instances delete",           // step 4: teardown — the demo leaves zero cost
		`MODE="plan"`,                               // plan-by-default
	} {
		requireContainsForClaudeGLMGCP(t, demo, want)
	}
}

func TestClaudeGLMGCPDemoApplyDoesNotDeleteBeforeWitness(t *testing.T) {
	root := repoRootFromTest(t)
	demo := readRepoTextForClaudeGLMGCP(t, root, "scripts", "gcp-glm-demo.sh")
	for _, want := range []string{
		"wait_for_remote_ready",
		"collect_remote_witness",
		"start_tunnel",
		"run_probe_turns",
		"powershell.exe",
		"summarize_probe_perf",
		"scrape_cache_value",
		"REMOTE_FAIL phase=",
		"DEMO complete; witnesses copied under",
	} {
		requireContainsForClaudeGLMGCP(t, demo, want)
	}
	applyIdx := strings.Index(demo, `bash "$ROOT/scripts/gcp-glm-serve.sh" --apply`)
	if applyIdx < 0 {
		t.Fatalf("apply marker missing from demo apply path")
	}
	tail := demo[applyIdx:]
	waitIdx := strings.Index(tail, "wait_for_remote_ready")
	collectIdx := strings.Index(tail, `collect_remote_witness "ready"`)
	probeIdx := strings.Index(tail, "run_probe_turns")
	perfIdx := strings.Index(tail, "summarize_probe_perf")
	scrapeIdx := strings.Index(tail, "scrape_cache_value")
	doneIdx := strings.Index(tail, "DEMO complete; witnesses copied under")
	for name, idx := range map[string]int{"wait": waitIdx, "collect": collectIdx, "probe": probeIdx, "perf": perfIdx, "scrape": scrapeIdx, "done": doneIdx} {
		if idx < 0 {
			t.Fatalf("%s marker missing from demo apply path", name)
		}
	}
	if !(waitIdx < collectIdx && collectIdx < probeIdx && probeIdx < perfIdx && perfIdx < scrapeIdx && scrapeIdx < doneIdx) {
		t.Fatalf("demo apply ordering is wrong: apply=%d wait=%d collect=%d probe=%d perf=%d scrape=%d done=%d", applyIdx, waitIdx, collectIdx, probeIdx, perfIdx, scrapeIdx, doneIdx)
	}
	if strings.Contains(demo, "cache-value scrape + teardown is the operator's next step") {
		t.Fatalf("demo apply still exits after provisioning and leaves the witness as an operator next step")
	}
}

// TestClaudeGLMGCPDemoPlanRendersWithoutCreds runs the demo orchestrator with no creds and
// asserts the rendered plan resolves the 8x H100 Mega shape, forces the pure fak-kernel native
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
		"a3-megagpu-8g",                             // the 8x H100 Mega machine type — the default tier resolved
		"glm52_fak_native_serve.sh",                 // SERVE=fak forced the pure kernel even on sm_90
		"EP_RANKS=8",                                // the H100 demo uses resident expert-parallel, not cpu-offload
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
