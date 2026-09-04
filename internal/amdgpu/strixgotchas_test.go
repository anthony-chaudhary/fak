package amdgpu

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTop20GotchasCountAndIntegrity(t *testing.T) {
	gotchas := Top20Gotchas()
	if len(gotchas) != 20 {
		t.Fatalf("expected exactly 20 gotchas, got %d", len(gotchas))
	}

	seenIDs := make(map[string]bool)
	for i, g := range gotchas {
		if g.ID == "" {
			t.Errorf("gotcha %d has empty ID", i)
		}
		if seenIDs[g.ID] {
			t.Errorf("duplicate gotcha ID %s", g.ID)
		}
		seenIDs[g.ID] = true

		if g.Title == "" {
			t.Errorf("gotcha %s has empty Title", g.ID)
		}
		if g.Category == "" {
			t.Errorf("gotcha %s has empty Category", g.ID)
		}
		if g.Severity == "" {
			t.Errorf("gotcha %s has empty Severity", g.ID)
		}
		if g.Symptoms == "" {
			t.Errorf("gotcha %s has empty Symptoms", g.ID)
		}
		if g.RootCause == "" {
			t.Errorf("gotcha %s has empty RootCause", g.ID)
		}
		if g.Remediation == "" {
			t.Errorf("gotcha %s has empty Remediation", g.ID)
		}
	}
}

func TestAuditHostGotchas_SafeEnvironment(t *testing.T) {
	env := GotchaProbeEnvironment{
		GOOS:          "linux",
		KernelCmdline: "BOOT_IMAGE=/vmlinuz root=/dev/nvme0n1p2 amdgpu.lockup_timeout=-1 ttm.pages_limit=31457280 amdgpu.gttsize=131072",
		ProcCPUInfo:   "flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid bus_lock_detect fsrm amd_ppin brs avx512_vp2intersect zero_fex invlpgb Ryzen AI MAX+ 395",
		TotalRAMBytes: 128 * 1024 * 1024 * 1024,
		GPUName:       "AMD Radeon 8060S Graphics (gfx1151)",
		EnvVars: map[string]string{
			"OLLAMA_VULKAN":               "1",
			"OLLAMA_IGPU_ENABLE":          "1",
			"HIP_VISIBLE_DEVICES":         "-1",
			"TORCH_BLAS_PREFER_HIPBLASLT": "1",
		},
		SysfsLockupVal:   "-1",
		SysfsTTMPagesVal: 31457280,
		MesaVersion:      "26.1.7",
		KernelVersion:    "6.17.0-35-generic",
		IsStrixHalo:      true,
	}

	report := AuditHostGotchas(env)
	if report == nil {
		t.Fatal("expected non-nil report")
	}

	if report.DefectCount != 0 {
		t.Errorf("expected 0 defects in safe environment, got %d", report.DefectCount)
	}
	if !report.ReadyForInference {
		t.Errorf("expected ReadyForInference = true in safe environment")
	}

	summary := report.Summary()
	if !strings.Contains(summary, "STRIX HALO") {
		t.Errorf("summary missing title")
	}
	if !strings.Contains(summary, "Ready: true") {
		t.Errorf("summary missing Ready: true")
	}

	jsonData, err := report.ToJSON()
	if err != nil {
		t.Fatalf("report.ToJSON failed: %v", err)
	}
	var unmarshaled map[string]any
	if err := json.Unmarshal(jsonData, &unmarshaled); err != nil {
		t.Fatalf("json unmarshal failed: %v", err)
	}
	if unmarshaled["ready_for_inference"] != true {
		t.Errorf("expected ready_for_inference true in JSON")
	}
}

func TestAuditHostGotchas_DefectDetectionAndFixPlan(t *testing.T) {
	env := GotchaProbeEnvironment{
		GOOS:          "linux",
		KernelCmdline: "BOOT_IMAGE=/vmlinuz root=/dev/nvme0n1p2 quiet splash",
		ProcCPUInfo:   "model name: AMD Ryzen AI MAX+ 395",
		TotalRAMBytes: 128 * 1024 * 1024 * 1024,
		GPUName:       "AMD Radeon 8060S (gfx1151)",
		EnvVars: map[string]string{
			"HSA_OVERRIDE_GFX_VERSION":        "11.0.0", // Defect!
			"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",      // Defect!
		},
		SysfsLockupVal:   "10",               // Defect! (10s watchdog)
		SysfsTTMPagesVal: 0,                  // Defect! (kernel default 50% ceiling)
		MesaVersion:      "24.0.5",           // Defect! (outdated mesa)
		KernelVersion:    "7.0.0-28-generic", // Defect! (kfd deadlock kernel)
		IsStrixHalo:      true,
	}

	report := AuditHostGotchas(env)
	if report == nil {
		t.Fatal("expected non-nil report")
	}

	if report.DefectCount < 5 {
		t.Errorf("expected at least 5 defects detected, got %d", report.DefectCount)
	}
	if report.ReadyForInference {
		t.Errorf("expected ReadyForInference = false when defects are present")
	}

	fixes := GenerateFixPlan(report)
	if len(fixes) == 0 {
		t.Fatal("expected generated fixes for detected defects, got none")
	}

	fixJoined := strings.Join(fixes, "\n")
	if !strings.Contains(fixJoined, "amdgpu.lockup_timeout=-1") {
		t.Errorf("expected lockup_timeout fix in plan")
	}
	if !strings.Contains(fixJoined, "ttm.pages_limit=31457280") {
		t.Errorf("expected ttm.pages_limit fix in plan")
	}
	if !strings.Contains(fixJoined, "unset HSA_OVERRIDE_GFX_VERSION") {
		t.Errorf("expected HSA_OVERRIDE_GFX_VERSION fix in plan")
	}
	if !strings.Contains(fixJoined, "unset GGML_CUDA_ENABLE_UNIFIED_MEMORY") {
		t.Errorf("expected GGML_CUDA_ENABLE_UNIFIED_MEMORY fix in plan")
	}
	if !strings.Contains(fixJoined, "kisak-mesa") {
		t.Errorf("expected kisak-mesa fix in plan")
	}
}

func TestAuditHostGotchas_WindowsEnvironment(t *testing.T) {
	env := GotchaProbeEnvironment{
		GOOS:          "windows",
		TotalRAMBytes: 128 * 1024 * 1024 * 1024,
		GPUName:       "AMD Radeon 8060S Graphics",
		IsStrixHalo:   true,
		EnvVars:       map[string]string{},
	}

	report := AuditHostGotchas(env)
	if report == nil {
		t.Fatal("expected non-nil report")
	}

	// On Windows, Linux-specific gotchas should be NOT_APPLICABLE or ADVISORY
	for _, f := range report.Findings {
		if f.Gotcha.ID == "GOTCHA_BIOS_UMA_GTT" {
			if f.Status != StatusAdvisory {
				t.Errorf("expected GOTCHA_BIOS_UMA_GTT to be ADVISORY on Windows, got %s", f.Status)
			}
		}
	}
}

func TestAuditHostGotchas_ZeroAndBoundaryEnvironment(t *testing.T) {
	// Adversarial test: verify that zero/empty environment and nil maps do not panic
	env := GotchaProbeEnvironment{}

	report := AuditHostGotchas(env)
	if report == nil {
		t.Fatal("expected non-nil report for zero environment")
	}
	if len(report.Findings) != 20 {
		t.Fatalf("expected 20 findings for zero environment, got %d", len(report.Findings))
	}

	summary := report.Summary()
	if summary == "" {
		t.Errorf("expected non-empty summary for zero environment")
	}

	jsonData, err := report.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed on zero environment: %v", err)
	}
	if len(jsonData) == 0 {
		t.Errorf("expected non-empty JSON for zero environment")
	}
}

func TestRunGotchasCLI_Invocation(t *testing.T) {
	var stdout, stderr strings.Builder

	// Invalid args should return exit code 2
	code := RunGotchasCLI(&stdout, &stderr, []string{"extra_arg"})
	if code != 2 {
		t.Errorf("expected code 2 for unexpected args, got %d", code)
	}

	// Flag parse error should return exit code 2
	stderr.Reset()
	code = RunGotchasCLI(&stdout, &stderr, []string{"--nonexistent-flag"})
	if code != 2 {
		t.Errorf("expected code 2 for invalid flag, got %d", code)
	}
}
