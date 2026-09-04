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
		SysfsLockupVal:            "-1",
		SysfsTTMPagesVal:          31457280,
		SysfsVRAMTotalBytes:       2 * 1024 * 1024 * 1024,
		MesaVersion:               "26.1.7",
		KernelVersion:             "6.17.0-35-generic",
		IsStrixHalo:               true,
		HasOllamaProcess:          true,
		VulkanEngineType:          "engtype_3D",
		SpecDraftUbatchConfigured: true,
		BatchFlagsConfigured:      true,
		NPUOffloadEnabled:         true,
		DirtyRingBufferActive:     true,
		DPMGovernorConfigured:     true,
		USB4Tuned:                 true,
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

func TestAuditHostGotchas_StrixHalo64GB_TTM(t *testing.T) {
	// Issue #11249: Verify dynamic 64GB vs 128GB memory threshold scaling for TTM
	env64Safe := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    64 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 14680064, // ~56 GiB
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report64Safe := AuditHostGotchas(env64Safe)
	foundTTM := false
	for _, f := range report64Safe.Findings {
		if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
			foundTTM = true
			if f.Status != StatusSafeConfigured {
				t.Fatalf("expected 64GB system with 14.6M pages to be SAFE_CONFIGURED, got %s: %s", f.Status, f.Details)
			}
			if !strings.Contains(f.Details, "56 GiB") {
				t.Errorf("expected details to cite 56 GiB, got: %s", f.Details)
			}
		}
	}
	if !foundTTM {
		t.Fatal("GOTCHA_TTM_50PCT_CEILING not evaluated")
	}

	// 64GB system with default 50% limit (0 or low) should flag defect and suggest 56GB/14680064
	env64Defect := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    64 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 0,
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report64Defect := AuditHostGotchas(env64Defect)
	for _, f := range report64Defect.Findings {
		if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
			if f.Status != StatusDefectDetected {
				t.Fatalf("expected 64GB system with 0 pages limit to be DEFECT_DETECTED, got %s", f.Status)
			}
			if !strings.Contains(f.Details, "56 GiB") {
				t.Errorf("expected defect details to cite 56 GiB target, got: %s", f.Details)
			}
		}
	}
	fixes := GenerateFixPlan(report64Defect)
	fixText := strings.Join(fixes, "\n")
	if !strings.Contains(fixText, "ttm.pages_limit=14680064") {
		t.Errorf("expected 64GB fix plan to target 14680064 pages, got: %s", fixText)
	}

	// Dynamic scaling contrast: on a 128GB system, 14680064 pages (~56 GiB) is INSUFFICIENT (<30M threshold)
	// and must be flagged as DEFECT_DETECTED with 120 GiB (31457280) target.
	env128Insufficient := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    128 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 14680064,
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report128Insufficient := AuditHostGotchas(env128Insufficient)
	for _, f := range report128Insufficient.Findings {
		if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
			if f.Status != StatusDefectDetected {
				t.Fatalf("expected 128GB system with 14.6M pages to be DEFECT_DETECTED, got %s", f.Status)
			}
			if !strings.Contains(f.Details, "120 GiB") {
				t.Errorf("expected 128GB defect details to cite 120 GiB target, got: %s", f.Details)
			}
		}
	}
	fixes128 := GenerateFixPlan(report128Insufficient)
	fixText128 := strings.Join(fixes128, "\n")
	if !strings.Contains(fixText128, "ttm.pages_limit=31457280") {
		t.Errorf("expected 128GB fix plan to target 31457280 pages, got: %s", fixText128)
	}

	// 128GB system with full 31457280 pages should be SAFE_CONFIGURED and cite 120 GiB
	env128Safe := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    128 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 31457280,
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report128Safe := AuditHostGotchas(env128Safe)
	for _, f := range report128Safe.Findings {
		if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
			if f.Status != StatusSafeConfigured {
				t.Fatalf("expected 128GB system with 31.4M pages to be SAFE_CONFIGURED, got %s", f.Status)
			}
			if !strings.Contains(f.Details, "120 GiB") {
				t.Errorf("expected 128GB details to cite 120 GiB, got: %s", f.Details)
			}
		}
	}

	// 96GB system with full 23068672 pages (~88 GiB) should be SAFE_CONFIGURED and cite 88 GiB
	env96Safe := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    96 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 23068672,
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report96Safe := AuditHostGotchas(env96Safe)
	for _, f := range report96Safe.Findings {
		if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
			if f.Status != StatusSafeConfigured {
				t.Fatalf("expected 96GB system with 23M pages to be SAFE_CONFIGURED, got %s", f.Status)
			}
			if !strings.Contains(f.Details, "88 GiB") {
				t.Errorf("expected 96GB details to cite 88 GiB, got: %s", f.Details)
			}
		}
	}

	// 96GB system with 0 pages should suggest 88 GiB / 23068672
	env96Defect := GotchaProbeEnvironment{
		GOOS:             "linux",
		TotalRAMBytes:    96 * 1024 * 1024 * 1024,
		SysfsTTMPagesVal: 0,
		IsStrixHalo:      true,
		EnvVars:          map[string]string{},
	}
	report96Defect := AuditHostGotchas(env96Defect)
	fixes96 := GenerateFixPlan(report96Defect)
	fixText96 := strings.Join(fixes96, "\n")
	if !strings.Contains(fixText96, "ttm.pages_limit=23068672") {
		t.Errorf("expected 96GB fix plan to target 23068672 pages, got: %s", fixText96)
	}
}

func TestAuditHostGotchas_LiveGotchasEvaluations(t *testing.T) {
	// Issue #11250: Test live/falsifiable evaluation for the 8 gotchas

	t.Run("GOTCHA_OLLAMA_IGPU_FALLBACK", func(t *testing.T) {
		// Not installed/running -> NOT_APPLICABLE
		envNoOllama := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true}
		rep := AuditHostGotchas(envNoOllama)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_OLLAMA_IGPU_FALLBACK" {
				if f.Status != StatusNotApplicable {
					t.Errorf("expected NOT_APPLICABLE when Ollama not installed/running, got %s", f.Status)
				}
			}
		}

		// Ollama running but missing env vars -> DEFECT_DETECTED
		envOllamaDefect := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsStrixHalo:      true,
			HasOllamaProcess: true,
			EnvVars:          map[string]string{},
		}
		rep = AuditHostGotchas(envOllamaDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_OLLAMA_IGPU_FALLBACK" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when Ollama running without env vars, got %s", f.Status)
				}
			}
		}

		// Ollama running with env vars -> SAFE_CONFIGURED
		envOllamaSafe := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsStrixHalo:      true,
			HasOllamaProcess: true,
			EnvVars: map[string]string{
				"OLLAMA_VULKAN":      "1",
				"OLLAMA_IGPU_ENABLE": "1",
			},
		}
		rep = AuditHostGotchas(envOllamaSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_OLLAMA_IGPU_FALLBACK" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when Ollama properly exported, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_VULKAN_3D_ENGTYPE", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsStrixHalo:      true,
			VulkanEngineType: "engtype_Compute",
		}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_VULKAN_3D_ENGTYPE" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED for engtype_Compute, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsStrixHalo:      true,
			VulkanEngineType: "engtype_3D",
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_VULKAN_3D_ENGTYPE" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED for engtype_3D, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_SPEC_GRAPH_TIMEOUT", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, SpecDraftUbatchConfigured: false}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_SPEC_GRAPH_TIMEOUT" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when SpecDraftUbatchConfigured=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, SpecDraftUbatchConfigured: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_SPEC_GRAPH_TIMEOUT" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when SpecDraftUbatchConfigured=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_BATCH_CLAMP_SILENT", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, BatchFlagsConfigured: false}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_BATCH_CLAMP_SILENT" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when BatchFlagsConfigured=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, BatchFlagsConfigured: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_BATCH_CLAMP_SILENT" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when BatchFlagsConfigured=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_NPU_BANDWIDTH_CONTENTION", func(t *testing.T) {
		envAdvisory := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, NPUOffloadEnabled: false}
		rep := AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_NPU_BANDWIDTH_CONTENTION" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY when NPUOffloadEnabled=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, NPUOffloadEnabled: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_NPU_BANDWIDTH_CONTENTION" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when NPUOffloadEnabled=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_NVME_HIGH_WAF", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, DirtyRingBufferActive: false}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_NVME_HIGH_WAF" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when DirtyRingBufferActive=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, DirtyRingBufferActive: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_NVME_HIGH_WAF" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when DirtyRingBufferActive=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_THERMAL_CLOCK_HUNTING", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, DPMGovernorConfigured: false}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_THERMAL_CLOCK_HUNTING" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when DPMGovernorConfigured=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, DPMGovernorConfigured: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_THERMAL_CLOCK_HUNTING" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when DPMGovernorConfigured=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_USB4_CLUSTER_LATENCY", func(t *testing.T) {
		envAdvisory := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, USB4Tuned: false}
		rep := AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_USB4_CLUSTER_LATENCY" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY when USB4Tuned=false, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, USB4Tuned: true}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_USB4_CLUSTER_LATENCY" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when USB4Tuned=true, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_WC_CPU_READ_COLLAPSE", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, ProcCPUInfo: "flags: sse sse2"}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_WC_CPU_READ_COLLAPSE" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when CPU lacks avx512, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{GOOS: "linux", IsStrixHalo: true, ProcCPUInfo: "flags: avx512f"}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_WC_CPU_READ_COLLAPSE" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when CPU has avx512, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_GFX1151_OVERRIDE", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
		}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_GFX1151_OVERRIDE" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED for HSA_OVERRIDE_GFX_VERSION=11.0.0, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_GFX1151_OVERRIDE" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED for HSA_OVERRIDE_GFX_VERSION=11.5.1, got %s", f.Status)
				}
			}
		}

		envAdvisory := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{"HSA_OVERRIDE_GFX_VERSION": "12.0.0"},
		}
		rep = AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_GFX1151_OVERRIDE" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY for non-standard HSA_OVERRIDE_GFX_VERSION, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_GGML_UNIFIED_CORRUPT", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0"},
		}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_GGML_UNIFIED_CORRUPT" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED when GGML_CUDA_ENABLE_UNIFIED_MEMORY is defined, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{},
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_GGML_UNIFIED_CORRUPT" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when GGML_CUDA_ENABLE_UNIFIED_MEMORY is unset, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_IOMMU_DISABLE_SIDE_EFFECTS", func(t *testing.T) {
		envAdvisory := GotchaProbeEnvironment{
			GOOS:          "linux",
			IsStrixHalo:   true,
			KernelCmdline: "amd_iommu=off",
		}
		rep := AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_IOMMU_DISABLE_SIDE_EFFECTS" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY when amd_iommu=off, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:          "linux",
			IsStrixHalo:   true,
			KernelCmdline: "quiet splash",
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_IOMMU_DISABLE_SIDE_EFFECTS" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when IOMMU enabled, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_MESA_RADV_STALE", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			MesaVersion: "24.2.1",
		}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_MESA_RADV_STALE" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED for Mesa 24.2.1, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			MesaVersion: "25.3.0",
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_MESA_RADV_STALE" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED for Mesa 25.3.0, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_KERNEL7_KFD_DEADLOCK", func(t *testing.T) {
		envDefect := GotchaProbeEnvironment{
			GOOS:          "linux",
			IsStrixHalo:   true,
			KernelVersion: "7.0.0-28-generic",
		}
		rep := AuditHostGotchas(envDefect)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_KERNEL7_KFD_DEADLOCK" {
				if f.Status != StatusDefectDetected {
					t.Errorf("expected DEFECT_DETECTED for kernel 7.0.0-28, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:          "linux",
			IsStrixHalo:   true,
			KernelVersion: "6.17.0-35-generic",
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_KERNEL7_KFD_DEADLOCK" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED for kernel 6.17.0, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_CPU_STORM_INVLPGB", func(t *testing.T) {
		envAdvisory := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			ProcCPUInfo: "model name: Zen 5\nflags: avx512f",
		}
		rep := AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_CPU_STORM_INVLPGB" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY when invlpgb missing, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			ProcCPUInfo: "model name: Zen 5\nflags: avx512f invlpgb",
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_CPU_STORM_INVLPGB" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when invlpgb present, got %s", f.Status)
				}
			}
		}
	})

	t.Run("GOTCHA_VLLM_HIPBLASLT", func(t *testing.T) {
		envAdvisory := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{},
		}
		rep := AuditHostGotchas(envAdvisory)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_VLLM_HIPBLASLT" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected ADVISORY when TORCH_BLAS_PREFER_HIPBLASLT unset, got %s", f.Status)
				}
			}
		}

		envSafe := GotchaProbeEnvironment{
			GOOS:        "linux",
			IsStrixHalo: true,
			EnvVars:     map[string]string{"TORCH_BLAS_PREFER_HIPBLASLT": "1"},
		}
		rep = AuditHostGotchas(envSafe)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_VLLM_HIPBLASLT" {
				if f.Status != StatusSafeConfigured {
					t.Errorf("expected SAFE_CONFIGURED when TORCH_BLAS_PREFER_HIPBLASLT=1, got %s", f.Status)
				}
			}
		}
	})
}

func TestAuditHostGotchas_ContainerAndWSL2(t *testing.T) {
	// Issue #11247: Container and WSL2 detection and remediation isolation

	t.Run("ContainerEnvironment", func(t *testing.T) {
		env := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsContainer:      true,
			IsStrixHalo:      true,
			SysfsLockupVal:   "10", // would be defect on bare metal
			SysfsTTMPagesVal: 0,    // would be defect on bare metal
		}
		rep := AuditHostGotchas(env)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_RING_TIMEOUT" {
				if f.Status != StatusNotApplicable {
					t.Errorf("expected GOTCHA_RING_TIMEOUT to be NOT_APPLICABLE in container, got %s", f.Status)
				}
				if !strings.Contains(f.Details, "container") {
					t.Errorf("expected details to cite container, got: %s", f.Details)
				}
			}
			if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
				if f.Status != StatusNotApplicable {
					t.Errorf("expected GOTCHA_TTM_50PCT_CEILING to be NOT_APPLICABLE in container, got %s", f.Status)
				}
				if !strings.Contains(f.Details, "container") {
					t.Errorf("expected details to cite container, got: %s", f.Details)
				}
			}
		}

		// Fix plan must not emit update-grub for container
		fixes := GenerateFixPlan(rep)
		fixText := strings.Join(fixes, "\n")
		if strings.Contains(fixText, "update-grub") || strings.Contains(fixText, "grub-mkconfig") {
			t.Errorf("fix plan emitted bootloader update inside container environment: %s", fixText)
		}

		// Rollback script must not emit update-grub for container
		rollbacks := GenerateRollbackScript(rep)
		rbText := strings.Join(rollbacks, "\n")
		if strings.Contains(rbText, "update-grub") || strings.Contains(rbText, "grub-mkconfig") || strings.Contains(rbText, "/etc/default/grub") {
			t.Errorf("rollback emitted grub modifications inside container environment: %s", rbText)
		}

		// Container detection via cgroup when .dockerenv is absent
		mockCgroupFS := NewMockFS()
		mockCgroupFS.files["/proc/version"] = []byte("Linux version 6.17.0 (test@build) #1 SMP\n")
		mockCgroupFS.files["/proc/1/cgroup"] = []byte("0::/docker/e938dfbc1234\n")
		cgroupEnv := BuildHostProbeEnvironmentWithFS(mockCgroupFS)
		if !cgroupEnv.IsContainer {
			t.Errorf("expected container detection via /proc/1/cgroup")
		}
	})

	t.Run("WSL2Environment", func(t *testing.T) {
		env := GotchaProbeEnvironment{
			GOOS:             "linux",
			IsWSL2:           true,
			IsStrixHalo:      true,
			SysfsLockupVal:   "10",
			SysfsTTMPagesVal: 0,
		}
		rep := AuditHostGotchas(env)
		for _, f := range rep.Findings {
			if f.Gotcha.ID == "GOTCHA_RING_TIMEOUT" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected GOTCHA_RING_TIMEOUT to be ADVISORY in WSL2, got %s", f.Status)
				}
				if !strings.Contains(f.Details, "WSL2") || !strings.Contains(f.Details, ".wslconfig") {
					t.Errorf("expected details to explain WSL2 and .wslconfig, got: %s", f.Details)
				}
			}
			if f.Gotcha.ID == "GOTCHA_TTM_50PCT_CEILING" {
				if f.Status != StatusAdvisory {
					t.Errorf("expected GOTCHA_TTM_50PCT_CEILING to be ADVISORY in WSL2, got %s", f.Status)
				}
				if !strings.Contains(f.Details, "WSL2") || !strings.Contains(f.Details, ".wslconfig") {
					t.Errorf("expected details to explain WSL2 and .wslconfig, got: %s", f.Details)
				}
			}
		}

		fixes := GenerateFixPlan(rep)
		fixText := strings.Join(fixes, "\n")
		if strings.Contains(fixText, "update-grub") || strings.Contains(fixText, "grub-mkconfig") {
			t.Errorf("fix plan emitted bootloader update in WSL2 environment: %s", fixText)
		}

		rollbacks := GenerateRollbackScript(rep)
		rbText := strings.Join(rollbacks, "\n")
		if strings.Contains(rbText, "update-grub") || strings.Contains(rbText, "grub-mkconfig") || strings.Contains(rbText, "/etc/default/grub") {
			t.Errorf("rollback emitted grub modifications in WSL2 environment: %s", rbText)
		}
	})
}

func TestAuditHostGotchas_MultiDistroBootloaderAndPackageManager(t *testing.T) {
	// Issue #11248: Multi-distro bootloader and package manager support

	testCases := []struct {
		distro        string
		bootloaderCmd string
		mesaCmd       string
	}{
		{distro: "arch", bootloaderCmd: "grub-mkconfig -o /boot/grub/grub.cfg", mesaCmd: "pacman -S mesa vulkan-radeon"},
		{distro: "fedora", bootloaderCmd: "grub2-mkconfig -o /boot/grub2/grub.cfg", mesaCmd: "dnf upgrade mesa-dri-drivers"},
		{distro: "opensuse", bootloaderCmd: "grub2-mkconfig -o /boot/grub2/grub.cfg", mesaCmd: "zypper update Mesa"},
		{distro: "nixos", bootloaderCmd: "nixos-rebuild", mesaCmd: "pkgs.mesa"},
		{distro: "ubuntu", bootloaderCmd: "update-grub", mesaCmd: "ppa:kisak/kisak-mesa"},
	}

	for _, tc := range testCases {
		t.Run(tc.distro, func(t *testing.T) {
			rep := &GotchaAuditReport{
				Platform: "linux",
				DistroID: tc.distro,
				Findings: []GotchaAuditFinding{
					{
						Gotcha: StrixGotcha{ID: "GOTCHA_RING_TIMEOUT"},
						Status: StatusDefectDetected,
					},
					{
						Gotcha: StrixGotcha{ID: "GOTCHA_MESA_RADV_STALE"},
						Status: StatusDefectDetected,
					},
				},
			}
			fixes := GenerateFixPlan(rep)
			joined := strings.Join(fixes, "\n")
			if !strings.Contains(joined, tc.bootloaderCmd) {
				t.Errorf("distro %s: expected bootloader command %q in fix plan, got: %s", tc.distro, tc.bootloaderCmd, joined)
			}
			if !strings.Contains(joined, tc.mesaCmd) {
				t.Errorf("distro %s: expected mesa command %q in fix plan, got: %s", tc.distro, tc.mesaCmd, joined)
			}

			rollbacks := GenerateRollbackScript(rep)
			rbJoined := strings.Join(rollbacks, "\n")
			if tc.distro == "nixos" {
				if strings.Contains(rbJoined, "update-grub") || strings.Contains(rbJoined, "grub-mkconfig") {
					t.Errorf("distro nixos: expected no grub commands in rollback, got: %s", rbJoined)
				}
			} else {
				if !strings.Contains(rbJoined, tc.bootloaderCmd) {
					t.Errorf("distro %s: expected bootloader command %q in rollback, got: %s", tc.distro, tc.bootloaderCmd, rbJoined)
				}
			}
		})
	}
}

func TestAuditHostGotchas_LinuxVRAMCarveout(t *testing.T) {
	// Issue #11246: Linux VRAM BAR probing & BIOS UMA evaluation

	// Fixed carveout of 96 GiB on Linux should trigger defect
	envLargeCarveout := GotchaProbeEnvironment{
		GOOS:                "linux",
		IsStrixHalo:         true,
		SysfsVRAMTotalBytes: 96 * 1024 * 1024 * 1024,
	}
	rep := AuditHostGotchas(envLargeCarveout)
	for _, f := range rep.Findings {
		if f.Gotcha.ID == "GOTCHA_BIOS_UMA_GTT" {
			if f.Status != StatusDefectDetected {
				t.Fatalf("expected >=64GB VRAM carveout on Linux to be DEFECT_DETECTED, got %s", f.Status)
			}
			if !strings.Contains(f.Details, ">=64GB") {
				t.Errorf("expected details to mention >=64GB, got: %s", f.Details)
			}
		}
	}

	// Small carveout of 2 GiB on Linux should evaluate SAFE_CONFIGURED
	envSmallCarveout := GotchaProbeEnvironment{
		GOOS:                "linux",
		IsStrixHalo:         true,
		SysfsVRAMTotalBytes: 2 * 1024 * 1024 * 1024,
	}
	rep = AuditHostGotchas(envSmallCarveout)
	for _, f := range rep.Findings {
		if f.Gotcha.ID == "GOTCHA_BIOS_UMA_GTT" {
			if f.Status != StatusSafeConfigured {
				t.Fatalf("expected 2GB VRAM carveout on Linux to be SAFE_CONFIGURED, got %s", f.Status)
			}
		}
	}
}

func TestAuditHostGotchas_NonStrixHostFiltering(t *testing.T) {
	// Issue #11250: Non-Strix host filtering
	envNonStrix := GotchaProbeEnvironment{
		GOOS:        "linux",
		IsStrixHalo: false,
		GPUName:     "NVIDIA GeForce RTX 4090",
		ProcCPUInfo: "model name: 13th Gen Intel(R) Core(TM) i9-13900K",
	}
	rep := AuditHostGotchas(envNonStrix)
	if rep.DefectCount != 0 {
		t.Errorf("expected 0 defects on non-Strix host, got %d", rep.DefectCount)
	}
	for _, f := range rep.Findings {
		if f.Status != StatusNotApplicable {
			t.Errorf("expected all gotchas to be NOT_APPLICABLE on non-Strix host, got %s for %s", f.Status, f.Gotcha.ID)
		}
		if !strings.Contains(f.Details, "not AMD Strix Halo") {
			t.Errorf("expected non-Strix explanation in details, got: %s", f.Details)
		}
	}
}

func TestBuildHostProbeEnvironmentWithMockFS(t *testing.T) {
	// Issues #11246, #11247, #11248, #11250: Host probing using mock filesystem
	mockFS := NewMockFS()
	mockFS.files["/proc/version"] = []byte("Linux version 6.17.0-35-generic (buildd@lcy02-amd64-010) (gcc version 13.2.0) #35-Ubuntu SMP\n")
	mockFS.files["/proc/cmdline"] = []byte("BOOT_IMAGE=/vmlinuz root=/dev/nvme0n1p2 amdgpu.lockup_timeout=-1 ttm.pages_limit=31457280\n")
	mockFS.files["/proc/cpuinfo"] = []byte("model name: AMD Ryzen AI MAX+ 395 w/ Radeon 8060S\nflags: avx512f invlpgb\n")
	mockFS.files["/proc/meminfo"] = []byte("MemTotal:      134217728 kB\n")
	mockFS.files["/etc/os-release"] = []byte("ID=fedora\nVERSION_ID=41\n")
	mockFS.files["/sys/module/amdgpu/parameters/lockup_timeout"] = []byte("-1\n")
	mockFS.files["/sys/module/ttm/parameters/pages_limit"] = []byte("31457280\n")
	mockFS.files["/sys/class/drm/card0/device/mem_info_vram_total"] = []byte("2147483648\n")
	mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"] = []byte("high\n")
	mockFS.files["/usr/share/doc/mesa-vulkan-drivers/changelog.Debian.gz"] = []byte("mesa (25.3.1-1ubuntu1) noble; urgency=medium\n")
	mockFS.files["/.dockerenv"] = []byte("")
	mockFS.files["/usr/bin/ollama"] = []byte("")
	mockFS.files["/proc/1234/cmdline"] = []byte("ollama\x00serve\x00")

	env := BuildHostProbeEnvironmentWithFS(mockFS)
	if env.KernelVersion != "6.17.0-35-generic" {
		t.Errorf("expected KernelVersion '6.17.0-35-generic', got %q", env.KernelVersion)
	}
	if env.DistroID != "fedora" {
		t.Errorf("expected DistroID 'fedora', got %q", env.DistroID)
	}
	if env.SysfsVRAMTotalBytes != 2147483648 {
		t.Errorf("expected SysfsVRAMTotalBytes 2147483648, got %d", env.SysfsVRAMTotalBytes)
	}
	if env.MesaVersion != "25.3.1" {
		t.Errorf("expected MesaVersion '25.3.1', got %q", env.MesaVersion)
	}
	if !env.IsContainer {
		t.Errorf("expected IsContainer=true")
	}
	if !env.HasOllamaInstalled {
		t.Errorf("expected HasOllamaInstalled=true")
	}
	if !env.HasOllamaProcess {
		t.Errorf("expected HasOllamaProcess=true")
	}
	if !env.DPMGovernorConfigured {
		t.Errorf("expected DPMGovernorConfigured=true")
	}
	if !env.IsStrixHalo {
		t.Errorf("expected IsStrixHalo=true")
	}

	// Test VRAM BAR fallback via /sys/class/drm/card0/device/resource when mem_info_vram_total is absent
	mockBarFS := NewMockFS()
	mockBarFS.files["/proc/version"] = []byte("Linux version 6.17.0 (test@build) #1 SMP\n")
	// resource file format: start end flags (hex)
	// 512 MiB BAR: start 0x0000007fe0000000 end 0x0000007fffffffff (536870912 bytes)
	mockBarFS.files["/sys/class/drm/card0/device/resource"] = []byte("0x0000007fe0000000 0x0000007fffffffff 0x0014220c\n")
	envBar := BuildHostProbeEnvironmentWithFS(mockBarFS)
	if envBar.SysfsVRAMTotalBytes != 512*1024*1024 {
		t.Errorf("expected 512 MiB VRAM from resource fallback, got %d", envBar.SysfsVRAMTotalBytes)
	}

	// Test Mesa version probing via dpkg status
	mockDpkgFS := NewMockFS()
	mockDpkgFS.files["/proc/version"] = []byte("Linux version 6.17.0 (test@build) #1 SMP\n")
	mockDpkgFS.files["/var/lib/dpkg/status"] = []byte("Package: libgl1-mesa-dri\nStatus: install ok installed\nVersion: 26.0.2-1ubuntu1\n\nPackage: bash\nVersion: 5.2-1\n")
	envDpkg := BuildHostProbeEnvironmentWithFS(mockDpkgFS)
	if envDpkg.MesaVersion != "26.0.2" {
		t.Errorf("expected MesaVersion '26.0.2' from dpkg status, got %q", envDpkg.MesaVersion)
	}

	// Test Mesa version probing via pacman desc
	mockPacmanFS := NewMockFS()
	mockPacmanFS.files["/proc/version"] = []byte("Linux version 6.17.0 (test@build) #1 SMP\n")
	mockPacmanFS.files["/var/lib/pacman/local/mesa-25.2.0-1/desc"] = []byte("%NAME%\nmesa\n\n%VERSION%\n25.2.0-1\n")
	envPacman := BuildHostProbeEnvironmentWithFS(mockPacmanFS)
	if envPacman.MesaVersion != "25.2.0" {
		t.Errorf("expected MesaVersion '25.2.0' from pacman desc, got %q", envPacman.MesaVersion)
	}
}

func TestAuditHostGotchas_RollbackScriptAndBoundaryChecking(t *testing.T) {
	// Issue #11243: Rollback script generation & TTM page limit boundary checking

	rep := &GotchaAuditReport{
		Platform: "linux",
		DistroID: "ubuntu",
		Findings: []GotchaAuditFinding{
			{Gotcha: StrixGotcha{ID: "GOTCHA_RING_TIMEOUT"}, Status: StatusDefectDetected},
			{Gotcha: StrixGotcha{ID: "GOTCHA_TTM_50PCT_CEILING"}, Status: StatusDefectDetected},
			{Gotcha: StrixGotcha{ID: "GOTCHA_OLLAMA_IGPU_FALLBACK"}, Status: StatusDefectDetected},
			{Gotcha: StrixGotcha{ID: "GOTCHA_THERMAL_CLOCK_HUNTING"}, Status: StatusDefectDetected},
		},
	}
	rollbacks := GenerateRollbackScript(rep)
	if len(rollbacks) == 0 {
		t.Fatal("expected rollback commands, got none")
	}
	joined := strings.Join(rollbacks, "\n")
	if !strings.Contains(joined, "amdgpu.lockup_timeout") {
		t.Errorf("expected lockup_timeout removal in rollback, got: %s", joined)
	}
	if !strings.Contains(joined, "ttm.pages_limit") {
		t.Errorf("expected ttm.pages_limit removal in rollback, got: %s", joined)
	}
	if !strings.Contains(joined, "ollama.service.d/override.conf") {
		t.Errorf("expected ollama override removal in rollback, got: %s", joined)
	}
	if !strings.Contains(joined, "power_dpm_force_performance_level") {
		t.Errorf("expected DPM auto restore in rollback, got: %s", joined)
	}

	// Boundary check on 128GB system
	total128GB := uint64(128 * 1024 * 1024 * 1024)
	valid, err := ValidateTTMPagesLimit(31457280, total128GB) // ~120 GiB (valid)
	if !valid || err != nil {
		t.Errorf("expected 120 GiB on 128 GiB to be valid, got valid=%t, err=%v", valid, err)
	}

	valid, err = ValidateTTMPagesLimit(39321600, total128GB) // 150 GiB (> physical RAM)
	if valid || err == nil {
		t.Errorf("expected 150 GiB on 128 GiB to be invalid, got valid=%t, err=%v", valid, err)
	}

	valid, err = ValidateTTMPagesLimit(33030144, total128GB) // 126 GiB (leaves only 2 GiB reserve < 4 GiB)
	if valid || err == nil {
		t.Errorf("expected 126 GiB on 128 GiB (leaving 2GB reserve) to be invalid, got valid=%t, err=%v", valid, err)
	}

	// Boundary check on 64GB system
	total64GB := uint64(64 * 1024 * 1024 * 1024)
	valid, err = ValidateTTMPagesLimit(14680064, total64GB) // ~56 GiB (valid)
	if !valid || err != nil {
		t.Errorf("expected 56 GiB on 64 GiB to be valid, got valid=%t, err=%v", valid, err)
	}

	valid, err = ValidateTTMPagesLimit(18350080, total64GB) // ~70 GiB (> physical RAM)
	if valid || err == nil {
		t.Errorf("expected 70 GiB on 64 GiB to be invalid, got valid=%t, err=%v", valid, err)
	}

	valid, err = ValidateTTMPagesLimit(16252928, total64GB) // 62 GiB (leaves only 2 GiB reserve < 4 GiB)
	if valid || err == nil {
		t.Errorf("expected 62 GiB on 64 GiB (leaving 2GB reserve) to be invalid, got valid=%t, err=%v", valid, err)
	}

	// Boundary check with zero RAM (graceful pass)
	valid, err = ValidateTTMPagesLimit(1000, 0)
	if !valid || err != nil {
		t.Errorf("expected 0 total RAM to return valid=true, got valid=%t, err=%v", valid, err)
	}
}

func TestRunGotchasCLI_RollbackAndOutput(t *testing.T) {
	// Issue #11241 & #11243: Verify CLI --rollback and --fix-plan flags
	var stdout, stderr strings.Builder

	code := RunGotchasCLI(&stdout, &stderr, []string{"--rollback"})
	// Should execute and produce output
	if stdout.Len() == 0 && stderr.Len() == 0 {
		t.Errorf("expected CLI output with --rollback")
	}

	stdout.Reset()
	stderr.Reset()
	code = RunGotchasCLI(&stdout, &stderr, []string{"--json"})
	if code != 0 && code != 1 {
		t.Errorf("expected code 0 or 1, got %d", code)
	}
	var reportMap map[string]any
	if err := json.Unmarshal([]byte(stdout.String()), &reportMap); err != nil {
		t.Errorf("expected valid JSON output from CLI --json, err: %v", err)
	}
}

func TestAuditHostGotchas_ReadOnlySysfs(t *testing.T) {
	// Issue #11250: Mock sysfs handling read-only or empty files
	mockFS := NewMockFS()
	mockFS.files["/proc/version"] = []byte("Linux version 6.17.0 (test@build) #1 SMP\n")
	// No sysfs files created in mockFS; all ReadFile calls will fail with os.ErrNotExist
	env := BuildHostProbeEnvironmentWithFS(mockFS)
	rep := AuditHostGotchas(env)
	if rep == nil {
		t.Fatal("expected non-nil report for empty/read-only sysfs")
	}
	if len(rep.Findings) != 20 {
		t.Errorf("expected 20 findings, got %d", len(rep.Findings))
	}
}
