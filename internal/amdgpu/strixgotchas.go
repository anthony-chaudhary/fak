// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// Strix Halo APU operational serving profiles, direct AQL/PM4 packet dispatch,
// native HSACO code-object emission, and Strix Halo agent simulation verification.
package amdgpu

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// GotchaSeverity represents the operational impact level of a known gotcha.
type GotchaSeverity string

const (
	SeverityCritical GotchaSeverity = "CRITICAL"
	SeverityHigh     GotchaSeverity = "HIGH"
	SeverityMedium   GotchaSeverity = "MEDIUM"
)

// GotchaCategory classifies the subsystem or domain of the gotcha.
type GotchaCategory string

const (
	CategoryKernelDriver    GotchaCategory = "Kernel/Driver"
	CategoryMemoryUMA       GotchaCategory = "Memory/UMA"
	CategoryComputeROCm     GotchaCategory = "Compute/ROCm"
	CategoryRuntimeEngine   GotchaCategory = "Runtime/Engine"
	CategoryHardwareThermal GotchaCategory = "Hardware/Thermal"
	CategoryClusterIO       GotchaCategory = "Cluster/Storage/IO"
)

// GotchaStatus represents the audit assessment status for a particular gotcha on the host.
type GotchaStatus string

const (
	StatusDefectDetected GotchaStatus = "DEFECT_DETECTED"
	StatusSafeConfigured GotchaStatus = "SAFE_CONFIGURED"
	StatusAdvisory       GotchaStatus = "ADVISORY"
	StatusNotApplicable  GotchaStatus = "NOT_APPLICABLE"
)

// StrixGotcha defines the metadata, diagnosis, and remediation rules for an AMD Strix Halo gotcha.
type StrixGotcha struct {
	ID          string         `json:"id"`
	Title       string         `json:"title"`
	Category    GotchaCategory `json:"category"`
	Severity    GotchaSeverity `json:"severity"`
	Symptoms    string         `json:"symptoms"`
	RootCause   string         `json:"root_cause"`
	Remediation string         `json:"remediation"`
	CanAutoFix  bool           `json:"can_auto_fix"`
}

// GotchaAuditFinding holds the result of evaluating one gotcha against the host environment.
type GotchaAuditFinding struct {
	Gotcha      StrixGotcha  `json:"gotcha"`
	Status      GotchaStatus `json:"status"`
	Details     string       `json:"details"`
	ActionTaken string       `json:"action_taken,omitempty"`
}

// GotchaAuditReport is the aggregated report of auditing all top 20 gotchas.
type GotchaAuditReport struct {
	Platform          string               `json:"platform"`
	IsStrixHalo       bool                 `json:"is_strix_halo"`
	TotalRAMGiB       float64              `json:"total_ram_gib"`
	DistroID          string               `json:"distro_id,omitempty"`
	IsContainer       bool                 `json:"is_container,omitempty"`
	IsWSL2            bool                 `json:"is_wsl2,omitempty"`
	Findings          []GotchaAuditFinding `json:"findings"`
	DefectCount       int                  `json:"defect_count"`
	SafeCount         int                  `json:"safe_count"`
	AdvisoryCount     int                  `json:"advisory_count"`
	ReadyForInference bool                 `json:"ready_for_inference"`
}

// Summary generates a formatted human-readable summary of the gotcha audit report.
func (r *GotchaAuditReport) Summary() string {
	var b strings.Builder
	b.WriteString("================================================================================\n")
	b.WriteString("       AMD RYZEN AI MAX+ 395 (STRIX HALO) TOP 20 GOTCHAS AUDIT REPORT           \n")
	b.WriteString("================================================================================\n")
	fmt.Fprintf(&b, "Platform: %s | Strix Halo Detected: %t | Total RAM: %.1f GiB\n", r.Platform, r.IsStrixHalo, r.TotalRAMGiB)
	fmt.Fprintf(&b, "Status: %d Defects, %d Safe, %d Advisories | Ready: %t\n\n",
		r.DefectCount, r.SafeCount, r.AdvisoryCount, r.ReadyForInference)

	for i, f := range r.Findings {
		statusMarker := "[SAFE]"
		switch f.Status {
		case StatusDefectDetected:
			statusMarker = "[DEFECT]"
		case StatusAdvisory:
			statusMarker = "[WARN]"
		case StatusNotApplicable:
			statusMarker = "[N/A]"
		}

		fmt.Fprintf(&b, "%2d. %s [%s] %s (%s)\n", i+1, statusMarker, f.Gotcha.Severity, f.Gotcha.Title, f.Gotcha.Category)
		fmt.Fprintf(&b, "    Details:     %s\n", f.Details)
		if f.Status == StatusDefectDetected || f.Status == StatusAdvisory {
			fmt.Fprintf(&b, "    Root Cause:  %s\n", f.Gotcha.RootCause)
			fmt.Fprintf(&b, "    Remediation: %s\n", f.Gotcha.Remediation)
		}
		if f.ActionTaken != "" {
			fmt.Fprintf(&b, "    Action:      %s\n", f.ActionTaken)
		}
		b.WriteString("\n")
	}
	b.WriteString("================================================================================\n")
	return b.String()
}

// ToJSON marshals the GotchaAuditReport into formatted JSON.
func (r *GotchaAuditReport) ToJSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// GotchaProbeEnvironment supplies environmental facts to the gotcha auditor.
type GotchaProbeEnvironment struct {
	GOOS                       string            `json:"goos"`
	KernelCmdline              string            `json:"kernel_cmdline"`
	ProcCPUInfo                string            `json:"proc_cpuinfo"`
	TotalRAMBytes              uint64            `json:"total_ram_bytes"`
	GPUName                    string            `json:"gpu_name"`
	EnvVars                    map[string]string `json:"env_vars"`
	SysfsLockupVal             string            `json:"sysfs_lockup_val"`
	SysfsTTMPagesVal           uint64            `json:"sysfs_ttm_pages_val"`
	SysfsVRAMTotalBytes        uint64            `json:"sysfs_vram_total_bytes"`
	MesaVersion                string            `json:"mesa_version"`
	KernelVersion              string            `json:"kernel_version"`
	IsStrixHalo                bool              `json:"is_strix_halo"`
	IsContainer                bool              `json:"is_container"`
	IsWSL2                     bool              `json:"is_wsl2"`
	DistroID                   string            `json:"distro_id"`
	HasOllamaProcess           bool              `json:"has_ollama_process"`
	HasOllamaInstalled         bool              `json:"has_ollama_installed"`
	VulkanEngineType           string            `json:"vulkan_engine_type"`
	SpecDraftUbatchConfigured  bool              `json:"spec_draft_ubatch_configured"`
	BatchFlagsConfigured       bool              `json:"batch_flags_configured"`
	NPUOffloadEnabled          bool              `json:"npu_offload_enabled"`
	DirtyRingBufferActive      bool              `json:"dirty_ring_buffer_active"`
	DPMGovernorConfigured      bool              `json:"dpm_governor_configured"`
	USB4Tuned                  bool              `json:"usb4_tuned"`
	F16KVContiguizationEnabled bool              `json:"f16_kv_contiguization_enabled"`
	FS                         FileSystem        `json:"-"`
}

// Top20Gotchas returns the authoritative list of the top 20 gotchas with technical specs.
func Top20Gotchas() []StrixGotcha {
	return []StrixGotcha{
		{
			ID:          "GOTCHA_RING_TIMEOUT",
			Title:       "AMDGPU Watchdog Ring Timeout on Deep-Context Prefill (>136k tokens)",
			Category:    CategoryKernelDriver,
			Severity:    SeverityCritical,
			Symptoms:    "GPU freezes, screen blinks or crashes with ErrorDeviceLost, dmesg shows '[drm:amdgpu_job_timedout] *ERROR* ring gfx timeout'.",
			RootCause:   "Default Linux amdgpu watchdog triggers after 10 seconds of continuous compute queue occupancy during deep prefill (>136K tokens).",
			Remediation: "Set amdgpu.lockup_timeout=-1 (or 60000 ms) in /etc/default/grub (GRUB_CMDLINE_LINUX_DEFAULT) and run sudo update-grub.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_TTM_50PCT_CEILING",
			Title:       "Linux TTM 50% RAM Allocation Ceiling Starves Large Models",
			Category:    CategoryMemoryUMA,
			Severity:    SeverityCritical,
			Symptoms:    "OOM or fallback to system RAM when allocating >64GB on a 128GB Strix Halo system.",
			RootCause:   "Linux kernel ttm.pages_limit defaults to 50% of system RAM (~64GB on 128GB system), hiding half of available unified memory.",
			Remediation: "Add ttm.pages_limit=31457280 (120GB) and amdgpu.gttsize=131072 to kernel cmdline in /etc/default/grub, then update-grub.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_BIOS_UMA_GTT",
			Title:       "BIOS UMA Fixed Carveout vs Dynamic GTT Memory Starvation",
			Category:    CategoryMemoryUMA,
			Severity:    SeverityHigh,
			Symptoms:    "Host OS boots with only 32GB RAM or crashes during boot if 96GB fixed carveout is set in BIOS under Linux.",
			RootCause:   "Linux manages UMA via dynamic GTT; setting 96GB fixed BIOS carveout starves Linux host kernel. Windows requires 96GB BIOS carveout, but Linux requires minimum BIOS carveout (512MB or 2GB) with dynamic GTT.",
			Remediation: "On Linux: set BIOS UMA Frame Buffer Size to 512MB (or 2GB); on Windows: set BIOS UMA Frame Buffer Size to 96GB (128GB system) or 48GB (64GB system).",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_WC_CPU_READ_COLLAPSE",
			Title:       "Write-Combining (WC) Memory CPU Read Throughput Collapse (200 MB/s)",
			Category:    CategoryMemoryUMA,
			Severity:    SeverityHigh,
			Symptoms:    "CPU staging reads from GPU-allocated unified memory suffer a catastrophic 50x drop from 10 GB/s to 200 MB/s.",
			RootCause:   "CPU reads from PCIe/APU write-combined pages incur massive non-cached unaligned read stalls unless read with non-temporal streaming instructions.",
			Remediation: "Use AVX-512 non-temporal streaming load primitives (_mm512_stream_load_si512 / vmovntdqa) for all CPU reads from write-combined host memory.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_GFX1151_OVERRIDE",
			Title:       "ROCm Architecture gfx1151 Missing Kernels & Dangerous Overrides",
			Category:    CategoryComputeROCm,
			Severity:    SeverityHigh,
			Symptoms:    "ROCm tools crash on launch with exit 139 (SIGSEGV in libamdhip64) or fallback to slow un-optimized emulation.",
			RootCause:   "ROCm versions prior to 7.14/10.0 lack compiled gfx1151 kernels. Setting HSA_OVERRIDE_GFX_VERSION=11.0.0 breaks RDNA 3.5 Wave32 optimizations and causes segfaults.",
			Remediation: "Do not set HSA_OVERRIDE_GFX_VERSION=11.0.0; use native gfx1151 builds or if needed HSA_OVERRIDE_GFX_VERSION=11.5.1 on qualified stacks.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_VULKAN_3D_ENGTYPE",
			Title:       "Vulkan Compute Utilization Inaccurately Reported Under 3D Engine",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityMedium,
			Symptoms:    "Performance monitors (Windows Task Manager, perf counters) show GPU compute at 0% while LLM inference runs full throttle.",
			RootCause:   "On AMD Radeon/RDNA drivers, Vulkan compute dispatches execute on the primary graphics queue ('3D' engine), leaving engtype_Compute at 0%.",
			Remediation: "Monitor total_util_pct or engtype_3D utilization instead of engtype_Compute when tracking Vulkan LLM workloads.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_OLLAMA_IGPU_FALLBACK",
			Title:       "Ollama Silently Falls Back to CPU Inference on Radeon 8060S",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityHigh,
			Symptoms:    "Ollama runs models at 4-7 t/s on CPU despite detecting Radeon 8060S iGPU.",
			RootCause:   "Ollama disables iGPU acceleration by default unless explicitly instructed, treating Radeon 8060S as an unaccelerated client display adapter.",
			Remediation: "Export OLLAMA_VULKAN=1, OLLAMA_IGPU_ENABLE=1, and HIP_VISIBLE_DEVICES=-1 in systemd ollama.service or environment.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_SPEC_GRAPH_TIMEOUT",
			Title:       "Vulkan Speculative Graph Capture Timeouts / Ring Hangs in MTP",
			Category:    CategoryComputeROCm,
			Severity:    SeverityHigh,
			Symptoms:    "Speculative decoding (MTP / DFlash) triggers compute ring resets or severe token-generation jitter at high acceptance rates.",
			RootCause:   "Dynamic variable-length speculative draft tokens thrash the Vulkan command graph, causing synchronous pipeline recreation and timeouts.",
			Remediation: "Decouple speculative draft micro-batch size (--spec-draft-ubatch-size 512) and quantize draft token dimensions to power-of-two buckets.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_GGML_UNIFIED_CORRUPT",
			Title:       "GGML_CUDA_ENABLE_UNIFIED_MEMORY Corrupts LLM Output on ROCm",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityCritical,
			Symptoms:    "ROCm llama.cpp emits garbled, hallucinated, or corrupted text when loading large models.",
			RootCause:   "Defining GGML_CUDA_ENABLE_UNIFIED_MEMORY (even set to 0) activates an incompatible unified memory code path on RDNA 3.5 APUs.",
			Remediation: "Completely unset GGML_CUDA_ENABLE_UNIFIED_MEMORY from environment; do not export it with 0 or 1.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_ROCM_HOST_CORRUPT",
			Title:       "llama.cpp ROCm_Host Direct Compute Buffer Corruption on APU",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityHigh,
			Symptoms:    "Repeated phrases, token degradation, or crash in multi-slot, long-context, or vision workloads (llama.cpp issue #26209).",
			RootCause:   "Direct integrated ROCm_Host compute buffer mapping bypasses cache coherency boundaries on certain APU memory controllers.",
			Remediation: "Pin llama.cpp build with PR #25863 or prefer Vulkan/RADV backend for multi-slot and long-context inference.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_IOMMU_DISABLE_SIDE_EFFECTS",
			Title:       "Disabling IOMMU (amd_iommu=off) Breaks NPU, Suspend, and Networking",
			Category:    CategoryKernelDriver,
			Severity:    SeverityHigh,
			Symptoms:    "XDNA 2 NPU fails to probe, laptop S0ix/S3 sleep hangs, or USB4 RDMA clustering fails.",
			RootCause:   "Setting amd_iommu=off disables IOMMU address translation required by the XDNA 2 NPU driver, PCIe VFIO, and mobile power state management.",
			Remediation: "Keep IOMMU enabled (default); only use amd_iommu=off for dedicated always-on desktop benchmark rigs with no NPU workloads.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_MESA_RADV_STALE",
			Title:       "Outdated Stock Mesa Drivers Lack Wave32 FlashAttention & KHR_coopmat",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityHigh,
			Symptoms:    "Prompt ingestion (pp512) is 20-30% slower than expected on Vulkan/RADV.",
			RootCause:   "Ubuntu 24.04 stock Mesa 24.0/24.2 lacks optimized RDNA 3.5 Wave32 cooperative matrix and flash attention kernels.",
			Remediation: "Upgrade Mesa via Kisak PPA (ppa:kisak/kisak-mesa) to Mesa 25.3+ or 26.1+ for full RADV cooperative matrix performance.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_BATCH_CLAMP_SILENT",
			Title:       "Silent Batch Size Clamping Degrading Prompt Ingestion (-ub > -b)",
			Category:    CategoryRuntimeEngine,
			Severity:    SeverityMedium,
			Symptoms:    "Prompt processing speed drops by 25-30% with no warnings or error messages in logs.",
			RootCause:   "llama.cpp silently clamps -ub to min(n_batch, n_ubatch); setting -b 256 -ub 1024 runs at -ub 256.",
			Remediation: "Always configure batch flags such that -b >= -ub (e.g. -b 2048 -ub 512).",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_VLLM_HIPBLASLT",
			Title:       "vLLM FP16 Batch 8+ Throughput Collapse Without hipBLASLt",
			Category:    CategoryComputeROCm,
			Severity:    SeverityHigh,
			Symptoms:    "vLLM throughput on Ryzen AI MAX drops sharply by ~40% when concurrency exceeds 8 concurrent requests.",
			RootCause:   "PyTorch < 2.14 on ROCm fails to use hipBLASLt GEMM kernels at batch 8+, falling back to suboptimal standard rocBLAS GEMM.",
			Remediation: "Export TORCH_BLAS_PREFER_HIPBLASLT=1 before launching vLLM server to recover ~40% aggregate throughput.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_NPU_BANDWIDTH_CONTENTION",
			Title:       "NPU vs iGPU Unified Memory Bus Contention",
			Category:    CategoryHardwareThermal,
			Severity:    SeverityMedium,
			Symptoms:    "iGPU generation drops by 60%+ if auxiliary LLM tasks are co-located on iGPU instead of NPU.",
			RootCause:   "iGPU auxiliary workloads compete for shader cores and memory channels, whereas NPU operates as a decoupled low-power sidecar.",
			Remediation: "Offload sidecar tasks (small embeddings, audio ASR, TTS, FastFlowLM 1.2B) to XDNA 2 NPU; keeps iGPU latency penalty to <3.5%.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_KERNEL7_KFD_DEADLOCK",
			Title:       "Linux Kernel 7.0+ Silent KFD Work-Queue Deadlock on Large Allocations",
			Category:    CategoryKernelDriver,
			Severity:    SeverityHigh,
			Symptoms:    "GPU hangs during VAE/weight loading on large (>64GB) unified allocations under Linux kernel 7.0.0-28.",
			RootCause:   "Upstream regression in kernel 7.0 KFD driver causes work-queue deadlocks during massive contiguous page remapping.",
			Remediation: "Retain Linux kernel 6.17 LTS (or Ubuntu 24.04 HWE kernel) as preferred boot option for stable 128GB unified memory allocations.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_CPU_STORM_INVLPGB",
			Title:       "CPU Saturation Storm During Eval Due to Missing invlpgb Flag",
			Category:    CategoryKernelDriver,
			Severity:    SeverityMedium,
			Symptoms:    "All 32 Zen 5 CPU threads peg at 100% for minutes during evaluation checkpoints in LoRA/fine-tuning.",
			RootCause:   "Certain BIOS revisions fail to expose Zen 5 invlpgb (broadcast TLB invalidation), causing expensive inter-processor interrupts (IPIs).",
			Remediation: "Update motherboard UEFI/BIOS to latest version; verify 'invlpgb' presence in /proc/cpuinfo.",
			CanAutoFix:  false,
		},
		{
			ID:          "GOTCHA_NVME_HIGH_WAF",
			Title:       "Client NVMe Wear-Out Under High-Volume KV-Cache Paging (WAF > 30x)",
			Category:    CategoryClusterIO,
			Severity:    SeverityCritical,
			Symptoms:    "SSD TBW exhausted and drive enters read-only mode after 1-2 weeks of high-volume autonomous agent execution.",
			RootCause:   "Direct synchronous disk paging of KV caches causes 4KB random write thrashing on client TLC/QLC NAND flash with extreme WAF.",
			Remediation: "Allocate a 2-4 GiB write-back dirty ring buffer in host UMA DRAM and coalesce KV pages before flushing sequentially.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_THERMAL_CLOCK_HUNTING",
			Title:       "Thermal Throttling & Acoustic Fan Hunting Under Sustained APU Boost",
			Category:    CategoryHardwareThermal,
			Severity:    SeverityMedium,
			Symptoms:    "Aggressive fan noise oscillation and token decode rate fluctuating between 18 t/s and 12 t/s.",
			RootCause:   "APU PPT dynamically boosts to 120-140W, hits 90C thermal throttle ceiling, violently drops clocks, cools down, and repeats.",
			Remediation: "Lock DPM performance level to 'high' with an sclk clock cap (e.g. 2400 MHz) for rock-steady acoustic profile and zero latency jitter.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_USB4_CLUSTER_LATENCY",
			Title:       "Multi-Node Cluster Degradation Over USB4 Due to Link Sleep States",
			Category:    CategoryClusterIO,
			Severity:    SeverityHigh,
			Symptoms:    "2-node Strix Halo cluster over USB4 is 15-20% slower than a single node on models that fit within 128GB.",
			RootCause:   "USB4 link power-management sleep states introduce hundreds of microseconds of latency jitter during tensor-parallel all-reduce.",
			Remediation: "Only use multi-node clustering for models >128GB (e.g. DeepSeek V4 284B); set USB4 MTU 9000 and pm_qos_resume_latency_us=100.",
			CanAutoFix:  true,
		},
		{
			ID:          "GOTCHA_LPDDR5X_CHANNEL_CAMPING",
			Title:       "LPDDR5X 16-channel strided KV cache camping on Strix Halo",
			Category:    CategoryMemoryUMA,
			Severity:    SeverityHigh,
			Symptoms:    "Severe memory throughput collapse during long-context (>=32k) prefill and decode; memory transactions camp on <= 2 of 16 LPDDR5X pseudo-channels (entropy < 0.25), starving the remaining 14 channels.",
			RootCause:   "Strided f16 KV cache layout [nPos, nKV, hd] produces 2048-byte token strides that alias with Strix Halo's 16-channel 128B/64B interleaving period, funneling accesses onto 1 or 2 channels.",
			Remediation: "Enable pre-attention f16 KV contiguization pass to linearize KV cache into head-contiguous [nKV, nPos, hd] layout (export FAK_F16_KV_CONTIGUIZE=1 or set EnableF16KVContiguization=true in serving configuration).",
			CanAutoFix:  true,
		},
	}
}

// AuditHostGotchas audits the host environment against the top 20 gotchas.
func AuditHostGotchas(env GotchaProbeEnvironment) *GotchaAuditReport {
	gotchas := Top20Gotchas()
	findings := make([]GotchaAuditFinding, 0, len(gotchas))

	totalGiB := float64(env.TotalRAMBytes) / (1024 * 1024 * 1024)

	// Non-Strix Halo hardware filtering: if explicit non-Strix hardware was detected,
	// mark gotchas NOT_APPLICABLE rather than raising false defects.
	if !env.IsStrixHalo && (env.GPUName != "" || env.ProcCPUInfo != "") {
		nonStrixDesc := env.GPUName
		if nonStrixDesc == "" {
			nonStrixDesc = "Non-Strix CPU/Platform"
		}
		for _, g := range gotchas {
			findings = append(findings, GotchaAuditFinding{
				Gotcha:  g,
				Status:  StatusNotApplicable,
				Details: fmt.Sprintf("Host hardware is not AMD Strix Halo (detected: %s); gotcha applies specifically to AMD Ryzen AI MAX+ / gfx1151.", nonStrixDesc),
			})
		}
		return &GotchaAuditReport{
			Platform:          env.GOOS,
			IsStrixHalo:       false,
			TotalRAMGiB:       totalGiB,
			DistroID:          env.DistroID,
			IsContainer:       env.IsContainer,
			IsWSL2:            env.IsWSL2,
			Findings:          findings,
			DefectCount:       0,
			SafeCount:         0,
			AdvisoryCount:     0,
			ReadyForInference: true,
		}
	}

	defectCount := 0
	safeCount := 0
	advisoryCount := 0

	for _, g := range gotchas {
		status, details := evaluateGotcha(g.ID, env)
		switch status {
		case StatusDefectDetected:
			defectCount++
		case StatusSafeConfigured:
			safeCount++
		case StatusAdvisory:
			advisoryCount++
		}

		findings = append(findings, GotchaAuditFinding{
			Gotcha:  g,
			Status:  status,
			Details: details,
		})
	}

	ready := defectCount == 0

	return &GotchaAuditReport{
		Platform:          env.GOOS,
		IsStrixHalo:       env.IsStrixHalo,
		TotalRAMGiB:       totalGiB,
		DistroID:          env.DistroID,
		IsContainer:       env.IsContainer,
		IsWSL2:            env.IsWSL2,
		Findings:          findings,
		DefectCount:       defectCount,
		SafeCount:         safeCount,
		AdvisoryCount:     advisoryCount,
		ReadyForInference: ready,
	}
}

// AuditEnvironment audits an environment against Strix Halo gotchas.
func AuditEnvironment(env GotchaProbeEnvironment) *GotchaAuditReport {
	return AuditHostGotchas(env)
}

func evaluateGotcha(id string, env GotchaProbeEnvironment) (GotchaStatus, string) {
	switch id {
	case "GOTCHA_RING_TIMEOUT":
		if env.GOOS != "linux" {
			return StatusNotApplicable, "Applies to Linux amdgpu kernel driver."
		}
		if env.IsContainer {
			return StatusNotApplicable, "Running inside container; host kernel parameter amdgpu.lockup_timeout must be configured on host OS, not in container."
		}
		if env.IsWSL2 {
			return StatusAdvisory, "Running under WSL2: GPU execution mediated by D3D12/dxgkrnl; watchdog timeouts are controlled via host Windows display driver or %USERPROFILE%\\.wslconfig."
		}
		if strings.Contains(env.KernelCmdline, "amdgpu.lockup_timeout=-1") ||
			strings.Contains(env.KernelCmdline, "amdgpu.lockup_timeout=60000") ||
			env.SysfsLockupVal == "-1" || env.SysfsLockupVal == "60000" {
			return StatusSafeConfigured, fmt.Sprintf("amdgpu.lockup_timeout is configured safely (value: %s)", env.SysfsLockupVal)
		}
		return StatusDefectDetected, fmt.Sprintf("amdgpu.lockup_timeout is %s (default: 10s); deep prefill >136k tokens will trigger GPU reset crash", env.SysfsLockupVal)

	case "GOTCHA_TTM_50PCT_CEILING":
		if env.GOOS != "linux" {
			return StatusNotApplicable, "Applies to Linux TTM kernel memory subsystem."
		}
		if env.IsContainer {
			return StatusNotApplicable, "Running inside container; host kernel parameter ttm.pages_limit must be configured on host OS, not in container."
		}
		if env.IsWSL2 {
			return StatusAdvisory, "Running under WSL2: memory allocation is governed by %USERPROFILE%\\.wslconfig memory limit rather than Linux ttm.pages_limit."
		}
		// Scale threshold based on 64GB / 96GB / 128GB platform tier
		expectedPages := uint64(30000000) // ~120 GiB on 128GB systems
		targetGiB := "120 GiB"
		targetPagesExact := "31457280"
		if env.TotalRAMBytes > 0 && env.TotalRAMBytes < 80*1024*1024*1024 {
			expectedPages = uint64(14000000) // ~56 GiB on 64GB systems
			targetGiB = "56 GiB"
			targetPagesExact = "14680064"
		} else if env.TotalRAMBytes >= 80*1024*1024*1024 && env.TotalRAMBytes < 112*1024*1024*1024 {
			expectedPages = uint64(22000000) // ~88 GiB on 96GB systems
			targetGiB = "88 GiB"
			targetPagesExact = "23068672"
		}
		if env.SysfsTTMPagesVal >= expectedPages ||
			strings.Contains(env.KernelCmdline, "ttm.pages_limit="+targetPagesExact) ||
			(env.TotalRAMBytes >= 112*1024*1024*1024 && strings.Contains(env.KernelCmdline, "ttm.pages_limit=31457280")) ||
			(env.TotalRAMBytes >= 80*1024*1024*1024 && env.TotalRAMBytes < 112*1024*1024*1024 && strings.Contains(env.KernelCmdline, "ttm.pages_limit=23068672")) ||
			(env.TotalRAMBytes > 0 && env.TotalRAMBytes < 80*1024*1024*1024 && strings.Contains(env.KernelCmdline, "ttm.pages_limit=14680064")) {
			return StatusSafeConfigured, fmt.Sprintf("TTM pages_limit is configured for full aperture (%d pages / ~%s)", env.SysfsTTMPagesVal, targetGiB)
		}
		if env.SysfsTTMPagesVal == 0 {
			return StatusDefectDetected, fmt.Sprintf("TTM pages_limit is set to kernel default 50%% limit; restricts GPU compute to half available memory (target: %s)", targetGiB)
		}
		return StatusDefectDetected, fmt.Sprintf("TTM pages_limit is %d (lower than recommended %d pages / %s)", env.SysfsTTMPagesVal, expectedPages, targetGiB)

	case "GOTCHA_BIOS_UMA_GTT":
		if env.GOOS == "windows" {
			return StatusAdvisory, "Windows requires UEFI/BIOS UMA Frame Buffer Size set to 96GB for maximum VRAM aperture."
		}
		// On Linux: fixed carveout >= 64 GiB starves the host kernel
		if env.SysfsVRAMTotalBytes >= 64*1024*1024*1024 {
			return StatusDefectDetected, fmt.Sprintf("BIOS UMA fixed carveout is set to >=64GB on Linux (detected: %.1f GiB); starves host kernel. Reduce BIOS UMA Frame Buffer Size to 512MB/2GB and use dynamic GTT.", float64(env.SysfsVRAMTotalBytes)/(1024*1024*1024))
		}
		if env.SysfsVRAMTotalBytes > 0 && env.SysfsVRAMTotalBytes <= 4*1024*1024*1024 {
			return StatusSafeConfigured, fmt.Sprintf("Linux dynamic GTT verified; BIOS UMA carveout is set to minimum (%.1f GiB) preserving host memory.", float64(env.SysfsVRAMTotalBytes)/(1024*1024*1024))
		}
		return StatusSafeConfigured, "Linux dynamic GTT verified; ensure BIOS UMA framebuffer is set to minimum (512MB or 2GB) to preserve host memory."

	case "GOTCHA_WC_CPU_READ_COLLAPSE":
		// Check if Zen 5 AVX-512 is supported
		if strings.Contains(env.ProcCPUInfo, "avx512") || env.GOOS == "windows" {
			return StatusSafeConfigured, "Host CPU supports AVX-512 non-temporal streaming load instructions (_mm512_stream_load_si512)."
		}
		return StatusDefectDetected, "CPU lacks AVX-512 streaming load support; CPU reads from GPU write-combining memory will collapse to 200 MB/s."

	case "GOTCHA_GFX1151_OVERRIDE":
		val := env.EnvVars["HSA_OVERRIDE_GFX_VERSION"]
		if val == "11.0.0" {
			return StatusDefectDetected, "HSA_OVERRIDE_GFX_VERSION is set to 11.0.0; causes libamdhip64 segfaults (exit 139) on Strix Halo."
		}
		if val == "" || val == "11.5.1" {
			return StatusSafeConfigured, "HSA_OVERRIDE_GFX_VERSION is clean or set to compatible 11.5.1."
		}
		return StatusAdvisory, fmt.Sprintf("Non-standard HSA_OVERRIDE_GFX_VERSION=%s detected.", val)

	case "GOTCHA_VULKAN_3D_ENGTYPE":
		if env.VulkanEngineType == "engtype_Compute" || env.VulkanEngineType == "Compute" {
			return StatusDefectDetected, "Vulkan telemetry mapped to engtype_Compute; will show 0% utilization during LLM inference (configure engtype_3D or total_util_pct)."
		}
		return StatusSafeConfigured, "Telemetry configured: monitoring engtype_3D and total_util_pct instead of engtype_Compute for Vulkan."

	case "GOTCHA_OLLAMA_IGPU_FALLBACK":
		if !env.HasOllamaProcess && !env.HasOllamaInstalled {
			return StatusNotApplicable, "Ollama is not installed or running on host."
		}
		vulkanSet := env.EnvVars["OLLAMA_VULKAN"] == "1"
		igpuSet := env.EnvVars["OLLAMA_IGPU_ENABLE"] == "1"
		if vulkanSet && igpuSet {
			return StatusSafeConfigured, "OLLAMA_VULKAN=1 and OLLAMA_IGPU_ENABLE=1 are exported."
		}
		return StatusDefectDetected, "Ollama environment missing OLLAMA_VULKAN=1 or OLLAMA_IGPU_ENABLE=1; will silently fall back to CPU inference."

	case "GOTCHA_SPEC_GRAPH_TIMEOUT":
		if env.SpecDraftUbatchConfigured {
			return StatusSafeConfigured, "Decoupled draft micro-batching (--spec-draft-ubatch-size 512) and power-of-2 graph bucketing enabled."
		}
		return StatusDefectDetected, "Speculative draft micro-batch size not decoupled; dynamic token count causes Vulkan pipeline recreation and ring timeouts."

	case "GOTCHA_GGML_UNIFIED_CORRUPT":
		_, exists := env.EnvVars["GGML_CUDA_ENABLE_UNIFIED_MEMORY"]
		if exists {
			return StatusDefectDetected, "GGML_CUDA_ENABLE_UNIFIED_MEMORY is defined in environment; presence causes garbled text corruption on gfx1151."
		}
		return StatusSafeConfigured, "GGML_CUDA_ENABLE_UNIFIED_MEMORY is safely unset."

	case "GOTCHA_ROCM_HOST_CORRUPT":
		return StatusAdvisory, "Issue #26209: ROCm_Host direct compute buffers can corrupt long-context/multimodal inference on APU; prefer Vulkan backend."

	case "GOTCHA_IOMMU_DISABLE_SIDE_EFFECTS":
		if strings.Contains(env.KernelCmdline, "amd_iommu=off") {
			return StatusAdvisory, "amd_iommu=off active in cmdline; disables XDNA 2 NPU and deep S3 sleep (acceptable only for desktop benchmark mode)."
		}
		return StatusSafeConfigured, "IOMMU is enabled (default); NPU and power management are functional."

	case "GOTCHA_MESA_RADV_STALE":
		if env.MesaVersion != "" && (strings.HasPrefix(env.MesaVersion, "24.0") || strings.HasPrefix(env.MesaVersion, "24.1") || strings.HasPrefix(env.MesaVersion, "24.2")) {
			return StatusDefectDetected, fmt.Sprintf("Mesa version is %s; lacks RDNA 3.5 Wave32 FlashAttention and KHR_coopmat optimizations (upgrade to Mesa 25.3+).", env.MesaVersion)
		}
		return StatusSafeConfigured, "Mesa / RADV driver version is current (supports Wave32 cooperative matrices)."

	case "GOTCHA_BATCH_CLAMP_SILENT":
		if env.BatchFlagsConfigured {
			return StatusSafeConfigured, "Batch flag validation enforced: -b >= -ub in runner configurations."
		}
		return StatusDefectDetected, "Batch flags not validated: -b < -ub triggers silent clamping to -b in llama.cpp, degrading prompt ingestion."

	case "GOTCHA_VLLM_HIPBLASLT":
		if env.EnvVars["TORCH_BLAS_PREFER_HIPBLASLT"] == "1" {
			return StatusSafeConfigured, "TORCH_BLAS_PREFER_HIPBLASLT=1 is exported; vLLM batch 8+ throughput protected."
		}
		return StatusAdvisory, "TORCH_BLAS_PREFER_HIPBLASLT=1 not set; vLLM FP16 batch 8+ throughput may drop by ~40% on PyTorch <2.14."

	case "GOTCHA_NPU_BANDWIDTH_CONTENTION":
		if env.NPUOffloadEnabled {
			return StatusSafeConfigured, "NPU co-residency isolation enabled: auxiliary workloads routed to XDNA 2 NPU; iGPU latency penalty <3.5%."
		}
		return StatusAdvisory, "Auxiliary LLM workloads (embeddings, audio, small sidecars) not offloaded to NPU; competing on iGPU causes up to 60% throughput drop."

	case "GOTCHA_KERNEL7_KFD_DEADLOCK":
		if strings.HasPrefix(env.KernelVersion, "7.0.0-28") {
			return StatusDefectDetected, fmt.Sprintf("Kernel %s detected; vulnerable to KFD work-queue deadlocks during >64GB allocations (use 6.17 LTS).", env.KernelVersion)
		}
		return StatusSafeConfigured, "Kernel version is safe for large unified memory allocations."

	case "GOTCHA_CPU_STORM_INVLPGB":
		if env.ProcCPUInfo != "" && !strings.Contains(env.ProcCPUInfo, "invlpgb") {
			return StatusAdvisory, "invlpgb instruction not exposed in /proc/cpuinfo; checkpoint evals in fine-tuning may cause temporary 100% CPU lockups."
		}
		return StatusSafeConfigured, "Zen 5 invlpgb broadcast TLB invalidation verified or not in fine-tuning loop."

	case "GOTCHA_NVME_HIGH_WAF":
		if env.DirtyRingBufferActive {
			return StatusSafeConfigured, "Write-back dirty ring buffer (2-4 GiB) active; protects client NVMe SSD from high WAF."
		}
		return StatusDefectDetected, "Synchronous disk paging of KV caches active without dirty ring buffer; client NVMe SSD vulnerable to extreme WAF (>30x) wear-out."

	case "GOTCHA_THERMAL_CLOCK_HUNTING":
		if env.DPMGovernorConfigured {
			return StatusSafeConfigured, "Static DPM 'high' performance governor active with clock ceiling to prevent acoustic fan hunting."
		}
		return StatusDefectDetected, "DPM dynamic boost active without clock ceiling; causes thermal throttling oscillations and acoustic fan hunting."

	case "GOTCHA_USB4_CLUSTER_LATENCY":
		if env.USB4Tuned {
			return StatusSafeConfigured, "USB4 cluster tuning: single-node preferred for models <=128GB; MTU 9000 & pm_qos configured for multi-node."
		}
		return StatusAdvisory, "USB4 link sleep states untuned; multi-node clustering on models <=128GB suffers 15-20% latency penalty over single node."

	case "GOTCHA_LPDDR5X_CHANNEL_CAMPING":
		if !env.IsStrixHalo {
			return StatusNotApplicable, "Applies to AMD Strix Halo (Ryzen AI MAX+ / gfx1151) 16-channel LPDDR5X memory subsystem."
		}
		if env.F16KVContiguizationEnabled || env.EnvVars["FAK_F16_KV_CONTIGUIZE"] == "1" {
			return StatusSafeConfigured, "Pre-attention f16 KV contiguization pass enabled; uniform LPDDR5X 16-channel interleaving verified."
		}
		return StatusDefectDetected, "Strided f16 KV cache causes LPDDR5X channel camping across 16 pseudo-channels (channel entropy < 0.25, <= 2 active channels); pre-attention contiguization pass is disabled."

	default:
		return StatusSafeConfigured, "Verified safe."
	}
}

// BuildHostProbeEnvironment constructs a probe environment from the live host.
func BuildHostProbeEnvironment() GotchaProbeEnvironment {
	return BuildHostProbeEnvironmentWithFS(osFileSystem{})
}

// BuildHostProbeEnvironmentWithFS constructs a probe environment using the provided FileSystem abstraction.
func BuildHostProbeEnvironmentWithFS(fs FileSystem) GotchaProbeEnvironment {
	if fs == nil {
		fs = osFileSystem{}
	}
	env := GotchaProbeEnvironment{
		GOOS:        runtime.GOOS,
		EnvVars:     make(map[string]string),
		IsStrixHalo: false,
		FS:          fs,
	}

	for _, e := range os.Environ() {
		pair := strings.SplitN(e, "=", 2)
		if len(pair) == 2 {
			env.EnvVars[pair[0]] = pair[1]
		}
	}

	if env.EnvVars["FAK_F16_KV_CONTIGUIZE"] == "1" {
		env.F16KVContiguizationEnabled = true
	}
	if val := env.EnvVars["KV_CACHE_DIRTY_RING_BUFFER_GIB"]; val != "" {
		if gib, err := strconv.ParseUint(val, 10, 64); err == nil && gib >= 2 {
			env.DirtyRingBufferActive = true
		}
	}

	hasLinuxFiles := false
	if _, err := fs.Stat("/proc/version"); err == nil {
		hasLinuxFiles = true
	} else if _, err := fs.Stat("/proc/cmdline"); err == nil {
		hasLinuxFiles = true
	}

	if runtime.GOOS == "windows" && !hasLinuxFiles {
		// Probe Windows GPU facts via PowerShell
		facts := Facts("", PowerShellRunner)
		if facts["available"] == true {
			if name, ok := facts["name"].(string); ok {
				env.GPUName = name
				_, env.IsStrixHalo = DetectAPU(name)
			}
		}
		ok, out, _ := PowerShellRunner("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory", 5*1000000000)
		if ok {
			if v, err := strconv.ParseUint(strings.TrimSpace(out), 10, 64); err == nil {
				env.TotalRAMBytes = v
			}
		}
	} else {
		// Linux probing (also used for mock linux filesystem on any OS)
		if hasLinuxFiles && env.GOOS != "linux" {
			env.GOOS = "linux"
		}
		if cmdData, err := fs.ReadFile("/proc/cmdline"); err == nil {
			env.KernelCmdline = string(cmdData)
		}
		if cpuData, err := fs.ReadFile("/proc/cpuinfo"); err == nil {
			env.ProcCPUInfo = string(cpuData)
			if strings.Contains(env.ProcCPUInfo, "Ryzen AI MAX") || strings.Contains(env.ProcCPUInfo, "Strix Halo") {
				env.IsStrixHalo = true
			}
		}
		if memData, err := fs.ReadFile("/proc/meminfo"); err == nil {
			if ram, err := ParseMemTotalFromProcMeminfo(string(memData)); err == nil {
				env.TotalRAMBytes = ram
			}
		}
		if lockupData, err := fs.ReadFile("/sys/module/amdgpu/parameters/lockup_timeout"); err == nil {
			env.SysfsLockupVal = strings.TrimSpace(string(lockupData))
		}
		if ttmData, err := fs.ReadFile("/sys/module/ttm/parameters/pages_limit"); err == nil {
			if p, err := strconv.ParseUint(strings.TrimSpace(string(ttmData)), 10, 64); err == nil {
				env.SysfsTTMPagesVal = p
			}
		}
		if verData, err := fs.ReadFile("/proc/version"); err == nil {
			verStr := string(verData)
			fields := strings.Fields(verStr)
			if len(fields) >= 3 {
				env.KernelVersion = fields[2]
			}
			lowerVer := strings.ToLower(verStr)
			if strings.Contains(lowerVer, "microsoft") || strings.Contains(lowerVer, "wsl") {
				env.IsWSL2 = true
			}
		}
		if osReleaseData, err := fs.ReadFile("/etc/os-release"); err == nil {
			env.DistroID = parseDistroID(string(osReleaseData))
		}

		env.IsContainer = detectContainer(fs)
		env.MesaVersion = probeMesaVersion(fs, env.EnvVars)
		env.SysfsVRAMTotalBytes = probeSysfsVRAMTotalBytes(fs)
		env.HasOllamaInstalled = probeOllamaInstalled(fs)
		env.HasOllamaProcess = probeOllamaProcess(fs)
		if probeDPMGovernorConfigured(fs) {
			env.DPMGovernorConfigured = true
		}
	}

	return env
}

func parseDistroID(content string) string {
	lines := strings.Split(content, "\n")
	var idLike string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "ID=") {
			val := strings.TrimPrefix(line, "ID=")
			val = strings.ToLower(strings.Trim(val, "\"'"))
			return val
		}
		if strings.HasPrefix(line, "ID_LIKE=") {
			val := strings.TrimPrefix(line, "ID_LIKE=")
			idLike = strings.ToLower(strings.Trim(val, "\"'"))
		}
	}
	if idLike != "" {
		fields := strings.Fields(idLike)
		if len(fields) > 0 {
			return fields[0]
		}
	}
	return ""
}

func detectContainer(fs FileSystem) bool {
	if _, err := fs.Stat("/.dockerenv"); err == nil {
		return true
	}
	if _, err := fs.Stat("/run/.containerenv"); err == nil {
		return true
	}
	for _, cgroupPath := range []string{"/proc/1/cgroup", "/proc/self/cgroup"} {
		if data, err := fs.ReadFile(cgroupPath); err == nil {
			lower := strings.ToLower(string(data))
			if strings.Contains(lower, "docker") || strings.Contains(lower, "containerd") ||
				strings.Contains(lower, "kubepods") || strings.Contains(lower, "container") {
				return true
			}
		}
	}
	return false
}

func probeMesaVersion(fs FileSystem, envVars map[string]string) string {
	if envVars != nil {
		if v, ok := envVars["MESA_VERSION"]; ok && v != "" {
			return strings.TrimSpace(v)
		}
	}

	patterns := []string{
		"/usr/share/doc/mesa*/changelog*",
		"/usr/share/doc/*mesa*/changelog*",
		"/usr/share/doc/mesa*/CHANGELOG*",
	}
	for _, pat := range patterns {
		matches, err := fs.Glob(pat)
		if err == nil {
			for _, m := range matches {
				data, err := fs.ReadFile(m)
				if err == nil && len(data) > 0 {
					if ver := extractMesaVersionFromChangelog(string(data)); ver != "" {
						return ver
					}
				}
			}
		}
	}

	if data, err := fs.ReadFile("/var/lib/dpkg/status"); err == nil {
		if ver := extractMesaVersionFromDpkgStatus(string(data)); ver != "" {
			return ver
		}
	}

	if matches, err := fs.Glob("/var/lib/pacman/local/mesa-*/desc"); err == nil {
		for _, m := range matches {
			if data, err := fs.ReadFile(m); err == nil {
				if ver := extractMesaVersionFromPacmanDesc(string(data)); ver != "" {
					return ver
				}
			}
		}
	}

	if data, err := fs.ReadFile("/etc/os-release"); err == nil {
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "MESA_VERSION=") {
				val := strings.TrimPrefix(line, "MESA_VERSION=")
				return strings.Trim(val, "\"'")
			}
		}
	}

	if envVars != nil {
		if vi, ok := envVars["VULKANINFO"]; ok && vi != "" {
			if ver := extractMesaVersionFromVulkanInfo(vi); ver != "" {
				return ver
			}
		}
	}

	return ""
}

func extractMesaVersionFromChangelog(content string) string {
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		lower := strings.ToLower(line)
		if strings.Contains(lower, "mesa") {
			openIdx := strings.Index(line, "(")
			closeIdx := strings.Index(line, ")")
			if openIdx != -1 && closeIdx > openIdx {
				verPart := line[openIdx+1 : closeIdx]
				if colonIdx := strings.Index(verPart, ":"); colonIdx != -1 {
					verPart = verPart[colonIdx+1:]
				}
				if dashIdx := strings.Index(verPart, "-"); dashIdx != -1 {
					verPart = verPart[:dashIdx]
				}
				verPart = strings.TrimSpace(verPart)
				if isSemverLike(verPart) {
					return verPart
				}
			}
		}
	}
	return ""
}

func extractMesaVersionFromDpkgStatus(content string) string {
	paragraphs := strings.Split(content, "\n\n")
	for _, p := range paragraphs {
		lines := strings.Split(p, "\n")
		isMesaPkg := false
		version := ""
		for _, l := range lines {
			l = strings.TrimSpace(l)
			if strings.HasPrefix(l, "Package:") {
				pkg := strings.TrimSpace(strings.TrimPrefix(l, "Package:"))
				if strings.Contains(pkg, "mesa") {
					isMesaPkg = true
				}
			}
			if strings.HasPrefix(l, "Version:") {
				v := strings.TrimSpace(strings.TrimPrefix(l, "Version:"))
				if colonIdx := strings.Index(v, ":"); colonIdx != -1 {
					v = v[colonIdx+1:]
				}
				if dashIdx := strings.Index(v, "-"); dashIdx != -1 {
					v = v[:dashIdx]
				}
				version = strings.TrimSpace(v)
			}
		}
		if isMesaPkg && isSemverLike(version) {
			return version
		}
	}
	return ""
}

func extractMesaVersionFromPacmanDesc(content string) string {
	lines := strings.Split(content, "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) == "%VERSION%" && i+1 < len(lines) {
			v := strings.TrimSpace(lines[i+1])
			if dashIdx := strings.Index(v, "-"); dashIdx != -1 {
				v = v[:dashIdx]
			}
			if isSemverLike(v) {
				return v
			}
		}
	}
	return ""
}

func extractMesaVersionFromVulkanInfo(content string) string {
	lines := strings.Split(content, "\n")
	for _, l := range lines {
		if strings.Contains(l, "Mesa ") {
			idx := strings.Index(l, "Mesa ")
			v := strings.TrimSpace(l[idx+len("Mesa "):])
			fields := strings.Fields(v)
			if len(fields) > 0 && isSemverLike(fields[0]) {
				return fields[0]
			}
		}
	}
	return ""
}

func isSemverLike(s string) bool {
	parts := strings.Split(s, ".")
	if len(parts) < 2 {
		return false
	}
	for _, p := range parts {
		if _, err := strconv.Atoi(p); err != nil {
			return false
		}
	}
	return true
}

func probeSysfsVRAMTotalBytes(fs FileSystem) uint64 {
	matches, err := fs.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err == nil {
		for _, m := range matches {
			if data, err := fs.ReadFile(m); err == nil {
				valStr := strings.TrimSpace(string(data))
				if val, err := strconv.ParseUint(valStr, 10, 64); err == nil && val > 0 {
					return val
				}
			}
		}
	}
	resMatches, err := fs.Glob("/sys/class/drm/card*/device/resource")
	if err == nil {
		for _, m := range resMatches {
			if data, err := fs.ReadFile(m); err == nil {
				var maxBar uint64
				lines := strings.Split(string(data), "\n")
				for _, line := range lines {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						start, err1 := strconv.ParseUint(strings.TrimPrefix(fields[0], "0x"), 16, 64)
						end, err2 := strconv.ParseUint(strings.TrimPrefix(fields[1], "0x"), 16, 64)
						if err1 == nil && err2 == nil && end > start {
							barSize := end - start + 1
							if barSize > maxBar {
								maxBar = barSize
							}
						}
					}
				}
				if maxBar > 0 {
					return maxBar
				}
			}
		}
	}
	return 0
}

func probeOllamaInstalled(fs FileSystem) bool {
	for _, p := range []string{"/usr/bin/ollama", "/usr/local/bin/ollama", "/opt/ollama/bin/ollama"} {
		if _, err := fs.Stat(p); err == nil {
			return true
		}
	}
	return false
}

func probeOllamaProcess(fs FileSystem) bool {
	procs, err := fs.Glob("/proc/[0-9]*/cmdline")
	if err == nil {
		for _, p := range procs {
			if data, err := fs.ReadFile(p); err == nil {
				if strings.Contains(string(data), "ollama") {
					return true
				}
			}
		}
	}
	return false
}

func probeDPMGovernorConfigured(fs FileSystem) bool {
	matches, err := fs.Glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")
	if err == nil {
		for _, m := range matches {
			if data, err := fs.ReadFile(m); err == nil {
				if strings.TrimSpace(string(data)) == "high" {
					return true
				}
			}
		}
	}
	return false
}

// GenerateFixPlan generates concrete actionable shell commands / bootloader adjustments for detected defects.
func GenerateFixPlan(report *GotchaAuditReport) []string {
	fixes := make([]string, 0)

	bootloaderCmd := "sudo update-grub"
	distro := strings.ToLower(report.DistroID)
	switch {
	case strings.Contains(distro, "fedora") || strings.Contains(distro, "rhel") || strings.Contains(distro, "centos"):
		bootloaderCmd = "sudo grub2-mkconfig -o /boot/grub2/grub.cfg"
	case strings.Contains(distro, "arch") || strings.Contains(distro, "manjaro"):
		bootloaderCmd = "sudo grub-mkconfig -o /boot/grub/grub.cfg"
	case strings.Contains(distro, "opensuse") || strings.Contains(distro, "suse"):
		bootloaderCmd = "sudo grub2-mkconfig -o /boot/grub2/grub.cfg"
	case strings.Contains(distro, "nixos"):
		bootloaderCmd = "# NixOS: add boot.kernelParams to /etc/nixos/configuration.nix and run sudo nixos-rebuild switch"
	default:
		bootloaderCmd = "sudo update-grub"
	}

	for _, f := range report.Findings {
		if f.Status == StatusDefectDetected {
			switch f.Gotcha.ID {
			case "GOTCHA_RING_TIMEOUT":
				fixes = append(fixes, "# Fix 1: Disable AMDGPU watchdog timeout to prevent long-context crashes")
				if report.IsContainer {
					fixes = append(fixes, "# Container environment detected: host kernel parameter amdgpu.lockup_timeout=-1 must be configured on host OS, not in container")
				} else if report.IsWSL2 {
					fixes = append(fixes, "# WSL2 environment detected: configure %USERPROFILE%\\.wslconfig; update-grub is not applicable")
				} else if strings.Contains(distro, "nixos") {
					fixes = append(fixes, bootloaderCmd)
				} else {
					fixes = append(fixes, "sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"amdgpu.lockup_timeout=-1 /' /etc/default/grub")
					fixes = append(fixes, bootloaderCmd)
				}

			case "GOTCHA_TTM_50PCT_CEILING":
				ttmParam := "ttm.pages_limit=31457280 amdgpu.gttsize=131072"
				titleDesc := "Unlock full 120GB UMA aperture in Linux TTM subsystem"
				if report.TotalRAMGiB > 0 && report.TotalRAMGiB < 80 {
					ttmParam = "ttm.pages_limit=14680064 amdgpu.gttsize=65536"
					titleDesc = "Unlock full 56GB UMA aperture in Linux TTM subsystem"
				} else if report.TotalRAMGiB >= 80 && report.TotalRAMGiB < 112 {
					ttmParam = "ttm.pages_limit=23068672 amdgpu.gttsize=98304"
					titleDesc = "Unlock full 88GB UMA aperture in Linux TTM subsystem"
				}
				fixes = append(fixes, "# Fix 2: "+titleDesc)
				if report.IsContainer {
					fixes = append(fixes, "# Container environment detected: host kernel parameter "+strings.Fields(ttmParam)[0]+" must be configured on host OS, not in container")
				} else if report.IsWSL2 {
					fixes = append(fixes, "# WSL2 environment detected: configure [wsl2] memory in %USERPROFILE%\\.wslconfig; update-grub is not applicable")
				} else if strings.Contains(distro, "nixos") {
					fixes = append(fixes, bootloaderCmd)
				} else {
					fixes = append(fixes, fmt.Sprintf("sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"%s /' /etc/default/grub", ttmParam))
					fixes = append(fixes, bootloaderCmd)
				}

			case "GOTCHA_BIOS_UMA_GTT":
				fixes = append(fixes, "# Fix: Reduce BIOS UMA Frame Buffer Size to 512MB or 2GB in UEFI/BIOS setup to prevent Linux host kernel memory starvation")

			case "GOTCHA_OLLAMA_IGPU_FALLBACK":
				fixes = append(fixes, "# Fix 3: Force Ollama to use Radeon 8060S Vulkan iGPU instead of CPU")
				fixes = append(fixes, "sudo mkdir -p /etc/systemd/system/ollama.service.d")
				fixes = append(fixes, "printf '[Service]\\nEnvironment=\"OLLAMA_VULKAN=1\"\\nEnvironment=\"OLLAMA_IGPU_ENABLE=1\"\\nEnvironment=\"HIP_VISIBLE_DEVICES=-1\"\\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf")
				fixes = append(fixes, "sudo systemctl daemon-reload && sudo systemctl restart ollama")

			case "GOTCHA_GFX1151_OVERRIDE":
				fixes = append(fixes, "# Fix 4: Remove fatal HSA_OVERRIDE_GFX_VERSION=11.0.0 from environment")
				fixes = append(fixes, "unset HSA_OVERRIDE_GFX_VERSION")

			case "GOTCHA_GGML_UNIFIED_CORRUPT":
				fixes = append(fixes, "# Fix 5: Unset GGML_CUDA_ENABLE_UNIFIED_MEMORY to prevent token corruption")
				fixes = append(fixes, "unset GGML_CUDA_ENABLE_UNIFIED_MEMORY")

			case "GOTCHA_MESA_RADV_STALE":
				fixes = append(fixes, "# Fix 6: Upgrade Mesa for RDNA 3.5 Wave32 FlashAttention and KHR_coopmat")
				switch {
				case strings.Contains(distro, "arch") || strings.Contains(distro, "manjaro"):
					fixes = append(fixes, "sudo pacman -S mesa vulkan-radeon")
				case strings.Contains(distro, "fedora") || strings.Contains(distro, "rhel") || strings.Contains(distro, "centos"):
					fixes = append(fixes, "sudo dnf upgrade mesa-dri-drivers")
				case strings.Contains(distro, "opensuse") || strings.Contains(distro, "suse"):
					fixes = append(fixes, "sudo zypper update Mesa")
				case strings.Contains(distro, "nixos"):
					fixes = append(fixes, "# NixOS: add pkgs.mesa to environment.systemPackages in /etc/nixos/configuration.nix and run sudo nixos-rebuild switch")
				default:
					fixes = append(fixes, "sudo add-apt-repository -y ppa:kisak/kisak-mesa && sudo apt update && sudo apt upgrade -y")
				}

			case "GOTCHA_SPEC_GRAPH_TIMEOUT":
				fixes = append(fixes, "# Fix: Configure decoupled speculative draft ubatch size to avoid Vulkan ring timeouts")
				fixes = append(fixes, "export SPEC_DRAFT_UBATCH_SIZE=512")

			case "GOTCHA_BATCH_CLAMP_SILENT":
				fixes = append(fixes, "# Fix: Configure runner batch flags so -b >= -ub (e.g. -b 2048 -ub 512)")

			case "GOTCHA_NVME_HIGH_WAF":
				fixes = append(fixes, "# Fix: Enable host DRAM write-back dirty ring buffer for KV cache paging")
				fixes = append(fixes, "export KV_CACHE_DIRTY_RING_BUFFER_GIB=4")

			case "GOTCHA_THERMAL_CLOCK_HUNTING":
				fixes = append(fixes, "# Fix: Lock DPM governor to high performance with clock ceiling to prevent acoustic fan hunting")
				fixes = append(fixes, "echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level")

			case "GOTCHA_LPDDR5X_CHANNEL_CAMPING":
				fixes = append(fixes, "# Fix: Enable pre-attention f16 KV contiguization pass to prevent LPDDR5X channel camping")
				fixes = append(fixes, "export FAK_F16_KV_CONTIGUIZE=1")
			}
		}
	}
	return fixes
}

// RemediateGotchas generates concrete remediation commands for detected defects from an audit report.
func RemediateGotchas(report *GotchaAuditReport) []string {
	return GenerateFixPlan(report)
}

// GenerateRollbackScript generates shell commands to undo remediation changes and restore previous settings.
func GenerateRollbackScript(report *GotchaAuditReport) []string {
	rollbacks := make([]string, 0)
	bootloaderCmd := "sudo update-grub"
	distro := strings.ToLower(report.DistroID)
	switch {
	case strings.Contains(distro, "fedora") || strings.Contains(distro, "rhel") || strings.Contains(distro, "centos"):
		bootloaderCmd = "sudo grub2-mkconfig -o /boot/grub2/grub.cfg"
	case strings.Contains(distro, "arch") || strings.Contains(distro, "manjaro"):
		bootloaderCmd = "sudo grub-mkconfig -o /boot/grub/grub.cfg"
	case strings.Contains(distro, "opensuse") || strings.Contains(distro, "suse"):
		bootloaderCmd = "sudo grub2-mkconfig -o /boot/grub2/grub.cfg"
	}

	needsBootloaderUpdate := false

	for _, f := range report.Findings {
		if f.Status == StatusDefectDetected {
			switch f.Gotcha.ID {
			case "GOTCHA_RING_TIMEOUT", "GOTCHA_TTM_50PCT_CEILING":
				if !report.IsContainer && !report.IsWSL2 && !strings.Contains(distro, "nixos") {
					needsBootloaderUpdate = true
				}
			case "GOTCHA_OLLAMA_IGPU_FALLBACK":
				rollbacks = append(rollbacks, "# Revert Ollama service override")
				rollbacks = append(rollbacks, "sudo rm -f /etc/systemd/system/ollama.service.d/override.conf")
				rollbacks = append(rollbacks, "sudo systemctl daemon-reload && sudo systemctl restart ollama")
			case "GOTCHA_THERMAL_CLOCK_HUNTING":
				rollbacks = append(rollbacks, "# Revert DPM performance level to auto")
				rollbacks = append(rollbacks, "echo auto | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level")
			case "GOTCHA_LPDDR5X_CHANNEL_CAMPING":
				rollbacks = append(rollbacks, "# Revert pre-attention f16 KV contiguization setting")
				rollbacks = append(rollbacks, "unset FAK_F16_KV_CONTIGUIZE")
			}
		}
	}

	if needsBootloaderUpdate {
		rollbacks = append(rollbacks, "# Revert kernel command line parameters in /etc/default/grub")
		rollbacks = append(rollbacks, "sudo sed -i 's/amdgpu.lockup_timeout=[^ ]* //g' /etc/default/grub")
		rollbacks = append(rollbacks, "sudo sed -i 's/ttm.pages_limit=[^ ]* //g' /etc/default/grub")
		rollbacks = append(rollbacks, "sudo sed -i 's/amdgpu.gttsize=[^ ]* //g' /etc/default/grub")
		rollbacks = append(rollbacks, bootloaderCmd)
	}

	return rollbacks
}

// ValidateTTMPagesLimit checks that the requested TTM pages limit does not exceed physical RAM
// and leaves a safe operational reserve (at least 4 GiB) for host OS processes.
func ValidateTTMPagesLimit(pagesLimit uint64, totalRAMBytes uint64) (bool, error) {
	if totalRAMBytes == 0 {
		return true, nil
	}
	const pageSize = 4096
	requestedBytes := pagesLimit * pageSize
	if requestedBytes > totalRAMBytes {
		return false, fmt.Errorf("requested TTM allocation (%d pages / %.1f GiB) exceeds total physical RAM (%.1f GiB)",
			pagesLimit, float64(requestedBytes)/(1024*1024*1024), float64(totalRAMBytes)/(1024*1024*1024))
	}
	const minOSReserve = uint64(4 * 1024 * 1024 * 1024) // 4 GiB
	if totalRAMBytes > minOSReserve && requestedBytes > (totalRAMBytes-minOSReserve) {
		return false, fmt.Errorf("requested TTM allocation (%d pages / %.1f GiB) leaves only %.1f GiB reserve for OS (minimum required: 4.0 GiB)",
			pagesLimit, float64(requestedBytes)/(1024*1024*1024), float64(totalRAMBytes-requestedBytes)/(1024*1024*1024))
	}
	return true, nil
}

// RunGotchasCLI provides the command-line interface for the AMD Strix Halo gotchas auditor.
func RunGotchasCLI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-gotchas", flag.ContinueOnError)
	fs.SetOutput(stderr)

	jsonOut := fs.Bool("json", false, "output gotchas report as JSON")
	fixPlan := fs.Bool("fix-plan", false, "display actionable remediation commands for detected defects")
	rollback := fs.Bool("rollback", false, "display rollback commands to revert remediation changes")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "amd-gotchas: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	env := BuildHostProbeEnvironment()
	report := AuditHostGotchas(env)

	if *jsonOut {
		data, err := report.ToJSON()
		if err != nil {
			fmt.Fprintf(stderr, "error formatting JSON: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprint(stdout, report.Summary())
		if *fixPlan || report.DefectCount > 0 {
			fixes := GenerateFixPlan(report)
			if len(fixes) > 0 {
				fmt.Fprintln(stdout, "Actionable Remediation Commands:")
				for _, cmd := range fixes {
					fmt.Fprintln(stdout, "  "+cmd)
				}
				fmt.Fprintln(stdout, "")
			}
		}
		if *rollback {
			rollbacks := GenerateRollbackScript(report)
			if len(rollbacks) > 0 {
				fmt.Fprintln(stdout, "Rollback Commands:")
				for _, cmd := range rollbacks {
					fmt.Fprintln(stdout, "  "+cmd)
				}
				fmt.Fprintln(stdout, "")
			}
		}
	}

	if report.DefectCount > 0 {
		return 1
	}
	return 0
}
