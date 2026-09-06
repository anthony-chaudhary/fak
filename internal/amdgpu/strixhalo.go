// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// and Strix Halo APU operational serving profiles.
package amdgpu

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// PresetStrixHalo128 is the hardware tuning preset for 128 GiB AMD Strix Halo APUs.
	PresetStrixHalo128 = "strix-halo-128"
	// PresetStrixHalo64 is the hardware tuning preset for 64 GiB AMD Strix Halo APUs.
	PresetStrixHalo64 = "strix-halo-64"
)

// StrixHaloPlatform represents the hardware memory configuration tier of AMD Strix Halo APUs.
type StrixHaloPlatform string

const (
	StrixHalo128GB StrixHaloPlatform = PresetStrixHalo128
	StrixHalo64GB  StrixHaloPlatform = PresetStrixHalo64
)

// String returns the string representation of the platform.
func (p StrixHaloPlatform) String() string {
	return string(p)
}

// ParseStrixHaloPlatform maps a string identifier to a StrixHaloPlatform.
func ParseStrixHaloPlatform(s string) (StrixHaloPlatform, error) {
	lower := strings.ToLower(strings.TrimSpace(s))
	switch lower {
	case "strix-halo-128", "strix-halo-128gb", "128", "128gb", "128gib":
		return StrixHalo128GB, nil
	case "strix-halo-64", "strix-halo-64gb", "64", "64gb", "64gib":
		return StrixHalo64GB, nil
	default:
		return "", fmt.Errorf("amdgpu: unknown Strix Halo platform %q (valid: %s, %s)", s, StrixHalo128GB, StrixHalo64GB)
	}
}

// StrixHaloServingConfig specifies the operational serving parameters and hardware resource
// allocations for AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151).
type StrixHaloServingConfig struct {
	Platform                  StrixHaloPlatform   `json:"platform"`
	UMAAllocatedGiB           float64             `json:"uma_allocated_gib"`
	OSReservedGiB             float64             `json:"os_reserved_gib"`
	KVPrecision               compute.KVPrecision `json:"kv_precision"`
	KVPrecisionLabel          string              `json:"kv_precision_label"`
	EnableF16KVContiguization bool                `json:"enable_f16_kv_contiguization"`
	ContiguizationScratchGiB  float64             `json:"contiguization_scratch_gib"`
	ContiguizationMinContext  int                 `json:"contiguization_min_context"`
	DecoupledDraftUBatch      int                 `json:"decoupled_draft_ubatch"`
	PrefillChunkTokens        int                 `json:"prefill_chunk_tokens"`
	WatchdogTimeoutMs         int64               `json:"watchdog_timeout_ms"`
	MaxConcurrentAgents       int                 `json:"max_concurrent_agents"`
	MaxContextTokens          int                 `json:"max_context_tokens"`
	MaxDepthOfTurns           int                 `json:"max_depth_of_turns"`
}

// CalculateStrixHaloServingEnvelope derives the optimal Strix Halo serving configuration,
// verifying that model weights, runtime scratchpads, and target KV context fit within
// the unified memory aperture while computing safe concurrency and turn depth limits.
func CalculateStrixHaloServingEnvelope(
	platform StrixHaloPlatform,
	modelWeightsBytes int64,
	targetContextTokens int,
	targetConcurrency int,
) (*StrixHaloServingConfig, error) {
	var (
		umaAllocatedGiB float64
		osReservedGiB   float64
		scratchGiB      float64
	)

	switch platform {
	case StrixHalo128GB:
		umaAllocatedGiB = 120.0
		osReservedGiB = 8.0
		scratchGiB = 8.0
	case StrixHalo64GB:
		umaAllocatedGiB = 56.0
		osReservedGiB = 8.0
		scratchGiB = 4.0
	default:
		return nil, fmt.Errorf("amdgpu: unsupported Strix Halo platform %q", platform)
	}

	umaAllocatedBytes := int64(umaAllocatedGiB * 1024 * 1024 * 1024)
	scratchBytes := int64(scratchGiB * 1024 * 1024 * 1024)

	if modelWeightsBytes <= 0 {
		return nil, fmt.Errorf("amdgpu: invalid model weights size (%d bytes); must be positive", modelWeightsBytes)
	}
	if modelWeightsBytes > umaAllocatedBytes {
		return nil, fmt.Errorf("amdgpu: model weights (%d bytes / %.2f GiB) exceed GPU UMA aperture (%d bytes / %.2f GiB)",
			modelWeightsBytes, float64(modelWeightsBytes)/(1<<30), umaAllocatedBytes, umaAllocatedGiB)
	}
	if modelWeightsBytes+scratchBytes > umaAllocatedBytes {
		return nil, fmt.Errorf("amdgpu: model weights (%d bytes / %.2f GiB) plus scratch (%d bytes / %.2f GiB) exceed GPU UMA aperture (%.2f GiB)",
			modelWeightsBytes, float64(modelWeightsBytes)/(1<<30), scratchBytes, scratchGiB, umaAllocatedGiB)
	}

	availableKVBudget := umaAllocatedBytes - modelWeightsBytes - scratchBytes

	// Estimate model geometry for KV computation (heads, layers, head dim)
	layers, kvHeads, headDim := estimateModelGeometry(modelWeightsBytes)
	elementsPerToken := int64(layers * kvHeads * headDim * 2) // 2 for K and V

	// FP16 is preferred on UMA (2 bytes per element) due to zero quantization distortion and 200+ GB/s bus
	kvBytesFP16 := elementsPerToken * 2
	// Q8_0 fallback (1 byte per element)
	kvBytesQ8 := elementsPerToken * 1

	if targetContextTokens <= 0 {
		if platform == StrixHalo128GB {
			targetContextTokens = 262144
		} else {
			targetContextTokens = 131072
		}
	}

	requiredKVFP16 := int64(targetContextTokens) * kvBytesFP16
	requiredKVQ8 := int64(targetContextTokens) * kvBytesQ8

	// F16 KV contiguization scratch buffer (per-layer K and V head-contiguous buffers)
	// prevents LPDDR5X 16-channel camping on Strix Halo APUs.
	contiguizationScratchBytes := int64(2 * kvHeads * targetContextTokens * headDim * 2)

	var selectedPrecision compute.KVPrecision
	var precisionLabel string
	var chosenKVBytesPerToken int64
	var enableContiguization bool
	var contiguizationScratchGiB float64
	var contiguizationMinContext int

	if requiredKVFP16+contiguizationScratchBytes <= availableKVBudget {
		selectedPrecision = compute.KVPrecisionF32
		precisionLabel = "f16"
		chosenKVBytesPerToken = kvBytesFP16
		enableContiguization = true
		contiguizationScratchGiB = float64(contiguizationScratchBytes) / (1024 * 1024 * 1024)
		contiguizationMinContext = compute.ContiguizationMinContext
		availableKVBudget -= contiguizationScratchBytes
	} else if requiredKVFP16 <= availableKVBudget {
		selectedPrecision = compute.KVPrecisionF32
		precisionLabel = "f16"
		chosenKVBytesPerToken = kvBytesFP16
		enableContiguization = true
		contiguizationScratchGiB = float64(contiguizationScratchBytes) / (1024 * 1024 * 1024)
		contiguizationMinContext = compute.ContiguizationMinContext
	} else if requiredKVQ8 <= availableKVBudget {
		selectedPrecision = compute.KVPrecisionQ8
		precisionLabel = "q8"
		chosenKVBytesPerToken = kvBytesQ8
	} else {
		return nil, fmt.Errorf("amdgpu: target context (%d tokens) requires %.2f GiB (FP16) or %.2f GiB (Q8), exceeding available KV budget of %.2f GiB",
			targetContextTokens, float64(requiredKVFP16)/(1<<30), float64(requiredKVQ8)/(1<<30), float64(availableKVBudget)/(1<<30))
	}

	// Validate or derive concurrency
	if targetConcurrency <= 0 {
		// Default concurrency based on context token distribution (~4096 tokens per agent)
		targetConcurrency = targetContextTokens / 4096
		if targetConcurrency < 1 {
			targetConcurrency = 1
		}
	}

	// Ensure concurrency does not exceed total budget assuming minimum working context (512 tokens/agent)
	const minTokensPerAgent = 512
	if int64(targetConcurrency*minTokensPerAgent)*chosenKVBytesPerToken > availableKVBudget {
		return nil, fmt.Errorf("amdgpu: target concurrency (%d agents) exceeds KV memory budget at minimum %d tokens/agent",
			targetConcurrency, minTokensPerAgent)
	}

	// Max depth of turns: in deep turns (>100k-262k tokens), compute turn capacity at ~2048 tokens/turn
	const tokensPerTurn = 2048
	maxDepthOfTurns := targetContextTokens / tokensPerTurn
	if maxDepthOfTurns < 1 {
		maxDepthOfTurns = 1
	}

	return &StrixHaloServingConfig{
		Platform:                  platform,
		UMAAllocatedGiB:           umaAllocatedGiB,
		OSReservedGiB:             osReservedGiB,
		KVPrecision:               selectedPrecision,
		KVPrecisionLabel:          precisionLabel,
		EnableF16KVContiguization: enableContiguization,
		ContiguizationScratchGiB:  contiguizationScratchGiB,
		ContiguizationMinContext:  contiguizationMinContext,
		DecoupledDraftUBatch:      512,
		PrefillChunkTokens:        1024,
		WatchdogTimeoutMs:         -1,
		MaxConcurrentAgents:       targetConcurrency,
		MaxContextTokens:          targetContextTokens,
		MaxDepthOfTurns:           maxDepthOfTurns,
	}, nil
}

func estimateModelGeometry(modelWeightsBytes int64) (layers int, kvHeads int, headDim int) {
	headDim = 128
	kvHeads = 8
	gib := float64(modelWeightsBytes) / (1024 * 1024 * 1024)
	switch {
	case gib >= 25.0: // ~32B-70B
		layers = 64
		if gib >= 35.0 {
			layers = 80
		}
	case gib >= 13.0: // ~27B (Q4_K ~15-18 GiB)
		layers = 64
	case gib >= 7.0: // ~14B (Q4_K ~8.5-10 GiB)
		layers = 48
	default: // ~7B-8B (Q4_K ~4.5-5.5 GiB) or smaller
		layers = 32
	}
	return layers, kvHeads, headDim
}

// InspectHostStrixHalo probes the host system to detect whether it is an AMD Strix Halo APU
// (gfx1151 / Ryzen AI MAX+ / Radeon 8060S), determines system memory size, and returns
// the tailored StrixHaloServingConfig.
func InspectHostStrixHalo() (*StrixHaloServingConfig, error) {
	return inspectHostStrixHaloInternal(runtime.GOOS, Facts, PowerShellRunner, os.ReadFile)
}

func inspectHostStrixHaloInternal(
	goos string,
	factsFn func(string, Runner) map[string]any,
	runner Runner,
	readFileFn func(string) ([]byte, error),
) (*StrixHaloServingConfig, error) {
	var isStrix bool
	var detectedName string
	var totalRAMBytes uint64

	if goos == "windows" {
		if factsFn != nil {
			facts := factsFn("", runner)
			if facts["available"] == true {
				if name, ok := facts["name"].(string); ok {
					detectedName = name
					_, isStrix = DetectAPU(name)
					if !isStrix {
						lower := strings.ToLower(name)
						if strings.Contains(lower, "gfx1151") || strings.Contains(lower, "8060s") || strings.Contains(lower, "8050s") {
							isStrix = true
						}
					}
				}
			}
		}
		if runner != nil {
			ok, out, _ := runner("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory", 5*time.Second)
			if ok {
				if v, err := strconv.ParseUint(strings.TrimSpace(out), 10, 64); err == nil {
					totalRAMBytes = v
				}
			}
		}
	} else {
		// Linux sysfs / proc inspection
		if readFileFn != nil {
			if cpuinfo, err := readFileFn("/proc/cpuinfo"); err == nil {
				s := string(cpuinfo)
				if strings.Contains(s, "Ryzen AI MAX") || strings.Contains(s, "Strix Halo") {
					isStrix = true
					detectedName = "AMD Strix Halo APU (CPU model)"
				}
			}
			if meminfo, err := readFileFn("/proc/meminfo"); err == nil {
				if ram, err := ParseMemTotalFromProcMeminfo(string(meminfo)); err == nil {
					totalRAMBytes = ram
				}
			}
		}
	}

	if !isStrix {
		if detectedName == "" {
			detectedName = "unknown device"
		}
		return nil, fmt.Errorf("amdgpu: host is not an AMD Strix Halo platform (detected: %s)", detectedName)
	}

	// Classify 128GB vs 64GB
	platform := StrixHalo128GB
	if totalRAMBytes > 0 && totalRAMBytes < 96*1024*1024*1024 {
		platform = StrixHalo64GB
	}

	if platform == StrixHalo128GB {
		// Default 128GB configuration for 27B model (Q4_K ~16 GiB)
		return CalculateStrixHaloServingEnvelope(StrixHalo128GB, 16*1024*1024*1024, 262144, 64)
	}
	// Default 64GB configuration for 14B model (Q4_K ~9 GiB)
	return CalculateStrixHaloServingEnvelope(StrixHalo64GB, 9*1024*1024*1024, 131072, 32)
}

// DetectAPU inspects an AMD device name and reports whether it is an APU (integrated GPU sharing system RAM)
// and specifically whether it is an AMD Strix Halo platform (Ryzen AI MAX+ 395, 390, 8060S, etc.).
func DetectAPU(name string) (isAPU bool, isStrixHalo bool) {
	lower := strings.ToLower(name)
	if lower == "" {
		return false, false
	}
	if strings.Contains(lower, "strix halo") ||
		strings.Contains(lower, "ryzen ai max") ||
		strings.Contains(lower, "gfx1151") ||
		strings.Contains(lower, "8060s") ||
		strings.Contains(lower, "8050s") {
		return true, true
	}
	if strings.Contains(lower, "apu") ||
		strings.Contains(lower, "ryzen") ||
		strings.Contains(lower, "radeon graphics") ||
		strings.Contains(lower, "radeon(tm) graphics") ||
		strings.Contains(lower, "phoenix") ||
		strings.Contains(lower, "strix") ||
		strings.Contains(lower, "hawk point") ||
		strings.Contains(lower, "rembrandt") ||
		strings.Contains(lower, "cezanne") ||
		strings.Contains(lower, "890m") ||
		strings.Contains(lower, "880m") ||
		strings.Contains(lower, "780m") ||
		strings.Contains(lower, "760m") ||
		strings.Contains(lower, "740m") ||
		strings.Contains(lower, "680m") ||
		strings.Contains(lower, "660m") {
		return true, false
	}
	return false, false
}
