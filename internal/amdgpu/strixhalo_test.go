package amdgpu

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestParseStrixHaloPlatform(t *testing.T) {
	tests := []struct {
		input string
		want  StrixHaloPlatform
	}{
		{"strix-halo-128", StrixHalo128GB},
		{"strix-halo-128gb", StrixHalo128GB},
		{"128", StrixHalo128GB},
		{"128GB", StrixHalo128GB},
		{"strix-halo-64", StrixHalo64GB},
		{"strix-halo-64gb", StrixHalo64GB},
		{"64", StrixHalo64GB},
		{"64gib", StrixHalo64GB},
	}

	for _, tc := range tests {
		got, err := ParseStrixHaloPlatform(tc.input)
		if err != nil {
			t.Fatalf("ParseStrixHaloPlatform(%q) error: %v", tc.input, err)
		}
		if got != tc.want {
			t.Fatalf("ParseStrixHaloPlatform(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}

	if _, err := ParseStrixHaloPlatform("invalid"); err == nil {
		t.Fatal("ParseStrixHaloPlatform(\"invalid\") expected error, got nil")
	}
}

// TestStrixHalo128GB_27BModel proves that a 27B model (Q4_K ~16 GiB) supports 32-64 concurrent
// agents and 262k context tokens within the 120 GiB UMA aperture.
func TestStrixHalo128GB_27BModel(t *testing.T) {
	const modelWeightsBytes = int64(16) * 1024 * 1024 * 1024 // 16 GiB Q4_K
	const targetContextTokens = 262144                       // 262k tokens

	for _, concurrency := range []int{32, 64} {
		cfg, err := CalculateStrixHaloServingEnvelope(
			StrixHalo128GB,
			modelWeightsBytes,
			targetContextTokens,
			concurrency,
		)
		if err != nil {
			t.Fatalf("CalculateStrixHaloServingEnvelope failed for concurrency %d: %v", concurrency, err)
		}

		if cfg.Platform != StrixHalo128GB {
			t.Errorf("Platform = %q, want %q", cfg.Platform, StrixHalo128GB)
		}
		if cfg.UMAAllocatedGiB != 120.0 {
			t.Errorf("UMAAllocatedGiB = %.1f, want 120.0", cfg.UMAAllocatedGiB)
		}
		if cfg.OSReservedGiB != 8.0 {
			t.Errorf("OSReservedGiB = %.1f, want 8.0", cfg.OSReservedGiB)
		}
		if cfg.KVPrecision != compute.KVPrecisionF32 {
			t.Errorf("KVPrecision = %v, want %v (F32/FP16 preferred on UMA)", cfg.KVPrecision, compute.KVPrecisionF32)
		}
		if cfg.KVPrecisionLabel != "f16" {
			t.Errorf("KVPrecisionLabel = %q, want \"f16\"", cfg.KVPrecisionLabel)
		}
		if !cfg.EnableF16KVContiguization {
			t.Errorf("EnableF16KVContiguization = false, want true for f16")
		}
		if cfg.ContiguizationScratchGiB != 1.0 {
			t.Errorf("ContiguizationScratchGiB = %.2f, want 1.0", cfg.ContiguizationScratchGiB)
		}
		if cfg.ContiguizationMinContext != 32768 {
			t.Errorf("ContiguizationMinContext = %d, want 32768", cfg.ContiguizationMinContext)
		}
		if cfg.DecoupledDraftUBatch != 512 {
			t.Errorf("DecoupledDraftUBatch = %d, want 512", cfg.DecoupledDraftUBatch)
		}
		if cfg.PrefillChunkTokens != 1024 {
			t.Errorf("PrefillChunkTokens = %d, want 1024", cfg.PrefillChunkTokens)
		}
		if cfg.WatchdogTimeoutMs != -1 {
			t.Errorf("WatchdogTimeoutMs = %d, want -1", cfg.WatchdogTimeoutMs)
		}
		if cfg.MaxConcurrentAgents != concurrency {
			t.Errorf("MaxConcurrentAgents = %d, want %d", cfg.MaxConcurrentAgents, concurrency)
		}
		if cfg.MaxContextTokens != targetContextTokens {
			t.Errorf("MaxContextTokens = %d, want %d", cfg.MaxContextTokens, targetContextTokens)
		}
		if cfg.MaxDepthOfTurns != 128 {
			t.Errorf("MaxDepthOfTurns = %d, want 128", cfg.MaxDepthOfTurns)
		}
	}
}

// TestStrixHalo64GB_14BModel proves that a 14B model (Q4_K ~9 GiB) supports 16-32 concurrent
// agents and deep context within the 56 GiB UMA aperture.
func TestStrixHalo64GB_14BModel(t *testing.T) {
	const modelWeightsBytes = int64(9) * 1024 * 1024 * 1024 // 9 GiB Q4_K
	const targetContextTokens = 131072                      // 128k tokens

	for _, concurrency := range []int{16, 32} {
		cfg, err := CalculateStrixHaloServingEnvelope(
			StrixHalo64GB,
			modelWeightsBytes,
			targetContextTokens,
			concurrency,
		)
		if err != nil {
			t.Fatalf("CalculateStrixHaloServingEnvelope failed for concurrency %d: %v", concurrency, err)
		}

		if cfg.Platform != StrixHalo64GB {
			t.Errorf("Platform = %q, want %q", cfg.Platform, StrixHalo64GB)
		}
		if cfg.UMAAllocatedGiB != 56.0 {
			t.Errorf("UMAAllocatedGiB = %.1f, want 56.0", cfg.UMAAllocatedGiB)
		}
		if cfg.OSReservedGiB != 8.0 {
			t.Errorf("OSReservedGiB = %.1f, want 8.0", cfg.OSReservedGiB)
		}
		if cfg.KVPrecision != compute.KVPrecisionF32 {
			t.Errorf("KVPrecision = %v, want %v (F32/FP16 preferred on UMA)", cfg.KVPrecision, compute.KVPrecisionF32)
		}
		if cfg.KVPrecisionLabel != "f16" {
			t.Errorf("KVPrecisionLabel = %q, want \"f16\"", cfg.KVPrecisionLabel)
		}
		if !cfg.EnableF16KVContiguization {
			t.Errorf("EnableF16KVContiguization = false, want true for f16")
		}
		if cfg.ContiguizationScratchGiB != 0.5 {
			t.Errorf("ContiguizationScratchGiB = %.2f, want 0.5", cfg.ContiguizationScratchGiB)
		}
		if cfg.ContiguizationMinContext != 32768 {
			t.Errorf("ContiguizationMinContext = %d, want 32768", cfg.ContiguizationMinContext)
		}
		if cfg.DecoupledDraftUBatch != 512 {
			t.Errorf("DecoupledDraftUBatch = %d, want 512", cfg.DecoupledDraftUBatch)
		}
		if cfg.PrefillChunkTokens != 1024 {
			t.Errorf("PrefillChunkTokens = %d, want 1024", cfg.PrefillChunkTokens)
		}
		if cfg.WatchdogTimeoutMs != -1 {
			t.Errorf("WatchdogTimeoutMs = %d, want -1", cfg.WatchdogTimeoutMs)
		}
		if cfg.MaxConcurrentAgents != concurrency {
			t.Errorf("MaxConcurrentAgents = %d, want %d", cfg.MaxConcurrentAgents, concurrency)
		}
		if cfg.MaxContextTokens != targetContextTokens {
			t.Errorf("MaxContextTokens = %d, want %d", cfg.MaxContextTokens, targetContextTokens)
		}
		if cfg.MaxDepthOfTurns != 64 {
			t.Errorf("MaxDepthOfTurns = %d, want 64", cfg.MaxDepthOfTurns)
		}
	}
}

// TestStrixHaloErrorCasesWhenWeightsExceedAperture verifies that models too large for the UMA aperture
// or invalid inputs fail-closed with descriptive errors.
func TestStrixHaloErrorCasesWhenWeightsExceedAperture(t *testing.T) {
	// 125 GiB exceeds 120 GiB aperture on 128GB platform
	_, err := CalculateStrixHaloServingEnvelope(StrixHalo128GB, int64(125)*1024*1024*1024, 4096, 1)
	if err == nil {
		t.Fatal("expected error when weights exceed 120 GiB aperture, got nil")
	}
	if !strings.Contains(err.Error(), "exceed") {
		t.Errorf("unexpected error message: %v", err)
	}

	// 60 GiB exceeds 56 GiB aperture on 64GB platform
	_, err = CalculateStrixHaloServingEnvelope(StrixHalo64GB, int64(60)*1024*1024*1024, 4096, 1)
	if err == nil {
		t.Fatal("expected error when weights exceed 56 GiB aperture, got nil")
	}

	// 115 GiB weights + 8 GiB scratch = 123 GiB > 120 GiB aperture
	_, err = CalculateStrixHaloServingEnvelope(StrixHalo128GB, int64(115)*1024*1024*1024, 4096, 1)
	if err == nil {
		t.Fatal("expected error when weights + scratch exceed aperture, got nil")
	}

	// Invalid model weights (0 or negative)
	_, err = CalculateStrixHaloServingEnvelope(StrixHalo128GB, 0, 4096, 1)
	if err == nil {
		t.Fatal("expected error for 0-byte weights, got nil")
	}
	_, err = CalculateStrixHaloServingEnvelope(StrixHalo128GB, -100, 4096, 1)
	if err == nil {
		t.Fatal("expected error for negative weights, got nil")
	}

	// Unsupported platform
	_, err = CalculateStrixHaloServingEnvelope("unknown-platform", int64(16)*1024*1024*1024, 4096, 1)
	if err == nil {
		t.Fatal("expected error for unsupported platform, got nil")
	}
}

// TestStrixHaloQ8Fallback verifies fallback to Q8_0 when context is too large for FP16
// but still fits in Q8_0 within the remaining budget.
func TestStrixHaloQ8Fallback(t *testing.T) {
	// For 64GB platform (56 GiB aperture):
	// Model weights: 30 GiB. Scratch: 4 GiB. Available KV: 22 GiB.
	// Model geometry ~32B/70B (layers 64, kvHeads 8, headDim 128) => elementsPerToken = 131,072.
	// FP16: 262,144 bytes/token (0.25 MiB).
	// Q8_0: 131,072 bytes/token (0.125 MiB).
	// With 100,000 tokens:
	// FP16 requires 100,000 * 262,144 = 26,214,400,000 bytes (~24.4 GiB > 22 GiB) -> Does not fit FP16!
	// Q8_0 requires 100,000 * 131,072 = 13,107,200,000 bytes (~12.2 GiB <= 22 GiB) -> Fits Q8_0!
	cfg, err := CalculateStrixHaloServingEnvelope(
		StrixHalo64GB,
		int64(30)*1024*1024*1024,
		100000,
		16,
	)
	if err != nil {
		t.Fatalf("expected Q8_0 fallback to succeed, got error: %v", err)
	}
	if cfg.KVPrecision != compute.KVPrecisionQ8 {
		t.Errorf("KVPrecision = %v, want %v", cfg.KVPrecision, compute.KVPrecisionQ8)
	}
	if cfg.KVPrecisionLabel != "q8" {
		t.Errorf("KVPrecisionLabel = %q, want \"q8\"", cfg.KVPrecisionLabel)
	}
	if cfg.EnableF16KVContiguization {
		t.Errorf("EnableF16KVContiguization = true, want false when Q8 fallback is selected")
	}
	if cfg.ContiguizationScratchGiB != 0 {
		t.Errorf("ContiguizationScratchGiB = %f, want 0 when Q8 fallback is selected", cfg.ContiguizationScratchGiB)
	}
}

func TestInspectHostStrixHaloUppercaseLinuxCPU(t *testing.T) {
	readFile := func(path string) ([]byte, error) {
		switch path {
		case "/proc/cpuinfo":
			return []byte("model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"), nil
		case "/proc/meminfo":
			return []byte("MemTotal:       125829120 kB\n"), nil
		default:
			return nil, errors.New("unexpected path " + path)
		}
	}
	config, err := inspectHostStrixHaloInternal("linux", nil, nil, readFile)
	if err != nil {
		t.Fatalf("live Halo CPU model was rejected: %v", err)
	}
	if config == nil {
		t.Fatal("missing Strix Halo serving configuration")
	}
}

// TestInspectHostStrixHaloDedicatedGPUReservation verifies issue #11916:
// on a 128 GiB physical machine where a 64 GiB dedicated GPU reservation reduces
// OS-visible MemTotal to ~56-64 GiB, the platform tier must still be classified
// as StrixHalo128GB, but UMA allocation is capped to the dedicated VRAM (64 GiB)
// so that the reserved bytes are never budgeted twice.
func TestInspectHostStrixHaloDedicatedGPUReservation(t *testing.T) {
	readFile := func(path string) ([]byte, error) {
		switch path {
		case "/proc/cpuinfo":
			return []byte("model name\t: AMD Ryzen AI MAX+ 395 16-Core Processor\n"), nil
		case "/proc/meminfo":
			// 64 GiB OS-visible MemTotal
			return []byte("MemTotal:       67108864 kB\nMemFree:        50000000 kB\n"), nil
		case "/sys/class/drm/card0/device/mem_info_vram_total":
			// 64 GiB dedicated GPU reservation
			return []byte("68719476736\n"), nil
		default:
			return nil, errors.New("file not found: " + path)
		}
	}

	cfg, err := inspectHostStrixHaloInternal("linux", nil, nil, readFile)
	if err != nil {
		t.Fatalf("failed to inspect host with dedicated GPU reservation: %v", err)
	}

	// 1. Hardware platform tier must be preserved as 128GB
	if cfg.Platform != StrixHalo128GB {
		t.Errorf("Platform = %q, want %q (physical tier must be preserved)", cfg.Platform, StrixHalo128GB)
	}

	// 2. Independently witnessed installed capacity and reservation reported separately
	if cfg.InstalledRAMGiB != 128.0 {
		t.Errorf("InstalledRAMGiB = %.1f, want 128.0", cfg.InstalledRAMGiB)
	}
	if cfg.ReservedGPURAMGiB != 64.0 {
		t.Errorf("ReservedGPURAMGiB = %.1f, want 64.0", cfg.ReservedGPURAMGiB)
	}
	if cfg.AllocatableHostRAMGiB != 64.0 {
		t.Errorf("AllocatableHostRAMGiB = %.1f, want 64.0", cfg.AllocatableHostRAMGiB)
	}

	// 3. UMAAllocatedGiB must reflect the dedicated GPU VRAM (64 GiB), NOT 120 GiB (which would double-budget host memory)
	if cfg.UMAAllocatedGiB != 64.0 {
		t.Errorf("UMAAllocatedGiB = %.1f, want 64.0 (must not budget reserved bytes twice)", cfg.UMAAllocatedGiB)
	}
}

// TestInspectHostStrixHaloInternal verifies host detection logic for Strix Halo / gfx1151.
func TestInspectHostStrixHaloInternal(t *testing.T) {
	fakeWindowsRunner128GB := func(script string, timeout time.Duration) (bool, string, string) {
		if strings.Contains(script, "TotalPhysicalMemory") {
			return true, "137438953472", "" // 128 GiB
		}
		return false, "", "unrecognized script"
	}

	fakeWindowsRunner64GB := func(script string, timeout time.Duration) (bool, string, string) {
		if strings.Contains(script, "TotalPhysicalMemory") {
			return true, "68719476736", "" // 64 GiB
		}
		return false, "", "unrecognized script"
	}

	// Case 1: Windows with AMD Radeon 8060S (Strix Halo 128GB)
	facts8060S := func(filter string, r Runner) map[string]any {
		return map[string]any{
			"available": true,
			"name":      "AMD Radeon 8060S Graphics",
		}
	}
	cfg128, err := inspectHostStrixHaloInternal("windows", facts8060S, fakeWindowsRunner128GB, nil)
	if err != nil {
		t.Fatalf("expected Strix Halo 128GB detection, got error: %v", err)
	}
	if cfg128.Platform != StrixHalo128GB {
		t.Errorf("Platform = %q, want %q", cfg128.Platform, StrixHalo128GB)
	}
	if cfg128.UMAAllocatedGiB != 120.0 {
		t.Errorf("UMAAllocatedGiB = %.1f, want 120.0", cfg128.UMAAllocatedGiB)
	}

	// Case 2: Windows with Ryzen AI MAX+ 395 (Strix Halo 64GB)
	factsRyzenMax := func(filter string, r Runner) map[string]any {
		return map[string]any{
			"available": true,
			"name":      "AMD Ryzen AI MAX+ 395 with Radeon 8060S",
		}
	}
	cfg64, err := inspectHostStrixHaloInternal("windows", factsRyzenMax, fakeWindowsRunner64GB, nil)
	if err != nil {
		t.Fatalf("expected Strix Halo 64GB detection, got error: %v", err)
	}
	if cfg64.Platform != StrixHalo64GB {
		t.Errorf("Platform = %q, want %q", cfg64.Platform, StrixHalo64GB)
	}
	if cfg64.UMAAllocatedGiB != 56.0 {
		t.Errorf("UMAAllocatedGiB = %.1f, want 56.0", cfg64.UMAAllocatedGiB)
	}

	// Case 3: Windows with gfx1151 ISA name
	factsGfx1151 := func(filter string, r Runner) map[string]any {
		return map[string]any{
			"available": true,
			"name":      "AMD Radeon Graphics (gfx1151)",
		}
	}
	cfgGfx, err := inspectHostStrixHaloInternal("windows", factsGfx1151, fakeWindowsRunner128GB, nil)
	if err != nil {
		t.Fatalf("expected gfx1151 detection, got error: %v", err)
	}
	if cfgGfx.Platform != StrixHalo128GB {
		t.Errorf("Platform = %q, want %q", cfgGfx.Platform, StrixHalo128GB)
	}

	// Case 4: Windows with non-Strix GPU (NVIDIA RTX 4090) -> error
	factsRTX := func(filter string, r Runner) map[string]any {
		return map[string]any{
			"available": true,
			"name":      "NVIDIA GeForce RTX 4090",
		}
	}
	_, err = inspectHostStrixHaloInternal("windows", factsRTX, fakeWindowsRunner128GB, nil)
	if err == nil {
		t.Fatal("expected error on NVIDIA RTX 4090, got nil")
	}

	// Case 5: Linux with /proc/cpuinfo Ryzen AI MAX and 128GB RAM
	fakeLinuxFS128 := func(path string) ([]byte, error) {
		switch path {
		case "/proc/cpuinfo":
			return []byte("model name : AMD Ryzen AI MAX+ 395 16-Core Processor\n"), nil
		case "/proc/meminfo":
			return []byte("MemTotal:       131072000 kB\nMemFree:        120000000 kB\n"), nil
		default:
			return nil, errors.New("file not found")
		}
	}
	cfgLinux, err := inspectHostStrixHaloInternal("linux", nil, nil, fakeLinuxFS128)
	if err != nil {
		t.Fatalf("expected Linux Strix Halo detection, got error: %v", err)
	}
	if cfgLinux.Platform != StrixHalo128GB {
		t.Errorf("Linux Platform = %q, want %q", cfgLinux.Platform, StrixHalo128GB)
	}

	// Case 6: Live call on current host should not panic (returns either config or error)
	_, _ = InspectHostStrixHalo()
}
