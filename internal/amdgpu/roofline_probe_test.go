package amdgpu

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// TestMALLCacheWorkingSetVsDRAMWorkingSetBandwidthDistinction verifies that working sets
// fitting within the 32 MB MALL cache achieve >800 GB/s while working sets exceeding it
// drop into sustained DRAM bandwidth (~210-224 GB/s).
func TestMALLCacheWorkingSetVsDRAMWorkingSetBandwidthDistinction(t *testing.T) {
	cfg := DefaultProbeConfig()
	cfg.ForceMock = true
	cfg.WorkingSetSizesMB = []int{16, 64}

	receipt, err := RunRooflineProbe(cfg)
	if err != nil {
		t.Fatalf("RunRooflineProbe failed: %v", err)
	}

	// 1. Verify 16 MB working set hits MALL cache (> 800 GB/s)
	if receipt.MALLHitBandwidthGBps < StrixHaloMALLHitBandwidthFloorGBps {
		t.Errorf("MALL hit bandwidth = %.2f GB/s, want >= %.2f GB/s",
			receipt.MALLHitBandwidthGBps, StrixHaloMALLHitBandwidthFloorGBps)
	}

	// 2. Verify 64 MB working set drops to sustained DRAM bandwidth (~210-224 GB/s)
	if receipt.DRAMSustainedBandwidthGBps < StrixHaloSustainedDRAMBandwidthFloorGBps ||
		receipt.DRAMSustainedBandwidthGBps > StrixHaloSustainedDRAMBandwidthCeilGBps {
		t.Errorf("DRAM sustained bandwidth = %.2f GB/s, want in range [%.2f, %.2f] GB/s",
			receipt.DRAMSustainedBandwidthGBps,
			StrixHaloSustainedDRAMBandwidthFloorGBps,
			StrixHaloSustainedDRAMBandwidthCeilGBps)
	}

	// 3. Verify MALL hit bandwidth strictly exceeds DRAM bandwidth with a significant drop ratio (> 3x)
	if receipt.MALLHitBandwidthGBps <= receipt.DRAMSustainedBandwidthGBps {
		t.Fatalf("MALL bandwidth %.2f GB/s must exceed DRAM bandwidth %.2f GB/s",
			receipt.MALLHitBandwidthGBps, receipt.DRAMSustainedBandwidthGBps)
	}
	dropRatio := receipt.MALLHitBandwidthGBps / receipt.DRAMSustainedBandwidthGBps
	if dropRatio < 3.0 {
		t.Errorf("MALL-to-DRAM bandwidth drop ratio = %.2fx, want >= 3.0x", dropRatio)
	}

	// 4. Verify individual sweep points
	var found16MB, found64MB bool
	for _, pt := range receipt.SweepPoints {
		if pt.SizeMB == 16 {
			found16MB = true
			if pt.Residency != "MALL" {
				t.Errorf("16 MB sweep point residency = %q, want 'MALL'", pt.Residency)
			}
			if pt.BandwidthGBps < StrixHaloMALLHitBandwidthFloorGBps {
				t.Errorf("16 MB sweep point bandwidth = %.2f GB/s, want >= 800 GB/s", pt.BandwidthGBps)
			}
		}
		if pt.SizeMB == 64 {
			found64MB = true
			if pt.Residency != "DRAM" {
				t.Errorf("64 MB sweep point residency = %q, want 'DRAM'", pt.Residency)
			}
			if pt.BandwidthGBps < StrixHaloSustainedDRAMBandwidthFloorGBps || pt.BandwidthGBps > StrixHaloSustainedDRAMBandwidthCeilGBps {
				t.Errorf("64 MB sweep point bandwidth = %.2f GB/s, want in [%.2f, %.2f]",
					pt.BandwidthGBps, StrixHaloSustainedDRAMBandwidthFloorGBps, StrixHaloSustainedDRAMBandwidthCeilGBps)
			}
		}
	}
	if !found16MB || !found64MB {
		t.Errorf("missing expected sweep points: found16MB=%v, found64MB=%v", found16MB, found64MB)
	}

	// 5. Test DetectMALLInflection boundary detection
	multiPoints := []SweepPoint{
		{SizeMB: 16, BandwidthGBps: 825.40, Residency: "MALL"},
		{SizeMB: 24, BandwidthGBps: 818.10, Residency: "MALL"},
		{SizeMB: 32, BandwidthGBps: 806.50, Residency: "MALL"},
		{SizeMB: 48, BandwidthGBps: 216.50, Residency: "DRAM"},
		{SizeMB: 64, BandwidthGBps: 214.20, Residency: "DRAM"},
	}
	boundaryMB, mallBW, dramBW, err := DetectMALLInflection(multiPoints)
	if err != nil {
		t.Fatalf("DetectMALLInflection failed: %v", err)
	}
	if boundaryMB != 32 {
		t.Errorf("boundary detected = %d MB, want 32 MB", boundaryMB)
	}
	if mallBW < 800.0 {
		t.Errorf("mallBW = %.2f GB/s, want >= 800.0", mallBW)
	}
	if dramBW < 210.0 || dramBW > 224.0 {
		t.Errorf("dramBW = %.2f GB/s, want in [210, 224]", dramBW)
	}

	// Edge case: < 2 points
	if _, _, _, err := DetectMALLInflection([]SweepPoint{{SizeMB: 16, BandwidthGBps: 825.40}}); err == nil {
		t.Error("expected error for < 2 sweep points, got nil")
	}

	// Edge case: Flat bandwidth (no inflection)
	flatPoints := []SweepPoint{
		{SizeMB: 16, BandwidthGBps: 210.0},
		{SizeMB: 64, BandwidthGBps: 209.0},
	}
	if _, _, _, err := DetectMALLInflection(flatPoints); err == nil {
		t.Error("expected error for flat bandwidth sweep, got nil")
	}
}

// TestWMMA_BF16_FP8_TFLOPS_Measurements verifies synthetic WMMA matrix multiply ceilings,
// theoretical peak alignment (59.4 TFLOPS BF16, 118.8 TFLOPS FP8), 2:1 FP8-to-BF16 ratio,
// and arithmetic intensity ridge points.
func TestWMMA_BF16_FP8_TFLOPS_Measurements(t *testing.T) {
	cfg := DefaultProbeConfig()
	cfg.ForceMock = true

	receipt, err := RunRooflineProbe(cfg)
	if err != nil {
		t.Fatalf("RunRooflineProbe failed: %v", err)
	}

	// 1. Verify BF16 theoretical peak and sustained throughput
	if receipt.BF16TFLOPS != StrixHaloBF16TheoreticalPeakTFLOPS {
		t.Errorf("BF16 theoretical peak = %.2f TFLOPS, want %.2f TFLOPS",
			receipt.BF16TFLOPS, StrixHaloBF16TheoreticalPeakTFLOPS)
	}
	if receipt.ComputeCeiling.BF16TFLOPS < 50.0 {
		t.Errorf("BF16 sustained throughput = %.2f TFLOPS, want >= 50.0 TFLOPS",
			receipt.ComputeCeiling.BF16TFLOPS)
	}

	// 2. Verify FP8 theoretical peak and sustained throughput
	if receipt.FP8TFLOPS != StrixHaloFP8TheoreticalPeakTFLOPS {
		t.Errorf("FP8 theoretical peak = %.2f TFLOPS, want %.2f TFLOPS",
			receipt.FP8TFLOPS, StrixHaloFP8TheoreticalPeakTFLOPS)
	}
	if receipt.ComputeCeiling.FP8TFLOPS < 100.0 {
		t.Errorf("FP8 sustained throughput = %.2f TFLOPS, want >= 100.0 TFLOPS",
			receipt.ComputeCeiling.FP8TFLOPS)
	}

	// 3. Verify exact 2:1 ratio for FP8 vs BF16
	ratio := receipt.FP8TFLOPS / receipt.BF16TFLOPS
	if math.Abs(ratio-2.0) > 0.001 {
		t.Errorf("FP8-to-BF16 peak ratio = %.4f, want 2.0000", ratio)
	}

	// 4. Verify empirical ridge point calculation: RidgePoint = (BF16 TFLOPS * 1000) / DRAM Bandwidth
	expectedRidge := math.Round(((receipt.BF16TFLOPS*1000.0)/receipt.DRAMSustainedBandwidthGBps)*100) / 100
	if math.Abs(receipt.RidgePoint-expectedRidge) > 0.01 {
		t.Errorf("receipt.RidgePoint = %.2f FLOP/Byte, want %.2f FLOP/Byte", receipt.RidgePoint, expectedRidge)
	}

	// Ridge point on Strix Halo is typically in ~265 - 285 FLOP/Byte range
	if receipt.RidgePoint < 250.0 || receipt.RidgePoint > 300.0 {
		t.Errorf("receipt.RidgePoint = %.2f FLOP/Byte outside expected [250, 300] range", receipt.RidgePoint)
	}
}

// TestEmpiricalReceipt_ValidationAndSerialization verifies schema compliance, cryptographic
// digest calculation, signature generation, JSON round-trip, and tampering detection.
func TestEmpiricalReceipt_ValidationAndSerialization(t *testing.T) {
	cfg := DefaultProbeConfig()
	cfg.ForceMock = true

	receipt, err := RunRooflineProbe(cfg)
	if err != nil {
		t.Fatalf("RunRooflineProbe failed: %v", err)
	}

	// 1. Verify basic properties
	if receipt.Schema != EmpiricalRooflineSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, EmpiricalRooflineSchema)
	}
	if receipt.Device != DefaultStrixHaloArch {
		t.Errorf("device = %q, want %q", receipt.Device, DefaultStrixHaloArch)
	}
	if receipt.ComputeUnits != StrixHaloDefaultComputeUnits {
		t.Errorf("compute_units = %d, want %d", receipt.ComputeUnits, StrixHaloDefaultComputeUnits)
	}
	if receipt.BusWidthBits != StrixHaloBusWidthBits {
		t.Errorf("bus_width_bits = %d, want %d", receipt.BusWidthBits, StrixHaloBusWidthBits)
	}
	if receipt.MemoryType != StrixHaloMemoryType {
		t.Errorf("memory_type = %q, want %q", receipt.MemoryType, StrixHaloMemoryType)
	}
	if !receipt.Verified {
		t.Errorf("receipt.Verified = false, want true")
	}
	if !strings.HasPrefix(receipt.Digest, "sha256:") {
		t.Errorf("digest %q missing 'sha256:' prefix", receipt.Digest)
	}
	if !strings.HasPrefix(receipt.Signature, "fak-sig:sha256:") {
		t.Errorf("signature %q missing 'fak-sig:sha256:' prefix", receipt.Signature)
	}

	// 2. Invariant verification
	if err := receipt.Verify(); err != nil {
		t.Fatalf("receipt.Verify() failed: %v", err)
	}

	// 3. JSON serialization roundtrip
	jsonBytes, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt.JSON() error: %v", err)
	}

	var unmarshaled EmpiricalRooflineReceipt
	if err := json.Unmarshal(jsonBytes, &unmarshaled); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if err := unmarshaled.Verify(); err != nil {
		t.Errorf("unmarshaled.Verify() failed: %v", err)
	}
	if unmarshaled.MALLHitBandwidthGBps != receipt.MALLHitBandwidthGBps {
		t.Errorf("MALLHitBandwidthGBps = %.2f, want %.2f",
			unmarshaled.MALLHitBandwidthGBps, receipt.MALLHitBandwidthGBps)
	}
	if unmarshaled.DRAMSustainedBandwidthGBps != receipt.DRAMSustainedBandwidthGBps {
		t.Errorf("DRAMSustainedBandwidthGBps = %.2f, want %.2f",
			unmarshaled.DRAMSustainedBandwidthGBps, receipt.DRAMSustainedBandwidthGBps)
	}
	if unmarshaled.BF16TFLOPS != receipt.BF16TFLOPS {
		t.Errorf("BF16TFLOPS = %.2f, want %.2f", unmarshaled.BF16TFLOPS, receipt.BF16TFLOPS)
	}
	if unmarshaled.FP8TFLOPS != receipt.FP8TFLOPS {
		t.Errorf("FP8TFLOPS = %.2f, want %.2f", unmarshaled.FP8TFLOPS, receipt.FP8TFLOPS)
	}
	if unmarshaled.RidgePoint != receipt.RidgePoint {
		t.Errorf("RidgePoint = %.2f, want %.2f", unmarshaled.RidgePoint, receipt.RidgePoint)
	}

	// 4. Verify explicit schema dictionary keys exist in raw JSON
	var rawMap map[string]any
	if err := json.Unmarshal(jsonBytes, &rawMap); err != nil {
		t.Fatalf("unmarshal to map failed: %v", err)
	}
	requiredKeys := []string{
		"schema", "device", "mall_hit_bandwidth_gbps", "dram_sustained_bandwidth_gbps",
		"bf16_tflops", "fp8_tflops", "ridge_point", "timestamp", "signature", "digest", "verified",
	}
	for _, key := range requiredKeys {
		if _, ok := rawMap[key]; !ok {
			t.Errorf("missing required key %q in JSON output", key)
		}
	}

	// 5. Tampering tests
	t.Run("tampered schema fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.Schema = "invalid.schema/v2"
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for tampered schema, got nil")
		}
	})

	t.Run("tampered device fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.Device = "gfx1100"
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for tampered device, got nil")
		}
	})

	t.Run("tampered compute units fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.ComputeUnits = 32
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for tampered compute units, got nil")
		}
	})

	t.Run("tampered low DRAM bandwidth fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.DRAMSustainedBandwidthGBps = 150.0
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for low DRAM bandwidth, got nil")
		}
	})

	t.Run("tampered MALL <= DRAM bandwidth fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.MALLHitBandwidthGBps = 200.0
		if err := tampered.Verify(); err == nil {
			t.Error("expected error when MALL bandwidth <= DRAM bandwidth, got nil")
		}
	})

	t.Run("tampered BF16 TFLOPS fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.BF16TFLOPS = 0
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for zero BF16 TFLOPS, got nil")
		}
	})

	t.Run("tampered digest fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
		if err := tampered.Verify(); err == nil {
			t.Error("expected error for altered digest, got nil")
		}
	})

	t.Run("unverified receipt fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.Verified = false
		if err := tampered.Verify(); err == nil {
			t.Error("expected error when Verified=false, got nil")
		}
	})

	t.Run("physical execution without witness fails verify", func(t *testing.T) {
		tampered := *receipt
		tampered.Simulated = false
		tampered.ExecutionWitness = ""
		err := tampered.Verify()
		if err == nil {
			t.Error("expected error when Simulated=false without ExecutionWitness, got nil")
		}
		if !strings.Contains(err.Error(), "lacks execution witness") {
			t.Errorf("expected error mentioning 'lacks execution witness', got: %v", err)
		}
	})
}

// TestFallbackBehavior_MockPlatform verifies automatic software probe activation
// on development machines, explicit mock forcing, and error handling when mock fallback is disabled.
func TestFallbackBehavior_MockPlatform(t *testing.T) {
	// 1. Explicit ForceMock: true returns a verified simulated receipt
	t.Run("force mock produces verified receipt", func(t *testing.T) {
		receipt, err := RunRooflineProbe(ProbeConfig{ForceMock: true})
		if err != nil {
			t.Fatalf("RunRooflineProbe(ForceMock) failed: %v", err)
		}
		if !receipt.Simulated {
			t.Error("receipt.Simulated = false, want true")
		}
		if err := receipt.Verify(); err != nil {
			t.Errorf("receipt.Verify() failed: %v", err)
		}
	})

	// 2. Default configuration automatically activates software probe on non-gfx1151 dev host
	t.Run("default configuration activates fallback on dev machine", func(t *testing.T) {
		receipt, err := RunRooflineProbe(DefaultProbeConfig())
		if err != nil {
			t.Fatalf("RunRooflineProbe(DefaultProbeConfig) failed: %v", err)
		}
		if err := receipt.Verify(); err != nil {
			t.Errorf("receipt.Verify() failed: %v", err)
		}
	})

	// 3. Disabling mock fallback on a non-physical host returns ErrDeviceNotFound
	t.Run("disabling mock fallback on non-gfx1151 returns ErrDeviceNotFound", func(t *testing.T) {
		if isPhysicalGFX1151() {
			t.Skip("skipping on physical gfx1151 device")
		}
		cfg := DefaultProbeConfig()
		cfg.MockFallback = false
		cfg.ForceMock = false
		_, err := RunRooflineProbe(cfg)
		if err == nil {
			t.Error("expected error when MockFallback=false on non-physical host, got nil")
		}
		if !errors.Is(err, ErrDeviceNotFound) {
			t.Errorf("expected ErrDeviceNotFound, got: %v", err)
		}
	})

	// 4. Pre-cancelled context returns context error
	t.Run("context cancellation terminates probe", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := RunRooflineProbeWithContext(ctx, DefaultProbeConfig())
		if err == nil {
			t.Error("expected error with cancelled context, got nil")
		}
	})
}

// TestRunRooflineProbeCLI verifies the CLI command execution, formatting, and file verification.
func TestRunRooflineProbeCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. JSON output
	code := RunRooflineProbeCLI(&stdout, &stderr, []string{"--device=gfx1151", "--json", "--mock"})
	if code != 0 {
		t.Fatalf("CLI exited with %d; stderr: %s", code, stderr.String())
	}
	var receipt EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal CLI JSON output: %v", err)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("receipt.Device = %q, want 'gfx1151'", receipt.Device)
	}
	if err := receipt.Verify(); err != nil {
		t.Errorf("CLI receipt failed verification: %v", err)
	}

	// 2. Subcommand prefix: `roofline --device=gfx1151 --json`
	stdout.Reset()
	stderr.Reset()
	code = RunRooflineProbeCLI(&stdout, &stderr, []string{"roofline", "--device=gfx1151", "--json", "--mock"})
	if code != 0 {
		t.Fatalf("CLI with subcommand prefix exited with %d: %s", code, stderr.String())
	}

	// 3. Human readable output
	stdout.Reset()
	stderr.Reset()
	code = RunRooflineProbeCLI(&stdout, &stderr, []string{"--device=gfx1151", "--mock"})
	if code != 0 {
		t.Fatalf("CLI text output exited with %d: %s", code, stderr.String())
	}
	outText := stdout.String()
	for _, substr := range []string{
		"AMD Strix Halo (gfx1151)",
		"Coalesced 256-bit DRAM Read Streaming",
		"Stepped MALL Cache Boundary Sweep",
		"Synthetic WMMA Compute Ceilings",
		"Empirical Ridge Point",
		"VERIFIED",
	} {
		if !strings.Contains(outText, substr) {
			t.Errorf("output missing expected text %q; output was:\n%s", substr, outText)
		}
	}

	// 4. Write to file and verify via --verify
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "probe_receipt.json")

	stdout.Reset()
	stderr.Reset()
	code = RunRooflineProbeCLI(&stdout, &stderr, []string{"--device=gfx1151", "--json", "--mock", "--out=" + outPath})
	if code != 0 {
		t.Fatalf("CLI --out exited with %d: %s", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = RunRooflineProbeCLI(&stdout, &stderr, []string{"--verify=" + outPath})
	if code != 0 {
		t.Fatalf("CLI --verify exited with %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "VALID") {
		t.Errorf("expected 'VALID' in verification output, got: %s", stdout.String())
	}

	// 5. Invalid flag returns code 2
	stdout.Reset()
	stderr.Reset()
	code = RunRooflineProbeCLI(&stdout, &stderr, []string{"--unrecognized-flag"})
	if code != 2 {
		t.Errorf("expected exit code 2 for invalid flag, got %d", code)
	}
}

// TestRooflineProbe_RunPhysicalProbe_Unwitnessed verifies that runPhysicalProbe fails closed
// and returns ErrPhysicalExecutionUnwitnessed instead of relabeling analytical probe receipts.
func TestRooflineProbe_RunPhysicalProbe_Unwitnessed(t *testing.T) {
	cfg := DefaultProbeConfig()

	// 1. Normal call fails closed with ErrPhysicalExecutionUnwitnessed
	receipt, err := runPhysicalProbe(context.Background(), cfg)
	if receipt != nil {
		t.Errorf("expected nil receipt from runPhysicalProbe, got %+v", receipt)
	}
	if !errors.Is(err, ErrPhysicalExecutionUnwitnessed) {
		t.Fatalf("expected ErrPhysicalExecutionUnwitnessed, got: %v", err)
	}

	// 2. Pre-cancelled context returns context error
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = runPhysicalProbe(ctx, cfg)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

// TestRooflineProbe_Verify_PhysicalRequiresExecutionWitness verifies that receipts claiming physical device
// execution (Simulated=false) must supply an ExecutionWitness to pass Verify().
func TestRooflineProbe_Verify_PhysicalRequiresExecutionWitness(t *testing.T) {
	cfg := DefaultProbeConfig()
	cfg.ForceMock = true

	receipt, err := RunRooflineProbe(cfg)
	if err != nil {
		t.Fatalf("RunRooflineProbe failed: %v", err)
	}

	// Claim physical execution without an execution witness
	unwitnessed := *receipt
	unwitnessed.Simulated = false
	unwitnessed.ExecutionWitness = ""
	if err := unwitnessed.Sign(nil); err != nil {
		t.Fatalf("Sign failed: %v", err)
	}

	err = unwitnessed.Verify()
	if err == nil {
		t.Fatal("expected error when Simulated=false without ExecutionWitness, got nil")
	}
	if !strings.Contains(err.Error(), "lacks execution witness") {
		t.Errorf("expected error mentioning 'lacks execution witness', got: %v", err)
	}

	// Supply a valid execution witness and re-sign -> should pass Verify()
	witnessed := *receipt
	witnessed.Simulated = false
	witnessed.ExecutionWitness = "rocm:kfd:gfx1151:kernel_dispatch:0xdeadbeef"
	if err := witnessed.Sign(nil); err != nil {
		t.Fatalf("Sign failed: %v", err)
	}
	if err := witnessed.Verify(); err != nil {
		t.Errorf("expected successful Verify() with ExecutionWitness, got: %v", err)
	}
}

// TestRooflineProbe_PhysicalHostFallback verifies that when physical GFX1151 detection is active,
// runPhysicalProbe returns ErrPhysicalExecutionUnwitnessed, falling back to mock software probe
// if MockFallback is enabled, or returning ErrPhysicalExecutionUnwitnessed if MockFallback is disabled.
func TestRooflineProbe_PhysicalHostFallback(t *testing.T) {
	origFn := isPhysicalGFX1151Fn
	defer func() { isPhysicalGFX1151Fn = origFn }()
	isPhysicalGFX1151Fn = func() bool { return true }

	// 1. With MockFallback=true (default): falls through to software calibration with Simulated=true
	cfgFallback := DefaultProbeConfig()
	cfgFallback.MockFallback = true
	cfgFallback.ForceMock = false

	receipt, err := RunRooflineProbe(cfgFallback)
	if err != nil {
		t.Fatalf("RunRooflineProbe with MockFallback=true failed: %v", err)
	}
	if !receipt.Simulated {
		t.Errorf("expected fallback receipt to have Simulated=true, got %v", receipt.Simulated)
	}
	if err := receipt.Verify(); err != nil {
		t.Errorf("fallback receipt failed Verify(): %v", err)
	}

	// 2. With MockFallback=false: returns ErrPhysicalExecutionUnwitnessed
	cfgStrict := DefaultProbeConfig()
	cfgStrict.MockFallback = false
	cfgStrict.ForceMock = false

	_, err = RunRooflineProbe(cfgStrict)
	if err == nil {
		t.Fatal("expected error when MockFallback=false on physical host with unwitnessed probe, got nil")
	}
	if !errors.Is(err, ErrPhysicalExecutionUnwitnessed) {
		t.Fatalf("expected ErrPhysicalExecutionUnwitnessed, got: %v", err)
	}
}
