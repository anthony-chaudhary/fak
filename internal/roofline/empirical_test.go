package roofline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestMeasureEmpiricalRoofline_SchemaCompliance verifies schema, invariants, and cryptographic digest.
func TestMeasureEmpiricalRoofline_SchemaCompliance(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	receipt, err := MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("MeasureEmpiricalRoofline failed: %v", err)
	}

	if receipt.Schema != EmpiricalRooflineSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, EmpiricalRooflineSchema)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("device = %q, want gfx1151", receipt.Device)
	}
	if receipt.ComputeUnits != 40 {
		t.Errorf("compute units = %d, want 40", receipt.ComputeUnits)
	}
	if receipt.BusWidthBits != 256 {
		t.Errorf("bus width = %d, want 256", receipt.BusWidthBits)
	}
	if receipt.MemoryType != "LPDDR5X-8533" {
		t.Errorf("memory type = %q, want LPDDR5X-8533", receipt.MemoryType)
	}
	if receipt.Architecture != "RDNA 3.5" {
		t.Errorf("architecture = %q, want RDNA 3.5", receipt.Architecture)
	}
	if !receipt.Verified {
		t.Errorf("receipt.Verified = false, want true")
	}
	if !strings.HasPrefix(receipt.Digest, "sha256:") {
		t.Errorf("digest %q does not start with sha256:", receipt.Digest)
	}

	// Verify the receipt passes internal cryptographic verification
	if err := receipt.Verify(); err != nil {
		t.Errorf("receipt.Verify() failed: %v", err)
	}

	// Test JSON serialization round-trip
	raw, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt.JSON() error: %v", err)
	}

	var decoded EmpiricalRooflineReceipt
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if decoded.Schema != receipt.Schema {
		t.Errorf("decoded schema = %q, want %q", decoded.Schema, receipt.Schema)
	}
	if decoded.Digest != receipt.Digest {
		t.Errorf("decoded digest = %q, want %q", decoded.Digest, receipt.Digest)
	}
	if err := decoded.Verify(); err != nil {
		t.Errorf("decoded.Verify() failed: %v", err)
	}
}

// TestMeasureEmpiricalRoofline_TamperingFailsVerification checks that altered fields fail Verify().
func TestMeasureEmpiricalRoofline_TamperingFailsVerification(t *testing.T) {
	ctx := context.Background()
	receipt, err := MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("MeasureEmpiricalRoofline failed: %v", err)
	}

	// 1. Alter schema
	tampered := *receipt
	tampered.Schema = "fak.roofline.empirical/v2"
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error for tampered schema, got nil")
	}

	// 2. Alter DRAM bandwidth
	tampered = *receipt
	tampered.DRAMBandwidth.SustainedGBps = -10
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error for negative DRAM bandwidth, got nil")
	}

	// 3. Alter MALL bandwidth to be lower than DRAM
	tampered = *receipt
	tampered.MALLSweep.WithinMALLBandwidthGBps = 50.0
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error when MALL bandwidth <= DRAM bandwidth, got nil")
	}

	// 4. Invalidate boundary
	tampered = *receipt
	tampered.MALLSweep.BoundaryDetectedMB = 0
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error when boundaryDetected <= 0, got nil")
	}

	// 5. Alter compute ceiling
	tampered = *receipt
	tampered.ComputeCeiling.FP16TFLOPS = 0
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error when FP16TFLOPS <= 0, got nil")
	}

	// 6. Corrupt digest
	tampered = *receipt
	tampered.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error for corrupted digest, got nil")
	}

	// 7. Not verified
	tampered = *receipt
	tampered.Verified = false
	if err := tampered.Verify(); err == nil {
		t.Errorf("expected error when Verified=false, got nil")
	}
}

// TestMALLBoundaryDetection verifies that DetectMALLBoundary correctly identifies the 32MB cache boundary.
func TestMALLBoundaryDetection(t *testing.T) {
	// Standard stepped working set sweep from 16MB to 64MB
	points := []SweepPoint{
		{SizeMB: 16, BandwidthGBps: 808.50, Residency: "MALL"},
		{SizeMB: 24, BandwidthGBps: 802.10, Residency: "MALL"},
		{SizeMB: 32, BandwidthGBps: 794.60, Residency: "MALL"},
		{SizeMB: 48, BandwidthGBps: 211.50, Residency: "DRAM"},
		{SizeMB: 64, BandwidthGBps: 209.40, Residency: "DRAM"},
	}

	boundaryMB, withinMALLBW, dramBW, err := DetectMALLBoundary(points)
	if err != nil {
		t.Fatalf("DetectMALLBoundary failed: %v", err)
	}

	if boundaryMB != 32 {
		t.Errorf("boundaryMB = %d, want 32", boundaryMB)
	}

	// Expect within-MALL bandwidth around ~800 GB/s (e.g. 790 - 815)
	if withinMALLBW < 790.0 || withinMALLBW > 815.0 {
		t.Errorf("withinMALLBW = %.2f GB/s, want ~800 GB/s", withinMALLBW)
	}

	// Expect DRAM spill bandwidth around ~210 GB/s (e.g. 205 - 215)
	if dramBW < 205.0 || dramBW > 215.0 {
		t.Errorf("dramBW = %.2f GB/s, want ~210 GB/s", dramBW)
	}

	dropRatio := withinMALLBW / dramBW
	if dropRatio < 3.5 || dropRatio > 4.2 {
		t.Errorf("dropRatio = %.2fx, want ~3.8x", dropRatio)
	}
}

// TestMALLBoundaryDetection_EdgeCases tests edge cases for boundary detection.
func TestMALLBoundaryDetection_EdgeCases(t *testing.T) {
	// 1. Single point
	_, _, _, err := DetectMALLBoundary([]SweepPoint{{SizeMB: 16, BandwidthGBps: 800}})
	if err == nil {
		t.Errorf("expected error for single point, got nil")
	}

	// 2. Flat profile without significant drop
	flatPoints := []SweepPoint{
		{SizeMB: 16, BandwidthGBps: 210.0},
		{SizeMB: 32, BandwidthGBps: 208.0},
		{SizeMB: 64, BandwidthGBps: 205.0},
	}
	_, _, _, err = DetectMALLBoundary(flatPoints)
	if err == nil {
		t.Errorf("expected error for flat profile, got nil")
	}
}

// TestDRAMBandwidthCalculation verifies theoretical peak and sustained DRAM bandwidth math.
func TestDRAMBandwidthCalculation(t *testing.T) {
	// Strix Halo: 256-bit bus, LPDDR5X-8533 MT/s
	// (256 / 8 bytes) * 8533.333333 * 10^6 / 10^9 = 32 * 8.533 = 273.056 GB/s
	theoreticalPeak := CalculateTheoreticalDRAMBandwidth(256, 8533.0)
	if math.Abs(theoreticalPeak-273.056) > 0.1 {
		t.Errorf("theoretical peak = %.3f GB/s, want ~273.056 GB/s", theoreticalPeak)
	}

	ctx := context.Background()
	receipt, err := MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("MeasureEmpiricalRoofline failed: %v", err)
	}

	// Sustainable DRAM bandwidth ~210 GB/s on 256-bit bus vs theoretical 273.056 GB/s
	if math.Abs(receipt.DRAMBandwidth.SustainedGBps-210.0) > 1.0 {
		t.Errorf("sustained DRAM BW = %.2f GB/s, want ~210.0 GB/s", receipt.DRAMBandwidth.SustainedGBps)
	}
	if receipt.DRAMBandwidth.ActiveCUs != 40 {
		t.Errorf("active CUs = %d, want 40", receipt.DRAMBandwidth.ActiveCUs)
	}
	if receipt.DRAMBandwidth.Efficiency < 0.75 || receipt.DRAMBandwidth.Efficiency > 0.80 {
		t.Errorf("efficiency = %.4f, want ~0.7691", receipt.DRAMBandwidth.Efficiency)
	}

	// Knee point verification
	// DRAM Knee FP16: 60,000 / 210.0 = 285.71 FLOP/byte
	if math.Abs(receipt.KneePoints.DRAMKneeFP16FLOPPerByte-285.71) > 1.0 {
		t.Errorf("DRAM FP16 knee = %.2f FLOP/B, want ~285.71", receipt.KneePoints.DRAMKneeFP16FLOPPerByte)
	}
	// DRAM Knee FP8: 120,000 / 210.0 = 571.43 OP/byte
	if math.Abs(receipt.KneePoints.DRAMKneeFP8OPPerByte-571.43) > 1.0 {
		t.Errorf("DRAM FP8 knee = %.2f OP/B, want ~571.43", receipt.KneePoints.DRAMKneeFP8OPPerByte)
	}
	// MALL Knee FP16: 60,000 / ~801.7 = ~74.84 FLOP/byte
	if receipt.KneePoints.MALLKneeFP16FLOPPerByte < 70.0 || receipt.KneePoints.MALLKneeFP16FLOPPerByte > 80.0 {
		t.Errorf("MALL FP16 knee = %.2f FLOP/B, want ~75.0", receipt.KneePoints.MALLKneeFP16FLOPPerByte)
	}
	// MALL Knee FP8: 120,000 / ~801.7 = ~149.68 OP/byte
	if receipt.KneePoints.MALLKneeFP8OPPerByte < 140.0 || receipt.KneePoints.MALLKneeFP8OPPerByte > 160.0 {
		t.Errorf("MALL FP8 knee = %.2f OP/B, want ~150.0", receipt.KneePoints.MALLKneeFP8OPPerByte)
	}
}

// TestWMMAComputeCeilings verifies synthetic WMMA matrix-multiply throughput numbers.
func TestWMMAComputeCeilings(t *testing.T) {
	ctx := context.Background()
	receipt, err := MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("MeasureEmpiricalRoofline failed: %v", err)
	}

	if receipt.ComputeCeiling.BlockDimensions != "16x16x16" {
		t.Errorf("block dimensions = %q, want 16x16x16", receipt.ComputeCeiling.BlockDimensions)
	}
	if math.Abs(receipt.ComputeCeiling.FP16TFLOPS-60.0) > 1.0 {
		t.Errorf("FP16 TFLOPS = %.2f, want ~60.0", receipt.ComputeCeiling.FP16TFLOPS)
	}
	if math.Abs(receipt.ComputeCeiling.FP8TOPS-120.0) > 1.0 {
		t.Errorf("FP8 TOPS = %.2f, want ~120.0", receipt.ComputeCeiling.FP8TOPS)
	}
	if math.Abs(receipt.ComputeCeiling.INT8TOPS-120.0) > 1.0 {
		t.Errorf("INT8 TOPS = %.2f, want ~120.0", receipt.ComputeCeiling.INT8TOPS)
	}
	if receipt.ComputeCeiling.ActiveCUs != 40 {
		t.Errorf("active CUs = %d, want 40", receipt.ComputeCeiling.ActiveCUs)
	}
}

// TestArchitectureNormalization verifies normalization of supported device strings.
func TestArchitectureNormalization(t *testing.T) {
	valid := []string{
		"gfx1151",
		"GFX1151",
		"strix-halo",
		"Strix-Halo",
		"strixhalo",
		"strix-halo-128",
		"strix-halo-64",
		"ryzen ai max+ 395",
		"Ryzen AI Max+ 395",
		"radeon 8060s",
		"",
	}

	for _, v := range valid {
		norm, err := NormalizeArchitecture(v)
		if err != nil {
			t.Errorf("NormalizeArchitecture(%q) unexpected error: %v", v, err)
		}
		if norm != DefaultArchStrixHalo {
			t.Errorf("NormalizeArchitecture(%q) = %q, want %q", v, norm, DefaultArchStrixHalo)
		}
	}

	invalid := []string{
		"nvidia_h100",
		"blackwell",
		"intel_gaudi",
		"unknown_chipset",
	}

	for _, inv := range invalid {
		if _, err := NormalizeArchitecture(inv); err == nil {
			t.Errorf("NormalizeArchitecture(%q) expected error, got nil", inv)
		}
	}
}

// TestContextCancellation verifies that MeasureEmpiricalRoofline honors context cancellation.
func TestContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := MeasureEmpiricalRoofline(ctx, "gfx1151")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

// TestCLIInvocation tests the RunCLI function with various flag combinations.
func TestCLIInvocation(t *testing.T) {
	// 1. JSON mode
	var stdout, stderr bytes.Buffer
	code := RunCLI(&stdout, &stderr, []string{"--device=gfx1151", "--json"})
	if code != 0 {
		t.Fatalf("RunCLI --json failed with code %d: %s", code, stderr.String())
	}

	var receipt EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode JSON output: %v", err)
	}
	if receipt.Schema != EmpiricalRooflineSchema {
		t.Errorf("receipt schema = %q, want %q", receipt.Schema, EmpiricalRooflineSchema)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("receipt device = %q, want gfx1151", receipt.Device)
	}

	// 2. Human readable mode
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"--device=gfx1151"})
	if code != 0 {
		t.Fatalf("RunCLI human mode failed with code %d: %s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "AMD Strix Halo (gfx1151)") {
		t.Errorf("human output missing device header: %s", outStr)
	}
	if !strings.Contains(outStr, "32 MB") {
		t.Errorf("human output missing 32 MB MALL boundary: %s", outStr)
	}

	// 3. Subcommand prefix "roofline"
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"roofline", "--device=gfx1151", "--json"})
	if code != 0 {
		t.Fatalf("RunCLI with subcommand prefix failed: %s", stderr.String())
	}

	// 4. File output and verify
	tmpDir := t.TempDir()
	receiptFile := filepath.Join(tmpDir, "test_receipt.json")

	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"--device=gfx1151", "--out=" + receiptFile})
	if code != 0 {
		t.Fatalf("RunCLI with --out failed: %s", stderr.String())
	}

	if _, err := os.Stat(receiptFile); err != nil {
		t.Fatalf("receipt file not created at %s: %v", receiptFile, err)
	}

	// Verify using --verify
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"--verify=" + receiptFile})
	if code != 0 {
		t.Fatalf("RunCLI --verify failed: %s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "VALID") {
		t.Errorf("expected VALID in verify output, got: %s", stdout.String())
	}

	// 5. Invalid device
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"--device=nonexistent_device"})
	if code == 0 {
		t.Errorf("expected error exit code for invalid device, got 0")
	}

	// 6. Help
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"--help"})
	if code != 0 && code != 2 {
		t.Errorf("unexpected exit code for --help: %d", code)
	}
}
