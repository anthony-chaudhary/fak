package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rocm_arch_test.go — host-tractable witnesses for the ROCm device-arch taxonomy (#266).
// Every assertion here is hardware-independent: it checks the gfx→family mapping, the
// CDNA/RDNA split, and the offload-arch normalization that a HIP build keys on, none of
// which needs an AMD GPU. The device run (a HIP kernel actually executing on AMD silicon)
// is the deferred half — see ROCM-C002-NOTES.md.

func TestROCmArchLookupKnown(t *testing.T) {
	cases := []struct {
		gfx        string
		family     ROCmFamily
		wavefront  int
		datacenter bool
	}{
		{"gfx906", ROCmGCN5, 64, true},
		{"gfx908", ROCmCDNA1, 64, true},
		{"gfx90a", ROCmCDNA2, 64, true},
		{"gfx942", ROCmCDNA3, 64, true},
		{"gfx1010", ROCmRDNA1, 32, false},
		{"gfx1030", ROCmRDNA2, 32, false},
		{"gfx1032", ROCmRDNA2, 32, false},
		{"gfx1100", ROCmRDNA3, 32, false},
		{"gfx1102", ROCmRDNA3, 32, false},
		{"gfx1151", ROCmRDNA3_5, 32, false},
	}
	for _, c := range cases {
		a, ok := LookupROCmArch(c.gfx)
		if !ok {
			t.Fatalf("LookupROCmArch(%q): not found, want supported", c.gfx)
		}
		if a.Family != c.family {
			t.Errorf("%s: family = %v, want %v", c.gfx, a.Family, c.family)
		}
		if a.Wavefront != c.wavefront {
			t.Errorf("%s: wavefront = %d, want %d", c.gfx, a.Wavefront, c.wavefront)
		}
		if a.Family.Datacenter() != c.datacenter {
			t.Errorf("%s: Datacenter() = %v, want %v", c.gfx, a.Family.Datacenter(), c.datacenter)
		}
	}
}

// TestROCmCDNARDNAInvariant pins the load-bearing split: every supported target is exactly
// one of CDNA, RDNA, or the GCN5 datacenter part — never both, never neither — and the
// wavefront width is the one its family mandates (64 for GCN/CDNA, 32 for RDNA). A new row
// added to the table with a mistyped family or wavefront fails here.
func TestROCmCDNARDNAInvariant(t *testing.T) {
	for _, a := range KnownROCmArches() {
		if a.Family.IsCDNA() && a.Family.IsRDNA() {
			t.Errorf("%s: family %v claims both CDNA and RDNA", a.GFX, a.Family)
		}
		isGCN5 := a.Family == ROCmGCN5
		if !a.Family.IsCDNA() && !a.Family.IsRDNA() && !isGCN5 {
			t.Errorf("%s: family %v is neither CDNA, RDNA, nor GCN5", a.GFX, a.Family)
		}
		if a.Family.IsRDNA() {
			if a.Wavefront != 32 {
				t.Errorf("%s: RDNA wavefront = %d, want 32", a.GFX, a.Wavefront)
			}
			if a.Family.Datacenter() {
				t.Errorf("%s: RDNA classified as datacenter", a.GFX)
			}
		} else { // GCN5 or CDNA — the 64-lane Instinct datacenter line
			if a.Wavefront != 64 {
				t.Errorf("%s: %v wavefront = %d, want 64", a.GFX, a.Family, a.Wavefront)
			}
			if !a.Family.Datacenter() {
				t.Errorf("%s: %v not classified as datacenter", a.GFX, a.Family)
			}
		}
	}
}

// TestROCmOffloadArchNormalization checks that the offload-arch selector accepts the noisy
// forms ROCm actually reports (case, whitespace, a target-feature suffix) and canonicalizes
// them to the bare gfx token hipcc wants — while still rejecting an unsupported part.
func TestROCmOffloadArchNormalization(t *testing.T) {
	for _, in := range []string{"gfx90a", "GFX90A", "  gfx90a ", "gfx90a:sramecc+:xnack-"} {
		got, ok := ROCmOffloadArch(in)
		if !ok || got != "gfx90a" {
			t.Errorf("ROCmOffloadArch(%q) = (%q,%v), want (gfx90a,true)", in, got, ok)
		}
	}
	for _, in := range []string{"gfx700", "sm_90", "", "gfx", "rdna3"} {
		if got, ok := ROCmOffloadArch(in); ok {
			t.Errorf("ROCmOffloadArch(%q) = (%q,true), want unsupported", in, got)
		}
	}
}

// TestROCmRX7600IsRDNA3 ties the taxonomy to a real, already-witnessed AMD card: the RX 7600
// (gfx1102) that the Vulkan backend runs on with numerical parity (VULKAN-AMD-RESULTS.md) is
// the consumer-RDNA3 target the ROCm backend will compile for via --offload-arch=gfx1102.
func TestROCmRX7600IsRDNA3(t *testing.T) {
	a, ok := LookupROCmArch("gfx1102")
	if !ok {
		t.Fatal("gfx1102 (RX 7600) must be a supported target")
	}
	if a.Family != ROCmRDNA3 || a.Family.Datacenter() || a.Wavefront != 32 {
		t.Errorf("gfx1102 = %+v, want RDNA3 consumer wave32", a)
	}
	if off, ok := ROCmOffloadArch("gfx1102"); !ok || off != "gfx1102" {
		t.Errorf("ROCmOffloadArch(gfx1102) = (%q,%v), want (gfx1102,true)", off, ok)
	}
}

// TestROCmGfx1151IsRDNA3_5 witnesses AMD Strix Halo (Ryzen AI Max+ 395, 40 CUs RDNA 3.5)
// recognition in the ROCm architecture taxonomy.
func TestROCmGfx1151IsRDNA3_5(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 (Strix Halo APU) must be a supported target")
	}
	if a.Family != ROCmRDNA3_5 {
		t.Errorf("gfx1151 family = %v, want ROCmRDNA3_5", a.Family)
	}
	if a.Family.String() != "RDNA3.5" {
		t.Errorf("gfx1151 family string = %q, want RDNA3.5", a.Family.String())
	}
	if a.Wavefront != 32 {
		t.Errorf("gfx1151 wavefront = %d, want 32", a.Wavefront)
	}
	if a.Family.Datacenter() {
		t.Errorf("gfx1151 classified as datacenter, want false")
	}
	if !a.Family.IsRDNA() {
		t.Errorf("gfx1151 IsRDNA() = false, want true")
	}
	if a.Family.IsCDNA() {
		t.Errorf("gfx1151 IsCDNA() = true, want false")
	}
	if off, ok := ROCmOffloadArch("gfx1151"); !ok || off != "gfx1151" {
		t.Errorf("ROCmOffloadArch(gfx1151) = (%q,%v), want (gfx1151,true)", off, ok)
	}
}

// TestROCmGfx1151CompilerFlagsAndLDS verifies compiler flags and LDS tuning for 4-token speculative tree masks.
func TestROCmGfx1151CompilerFlagsAndLDS(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found")
	}
	flags := a.CompilerFlags()
	hasWave32 := false
	hasOffload := false
	for _, f := range flags {
		if f == "-mwavefrontsize32" {
			hasWave32 = true
		}
		if f == "--offload-arch=gfx1151" {
			hasOffload = true
		}
	}
	if !hasWave32 {
		t.Errorf("flags %+v missing -mwavefrontsize32", flags)
	}
	if !hasOffload {
		t.Errorf("flags %+v missing --offload-arch=gfx1151", flags)
	}

	lds := a.TuneMTPTreeMaskLDS(4, 128)
	if lds.DraftDepth != 4 {
		t.Errorf("lds.DraftDepth = %d, want 4", lds.DraftDepth)
	}
	if lds.TotalLDSBytes <= 0 || lds.TotalLDSBytes > 65536 {
		t.Errorf("lds.TotalLDSBytes = %d, must fit in 64KB CU LDS", lds.TotalLDSBytes)
	}
	if lds.MaxWavesPerCU != 32 {
		t.Errorf("lds.MaxWavesPerCU = %d, want 32", lds.MaxWavesPerCU)
	}
}

// TestTuneQSASparseGather verifies QSA sparse row gather tuning on AMD Strix Halo (gfx1151).
func TestTuneQSASparseGather(t *testing.T) {
	if !HasQSASparseRowGather() {
		t.Fatal("HasQSASparseRowGather must be true")
	}

	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found")
	}

	cfg := a.TuneQSASparseGather(256, 2, 2)
	if cfg.Arch != "gfx1151" {
		t.Errorf("cfg.Arch = %q, want gfx1151", cfg.Arch)
	}
	if cfg.TopKTokens != 2048 {
		t.Errorf("cfg.TopKTokens = %d, want 2048", cfg.TopKTokens)
	}
	if cfg.LocalTailTokens != 256 {
		t.Errorf("cfg.LocalTailTokens = %d, want 256", cfg.LocalTailTokens)
	}
	if cfg.TotalGatherTokens != 2304 {
		t.Errorf("cfg.TotalGatherTokens = %d, want 2304", cfg.TotalGatherTokens)
	}
	if cfg.NumTiles != 9 {
		t.Errorf("cfg.NumTiles = %d, want 9 (2304 / 256)", cfg.NumTiles)
	}
	if cfg.RadixLDSBytes != 1024 {
		t.Errorf("cfg.RadixLDSBytes = %d, want 1024", cfg.RadixLDSBytes)
	}
	if !cfg.FitsInInfinityCache {
		t.Errorf("cfg.FitsInInfinityCache = false, gathered scratch (%d bytes) must fit in 32MB MALL", cfg.GatherScratchBytes)
	}
	if cfg.MaxWavesPerCU != 32 {
		t.Errorf("cfg.MaxWavesPerCU = %d, want 32", cfg.MaxWavesPerCU)
	}
	if cfg.BandwidthSavingEst <= 0.50 {
		t.Errorf("cfg.BandwidthSavingEst = %f, want > 0.50", cfg.BandwidthSavingEst)
	}
}

// TestRadixTopKBlockSelect_DynamicGating verifies that below 16k tokens, dense attention is preserved.
func TestRadixTopKBlockSelect_DynamicGating(t *testing.T) {
	totalBlocks := 200 // 200 * 64 = 12,800 tokens (< 16,384)
	scores := make([]float32, totalBlocks)
	for i := range scores {
		scores[i] = float32(i) * 0.1
	}

	selected, receipt, err := RadixTopKBlockSelect(scores, totalBlocks, 32, 4)
	if err != nil {
		t.Fatalf("RadixTopKBlockSelect: %v", err)
	}
	if !receipt.DynamicGatingBypassed {
		t.Error("receipt.DynamicGatingBypassed must be true for < 16,384 tokens")
	}
	if len(selected) != totalBlocks {
		t.Errorf("len(selected) = %d, want all %d blocks", len(selected), totalBlocks)
	}
	for i, b := range selected {
		if b != int32(i) {
			t.Fatalf("selected[%d] = %d, want %d", i, b, i)
		}
	}
}

// TestRadixTopKBlockSelect_LongContext verifies top-k (2048 tok) + tail (256 tok) selection at long context.
func TestRadixTopKBlockSelect_LongContext(t *testing.T) {
	// 78k context: 78,000 / 64 = 1219 blocks
	totalBlocks := 1219
	scores := make([]float32, totalBlocks)
	for i := range scores {
		scores[i] = 1.0 // default baseline score
	}

	// Elevate 32 specific candidate blocks to high score
	highBlocks := []int{10, 25, 42, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
	for _, hb := range highBlocks {
		scores[hb] = 99.0
	}

	topKBlocks := 32
	tailBlocks := 4
	selected, receipt, err := RadixTopKBlockSelect(scores, totalBlocks, topKBlocks, tailBlocks)
	if err != nil {
		t.Fatalf("RadixTopKBlockSelect: %v", err)
	}
	if receipt.DynamicGatingBypassed {
		t.Error("DynamicGatingBypassed must be false at 78k context")
	}

	// Check tail blocks are present: totalBlocks-4 .. totalBlocks-1
	for b := totalBlocks - 4; b < totalBlocks; b++ {
		found := false
		for _, s := range selected {
			if s == int32(b) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("tail block %d missing from selection", b)
		}
	}

	// Check high score candidate blocks are present
	for _, hb := range highBlocks {
		found := false
		for _, s := range selected {
			if s == int32(hb) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("high score block %d missing from selection", hb)
		}
	}

	// Verify ascending ordering
	for i := 1; i < len(selected); i++ {
		if selected[i] <= selected[i-1] {
			t.Errorf("selected not strictly ascending: selected[%d]=%d <= selected[%d]=%d", i, selected[i], i-1, selected[i-1])
		}
	}

	if receipt.BandwidthSavingsRatio <= 0.50 {
		t.Errorf("BandwidthSavingsRatio = %f, want > 0.50 (>50%% savings)", receipt.BandwidthSavingsRatio)
	}
	if receipt.DRAMBytesEliminated <= 0 {
		t.Errorf("DRAMBytesEliminated = %d, want > 0", receipt.DRAMBytesEliminated)
	}
}

// TestSparseRowGatherKV_ContiguousReconstruction tests extracting selected rows into contiguous scratch.
func TestSparseRowGatherKV_ContiguousReconstruction(t *testing.T) {
	totalTokens := 1024
	blockSize := 64
	numKVHeads := 2
	headDim := 128
	rowWidth := numKVHeads * headDim

	srcK := make([]float32, totalTokens*rowWidth)
	srcV := make([]float32, totalTokens*rowWidth)
	for i := range srcK {
		srcK[i] = float32(i) * 0.01
		srcV[i] = float32(i) * 0.02
	}

	selectedBlocks := []int32{0, 3, 7, 15}
	gK, gV, err := SparseRowGatherKV(srcK, srcV, selectedBlocks, blockSize, numKVHeads, headDim, totalTokens)
	if err != nil {
		t.Fatalf("SparseRowGatherKV: %v", err)
	}

	expectedTokens := len(selectedBlocks) * blockSize
	if len(gK) != expectedTokens*rowWidth || len(gV) != expectedTokens*rowWidth {
		t.Fatalf("gathered len K=%d, V=%d, want %d", len(gK), len(gV), expectedTokens*rowWidth)
	}

	// Verify exact row contents
	dstTok := 0
	for _, b := range selectedBlocks {
		srcStartTok := int(b) * blockSize
		for row := 0; row < blockSize; row++ {
			srcRowOffset := (srcStartTok + row) * rowWidth
			dstRowOffset := (dstTok + row) * rowWidth
			for c := 0; c < rowWidth; c++ {
				if gK[dstRowOffset+c] != srcK[srcRowOffset+c] {
					t.Fatalf("K mismatch at block %d row %d c %d: got %f, want %f", b, row, c, gK[dstRowOffset+c], srcK[srcRowOffset+c])
				}
				if gV[dstRowOffset+c] != srcV[srcRowOffset+c] {
					t.Fatalf("V mismatch at block %d row %d c %d: got %f, want %f", b, row, c, gV[dstRowOffset+c], srcV[srcRowOffset+c])
				}
			}
		}
		dstTok += blockSize
	}
}

// TestROCm_Ticket514_Wave32WMMAAndPad2 verifies Ticket #514 requirements:
// - CompilerFlags() for gfx1151 enforces -mwavefrontsize32 and native Wave32 SIMD.
// - Cooperative matrix tile geometry tuned for RDNA 3.5 dual-issue WMMA (16x16x16 and 16x16x32 primitives).
// - LDS bank conflict Pad-2 alignment helper (+13% matmul speedup) with numerical bit-identity.
func TestROCm_Ticket514_Wave32WMMAAndPad2(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found in ROCm arch taxonomy")
	}

	// 1. Verify CompilerFlags() enforces -mwavefrontsize32 and native Wave32 SIMD
	flags := a.CompilerFlags()
	hasWave32 := false
	for _, f := range flags {
		if f == "-mwavefrontsize32" {
			hasWave32 = true
		}
	}
	if !hasWave32 {
		t.Fatalf("CompilerFlags() for gfx1151 missing -mwavefrontsize32: %v", flags)
	}
	if !a.HasNativeWave32WMMA() {
		t.Fatalf("gfx1151 HasNativeWave32WMMA() = false, want true")
	}

	// 2. Verify cooperative matrix tile geometry for dual-issue WMMA
	tiles := a.SupportedWMMATiles()
	if len(tiles) != 2 {
		t.Fatalf("SupportedWMMATiles() length = %d, want 2", len(tiles))
	}
	has16x16x16 := false
	has16x16x32 := false
	for _, tile := range tiles {
		if tile.SubgroupSize != 32 {
			t.Errorf("tile.SubgroupSize = %d, want 32", tile.SubgroupSize)
		}
		if tile.Primitive == WMMAPrimitive16x16x16 {
			has16x16x16 = true
			if tile.M != 16 || tile.N != 16 || tile.K != 16 || tile.Lanes != 32 || tile.SubgroupSize != 32 || !tile.DualIssue {
				t.Errorf("invalid 16x16x16 tile config: %+v", tile)
			}
		}
		if tile.Primitive == WMMAPrimitive16x16x32 {
			has16x16x32 = true
			if tile.M != 16 || tile.N != 16 || tile.K != 32 || tile.Lanes != 32 || tile.SubgroupSize != 32 || !tile.DualIssue {
				t.Errorf("invalid 16x16x32 tile config: %+v", tile)
			}
		}
	}
	if !has16x16x16 || !has16x16x32 {
		t.Fatalf("missing WMMA primitives: has16x16x16=%v has16x16x32=%v", has16x16x16, has16x16x32)
	}

	// Verify GEMM tuning selects correct primitives
	fp16Cfg, err := a.TuneCooperativeMatrixGEMM(128, 128, 128, "fp16")
	if err != nil {
		t.Fatalf("TuneCooperativeMatrixGEMM(fp16): %v", err)
	}
	if fp16Cfg.Primitive != WMMAPrimitive16x16x16 || fp16Cfg.TileK != 16 || !fp16Cfg.DualIssue {
		t.Errorf("fp16Cfg = %+v, want 16x16x16 dual-issue", fp16Cfg)
	}
	if fp16Cfg.SubgroupSize != 32 {
		t.Errorf("fp16Cfg.SubgroupSize = %d, want 32", fp16Cfg.SubgroupSize)
	}
	if fp16Cfg.PaddedStride != fp16Cfg.UnpaddedStride+2 {
		t.Errorf("fp16Cfg padded stride %d != unpadded %d + 2", fp16Cfg.PaddedStride, fp16Cfg.UnpaddedStride)
	}

	int8Cfg, err := a.TuneCooperativeMatrixGEMM(128, 128, 128, "int8")
	if err != nil {
		t.Fatalf("TuneCooperativeMatrixGEMM(int8): %v", err)
	}
	if int8Cfg.Primitive != WMMAPrimitive16x16x32 || int8Cfg.TileK != 32 || !int8Cfg.DualIssue {
		t.Errorf("int8Cfg = %+v, want 16x16x32 dual-issue", int8Cfg)
	}
	if int8Cfg.SubgroupSize != 32 {
		t.Errorf("int8Cfg.SubgroupSize = %d, want 32", int8Cfg.SubgroupSize)
	}

	// 3. Verify LDS bank conflict Pad-2 alignment helper (+13% matmul speedup)
	unpaddedCols := 16
	unpaddedReport := AnalyzeLDSBankConflicts(unpaddedCols, false)
	paddedReport := AnalyzeLDSBankConflicts(unpaddedCols, true)

	if unpaddedReport.ActiveBanks >= 16 {
		t.Errorf("unpadded active banks = %d, expected bank collision (<= 8)", unpaddedReport.ActiveBanks)
	}
	if unpaddedReport.BankConflictStalls == 0 {
		t.Errorf("unpadded bank conflict stalls = 0, expected stalls")
	}

	if paddedReport.ActiveBanks != 16 {
		t.Errorf("padded active banks = %d, want 16 (expanded bank coverage)", paddedReport.ActiveBanks)
	}
	if paddedReport.SpeedupEstimate != 1.13 {
		t.Errorf("padded speedup estimate = %f, want 1.13 (+13%% matmul speedup)", paddedReport.SpeedupEstimate)
	}
	if paddedReport.PaddedStrideWords != LDSBankPad2Stride(unpaddedCols) {
		t.Errorf("padded stride = %d, want %d", paddedReport.PaddedStrideWords, LDSBankPad2Stride(unpaddedCols))
	}

	// 4. Verify in-tree numerical bit-identity (zero degradation)
	M, N, K := 16, 16, 32
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = float32(i)*0.05 - 1.0
	}
	for i := range B {
		B[i] = float32(i)*0.03 - 0.5
	}

	C, report, err := VerifyLDSBankPad2MatMul(A, B, M, N, K)
	if err != nil {
		t.Fatalf("VerifyLDSBankPad2MatMul failed: %v", err)
	}
	if len(C) != M*N {
		t.Fatalf("len(C) = %d, want %d", len(C), M*N)
	}
	if report.ActiveBanks != 16 {
		t.Errorf("report.ActiveBanks = %d, want 16", report.ActiveBanks)
	}
	if report.SpeedupEstimate != 1.13 {
		t.Errorf("report.SpeedupEstimate = %f, want 1.13", report.SpeedupEstimate)
	}
}

// TestROCm_Ticket12081_Wave32WMMA_ShaderAndPipeline verifies the Wave32 Cooperative Matrix Retiling (WMMA)
// micro-architectural optimization for RDNA 3.5 (gfx1151) (#12081):
// 1. coopmat_wave32_wmma.comp shader source verification (GL extensions, workgroup size 32x4x1, Pad-2 stride 34, attribution).
// 2. SubgroupSize == 32 propagation across WMMATileGeometry and CooperativeMatrixConfig.
// 3. Pipeline generation and validation helpers.
func TestROCm_Ticket12081_Wave32WMMA_ShaderAndPipeline(t *testing.T) {
	// 1. Verify coopmat_wave32_wmma.comp shader file
	shaderPath := filepath.Join("shaders", "coopmat_wave32_wmma.comp")
	content, err := os.ReadFile(shaderPath)
	if err != nil {
		// Try relative to repo root if test running in different directory
		shaderPath = filepath.Join("..", "..", "internal", "compute", "shaders", "coopmat_wave32_wmma.comp")
		content, err = os.ReadFile(shaderPath)
	}
	if err != nil {
		t.Fatalf("failed to read coopmat_wave32_wmma.comp: %v", err)
	}
	shaderSrc := string(content)

	requiredShaderTokens := []string{
		"#version 450",
		"Nathanw1014",
		"MIT License",
		"GL_KHR_cooperative_matrix",
		"GL_KHR_memory_scope_semantics",
		"GL_KHR_shader_subgroup_basic",
		"GL_EXT_shader_explicit_arithmetic_types_float16",
		"GL_EXT_shader_explicit_arithmetic_types_int8",
		"layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;",
		"uint M;",
		"uint N;",
		"uint K;",
		"float alpha;",
		"float beta;",
		"STRIDE_N = TILE_N + LDS_PAD_2;",
		"32 + 2 = 34",
		"coopMatLoad",
		"coopMatMulAdd",
		"coopMatStore",
	}

	for _, tok := range requiredShaderTokens {
		if !strings.Contains(shaderSrc, tok) {
			t.Errorf("coopmat_wave32_wmma.comp missing required token: %q", tok)
		}
	}

	// 2. Verify SubgroupSize == 32 in SupportedWMMATiles and TuneCooperativeMatrixGEMM
	arch, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found")
	}

	tiles := arch.SupportedWMMATiles()
	if len(tiles) == 0 {
		t.Fatal("SupportedWMMATiles() returned empty slice")
	}
	for _, tile := range tiles {
		if tile.SubgroupSize != 32 {
			t.Errorf("tile.SubgroupSize = %d, want 32", tile.SubgroupSize)
		}
	}

	cfgFP16, err := arch.TuneCooperativeMatrixGEMM(64, 64, 64, "fp16")
	if err != nil {
		t.Fatalf("TuneCooperativeMatrixGEMM(fp16): %v", err)
	}
	if cfgFP16.SubgroupSize != 32 {
		t.Errorf("cfgFP16.SubgroupSize = %d, want 32", cfgFP16.SubgroupSize)
	}
	if cfgFP16.WavesPerWorkgroup != 4 {
		t.Errorf("cfgFP16.WavesPerWorkgroup = %d, want 4", cfgFP16.WavesPerWorkgroup)
	}
	if cfgFP16.TileM != 32 || cfgFP16.TileN != 32 {
		t.Errorf("cfgFP16 tile dimensions = (%d, %d), want (32, 32)", cfgFP16.TileM, cfgFP16.TileN)
	}
	if cfgFP16.PaddedStride != 34 {
		t.Errorf("cfgFP16.PaddedStride = %d, want 34", cfgFP16.PaddedStride)
	}

	cfgINT8, err := arch.TuneCooperativeMatrixGEMM(64, 64, 64, "int8")
	if err != nil {
		t.Fatalf("TuneCooperativeMatrixGEMM(int8): %v", err)
	}
	if cfgINT8.SubgroupSize != 32 {
		t.Errorf("cfgINT8.SubgroupSize = %d, want 32", cfgINT8.SubgroupSize)
	}
	if cfgINT8.TileK != 32 {
		t.Errorf("cfgINT8.TileK = %d, want 32", cfgINT8.TileK)
	}

	// 3. ValidateCooperativeMatrixConfig
	if err := ValidateCooperativeMatrixConfig(cfgFP16); err != nil {
		t.Errorf("ValidateCooperativeMatrixConfig(cfgFP16) unexpected error: %v", err)
	}
	if err := ValidateCooperativeMatrixConfig(cfgINT8); err != nil {
		t.Errorf("ValidateCooperativeMatrixConfig(cfgINT8) unexpected error: %v", err)
	}

	// Verify fail-closed validation on mismatched subgroup size
	badCfg := cfgFP16
	badCfg.SubgroupSize = 64
	if err := ValidateCooperativeMatrixConfig(badCfg); err == nil {
		t.Errorf("ValidateCooperativeMatrixConfig with SubgroupSize=64 should have failed")
	}

	// 4. Test pipeline descriptor generation
	pipelineFP16, err := arch.GenerateWave32WMMAPipelineDescriptor(128, 128, 128, "fp16")
	if err != nil {
		t.Fatalf("GenerateWave32WMMAPipelineDescriptor(fp16): %v", err)
	}
	if pipelineFP16.SubgroupSize != 32 {
		t.Errorf("pipelineFP16.SubgroupSize = %d, want 32", pipelineFP16.SubgroupSize)
	}
	if pipelineFP16.WorkgroupLocalX != 32 || pipelineFP16.WorkgroupLocalY != 4 || pipelineFP16.WorkgroupLocalZ != 1 {
		t.Errorf("pipelineFP16 workgroup local size = (%d, %d, %d), want (32, 4, 1)", pipelineFP16.WorkgroupLocalX, pipelineFP16.WorkgroupLocalY, pipelineFP16.WorkgroupLocalZ)
	}
	if pipelineFP16.Pad2Stride != 34 {
		t.Errorf("pipelineFP16.Pad2Stride = %d, want 34", pipelineFP16.Pad2Stride)
	}

	// 5. Test push constants encoding and validation
	pc, err := NewCooperativeMatrixPushConstants(128, 128, 64, 1.0, 0.0)
	if err != nil {
		t.Fatalf("NewCooperativeMatrixPushConstants: %v", err)
	}
	if pc.Size() != 32 {
		t.Errorf("pc.Size() = %d, want 32", pc.Size())
	}
	encoded := pc.Encode()
	if len(encoded) != 32 {
		t.Errorf("len(encoded) = %d, want 32", len(encoded))
	}
	if err := pc.Validate(); err != nil {
		t.Errorf("pc.Validate() error: %v", err)
	}

	zeroPC := CooperativeMatrixPushConstants{}
	if err := zeroPC.Validate(); err == nil {
		t.Errorf("zeroPC.Validate() should have failed for zero dimensions")
	}
}

// TestROCmArchCompilerFlagsGfx1151Wave32 witnesses the required Wave32 and target CPU compiler flags
// for AMD Strix Halo (gfx1151 / RDNA 3.5) (#12187):
// - `CompilerFlags()` for `gfx1151` must mandate `-mwavefrontsize32` and `-mcpu=gfx1151`.
// - Targets discrete RDNA 3 (gfx1100) or CDNA (gfx90a/gfx942) remain untouched.
func TestROCmArchCompilerFlagsGfx1151Wave32(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("LookupROCmArch(gfx1151): not found, want supported")
	}
	if a.Family != ROCmRDNA3_5 {
		t.Errorf("gfx1151 family = %v, want ROCmRDNA3_5", a.Family)
	}
	if a.Wavefront != 32 {
		t.Errorf("gfx1151 wavefront = %d, want 32", a.Wavefront)
	}
	if !a.HasNativeWave32WMMA() {
		t.Errorf("gfx1151 HasNativeWave32WMMA() = false, want true")
	}

	flags := a.CompilerFlags()
	hasWave32 := false
	hasMCPU := false
	hasOffload := false
	for _, f := range flags {
		if f == "-mwavefrontsize32" {
			hasWave32 = true
		}
		if f == "-mcpu=gfx1151" {
			hasMCPU = true
		}
		if f == "--offload-arch=gfx1151" {
			hasOffload = true
		}
	}
	if !hasWave32 {
		t.Errorf("flags %+v missing -mwavefrontsize32", flags)
	}
	if !hasMCPU {
		t.Errorf("flags %+v missing -mcpu=gfx1151", flags)
	}
	if !hasOffload {
		t.Errorf("flags %+v missing --offload-arch=gfx1151", flags)
	}

	// Verify discrete desktop RDNA 3 (gfx1100) does not receive -mcpu=gfx1151
	rdna3, ok := LookupROCmArch("gfx1100")
	if !ok {
		t.Fatal("LookupROCmArch(gfx1100) not found")
	}
	for _, f := range rdna3.CompilerFlags() {
		if f == "-mcpu=gfx1151" {
			t.Errorf("gfx1100 unexpectedly includes -mcpu=gfx1151: %v", rdna3.CompilerFlags())
		}
	}

	// Verify CDNA (gfx90a) does not receive -mwavefrontsize32 or -mcpu=gfx1151
	cdna, ok := LookupROCmArch("gfx90a")
	if !ok {
		t.Fatal("LookupROCmArch(gfx90a) not found")
	}
	for _, f := range cdna.CompilerFlags() {
		if f == "-mcpu=gfx1151" || f == "-mwavefrontsize32" {
			t.Errorf("gfx90a unexpectedly includes RDNA flags: %v", cdna.CompilerFlags())
		}
	}
}
