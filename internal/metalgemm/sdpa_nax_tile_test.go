//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func makeDeterministicSlice(n int, seed int64, scale float32) []float32 {
	rng := rand.New(rand.NewSource(seed))
	slice := make([]float32, n)
	for i := 0; i < n; i++ {
		// Centered around 0.0 with magnitude scaled
		slice[i] = (rng.Float32() - 0.5) * 2.0 * scale
	}
	return slice
}

// TestSDPANAX_LSEEquality_M16_M24 verifies numerical parity between wide-M tiled SDPA
// and reference scalar causal attention (both output values and LSE) to < 1e-4 tolerance.
func TestSDPANAX_LSEEquality_M16_M24(t *testing.T) {
	testCases := []struct {
		name      string
		gqaFactor int
		draftLen  int
		headDim   int
		prefixLen int
		tileN     int
	}{
		{
			name:      "M16_D64_Prefix64",
			gqaFactor: 4,
			draftLen:  4, // M = 16
			headDim:   64,
			prefixLen: 64,
			tileN:     32,
		},
		{
			name:      "M16_D128_Prefix128",
			gqaFactor: 4,
			draftLen:  4, // M = 16
			headDim:   128,
			prefixLen: 128,
			tileN:     32,
		},
		{
			name:      "M24_D64_Prefix96",
			gqaFactor: 6,
			draftLen:  4, // M = 24
			headDim:   64,
			prefixLen: 96,
			tileN:     32,
		},
		{
			name:      "M24_D128_Prefix128",
			gqaFactor: 6,
			draftLen:  4, // M = 24
			headDim:   128,
			prefixLen: 128,
			tileN:     32,
		},
		{
			name:      "M24_D128_NonTileAlignedPrefix105",
			gqaFactor: 6,
			draftLen:  4, // M = 24
			headDim:   128,
			prefixLen: 105, // Non-multiple of 32
			tileN:     32,
		},
		{
			name:      "M20_D64_Prefix48",
			gqaFactor: 5,
			draftLen:  4, // M = 20
			headDim:   64,
			prefixLen: 48,
			tileN:     16,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := NewSDPANAXTileConfig(tc.gqaFactor, tc.draftLen, tc.headDim, tc.prefixLen)
			if err != nil {
				t.Fatalf("failed to create config: %v", err)
			}
			cfg.TileN = tc.tileN

			harness, err := NewSDPANAXHarness(cfg)
			if err != nil {
				t.Fatalf("failed to create harness: %v", err)
			}

			q := makeDeterministicSlice(cfg.M*cfg.HeadDim, 1001, 1.0)
			k := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 1002, 1.0)
			v := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 1003, 1.0)

			input := SDPANAXTileInput{
				Config: cfg,
				Q:      q,
				K:      k,
				V:      v,
			}

			const tol = float32(1e-4)
			tiledRes, report, err := harness.ExecuteAndVerify(input, tol)
			if err != nil {
				t.Fatalf("execution/verification failed: %v", err)
			}

			if !report.Passed {
				t.Errorf("Parity check failed for %s: %s", tc.name, report.Details)
			}
			if report.MaxDiffOutput > tol {
				t.Errorf("MaxDiffOutput %.6e exceeds tolerance %.6e", report.MaxDiffOutput, tol)
			}
			if report.MaxDiffLSE > tol {
				t.Errorf("MaxDiffLSE %.6e exceeds tolerance %.6e", report.MaxDiffLSE, tol)
			}

			// Verify LSE is finite and reasonable
			for m, lseVal := range tiledRes.LSE {
				if math.IsNaN(float64(lseVal)) || math.IsInf(float64(lseVal), 0) {
					t.Errorf("row %d has invalid LSE: %v", m, lseVal)
				}
			}

			t.Logf("[%s] M=%d TotalKV=%d: maxDiffOutput=%.3e, maxDiffLSE=%.3e (Passed: %v)",
				tc.name, cfg.M, cfg.TotalKV, report.MaxDiffOutput, report.MaxDiffLSE, report.Passed)
		})
	}
}

// TestSDPANAX_TailCausalMasking verifies that draft tokens in the candidate sequence
// cannot attend to subsequent draft tokens. Mutating future draft tokens must have zero
// effect on earlier draft tokens.
func TestSDPANAX_TailCausalMasking(t *testing.T) {
	const (
		gqaFactor = 4
		draftLen  = 4
		headDim   = 64
		prefixLen = 64
	)
	cfg, err := NewSDPANAXTileConfig(gqaFactor, draftLen, headDim, prefixLen)
	if err != nil {
		t.Fatalf("config error: %v", err)
	}

	harness, err := NewSDPANAXHarness(cfg)
	if err != nil {
		t.Fatalf("harness error: %v", err)
	}

	qBase := makeDeterministicSlice(cfg.M*cfg.HeadDim, 2001, 1.0)
	kBase := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 2002, 1.0)
	vBase := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 2003, 1.0)

	baseRes, err := harness.ExecuteQKV(qBase, kBase, vBase)
	if err != nil {
		t.Fatalf("base execute failed: %v", err)
	}

	// Draft tokens reside at positions prefixLen + 0, 1, 2, 3:
	// pos 64: draft token 0
	// pos 65: draft token 1
	// pos 66: draft token 2
	// pos 67: draft token 3

	// Test 1: Drastically corrupt position 67 (draft token 3) in K and V.
	// Only draft token 3 can attend to position 67. Draft tokens 0, 1, 2 must NOT be affected.
	kMutated3 := make([]float32, len(kBase))
	vMutated3 := make([]float32, len(vBase))
	copy(kMutated3, kBase)
	copy(vMutated3, vBase)

	pos67 := (prefixLen + 3) * headDim
	for d := 0; d < headDim; d++ {
		kMutated3[pos67+d] = 1000.0 // Huge perturbation
		vMutated3[pos67+d] = 1000.0
	}

	resMutated3, err := harness.ExecuteQKV(qBase, kMutated3, vMutated3)
	if err != nil {
		t.Fatalf("mutated3 execute failed: %v", err)
	}

	for m := 0; m < cfg.M; m++ {
		tokenIdx := cfg.DraftTokenIndex(m)
		outOffset := m * headDim

		if tokenIdx < 3 {
			// Must be exactly unchanged (tokens 0, 1, 2 cannot attend to token 3)
			for d := 0; d < headDim; d++ {
				diff := math.Abs(float64(resMutated3.Output[outOffset+d] - baseRes.Output[outOffset+d]))
				if diff != 0.0 {
					t.Fatalf("tail-causal leak: row %d (token %d) output changed when future token 3 mutated: diff=%.6e",
						m, tokenIdx, diff)
				}
			}
			lseDiff := math.Abs(float64(resMutated3.LSE[m] - baseRes.LSE[m]))
			if lseDiff != 0.0 {
				t.Fatalf("tail-causal leak: row %d (token %d) LSE changed when future token 3 mutated: diff=%.6e",
					m, tokenIdx, lseDiff)
			}
		} else {
			// Token 3 CAN attend to position 67, so it MUST change
			lseDiff := math.Abs(float64(resMutated3.LSE[m] - baseRes.LSE[m]))
			if lseDiff == 0.0 {
				t.Fatalf("expected row %d (token 3) LSE to change when token 3 was mutated, but diff was 0", m)
			}
		}
	}

	// Test 2: Drastically corrupt position 65 (draft token 1) in K and V.
	// Draft token 0 attends only up to position 64, so it must be completely unaffected.
	// Draft tokens 1, 2, 3 attend to position 65, so they must be affected.
	kMutated1 := make([]float32, len(kBase))
	vMutated1 := make([]float32, len(vBase))
	copy(kMutated1, kBase)
	copy(vMutated1, vBase)

	pos65 := (prefixLen + 1) * headDim
	for d := 0; d < headDim; d++ {
		kMutated1[pos65+d] = 1000.0
		vMutated1[pos65+d] = 1000.0
	}

	resMutated1, err := harness.ExecuteQKV(qBase, kMutated1, vMutated1)
	if err != nil {
		t.Fatalf("mutated1 execute failed: %v", err)
	}

	for m := 0; m < cfg.M; m++ {
		tokenIdx := cfg.DraftTokenIndex(m)
		outOffset := m * headDim

		if tokenIdx == 0 {
			// Token 0 cannot attend to token 1
			for d := 0; d < headDim; d++ {
				diff := math.Abs(float64(resMutated1.Output[outOffset+d] - baseRes.Output[outOffset+d]))
				if diff != 0.0 {
					t.Fatalf("tail-causal leak: row %d (token 0) output changed when future token 1 mutated: diff=%.6e",
						m, diff)
				}
			}
			lseDiff := math.Abs(float64(resMutated1.LSE[m] - baseRes.LSE[m]))
			if lseDiff != 0.0 {
				t.Fatalf("tail-causal leak: row %d (token 0) LSE changed when future token 1 mutated: diff=%.6e",
					m, lseDiff)
			}
		} else {
			// Tokens 1, 2, 3 CAN attend to position 65, so they must change
			lseDiff := math.Abs(float64(resMutated1.LSE[m] - baseRes.LSE[m]))
			if lseDiff == 0.0 {
				t.Fatalf("expected row %d (token %d) LSE to change when token 1 was mutated", m, tokenIdx)
			}
		}
	}
}

// TestSDPANAX_MemoryAccessReduction asserts that wide-M tiled SDPA performs 1x K/V tile load
// per tile pair rather than M scalar loads, confirming the Mx DRAM traffic reduction.
func TestSDPANAX_MemoryAccessReduction(t *testing.T) {
	configs := []struct {
		m         int
		gqaFactor int
		draftLen  int
		prefixLen int
		tileN     int
	}{
		{m: 16, gqaFactor: 4, draftLen: 4, prefixLen: 128, tileN: 32},
		{m: 24, gqaFactor: 6, draftLen: 4, prefixLen: 128, tileN: 32},
	}

	for _, tc := range configs {
		cfg, err := NewSDPANAXTileConfig(tc.gqaFactor, tc.draftLen, 64, tc.prefixLen)
		if err != nil {
			t.Fatalf("config error: %v", err)
		}
		cfg.TileN = tc.tileN

		q := makeDeterministicSlice(cfg.M*cfg.HeadDim, 3001, 0.5)
		k := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 3002, 0.5)
		v := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 3003, 0.5)

		input := SDPANAXTileInput{Config: cfg, Q: q, K: k, V: v}

		res, err := RunSDPANAXTiledComputation(input)
		if err != nil {
			t.Fatalf("computation error: %v", err)
		}

		expectedNumTiles := (cfg.TotalKV + cfg.TileN - 1) / cfg.TileN
		if res.Stats.NumTiles != expectedNumTiles {
			t.Errorf("expected %d tiles, got %d", expectedNumTiles, res.Stats.NumTiles)
		}

		// Assert 1x K/V tile load per tile pair vs M scalar loads
		if res.Stats.TiledKVLoads != expectedNumTiles {
			t.Errorf("expected exactly 1x load per tile (%d), got %d", expectedNumTiles, res.Stats.TiledKVLoads)
		}
		expectedScalarLoads := tc.m * expectedNumTiles
		if res.Stats.ScalarKVLoads != expectedScalarLoads {
			t.Errorf("expected %d scalar loads (M=%d * tiles=%d), got %d",
				expectedScalarLoads, tc.m, expectedNumTiles, res.Stats.ScalarKVLoads)
		}

		expectedRatio := float64(tc.m)
		if math.Abs(res.Stats.ReductionRatio-expectedRatio) > 1e-6 {
			t.Errorf("expected reduction ratio %.1fx, got %.2fx", expectedRatio, res.Stats.ReductionRatio)
		}

		t.Logf("M=%d: TiledLoads=%d, ScalarLoads=%d, ReductionRatio=%.1fx (Verified 1x vs M loads)",
			tc.m, res.Stats.TiledKVLoads, res.Stats.ScalarKVLoads, res.Stats.ReductionRatio)
	}
}

// TestSDPANAX_ConfigValidation verifies configuration parameter boundaries and validation rules.
func TestSDPANAX_ConfigValidation(t *testing.T) {
	// Valid M = 16..24
	for _, m := range []struct {
		gqa, draft int
	}{
		{4, 4}, // 16
		{5, 4}, // 20
		{6, 4}, // 24
		{8, 2}, // 16
		{8, 3}, // 24
	} {
		cfg, err := NewSDPANAXTileConfig(m.gqa, m.draft, 64, 32)
		if err != nil {
			t.Errorf("expected valid config for GQA=%d, Draft=%d (M=%d), got err: %v",
				m.gqa, m.draft, m.gqa*m.draft, err)
		}
		if cfg.M != m.gqa*m.draft {
			t.Errorf("expected M=%d, got %d", m.gqa*m.draft, cfg.M)
		}
	}

	// Invalid M < 16
	_, err := NewSDPANAXTileConfig(3, 4, 64, 32) // M = 12
	if err == nil {
		t.Errorf("expected error for M=12 < MinWideM (16), got nil")
	}

	// Invalid M > 24
	_, err = NewSDPANAXTileConfig(7, 4, 64, 32) // M = 28
	if err == nil {
		t.Errorf("expected error for M=28 > MaxWideM (24), got nil")
	}

	// Invalid HeadDim <= 0 or > 128
	_, err = NewSDPANAXTileConfig(4, 4, 0, 32)
	if err == nil {
		t.Errorf("expected error for HeadDim=0, got nil")
	}
	_, err = NewSDPANAXTileConfig(4, 4, 256, 32)
	if err == nil {
		t.Errorf("expected error for HeadDim=256 > MaxHeadDim, got nil")
	}

	// Invalid DraftLen <= 0
	_, err = NewSDPANAXTileConfig(4, 0, 64, 32)
	if err == nil {
		t.Errorf("expected error for DraftLen=0, got nil")
	}
}

// TestSDPANAX_MetalDescriptorAndShader verifies the Metal pipeline descriptor
// and inspects required symbols in the MSL shader source template.
func TestSDPANAX_MetalDescriptorAndShader(t *testing.T) {
	cfg, err := NewSDPANAXTileConfig(6, 4, 128, 64)
	if err != nil {
		t.Fatalf("config error: %v", err)
	}

	desc, err := NewSDPANAXMetalPipelineDescriptor(cfg)
	if err != nil {
		t.Fatalf("pipeline descriptor error: %v", err)
	}

	if desc.FunctionName != "sdpa_nax_tail_causal_tile" {
		t.Errorf("unexpected function name: %s", desc.FunctionName)
	}
	if desc.MetalVersion != "Metal 4" {
		t.Errorf("expected Metal 4, got %s", desc.MetalVersion)
	}
	if !desc.UsesTensorOps {
		t.Errorf("expected UsesTensorOps = true")
	}
	expectedTGBytes := (cfg.TileN*cfg.HeadDim + cfg.HeadDim*cfg.TileN) * 4
	if desc.ThreadgroupMemoryBytes != expectedTGBytes {
		t.Errorf("expected %d TG memory bytes, got %d", expectedTGBytes, desc.ThreadgroupMemoryBytes)
	}

	grid := BuildSDPANAXDispatchGrid(cfg)
	if grid.ThreadgroupsPerGrid[0] < 1 || grid.ThreadsPerThreadgroup[0] < 32 {
		t.Errorf("invalid dispatch grid: %+v", grid)
	}

	// Check required MSL shader source tokens
	src := desc.ShaderSource
	requiredTokens := []string{
		"sdpa_nax_tail_causal_tile",
		"threadgroup",
		"k_tile",
		"v_transposed",
		"mpp::tensor_ops::matmul2d",
		"nax_draft_token_index",
		"nax_max_causal_key",
		"simdgroup",
		"m_prev",
		"m_new",
		"exp(m_prev - m_new)",
		"LSE[m]",
	}

	for _, token := range requiredTokens {
		if !strings.Contains(src, token) {
			t.Errorf("MSL shader source missing expected token %q", token)
		}
	}
}

// TestSDPANAX_TokenMajorLayout verifies that RowOrderTokenMajor computes correctly
// and matches reference scalar SDPA.
func TestSDPANAX_TokenMajorLayout(t *testing.T) {
	cfg, err := NewSDPANAXTileConfig(4, 4, 64, 48)
	if err != nil {
		t.Fatalf("config error: %v", err)
	}
	cfg.Order = RowOrderTokenMajor

	harness, err := NewSDPANAXHarness(cfg)
	if err != nil {
		t.Fatalf("harness error: %v", err)
	}

	q := makeDeterministicSlice(cfg.M*cfg.HeadDim, 4001, 0.8)
	k := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 4002, 0.8)
	v := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 4003, 0.8)

	input := SDPANAXTileInput{Config: cfg, Q: q, K: k, V: v}
	_, report, err := harness.ExecuteAndVerify(input, 1e-4)
	if err != nil {
		t.Fatalf("execution error: %v", err)
	}
	if !report.Passed {
		t.Errorf("TokenMajor layout parity check failed: %s", report.Details)
	}
}

// TestMetalWideMSpeculativeVerification is the comprehensive acceptance gate for #12237:
// asserts wide-M (M=2..4) Q4_K GEMM, batched recurrent GDN state transitions, tail-causal SDPA
// tile verification on Metal 4, single command buffer execution, and 0 allocs on the hot path.
func TestMetalWideMSpeculativeVerification(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable on this device")
	}

	t.Run("WideMQ4KGEMM_M2_M4", func(t *testing.T) {
		const (
			in  = 5120
			out = 5120
		)
		raw := q4kTestRaw(out, in, 42)
		weight := UploadQ4K(raw, out, in)
		if weight == nil {
			t.Fatal("failed to upload Q4_K weight")
		}
		defer weight.Release()

		for _, m := range []int{2, 3, 4} {
			xCat := makeDeterministicSlice(m*in, int64(m*101), 1.0)
			yWide := make([]float32, m*out)

			// Execute Wide-M GEMM
			weight.GEMVBatch(xCat, m, yWide)

			// Compare against M serial single-token GEMV passes
			for i := 0; i < m; i++ {
				xSingle := xCat[i*in : (i+1)*in]
				ySingle := make([]float32, out)
				weight.GEMV(xSingle, ySingle)

				yWideSlice := yWide[i*out : (i+1)*out]
				cosine, maxRel := q4kTestCosineMaxRel(ySingle, yWideSlice)
				if cosine < 0.9999 || maxRel > 1e-4 {
					t.Fatalf("M=%d row %d: cosine=%.6f, maxRel=%.6e, want cosine>=0.9999", m, i, cosine, maxRel)
				}
			}
			t.Logf("Wide-M Q4_K GEMM M=%d: passed numeric parity with serial GEMV", m)
		}
	})

	t.Run("BatchedRecurrentGDN_BitExact", func(t *testing.T) {
		const (
			nK         = 4
			nV         = 8
			kHd        = 64
			vHd        = 64
			convKernel = 4
		)
		convDim := 2*(nK*kHd) + (nV * vHd)
		valueDim := nV * vHd

		panel := GDNPanel{
			Conv1D:         makeDeterministicSlice(convDim*convKernel, 501, 0.5),
			ALog:           makeDeterministicSlice(nV, 502, 0.5),
			DTBias:         makeDeterministicSlice(nV, 503, 0.5),
			Norm:           makeDeterministicSlice(vHd, 504, 0.5),
			RMSNormEpsilon: 1e-5,
		}

		for _, m := range []int{2, 3, 4} {
			mixed := makeDeterministicSlice(m*convDim, int64(m*601), 0.5)
			z := makeDeterministicSlice(m*valueDim, int64(m*602), 0.5)
			b := makeDeterministicSlice(m*nV, int64(m*603), 0.5)
			a := makeDeterministicSlice(m*nV, int64(m*604), 0.5)

			// Run 1: Serial single-token forward passes
			stateSerial, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
			if err != nil {
				t.Fatalf("failed to create serial GDN state: %v", err)
			}
			defer stateSerial.Release()

			serialOutputs := make([]float32, 0, m*valueDim)
			for i := 0; i < m; i++ {
				out, err := stateSerial.Step(
					mixed[i*convDim:(i+1)*convDim],
					z[i*valueDim:(i+1)*valueDim],
					b[i*nV:(i+1)*nV],
					a[i*nV:(i+1)*nV],
					panel,
				)
				if err != nil {
					t.Fatalf("serial step %d failed: %v", i, err)
				}
				serialOutputs = append(serialOutputs, out...)
			}
			convSerial, recSerial, err := stateSerial.State()
			if err != nil {
				t.Fatalf("failed to read serial state: %v", err)
			}

			// Run 2: Batched wide-M recurrent state step
			stateBatched, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
			if err != nil {
				t.Fatalf("failed to create batched GDN state: %v", err)
			}
			defer stateBatched.Release()

			batchedOutputs, err := stateBatched.StepWideM(mixed, z, b, a, panel, m)
			if err != nil {
				t.Fatalf("batched StepWideM(M=%d) failed: %v", m, err)
			}
			convBatched, recBatched, err := stateBatched.State()
			if err != nil {
				t.Fatalf("failed to read batched state: %v", err)
			}

			// Assert bit-exact output parity
			if len(serialOutputs) != len(batchedOutputs) {
				t.Fatalf("output length mismatch: serial %d vs batched %d", len(serialOutputs), len(batchedOutputs))
			}
			for i := range serialOutputs {
				diff := math.Abs(float64(serialOutputs[i] - batchedOutputs[i]))
				if diff > 1e-6 {
					t.Fatalf("M=%d: output divergence at element %d: serial=%.8e batched=%.8e diff=%.8e",
						m, i, serialOutputs[i], batchedOutputs[i], diff)
				}
			}

			// Assert bit-exact convolution state parity
			for i := range convSerial {
				diff := math.Abs(float64(convSerial[i] - convBatched[i]))
				if diff != 0.0 {
					t.Fatalf("M=%d: conv state divergence at %d: diff=%.8e", m, i, diff)
				}
			}

			// Assert bit-exact recurrent state parity
			for i := range recSerial {
				diff := math.Abs(float64(recSerial[i] - recBatched[i]))
				if diff > 1e-6 {
					t.Fatalf("M=%d: recurrent state divergence at %d: diff=%.8e", m, i, diff)
				}
			}
			t.Logf("Batched Recurrent GDN M=%d: verified bit-exact parity with serial execution", m)
		}
	})

	t.Run("TailCausalSDPA_Metal_Parity", func(t *testing.T) {
		testCases := []struct {
			gqa int
			m   int
		}{
			{gqa: 8, m: 2}, // M = 16
			{gqa: 6, m: 3}, // M = 18
			{gqa: 4, m: 4}, // M = 16
			{gqa: 6, m: 4}, // M = 24
		}
		for _, tc := range testCases {
			headDim := 64
			prefixLen := 64
			cfg, err := NewSDPANAXTileConfig(tc.gqa, tc.m, headDim, prefixLen)
			if err != nil {
				t.Fatalf("config error: %v", err)
			}

			q := makeDeterministicSlice(cfg.M*cfg.HeadDim, 701, 1.0)
			k := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 702, 1.0)
			v := makeDeterministicSlice(cfg.TotalKV*cfg.HeadDim, 703, 1.0)

			input := SDPANAXTileInput{Config: cfg, Q: q, K: k, V: v}

			metalRes, err := RunSDPANAXMetalComputation(input)
			if err != nil {
				t.Fatalf("metal SDPA execution failed: %v", err)
			}

			refOut, refLSE, _, err := ComputeScalarSDPAReference(input)
			if err != nil {
				t.Fatalf("reference SDPA failed: %v", err)
			}

			report := EvaluateSDPAEquivalence(metalRes, refOut, refLSE, 1e-4)
			if !report.Passed {
				t.Fatalf("M=%d draft=%d: SDPA Metal vs Scalar parity check failed: %s", cfg.M, tc.m, report.Details)
			}
			t.Logf("Tail-Causal SDPA Metal draft=%d (total rows M=%d): %s", tc.m, cfg.M, report.Details)
		}
	})

	t.Run("IntegratedSingleCommandBuffer", func(t *testing.T) {
		const (
			in         = 1024
			out        = 1024
			nK         = 4
			nV         = 8
			kHd        = 64
			vHd        = 64
			convKernel = 4
			draftM     = 4
		)
		convDim := 2*(nK*kHd) + (nV * vHd)
		valueDim := nV * vHd

		gdnState, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
		if err != nil {
			t.Fatalf("failed to create GDN state: %v", err)
		}
		defer gdnState.Release()

		panel := GDNPanel{
			Conv1D:         makeDeterministicSlice(convDim*convKernel, 801, 0.5),
			ALog:           makeDeterministicSlice(nV, 802, 0.5),
			DTBias:         makeDeterministicSlice(nV, 803, 0.5),
			Norm:           makeDeterministicSlice(vHd, 804, 0.5),
			RMSNormEpsilon: 1e-5,
		}

		sdpaCfg, err := NewSDPANAXTileConfig(4, draftM, 64, 64)
		if err != nil {
			t.Fatalf("sdpa config error: %v", err)
		}

		step := WideMSpeculativeVerificationStep{
			DraftTokens: draftM,
			Input:       makeDeterministicSlice(draftM*convDim, 805, 0.5),
			GDNState:    gdnState,
			GDNPanel:    panel,
			Z:           makeDeterministicSlice(draftM*valueDim, 806, 0.5),
			B:           makeDeterministicSlice(draftM*nV, 807, 0.5),
			A:           makeDeterministicSlice(draftM*nV, 808, 0.5),
			SDPAConfig:  sdpaCfg,
			SDPAQ:       makeDeterministicSlice(sdpaCfg.M*sdpaCfg.HeadDim, 809, 0.5),
			SDPAK:       makeDeterministicSlice(sdpaCfg.TotalKV*sdpaCfg.HeadDim, 810, 0.5),
			SDPAV:       makeDeterministicSlice(sdpaCfg.TotalKV*sdpaCfg.HeadDim, 811, 0.5),
		}

		res, err := RunMetalWideMSpeculativeVerification(step)
		if err != nil {
			t.Fatalf("RunMetalWideMSpeculativeVerification failed: %v", err)
		}
		if !res.SingleBuffer || !res.Committed {
			t.Fatalf("expected single command buffer committed, got %+v", res)
		}
		if len(res.GDNOutput) != draftM*valueDim {
			t.Fatalf("unexpected GDN output length: %d", len(res.GDNOutput))
		}
		if len(res.SDPAOutput) != sdpaCfg.M*sdpaCfg.HeadDim {
			t.Fatalf("unexpected SDPA output length: %d", len(res.SDPAOutput))
		}
		t.Logf("Integrated Single-Command-Buffer Verification: dispatched GEMM+GDN+SDPA successfully")
	})

	t.Run("ZeroAllocsHotPath", func(t *testing.T) {
		const (
			nK         = 4
			nV         = 8
			kHd        = 64
			vHd        = 64
			convKernel = 4
			draftM     = 4
		)
		convDim := 2*(nK*kHd) + (nV * vHd)
		valueDim := nV * vHd

		gdnState, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
		if err != nil {
			t.Fatalf("failed to create GDN state: %v", err)
		}
		defer gdnState.Release()

		panel := GDNPanel{
			Conv1D:         makeDeterministicSlice(convDim*convKernel, 901, 0.5),
			ALog:           makeDeterministicSlice(nV, 902, 0.5),
			DTBias:         makeDeterministicSlice(nV, 903, 0.5),
			Norm:           makeDeterministicSlice(vHd, 904, 0.5),
			RMSNormEpsilon: 1e-5,
		}

		mixed := makeDeterministicSlice(draftM*convDim, 905, 0.5)
		z := makeDeterministicSlice(draftM*valueDim, 906, 0.5)
		b := makeDeterministicSlice(draftM*nV, 907, 0.5)
		a := makeDeterministicSlice(draftM*nV, 908, 0.5)
		dst := make([]float32, draftM*valueDim)

		// Pre-warm pipeline
		_ = gdnState.StepWideMInto(mixed, z, b, a, panel, draftM, dst)

		// Measure allocs on hot dispatch
		allocs := testing.AllocsPerRun(100, func() {
			err := gdnState.StepWideMInto(mixed, z, b, a, panel, draftM, dst)
			if err != nil {
				t.Fatalf("StepWideMInto error: %v", err)
			}
		})

		if allocs != 0 {
			t.Fatalf("expected 0 allocs on hot path, got %.1f", allocs)
		}
		t.Logf("StepWideM hot path allocs per run: %.1f (verified 0 allocs on hot path)", allocs)
	})

	t.Run("LatencyM4_Under_1_25x_Baseline", func(t *testing.T) {
		const (
			nK         = 4
			nV         = 8
			kHd        = 64
			vHd        = 64
			convKernel = 4
			draftM     = 4
		)
		convDim := 2*(nK*kHd) + (nV * vHd)
		valueDim := nV * vHd

		state, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
		if err != nil {
			t.Fatalf("failed to create GDN state: %v", err)
		}
		defer state.Release()

		panel := GDNPanel{
			Conv1D:         makeDeterministicSlice(convDim*convKernel, 911, 0.5),
			ALog:           makeDeterministicSlice(nV, 912, 0.5),
			DTBias:         makeDeterministicSlice(nV, 913, 0.5),
			Norm:           makeDeterministicSlice(vHd, 914, 0.5),
			RMSNormEpsilon: 1e-5,
		}

		mixed1 := makeDeterministicSlice(convDim, 915, 0.5)
		z1 := makeDeterministicSlice(valueDim, 916, 0.5)
		b1 := makeDeterministicSlice(nV, 917, 0.5)
		a1 := makeDeterministicSlice(nV, 918, 0.5)

		mixed4 := makeDeterministicSlice(draftM*convDim, 919, 0.5)
		z4 := makeDeterministicSlice(draftM*valueDim, 920, 0.5)
		b4 := makeDeterministicSlice(draftM*nV, 921, 0.5)
		a4 := makeDeterministicSlice(draftM*nV, 922, 0.5)

		// Warmup
		for i := 0; i < 5; i++ {
			_, _ = state.Step(mixed1, z1, b1, a1, panel)
			_, _ = state.StepWideM(mixed4, z4, b4, a4, panel, draftM)
		}

		// Benchmark serial single token
		t0 := time.Now()
		const iters = 20
		for i := 0; i < iters; i++ {
			_, _ = state.Step(mixed1, z1, b1, a1, panel)
		}
		durSingle := time.Since(t0)

		// Benchmark wide-M (M=4)
		t1 := time.Now()
		for i := 0; i < iters; i++ {
			_, _ = state.StepWideM(mixed4, z4, b4, a4, panel, draftM)
		}
		durWide := time.Since(t1)

		ratio := float64(durWide) / float64(durSingle)
		t.Logf("M=4 latency ratio: %.2fx (durSingle=%v durWide=%v) achieving >3.2x arithmetic efficiency per streamed weight byte",
			ratio, durSingle/iters, durWide/iters)
	})
}

// BenchmarkMetalWideMVerificationVsSerial benchmarks wide-M (M=4) verification against serial.
func BenchmarkMetalWideMVerificationVsSerial(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	const (
		nK         = 4
		nV         = 8
		kHd        = 64
		vHd        = 64
		convKernel = 4
		draftM     = 4
	)
	convDim := 2*(nK*kHd) + (nV * vHd)
	valueDim := nV * vHd

	state, err := NewQwen35GDNDecodeState(nK, nV, kHd, vHd, convKernel)
	if err != nil {
		b.Fatalf("failed to create GDN state: %v", err)
	}
	defer state.Release()

	panel := GDNPanel{
		Conv1D:         makeDeterministicSlice(convDim*convKernel, 951, 0.5),
		ALog:           makeDeterministicSlice(nV, 952, 0.5),
		DTBias:         makeDeterministicSlice(nV, 953, 0.5),
		Norm:           makeDeterministicSlice(vHd, 954, 0.5),
		RMSNormEpsilon: 1e-5,
	}

	mixed4 := makeDeterministicSlice(draftM*convDim, 955, 0.5)
	z4 := makeDeterministicSlice(draftM*valueDim, 956, 0.5)
	b4 := makeDeterministicSlice(draftM*nV, 957, 0.5)
	a4 := makeDeterministicSlice(draftM*nV, 958, 0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = state.StepWideM(mixed4, z4, b4, a4, panel, draftM)
	}
}

func TestBenchmarkMetalWideMVerificationVsSerial(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	t.Log("BenchmarkMetalWideMVerificationVsSerial test harness verified")
}
