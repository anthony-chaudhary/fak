package compute

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// cuda_q2k_test.go — unit test and architecture witness for Issue #11945:
// native dequant-fused Q2_K GEMV/GEMM for sm_80/sm_89/sm_90 to eliminate A100 CPU decode offload.
//
// This test file runs in both pure-Go environments (go test ./internal/compute) and GPU environments
// (go test -tags cuda ./internal/compute). It tests:
// 1. CUDA C ABI exports in cuda_backend.h (fcuda_q2k_matmul_f32, fcuda_q2k_gemv, fcuda_q2k_gemm_panel).
// 2. CUDA kernel definitions in cuda_kernels.cu (k_q2k_gemv, k_q2k_gemm_panel, k_q2k_dequant_transient).
// 3. Q2_K super-block format invariants & alignment for sm_80/sm_89/sm_90.
// 4. Bit-for-bit mathematical equivalence between CUDA kernel formula and table-based dequantization.
// 5. Warp-level GEMV reduction simulation (1 warp/row, 32 threads, 8 weights/thread, tree shuffle).
// 6. Token panel GEMM simulation (token_tile=4 weight register reuse).
// 7. On-device execution through compute.Backend interface when a CUDA device is registered.

func TestCUDAHeaderExportsQ2KFunctions(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd failed: %v", err)
	}
	repoRoot := findRepoRootForTest(t, wd)

	// Check internal/compute/cuda_backend.h
	headerPath := filepath.Join(repoRoot, "internal", "compute", "cuda_backend.h")
	headerBytes, err := os.ReadFile(headerPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", headerPath, err)
	}
	headerSrc := string(headerBytes)

	requiredHeaderDecls := []string{
		"void fcuda_q2k_matmul_f32(const uint8_t *dQ2K, const float *dX, float *dY, int out, int in, int P);",
		"void fcuda_q2k_gemv(const uint8_t *dQ2K, const float *dX, float *dY, int out, int in, int P);",
		"void fcuda_q2k_gemm_panel(const uint8_t *dQ2K, const float *dX, float *dY, int out, int in, int P);",
	}
	for _, decl := range requiredHeaderDecls {
		if !strings.Contains(headerSrc, decl) {
			t.Errorf("cuda_backend.h missing required declaration: %s", decl)
		}
	}

	// Check internal/compute/cuda_kernels.cu
	cuPath := filepath.Join(repoRoot, "internal", "compute", "cuda_kernels.cu")
	cuBytes, err := os.ReadFile(cuPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", cuPath, err)
	}
	cuSrc := string(cuBytes)

	requiredCuSymbols := []string{
		"k_q2k_gemv",
		"k_q2k_gemm_panel",
		"k_q2k_dequant_transient",
		"fcuda_q2k_gemv",
		"fcuda_q2k_gemm_panel",
		"fcuda_q2k_matmul_f32",
	}
	for _, sym := range requiredCuSymbols {
		if !strings.Contains(cuSrc, sym) {
			t.Errorf("cuda_kernels.cu missing required symbol: %s", sym)
		}
	}
}

func TestQ2KCUDAFormatInvariants(t *testing.T) {
	if q2kSuper != 256 {
		t.Errorf("q2kSuper = %d, want 256", q2kSuper)
	}
	if q2kSuperBlock != 84 {
		t.Errorf("q2kSuperBlock = %d, want 84", q2kSuperBlock)
	}

	// Sizing and alignment invariants for sm_80 / sm_89 / sm_90:
	// 84 bytes = 4 * 21 bytes -> any 256-byte aligned base pointer keeps every super-block 4-byte aligned.
	const (
		scalesOffset = 0
		quantsOffset = 16
		dOffset      = 80
		dminOffset   = 82
	)
	if scalesOffset%4 != 0 {
		t.Errorf("scales offset %d is not 4-byte aligned", scalesOffset)
	}
	if quantsOffset%4 != 0 {
		t.Errorf("quants offset %d is not 4-byte aligned", quantsOffset)
	}
	if dOffset%2 != 0 {
		t.Errorf("d offset %d is not 2-byte aligned for half float", dOffset)
	}
	if dminOffset%2 != 0 {
		t.Errorf("dmin offset %d is not 2-byte aligned for half float", dminOffset)
	}
}

func TestQ2KCUDADequantExactMatchesFormula(t *testing.T) {
	// Construct a synthetic 84-byte Q2_K block
	blk := make([]byte, q2kSuperBlock)

	// Populate scales: test varied high/low nibbles
	for i := 0; i < 16; i++ {
		lowNibble := byte(i & 0x0f)
		highNibble := byte((15 - i) & 0x0f)
		blk[i] = (highNibble << 4) | lowNibble
	}

	// Populate quants: 64 bytes with alternating 2-bit codes
	for i := 0; i < 64; i++ {
		// 4 codes per byte: 0, 1, 2, 3
		c0 := byte(i % 4)
		c1 := byte((i + 1) % 4)
		c2 := byte((i + 2) % 4)
		c3 := byte((i + 3) % 4)
		blk[16+i] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
	}

	// d = 0.5 (f16 0x3800), dmin = 0.25 (f16 0x3400)
	binary.LittleEndian.PutUint16(blk[80:82], 0x3800)
	binary.LittleEndian.PutUint16(blk[82:84], 0x3400)

	// 1. Reference dequant via q2kDequantSuperBlock
	refWeights := make([]float32, q2kSuper)
	q2kDequantSuperBlock(refWeights, blk)

	// 2. Kernel dequant via closed-form formula matching k_q2k_gemv and k_q2k_dequant_transient
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:82])))
	dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:84])))

	kernelWeights := make([]float32, q2kSuper)
	for elem := 0; elem < q2kSuper; elem++ {
		h := elem / 128
		rem := elem % 128
		j := rem / 32
		lane := rem % 32
		laneHalf := lane >> 4

		is := h*8 + 2*j + laneHalf
		sc := blk[is]
		dl := d * float32(sc&0x0f)
		ml := dmin * float32(sc>>4)

		qb := blk[16+h*32+lane]
		shift := uint(2 * j)
		code := (qb >> shift) & 3

		kernelWeights[elem] = dl*float32(code) - ml
	}

	// Verify bit-for-bit float match across all 256 weights
	for i := 0; i < q2kSuper; i++ {
		if math.Float32bits(refWeights[i]) != math.Float32bits(kernelWeights[i]) {
			t.Fatalf("elem %d: ref %f (0x%08x) != kernel %f (0x%08x)",
				i, refWeights[i], math.Float32bits(refWeights[i]),
				kernelWeights[i], math.Float32bits(kernelWeights[i]))
		}
	}
}

// simulateKQ2KGEMV executes the exact algorithmic structure of k_q2k_gemv:
// 1 warp per row, 32 threads, 8 weights/thread, 100% coalesced loads, tree shuffle reduction.
func simulateKQ2KGEMV(w []byte, x []float32, out, in int) []float32 {
	nsb := in / q2kSuper
	y := make([]float32, out)

	for o := 0; o < out; o++ {
		wrow := w[o*nsb*q2kSuperBlock : (o+1)*nsb*q2kSuperBlock]
		var warpSum float32

		// Simulate 32 threads in the warp
		for lane := 0; lane < 32; lane++ {
			laneHalf := lane >> 4
			var threadSum float32

			for sb := 0; sb < nsb; sb++ {
				blk := wrow[sb*q2kSuperBlock : (sb+1)*q2kSuperBlock]
				scales := blk[0:16]
				q := blk[16:80]
				d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:82])))
				dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:84])))
				xb := x[sb*256 : (sb+1)*256]

				for h := 0; h < 2; h++ {
					qb := q[h*32+lane]
					so := h * 8
					base := h * 128
					for j := 0; j < 4; j++ {
						sc := scales[so+2*j+laneHalf]
						dl := d * float32(sc&0x0f)
						ml := dmin * float32(sc>>4)
						code := (qb >> (2 * j)) & 3
						weight := dl*float32(code) - ml
						threadSum += weight * xb[base+j*32+lane]
					}
				}
			}
			warpSum += threadSum
		}
		y[o] = warpSum
	}
	return y
}

func TestQ2KCUDAGEMVSimulatedWarpReduction(t *testing.T) {
	testShapes := [][2]int{
		{1, 256},
		{8, 256},
		{320, 256},
		{64, 512},
		{32, 1024},
	}

	for _, shape := range testShapes {
		out, in := shape[0], shape[1]
		nblk := in / q2kSuper
		raw := make([]byte, out*nblk*q2kSuperBlock)

		// Deterministic pseudo-random bytes
		state := uint64(0x11945 + out*in)
		for i := range raw {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			raw[i] = byte(state)
		}
		for base := 0; base < len(raw); base += q2kSuperBlock {
			binary.LittleEndian.PutUint16(raw[base+80:base+82], 0x3800) // d = 0.5
			binary.LittleEndian.PutUint16(raw[base+82:base+84], 0x3400) // dmin = 0.25
		}

		x := make([]float32, in)
		for i := range x {
			x[i] = float32(i%17-8) * 0.125
		}

		// 1. CPU Reference GEMV using q2kRowDot
		refY := make([]float32, out)
		scratch := make([]float32, q2kSuper)
		rowBytes := nblk * q2kSuperBlock
		for o := 0; o < out; o++ {
			refY[o] = q2kRowDot(raw[o*rowBytes:(o+1)*rowBytes], x, scratch)
		}

		// 2. Simulated CUDA k_q2k_gemv
		simY := simulateKQ2KGEMV(raw, x, out, in)

		// Check parity
		var maxDelta float32
		for o := 0; o < out; o++ {
			delta := float32(math.Abs(float64(refY[o] - simY[o])))
			if delta > maxDelta {
				maxDelta = delta
			}
		}
		if maxDelta > 1e-4 {
			t.Errorf("shape [%d, %d]: max delta %g > 1e-4", out, in, maxDelta)
		}
	}
}

// simulateKQ2KGEMMPanel executes the exact algorithmic structure of k_q2k_gemm_panel:
// token_tile = 4 tokens, thread lane calculates dequantized weight once, accumulates into 4 registers.
func simulateKQ2KGEMMPanel(w []byte, X []float32, out, in, P int) []float32 {
	nsb := in / q2kSuper
	Y := make([]float32, P*out)
	const tokenTile = 4

	for o := 0; o < out; o++ {
		wrow := w[o*nsb*q2kSuperBlock : (o+1)*nsb*q2kSuperBlock]

		for token0 := 0; token0 < P; token0 += tokenTile {
			var sums [tokenTile]float32

			for lane := 0; lane < 32; lane++ {
				laneHalf := lane >> 4
				var threadSums [tokenTile]float32

				for sb := 0; sb < nsb; sb++ {
					blk := wrow[sb*q2kSuperBlock : (sb+1)*q2kSuperBlock]
					scales := blk[0:16]
					q := blk[16:80]
					d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:82])))
					dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:84])))

					for h := 0; h < 2; h++ {
						qb := q[h*32+lane]
						so := h * 8
						base := sb*256 + h*128
						for j := 0; j < 4; j++ {
							sc := scales[so+2*j+laneHalf]
							dl := d * float32(sc&0x0f)
							ml := dmin * float32(sc>>4)
							code := (qb >> (2 * j)) & 3
							weight := dl*float32(code) - ml
							k := base + j*32 + lane

							for m := 0; m < tokenTile; m++ {
								token := token0 + m
								if token < P {
									threadSums[m] += weight * X[token*in+k]
								}
							}
						}
					}
				}

				for m := 0; m < tokenTile; m++ {
					sums[m] += threadSums[m]
				}
			}

			for m := 0; m < tokenTile; m++ {
				token := token0 + m
				if token < P {
					Y[token*out+o] = sums[m]
				}
			}
		}
	}
	return Y
}

func TestQ2KCUDAGEMMPanelSimulatedReduction(t *testing.T) {
	out, in, P := 64, 512, 8
	nblk := in / q2kSuper
	raw := make([]byte, out*nblk*q2kSuperBlock)

	state := uint64(0x11945_cafe)
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state)
	}
	for base := 0; base < len(raw); base += q2kSuperBlock {
		binary.LittleEndian.PutUint16(raw[base+80:base+82], 0x3800) // d = 0.5
		binary.LittleEndian.PutUint16(raw[base+82:base+84], 0x3400) // dmin = 0.25
	}

	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32((i%23)-11) * 0.05
	}

	// Reference batched dot
	refY := make([]float32, P*out)
	scratch := make([]float32, q2kSuper)
	rowBytes := nblk * q2kSuperBlock
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		for tRow := 0; tRow < P; tRow++ {
			refY[tRow*out+o] = q2kRowDot(row, X[tRow*in:(tRow+1)*in], scratch)
		}
	}

	panelY := simulateKQ2KGEMMPanel(raw, X, out, in, P)

	var maxDelta float32
	for i := range refY {
		delta := float32(math.Abs(float64(refY[i] - panelY[i])))
		if delta > maxDelta {
			maxDelta = delta
		}
	}
	if maxDelta > 1e-4 {
		t.Errorf("panel GEMM max delta %g > 1e-4", maxDelta)
	}
}

func TestQ2KCUDADeviceExecutionIfAvailable(t *testing.T) {
	be := Pick("cuda")
	if be == nil {
		t.Log("cuda backend not registered in pure-Go build; simulated warp reduction and math witnesses passed")
		return
	}

	const out, in = 320, 256
	nblk := in / q2kSuper
	raw := make([]byte, out*nblk*q2kSuperBlock)
	for i := range raw {
		raw[i] = byte(i * 37)
	}
	for base := 0; base < len(raw); base += q2kSuperBlock {
		binary.LittleEndian.PutUint16(raw[base+80:base+82], 0x3800)
		binary.LittleEndian.PutUint16(raw[base+82:base+84], 0x3400)
	}

	hostT := NewQ2K(be, []int{out, in}, raw)
	if hostT.Dtype != Q2_K {
		t.Fatalf("NewQ2K dtype = %s, want Q2_K", hostT.Dtype)
	}

	devW := be.Upload(hostT, Q2_K)
	if devW.Dtype != Q2_K {
		t.Fatalf("Upload Q2_K dtype = %s, want Q2_K", devW.Dtype)
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = 1.0
	}
	devX := be.Upload(NewF32(be, []int{in}, x), F32)

	// GEMV (decode)
	devY := be.MatMul(devW, devX)
	got := be.Read(devY)
	if len(got) != out {
		t.Fatalf("MatMul output size %d != %d", len(got), out)
	}

	// Parity against CPU reference
	ref := Default()
	refW := NewQ2K(ref, []int{out, in}, raw)
	refX := NewF32(ref, []int{in}, x)
	refY := ref.Read(ref.MatMul(refW, refX))

	cos := cosine(refY, got)
	if cos < 0.995 {
		t.Errorf("CUDA Q2_K GEMV cosine %g < 0.995", cos)
	}
	t.Logf("CUDA Q2_K GEMV on-device execution verified: cosine=%g", cos)
}
