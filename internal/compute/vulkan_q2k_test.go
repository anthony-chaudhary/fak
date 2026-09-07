package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// TestVulkanQ2KShaderInvariants verifies that the Q2_K compute shader uses one
// cooperative Wave32 per output row, reuses each dequantized weight across a
// four-token panel, and preserves the 84-byte super-block layout.
func TestVulkanQ2KShaderInvariants(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd failed: %v", err)
	}
	repoRoot := findRepoRootForTest(t, wd)
	shaderPath := filepath.Join(repoRoot, "internal", "compute", "shaders", "q2k_matmul.comp")

	shaderBytes, err := os.ReadFile(shaderPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", shaderPath, err)
	}
	src := string(shaderBytes)

	requiredClauses := []string{
		"#version 450",
		"#extension GL_KHR_shader_subgroup_arithmetic : require",
		"#extension GL_KHR_shader_subgroup_shuffle : require",
		"layout(local_size_x = 32) in;",
		"layout(std430, set = 0, binding = 0) readonly buffer Q2K",
		"layout(std430, set = 0, binding = 1) readonly buffer X",
		"layout(std430, set = 0, binding = 2) writeonly buffer Y",
		"int outDim;",
		"int inDim;",
		"int tokens;",
		"const uint TOKEN_TILE = 4u;",
		"uint baseWord = (row * blocks + sb) * 21u;",
		"raw[baseWord + 20u]",
		"subgroupShuffle(qWord, lane >> 2u)",
		"subgroupAdd(sums[m])",
	}

	for _, clause := range requiredClauses {
		if !strings.Contains(src, clause) {
			t.Errorf("q2k_matmul.comp missing required clause: %q", clause)
		}
	}

	for _, forbidden := range []string{
		"layout(local_size_x = 64) in;",
		"gl_GlobalInvocationID",
	} {
		if strings.Contains(src, forbidden) {
			t.Errorf("q2k_matmul.comp retained scalar-dispatch clause %q", forbidden)
		}
	}

	shimPath := filepath.Join(repoRoot, "internal", "compute", "vulkan_shim.cpp")
	shimBytes, err := os.ReadFile(shimPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", shimPath, err)
	}
	shim := string(shimBytes)
	for _, clause := range []string{
		"ok &= buildKernel(g_kern[K_Q2K_MATMUL]",
		"const size_t tokenPanels = ((size_t)P + 3u) / 4u;",
		"(size_t)out * tokenPanels",
	} {
		if !strings.Contains(shim, clause) {
			t.Errorf("vulkan_shim.cpp missing Wave32 panel clause %q", clause)
		}
	}
	if strings.Contains(shim, "dispatch(g_kern[K_Q2K_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)(((size_t)out * P + 63) / 64))") {
		t.Error("vulkan_shim.cpp retained scalar Q2_K dispatch geometry")
	}

	buildPath := filepath.Join(repoRoot, "internal", "compute", "build_vulkan.ps1")
	buildBytes, err := os.ReadFile(buildPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", buildPath, err)
	}
	if !strings.Contains(string(buildBytes), `"q2k_matmul"`) {
		t.Error("build_vulkan.ps1 does not compile q2k_matmul.comp")
	}
}

// TestVulkanQ2KShaderSimulatedEquivalence simulates the Wave32 four-token panel
// in pure Go and verifies decode, complete-panel, and panel-tail equivalence with
// the reference q2kRowDot implementation.
func TestVulkanQ2KShaderSimulatedEquivalence(t *testing.T) {
	const (
		out  = 4
		in   = 512
		maxP = 7
	)
	rng := rand.New(rand.NewSource(2026))

	raw := make([]byte, out*(in/256)*84)
	for b := 0; b < out*(in/256); b++ {
		blk := raw[b*84 : (b+1)*84]
		for i := 0; i < 16; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		for i := 16; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], 1.5)
		binaryPutFloat16(blk[82:84], 0.5)
	}

	X := make([]float32, maxP*in)
	for i := range X {
		X[i] = rng.Float32()*2 - 1
	}

	for _, P := range []int{1, 3, 4, 7} {
		got := simulateVulkanQ2KWave32Panel(raw, X[:P*in], out, in, P)
		want := make([]float32, P*out)
		scratch := make([]float32, 256)
		rowBytes := (in / 256) * 84
		for token := 0; token < P; token++ {
			for row := 0; row < out; row++ {
				want[token*out+row] = q2kRowDot(
					raw[row*rowBytes:(row+1)*rowBytes],
					X[token*in:(token+1)*in],
					scratch,
				)
			}
		}

		if c := cosineC(got, want); c < 0.999990 {
			t.Fatalf("P=%d: Wave32 panel cosine %.8f < 0.999990", P, c)
		}
		for i := range got {
			if delta := math.Abs(float64(got[i] - want[i])); delta > 1e-3 {
				t.Fatalf("P=%d index=%d: want %g, got %g (delta %g)", P, i, want[i], got[i], delta)
			}
		}
	}
}

func simulateVulkanQ2KWave32Panel(raw []byte, X []float32, out, in, P int) []float32 {
	const tokenTile = 4
	blocks := in / 256
	rowBytes := blocks * 84
	Y := make([]float32, P*out)

	for token0 := 0; token0 < P; token0 += tokenTile {
		for row := 0; row < out; row++ {
			var laneSums [tokenTile][32]float32
			wrow := raw[row*rowBytes : (row+1)*rowBytes]
			for sb := 0; sb < blocks; sb++ {
				blk := wrow[sb*84 : (sb+1)*84]
				d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:])))
				minValue := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:])))

				for lane := 0; lane < 32; lane++ {
					laneHalf := lane >> 4
					for h := 0; h < 2; h++ {
						qb := blk[16+h*32+lane]
						for j := 0; j < 4; j++ {
							sc := blk[h*8+2*j+laneHalf]
							dl := d * float32(sc&15)
							ml := minValue * float32(sc>>4)
							code := (qb >> (2 * j)) & 3
							weight := dl*float32(code) - ml
							k := sb*256 + h*128 + j*32 + lane
							for m := 0; m < tokenTile && token0+m < P; m++ {
								laneSums[m][lane] += weight * X[(token0+m)*in+k]
							}
						}
					}
				}
			}

			for m := 0; m < tokenTile && token0+m < P; m++ {
				var total float32
				for lane := 0; lane < 32; lane++ {
					total += laneSums[m][lane]
				}
				Y[(token0+m)*out+row] = total
			}
		}
	}
	return Y
}

func binaryPutFloat16(b []byte, f float32) {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x1
	exp := int((bits>>23)&0xff) - 127
	frac := bits & 0x7fffff

	var h uint16
	if exp > 15 {
		h = uint16(sign<<15 | 0x1f<<10)
	} else if exp < -14 {
		h = uint16(sign << 15)
	} else {
		h = uint16(sign<<15 | uint32(exp+15)<<10 | (frac >> 13))
	}
	b[0] = byte(h)
	b[1] = byte(h >> 8)
}

func makeVulkanQ2KFixture(rng *rand.Rand, out, in int) ([]byte, Tensor) {
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < out*(in/q2kSuper); b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], float32(rng.Float64()+0.5))
		binaryPutFloat16(blk[82:84], float32(rng.Float64()*0.5+0.1))
	}
	return raw, NewQ2K(Default(), []int{out, in}, raw)
}

func TestVulkanQ2KMatMulMatchesCPUReference(t *testing.T) {
	v, ok := Lookup("vulkan")
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	rng := rand.New(rand.NewSource(9718))

	for _, tc := range []struct {
		name string
		out  int
		P    int
	}{
		{name: "decode", out: 64, P: 1},
		{name: "decode_row_tail", out: 67, P: 1},
		{name: "prefill_panel_tail", out: 67, P: 3},
		{name: "prefill_full_panel", out: 67, P: 4},
		{name: "prefill_multi_panel_tail", out: 67, P: 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const in = 512
			_, hw := makeVulkanQ2KFixture(rng, tc.out, in)
			dw := v.Upload(hw, Q2_K)
			defer v.Free(dw)

			X := make([]float32, tc.P*in)
			for i := range X {
				X[i] = rng.Float32()*2 - 1
			}
			dx := v.Upload(NewF32(Default(), []int{tc.P, in}, X), F32)
			defer v.Free(dx)

			var dy Tensor
			if tc.P == 1 {
				dy = v.MatMul(dw, dx)
			} else {
				dy = v.BatchedMatMul(dw, dx, tc.P)
			}
			defer v.Free(dy)
			got := v.Read(dy)

			ref := Default()
			var want []float32
			if tc.P == 1 {
				want = ref.Read(ref.MatMul(hw, NewF32(ref, []int{in}, X)))
			} else {
				want = ref.Read(ref.BatchedMatMul(hw, NewF32(ref, []int{tc.P, in}, X), tc.P))
			}
			if c := cosineC(got, want); c < 0.999990 {
				t.Fatalf("cosine %.8f < 0.999990", c)
			}
			for token := 0; token < tc.P; token++ {
				gotRow := got[token*tc.out : (token+1)*tc.out]
				wantRow := want[token*tc.out : (token+1)*tc.out]
				if a, b := argmaxF32(gotRow), argmaxF32(wantRow); a != b {
					t.Fatalf("token %d argmax=%d want %d", token, a, b)
				}
			}
		})
	}
}

func BenchmarkVulkanQ2KMatMul(b *testing.B) {
	v, ok := Lookup("vulkan")
	if !ok {
		b.Skip("Vulkan backend unavailable")
	}
	for _, tc := range []struct {
		name string
		P    int
	}{
		{name: "decode_5120x5120_p1", P: 1},
		{name: "prefill_5120x5120_p4", P: 4},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const out, in = 5120, 5120
			rng := rand.New(rand.NewSource(9718 + int64(tc.P)))
			_, hw := makeVulkanQ2KFixture(rng, out, in)
			dw := v.Upload(hw, Q2_K)
			defer v.Free(dw)

			X := make([]float32, tc.P*in)
			for i := range X {
				X[i] = rng.Float32()*2 - 1
			}
			dx := v.Upload(NewF32(Default(), []int{tc.P, in}, X), F32)
			defer v.Free(dx)

			b.ReportMetric(float64(tc.P), "tokens/op")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var dy Tensor
				if tc.P == 1 {
					dy = v.MatMul(dw, dx)
				} else {
					dy = v.BatchedMatMul(dw, dx, tc.P)
				}
				v.Free(dy)
			}
		})
	}
}
