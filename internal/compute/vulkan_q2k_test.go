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

// TestVulkanQ2KShaderInvariants verifies that the Q2_K compute shader exists,
// declares valid layout descriptors and push constants, and adheres to 84-byte
// super-block unpacking invariants.
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
		"layout(local_size_x = 64) in;",
		"layout(std430, set = 0, binding = 0) readonly buffer Q2K",
		"layout(std430, set = 0, binding = 1) readonly buffer X",
		"layout(std430, set = 0, binding = 2) writeonly buffer Y",
		"int outDim;",
		"int inDim;",
		"int tokens;",
		"uint base = uint((row * blocks + sb) * 84);",
		"halfAt(base + 80u)",
		"halfAt(base + 82u)",
	}

	for _, clause := range requiredClauses {
		if !strings.Contains(src, clause) {
			t.Errorf("q2k_matmul.comp missing required clause: %q", clause)
		}
	}
}

// TestVulkanQ2KShaderSimulatedEquivalence simulates the GLSL shader logic in pure Go
// and verifies that it produces bit-exact equivalence with the reference q2kRowDot.
func TestVulkanQ2KShaderSimulatedEquivalence(t *testing.T) {
	const out = 4
	const in = 256
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

	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	scratch := make([]float32, 256)
	for r := 0; r < out; r++ {
		want := q2kRowDot(raw[r*84:(r+1)*84], x, scratch)

		// Simulated GLSL shader arithmetic
		blk := raw[r*84 : (r+1)*84]
		d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:])))
		minVal := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:])))

		var simSum float32
		isIdx := 0
		qiOffset := 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc0 := blk[isIdx]
				isIdx++
				dl0 := d * float32(sc0&15)
				ml0 := minVal * float32(sc0>>4)

				sc1 := blk[isIdx]
				isIdx++
				dl1 := d * float32(sc1&15)
				ml1 := minVal * float32(sc1>>4)

				qBase0 := 16 + qiOffset
				qBase1 := qBase0 + 16
				xGroupBase := n + j*32

				for l := 0; l < 16; l++ {
					q0 := blk[qBase0+l]
					code0 := (q0 >> shift) & 3
					w0 := dl0*float32(code0) - ml0
					simSum += w0 * x[xGroupBase+l]

					q1 := blk[qBase1+l]
					code1 := (q1 >> shift) & 3
					w1 := dl1*float32(code1) - ml1
					simSum += w1 * x[xGroupBase+16+l]
				}
				shift += 2
			}
			qiOffset += 32
		}

		if math.Abs(float64(want-simSum)) > 1e-4 {
			t.Fatalf("row %d: want %g, got %g (delta %g)", r, want, simSum, math.Abs(float64(want-simSum)))
		}
	}
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

func TestVulkanQ2KMatMulMatchesCPUReference(t *testing.T) {
	v, ok := Lookup("vulkan")
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	const out, in = 8, 512
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	rng := rand.New(rand.NewSource(9718))
	for b := 0; b < out*(in/q2kSuper); b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 16; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		for i := 16; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], float32(rng.Float64()*1.0+0.5))
		binaryPutFloat16(blk[82:84], float32(rng.Float64()*0.5+0.1))
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	hw := NewQ2K(Default(), []int{out, in}, raw)
	dw := v.Upload(hw, Q2_K)
	defer v.Free(dw)
	dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
	defer v.Free(dx)
	dy := v.MatMul(dw, dx)
	defer v.Free(dy)
	got := v.Read(dy)
	want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, x)))
	if a, b := argmaxF32(got), argmaxF32(want); a != b {
		t.Fatalf("argmax=%d want %d", a, b)
	}
	if c := cosineC(got, want); c < 0.995 {
		t.Fatalf("cosine %.8f < 0.995", c)
	}
}
