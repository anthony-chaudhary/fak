//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// v41ExactRMSNormHostRow is the host serial RMSNorm (norm.SerialRMSNorm): an
// in-order binary32 sum of squares, float64 sqrt and divide, (x*inv)*w.
func v41ExactRMSNormHostRow(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// v41ExactRMSNormWitnessRows spans the whole finite binary32 range by row: zero
// rows, rows whose squares are subnormal, a single dominant lane, BF16-rounded
// rows like the compressor's, and rows whose sum of squares overflows to +Inf.
func v41ExactRMSNormWitnessRows(rng *rand.Rand, rows, width int) []float32 {
	x := make([]float32, rows*width)
	for r := 0; r < rows; r++ {
		row := x[r*width : (r+1)*width]
		scale := math.Ldexp(1, rng.Intn(210)-145)
		switch r % 8 {
		case 0:
			if r%64 == 0 {
				continue // all-zero row: mean+eps is exactly eps
			}
			fallthrough
		default:
			for i := range row {
				row[i] = float32((rng.Float64()*2 - 1) * scale)
			}
		case 3:
			row[rng.Intn(width)] = float32((rng.Float64() + 0.5) * scale)
		case 5:
			for i := range row {
				v := float32((rng.Float64()*2 - 1) * scale)
				row[i] = math.Float32frombits((math.Float32bits(v) + 0x7fff + (math.Float32bits(v)>>16)&1) &^ 0xffff)
			}
		case 7:
			for i := range row {
				row[i] = math.Float32frombits(uint32(rng.Intn(0x800000)) | uint32(rng.Intn(2))<<31)
			}
		}
	}
	return x
}

// fak-test:runtime integration est=3s lane=optin
// Ref #13668. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// physical witness. Every output bit equals the host serial RMSNorm at the V4.1
// compressor (512) and index-key (128) widths: no tolerance.
func TestVulkanV41ExactRMSNormBitExact(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41ExactRMSNorm(128, 1e-6) {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 exact RMSNorm module or binary32 float controls")
		}
		t.Skip("V4.1 exact RMSNorm module not loaded")
	}
	if v.SupportsV41ExactRMSNorm(384, 1e-6) || v.SupportsV41ExactRMSNorm(128, 0) {
		t.Fatal("exact RMSNorm claims a width/eps outside its exact domain")
	}
	upload := func(t *testing.T, shape []int, x []float32) Tensor {
		d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 exact RMSNorm witness input")
		t.Cleanup(func() { v.Free(d) })
		return d
	}
	for _, tc := range []struct {
		width, rows int
		eps         float32
	}{{128, 1 << 15, 1e-6}, {512, 1 << 12, 1e-6}, {128, 1 << 12, 1e-5}, {512, 1, 1e-6}} {
		t.Run(fmt.Sprintf("width%d_rows%d_eps%g", tc.width, tc.rows, tc.eps), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(13668 + tc.width + tc.rows)))
			x := v41ExactRMSNormWitnessRows(rng, tc.rows, tc.width)
			w := make([]float32, tc.width)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			w[0] = math.Float32frombits(0x00000123)
			shape := []int{tc.rows, tc.width}
			if tc.rows == 1 {
				shape = []int{tc.width}
			}
			dx, dw := upload(t, shape, x), upload(t, []int{tc.width}, w)
			out, err := v.V41ExactRMSNorm(dx, dw, tc.eps)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { v.Free(out) })
			if !out.buf.(*vulkanBuf).v41CheckedRead || len(out.Shape) != len(shape) {
				t.Fatalf("output %v is not a fresh checked tensor shaped %v", out.Shape, shape)
			}
			got := v.Read(out)
			mismatches := 0
			for r := 0; r < tc.rows; r++ {
				want := v41ExactRMSNormHostRow(x[r*tc.width:(r+1)*tc.width], w, tc.eps)
				for i, wv := range want {
					if gv := got[r*tc.width+i]; math.Float32bits(gv) != math.Float32bits(wv) {
						if mismatches < 5 {
							t.Errorf("row %d lane %d: device %#08x (%g), host %#08x (%g)", r, i, math.Float32bits(gv), gv, math.Float32bits(wv), wv)
						}
						mismatches++
					}
				}
			}
			if mismatches > 0 {
				t.Fatalf("%d of %d lanes differ from the host serial RMSNorm", mismatches, tc.rows*tc.width)
			}
			// Control, not a gate: the generic tree-reduction RMSNorm on the same
			// rows, showing why these seams need the dedicated exact module.
			generic := v.RMSNorm(dx, dw, tc.eps)
			genericValues := append([]float32(nil), v.Read(generic)...)
			v.Free(generic)
			genericDiff := 0
			for r := 0; r < tc.rows; r++ {
				want := v41ExactRMSNormHostRow(x[r*tc.width:(r+1)*tc.width], w, tc.eps)
				for i, wv := range want {
					if math.Float32bits(genericValues[r*tc.width+i]) != math.Float32bits(wv) {
						genericDiff++
					}
				}
			}
			t.Logf("exact module: 0 of %d lanes differ; generic RMSNorm control: %d differ", tc.rows*tc.width, genericDiff)
		})
	}
	if _, err := v.V41ExactRMSNorm(upload(t, []int{96}, make([]float32, 96)), upload(t, []int{96}, make([]float32, 96)), 1e-6); err == nil {
		t.Fatal("non-power-of-two width dispatched")
	}
}
