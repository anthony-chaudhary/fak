//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// q8FFNDevice resolves the Vulkan backend for the fused Q8 FFN-down path and
// skips (or fails, when a device is required) exactly like the other Vulkan
// fusion tests.
func q8FFNDevice(t *testing.T) *vulkanBackend {
	t.Helper()
	be, ok := Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered")
		}
		t.Skip("Vulkan backend unavailable")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(be.Tier()), strings.ToLower(expected)) {
			t.Fatalf("device %q does not match required %q", be.Tier(), expected)
		}
	}
	v := be.(*vulkanBackend)
	if !v.haveQ8 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required device lacks Q8 support")
		}
		t.Skip("device lacks Q8 support")
	}
	return v
}

// q8FusedFFNCase holds one deterministic fixture: a prequantized Q8_0 down
// weight, gate/up activations, and the residual destination.
type q8FusedFFNCase struct {
	name   string
	out    int
	in     int
	batch  int
	codes  []int8
	scales []float32
	gate   []float32
	up     []float32
	dst    []float32
}

// buildQ8FusedFFNCase quantizes each weight row independently with the exact
// cpu reference scheme (quantizeVecQ8), and synthesizes gate/up where selected
// blocks are forced to zero (both gate and up zero => zero activation block, so
// the amax==0 branch that writes q=0/d=0 is exercised).
func buildQ8FusedFFNCase(name string, out, in, batch int, seed int64, zeroBlocks bool) q8FusedFFNCase {
	rng := rand.New(rand.NewSource(seed))
	nblk := in / 32
	codes := make([]int8, out*in)
	scales := make([]float32, out*nblk)
	rowWeights := make([]float32, in)
	for row := 0; row < out; row++ {
		for i := range rowWeights {
			rowWeights[i] = (rng.Float32()*2 - 1) * .25
		}
		rowCodes, rowScales := quantizeVecQ8(rowWeights, 32)
		copy(codes[row*in:], rowCodes)
		copy(scales[row*nblk:], rowScales)
	}
	gate := make([]float32, batch*in)
	up := make([]float32, batch*in)
	for i := range gate {
		gate[i] = float32(math.Sin(float64(i)*0.017+0.2)) * 0.7
		up[i] = float32(math.Cos(float64(i)*0.011+0.4)) * 0.7
		if zeroBlocks && (i/32)%7 == 3 {
			gate[i] = 0
			up[i] = 0
		}
	}
	dst := make([]float32, batch*out)
	for i := range dst {
		dst[i] = float32(math.Sin(float64(i)*0.05)) * 0.3
	}
	return q8FusedFFNCase{name: name, out: out, in: in, batch: batch, codes: codes, scales: scales, gate: gate, up: up, dst: dst}
}

// q8FusedFFNWant builds the independent CPU oracle: silu(gate)*up projected
// through the same Q8_0 weight, accumulated into the residual. Mirrors the
// fused kernel's per-block activation quantization.
func q8FusedFFNWant(t *testing.T, c q8FusedFFNCase) []float32 {
	t.Helper()
	be := Default()
	hw := NewQ8(be, []int{c.out, c.in}, c.codes, c.scales, 32)
	hgate := NewF32(be, []int{c.batch, c.in}, c.gate)
	hup := NewF32(be, []int{c.batch, c.in}, c.up)
	sw := be.SwiGLU(hgate, hup)
	proj := be.Read(be.BatchedMatMul(hw, sw, c.batch))
	want := append([]float32(nil), c.dst...)
	for i := range want {
		want[i] += proj[i]
	}
	return want
}

func assertQ8FusedParity(t *testing.T, got, want []float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: output elements=%d, want %d", label, len(got), len(want))
	}
	var sqErr, sqRef, maxAbs float64
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("%s: non-finite output at %d: %v", label, i, got[i])
		}
		d := float64(got[i] - want[i])
		sqErr += d * d
		sqRef += float64(want[i]) * float64(want[i])
		if a := math.Abs(d); a > maxAbs {
			maxAbs = a
		}
	}
	if !(sqRef > 0) {
		t.Fatalf("%s: zero reference norm", label)
	}
	relL2 := math.Sqrt(sqErr / sqRef)
	if relL2 > 1e-4 {
		t.Fatalf("%s: relative L2=%g > 1e-4", label, relL2)
	}
	if maxAbs > 1e-3 {
		t.Fatalf("%s: max abs=%g > 1e-3", label, maxAbs)
	}
}

// TestVulkanQ8FusedFFNDown witnesses the cooperative fused Q8 FFN-down kernel:
// 32 lanes per output, eight outputs per workgroup, ordered block reduction,
// and a two-dimensional output/token grid, all against the CPU oracle.
func TestVulkanQ8FusedFFNDown(t *testing.T) {
	v := q8FFNDevice(t)
	cases := []q8FusedFFNCase{
		buildQ8FusedFFNCase("p1_5120x17408", 5120, 17408, 1, 12697, true),
		buildQ8FusedFFNCase("p3_5120x17408", 5120, 17408, 3, 12698, true),
		buildQ8FusedFFNCase("p1_odd_out_259x2080", 259, 2080, 1, 12699, true),
		buildQ8FusedFFNCase("p2_odd_out_259x2080", 259, 2080, 2, 12700, true),
		buildQ8FusedFFNCase("p1_mixed_out_13x256", 13, 256, 1, 12701, false),
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			be := Default()
			hw := NewQ8(be, []int{tc.out, tc.in}, tc.codes, tc.scales, 32)
			hgate := NewF32(be, []int{tc.batch, tc.in}, tc.gate)
			hup := NewF32(be, []int{tc.batch, tc.in}, tc.up)
			hdst := NewF32(be, []int{tc.batch, tc.out}, tc.dst)
			w := v.Upload(hw, Q8_0)
			defer v.Free(w)
			gate := v.Upload(hgate, F32)
			defer v.Free(gate)
			up := v.Upload(hup, F32)
			defer v.Free(up)
			dst := v.Upload(hdst, F32)
			defer v.Free(dst)

			v.SwiGLUMatMulAddInPlace(dst, w, gate, up)
			got := v.Read(dst)
			want := q8FusedFFNWant(t, tc)
			assertQ8FusedParity(t, got, want, tc.name)

			for token := 0; token < tc.batch; token++ {
				lo, hi := token*tc.out, (token+1)*tc.out
				if argmaxF32(got[lo:hi]) != argmaxF32(want[lo:hi]) {
					t.Fatalf("%s: token %d argmax mismatch", tc.name, token)
				}
			}
		})
	}
}

// TestVulkanQ8BatchedRetainsControl proves the cooperative fused kernel preserves
// the batched (P>1) fused Q8 path — a regressed batched organization would show
// as token-row cross-talk or a window-boundary error.
func TestVulkanQ8Batched(t *testing.T) {
	v := q8FFNDevice(t)
	for _, batch := range []int{2, 3, 8} {
		tc := buildQ8FusedFFNCase("batched", 259, 2080, batch, int64(12800+batch), true)
		be := Default()
		hw := NewQ8(be, []int{tc.out, tc.in}, tc.codes, tc.scales, 32)
		hgate := NewF32(be, []int{tc.batch, tc.in}, tc.gate)
		hup := NewF32(be, []int{tc.batch, tc.in}, tc.up)
		hdst := NewF32(be, []int{tc.batch, tc.out}, tc.dst)
		w := v.Upload(hw, Q8_0)
		defer v.Free(w)
		gate := v.Upload(hgate, F32)
		defer v.Free(gate)
		up := v.Upload(hup, F32)
		defer v.Free(up)
		dst := v.Upload(hdst, F32)
		defer v.Free(dst)
		v.SwiGLUMatMulAddInPlace(dst, w, gate, up)
		got := v.Read(dst)
		want := q8FusedFFNWant(t, tc)
		assertQ8FusedParity(t, got, want, "batched")
	}
}

// TestVulkanQ8FusedFFNDownLargeTokenGrid proves the two-dimensional grid keeps
// P beyond the 65,535-group single-dimension limit expressible (token rows on Y).
func TestVulkanQ8FusedFFNDownLargeTokenGrid(t *testing.T) {
	v := q8FFNDevice(t)
	const batch = 70000
	out, in := 13, 256
	rng := rand.New(rand.NewSource(12702))
	codes := make([]int8, out*in)
	scales := make([]float32, out*(in/32))
	rowWeights := make([]float32, in)
	for row := 0; row < out; row++ {
		for i := range rowWeights {
			rowWeights[i] = (rng.Float32()*2 - 1) * .25
		}
		rowCodes, rowScales := quantizeVecQ8(rowWeights, 32)
		copy(codes[row*in:], rowCodes)
		copy(scales[row*(in/32):], rowScales)
	}
	gate := make([]float32, batch*in)
	up := make([]float32, batch*in)
	for i := range gate {
		gate[i] = float32(math.Sin(float64(i)*0.017+0.2)) * 0.7
		up[i] = float32(math.Cos(float64(i)*0.011+0.4)) * 0.7
	}
	dst := make([]float32, batch*out)
	be := Default()
	hw := NewQ8(be, []int{out, in}, codes, scales, 32)
	hgate := NewF32(be, []int{batch, in}, gate)
	hup := NewF32(be, []int{batch, in}, up)
	hdst := NewF32(be, []int{batch, out}, dst)
	w := v.Upload(hw, Q8_0)
	defer v.Free(w)
	dgate := v.Upload(hgate, F32)
	defer v.Free(dgate)
	dup := v.Upload(hup, F32)
	defer v.Free(dup)
	ddst := v.Upload(hdst, F32)
	defer v.Free(ddst)
	v.SwiGLUMatMulAddInPlace(ddst, w, dgate, dup)
	got := v.Read(ddst)
	// CPU oracle on a bounded prefix (a full 70000-row oracle is unnecessary for a
	// grid-expressibility witness): the first and last token rows must be finite.
	for _, token := range []int{0, batch - 1} {
		lo, hi := token*out, (token+1)*out
		for i := lo; i < hi; i++ {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("token %d index %d non-finite: %v", token, i, got[i])
			}
		}
	}
	// A truncated oracle over the first three token rows guards real correctness.
	trunc := q8FusedFFNCase{name: "large_grid_prefix", out: out, in: in, batch: 3,
		codes: codes, scales: scales, gate: gate[:3*in], up: up[:3*in], dst: dst[:3*out]}
	want := q8FusedFFNWant(t, trunc)
	assertQ8FusedParity(t, got[:3*out], want, "large_grid_prefix")
}
