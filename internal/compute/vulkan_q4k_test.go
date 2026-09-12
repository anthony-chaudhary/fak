package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
)

type q4FusedBackend interface {
	Backend
	RMSNormMatMul2(Tensor, Tensor, Tensor, Tensor, float32) (Tensor, Tensor)
	SwiGLUMatMulAddInPlace(Tensor, Tensor, Tensor, Tensor)
}

func q4Device(t *testing.T) Backend {
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
	return be
}

func q4FusedDevice(t *testing.T) q4FusedBackend {
	t.Helper()
	be := q4Device(t)
	v, ok := be.(q4FusedBackend)
	if !ok {
		t.Fatal("Vulkan backend lacks fused operations")
	}
	return v
}

type q4kParityOracleEvent struct {
	Schema         string                  `json:"schema"`
	Selector       string                  `json:"selector"`
	TestName       string                  `json:"test_name"`
	OracleKind     string                  `json:"oracle_kind"`
	Engine         string                  `json:"engine"`
	DeviceObserved bool                    `json:"device_observed"`
	CaseCount      int                     `json:"case_count"`
	Passed         bool                    `json:"passed"`
	Observed       q4kParityOracleObserved `json:"observed"`
	Bounds         q4kParityOracleBounds   `json:"bounds"`
}

type q4kParityOracleObserved struct {
	Cosine      float64 `json:"cosine"`
	ArgmaxExact bool    `json:"argmax_exact"`
}

type q4kParityOracleBounds struct {
	MinCosine          float64 `json:"min_cosine"`
	RequireArgmaxExact bool    `json:"require_argmax_exact"`
}

func formatQ4KMatMulParityOracle(cosine float64, argmaxExact bool) ([]byte, error) {
	passed := argmaxExact && cosine >= 0.995
	event := q4kParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "q4k_matmul",
		TestName:       "TestVulkanQ4KMatMulMatchesCPUReference",
		OracleKind:     "cosine_argmax",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      1,
		Passed:         passed,
		Observed: q4kParityOracleObserved{
			Cosine:      cosine,
			ArgmaxExact: argmaxExact,
		},
		Bounds: q4kParityOracleBounds{
			MinCosine:          0.995,
			RequireArgmaxExact: true,
		},
	}
	return json.Marshal(event)
}

func TestVulkanQ4KMatMulMatchesCPUReference(t *testing.T) {
	v := q4Device(t)
	const out, in = 12, 768
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	rng := rand.New(rand.NewSource(9715))
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	hw := NewQ4K(Default(), []int{out, in}, raw)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
	defer v.Free(dx)
	dy := v.MatMul(dw, dx)
	defer v.Free(dy)
	got := v.Read(dy)
	want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, x)))
	gotArgmax, wantArgmax := argmaxF32(got), argmaxF32(want)
	argmaxExact := gotArgmax == wantArgmax
	if !argmaxExact {
		t.Fatalf("argmax=%d want %d", gotArgmax, wantArgmax)
	}
	c := cosineC(got, want)
	if c < 0.995 {
		t.Fatalf("cosine %.8f < 0.995", c)
	}
	oracleJSON, err := formatQ4KMatMulParityOracle(float64(c), argmaxExact)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
}

func TestVulkanQ4KMatMulParityOracleFormat(t *testing.T) {
	raw, err := formatQ4KMatMulParityOracle(0.998, true)
	if err != nil {
		t.Fatalf("formatQ4KMatMulParityOracle failed: %v", err)
	}
	var parsed struct {
		Schema         string `json:"schema"`
		Selector       string `json:"selector"`
		TestName       string `json:"test_name"`
		OracleKind     string `json:"oracle_kind"`
		Engine         string `json:"engine"`
		DeviceObserved bool   `json:"device_observed"`
		CaseCount      int    `json:"case_count"`
		Passed         bool   `json:"passed"`
		Observed       struct {
			Cosine      float64 `json:"cosine"`
			ArgmaxExact bool    `json:"argmax_exact"`
		} `json:"observed"`
		Bounds struct {
			MinCosine          float64 `json:"min_cosine"`
			RequireArgmaxExact bool    `json:"require_argmax_exact"`
		} `json:"bounds"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal formatted oracle failed: %v", err)
	}
	if parsed.Schema != "fak.strix.subkernel-parity/v1" {
		t.Errorf("schema = %q, want fak.strix.subkernel-parity/v1", parsed.Schema)
	}
	if parsed.Selector != "q4k_matmul" {
		t.Errorf("selector = %q, want q4k_matmul", parsed.Selector)
	}
	if parsed.TestName != "TestVulkanQ4KMatMulMatchesCPUReference" {
		t.Errorf("test_name = %q, want TestVulkanQ4KMatMulMatchesCPUReference", parsed.TestName)
	}
	if parsed.OracleKind != "cosine_argmax" {
		t.Errorf("oracle_kind = %q, want cosine_argmax", parsed.OracleKind)
	}
	if parsed.Engine != "fak-native/vulkan" {
		t.Errorf("engine = %q, want fak-native/vulkan", parsed.Engine)
	}
	if !parsed.DeviceObserved {
		t.Errorf("device_observed must be true")
	}
	if parsed.CaseCount != 1 {
		t.Errorf("case_count = %d, want 1", parsed.CaseCount)
	}
	if !parsed.Passed {
		t.Errorf("passed must be true")
	}
	if parsed.Observed.Cosine != 0.998 {
		t.Errorf("observed cosine = %f, want 0.998", parsed.Observed.Cosine)
	}
	if !parsed.Observed.ArgmaxExact {
		t.Errorf("observed argmax_exact must be true")
	}
	if parsed.Bounds.MinCosine != 0.995 {
		t.Errorf("bounds min_cosine = %f, want 0.995", parsed.Bounds.MinCosine)
	}
	if !parsed.Bounds.RequireArgmaxExact {
		t.Errorf("bounds require_argmax_exact must be true")
	}

	failRaw, err := formatQ4KMatMulParityOracle(0.990, true)
	if err != nil {
		t.Fatalf("format failed oracle failed: %v", err)
	}
	if err := json.Unmarshal(failRaw, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when cosine < 0.995")
	}

	failRaw2, err := formatQ4KMatMulParityOracle(0.998, false)
	if err != nil {
		t.Fatalf("format failed oracle 2 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw2, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 2 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when argmax_exact=false")
	}
}

// TestVulkanQ4KLongDot exercises the scalar Q4_K row dot on the long
// in=5120 / in=17408 decode shapes where a single loop-carried accumulator is
// most exposed. The device result (now eight independent accumulation chains)
// is compared against the canonical CPU Q4_K reference; the oracle is the CPU
// path, not a duplicate of the shader arithmetic, so reassociation damage would
// show up as a parity failure.
func TestVulkanQ4KLongDot(t *testing.T) {
	v := q4Device(t)
	const out = 7 // odd row count crosses a 64-lane workgroup boundary
	for _, in := range []int{5120, 17408} {
		t.Run(fmtInt(in), func(t *testing.T) {
			blocks := in / q4kSuper
			raw := make([]byte, out*blocks*q4kSuperBlock)
			rng := rand.New(rand.NewSource(int64(in) + 12685))
			for b := 0; b < out*blocks; b++ {
				randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
			}
			// Mixed-sign activations exercise cancellation across the eight chains.
			x := make([]float32, in)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			hw := NewQ4K(Default(), []int{out, in}, raw)
			dw := v.Upload(hw, Q4_K)
			defer v.Free(dw)
			dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
			defer v.Free(dx)
			dy := v.MatMul(dw, dx)
			defer v.Free(dy)
			got := v.Read(dy)
			want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, x)))
			if len(got) != out || len(want) != out {
				t.Fatalf("output len got=%d want=%d, expected %d", len(got), len(want), out)
			}
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("row %d is non-finite: %v", i, got[i])
				}
			}
			if ga, wa := argmaxF32(got), argmaxF32(want); ga != wa {
				t.Fatalf("argmax=%d want %d", ga, wa)
			}
			c := cosineC(got, want)
			if c < 0.99999 {
				t.Fatalf("cosine %.10f < 0.99999", c)
			}
			if l2 := relativeL2(got, want); l2 > 1e-4 {
				t.Fatalf("relative L2 %.10f > 1e-4", l2)
			}
		})
	}
}

func fmtInt(n int) string {
	return strconv.Itoa(n)
}

func relativeL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}
func TestStrixQuantParityEmitterContract(t *testing.T) {
	t.Run("q4k_matmul", func(t *testing.T) {
		TestVulkanQ4KMatMulParityOracleFormat(t)
	})
	t.Run("q2k_matmul", func(t *testing.T) {
		TestVulkanQ2KMatMulParityOracleFormat(t)
	})
}

func TestVulkanQ4KBatchedMatMulMultipleTokensMatchesCPUReference(t *testing.T) {
	v := q4Device(t)
	// out deliberately crosses a 64-lane workgroup boundary. With P > 1 the shader
	// must recover both token and row from the flattened X dispatch index.
	const out, in, P = 70, 768, 3
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	rng := rand.New(rand.NewSource(11803))
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	x := make([]float32, P*in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	hw := NewQ4K(Default(), []int{out, in}, raw)
	hx := NewF32(Default(), []int{P, in}, x)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	dx := v.Upload(hx, F32)
	defer v.Free(dx)
	dy := v.BatchedMatMul(dw, dx, P)
	defer v.Free(dy)

	got := v.Read(dy)
	want := Default().Read(Default().BatchedMatMul(hw, hx, P))
	if len(got) != P*out {
		t.Fatalf("output len=%d, want %d", len(got), P*out)
	}
	for token := 0; token < P; token++ {
		gotRow := got[token*out : (token+1)*out]
		wantRow := want[token*out : (token+1)*out]
		for row, value := range gotRow {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("token %d row %d is non-finite: %v", token, row, value)
			}
		}
		if gotArgmax, wantArgmax := argmaxF32(gotRow), argmaxF32(wantRow); gotArgmax != wantArgmax {
			t.Fatalf("token %d argmax=%d, want %d", token, gotArgmax, wantArgmax)
		}
		if cosine := cosineC(gotRow, wantRow); cosine < 0.995 {
			t.Fatalf("token %d cosine %.8f < 0.995", token, cosine)
		}
	}
}

func TestVulkanQ4KRMSNormMatMul2MatchesCPUReference(t *testing.T) {
	v := q4FusedDevice(t)
	inDims := []int{256, 512, 768}
	const out0, out1 = 12, 16
	rng := rand.New(rand.NewSource(9716))

	for _, in := range inDims {
		nblk := in / q4kSuper
		newWeight := func(out int, seedOffset int) Tensor {
			raw := make([]byte, out*nblk*q4kSuperBlock)
			for b := 0; b < out*nblk; b++ {
				fillCraftedQ4KBlock(raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+seedOffset)
			}
			return NewQ4K(Default(), []int{out, in}, raw)
		}
		hw0, hw1 := newWeight(out0, 0), newWeight(out1, 100)
		x := make([]float32, in)
		norm := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
			norm[i] = 0.5 + rng.Float32()
		}
		dw0 := v.Upload(hw0, Q4_K)
		defer v.Free(dw0)
		dw1 := v.Upload(hw1, Q4_K)
		defer v.Free(dw1)
		dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
		defer v.Free(dx)
		dnorm := v.Upload(NewF32(Default(), []int{in}, norm), F32)
		defer v.Free(dnorm)
		const eps = float32(1e-6)
		got0, got1 := v.RMSNormMatMul2(dw0, dw1, dx, dnorm, eps)
		defer v.Free(got0)
		defer v.Free(got1)
		hx := NewF32(Default(), []int{in}, x)
		hnorm := NewF32(Default(), []int{in}, norm)
		xn := Default().RMSNorm(hx, hnorm, eps)
		want0 := Default().Read(Default().MatMul(hw0, xn))
		want1 := Default().Read(Default().MatMul(hw1, xn))
		for name, pair := range map[string][2][]float32{
			"projection 0": {v.Read(got0), want0},
			"projection 1": {v.Read(got1), want1},
		} {
			got, want := pair[0], pair[1]
			var sqErr, sqRef float64
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("%s in=%d non-finite value at %d: %v", name, in, i, got[i])
				}
				diff := float64(got[i] - want[i])
				sqErr += diff * diff
				sqRef += float64(want[i]) * float64(want[i])
			}
			if sqRef <= 0 {
				t.Fatalf("%s in=%d reference norm is zero", name, in)
			}
			relL2 := math.Sqrt(sqErr / sqRef)
			cos := cosineC(got, want)
			if argmaxF32(got) != argmaxF32(want) {
				t.Fatalf("%s in=%d argmax mismatch: got %d, want %d", name, in, argmaxF32(got), argmaxF32(want))
			}
			if relL2 > 1e-4 {
				t.Fatalf("%s in=%d relative L2 %g > 1e-4", name, in, relL2)
			}
			if cos < 0.995 {
				t.Fatalf("%s in=%d cosine %.8f < 0.995", name, in, cos)
			}
		}
	}
}

func TestVulkanQ4KSwiGLUMatMulAddInPlaceMatchesCPUReference(t *testing.T) {
	v := q4FusedDevice(t)
	inDims := []int{256, 512, 768}
	const out = 16
	rng := rand.New(rand.NewSource(9717))

	for _, in := range inDims {
		nblk := in / q4kSuper
		raw := make([]byte, out*nblk*q4kSuperBlock)
		for b := 0; b < out*nblk; b++ {
			fillCraftedQ4KBlock(raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+200)
		}
		gate, up := make([]float32, in), make([]float32, in)
		dst := make([]float32, out)
		for i := range gate {
			gate[i], up[i] = rng.Float32()*2-1, rng.Float32()*2-1
		}
		for i := range dst {
			dst[i] = rng.Float32()*2 - 1
		}
		hw := NewQ4K(Default(), []int{out, in}, raw)
		dw := v.Upload(hw, Q4_K)
		defer v.Free(dw)
		dgate := v.Upload(NewF32(Default(), []int{in}, gate), F32)
		defer v.Free(dgate)
		dup := v.Upload(NewF32(Default(), []int{in}, up), F32)
		defer v.Free(dup)
		ddst := v.Upload(NewF32(Default(), []int{out}, dst), F32)
		defer v.Free(ddst)
		v.SwiGLUMatMulAddInPlace(ddst, dw, dgate, dup)
		sw := Default().SwiGLU(NewF32(Default(), []int{in}, gate), NewF32(Default(), []int{in}, up))
		proj := Default().Read(Default().MatMul(hw, sw))
		want := append([]float32(nil), dst...)
		for i := range want {
			want[i] += proj[i]
		}
		got := v.Read(ddst)
		var sqErr, sqRef float64
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("in=%d non-finite value at %d: %v", in, i, got[i])
			}
			diff := float64(got[i] - want[i])
			sqErr += diff * diff
			sqRef += float64(want[i]) * float64(want[i])
		}
		if sqRef <= 0 {
			t.Fatalf("in=%d reference norm is zero", in)
		}
		relL2 := math.Sqrt(sqErr / sqRef)
		cos := cosineC(got, want)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("in=%d argmax mismatch: got %d, want %d", in, argmaxF32(got), argmaxF32(want))
		}
		if relL2 > 1e-4 {
			t.Fatalf("in=%d relative L2 %g > 1e-4", in, relL2)
		}
		if cos < 0.995 {
			t.Fatalf("in=%d cosine %.8f < 0.995", in, cos)
		}
	}
}

// fillCraftedQ4KBlock fills a 144-byte super-block explicitly exercising all 8 scale/min groups,
// both nibbles (even/odd groups), and varied lane patterns.
func fillCraftedQ4KBlock(blk []byte, blockIdx int) {
	// Bytes 0..1: d (f16) ~ 0.05
	blk[0] = 0x00
	blk[1] = 0x31
	// Bytes 2..3: dm (f16) ~ 0.025
	blk[2] = 0x00
	blk[3] = 0x29

	// Bytes 4..15: 12 bytes of packed scales and mins for groups 0..7.
	for j := 0; j < 4; j++ {
		scVal := byte((j*7 + 11 + blockIdx) & 0x3F)
		mnHi := byte(((j + 1) & 0x03) << 6)
		blk[4+j] = scVal | mnHi

		mnVal := byte((j*5 + 7 + blockIdx) & 0x3F)
		scHi := byte(((j + 2) & 0x03) << 6)
		blk[4+j+4] = mnVal | scHi

		scLo := byte((j*3 + 1 + blockIdx) & 0x0F)
		mnLo := byte(((j*4 + 2 + blockIdx) & 0x0F) << 4)
		blk[4+j+8] = scLo | mnLo
	}

	// Bytes 16..143: 128 bytes of quantized nibbles (4 pairs of groups).
	for p := 0; p < 4; p++ {
		for lane := 0; lane < 32; lane++ {
			loNibble := byte((lane*3 + p*5 + blockIdx*7) & 0x0F)
			hiNibble := byte(((lane*7 + p*11 + blockIdx*3) & 0x0F) << 4)
			blk[16+p*32+lane] = loNibble | hiNibble
		}
	}
}

func TestVulkanQ4KWave32CraftedFixtures(t *testing.T) {
	v := q4Device(t)
	inDims := []int{256, 512, 768}
	outDims := []int{1, 2, 3, 5, 8, 13}

	for _, in := range inDims {
		for _, out := range outDims {
			t.Run(t.Name(), func(t *testing.T) {
				nblk := in / q4kSuper
				raw := make([]byte, out*nblk*q4kSuperBlock)
				for b := 0; b < out*nblk; b++ {
					fillCraftedQ4KBlock(raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b)
				}
				x := make([]float32, in)
				for i := range x {
					x[i] = float32(math.Sin(float64(i)*0.07+0.3)) * 0.5
				}

				hw := NewQ4K(Default(), []int{out, in}, raw)
				hx := NewF32(Default(), []int{in}, x)
				dw := v.Upload(hw, Q4_K)
				defer v.Free(dw)
				dx := v.Upload(hx, F32)
				defer v.Free(dx)

				// Candidate P=1 execution
				t.Setenv("FAK_VULKAN_Q4K_ARM", "candidate")
				dy := v.MatMul(dw, dx)
				defer v.Free(dy)
				got := v.Read(dy)
				want := Default().Read(Default().MatMul(hw, hx))

				if len(got) != out {
					t.Fatalf("in=%d out=%d len(got)=%d want %d", in, out, len(got), out)
				}
				var sqErr, sqRef float64
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						t.Fatalf("in=%d out=%d index %d non-finite: %v", in, out, i, got[i])
					}
					diff := float64(got[i] - want[i])
					sqErr += diff * diff
					sqRef += float64(want[i]) * float64(want[i])
				}
				if sqRef <= 0 {
					t.Fatalf("in=%d out=%d reference norm is zero", in, out)
				}
				relL2 := math.Sqrt(sqErr / sqRef)
				cos := cosineC(got, want)
				gotArgmax, wantArgmax := argmaxF32(got), argmaxF32(want)
				if gotArgmax != wantArgmax {
					t.Fatalf("in=%d out=%d argmax mismatch: got %d, want %d", in, out, gotArgmax, wantArgmax)
				}
				if relL2 > 1e-4 {
					t.Fatalf("in=%d out=%d relative L2 %g > 1e-4", in, out, relL2)
				}
				if cos < 0.99999 {
					t.Fatalf("in=%d out=%d cosine %g < 0.99999", in, out, cos)
				}
			})
		}
	}
}

func TestVulkanQ4KWave32CapabilityAdmission(t *testing.T) {
	v := q4Device(t)
	const out, in = 13, 512
	nblk := in / q4kSuper
	raw := make([]byte, out*nblk*q4kSuperBlock)
	for b := 0; b < out*nblk; b++ {
		fillCraftedQ4KBlock(raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.05+0.1)) * 0.4
	}

	hw := NewQ4K(Default(), []int{out, in}, raw)
	hx := NewF32(Default(), []int{in}, x)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	dx := v.Upload(hx, F32)
	defer v.Free(dx)
	want := Default().Read(Default().MatMul(hw, hx))

	// Arm 1: Candidate arm explicitly selected
	t.Run("CandidateArm", func(t *testing.T) {
		t.Setenv("FAK_VULKAN_Q4K_ARM", "candidate")
		dy := v.MatMul(dw, dx)
		defer v.Free(dy)
		got := v.Read(dy)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("candidate argmax mismatch: got %d want %d", argmaxF32(got), argmaxF32(want))
		}
		if cos := cosineC(got, want); cos < 0.99999 {
			t.Fatalf("candidate cosine %g < 0.99999", cos)
		}
	})

	// Arm 2: Scalar arm explicitly selected
	t.Run("ScalarArm", func(t *testing.T) {
		t.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
		dy := v.MatMul(dw, dx)
		defer v.Free(dy)
		got := v.Read(dy)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("scalar argmax mismatch: got %d want %d", argmaxF32(got), argmaxF32(want))
		}
		if cos := cosineC(got, want); cos < 0.995 {
			t.Fatalf("scalar cosine %g < 0.995", cos)
		}
	})

	// Multi-token fallback (P=3) selects scalar pipeline
	t.Run("MultiTokenFallbackP3", func(t *testing.T) {
		const P = 3
		xMulti := make([]float32, P*in)
		for i := range xMulti {
			xMulti[i] = float32(math.Sin(float64(i)*0.03)) * 0.3
		}
		hxMulti := NewF32(Default(), []int{P, in}, xMulti)
		dxMulti := v.Upload(hxMulti, F32)
		defer v.Free(dxMulti)
		wantMulti := Default().Read(Default().BatchedMatMul(hw, hxMulti, P))

		dyMulti := v.BatchedMatMul(dw, dxMulti, P)
		defer v.Free(dyMulti)
		gotMulti := v.Read(dyMulti)
		for tok := 0; tok < P; tok++ {
			gRow := gotMulti[tok*out : (tok+1)*out]
			wRow := wantMulti[tok*out : (tok+1)*out]
			if argmaxF32(gRow) != argmaxF32(wRow) {
				t.Fatalf("token %d argmax mismatch", tok)
			}
			if cos := cosineC(gRow, wRow); cos < 0.995 {
				t.Fatalf("token %d cosine %g < 0.995", tok, cos)
			}
		}
	})
}
