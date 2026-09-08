package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
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
	const out0, out1, in = 12, 16, 768
	rng := rand.New(rand.NewSource(9716))
	newWeight := func(out int) Tensor {
		raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
		for b := 0; b < out*(in/q4kSuper); b++ {
			randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
		}
		return NewQ4K(Default(), []int{out, in}, raw)
	}
	hw0, hw1 := newWeight(out0), newWeight(out1)
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
		if c := cosineC(pair[0], pair[1]); c < 0.995 {
			t.Fatalf("%s cosine %.8f < 0.995", name, c)
		}
	}
}
func TestVulkanQ4KSwiGLUMatMulAddInPlaceMatchesCPUReference(t *testing.T) {
	v := q4FusedDevice(t)
	const out, in = 16, 768
	rng := rand.New(rand.NewSource(9717))
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
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
	if c := cosineC(v.Read(ddst), want); c < 0.995 {
		t.Fatalf("cosine %.8f < 0.995", c)
	}
}
