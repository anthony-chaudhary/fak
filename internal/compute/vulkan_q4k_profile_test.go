//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// TestVulkanQ4KRealShapeProfile uses the 27B FFN geometry with synthetic Q4_K
// weights. It measures a resident primitive, not model inference or the mixed
// quantization types of the original artifact.
func TestVulkanQ4KRealShapeProfile(t *testing.T) {
	if os.Getenv("FAK_VULKAN_Q4K_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_Q4K_PROFILE=1 for the bounded physical-device profile")
	}
	started := time.Now()
	v := vk(t)
	const out, in = 17408, 5120
	rng := rand.New(rand.NewSource(11919))
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < len(raw)/q4kSuperBlock; b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * .01
	}
	hw, hx := NewQ4K(Default(), []int{out, in}, raw), NewF32(Default(), []int{in}, x)
	fixtureNS := time.Since(started).Nanoseconds()
	oracleStart := time.Now()
	hy := Default().MatMul(hw, hx)
	want := Default().Read(hy)
	defer Default().Free(hy)
	for i, y := range want {
		if math.IsNaN(float64(y)) || math.IsInf(float64(y), 0) {
			t.Fatalf("non-finite CPU reference at %d", i)
		}
	}
	oracleNS := time.Since(oracleStart).Nanoseconds()
	uploadStart := time.Now()
	w, a := v.Upload(hw, Q4_K), v.Upload(hx, F32)
	defer v.Free(w)
	defer v.Free(a)
	if !v.debugBufferDeviceLocal(w.buf.(*vulkanBuf)) {
		t.Fatal("profile requires device-local Q4_K weight storage")
	}
	wbuf, abuf := w.Buf(), a.Buf()
	uploadNS := time.Since(uploadStart).Nanoseconds()
	var samples []map[string]any
	for step := 0; step < 6; step++ {
		dispatchStart := time.Now()
		y := v.MatMul(w, a)
		got := v.Read(y) // synchronizes the dispatch and includes output transfer
		dispatchNS := time.Since(dispatchStart).Nanoseconds()
		checkStart := time.Now()
		if len(got) != len(want) {
			v.Free(y)
			t.Fatalf("output length %d, want %d", len(got), len(want))
		}
		var squaredError, squaredReference float64
		for i, z := range got {
			if math.IsNaN(float64(z)) || math.IsInf(float64(z), 0) {
				v.Free(y)
				t.Fatalf("step %d non-finite output at %d", step, i)
			}
			delta, reference := float64(z)-float64(want[i]), float64(want[i])
			squaredError += delta * delta
			squaredReference += reference * reference
		}
		if squaredReference <= 0 || math.IsNaN(squaredReference) || math.IsInf(squaredReference, 0) {
			v.Free(y)
			t.Fatal("CPU reference requires a finite nonzero norm")
		}
		cosine := cosineC(got, want)
		// Q4 inputs are identical, so this bounds reduction error rather than
		// quantization loss; unlike cosine it rejects a uniformly scaled result.
		relativeL2 := math.Sqrt(squaredError / squaredReference)
		if math.IsNaN(relativeL2) || math.IsInf(relativeL2, 0) || relativeL2 > 1e-4 || math.IsNaN(cosine) || math.IsInf(cosine, 0) || cosine < .99999 || argmaxF32(got) != argmaxF32(want) {
			v.Free(y)
			t.Fatalf("step %d Q4_K parity relativeL2=%g cosine=%g argmax=%d want=%d", step, relativeL2, cosine, argmaxF32(got), argmaxF32(want))
		}
		if w.Buf() != wbuf || a.Buf() != abuf {
			v.Free(y)
			t.Fatal("resident inputs changed identity")
		}
		validationNS := time.Since(checkStart).Nanoseconds()
		freeStart := time.Now()
		v.Free(y)
		samples = append(samples, map[string]any{"step": step, "warmup": step == 0,
			"dispatch_and_output_read_ns": dispatchNS, "validation_ns": validationNS,
			"output_free_ns": time.Since(freeStart).Nanoseconds(), "relative_l2": relativeL2, "cosine": cosine, "argmax": argmaxF32(want)})
	}
	receipt := map[string]any{
		"schema": "fak.vulkan.q4k-primitive-profile.v1", "engine": "fak-native-compute",
		"scope":  "synthetic Q4_K GEMV at real FFN geometry; no model execution",
		"device": v.Tier(), "shape": []int{out, in}, "batch": 1, "dtype": "Q4_K",
		"fixture_seed": 11919, "weight_payload_bytes": len(raw),
		"resident_input_payload_bytes": len(raw) + in*4, "output_payload_bytes": out * 4,
		"allocation_note":                        "payload excludes driver alignment and runtime overhead; host fixture retained",
		"resident_input_reuses_after_first_call": 5, "fixture_ns": fixtureNS,
		"cpu_q4_reference_ns": oracleNS, "upload_ns": uploadNS, "samples": samples,
		"test_body_ns": time.Since(started).Nanoseconds(),
		"timing_note":  "test body excludes process/backend initialization and deferred resident cleanup; capture whole-process resources separately",
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(b))
}
