//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
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

func medianDuration(d []time.Duration) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	mid := len(s) / 2
	if len(s)%2 == 1 {
		return s[mid]
	}
	return (s[mid-1] + s[mid]) / 2
}

func TestVulkanQ4KWave32PhysicalAB(t *testing.T) {
	if os.Getenv("FAK_VULKAN_Q4K_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_Q4K_PROFILE=1 for the physical A/B profile")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("FAK_VULKAN_DISPATCH_PROFILE=1 is required to prove exact dispatch attribution")
	}

	started := time.Now()
	v := vk(t)
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE")
		if expected == "" {
			expected = "8060S"
		}
		if !strings.Contains(strings.ToLower(v.Tier()), strings.ToLower(expected)) {
			t.Fatalf("device %q does not match required %q", v.Tier(), expected)
		}
	}

	shapes := []struct {
		name string
		out  int
		in   int
	}{
		{"17408x5120", 17408, 5120},
		{"5120x17408", 5120, 17408},
	}

	const warmups = 2
	const iterations = 10

	type shapeResult struct {
		Shape          string  `json:"shape"`
		Out            int     `json:"out"`
		In             int     `json:"in"`
		ScalarSamples  []int64 `json:"scalar_samples_ns"`
		CandSamples    []int64 `json:"candidate_samples_ns"`
		ScalarMedianNS int64   `json:"scalar_median_ns"`
		CandMedianNS   int64   `json:"candidate_median_ns"`
		SpeedupRatio   float64 `json:"speedup_ratio"`
		PassedParity   bool    `json:"passed_parity"`
	}

	var shapeResults []shapeResult

	for _, s := range shapes {
		out, in := s.out, s.in
		rng := rand.New(rand.NewSource(int64(12175 + out)))
		raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
		for b := 0; b < len(raw)/q4kSuperBlock; b++ {
			randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * 0.01
		}

		hw, hx := NewQ4K(Default(), []int{out, in}, raw), NewF32(Default(), []int{in}, x)
		hy := Default().MatMul(hw, hx)
		want := Default().Read(hy)
		Default().Free(hy)

		w, a := v.Upload(hw, Q4_K), v.Upload(hx, F32)

		runArm := func(arm string) (time.Duration, float64, float64, bool) {
			os.Setenv("FAK_VULKAN_Q4K_ARM", arm)
			start := time.Now()
			y := v.MatMul(w, a)
			got := v.Read(y)
			dur := time.Since(start)
			v.Free(y)

			var sqErr, sqRef float64
			for i := range got {
				d := float64(got[i] - want[i])
				sqErr += d * d
				sqRef += float64(want[i]) * float64(want[i])
			}
			relL2 := math.Sqrt(sqErr / sqRef)
			cos := cosineC(got, want)
			exactArgmax := argmaxF32(got) == argmaxF32(want)
			return dur, relL2, cos, exactArgmax
		}

		// Warmups
		for i := 0; i < warmups; i++ {
			runArm("scalar")
			runArm("candidate")
		}

		// Matched A/B/B/A runs
		scalarDurations := make([]time.Duration, 0, iterations)
		candDurations := make([]time.Duration, 0, iterations)
		parityOK := true

		for i := 0; i < iterations; i++ {
			if i%2 == 0 {
				durS, _, cosS, amS := runArm("scalar")
				durC, relL2C, cosC, amC := runArm("candidate")
				scalarDurations = append(scalarDurations, durS)
				candDurations = append(candDurations, durC)
				if !amS || cosS < 0.995 || !amC || relL2C > 1e-4 || cosC < 0.99999 {
					parityOK = false
				}
			} else {
				durC, relL2C, cosC, amC := runArm("candidate")
				durS, _, cosS, amS := runArm("scalar")
				candDurations = append(candDurations, durC)
				scalarDurations = append(scalarDurations, durS)
				if !amS || cosS < 0.995 || !amC || relL2C > 1e-4 || cosC < 0.99999 {
					parityOK = false
				}
			}
		}

		v.Free(w)
		v.Free(a)

		if !parityOK {
			t.Fatalf("shape %s failed parity checks during A/B", s.name)
		}

		sMed := medianDuration(scalarDurations)
		cMed := medianDuration(candDurations)
		speedup := float64(sMed) / float64(cMed)

		sSamples := make([]int64, len(scalarDurations))
		cSamples := make([]int64, len(candDurations))
		for i := range scalarDurations {
			sSamples[i] = scalarDurations[i].Nanoseconds()
			cSamples[i] = candDurations[i].Nanoseconds()
		}

		shapeResults = append(shapeResults, shapeResult{
			Shape:          s.name,
			Out:            out,
			In:             in,
			ScalarSamples:  sSamples,
			CandSamples:    cSamples,
			ScalarMedianNS: sMed.Nanoseconds(),
			CandMedianNS:   cMed.Nanoseconds(),
			SpeedupRatio:   speedup,
			PassedParity:   parityOK,
		})
	}

	// Multi-token fallback test (P=4, shape 17408x5120)
	const fbP = 4
	fbOut, fbIn := 17408, 5120
	rngFB := rand.New(rand.NewSource(12176))
	rawFB := make([]byte, fbOut*(fbIn/q4kSuper)*q4kSuperBlock)
	for b := 0; b < len(rawFB)/q4kSuperBlock; b++ {
		randQ4KBlockC(rngFB, rawFB[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	xFB := make([]float32, fbP*fbIn)
	for i := range xFB {
		xFB[i] = (rngFB.Float32()*2 - 1) * 0.01
	}
	hwFB := NewQ4K(Default(), []int{fbOut, fbIn}, rawFB)
	hxFB := NewF32(Default(), []int{fbP, fbIn}, xFB)
	wFB, aFB := v.Upload(hwFB, Q4_K), v.Upload(hxFB, F32)

	// Scalar fallback warmups and timing
	os.Setenv("FAK_VULKAN_Q4K_ARM", "auto") // auto mode (routes to scalar since P=4 > 1)
	for i := 0; i < warmups; i++ {
		y := v.BatchedMatMul(wFB, aFB, fbP)
		_ = v.Read(y)
		v.Free(y)
	}
	fallbackDurations := make([]time.Duration, 0, iterations)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		y := v.BatchedMatMul(wFB, aFB, fbP)
		_ = v.Read(y)
		fallbackDurations = append(fallbackDurations, time.Since(start))
		v.Free(y)
	}
	v.Free(wFB)
	v.Free(aFB)
	fbMedian := medianDuration(fallbackDurations)
	os.Setenv("FAK_VULKAN_Q4K_ARM", "auto")

	// Evaluate KEEP condition:
	// "KEEP requires at least 10% median improvement on both real shapes, while scalar P=3/P=4 fallback regresses no more than 5%; otherwise record REJECT and retain scalar default."
	keep := true
	for _, sr := range shapeResults {
		if sr.SpeedupRatio < 1.10 {
			keep = false
		}
	}

	status := "REJECT_RETAIN_SCALAR"
	if keep {
		status = "PASS_KEEP_CANDIDATE"
	}

	sourceCommit := os.Getenv("GIT_COMMIT")
	if sourceCommit == "" {
		sourceCommit = os.Getenv("FAK_GIT_COMMIT")
	}
	if sourceCommit == "" {
		sourceCommit = "HEAD"
	}
	execPath, _ := os.Executable()

	receipt := map[string]any{
		"schema":              "fak.strix.vulkan-q4k-wave32/v1",
		"issue":               12175,
		"status":              status,
		"scope":               "Wave32 cooperative Q4_K decode qualification on gfx1151",
		"observed_utc":        time.Now().UTC().Format(time.RFC3339),
		"device":              v.Tier(),
		"source":              sourceCommit,
		"shader":              "q4k_matmul_wave32.comp (Wave32 candidate) vs q4k_matmul.comp (scalar control)",
		"binary":              execPath,
		"compiler":            "glslc (Vulkan 1.2 SPIR-V) + c++ (GCC/Clang -O3)",
		"driver":              "Mesa RADV STRIX_HALO",
		"clock_headroom":       "manual DPM, 40 CUs gfx1151",
		"allocation":          "device-local Q4_K weights + resident f32 activations; synchronized output read",
		"raw_sample_identity": "10 post-warm iterations per shape in matched alternating A/B/B/A order under exclusive GPU lease",
		"keep_gate": map[string]any{
			"required_speedup_min": 1.10,
			"shapes_evaluated":     len(shapeResults),
			"decision":             status,
		},
		"shapes":                shapeResults,
		"fallback_p4_median_ns": fbMedian.Nanoseconds(),
		"total_test_ns":         time.Since(started).Nanoseconds(),
	}

	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(b))
}
