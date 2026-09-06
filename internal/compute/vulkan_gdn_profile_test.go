//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"
)

// TestVulkanGDNRealShapeProfile measures the recurrence shape from the 27B
// artifact separately from its much larger projection matrices. It is opt-in
// so ordinary correctness runs do not silently become hardware profiles.
func TestVulkanGDNRealShapeProfile(t *testing.T) {
	if os.Getenv("FAK_VULKAN_GDN_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_GDN_PROFILE=1 for the bounded physical-device profile")
	}
	started := time.Now()
	v := vk(t)
	const tokens, nK, nV, kHd, vHd, kernel = 1, 16, 48, 128, 128, 4
	const convDim, valueDim = 2*nK*kHd + nV*vHd, nV * vHd
	values := func(n int, scale float32) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((i%17)-8) * scale
		}
		return x
	}
	mixed, z := values(convDim, .025), values(valueDim, .04)
	beta, alpha := values(nV, .02), values(nV, .03)
	convW := values(convDim*kernel, .01)
	aLog, dtBias, norm := make([]float32, nV), values(nV, .01), make([]float32, vHd)
	for i := range aLog {
		aLog[i] = -1
	}
	for i := range norm {
		norm[i] = .9 + float32(i%5)*.05
	}
	hostC, hostR := make([]float32, (kernel-1)*convDim), make([]float32, nV*kHd*vHd)
	var resident []Tensor
	var payloadBytes int64
	cleanup := func() {
		for _, tensor := range resident {
			v.Free(tensor)
		}
		resident = nil
	}
	defer cleanup()
	upload := func(shape []int, x []float32, class MemoryClass) Tensor {
		tensor := v.UploadClass(NewF32(Default(), shape, x), F32, class, "GDN profile")
		resident = append(resident, tensor)
		payloadBytes += int64(len(x)) * 4
		return tensor
	}
	m := upload([]int{tokens, convDim}, mixed, MemoryActivation)
	zt := upload([]int{tokens, valueDim}, z, MemoryActivation)
	bt, at := upload([]int{tokens, nV}, beta, MemoryActivation), upload([]int{tokens, nV}, alpha, MemoryActivation)
	cw := upload([]int{convDim, kernel}, convW, MemoryWeights)
	al, dt := upload([]int{nV}, aLog, MemoryWeights), upload([]int{nV}, dtBias, MemoryWeights)
	nw := upload([]int{vHd}, norm, MemoryWeights)
	cs := upload([]int{kernel - 1, convDim}, hostC, MemoryKVCache)
	rs := upload([]int{nV, kHd, vHd}, hostR, MemoryKVCache)
	oldC, oldR := cs.Buf(), rs.Buf()
	setupNS := time.Since(started).Nanoseconds()
	check := func(name string, got, want []float32) {
		if len(got) != len(want) {
			t.Fatalf("%s length %d, want %d", name, len(got), len(want))
		}
		for i := range got {
			a, b := float64(got[i]), float64(want[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) || math.Abs(a-b) > 2e-4 {
				t.Fatalf("%s[%d]=%g want finite %g within 0.0002", name, i, a, b)
			}
		}
	}
	var samples []map[string]any
	for step := 0; step < 6; step++ {
		stepStart := time.Now()
		out, err := v.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		got := v.Read(out)
		dispatchNS := time.Since(stepStart).Nanoseconds()
		readStart := time.Now()
		gotC, gotR := v.Read(cs), v.Read(rs)
		readNS := time.Since(readStart).Nanoseconds()
		v.Free(out)
		oracleStart := time.Now()
		want, nextC, nextR := qwen35GDNPreprojectedOracle(mixed, z, beta, alpha, convW, aLog, dtBias, norm, hostC, hostR, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		oracleNS := time.Since(oracleStart).Nanoseconds()
		checkStart := time.Now()
		check("output", got, want)
		check("convolution state", gotC, nextC)
		check("recurrent state", gotR, nextR)
		if cs.Buf() != oldC || rs.Buf() != oldR {
			t.Fatal("persistent state identity changed")
		}
		hostC, hostR = nextC, nextR
		samples = append(samples, map[string]any{"warmup": step == 0, "dispatch_and_output_read_ns": dispatchNS, "state_readback_ns": readNS, "cpu_oracle_ns": oracleNS, "validation_ns": time.Since(checkStart).Nanoseconds(), "whole_step_ns": time.Since(stepStart).Nanoseconds()})
	}
	cleanupStart := time.Now()
	cleanup()
	receipt := map[string]any{
		"schema": "fak.vulkan.gdn-profile/1", "device": v.Tier(), "dtype": "f32", "model_loaded": false,
		"shape": []int{tokens, nK, nV, kHd, vHd, kernel}, "shape_order": "tokens,key_heads,value_heads,key_dim,value_dim,conv_kernel",
		"resident_tensor_payload_bytes": payloadBytes, "payload_excludes_driver_alignment_and_scratch": true,
		"warmups": 1, "measured_samples": 5, "resident_reuses_after_first_step": 5,
		"setup_ns": setupNS, "cleanup_ns": time.Since(cleanupStart).Nanoseconds(), "total_ns": time.Since(started).Nanoseconds(), "samples": samples,
		"quality": "finite output and evolved states match CPU oracle within 0.0002; persistent state identity preserved",
		"scope":   "preprojected recurrence only; synchronized host timings include transfers; no full-model throughput claim",
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(encoded))
}
