//go:build vulkan && (windows || linux) && cgo

package model

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type vulkanGLMKDAProfiler interface {
	VulkanDebugTransferBytes() (uint64, uint64)
	VulkanDebugResetDispatchProfile()
	VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
}

func glmKDAMedian(values []time.Duration) time.Duration {
	copyValues := append([]time.Duration(nil), values...)
	sort.Slice(copyValues, func(i, j int) bool { return copyValues[i] < copyValues[j] })
	return copyValues[len(copyValues)/2]
}

func glmKDAPercentile(values []time.Duration, percentile float64) time.Duration {
	copyValues := append([]time.Duration(nil), values...)
	sort.Slice(copyValues, func(i, j int) bool { return copyValues[i] < copyValues[j] })
	idx := int(math.Ceil(percentile*float64(len(copyValues)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(copyValues) {
		idx = len(copyValues) - 1
	}
	return copyValues[idx]
}

func glmKDACompare(got, want []float32) (maxAbs, cosine float64) {
	var dot, got2, want2 float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		delta := math.Abs(g - w)
		if delta > maxAbs {
			maxAbs = delta
		}
		dot += g * w
		got2 += g * g
		want2 += w * w
	}
	if got2 == 0 && want2 == 0 {
		return maxAbs, 1
	}
	return maxAbs, dot / math.Sqrt(got2*want2)
}

func TestVulkanGLMKDAWave32PhysicalAB(t *testing.T) {
	if os.Getenv("FAK_VULKAN_GLM_KDA_PHYSICAL") != "1" {
		t.Skip("set FAK_VULKAN_GLM_KDA_PHYSICAL=1 for the sanctioned physical A/B")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("FAK_VULKAN_DISPATCH_PROFILE=1 is required to prove exact dispatch attribution")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok {
		t.Fatal("vulkan backend is not registered; zero fallback is required")
	}
	if !strings.Contains(be.Tier(), "8060S") {
		t.Fatalf("backend tier %q is not the sanctioned Radeon 8060S device", be.Tier())
	}
	stepper, ok := be.(compute.VulkanGLMKDAStepper)
	if !ok {
		t.Fatal("vulkan backend lacks VulkanGLMKDAStepper; zero fallback is required")
	}
	if !stepper.VulkanGLMKDAWave32Available() {
		t.Fatal("physical backend cannot enforce a 32-lane compute subgroup; zero fallback is required")
	}
	profiler, ok := be.(vulkanGLMKDAProfiler)
	if !ok {
		t.Fatal("vulkan backend lacks physical transfer/dispatch observability")
	}

	const heads = 64
	const dim = compute.GLMKDAHeadDim
	const warmups = 4
	const iterations = 31
	state0 := make([]float32, heads*dim*dim)
	q := make([]float32, heads*dim)
	k := make([]float32, heads*dim)
	value := make([]float32, heads*dim)
	alpha := make([]float32, heads)
	beta := make([]float32, heads)
	for i := range state0 {
		state0[i] = float32(math.Sin(float64(i)*0.013+0.2)) * 0.002
	}
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.031+0.1)) * 0.02
		k[i] = float32(math.Cos(float64(i)*0.027+0.3)) * 0.02
		value[i] = float32(math.Sin(float64(i)*0.019+0.7)) * 0.03
	}
	for h := 0; h < heads; h++ {
		alpha[h] = 0.985 + float32(h%7)*0.001
		beta[h] = 0.10 + float32(h%5)*0.015
	}

	upload := func(shape []int, values []float32) compute.Tensor {
		return be.Upload(compute.NewF32(compute.Default(), shape, values), compute.F32)
	}
	dq, dk, dv := upload([]int{heads, dim}, q), upload([]int{heads, dim}, k), upload([]int{heads, dim}, value)
	da, db := upload([]int{heads}, alpha), upload([]int{heads}, beta)
	zero := make([]float32, heads*dim)
	parentState := upload([]int{heads, dim, dim}, state0)
	candidateState := upload([]int{heads, dim, dim}, state0)
	parentOut, candidateOut := upload([]int{heads, dim}, zero), upload([]int{heads, dim}, zero)
	warmParentState := upload([]int{heads, dim, dim}, state0)
	warmCandidateState := upload([]int{heads, dim, dim}, state0)
	warmParentOut, warmCandidateOut := upload([]int{heads, dim}, zero), upload([]int{heads, dim}, zero)
	all := []compute.Tensor{dq, dk, dv, da, db, parentState, candidateState, parentOut, candidateOut, warmParentState, warmCandidateState, warmParentOut, warmCandidateOut}
	defer func() {
		for _, tensor := range all {
			be.Free(tensor)
		}
	}()

	for i := 0; i < warmups; i++ {
		if err := stepper.VulkanGLMKDAStep(warmParentState, dq, dk, dv, da, db, warmParentOut, compute.VulkanGLMKDAReread); err != nil {
			t.Fatalf("parent warmup %d: %v", i, err)
		}
		if err := stepper.VulkanGLMKDAStep(warmCandidateState, dq, dk, dv, da, db, warmCandidateOut, compute.VulkanGLMKDAWave32Retain); err != nil {
			t.Fatalf("candidate warmup %d: %v", i, err)
		}
	}

	profiler.VulkanDebugResetDispatchProfile()
	h2dStart, d2hStart := profiler.VulkanDebugTransferBytes()
	parentTimes := make([]time.Duration, 0, iterations)
	candidateTimes := make([]time.Duration, 0, iterations)
	run := func(state, output compute.Tensor, variant compute.VulkanGLMKDAVariant, samples *[]time.Duration) {
		started := time.Now()
		if err := stepper.VulkanGLMKDAStep(state, dq, dk, dv, da, db, output, variant); err != nil {
			t.Fatalf("variant %d dispatch: %v", variant, err)
		}
		*samples = append(*samples, time.Since(started))
	}
	for i := 0; i < iterations; i++ {
		if i%2 == 0 {
			run(parentState, parentOut, compute.VulkanGLMKDAReread, &parentTimes)
			run(candidateState, candidateOut, compute.VulkanGLMKDAWave32Retain, &candidateTimes)
		} else {
			run(candidateState, candidateOut, compute.VulkanGLMKDAWave32Retain, &candidateTimes)
			run(parentState, parentOut, compute.VulkanGLMKDAReread, &parentTimes)
		}
	}
	h2dEnd, d2hEnd := profiler.VulkanDebugTransferBytes()
	profile := profiler.VulkanDebugDispatchProfileSnapshot()
	if h2dEnd != h2dStart || d2hEnd != d2hStart {
		t.Fatalf("timed operation crossed host/device boundary: H2D=%d D2H=%d", h2dEnd-h2dStart, d2hEnd-d2hStart)
	}
	if want := uint64(2 * iterations); profile.ComputeDispatches != want || profile.OtherGDNDispatches != want {
		t.Fatalf("dispatch attribution=%+v, want exactly %d compute/GDN dispatches", profile, want)
	}

	parentStateHost, candidateStateHost := be.Read(parentState), be.Read(candidateState)
	parentOutHost, candidateOutHost := be.Read(parentOut), be.Read(candidateOut)
	oracleState := append([]float32(nil), state0...)
	var oracleOut []float32
	for iteration := 0; iteration < iterations; iteration++ {
		oracleOut = make([]float32, heads*dim)
		for h := 0; h < heads; h++ {
			begin, end := h*dim, (h+1)*dim
			stateBegin, stateEnd := h*dim*dim, (h+1)*dim*dim
			got := StepGLM5NextKDAHead(oracleState[stateBegin:stateEnd], dim, dim, q[begin:end], k[begin:end], value[begin:end], alpha[h], beta[h])
			copy(oracleOut[begin:end], got)
		}
	}
	parentStateAbs, parentStateCos := glmKDACompare(parentStateHost, oracleState)
	candidateStateAbs, candidateStateCos := glmKDACompare(candidateStateHost, oracleState)
	parentOutAbs, parentOutCos := glmKDACompare(parentOutHost, oracleOut)
	candidateOutAbs, candidateOutCos := glmKDACompare(candidateOutHost, oracleOut)
	for name, result := range map[string]struct{ abs, cos float64 }{
		"parent_state": {parentStateAbs, parentStateCos}, "candidate_state": {candidateStateAbs, candidateStateCos},
		"parent_output": {parentOutAbs, parentOutCos}, "candidate_output": {candidateOutAbs, candidateOutCos},
	} {
		if result.abs > 2e-4 || result.cos < 0.999999 {
			t.Errorf("%s max_abs=%g cosine=%g, want <=2e-4 and >=0.999999", name, result.abs, result.cos)
		}
	}

	parentMedian, candidateMedian := glmKDAMedian(parentTimes), glmKDAMedian(candidateTimes)
	speedup := float64(parentMedian) / float64(candidateMedian)
	summary := struct {
		Backend, Tier                                string
		Heads, HeadDim, Iterations                   int
		ParentMedianNS, ParentP95NS                  int64
		CandidateMedianNS, CandidateP95NS            int64
		Speedup                                      float64
		ParentStateMaxAbs, ParentStateCosine         float64
		CandidateStateMaxAbs, CandidateStateCosine   float64
		ParentOutputMaxAbs, ParentOutputCosine       float64
		CandidateOutputMaxAbs, CandidateOutputCosine float64
		H2DBytes, D2HBytes, Dispatches               uint64
	}{
		be.Name(), be.Tier(), heads, dim, iterations,
		parentMedian.Nanoseconds(), glmKDAPercentile(parentTimes, .95).Nanoseconds(),
		candidateMedian.Nanoseconds(), glmKDAPercentile(candidateTimes, .95).Nanoseconds(), speedup,
		parentStateAbs, parentStateCos, candidateStateAbs, candidateStateCos,
		parentOutAbs, parentOutCos, candidateOutAbs, candidateOutCos,
		h2dEnd - h2dStart, d2hEnd - d2hStart, profile.ComputeDispatches,
	}
	raw, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("marshal physical summary: %v", err)
	}
	t.Logf("GLM_KDA_PHYSICAL_AB %s", raw)
	if speedup <= 1.01 {
		t.Fatalf("Wave32 retention speedup=%0.4fx, must exceed the matched reread parent by >1%% to retain candidate", speedup)
	}
}
