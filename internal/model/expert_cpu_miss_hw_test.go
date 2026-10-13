//go:build vulkan && (linux || windows) && cgo

// fak-test:runtime slow est=30s lane=optin

package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// assertUMADevice3343 refuses to label an uncontended/contended measurement as a UMA
// hardware witness unless the registered device is the integrated SoC GPU sharing host
// memory. A discrete-VRAM GPU would not exhibit UMA sharing, so accepting it would make the
// [HW-WITNESSED] label a lie. It fails closed: an unrecognized device tier is a hard failure,
// not a silent pass.
func assertUMADevice3343(t *testing.T, tier string) {
	t.Helper()
	low := strings.ToLower(tier)
	for _, marker := range []string{"8060s", "strix_halo", "strix-halo", "integrated"} {
		if strings.Contains(low, marker) {
			return
		}
	}
	t.Fatalf("UMA witness requires the integrated Strix Halo SoC GPU; got device tier %q", tier)
}

// Ticket #3343 DoD item 1: PHYSICAL measurement of UMA bandwidth sharing on the Strix
// Halo APU. The host CPU streaming copy and the GPU H2D DMA both draw from the same
// LPDDR5X, so each engine is slower when both run concurrently. This witness is gated
// behind FAK_UMA_HW_WITNESS=1 and the vulkan build tag because it needs real silicon;
// it must never run in a default lane. The receipt it emits is labelled
// ExpertCacheMeasurementHardware ([HW-WITNESSED]).
const (
	umaWorkBytes      = 64 << 20 // ~64 MiB working set: large enough to swamp caches, small enough to bound runtime
	umaBytesPerElem   = 4        // f32
	umaSoloReps       = 20       // copies/uploads per solo trial
	umaConcurrentReps = 10       // copies/uploads per channel per concurrent trial
	umaTrials         = 3        // independent trials; fastest solo wins, all concurrent pairs emitted
)

func TestUMA_3343_BandwidthSharingHardware(t *testing.T) {
	if os.Getenv("FAK_UMA_HW_WITNESS") != "1" {
		t.Skip("set FAK_UMA_HW_WITNESS=1 on a UMA appliance to run the physical bandwidth witness")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered (FAK_VULKAN_REQUIRE_DEVICE=1)")
		}
		t.Skip("vulkan backend not registered; UMA bandwidth witness needs the device upload path")
	}
	assertUMADevice3343(t, be.Tier())
	t.Logf("UMA witness backend: name=%s tier=%s", be.Name(), be.Tier())

	n := umaWorkBytes / umaBytesPerElem
	src := make([]float32, n)
	dst := make([]float32, n)
	for i := range src {
		src[i] = float32(i%97) * 0.5
	}
	host := compute.NewF32(compute.Default(), []int{n}, src)

	// streamingCopy moves 2*n*4 bytes (read src + write dst); count bytes moved once per copy.
	streamingCopy := func(reps int) (int64, int64) {
		started := time.Now()
		for i := 0; i < reps; i++ {
			copy(dst, src)
		}
		return int64(n) * umaBytesPerElem * int64(reps), time.Since(started).Nanoseconds()
	}
	// uploadLoop moves n*4 bytes per H2D DMA; the Read on the final tensor is the host
	// fence that guarantees the DMA actually landed before the timer stops.
	uploadLoop := func(reps int) (int64, int64) {
		started := time.Now()
		var last compute.Tensor
		for i := 0; i < reps; i++ {
			up := be.Upload(host, compute.F32)
			if i == reps-1 {
				last = up
			} else {
				be.Free(up)
			}
		}
		be.Read(last)
		be.Free(last)
		return int64(n) * umaBytesPerElem * int64(reps), time.Since(started).Nanoseconds()
	}

	var samples []UMABandwidthSharingSample

	// SOLO: one engine at a time, fastest trial is the uncontended reference.
	bestCPU := UMABandwidthSharingSample{Label: "solo-cpu", Concurrent: false}
	for trial := 0; trial < umaTrials; trial++ {
		b, ns := streamingCopy(umaSoloReps)
		if bestCPU.CPUBytes == 0 || float64(b)/float64(ns) > float64(bestCPU.CPUBytes)/float64(bestCPU.CPUNanos) {
			bestCPU.CPUBytes, bestCPU.CPUNanos = b, ns
		}
	}
	bestGPU := UMABandwidthSharingSample{Label: "solo-gpu", Concurrent: false}
	for trial := 0; trial < umaTrials; trial++ {
		b, ns := uploadLoop(umaSoloReps)
		if bestGPU.GPUBytes == 0 || float64(b)/float64(ns) > float64(bestGPU.GPUBytes)/float64(bestGPU.GPUNanos) {
			bestGPU.GPUBytes, bestGPU.GPUNanos = b, ns
		}
	}
	samples = append(samples, bestCPU, bestGPU)

	// CONCURRENT: CPU streaming copy and GPU H2D DMA in flight together, each channel
	// timing only its own span. Both contend for the same LPDDR5X; that is the claim.
	for pair := 0; pair < umaTrials; pair++ {
		start := make(chan struct{})
		var wg sync.WaitGroup
		var cpuBytes, cpuNanos, gpuBytes, gpuNanos int64
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			b, ns := streamingCopy(umaConcurrentReps)
			cpuBytes, cpuNanos = b, ns
		}()
		go func() {
			defer wg.Done()
			<-start
			b, ns := uploadLoop(umaConcurrentReps)
			gpuBytes, gpuNanos = b, ns
		}()
		close(start)
		wg.Wait()
		samples = append(samples, UMABandwidthSharingSample{
			Label:      "concurrent",
			CPUBytes:   cpuBytes,
			CPUNanos:   cpuNanos,
			GPUBytes:   gpuBytes,
			GPUNanos:   gpuNanos,
			Concurrent: true,
		})
	}

	receipt, err := MeasureUMABandwidthSharing(samples)
	if err != nil {
		t.Fatalf("MeasureUMABandwidthSharing: %v", err)
	}
	receipt.Measurement = ExpertCacheMeasurementHardware

	// HARD: well-formed, finite, and both engines actually moved bytes. A placebo backend
	// that moves nothing must fail here rather than emit a zero-rate "hardware" receipt.
	if receipt.Schema != "fak.uma-bandwidth-sharing/v1" {
		t.Errorf("receipt.Schema = %q, want fak.uma-bandwidth-sharing/v1", receipt.Schema)
	}
	if receipt.Samples != len(samples) {
		t.Errorf("receipt.Samples = %d, want %d", receipt.Samples, len(samples))
	}
	if !(receipt.SoloCPUGbps > 0) {
		t.Errorf("receipt.SoloCPUGbps = %v, want > 0 (no CPU bytes moved?)", receipt.SoloCPUGbps)
	}
	if !(receipt.SoloGPUGbps > 0) {
		t.Errorf("receipt.SoloGPUGbps = %v, want > 0 (no GPU DMA bytes moved?)", receipt.SoloGPUGbps)
	}
	for name, v := range map[string]float64{
		"solo_cpu_gbps":         receipt.SoloCPUGbps,
		"solo_gpu_gbps":         receipt.SoloGPUGbps,
		"concurrent_cpu_gbps":   receipt.ConcurrentCPUGbps,
		"concurrent_gpu_gbps":   receipt.ConcurrentGPUGbps,
		"cpu_contention_ratio":  receipt.CPUContentionRatio,
		"gpu_contention_ratio":  receipt.GPUContentionRatio,
		"shared_bandwidth_gbps": receipt.SharedBandwidthGbps,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			t.Errorf("receipt.%s = %v, want finite and >= 0", name, v)
		}
	}

	// ADVISORY: the physical UMA-sharing shape. Both ratios below 1.0 means both engines
	// were slowed by contention. Measurement noise can push a ratio >= 1; that is
	// reported, not fatal -- the hard assertion above already guards against a no-op.
	if receipt.CPUContentionRatio <= 0 || receipt.CPUContentionRatio >= 1.0 {
		t.Logf("ADVISORY: CPU contention ratio = %.3f not in (0,1); UMA sharing not observed on this run (noise or non-shared path)", receipt.CPUContentionRatio)
	}
	if receipt.GPUContentionRatio <= 0 || receipt.GPUContentionRatio >= 1.0 {
		t.Logf("ADVISORY: GPU contention ratio = %.3f not in (0,1); UMA sharing not observed on this run (noise or non-shared path)", receipt.GPUContentionRatio)
	}

	t.Logf("UMA bandwidth witness: solo CPU=%.2f GB/s solo GPU=%.2f GB/s concurrent CPU=%.2f GB/s concurrent GPU=%.2f GB/s shared=%.2f GB/s cpu_ratio=%.3f gpu_ratio=%.3f samples=%d",
		receipt.SoloCPUGbps, receipt.SoloGPUGbps, receipt.ConcurrentCPUGbps, receipt.ConcurrentGPUGbps,
		receipt.SharedBandwidthGbps, receipt.CPUContentionRatio, receipt.GPUContentionRatio, receipt.Samples)

	pretty, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal indented receipt: %v", err)
	}
	t.Logf("UMA_HW_RECEIPT_INDENTED:\n%s", pretty)
	compact, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("marshal compact receipt: %v", err)
	}
	t.Logf("UMA_HW_RECEIPT_JSON: %s", compact)
}

// TestUMA_3343_MissPathMicrobenchHardware measures the physical cost of the miss path for
// ONE V4.1 Flash Q2_K expert slab on the Strix Halo UMA APU. Both arms move the same expert
// bytes: the CPU arm streams the slab through host memory (the read the host GEMV performs
// on a cache miss) and the GPU arm H2D-DMAs the slab into device memory (what the device miss
// path pays). Because CPU and GPU draw from the same LPDDR5X, the two per-expert times are
// the inputs to the miss solver's serial-vs-overlap trade: K=94 experts/token serialized on
// the GPU is the GPU-only lower bound, and min(cpu,gpu) per expert is the OVERLAPPED lower
// bound that applies only IF the two channels were truly independent. This witness is gated
// behind FAK_UMA_HW_WITNESS=1 and the vulkan build tag (real silicon only), and its receipt
// is labelled ExpertCacheMeasurementHardware ([HW-WITNESSED]).
func TestUMA_3343_MissPathMicrobenchHardware(t *testing.T) {
	if os.Getenv("FAK_UMA_HW_WITNESS") != "1" {
		t.Skip("set FAK_UMA_HW_WITNESS=1 on a UMA appliance to run the miss-path microbench")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered (FAK_VULKAN_REQUIRE_DEVICE=1)")
		}
		t.Skip("vulkan backend not registered; miss-path microbench needs the device upload path")
	}
	assertUMADevice3343(t, be.Tier())
	t.Logf("miss-path microbench backend: name=%s tier=%s", be.Name(), be.Tier())

	// V4.1 Flash Q2_K expert geometry. gate/up are [2304,5120] at 84/256 bytes/weight; the
	// down projection is [5120,2304] at 110/256 bytes/weight. Compute slabBytes from the
	// geometry -- never hard-code the byte count.
	gateUpBytes := int64(2304) * int64(5120) * int64(84) / int64(256)
	downBytes := int64(5120) * int64(2304) * int64(110) / int64(256)
	slabBytes := gateUpBytes + gateUpBytes + downBytes
	t.Logf("MISSPATH slab: gate/up=%d B x2 down=%d B slab_bytes=%d", gateUpBytes, downBytes, slabBytes)

	const (
		missReps       = 10      // attempts per channel; fastest wins (the uncontended reference)
		maxElements    = 4 << 20 // 4,194,304 f32 = 16 MiB working-set cap
		missesPerToken = 94      // routed top-k experts per token at V4.1 Flash decode
	)

	// n f32 elements cover the slab. If slabBytes/4 exceeds the 16 MiB cap, clamp n to the
	// cap and SCALE the measured nanos back up by slabBytes/(n*4) -- the timed arm then moved
	// only the capped subset and the scaled value estimates the full-slab cost. The cap exists
	// to bound wall time and keep the working set in DRAM rather than spilling; it does not
	// trigger for the V4.1 slab above (n = 3,202,813 < 4,194,304).
	n := int(slabBytes / 4)
	scale := 1.0
	if n > maxElements {
		n = maxElements
		scale = float64(slabBytes) / float64(int64(n)*4)
		t.Logf("MISSPATH clamped working set to %d elems (%d B); scaling measured nanos by %.4f to estimate full slab", n, int64(n)*4, scale)
	}

	src := make([]float32, n)
	dst := make([]float32, n)
	for i := range src {
		src[i] = float32(i%97) * 0.5
	}

	// streamCopy times one streaming read/touch of the (possibly clamped) slab. This models
	// the EXACT expert bytes the host GEMV reads on a miss.
	streamCopy := func() int64 {
		started := time.Now()
		copy(dst, src)
		return time.Since(started).Nanoseconds()
	}

	host := compute.NewF32(compute.Default(), []int{n}, src)
	// uploadOnce times the H2D DMA of the same slab plus the Read fence that guarantees it
	// landed. Read is the only host fence, so the timer stops after the DMA is observable.
	uploadOnce := func() int64 {
		started := time.Now()
		u := be.Upload(host, compute.F32)
		be.Read(u)
		be.Free(u)
		return time.Since(started).Nanoseconds()
	}

	var cpuNanos, gpuNanos int64
	for i := 0; i < missReps; i++ {
		if ns := streamCopy(); cpuNanos == 0 || ns < cpuNanos {
			cpuNanos = ns
		}
	}
	for i := 0; i < missReps; i++ {
		if ns := uploadOnce(); gpuNanos == 0 || ns < gpuNanos {
			gpuNanos = ns
		}
	}
	if scale != 1.0 {
		cpuNanos = int64(float64(cpuNanos) * scale)
		gpuNanos = int64(float64(gpuNanos) * scale)
	}

	// Solo samples (one engine at a time) plus one concurrent pair, so
	// MeasureUMABandwidthSharing still validates the shared-memory picture. Bytes are the
	// expert slab for both arms: the scaled nanos represent the full slab even when clamped.
	samples := []UMABandwidthSharingSample{
		{Label: "solo-cpu", CPUBytes: slabBytes, CPUNanos: cpuNanos, Concurrent: false},
		{Label: "solo-gpu", GPUBytes: slabBytes, GPUNanos: gpuNanos, Concurrent: false},
	}
	{
		start := make(chan struct{})
		var wg sync.WaitGroup
		var concCPU, concGPU int64
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-start
			concCPU = streamCopy()
		}()
		go func() {
			defer wg.Done()
			<-start
			concGPU = uploadOnce()
		}()
		close(start)
		wg.Wait()
		samples = append(samples, UMABandwidthSharingSample{
			Label:      "concurrent",
			CPUBytes:   slabBytes,
			CPUNanos:   concCPU,
			GPUBytes:   slabBytes,
			GPUNanos:   concGPU,
			Concurrent: true,
		})
	}

	receipt, err := MeasureUMABandwidthSharing(samples)
	if err != nil {
		t.Fatalf("MeasureUMABandwidthSharing: %v", err)
	}
	receipt.Measurement = ExpertCacheMeasurementHardware

	// Derivation. perExpert*Ms are MEASURED hardware times for the one expert slab.
	perExpertCPUMs := float64(cpuNanos) / 1e6
	perExpertGPUMs := float64(gpuNanos) / 1e6
	cpuGbps := float64(slabBytes) / float64(cpuNanos)
	gpuGbps := float64(slabBytes) / float64(gpuNanos)
	// per_token_gpu_serial_ms is DERIVED: K misses serialized on the GPU, the GPU-only lower
	// bound. per_token_overlap_lower_bound_ms is DERIVED too: K * min(cpu,gpu), valid only if
	// the host CPU and the device DMA were independent. On a shared UMA they are not, so the
	// concurrent sample above is what tells you whether that lower bound is reachable.
	perTokenGPUSerialMs := float64(missesPerToken) * perExpertGPUMs
	overlapPerExpertMs := perExpertCPUMs
	if perExpertGPUMs < overlapPerExpertMs {
		overlapPerExpertMs = perExpertGPUMs
	}
	perTokenOverlapLowerBoundMs := float64(missesPerToken) * overlapPerExpertMs
	overlapImprovement := 0.0
	if perTokenOverlapLowerBoundMs > 0 {
		overlapImprovement = perTokenGPUSerialMs / perTokenOverlapLowerBoundMs
	}

	// HARD: the physical witness actually ran and moved bytes, and no derived value is NaN/Inf.
	if slabBytes <= 0 {
		t.Fatalf("slabBytes = %d, want > 0", slabBytes)
	}
	if cpuNanos <= 0 {
		t.Fatalf("cpuNanos = %d, want > 0 (CPU arm moved no bytes?)", cpuNanos)
	}
	if gpuNanos <= 0 {
		t.Fatalf("gpuNanos = %d, want > 0 (GPU DMA moved no bytes?)", gpuNanos)
	}
	for name, v := range map[string]float64{
		"solo_cpu_gbps":           receipt.SoloCPUGbps,
		"solo_gpu_gbps":           receipt.SoloGPUGbps,
		"concurrent_cpu_gbps":     receipt.ConcurrentCPUGbps,
		"concurrent_gpu_gbps":     receipt.ConcurrentGPUGbps,
		"cpu_contention_ratio":    receipt.CPUContentionRatio,
		"gpu_contention_ratio":    receipt.GPUContentionRatio,
		"shared_bandwidth_gbps":   receipt.SharedBandwidthGbps,
		"cpu_per_expert_ms":       perExpertCPUMs,
		"gpu_per_expert_ms":       perExpertGPUMs,
		"per_token_gpu_serial_ms": perTokenGPUSerialMs,
		"per_token_overlap_ms":    perTokenOverlapLowerBoundMs,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			t.Errorf("derived %s = %v, want finite and >= 0", name, v)
		}
	}

	t.Logf("MISSPATH MEASURED (hardware): slab_bytes=%d cpu_per_expert=%.3f ms gpu_per_expert=%.3f ms cpu=%.2f GB/s gpu=%.2f GB/s",
		slabBytes, perExpertCPUMs, perExpertGPUMs, cpuGbps, gpuGbps)
	t.Logf("MISSPATH DERIVED: misses_per_token=%d gpu_serial=%.3f ms/token overlap_lower_bound=%.3f ms/token serial->overlap improvement=%.2fx (lower bound only if CPU/GPU independent; see concurrent sample)",
		missesPerToken, perTokenGPUSerialMs, perTokenOverlapLowerBoundMs, overlapImprovement)
	t.Logf("MISSPATH concurrent (measured): cpu=%.2f GB/s gpu=%.2f GB/s shared=%.2f GB/s cpu_ratio=%.3f gpu_ratio=%.3f",
		receipt.ConcurrentCPUGbps, receipt.ConcurrentGPUGbps, receipt.SharedBandwidthGbps,
		receipt.CPUContentionRatio, receipt.GPUContentionRatio)

	receiptJSON := struct {
		Schema                      string  `json:"schema"`
		SlabBytes                   int64   `json:"slab_bytes"`
		CPUPerExpertMs              float64 `json:"cpu_per_expert_ms"`
		GPUPerExpertMs              float64 `json:"gpu_per_expert_ms"`
		MissesPerToken              int     `json:"misses_per_token"`
		PerTokenGPUSerialMs         float64 `json:"per_token_gpu_serial_ms"`
		PerTokenOverlapLowerBoundMs float64 `json:"per_token_overlap_lower_bound_ms"`
		CPUGbps                     float64 `json:"cpu_gbps"`
		GPUGbps                     float64 `json:"gpu_gbps"`
		Measurement                 string  `json:"measurement"`
		Notes                       string  `json:"notes"`
	}{
		Schema:                      "fak.v41-cpu-miss-misspath/v1",
		SlabBytes:                   slabBytes,
		CPUPerExpertMs:              perExpertCPUMs,
		GPUPerExpertMs:              perExpertGPUMs,
		MissesPerToken:              missesPerToken,
		PerTokenGPUSerialMs:         perTokenGPUSerialMs,
		PerTokenOverlapLowerBoundMs: perTokenOverlapLowerBoundMs,
		CPUGbps:                     cpuGbps,
		GPUGbps:                     gpuGbps,
		Measurement:                 "hardware",
		Notes: fmt.Sprintf(
			"cpu/gpu_per_expert_ms and *_gbps are physical; per_token_* are derived from K=%d. overlap lower bound is valid only if host CPU and device DMA are independent (shared UMA). clamped=%t scale=%.4f",
			missesPerToken, scale != 1.0, scale),
	}
	blob, err := json.Marshal(receiptJSON)
	if err != nil {
		t.Fatalf("marshal miss-path receipt: %v", err)
	}
	t.Logf("MISSPATH_HW_RECEIPT_JSON: %s", blob)
}
