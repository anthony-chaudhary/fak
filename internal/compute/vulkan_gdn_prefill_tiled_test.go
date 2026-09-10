//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

const (
	qwen35GDNTiledOutputEpsilon = float32(1e-5)
	qwen35GDNTiledProfileEnv    = "FAK_VULKAN_GDN_PREFILL_TILED_PROFILE"
)

type qwen35GDNTiledFixture struct {
	tokens, nK, nV, kHd, vHd, kernel int
	mixed, z, beta, alpha            []float32
	convW, aLog, dtBias, norm        []float32
	convState, recurrentState        []float32
}

func newQwen35GDNTiledFixture(tokens int) qwen35GDNTiledFixture {
	const nK, nV, kHd, vHd, kernel = 16, 48, 128, 128, 4
	convDim := 2*nK*kHd + nV*vHd
	valueDim := nV * vHd
	rng := rand.New(rand.NewSource(12533 + int64(tokens)))
	values := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = (rng.Float32()*2 - 1) * scale
		}
		return out
	}

	f := qwen35GDNTiledFixture{
		tokens: tokens, nK: nK, nV: nV, kHd: kHd, vHd: vHd, kernel: kernel,
		mixed:          values(tokens*convDim, .025),
		z:              values(tokens*valueDim, .6),
		beta:           values(tokens*nV, .3),
		alpha:          values(tokens*nV, .25),
		convW:          values(convDim*kernel, .15),
		aLog:           values(nV, .1),
		dtBias:         values(nV, .1),
		norm:           values(vHd, .1),
		convState:      values((kernel-1)*convDim, .015),
		recurrentState: values(nV*kHd*vHd, .004),
	}
	for i := range f.aLog {
		f.aLog[i] -= .9
	}
	for i := range f.norm {
		f.norm[i] += 1
	}
	return f
}

func newQwen35GDNTiledUnsupportedFixture(tokens int) qwen35GDNTiledFixture {
	const nK, nV, kHd, vHd, kernel = 1, 2, 4, 65, 3
	convDim := 2*nK*kHd + nV*vHd
	valueDim := nV * vHd
	rng := rand.New(rand.NewSource(1253308))
	values := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = (rng.Float32()*2 - 1) * scale
		}
		return out
	}
	f := qwen35GDNTiledFixture{
		tokens: tokens, nK: nK, nV: nV, kHd: kHd, vHd: vHd, kernel: kernel,
		mixed:          values(tokens*convDim, .025),
		z:              values(tokens*valueDim, .6),
		beta:           values(tokens*nV, .3),
		alpha:          values(tokens*nV, .25),
		convW:          values(convDim*kernel, .15),
		aLog:           values(nV, .1),
		dtBias:         values(nV, .1),
		norm:           values(vHd, .1),
		convState:      values((kernel-1)*convDim, .015),
		recurrentState: values(nV*kHd*vHd, .004),
	}
	for i := range f.aLog {
		f.aLog[i] -= .9
	}
	for i := range f.norm {
		f.norm[i] += 1
	}
	return f
}

func qwen35GDNTiledRows(values []float32, width, start, end int) []float32 {
	return values[start*width : end*width]
}

type qwen35GDNTiledDeviceCase struct {
	tb testing.TB
	v  *vulkanBackend
	f  qwen35GDNTiledFixture

	convW, aLog, dtBias, norm     Tensor
	convState, recurrentState     Tensor
	convStateID, recurrentStateID Buffer
}

type qwen35GDNTiledDevicePanel struct {
	mixed, z, beta, alpha Tensor
	tokens                int
}

func newQwen35GDNTiledDeviceCase(tb testing.TB, v *vulkanBackend, f qwen35GDNTiledFixture) *qwen35GDNTiledDeviceCase {
	tb.Helper()
	upload := func(shape []int, values []float32, class MemoryClass, name string) Tensor {
		tensor := v.UploadClass(NewF32(Default(), shape, values), F32, class, name)
		tb.Cleanup(func() { v.Free(tensor) })
		return tensor
	}
	convDim := 2*f.nK*f.kHd + f.nV*f.vHd
	d := &qwen35GDNTiledDeviceCase{tb: tb, v: v, f: f}
	d.convW = upload([]int{convDim, f.kernel}, f.convW, MemoryWeights, "tiled GDN convolution")
	d.aLog = upload([]int{f.nV}, f.aLog, MemoryWeights, "tiled GDN A log")
	d.dtBias = upload([]int{f.nV}, f.dtBias, MemoryWeights, "tiled GDN dt bias")
	d.norm = upload([]int{f.vHd}, f.norm, MemoryWeights, "tiled GDN output norm")
	d.convState = upload([]int{f.kernel - 1, convDim}, f.convState, MemoryKVCache, "tiled GDN convolution state")
	d.recurrentState = upload([]int{f.nV, f.kHd, f.vHd}, f.recurrentState, MemoryKVCache, "tiled GDN recurrent state")
	d.convStateID, d.recurrentStateID = d.convState.Buf(), d.recurrentState.Buf()
	return d
}

func (d *qwen35GDNTiledDeviceCase) uploadPanel(start, end int) qwen35GDNTiledDevicePanel {
	d.tb.Helper()
	tokens := end - start
	convDim := 2*d.f.nK*d.f.kHd + d.f.nV*d.f.vHd
	valueDim := d.f.nV * d.f.vHd
	upload := func(shape []int, values []float32, name string) Tensor {
		tensor := d.v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, name)
		d.tb.Cleanup(func() { d.v.Free(tensor) })
		return tensor
	}
	return qwen35GDNTiledDevicePanel{
		mixed:  upload([]int{tokens, convDim}, qwen35GDNTiledRows(d.f.mixed, convDim, start, end), "tiled GDN mixed"),
		z:      upload([]int{tokens, valueDim}, qwen35GDNTiledRows(d.f.z, valueDim, start, end), "tiled GDN z"),
		beta:   upload([]int{tokens, d.f.nV}, qwen35GDNTiledRows(d.f.beta, d.f.nV, start, end), "tiled GDN beta"),
		alpha:  upload([]int{tokens, d.f.nV}, qwen35GDNTiledRows(d.f.alpha, d.f.nV, start, end), "tiled GDN alpha"),
		tokens: tokens,
	}
}

func (d *qwen35GDNTiledDeviceCase) runPanel(panel qwen35GDNTiledDevicePanel) []float32 {
	d.tb.Helper()

	out, err := d.v.Qwen35GDNPreprojected(
		panel.mixed, panel.z, panel.beta, panel.alpha, d.convW, d.aLog, d.dtBias, d.norm, d.convState, d.recurrentState,
		panel.tokens, d.f.nK, d.f.nV, d.f.kHd, d.f.vHd, d.f.kernel, qwen35GDNTiledOutputEpsilon,
	)
	if err != nil {
		d.tb.Fatalf("Qwen35GDNPreprojected P%d failed: %v", panel.tokens, err)
	}
	if d.convState.Buf() != d.convStateID || d.recurrentState.Buf() != d.recurrentStateID {
		d.v.Free(out)
		d.tb.Fatal("GDN prefill replaced a durable state tensor instead of updating it in place")
	}
	got := d.v.Read(out)
	d.v.Free(out)
	return got
}

func (d *qwen35GDNTiledDeviceCase) run(start, end int) []float32 {
	d.tb.Helper()
	return d.runPanel(d.uploadPanel(start, end))
}

func qwen35GDNTiledCPUOracle(f qwen35GDNTiledFixture) (output, convState, recurrentState []float32) {
	// qwen35GDNPreprojectedOracle independently fixes Q/K L2 normalization at
	// 1e-6. The argument below is only the full-128-dimension output RMS epsilon.
	return qwen35GDNPreprojectedOracle(
		f.mixed, f.z, f.beta, f.alpha, f.convW, f.aLog, f.dtBias, f.norm,
		f.convState, f.recurrentState,
		f.tokens, f.nK, f.nV, f.kHd, f.vHd, f.kernel, qwen35GDNTiledOutputEpsilon,
	)
}

func qwen35GDNTiledAssertRelativeL2(tb testing.TB, name string, got, want []float32) {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("%s length=%d, want %d", name, len(got), len(want))
	}
	var squaredError, squaredReference float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			tb.Fatalf("%s[%d] is not finite: got=%g want=%g", name, i, got[i], want[i])
		}
		delta := g - w
		squaredError += delta * delta
		squaredReference += w * w
	}
	if squaredReference == 0 || math.IsNaN(squaredReference) || math.IsInf(squaredReference, 0) {
		tb.Fatalf("%s CPU oracle has no finite nonzero norm", name)
	}
	relativeL2 := math.Sqrt(squaredError / squaredReference)
	if math.IsNaN(relativeL2) || math.IsInf(relativeL2, 0) || relativeL2 > 1e-4 {
		tb.Fatalf("%s relative L2=%g, want <= 1e-4", name, relativeL2)
	}
}

func qwen35GDNTiledAssertTokenArgmax(tb testing.TB, got, want []float32, tokens, width int) {
	tb.Helper()
	meaningful := 0
	for token := 0; token < tokens; token++ {
		g, w := got[token*width:(token+1)*width], want[token*width:(token+1)*width]
		best, second := -math.MaxFloat64, -math.MaxFloat64
		for _, value := range w {
			v := float64(value)
			if v > best {
				second, best = best, v
			} else if v > second {
				second = v
			}
		}
		// Exact argmax is only informative when the CPU reference has a peak
		// separated from its runner-up by more than floating-point noise.
		if best-second <= 1e-6*math.Max(1, math.Abs(best)) {
			continue
		}
		meaningful++
		if argmaxF32(g) != argmaxF32(w) {
			tb.Fatalf("token %d argmax=%d, want %d (reference margin=%g)", token, argmaxF32(g), argmaxF32(w), best-second)
		}
	}
	if meaningful == 0 {
		tb.Fatal("CPU oracle produced no token with a meaningful argmax margin")
	}
}

func qwen35GDNTiledRequireAvailable(tb testing.TB, v *vulkanBackend) {
	tb.Helper()
	if v.VulkanDebugGDNTiledPrefillAvailable() {
		return
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		tb.Fatal("required Vulkan device does not expose the tiled GDN prefill pipeline")
	}
	tb.Skip("tiled GDN prefill pipeline unavailable in this Vulkan bundle")
}

func TestVulkanQwen35GDNTiledPrefillParityAndSplitContinuation(t *testing.T) {
	v := vk(t)
	qwen35GDNTiledRequireAvailable(t, v)
	defer v.VulkanDebugSetGDNTiledPrefillMode(-1)

	t.Run("P1FallsBackToOriginal", func(t *testing.T) {
		f := newQwen35GDNTiledFixture(1)
		want, wantC, wantR := qwen35GDNTiledCPUOracle(f)
		d := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugSetGDNTiledPrefillMode(1)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		got := d.run(0, 1)
		tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot()
		if tiled != 0 || original != 1 {
			t.Fatalf("P1 dispatch attribution tiled/original=%d/%d, want 0/1", tiled, original)
		}
		qwen35GDNTiledAssertRelativeL2(t, "P1 output", got, want)
		qwen35GDNTiledAssertRelativeL2(t, "P1 convolution state", v.Read(d.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "P1 recurrent state", v.Read(d.recurrentState), wantR)
	})

	t.Run("P8UnsupportedGeometryFallsBackToOriginal", func(t *testing.T) {
		f := newQwen35GDNTiledUnsupportedFixture(8)
		want, wantC, wantR := qwen35GDNTiledCPUOracle(f)
		d := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugSetGDNTiledPrefillMode(1)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		got := d.run(0, f.tokens)
		tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot()
		if tiled != 0 || original != 1 {
			t.Fatalf("unsupported P8 dispatch attribution tiled/original=%d/%d, want 0/1", tiled, original)
		}
		qwen35GDNTiledAssertRelativeL2(t, "unsupported P8 output", got, want)
		qwen35GDNTiledAssertRelativeL2(t, "unsupported P8 convolution state", v.Read(d.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "unsupported P8 recurrent state", v.Read(d.recurrentState), wantR)
		qwen35GDNTiledAssertTokenArgmax(t, got, want, f.tokens, f.nV*f.vHd)
	})

	t.Run("P8GroupedHeadsMatchesCPU", func(t *testing.T) {
		f := newQwen35GDNTiledFixture(8)
		if f.nV/f.nK != 3 {
			t.Fatalf("fixture group ratio=%d, want 3", f.nV/f.nK)
		}
		want, wantC, wantR := qwen35GDNTiledCPUOracle(f)
		d := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugSetGDNTiledPrefillMode(1)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		got := d.run(0, f.tokens)
		tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot()
		if tiled != 1 || original != 0 {
			t.Fatalf("P8 dispatch attribution tiled/original=%d/%d, want 1/0", tiled, original)
		}
		qwen35GDNTiledAssertRelativeL2(t, "P8 output", got, want)
		qwen35GDNTiledAssertRelativeL2(t, "P8 convolution state", v.Read(d.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "P8 recurrent state", v.Read(d.recurrentState), wantR)
		qwen35GDNTiledAssertTokenArgmax(t, got, want, f.tokens, f.nV*f.vHd)
	})

	t.Run("P8OriginalRouteWhenDisabled", func(t *testing.T) {
		f := newQwen35GDNTiledFixture(8)
		want, wantC, wantR := qwen35GDNTiledCPUOracle(f)
		d := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugSetGDNTiledPrefillMode(0)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		got := d.run(0, f.tokens)
		tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot()
		if tiled != 0 || original != 1 {
			t.Fatalf("disabled-route dispatch attribution tiled/original=%d/%d, want 0/1", tiled, original)
		}
		qwen35GDNTiledAssertRelativeL2(t, "original P8 output", got, want)
		qwen35GDNTiledAssertRelativeL2(t, "original P8 convolution state", v.Read(d.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "original P8 recurrent state", v.Read(d.recurrentState), wantR)
	})

	t.Run("P17SplitAt8PreservesContinuation", func(t *testing.T) {
		f := newQwen35GDNTiledFixture(17)
		want, wantC, wantR := qwen35GDNTiledCPUOracle(f)

		full := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugSetGDNTiledPrefillMode(1)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		gotFull := full.run(0, f.tokens)
		if tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot(); tiled != 1 || original != 0 {
			t.Fatalf("P17 dispatch attribution tiled/original=%d/%d, want 1/0", tiled, original)
		}

		split := newQwen35GDNTiledDeviceCase(t, v, f)
		v.VulkanDebugResetGDNTiledPrefillProfile()
		gotSplit := append(split.run(0, 8), split.run(8, 17)...)
		if tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot(); tiled != 2 || original != 0 {
			t.Fatalf("P17 split dispatch attribution tiled/original=%d/%d, want 2/0", tiled, original)
		}

		qwen35GDNTiledAssertRelativeL2(t, "P17 full output", gotFull, want)
		qwen35GDNTiledAssertRelativeL2(t, "P17 split output", gotSplit, want)
		qwen35GDNTiledAssertRelativeL2(t, "P17 full/split output", gotSplit, gotFull)
		qwen35GDNTiledAssertRelativeL2(t, "P17 full convolution state", v.Read(full.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "P17 split convolution state", v.Read(split.convState), wantC)
		qwen35GDNTiledAssertRelativeL2(t, "P17 full recurrent state", v.Read(full.recurrentState), wantR)
		qwen35GDNTiledAssertRelativeL2(t, "P17 split recurrent state", v.Read(split.recurrentState), wantR)
		qwen35GDNTiledAssertTokenArgmax(t, gotSplit, want, f.tokens, f.nV*f.vHd)
	})
}

func qwen35GDNTiledPercentile(samples []time.Duration, quantile float64) time.Duration {
	copyOf := append([]time.Duration(nil), samples...)
	sort.Slice(copyOf, func(i, j int) bool { return copyOf[i] < copyOf[j] })
	position := quantile * float64(len(copyOf)-1)
	lower := int(math.Floor(position))
	upper := int(math.Ceil(position))
	if lower == upper {
		return copyOf[lower]
	}
	weight := position - float64(lower)
	return time.Duration(float64(copyOf[lower])*(1-weight) + float64(copyOf[upper])*weight)
}

func qwen35GDNTiledMedianCI(samples []time.Duration) [2]int64 {
	const bootstrapSamples = 400
	rng := rand.New(rand.NewSource(12533 + int64(len(samples))))
	medians := make([]time.Duration, bootstrapSamples)
	resample := make([]time.Duration, len(samples))
	for i := range medians {
		for j := range resample {
			resample[j] = samples[rng.Intn(len(samples))]
		}
		medians[i] = qwen35GDNTiledPercentile(resample, .5)
	}
	return [2]int64{
		qwen35GDNTiledPercentile(medians, .025).Nanoseconds(),
		qwen35GDNTiledPercentile(medians, .975).Nanoseconds(),
	}
}

func qwen35GDNTiledDurationNS(samples []time.Duration) []int64 {
	out := make([]int64, len(samples))
	for i := range samples {
		out[i] = samples[i].Nanoseconds()
	}
	return out
}

func BenchmarkVulkanQwen35GDNTiledPrefillComponentAB(b *testing.B) {
	if os.Getenv(qwen35GDNTiledProfileEnv) != "1" {
		b.Skip("set " + qwen35GDNTiledProfileEnv + "=1 for the physical-device component A/B")
	}
	v, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		b.Skip("Vulkan backend unavailable")
	}
	qwen35GDNTiledRequireAvailable(b, v)
	defer v.VulkanDebugSetGDNTiledPrefillMode(-1)

	for _, tokens := range []int{32, 128, 512} {
		b.Run("P"+strconv.Itoa(tokens), func(b *testing.B) {
			f := newQwen35GDNTiledFixture(tokens)
			scalar := newQwen35GDNTiledDeviceCase(b, v, f)
			candidate := newQwen35GDNTiledDeviceCase(b, v, f)
			scalarPanel := scalar.uploadPanel(0, tokens)
			candidatePanel := candidate.uploadPanel(0, tokens)
			runArm := func(mode int, d *qwen35GDNTiledDeviceCase) (time.Duration, []float32) {
				v.VulkanDebugSetGDNTiledPrefillMode(mode)
				start := time.Now()
				var output []float32
				if mode == 0 {
					output = d.runPanel(scalarPanel) // Read synchronizes the physical dispatch.
				} else {
					output = d.runPanel(candidatePanel)
				}
				return time.Since(start), output
			}

			for i := 0; i < 2; i++ {
				runArm(0, scalar)
				runArm(1, candidate)
			}
			pairs := b.N
			if pairs < 6 {
				pairs = 6
			}
			scalarSamples := make([]time.Duration, 0, pairs)
			candidateSamples := make([]time.Duration, 0, pairs)
			orders := make([]string, 0, pairs)
			var scalarOutput, candidateOutput []float32
			v.VulkanDebugResetGDNTiledPrefillProfile()
			b.ResetTimer()
			for i := 0; i < pairs; i++ {
				if i%2 == 0 {
					var elapsed time.Duration
					elapsed, scalarOutput = runArm(0, scalar)
					scalarSamples = append(scalarSamples, elapsed)
					elapsed, candidateOutput = runArm(1, candidate)
					candidateSamples = append(candidateSamples, elapsed)
					orders = append(orders, "original,tiled")
				} else {
					var elapsed time.Duration
					elapsed, candidateOutput = runArm(1, candidate)
					candidateSamples = append(candidateSamples, elapsed)
					elapsed, scalarOutput = runArm(0, scalar)
					scalarSamples = append(scalarSamples, elapsed)
					orders = append(orders, "tiled,original")
				}
			}
			b.StopTimer()
			if tiled, original := v.VulkanDebugGDNTiledPrefillProfileSnapshot(); tiled != int64(pairs) || original != int64(pairs) {
				b.Fatalf("A/B dispatch attribution tiled/original=%d/%d, want %d/%d", tiled, original, pairs, pairs)
			}
			qwen35GDNTiledAssertRelativeL2(b, "measured A/B output", candidateOutput, scalarOutput)
			qwen35GDNTiledAssertRelativeL2(b, "measured A/B convolution state", v.Read(candidate.convState), v.Read(scalar.convState))
			qwen35GDNTiledAssertRelativeL2(b, "measured A/B recurrent state", v.Read(candidate.recurrentState), v.Read(scalar.recurrentState))
			qwen35GDNTiledAssertTokenArgmax(b, candidateOutput, scalarOutput, tokens, f.nV*f.vHd)

			originalP50 := qwen35GDNTiledPercentile(scalarSamples, .5)
			originalP90 := qwen35GDNTiledPercentile(scalarSamples, .9)
			tiledP50 := qwen35GDNTiledPercentile(candidateSamples, .5)
			tiledP90 := qwen35GDNTiledPercentile(candidateSamples, .9)
			receipt := map[string]any{
				"schema": "fak.vulkan.gdn-prefill-component-ab/1", "device": v.Tier(),
				"shape":           map[string]int{"tokens": tokens, "key_heads": f.nK, "value_heads": f.nV, "key_head_dim": f.kHd, "value_head_dim": f.vHd, "conv_kernel": f.kernel},
				"warmups_per_arm": 2, "pair_count": pairs, "pair_order": orders,
				"original_samples_ns": qwen35GDNTiledDurationNS(scalarSamples), "tiled_samples_ns": qwen35GDNTiledDurationNS(candidateSamples),
				"original_p50_ns": originalP50.Nanoseconds(), "original_p90_ns": originalP90.Nanoseconds(), "original_p50_ci95_ns": qwen35GDNTiledMedianCI(scalarSamples),
				"tiled_p50_ns": tiledP50.Nanoseconds(), "tiled_p90_ns": tiledP90.Nanoseconds(), "tiled_p50_ci95_ns": qwen35GDNTiledMedianCI(candidateSamples),
				"source_env": map[string]string{
					qwen35GDNTiledProfileEnv: os.Getenv(qwen35GDNTiledProfileEnv), "FAK_VULKAN_GDN_PREFILL_TILED": os.Getenv("FAK_VULKAN_GDN_PREFILL_TILED"),
					"FAK_VULKAN_REQUIRE_DEVICE": os.Getenv("FAK_VULKAN_REQUIRE_DEVICE"), "FAK_VULKAN_EXPECT_DEVICE": os.Getenv("FAK_VULKAN_EXPECT_DEVICE"),
				},
				"timing_scope":     "resident preprojected GDN call plus synchronized output read; fixture generation, input upload, warmup, and validation excluded",
				"state_trajectory": "paired arms start equal and advance once per pair; final output and durable states pass relative-L2 and argmax checks",
				"claim_scope":      "physical preprojected component only; no full-model token throughput claim",
			}
			encoded, err := json.Marshal(receipt)
			if err != nil {
				b.Fatal(err)
			}
			b.Log(string(encoded))
			b.ReportMetric(float64(originalP50.Nanoseconds()), "original_p50_ns")
			b.ReportMetric(float64(originalP90.Nanoseconds()), "original_p90_ns")
			b.ReportMetric(float64(tiledP50.Nanoseconds()), "tiled_p50_ns")
			b.ReportMetric(float64(tiledP90.Nanoseconds()), "tiled_p90_ns")
			b.ReportMetric(float64(pairs), "pairs")
			var pairTotal time.Duration
			for i := range scalarSamples {
				pairTotal += scalarSamples[i] + candidateSamples[i]
			}
			b.ReportMetric(float64(pairTotal.Nanoseconds())/float64(pairs), "ns/op")
		})
	}
}
