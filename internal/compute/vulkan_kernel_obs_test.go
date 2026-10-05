package compute

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// fak-test:runtime fast est=2s — pure in-memory event construction and Prometheus
// text rendering; no device, no dispatch, no model load.

const vulkanKernelBackendName = "vulkan"

// observeKernelBoth installs a process observer that fans each event to BOTH a
// stepobs recorder and a local slice. SetObserver replaces rather than chains, so a
// test that must assert on the raw event AND on the rendered family has to fan out
// itself. It restores the empty seam afterwards so subtests are order-independent.
func observeKernelBoth(t *testing.T, r *stepobs.Recorder) *[]computetrace.Event {
	t.Helper()
	var seen []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) {
		seen = append(seen, e)
		r.ObserveKernel(e)
	})
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	return &seen
}

// vkMatMulTensors builds a weight/activation pair of the requested weight dtype on
// the always-present CPU reference, so the emission seam can be exercised on any
// host with or without a Vulkan device. The seam is pure event construction: it
// never dispatches, so the payload only has to carry the right dtype and shape.
func vkMatMulTensors(t *testing.T, dt Dtype) (Tensor, Tensor) {
	t.Helper()
	c := Pick("cpu-ref")
	if c == nil {
		t.Fatal("cpu reference backend is absent")
	}
	// in=256 is the smallest reduction width that is simultaneously a valid F32,
	// Q8_0 (block 32) and Q4_K (super-block 256) shape, so one helper serves every
	// weight dtype a real Vulkan matmul dispatch accepts.
	const out, in = 4, 256
	var w Tensor
	switch dt {
	case F32:
		w = NewF32(c, []int{out, in}, make([]float32, out*in))
	case Q8_0:
		w = QuantizeQ8(c, []int{out, in}, make([]float32, out*in), 32)
	case Q4_K:
		w = NewQ4K(c, []int{out, in}, make([]byte, out*(in/q4kSuper)*q4kSuperBlock))
	default:
		t.Fatalf("vkMatMulTensors has no case for weight dtype %s", dt)
	}
	return w, NewF32(c, []int{in}, make([]float32, in))
}

// vkHistogramSuffixes are the exposition suffixes a histogram body carries. They are
// stripped so a family index keys on the FAMILY name, not on _bucket/_sum/_count.
var vkHistogramSuffixes = []string{"_bucket", "_sum", "_count"}

// vkRender renders a recorder's Prometheus text and returns (family -> its rendered
// lines including histogram components, whole render).
func vkRender(r *stepobs.Recorder) (map[string][]string, string) {
	var sb strings.Builder
	r.WritePrometheus(&sb)
	render := sb.String()
	families := map[string][]string{}
	for _, line := range strings.Split(render, "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		name, _, _ := strings.Cut(line, "{")
		name, _, _ = strings.Cut(name, " ")
		for _, suffix := range vkHistogramSuffixes {
			name = strings.TrimSuffix(name, suffix)
		}
		families[name] = append(families[name], line)
	}
	return families, render
}

// vkHistogramComponent returns the single rendered _bucket/_sum/_count line of a
// family carrying the exact label substring match.
func vkHistogramComponent(t *testing.T, lines []string, family, component, labelMatch string) string {
	t.Helper()
	var found []string
	for _, line := range lines {
		if strings.HasPrefix(line, family+component) && strings.Contains(line, labelMatch) {
			found = append(found, line)
		}
	}
	if len(found) != 1 {
		t.Fatalf("%s%s with %q: want exactly 1 rendered line, got %d", family, component, labelMatch, len(found))
	}
	return found[0]
}

// vkSingleSample returns the one rendered line of a family that carries no labels.
func vkSingleSample(t *testing.T, lines []string, family string) string {
	t.Helper()
	if len(lines) != 1 {
		t.Fatalf("%s: want exactly 1 rendered sample, got %d", family, len(lines))
	}
	return lines[0]
}

// vkMetricValue reads the value of a single unlabelled gauge/counter sample.
func vkMetricValue(t *testing.T, sample, family string) int64 {
	t.Helper()
	want := family + " "
	if !strings.HasPrefix(sample, want) {
		t.Fatalf("sample %q does not belong to family %q", sample, family)
	}
	v, err := strconv.ParseInt(strings.TrimPrefix(sample, want), 10, 64)
	if err != nil {
		t.Fatalf("family %q sample %q is not an integer: %v", family, sample, err)
	}
	return v
}

// deviceObservation is a live RADV/Vulkan performance-query surface that declares a
// nanoseconds compute-scope counter — the shape the Halo reports.
func deviceObservation() PhasePerformanceQueryObservation {
	return PhasePerformanceQueryObservation{
		Supported: true,
		Descriptors: []PhasePerformanceCounterDescriptor{
			{Index: 0, Name: "GPUTime", Unit: PhaseCounterUnitNanoseconds, Scope: PhaseCounterScopeCompute},
			{Index: 1, Name: "GPUMemoryBytes", Unit: PhaseCounterUnitBytes, Scope: PhaseCounterScopeCompute},
		},
	}
}

// TestVulkanKernelTimingIsDeviceMeasuredOrExplicitlyUnavailable pins the closed
// contract of the device timer: a nanoseconds-unit AVAILABLE reading is the only
// accepted device measurement, and every other verdict is the explicit unavailable
// marker with no duration at all.
func TestVulkanKernelTimingIsDeviceMeasuredOrExplicitlyUnavailable(t *testing.T) {
	t.Run("sampled nanoseconds counter is the device measurement", func(t *testing.T) {
		got := vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUTime": 4200})
		if !got.Available {
			t.Fatal("a sampled nanoseconds counter must resolve as an available device measurement")
		}
		if got.DurationNS != 4200 || got.Counter != "GPUTime" {
			t.Fatalf("device measurement = %d ns from %q; want 4200 ns from \"GPUTime\"", got.DurationNS, got.Counter)
		}
		if got.timerDomain() != VulkanTimerDomainDevice {
			t.Fatalf("timer domain = %q; want %q", got.timerDomain(), VulkanTimerDomainDevice)
		}
	})

	t.Run("unsupported surface is unavailable", func(t *testing.T) {
		got := vulkanKernelTimingFrom(
			PhasePerformanceQueryObservation{Reason: "device does not expose VK_KHR_performance_query"},
			vulkanKernelPhase, map[string]uint64{"GPUTime": 4200})
		if got.Available || got.DurationNS != 0 {
			t.Fatalf("unsupported surface resolved as a measurement: %+v", got)
		}
		if got.timerDomain() != VulkanTimerDomainUnavailable {
			t.Fatalf("timer domain = %q; want %q", got.timerDomain(), VulkanTimerDomainUnavailable)
		}
	})

	t.Run("supported but unsampled counter is unavailable", func(t *testing.T) {
		got := vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, nil)
		if got.Available || got.DurationNS != 0 || got.Counter != "" {
			t.Fatalf("unsampled counter resolved as a measurement: %+v", got)
		}
		if got.timerDomain() != VulkanTimerDomainUnavailable {
			t.Fatalf("timer domain = %q; want %q", got.timerDomain(), VulkanTimerDomainUnavailable)
		}
	})

	t.Run("a non-nanoseconds unit is never a duration", func(t *testing.T) {
		got := vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUMemoryBytes": 4096})
		if got.Available || got.DurationNS != 0 {
			t.Fatalf("a bytes counter was read as a duration: %+v", got)
		}
		if got.timerDomain() != VulkanTimerDomainUnavailable {
			t.Fatalf("timer domain = %q; want %q", got.timerDomain(), VulkanTimerDomainUnavailable)
		}
	})
}

// TestVulkanKernelEventCarriesNoHostProxy is the anti-fabrication pin: no code path
// through the Vulkan emitter can put a host-monotonic reading under a device timer
// domain. The only two domains the emitter can mint are the device query and the
// explicit unavailable marker.
func TestVulkanKernelEventCarriesNoHostProxy(t *testing.T) {
	w, x := vkMatMulTensors(t, F32)
	started := time.Unix(1700000000, 0).UTC()

	for _, timing := range []vulkanKernelTiming{
		{},
		vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, nil),
		vulkanKernelTimingFrom(PhasePerformanceQueryObservation{Reason: "no counters"}, vulkanKernelPhase, map[string]uint64{"GPUTime": 1}),
		vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUTime": 4096}),
	} {
		e := vulkanKernelEvent(vulkanKernelBackendName, w, x, started, timing)
		switch e.TimerDomain {
		case VulkanTimerDomainDevice:
			if !timing.Available {
				t.Fatal("device timer domain on an unavailable verdict")
			}
		case VulkanTimerDomainUnavailable:
			if timing.Available || e.DurationNS != 0 {
				t.Fatalf("unavailable event carries a duration: %+v", e)
			}
		default:
			t.Fatalf("event minted timer domain %q, which is neither the device query nor the unavailable marker", e.TimerDomain)
		}
		if e.TimerDomain == "host_monotonic" {
			t.Fatal("the Vulkan emitter must never report host_monotonic")
		}
	}
}

// TestVulkanKernelEventMatchesCPURefShape pins agreement with the CPU-reference
// emitter (cpuref.go MatMul): same Operation, same Phase, same <dtype>_matmul
// kernel name, so the two sites agree on the bounded key set.
func TestVulkanKernelEventMatchesCPURefShape(t *testing.T) {
	w, x := vkMatMulTensors(t, Q4_K)
	e := vulkanKernelEvent(vulkanKernelBackendName, w, x, time.Unix(1700000000, 0).UTC(),
		vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUTime": 77}))

	if e.Operation != "matmul" {
		t.Fatalf("Operation = %q; want \"matmul\"", e.Operation)
	}
	if e.Phase != "kernel" {
		t.Fatalf("Phase = %q; want \"kernel\"", e.Phase)
	}
	if want := Q4_K.String() + "_matmul"; e.Kernel != want {
		t.Fatalf("Kernel = %q; want %q", e.Kernel, want)
	}
	if e.Backend != vulkanKernelBackendName || e.Device != vulkanKernelDeviceLabel {
		t.Fatalf("Backend/Device = %q/%q; want %q/%q", e.Backend, e.Device, vulkanKernelBackendName, vulkanKernelDeviceLabel)
	}
	if e.Route != "device" {
		t.Fatalf("Route = %q; want \"device\"", e.Route)
	}
	if e.ProvenanceDigest != computetrace.Digest(vulkanKernelBackendName, Q4_K.String(), "matmul") {
		t.Fatal("provenance digest does not match the digest the reference site computes")
	}
}

// TestVulkanKernelDispatchFeedsObserverAndFlipsObserved is the end-to-end seam
// witness: one Vulkan dispatch emits exactly ONE kernel event, carrying a
// non-host_monotonic timer domain, and that event flips
// fak_engine_kernel_observed from 0 to 1 through the real stepobs renderer.
func TestVulkanKernelDispatchFeedsObserverAndFlipsObserved(t *testing.T) {
	rec := stepobs.New()
	rec.Attach()
	t.Cleanup(func() { computetrace.SetObserver(nil) })

	w, x := vkMatMulTensors(t, Q4_K)
	timing := vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUTime": 9000})

	before, _ := vkRender(rec)
	if got := vkMetricValue(t, vkSingleSample(t, before[stepobs.MetricKernelObserved], stepobs.MetricKernelObserved), stepobs.MetricKernelObserved); got != 0 {
		t.Fatalf("%s = %d before any dispatch; want 0", stepobs.MetricKernelObserved, got)
	}

	recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Unix(1700000000, 0).UTC(), timing)

	families, render := vkRender(rec)
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelObserved], stepobs.MetricKernelObserved), stepobs.MetricKernelObserved); got != 1 {
		t.Fatalf("%s = %d after one dispatch; want 1\n%s", stepobs.MetricKernelObserved, got, render)
	}

	calls := families[stepobs.MetricKernelCalls]
	if len(calls) != 1 {
		t.Fatalf("%s: want exactly one (kernel, backend) series for one dispatch, got %d:\n%s", stepobs.MetricKernelCalls, len(calls), render)
	}
	wantCall := stepobs.MetricKernelCalls + `{kernel="` + Q4_K.String() + `_matmul",backend="vulkan"} 1`
	if calls[0] != wantCall {
		t.Fatalf("call series = %q; want %q", calls[0], wantCall)
	}

	seconds := families[stepobs.MetricKernelSeconds]
	deviceLabels := `timer_domain="` + VulkanTimerDomainDevice + `"`
	count := vkHistogramComponent(t, seconds, stepobs.MetricKernelSeconds, "_count", deviceLabels)
	if !strings.HasSuffix(count, " 1") {
		t.Fatalf("%s = %q; want a count of exactly 1", stepobs.MetricKernelSeconds, count)
	}
	sum := vkHistogramComponent(t, seconds, stepobs.MetricKernelSeconds, "_sum", deviceLabels)
	if want := " 0.000009"; !strings.HasSuffix(sum, want) {
		t.Fatalf("%s = %q; want the 9000ns device measurement %q", stepobs.MetricKernelSeconds, sum, want)
	}
	for _, line := range seconds {
		if strings.Contains(line, `timer_domain="host_monotonic"`) {
			t.Fatalf("the Vulkan dispatch rendered a host_monotonic kernel duration: %q", line)
		}
	}
	if strings.Contains(render, `timer_domain="host_monotonic"`) {
		t.Fatalf("render carries a host_monotonic kernel series after a Vulkan dispatch:\n%s", render)
	}
}

// TestVulkanKernelAbsentProducerKeepsObservedAtZero is the negative pin: with no
// producer attached, the Vulkan emitter mints nothing and the family still reads
// 0. Absence must keep reading as absence — never as a fabricated observation.
func TestVulkanKernelAbsentProducerKeepsObservedAtZero(t *testing.T) {
	computetrace.SetObserver(nil)
	if computetrace.Emitting() {
		t.Fatal("Emitting() is true with neither a recorder nor an observer attached")
	}

	w, x := vkMatMulTensors(t, Q4_K)
	recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Now(), vulkanKernelTiming{})
	recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Now(),
		vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, map[string]uint64{"GPUTime": 5}))

	rec := stepobs.New()
	families, render := vkRender(rec)
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelObserved], stepobs.MetricKernelObserved), stepobs.MetricKernelObserved); got != 0 {
		t.Fatalf("%s = %d with no producer attached; want 0\n%s", stepobs.MetricKernelObserved, got, render)
	}
	if n := len(families[stepobs.MetricKernelCalls]); n != 0 {
		t.Fatalf("%s rendered %d series with no producer attached; want 0\n%s", stepobs.MetricKernelCalls, n, render)
	}
}

// TestVulkanKernelUnavailableTimerIsNotADeviceMeasurement pins the honest-telemetry
// contract on the render surface: a dispatch whose device timer could not be read
// still counts the CALL, still flips observed to 1, and reports zero duration under
// the explicit unavailable marker — so no device-duration consumer can read the
// unavailable dispatch as a measured kernel.
func TestVulkanKernelUnavailableTimerIsNotADeviceMeasurement(t *testing.T) {
	rec := stepobs.New()
	seen := observeKernelBoth(t, rec)

	w, x := vkMatMulTensors(t, Q4_K)
	timing := vulkanKernelTimingFrom(
		PhasePerformanceQueryObservation{Reason: "device does not expose VK_KHR_performance_query"},
		vulkanKernelPhase, nil)

	recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Now(), timing)

	if len(*seen) != 1 {
		t.Fatalf("want exactly 1 kernel event, got %d", len(*seen))
	}
	e := (*seen)[0]
	if e.TimerDomain != VulkanTimerDomainUnavailable {
		t.Fatalf("TimerDomain = %q; want the explicit marker %q", e.TimerDomain, VulkanTimerDomainUnavailable)
	}
	if e.DurationNS != 0 {
		t.Fatalf("unavailable event carries DurationNS=%d; want 0", e.DurationNS)
	}

	families, render := vkRender(rec)
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelObserved], stepobs.MetricKernelObserved), stepobs.MetricKernelObserved); got != 1 {
		t.Fatalf("%s = %d; a real kernel ran, so the family must read 1\n%s", stepobs.MetricKernelObserved, got, render)
	}
	calls := families[stepobs.MetricKernelCalls]
	if len(calls) != 1 || !strings.HasSuffix(calls[0], " 1") {
		t.Fatalf("%s must count the call exactly once: %v", stepobs.MetricKernelCalls, calls)
	}
	unavailableLabels := `timer_domain="` + VulkanTimerDomainUnavailable + `"`
	count := vkHistogramComponent(t, families[stepobs.MetricKernelSeconds], stepobs.MetricKernelSeconds, "_count", unavailableLabels)
	if !strings.HasSuffix(count, " 1") {
		t.Fatalf("%s = %q; the CALL must still be counted under the unavailable marker", stepobs.MetricKernelSeconds, count)
	}
	sum := vkHistogramComponent(t, families[stepobs.MetricKernelSeconds], stepobs.MetricKernelSeconds, "_sum", unavailableLabels)
	if !strings.HasSuffix(sum, " 0") {
		t.Fatalf("%s = %q; an unavailable timer must contribute exactly zero measured seconds", stepobs.MetricKernelSeconds, sum)
	}
	for _, line := range families[stepobs.MetricKernelSeconds] {
		if strings.Contains(line, `timer_domain="`+VulkanTimerDomainDevice+`"`) {
			t.Fatalf("the unavailable dispatch minted a device-duration series: %q\n%s", line, render)
		}
	}
}

// TestVulkanKernelKeySetDoesNotGrowPerDispatch pins the cardinality contract: repeated
// dispatches of the same dtype collapse onto ONE (kernel, backend, timer_domain) key,
// so the bounded set in internal/stepobs cannot grow with dispatch count. A
// per-dispatch string would add a key here.
func TestVulkanKernelKeySetDoesNotGrowPerDispatch(t *testing.T) {
	rec := stepobs.New()
	rec.Attach()
	t.Cleanup(func() { computetrace.SetObserver(nil) })

	w, x := vkMatMulTensors(t, Q8_0)
	samples := map[string]uint64{"GPUTime": 1000}
	for i := 0; i < 64; i++ {
		recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Now(),
			vulkanKernelTimingFrom(deviceObservation(), vulkanKernelPhase, samples))
	}

	families, render := vkRender(rec)
	if n := len(families[stepobs.MetricKernelCalls]); n != 1 {
		t.Fatalf("64 dispatches produced %d (kernel, backend) series; want 1\n%s", n, render)
	}
	snap := rec.Snapshot()
	if len(snap.KernelKeys) != 1 {
		t.Fatalf("64 dispatches produced %d keys %v; want exactly 1", len(snap.KernelKeys), snap.KernelKeys)
	}
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelKeysCapped], stepobs.MetricKernelKeysCapped), stepobs.MetricKernelKeysCapped); got != 0 {
		t.Fatalf("%s = %d; the emitter must stay well inside the bound\n%s", stepobs.MetricKernelKeysCapped, got, render)
	}
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelEventOverflowTotal], stepobs.MetricKernelEventOverflowTotal), stepobs.MetricKernelEventOverflowTotal); got != 0 {
		t.Fatalf("%s = %d; want 0\n%s", stepobs.MetricKernelEventOverflowTotal, got, render)
	}
	if snap.KernelEvents != 64 {
		t.Fatalf("KernelEvents = %d; want 64 (one per dispatch)", snap.KernelEvents)
	}
}

// TestVulkanKernelOverflowFoldsPastTheBound proves the bound is real rather than
// merely unexercised: once the (kernel, backend, timer_domain) set is full, further
// DISTINCT keys fold into the single overflow key and the overflow counter advances,
// so no producer can grow the family without limit.
func TestVulkanKernelOverflowFoldsPastTheBound(t *testing.T) {
	rec := stepobs.New()
	rec.Attach()
	t.Cleanup(func() { computetrace.SetObserver(nil) })

	for i := 0; i <= stepobs.MaxKernelKeys; i++ {
		computetrace.Record(computetrace.Event{
			Operation: "matmul", Phase: vulkanKernelPhase,
			Kernel:  "synthetic_" + strconv.Itoa(i) + "_matmul",
			Backend: vulkanKernelBackendName, Device: vulkanKernelDeviceLabel,
			DurationNS: 1000, TimerDomain: VulkanTimerDomainDevice,
		})
	}

	families, render := vkRender(rec)
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelKeysCapped], stepobs.MetricKernelKeysCapped), stepobs.MetricKernelKeysCapped); got != 1 {
		t.Fatalf("%s = %d past the bound; want 1\n%s", stepobs.MetricKernelKeysCapped, got, render)
	}
	if got := vkMetricValue(t, vkSingleSample(t, families[stepobs.MetricKernelEventOverflowTotal], stepobs.MetricKernelEventOverflowTotal), stepobs.MetricKernelEventOverflowTotal); got == 0 {
		t.Fatalf("%s = 0 past the bound; overflow must be counted\n%s", stepobs.MetricKernelEventOverflowTotal, render)
	}

	snap := rec.Snapshot()
	if !snap.KernelKeysCapped || snap.KernelOverflow == 0 {
		t.Fatalf("snapshot does not report the cap: capped=%v overflow=%d", snap.KernelKeysCapped, snap.KernelOverflow)
	}
	// The overflow key is one fixed constant, so folding never adds a real key.
	if len(snap.KernelKeys) != stepobs.MaxKernelKeys+1 {
		t.Fatalf("key count = %d; want %d (%d real keys + the single overflow key)", len(snap.KernelKeys), stepobs.MaxKernelKeys+1, stepobs.MaxKernelKeys)
	}

	// Past the bound the emitter's own key folds into the single overflow slot
	// rather than growing the set, and the fold is counted — not dropped.
	w, x := vkMatMulTensors(t, F32)
	beforeOverflow := rec.Snapshot().KernelOverflow
	beforeKeys := len(rec.Snapshot().KernelKeys)
	recordVulkanMatMulKernel(vulkanKernelBackendName, w, x, time.Now(), vulkanKernelTiming{})
	after := rec.Snapshot()
	if after.KernelOverflow != beforeOverflow+1 {
		t.Fatalf("overflow = %d after a dispatch past the bound; want %d", after.KernelOverflow, beforeOverflow+1)
	}
	if len(after.KernelKeys) != beforeKeys {
		t.Fatalf("key count grew from %d to %d past the bound", beforeKeys, len(after.KernelKeys))
	}
	if after.KernelEvents != uint64(stepobs.MaxKernelKeys)+2 {
		t.Fatalf("KernelEvents = %d; every event, folded or not, must be counted", after.KernelEvents)
	}
}
