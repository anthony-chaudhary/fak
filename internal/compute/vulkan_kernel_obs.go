package compute

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// Timer domains the Vulkan kernel seam may report. Both are fixed constants, so
// neither can mint a per-dispatch label or grow the bounded
// (kernel, backend, timer_domain) key set in internal/stepobs.
const (
	// VulkanTimerDomainDevice names the RADV/Vulkan performance-query nanoseconds
	// counter: a value the DEVICE measured for this dispatch.
	VulkanTimerDomainDevice = "vulkan_performance_query"
	// VulkanTimerDomainUnavailable is the typed-unavailable marker. It is NOT a
	// zero-duration measurement and NOT a host-monotonic proxy: it says the
	// device timer could not be read for this dispatch, which is the only honest
	// value when the shim exposes no per-dispatch counter read.
	VulkanTimerDomainUnavailable = "vulkan_performance_query_unavailable"
)

// vulkanKernelDeviceLabel is the fixed device string on every Vulkan kernel event.
// It is a constant because a per-dispatch or per-adapter string would make the
// event's device field an unbounded label source.
const vulkanKernelDeviceLabel = "vulkan:0"

// vulkanKernelPhase is both the Event.Phase value and the timestamped phase the
// performance-query readings are attributed to. One constant so the phase string
// cannot drift between the two.
const vulkanKernelPhase = "kernel"

// vulkanKernelTiming is one dispatch's device-timing verdict. Available=false
// means no device nanoseconds reading was observed, so DurationNS is not a
// measurement and the event MUST carry the unavailable timer domain.
type vulkanKernelTiming struct {
	DurationNS int64
	Counter    string
	Available  bool
}

// vulkanKernelTimingFrom resolves the device timing for one timestamped phase
// against a fail-closed performance-query observation.
//
// It is deliberately narrow: the observation's own Supported bit plus a
// nanoseconds-unit, AVAILABLE reading for this phase is the only accepted
// source. A supported query whose counter was not sampled for this phase, a
// non-nanoseconds unit, and an unsupported device all resolve to the typed
// unavailable verdict. No host clock is ever substituted — a host-monotonic
// number under a device timer domain is fabricated telemetry, which is the exact
// failure this seam exists to rule out.
func vulkanKernelTimingFrom(observation PhasePerformanceQueryObservation, phase string, samples map[string]uint64) vulkanKernelTiming {
	if !observation.Supported {
		return vulkanKernelTiming{}
	}
	for _, reading := range observation.ReadingsForPhase(phase, samples) {
		if !reading.Available || reading.Unit != PhaseCounterUnitNanoseconds {
			continue
		}
		return vulkanKernelTiming{DurationNS: int64(reading.Value), Counter: reading.Counter, Available: true}
	}
	return vulkanKernelTiming{}
}

// timerDomain names the clock this verdict was measured on: the device
// performance query when a reading exists, the explicit unavailable marker when
// it does not. It never returns host_monotonic.
func (t vulkanKernelTiming) timerDomain() string {
	if t.Available {
		return VulkanTimerDomainDevice
	}
	return VulkanTimerDomainUnavailable
}

// vulkanKernelEvent builds the kernel event for one Vulkan dispatch. The
// Operation/Phase/Kernel shape is identical to the CPU-reference site
// (cpuref.go MatMul), so the two emitters agree on the same kernel names.
func vulkanKernelEvent(backendName string, w, x Tensor, started time.Time, timing vulkanKernelTiming) computetrace.Event {
	return computetrace.Event{
		Operation:        "matmul",
		Phase:            vulkanKernelPhase,
		Backend:          backendName,
		Device:           vulkanKernelDeviceLabel,
		Kernel:           w.Dtype.String() + "_matmul",
		Route:            "device",
		StartedAt:        started.UTC(),
		DurationNS:       timing.DurationNS,
		TimerDomain:      timing.timerDomain(),
		InputDType:       x.Dtype.String(),
		WeightDType:      w.Dtype.String(),
		OutputDType:      F32.String(),
		BytesRead:        int64(w.Numel()*w.Dtype.Bytes() + x.Numel()*x.Dtype.Bytes()),
		Shapes:           [][]int{w.Shape, x.Shape},
		ProvenanceDigest: computetrace.Digest(backendName, w.Dtype.String(), "matmul"),
	}
}

// recordVulkanMatMulKernel is the single Vulkan kernel emission point. It is
// gated on computetrace.Emitting() — NOT computetrace.Enabled() — so a metrics
// observer alone is enough to be fed, while the bounded activation-sample
// capture that shares Enabled() stays off the decode hot path.
func recordVulkanMatMulKernel(backendName string, w, x Tensor, started time.Time, timing vulkanKernelTiming) {
	if !computetrace.Emitting() {
		return
	}
	computetrace.Record(vulkanKernelEvent(backendName, w, x, started, timing))
}
