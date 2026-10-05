package stepobs

import (
	"io"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// kernelKey is the bounded label set of one kernel series. All three members are
// free-form strings minted by the compute layer per GEMM, which is exactly why the key
// set is capped.
type kernelKey struct {
	kernel      string
	backend     string
	timerDomain string
}

// kernelOverflowKey is the single fixed key every rejected event folds into. It is a
// constant, never the offending string, so overflow cannot itself grow the cardinality.
var kernelOverflowKey = kernelKey{
	kernel:      OverflowKernelLabel,
	backend:     OverflowBackendLabel,
	timerDomain: OverflowTimerDomainLabel,
}

func (k kernelKey) render() string {
	return k.kernel + "|" + k.backend + "|" + k.timerDomain
}

func (k kernelKey) counterLabels() string {
	return "kernel=\"" + quoteLabel(k.kernel) + "\",backend=\"" + quoteLabel(k.backend) + "\""
}

func (k kernelKey) histogramLabels() string {
	return k.counterLabels() + ",timer_domain=\"" + quoteLabel(k.timerDomain) + "\""
}

type kernelAgg struct {
	calls   uint64
	seconds *histogram
}

// kernelRegistry folds kernel events into the bounded key set. overflowAgg is held
// separately from byKey rather than inserted as a key, so the bound applies to REAL keys
// only and the overflow series is distinguishable from a real kernel that happens to be
// named "overflow".
type kernelRegistry struct {
	byKey map[kernelKey]*kernelAgg
	keys  []kernelKey // first-seen order, so render output is stable across scrapes

	overflowAgg    *kernelAgg // lazily created, on the first rejected event
	overflowEvents uint64

	events uint64 // every event folded, INCLUDING the ones that overflowed
}

func (kr *kernelRegistry) init() {
	// The overflow slot is created lazily, on the first capped event, so a healthy
	// registry renders no overflow series at all.
	kr.byKey = make(map[kernelKey]*kernelAgg, MaxKernelKeys)
}

func (kr *kernelRegistry) observe(e computetrace.Event) {
	k := kernelKey{
		kernel:      normalizeLabel(e.Kernel, "unknown"),
		backend:     normalizeLabel(e.Backend, "unknown"),
		timerDomain: normalizeLabel(e.TimerDomain, "unknown"),
	}
	agg := kr.byKey[k]
	if agg == nil {
		if len(kr.keys) >= MaxKernelKeys {
			// The bound is already reached: fold this event into the shared overflow
			// slot instead of allocating the (MaxKernelKeys+1)th key.
			kr.overflowEvents++
			agg = kr.overflown()
		} else {
			agg = &kernelAgg{seconds: newHistogram(secondsBuckets)}
			kr.byKey[k] = agg
			kr.keys = append(kr.keys, k)
		}
	}
	agg.calls++
	// DurationNS is the duration THE PRODUCER measured, and timer_domain names the clock
	// it read. The device backends put a CUDA-event / command-buffer measurement there
	// and the host backends a monotonic wall reading, so one histogram plus the domain
	// label is honest; inventing a second series from DeviceDurationNS would imply a
	// second measurement this seam never took.
	agg.seconds.observe(seconds(time.Duration(e.DurationNS)))
	kr.events++
}

func (kr *kernelRegistry) overflown() *kernelAgg {
	if kr.overflowAgg == nil {
		kr.overflowAgg = &kernelAgg{seconds: newHistogram(secondsBuckets)}
	}
	return kr.overflowAgg
}

func (kr *kernelRegistry) capped() bool { return len(kr.keys) >= MaxKernelKeys }

// series is one renderable kernel series in stable order: real keys first-seen order,
// then the overflow slot when it exists.
func (kr *kernelRegistry) series() []kernelKey {
	out := make([]kernelKey, 0, len(kr.keys)+1)
	out = append(out, kr.keys...)
	if kr.overflowAgg != nil {
		out = append(out, kernelOverflowKey)
	}
	return out
}

func (kr *kernelRegistry) agg(k kernelKey) *kernelAgg {
	if k == kernelOverflowKey && kr.overflowAgg != nil {
		return kr.overflowAgg
	}
	return kr.byKey[k]
}

func (kr *kernelRegistry) write(w io.Writer) {
	keys := kr.series()

	helpType(w, MetricKernelCalls,
		"Kernel invocations observed by the sub-kernel trace seam, by bounded (kernel, backend) key. A missing (kernel, backend) pair means the producer never reported it, not that zero calls happened.",
		"counter")
	for _, k := range keys {
		writeString(w, MetricKernelCalls+"{"+k.counterLabels()+"} "+strconv.FormatUint(kr.agg(k).calls, 10)+"\n")
	}

	helpType(w, MetricKernelSeconds,
		"Duration of one observed kernel call in seconds, by bounded (kernel, backend, timer_domain). timer_domain names the clock the producer measured on (host_monotonic, cuda_event, metal_command_buffer).",
		"histogram")
	for _, k := range keys {
		kr.agg(k).seconds.write(w, MetricKernelSeconds, k.histogramLabels())
	}

	observed := 0
	if kr.events > 0 {
		observed = 1
	}
	helpType(w, MetricKernelObserved,
		"1 only when a kernel observer is genuinely attached AND fed at least one real kernel event; 0 means the sub-kernel producer is ABSENT, never that zero kernels ran. A device backend reports only while compute tracing is enabled, so 0 with no CPU-reference series is the expected reading on a device-only serve.",
		"gauge")
	writeString(w, MetricKernelObserved+" "+strconv.Itoa(observed)+"\n")

	capped := 0
	if kr.capped() {
		capped = 1
	}
	helpType(w, MetricKernelKeysCapped,
		"1 when the bounded (kernel, backend, timer_domain) key set has hit its bound (further distinct keys fold into the overflow key, never grow unbounded), 0 otherwise.",
		"gauge")
	writeString(w, MetricKernelKeysCapped+" "+strconv.Itoa(capped)+"\n")

	helpType(w, MetricKernelEventOverflowTotal,
		"Kernel events folded into the overflow key because a new (kernel, backend, timer_domain) key would exceed the bound.",
		"counter")
	writeString(w, MetricKernelEventOverflowTotal+" "+strconv.FormatUint(kr.overflowEvents, 10)+"\n")
}

// ObserveKernel folds one compute event into the kernel registry. It is the fan-out
// target installed on computetrace.SetObserver by Attach, so every GEMM the compute
// layer already records reaches /metrics without internal/compute being touched.
func (r *Recorder) ObserveKernel(e computetrace.Event) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kernel.observe(e)
}
