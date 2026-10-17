package compute

import (
	"regexp"
	"sort"
	"sync"
	"sync/atomic"
)

// DeviceWork is one backend's cumulative, process-lifetime device-work counters.
// SubmitWaitNS is host wall from queue submit to fence signal: the time the host
// observed the device busy with submitted work, not a device-clock reading.
type DeviceWork struct {
	Backend       string
	Dispatches    uint64
	ReadbackBytes uint64
	SubmitWaitNS  uint64
	// SubmitWaitObserved is true only when this reader accounts for every
	// relevant submission path throughout the counter lifetime. False means
	// unavailable, even when SubmitWaitNS is nonzero. The current Vulkan reader
	// leaves this false: its pipelined submits/fence waits are not all covered.
	SubmitWaitObserved bool
}

var deviceWorkReaders struct {
	mu        sync.RWMutex
	byBackend map[string]func() DeviceWork
}

// RegisterDeviceWorkReader installs the counter reader for an initialized
// backend. A backend that never registers has no dispatch series: absent, not 0.
// read must be lock-free; it runs on scrapes and around every decode loop.
// The returned func removes the reader (tests; a live backend never does).
func RegisterDeviceWorkReader(backend string, read func() DeviceWork) (unregister func()) {
	if backend == "" || read == nil {
		return func() {}
	}
	deviceWorkReaders.mu.Lock()
	defer deviceWorkReaders.mu.Unlock()
	if deviceWorkReaders.byBackend == nil {
		deviceWorkReaders.byBackend = make(map[string]func() DeviceWork)
	}
	deviceWorkReaders.byBackend[backend] = read
	return func() {
		deviceWorkReaders.mu.Lock()
		delete(deviceWorkReaders.byBackend, backend)
		deviceWorkReaders.mu.Unlock()
	}
}

// DeviceWorkSnapshot reads every registered backend, sorted by backend name.
func DeviceWorkSnapshot() []DeviceWork {
	deviceWorkReaders.mu.RLock()
	out := make([]DeviceWork, 0, len(deviceWorkReaders.byBackend))
	for backend, read := range deviceWorkReaders.byBackend {
		w := read()
		w.Backend = backend
		out = append(out, w)
	}
	deviceWorkReaders.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Backend < out[j].Backend })
	return out
}

// HostFallbackOtherStage absorbs stage names past the label cap or outside the
// label grammar, so a caller can never mint unbounded series.
const HostFallbackOtherStage = "other"

const maxHostFallbackStages = 32

var hostFallbackStageName = regexp.MustCompile(`^[a-z][a-z0-9_]{0,47}$`)

var hostFallbacks struct {
	mu      sync.RWMutex
	byStage map[string]*atomic.Uint64
}

// HostFallbackCount is one stage's cumulative host-fallback count.
type HostFallbackCount struct {
	Stage string
	Count uint64
}

// DeclareHostFallbackStage exposes a wired stage at 0 before it first fires, so
// "wired and never fell back" reads differently from "not instrumented".
func DeclareHostFallbackStage(stage string) { hostFallbackCounter(stage) }

// CountHostFallback records one decode-path stage that ran on the host where a
// device path was expected. Stage is a bounded snake_case label.
func CountHostFallback(stage string) { hostFallbackCounter(stage).Add(1) }

func hostFallbackCounter(stage string) *atomic.Uint64 {
	if !hostFallbackStageName.MatchString(stage) {
		stage = HostFallbackOtherStage
	}
	hostFallbacks.mu.RLock()
	c := hostFallbacks.byStage[stage]
	hostFallbacks.mu.RUnlock()
	if c != nil {
		return c
	}
	hostFallbacks.mu.Lock()
	defer hostFallbacks.mu.Unlock()
	if hostFallbacks.byStage == nil {
		hostFallbacks.byStage = make(map[string]*atomic.Uint64)
	}
	if c = hostFallbacks.byStage[stage]; c != nil {
		return c
	}
	if len(hostFallbacks.byStage) >= maxHostFallbackStages {
		stage = HostFallbackOtherStage
		if c = hostFallbacks.byStage[stage]; c != nil {
			return c
		}
	}
	c = new(atomic.Uint64)
	hostFallbacks.byStage[stage] = c
	return c
}

// HostFallbackCounts returns every declared or fired stage, sorted by stage.
func HostFallbackCounts() []HostFallbackCount {
	hostFallbacks.mu.RLock()
	out := make([]HostFallbackCount, 0, len(hostFallbacks.byStage))
	for stage, c := range hostFallbacks.byStage {
		out = append(out, HostFallbackCount{Stage: stage, Count: c.Load()})
	}
	hostFallbacks.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Stage < out[j].Stage })
	return out
}
