package compute

import (
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

// The CUDA kernel profile (FAK_CUDA_TIMESTAMP_PROFILE=1, the CUDA peer of
// FAK_VULKAN_TIMESTAMP_PROFILE). Every device op holds cudaMu, and all of its
// device work runs on the one in-order g_stream, so each cudaMu critical
// section is bracketed by a pair of CUDA events recorded on g_stream. A row is
// one backend op section (named by the function that took cudaMu), not one
// device kernel: GX and GY are 0, and a section that launches several kernels
// books their combined interval. Work a section puts on another stream or
// device is outside its interval.
//
// The interval is device-clock time between two stream positions. It is the
// section's own device time only while the stream stays busy across it: a
// section that begins on an idle stream would also book the host's launch
// latency, and one whose work finished before its end event was recorded would
// book host time after it. Both are detected when the event is recorded and
// counted Truncated instead of timed. Two cases are not detected and keep their
// idle in the interval: a stream that drains and refills inside one section,
// and one that drains between the idle query and the event record. In a
// host-bound decode most sections can end Truncated, so read Ops against
// Ops+Truncated as the timed fraction before trusting GPUNS as coverage.
//
// Field mapping: GPUNS sums the timed sections (disjoint, since g_stream is in
// order), and the rows sum to it. WallNS is the device-clock span of each
// drained batch, first timed start to last timed end, so it also covers the
// excluded sections and idle between sections. Vulkan's WallNS is host wall per
// submit instead, so compare GPUNS against a host window, never WallNS across
// backends. UntimedCompute counts sections captured into a CUDA graph.
// BarriersElided and UntimedTransfer stay 0.
//
// This file is the build-tag-free fold; the event pool and its cgo clock live
// behind -tags cuda (cuda.go, cuda_kernels.cu).

// cudaTimestampProfileEnv opts in with the same exact "1" grammar as the Vulkan
// control; any other value leaves the profile off.
const cudaTimestampProfileEnv = "FAK_CUDA_TIMESTAMP_PROFILE"

// cudaProfilePairs bounds pending event pairs; a full pool is drained (one
// host wait on the newest event) before the next section is timed. The waits
// spin (default event flags), so a drain counts as process CPU.
const cudaProfilePairs = 1024

func cudaTimestampProfileEnabled(value string) bool { return value == "1" }

// cudaProfileMark names one event in the pool: the start or end of a pair.
type cudaProfileMark struct {
	slot int
	end  bool
}

// cudaProfileClock is the device-clock seam. record enqueues the mark's event
// on g_stream and reports whether the stream had no work left (it then waits
// for the event, so the stream is idle again); elapsedNS waits for to and
// returns the device time from from to to; capturing reports an open stream
// capture, where an event is a graph dependency rather than a timing point.
type cudaProfileClock interface {
	record(m cudaProfileMark) (idle bool, err error)
	elapsedNS(from, to cudaProfileMark) (float64, error)
	capturing() bool
}

type cudaProfileRow struct {
	calls uint64
	ns    float64
}

// cudaKernelProfiler is plain state guarded by the cudaDeviceMutex that owns
// it: sections run while that mutex is held, and the reader takes it too.
type cudaKernelProfiler struct {
	clock  cudaProfileClock
	pairs  int
	usable bool // fixed at construction

	pending []string // section name of each recorded pair, in stream order

	open        bool
	openName    string
	openTimed   bool
	openCapture bool
	openIdle    bool

	batches, ops, truncated, untimedCompute uint64
	gpuNS, wallNS                           float64
	rows                                    map[string]*cudaProfileRow
}

// newCUDAKernelProfiler builds the fold. usable is false when the event pool
// could not be created: the reader then reports the profile unavailable rather
// than a zero profile.
func newCUDAKernelProfiler(clock cudaProfileClock, pairs int, usable bool) *cudaKernelProfiler {
	if clock == nil || pairs <= 0 {
		usable = false
	}
	return &cudaKernelProfiler{clock: clock, pairs: pairs, usable: usable, rows: make(map[string]*cudaProfileRow)}
}

// begin opens one section. A section that starts inside a stream capture is
// untimed: its work runs at graph replay, which no event pair brackets.
func (p *cudaKernelProfiler) begin(name string) {
	if !p.usable {
		return
	}
	p.open, p.openName, p.openTimed, p.openCapture, p.openIdle = true, name, false, false, false
	if p.clock.capturing() {
		p.openCapture = true
		return
	}
	if len(p.pending) >= p.pairs {
		p.drain()
	}
	idle, err := p.clock.record(cudaProfileMark{slot: len(p.pending)})
	if err != nil {
		return
	}
	p.openTimed, p.openIdle = true, idle
}

// end closes the open section. It is untimed when a capture opened inside it,
// since its end position then belongs to the graph, and truncated when an
// event failed or either boundary found the stream idle.
func (p *cudaKernelProfiler) end() {
	if !p.open {
		return
	}
	p.open = false
	switch {
	case p.openCapture:
		p.untimedCompute++
		return
	case !p.openTimed:
		p.truncated++
		return
	case p.clock.capturing():
		p.untimedCompute++
		return
	}
	idle, err := p.clock.record(cudaProfileMark{slot: len(p.pending), end: true})
	if err != nil || idle || p.openIdle {
		p.truncated++
		return
	}
	p.pending = append(p.pending, p.openName)
}

// drain folds every pending pair. The batch span (first start to last end) is
// read first and waits for the stream; if it fails, the whole batch is lost.
// A pair whose own interval fails stays out of the rows and counts truncated,
// so the rows and GPUNS add the same intervals.
func (p *cudaKernelProfiler) drain() {
	n := len(p.pending)
	if n == 0 {
		return
	}
	defer func() { p.pending = p.pending[:0] }()
	span, err := p.clock.elapsedNS(cudaProfileMark{slot: 0}, cudaProfileMark{slot: n - 1, end: true})
	if err != nil || !finiteNonnegative(span) {
		p.truncated += uint64(n)
		return
	}
	for i, name := range p.pending {
		ns, err := p.clock.elapsedNS(cudaProfileMark{slot: i}, cudaProfileMark{slot: i, end: true})
		if err != nil || !finiteNonnegative(ns) {
			p.truncated++
			continue
		}
		r := p.rows[name]
		if r == nil {
			r = &cudaProfileRow{}
			p.rows[name] = r
		}
		r.calls++
		r.ns += ns
		p.gpuNS += ns
		p.ops++
	}
	p.batches++
	p.wallNS += span
}

// snapshot drains and returns the cumulative profile. ok is false when the
// event pool is unusable, and while g_stream is capturing, when waiting on an
// event is not allowed. A profile that has timed nothing yet is a valid zero
// baseline.
func (p *cudaKernelProfiler) snapshot() (KernelProfile, bool) {
	if p == nil || !p.usable || p.clock.capturing() {
		return KernelProfile{}, false
	}
	p.drain()
	out := KernelProfile{
		Batches:        p.batches,
		Ops:            p.ops,
		Truncated:      p.truncated,
		UntimedCompute: p.untimedCompute,
		GPUNS:          p.gpuNS,
		WallNS:         p.wallNS,
		Rows:           make([]KernelRow, 0, len(p.rows)),
	}
	for name, r := range p.rows {
		out.Rows = append(out.Rows, KernelRow{Kernel: name, Calls: r.calls, NS: r.ns})
	}
	sortKernelRows(out.Rows)
	return out, true
}

// cudaDeviceMutex is cudaMu's type: a sync.Mutex whose critical sections are
// the profile's timed spans once a usable profiler is installed. With none
// installed, Lock and Unlock each add one atomic load to sync.Mutex. The reader
// locks the embedded Mutex directly so it never times itself.
type cudaDeviceMutex struct {
	sync.Mutex
	profile atomic.Pointer[cudaKernelProfiler]
	names   map[uintptr]string // caller PC -> section name, guarded by Mutex
}

func (m *cudaDeviceMutex) Lock() {
	m.Mutex.Lock()
	if p := m.profile.Load(); p != nil && p.usable {
		p.begin(m.sectionName())
	}
}

func (m *cudaDeviceMutex) Unlock() {
	if p := m.profile.Load(); p != nil {
		p.end()
	}
	m.Mutex.Unlock()
}

// readProfile returns the installed profiler's snapshot under the mutex.
func (m *cudaDeviceMutex) readProfile() (KernelProfile, bool) {
	p := m.profile.Load()
	if p == nil {
		return KernelProfile{}, false
	}
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	return p.snapshot()
}

// sectionName resolves the function that called Lock. Called with Mutex held.
func (m *cudaDeviceMutex) sectionName() string {
	var pcs [1]uintptr
	if runtime.Callers(3, pcs[:]) == 0 {
		return "unknown"
	}
	if name, ok := m.names[pcs[0]]; ok {
		return name
	}
	frame, _ := runtime.CallersFrames(pcs[:]).Next()
	name := cudaProfileSectionName(frame.Function)
	if m.names == nil {
		m.names = make(map[uintptr]string)
	}
	m.names[pcs[0]] = name
	return name
}

// cudaProfileCopyNames maps sections that move bytes onto the Vulkan transfer
// pseudo-kernel names, so the op-class fold files them as copy; MatMul is the P=1
// GEMV, which the Vulkan rows call matvec.
var cudaProfileCopyNames = map[string]string{
	"matmul":      "matvec",
	"read":        "readback",
	"upload":      "h2d_upload",
	"uploadclass": "h2d_upload",
	"clonetensor": "d2d_copy",
}

// cudaProfileSectionName turns a runtime function name into a stable,
// lowercase section name: the method name for a cudaBackend method, the
// receiver type and method otherwise, closures folded into their parent.
func cudaProfileSectionName(fn string) string {
	if i := strings.LastIndex(fn, "/"); i >= 0 {
		fn = fn[i+1:]
	}
	if i := strings.Index(fn, "."); i >= 0 {
		fn = fn[i+1:]
	}
	parts := strings.Split(fn, ".")
	for len(parts) > 1 && cudaProfileClosureSuffix(parts[len(parts)-1]) {
		parts = parts[:len(parts)-1]
	}
	if len(parts) > 2 {
		parts = parts[:2]
	}
	var name string
	switch {
	case len(parts) == 2:
		recv := strings.Trim(parts[0], "(*)")
		if recv == "cudaBackend" {
			name = parts[1]
		} else {
			name = recv + "_" + parts[1]
		}
	case len(parts) == 1 && parts[0] != "":
		name = parts[0]
	default:
		return "unknown"
	}
	name = strings.ToLower(name)
	if mapped, ok := cudaProfileCopyNames[name]; ok {
		return mapped
	}
	return name
}

// cudaProfileClosureSuffix matches the runtime's closure segments: func1, 2.
func cudaProfileClosureSuffix(s string) bool {
	s = strings.TrimPrefix(s, "func")
	return s != "" && strings.Trim(s, "0123456789") == ""
}
