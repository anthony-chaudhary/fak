// Package runtimeobs reads a fixed, availability-checked vector of Go
// runtime/metrics into a stable, versioned receipt (#10182).
//
// Aggregate goroutine/heap numbers cannot tell runnable saturation from blocked
// queues, thread growth, scheduler delay, or GC pressure, and a single "memory"
// number cannot tell bytes the Go GC manages from everything else the runtime
// mapped, or from what the OS actually holds resident (off-heap mmap/cgo
// payload). The receipt keeps those classes apart: Memory.HeapObjectsBytes is
// Go-heap bytes, Memory.TotalMappedBytes is every byte the Go runtime mapped,
// and Memory.ProcessRSSBytes is the OS-reported resident set.
//
// Contract:
//   - One runtime/metrics.Read of a reusable sample vector per Sample. Never
//     runtime.ReadMemStats, which stops the world.
//   - Every metric is resolved once, at New, against the runtime's catalog: a
//     name the catalog lacks falls through to its renamed aliases; a name the
//     catalog declares with another kind is refused. Per sample, a value the
//     reader returns as KindBad, with the wrong kind, or non-finite is refused.
//     A refused field marshals as JSON null and is named in
//     Receipt.Unavailable — never a silent zero.
//   - Cumulative counters and histograms are monotonic only within one process
//     epoch and one Go toolchain. Diff refuses to subtract across either
//     boundary: Go 1.26 changed the /sched/latencies:seconds sampling
//     population, so a cross-version latency comparison is not a conclusion.
//   - The Go 1.26 goroutine-state gauges are approximate and non-additive
//     (runtime/metrics docs): running+runnable+waiting+not_in_go need not equal
//     goroutines.
//
// Stdlib-only primitive leaf. Process RSS is not a runtime/metrics quantity;
// the caller injects a platform reader with WithRSSReader.
package runtimeobs

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/metrics"
	"sync"
	"time"
)

const (
	// Schema versions Receipt. Additive fields keep v1; a renamed or
	// re-meaning'd field bumps it.
	Schema = "fak.go-runtime.v1"
	// DeltaSchema versions Delta.
	DeltaSchema = "fak.go-runtime-delta.v1"
	// CPUBudgetFraction is the predeclared ceiling on synchronous collection
	// wall time as a fraction of sampler cadence. For the caller thread, elapsed
	// time conservatively bounds CPU time spent in Sample; it does not measure
	// secondary runtime work or total process CPU (#10182).
	CPUBudgetFraction = 0.01
)

// WithinBudget reports whether synchronous elapsed cost per sample stays
// below the cadence fraction. This bounds the caller thread's CPU use during
// Sample, but does not measure secondary runtime work or total process CPU.
func WithinBudget(cost, cadence time.Duration) bool {
	if cadence <= 0 {
		return false
	}
	return float64(cost) < CPUBudgetFraction*float64(cadence)
}

// Reason is the closed vocabulary for an unavailable receipt field.
type Reason string

const (
	// ReasonUnsupported: neither the canonical name nor any alias is in the
	// runtime's metric catalog (older/newer toolchain, or a removed metric).
	ReasonUnsupported Reason = "unsupported"
	// ReasonWrongKind: the catalog or the reader reported a value kind other
	// than the one this receipt's contract declares.
	ReasonWrongKind Reason = "wrong_kind"
	// ReasonBadValue: the reader returned KindBad, a nil histogram, or a
	// non-finite float.
	ReasonBadValue Reason = "bad_value"
	// ReasonNoReader: the field needs a platform reader that was not injected.
	ReasonNoReader Reason = "no_reader"
	// ReasonReadFailed: the injected platform reader could not read.
	ReasonReadFailed Reason = "read_failed"
)

// Unavailable names one receipt field that carries no value this sample, and why.
type Unavailable struct {
	Field  string `json:"field"`
	Reason Reason `json:"reason"`
	// Metric is the runtime/metrics name that was consulted (the canonical
	// name when none resolved).
	Metric string `json:"metric,omitempty"`
}

// Alias records a field resolved through a renamed (non-canonical) metric name.
type Alias struct {
	Field  string `json:"field"`
	Metric string `json:"metric"`
}

// Sched is the scheduler slice of the receipt.
type Sched struct {
	GOMAXPROCS         U64 `json:"gomaxprocs"`
	Goroutines         U64 `json:"goroutines"`
	GoroutinesRunning  U64 `json:"goroutines_running"`
	GoroutinesRunnable U64 `json:"goroutines_runnable"`
	GoroutinesWaiting  U64 `json:"goroutines_waiting"`
	GoroutinesNotInGo  U64 `json:"goroutines_not_in_go"`
	// GoroutinesCreated is cumulative since process start.
	GoroutinesCreated U64 `json:"goroutines_created"`
	// Threads is the count of Go-runtime-owned OS threads.
	Threads U64 `json:"threads"`
	// Latencies is the cumulative runnable-to-running delay histogram.
	Latencies Hist `json:"latencies_seconds"`
	// LatenciesSummary digests Latencies (count and conservative quantiles).
	LatenciesSummary Summary `json:"latencies_summary"`
}

// GC is the collector slice of the receipt.
type GC struct {
	Cycles        U64 `json:"cycles"`
	ForcedCycles  U64 `json:"forced_cycles"`
	HeapGoalBytes U64 `json:"heap_goal_bytes"`
	HeapLiveBytes U64 `json:"heap_live_bytes"`
	// MemoryLimitBytes is GOMEMLIMIT; math.MaxInt64 means no limit, which
	// MemoryLimitUnlimited states explicitly.
	MemoryLimitBytes     U64  `json:"memory_limit_bytes"`
	MemoryLimitUnlimited bool `json:"memory_limit_unlimited,omitempty"`
	// GOGCPercent is GOGC. The runtime reports GOGC=off as uint64(-1), which
	// GOGCOff states explicitly so no consumer reads it as a huge percentage.
	GOGCPercent U64  `json:"gogc_percent"`
	GOGCOff     bool `json:"gogc_off,omitempty"`
	// HeapAllocsBytes is cumulative bytes allocated on the Go heap.
	HeapAllocsBytes U64 `json:"heap_allocs_bytes"`
	// Pauses is the cumulative stop-the-world GC pause histogram.
	Pauses        Hist    `json:"pauses_seconds"`
	PausesSummary Summary `json:"pauses_summary"`
}

// CPU is the runtime's /cpu/classes estimate, cumulative seconds. These are
// runtime estimates comparable only with each other: Total is GOMAXPROCS
// integrated over wall time, so GC/Total is GC's share of available CPU.
type CPU struct {
	TotalSeconds           F64 `json:"total_seconds"`
	UserSeconds            F64 `json:"user_seconds"`
	IdleSeconds            F64 `json:"idle_seconds"`
	GCTotalSeconds         F64 `json:"gc_total_seconds"`
	GCMarkAssistSeconds    F64 `json:"gc_mark_assist_seconds"`
	GCMarkDedicatedSeconds F64 `json:"gc_mark_dedicated_seconds"`
	GCMarkIdleSeconds      F64 `json:"gc_mark_idle_seconds"`
	GCPauseSeconds         F64 `json:"gc_pause_seconds"`
}

// Memory separates Go-heap bytes from all runtime-mapped bytes from process RSS.
type Memory struct {
	// HeapObjectsBytes is the Go heap: live objects plus dead ones not yet swept.
	HeapObjectsBytes  U64 `json:"heap_objects_bytes"`
	HeapUnusedBytes   U64 `json:"heap_unused_bytes"`
	HeapFreeBytes     U64 `json:"heap_free_bytes"`
	HeapReleasedBytes U64 `json:"heap_released_bytes"`
	HeapStacksBytes   U64 `json:"heap_stacks_bytes"`
	OSStacksBytes     U64 `json:"os_stacks_bytes"`
	// TotalMappedBytes is every byte the Go runtime mapped (heap, stacks,
	// metadata, other). Memory the process maps outside the Go runtime
	// (mmap'd weights, cgo/accelerator allocations) is not in it.
	TotalMappedBytes U64 `json:"total_mapped_bytes"`
	// ProcessRSSBytes is the OS-reported resident set of the whole process.
	ProcessRSSBytes U64 `json:"process_rss_bytes"`
}

// Receipt is one typed sample of the Go runtime vector.
type Receipt struct {
	Schema    string `json:"schema"`
	GoVersion string `json:"go_version"`
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
	PID       int    `json:"pid"`
	// Epoch identifies the process lifetime cumulative series are monotonic
	// within; a different Epoch is a reset.
	Epoch             string `json:"epoch"`
	Seq               uint64 `json:"seq"`
	SampledAtUnixNano int64  `json:"sampled_at_unix_nano"`
	NumCPU            int    `json:"num_cpu"`

	Sched  Sched  `json:"sched"`
	GC     GC     `json:"gc"`
	CPU    CPU    `json:"cpu"`
	Memory Memory `json:"memory"`

	Unavailable []Unavailable `json:"unavailable,omitempty"`
	Aliased     []Alias       `json:"aliased,omitempty"`
}

// Reader is the injectable runtime/metrics.Read seam.
type Reader func([]metrics.Sample)

// Catalog is the injectable runtime/metrics.All seam.
type Catalog func() []metrics.Description

// spec binds one receipt field to its metric names (canonical first, then
// renamed aliases) and the value kind its contract declares.
type spec struct {
	field string
	names []string
	kind  metrics.ValueKind
	u64   func(*Receipt) *U64
	f64   func(*Receipt) *F64
	hist  func(*Receipt) *Hist
}

func u64Spec(field, name string, at func(*Receipt) *U64) spec {
	return spec{field: field, names: []string{name}, kind: metrics.KindUint64, u64: at}
}

func f64Spec(field, name string, at func(*Receipt) *F64) spec {
	return spec{field: field, names: []string{name}, kind: metrics.KindFloat64, f64: at}
}

func histSpec(field string, names []string, at func(*Receipt) *Hist) spec {
	return spec{field: field, names: names, kind: metrics.KindFloat64Histogram, hist: at}
}

// specs is the fixed metric vector. Order is the receipt's field order and is
// part of the stable contract (Unavailable lists follow it).
var specs = []spec{
	u64Spec("sched.gomaxprocs", "/sched/gomaxprocs:threads", func(r *Receipt) *U64 { return &r.Sched.GOMAXPROCS }),
	u64Spec("sched.goroutines", "/sched/goroutines:goroutines", func(r *Receipt) *U64 { return &r.Sched.Goroutines }),
	u64Spec("sched.goroutines_running", "/sched/goroutines/running:goroutines", func(r *Receipt) *U64 { return &r.Sched.GoroutinesRunning }),
	u64Spec("sched.goroutines_runnable", "/sched/goroutines/runnable:goroutines", func(r *Receipt) *U64 { return &r.Sched.GoroutinesRunnable }),
	u64Spec("sched.goroutines_waiting", "/sched/goroutines/waiting:goroutines", func(r *Receipt) *U64 { return &r.Sched.GoroutinesWaiting }),
	u64Spec("sched.goroutines_not_in_go", "/sched/goroutines/not-in-go:goroutines", func(r *Receipt) *U64 { return &r.Sched.GoroutinesNotInGo }),
	u64Spec("sched.goroutines_created", "/sched/goroutines-created:goroutines", func(r *Receipt) *U64 { return &r.Sched.GoroutinesCreated }),
	u64Spec("sched.threads", "/sched/threads/total:threads", func(r *Receipt) *U64 { return &r.Sched.Threads }),
	histSpec("sched.latencies_seconds", []string{"/sched/latencies:seconds"}, func(r *Receipt) *Hist { return &r.Sched.Latencies }),

	u64Spec("gc.cycles", "/gc/cycles/total:gc-cycles", func(r *Receipt) *U64 { return &r.GC.Cycles }),
	u64Spec("gc.forced_cycles", "/gc/cycles/forced:gc-cycles", func(r *Receipt) *U64 { return &r.GC.ForcedCycles }),
	u64Spec("gc.heap_goal_bytes", "/gc/heap/goal:bytes", func(r *Receipt) *U64 { return &r.GC.HeapGoalBytes }),
	u64Spec("gc.heap_live_bytes", "/gc/heap/live:bytes", func(r *Receipt) *U64 { return &r.GC.HeapLiveBytes }),
	u64Spec("gc.memory_limit_bytes", "/gc/gomemlimit:bytes", func(r *Receipt) *U64 { return &r.GC.MemoryLimitBytes }),
	u64Spec("gc.gogc_percent", "/gc/gogc:percent", func(r *Receipt) *U64 { return &r.GC.GOGCPercent }),
	u64Spec("gc.heap_allocs_bytes", "/gc/heap/allocs:bytes", func(r *Receipt) *U64 { return &r.GC.HeapAllocsBytes }),
	// Go 1.22 renamed /gc/pauses:seconds to /sched/pauses/total/gc:seconds; the
	// old name is kept as an alias so an older runtime still fills the field.
	histSpec("gc.pauses_seconds", []string{"/sched/pauses/total/gc:seconds", "/gc/pauses:seconds"}, func(r *Receipt) *Hist { return &r.GC.Pauses }),

	f64Spec("cpu.total_seconds", "/cpu/classes/total:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.TotalSeconds }),
	f64Spec("cpu.user_seconds", "/cpu/classes/user:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.UserSeconds }),
	f64Spec("cpu.idle_seconds", "/cpu/classes/idle:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.IdleSeconds }),
	f64Spec("cpu.gc_total_seconds", "/cpu/classes/gc/total:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.GCTotalSeconds }),
	f64Spec("cpu.gc_mark_assist_seconds", "/cpu/classes/gc/mark/assist:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.GCMarkAssistSeconds }),
	f64Spec("cpu.gc_mark_dedicated_seconds", "/cpu/classes/gc/mark/dedicated:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.GCMarkDedicatedSeconds }),
	f64Spec("cpu.gc_mark_idle_seconds", "/cpu/classes/gc/mark/idle:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.GCMarkIdleSeconds }),
	f64Spec("cpu.gc_pause_seconds", "/cpu/classes/gc/pause:cpu-seconds", func(r *Receipt) *F64 { return &r.CPU.GCPauseSeconds }),

	u64Spec("memory.heap_objects_bytes", "/memory/classes/heap/objects:bytes", func(r *Receipt) *U64 { return &r.Memory.HeapObjectsBytes }),
	u64Spec("memory.heap_unused_bytes", "/memory/classes/heap/unused:bytes", func(r *Receipt) *U64 { return &r.Memory.HeapUnusedBytes }),
	u64Spec("memory.heap_free_bytes", "/memory/classes/heap/free:bytes", func(r *Receipt) *U64 { return &r.Memory.HeapFreeBytes }),
	u64Spec("memory.heap_released_bytes", "/memory/classes/heap/released:bytes", func(r *Receipt) *U64 { return &r.Memory.HeapReleasedBytes }),
	u64Spec("memory.heap_stacks_bytes", "/memory/classes/heap/stacks:bytes", func(r *Receipt) *U64 { return &r.Memory.HeapStacksBytes }),
	u64Spec("memory.os_stacks_bytes", "/memory/classes/os-stacks:bytes", func(r *Receipt) *U64 { return &r.Memory.OSStacksBytes }),
	u64Spec("memory.total_mapped_bytes", "/memory/classes/total:bytes", func(r *Receipt) *U64 { return &r.Memory.TotalMappedBytes }),
}

const rssField = "memory.process_rss_bytes"

// gogcOff is how /gc/gogc:percent encodes GOGC=off: the runtime widens its
// int32 percent of -1 straight to uint64.
const gogcOff = math.MaxUint64

// processEpoch is fixed at package init, so every Collector in one process
// shares it and a restarted process never reuses it.
var processEpoch = fmt.Sprintf("%d@%d", os.Getpid(), time.Now().UnixNano())

// Option configures a Collector.
type Option func(*Collector)

// WithReader replaces runtime/metrics.Read (tests and fixtures).
func WithReader(r Reader) Option {
	return func(c *Collector) {
		if r != nil {
			c.read = r
		}
	}
}

// WithCatalog replaces runtime/metrics.All, e.g. to model an older toolchain.
func WithCatalog(cat Catalog) Option {
	return func(c *Collector) {
		if cat != nil {
			c.catalog = cat
		}
	}
}

// WithRSSReader injects the platform reader for the process resident set.
func WithRSSReader(fn func() (uint64, bool)) Option {
	return func(c *Collector) { c.rss = fn }
}

// WithClock replaces time.Now for SampledAtUnixNano.
func WithClock(now func() time.Time) Option {
	return func(c *Collector) {
		if now != nil {
			c.now = now
		}
	}
}

// Collector samples the resolved vector. It is safe for concurrent use; the
// sample vector is reused across reads, so Sample serializes.
type Collector struct {
	mu      sync.Mutex
	read    Reader
	catalog Catalog
	rss     func() (uint64, bool)
	now     func() time.Time
	seq     uint64

	active  []int            // specs index per resolved sample
	samples []metrics.Sample // reusable read vector, parallel to active
	static  []Unavailable    // catalog-time refusals, fixed for the collector's life
	aliased []Alias
}

// New resolves the vector against the runtime catalog once.
func New(opts ...Option) *Collector {
	c := &Collector{read: metrics.Read, catalog: metrics.All, now: time.Now}
	for _, o := range opts {
		o(c)
	}
	kinds := make(map[string]metrics.ValueKind)
	for _, d := range c.catalog() {
		kinds[d.Name] = d.Kind
	}
	for i := range specs {
		s := &specs[i]
		name, reason := resolve(s, kinds)
		if reason != "" {
			c.static = append(c.static, Unavailable{Field: s.field, Reason: reason, Metric: name})
			continue
		}
		if name != s.names[0] {
			c.aliased = append(c.aliased, Alias{Field: s.field, Metric: name})
		}
		c.active = append(c.active, i)
		c.samples = append(c.samples, metrics.Sample{Name: name})
	}
	return c
}

// resolve returns the first catalog name of the right kind. A wrong-kind
// canonical name does not stop the alias walk: an alias of the right kind wins.
func resolve(s *spec, kinds map[string]metrics.ValueKind) (string, Reason) {
	wrong := ""
	for _, n := range s.names {
		k, ok := kinds[n]
		if !ok {
			continue
		}
		if k == s.kind {
			return n, ""
		}
		if wrong == "" {
			wrong = n
		}
	}
	if wrong != "" {
		return wrong, ReasonWrongKind
	}
	return s.names[0], ReasonUnsupported
}

// Sample reads the vector once and returns an owned receipt: histogram
// buckets are copied out of the reader's reusable backing storage.
func (c *Collector) Sample() Receipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.samples) > 0 {
		c.read(c.samples)
	}
	c.seq++
	r := Receipt{
		Schema:            Schema,
		GoVersion:         runtime.Version(),
		GOOS:              runtime.GOOS,
		GOARCH:            runtime.GOARCH,
		PID:               os.Getpid(),
		Epoch:             processEpoch,
		Seq:               c.seq,
		SampledAtUnixNano: c.now().UnixNano(),
		NumCPU:            runtime.NumCPU(),
	}
	var dyn []Unavailable
	for i, si := range c.active {
		s := &specs[si]
		if reason := decode(s, &r, c.samples[i].Value); reason != "" {
			dyn = append(dyn, Unavailable{Field: s.field, Reason: reason, Metric: c.samples[i].Name})
		}
	}
	switch {
	case c.rss == nil:
		dyn = append(dyn, Unavailable{Field: rssField, Reason: ReasonNoReader})
	default:
		if v, ok := c.rss(); ok {
			r.Memory.ProcessRSSBytes = U64{V: v, OK: true}
		} else {
			dyn = append(dyn, Unavailable{Field: rssField, Reason: ReasonReadFailed})
		}
	}
	lim := r.GC.MemoryLimitBytes
	r.GC.MemoryLimitUnlimited = lim.OK && lim.V == math.MaxInt64
	gogc := r.GC.GOGCPercent
	r.GC.GOGCOff = gogc.OK && gogc.V == gogcOff
	r.Sched.LatenciesSummary = Summarize(r.Sched.Latencies)
	r.GC.PausesSummary = Summarize(r.GC.Pauses)
	if n := len(c.static) + len(dyn); n > 0 {
		r.Unavailable = make([]Unavailable, 0, n)
		r.Unavailable = append(append(r.Unavailable, c.static...), dyn...)
	}
	if len(c.aliased) > 0 {
		// Clipped: a caller's append reallocates instead of writing into the
		// collector's slice.
		r.Aliased = c.aliased[:len(c.aliased):len(c.aliased)]
	}
	return r
}

// decode stores v into s's field, or returns why it cannot. It checks the
// kind before any accessor, since metrics.Value accessors panic on mismatch.
func decode(s *spec, r *Receipt, v metrics.Value) Reason {
	k := v.Kind()
	if k == metrics.KindBad {
		return ReasonBadValue
	}
	if k != s.kind {
		return ReasonWrongKind
	}
	switch k {
	case metrics.KindUint64:
		*s.u64(r) = U64{V: v.Uint64(), OK: true}
	case metrics.KindFloat64:
		f := v.Float64()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return ReasonBadValue
		}
		*s.f64(r) = F64{V: f, OK: true}
	case metrics.KindFloat64Histogram:
		h := v.Float64Histogram()
		if h == nil || len(h.Buckets) != len(h.Counts)+1 {
			return ReasonBadValue
		}
		*s.hist(r) = sparse(h)
	default:
		return ReasonWrongKind
	}
	return ""
}

// sparse copies the non-empty buckets of h.
func sparse(h *metrics.Float64Histogram) Hist {
	out := Hist{OK: true}
	n := 0
	for _, c := range h.Counts {
		if c != 0 {
			n++
		}
	}
	if n > 0 {
		out.Buckets = make([]Bucket, 0, n)
	}
	for i, c := range h.Counts {
		if c == 0 {
			continue
		}
		out.Count += c
		out.Buckets = append(out.Buckets, Bucket{Lo: h.Buckets[i], Hi: h.Buckets[i+1], N: c})
	}
	return out
}

// Delta is the interval between two receipts of one process epoch. Cumulative
// fields become interval counts; a field unavailable on either side stays
// unavailable.
type Delta struct {
	Schema          string  `json:"schema"`
	GoVersion       string  `json:"go_version"`
	Epoch           string  `json:"epoch"`
	FromSeq         uint64  `json:"from_seq"`
	ToSeq           uint64  `json:"to_seq"`
	IntervalSeconds float64 `json:"interval_seconds"`

	GCCycles          U64 `json:"gc_cycles"`
	GCForcedCycles    U64 `json:"gc_forced_cycles"`
	HeapAllocsBytes   U64 `json:"heap_allocs_bytes"`
	GoroutinesCreated U64 `json:"goroutines_created"`

	CPUTotalSeconds        F64 `json:"cpu_total_seconds"`
	CPUGCSeconds           F64 `json:"cpu_gc_seconds"`
	CPUGCMarkAssistSeconds F64 `json:"cpu_gc_mark_assist_seconds"`
	CPUGCPauseSeconds      F64 `json:"cpu_gc_pause_seconds"`
	// GCCPUFraction is CPUGCSeconds/CPUTotalSeconds: GC's share of the CPU
	// the runtime had available over the interval.
	GCCPUFraction F64 `json:"gc_cpu_fraction"`

	SchedLatencies        Hist    `json:"sched_latencies_seconds"`
	SchedLatenciesSummary Summary `json:"sched_latencies_summary"`
	GCPauses              Hist    `json:"gc_pauses_seconds"`
	GCPausesSummary       Summary `json:"gc_pauses_summary"`
}

// Diff refusals. Each is a boundary across which cumulative series carry no
// interval meaning; a caller re-baselines rather than reporting a delta.
var (
	ErrSchemaMismatch   = errors.New("runtimeobs: receipt schema mismatch")
	ErrEpochChanged     = errors.New("runtimeobs: process epoch changed; cumulative series reset")
	ErrGoVersionChanged = errors.New("runtimeobs: Go toolchain changed; scheduler-latency populations are not comparable across versions")
	ErrOutOfOrder       = errors.New("runtimeobs: receipts out of order")
	ErrCounterReset     = errors.New("runtimeobs: cumulative series decreased within one epoch")
)

// Diff computes cur minus prev.
func Diff(prev, cur Receipt) (Delta, error) {
	switch {
	case prev.Schema != Schema || cur.Schema != Schema:
		return Delta{}, ErrSchemaMismatch
	case prev.GoVersion != cur.GoVersion:
		return Delta{}, ErrGoVersionChanged
	case prev.Epoch != cur.Epoch:
		return Delta{}, ErrEpochChanged
	case cur.Seq <= prev.Seq:
		return Delta{}, ErrOutOfOrder
	}
	d := Delta{
		Schema:          DeltaSchema,
		GoVersion:       cur.GoVersion,
		Epoch:           cur.Epoch,
		FromSeq:         prev.Seq,
		ToSeq:           cur.Seq,
		IntervalSeconds: float64(cur.SampledAtUnixNano-prev.SampledAtUnixNano) / 1e9,
	}
	var err error
	sub := func(a, b U64) U64 {
		if !a.OK || !b.OK {
			return U64{}
		}
		if b.V < a.V {
			err = ErrCounterReset
			return U64{}
		}
		return U64{V: b.V - a.V, OK: true}
	}
	subf := func(a, b F64) F64 {
		if !a.OK || !b.OK {
			return F64{}
		}
		if b.V < a.V {
			err = ErrCounterReset
			return F64{}
		}
		return F64{V: b.V - a.V, OK: true}
	}
	d.GCCycles = sub(prev.GC.Cycles, cur.GC.Cycles)
	d.GCForcedCycles = sub(prev.GC.ForcedCycles, cur.GC.ForcedCycles)
	d.HeapAllocsBytes = sub(prev.GC.HeapAllocsBytes, cur.GC.HeapAllocsBytes)
	d.GoroutinesCreated = sub(prev.Sched.GoroutinesCreated, cur.Sched.GoroutinesCreated)
	d.CPUTotalSeconds = subf(prev.CPU.TotalSeconds, cur.CPU.TotalSeconds)
	d.CPUGCSeconds = subf(prev.CPU.GCTotalSeconds, cur.CPU.GCTotalSeconds)
	d.CPUGCMarkAssistSeconds = subf(prev.CPU.GCMarkAssistSeconds, cur.CPU.GCMarkAssistSeconds)
	d.CPUGCPauseSeconds = subf(prev.CPU.GCPauseSeconds, cur.CPU.GCPauseSeconds)
	if d.CPUGCSeconds.OK && d.CPUTotalSeconds.OK && d.CPUTotalSeconds.V > 0 {
		d.GCCPUFraction = F64{V: d.CPUGCSeconds.V / d.CPUTotalSeconds.V, OK: true}
	}
	var herr error
	if d.SchedLatencies, herr = cur.Sched.Latencies.Since(prev.Sched.Latencies); herr != nil {
		err = herr
	}
	if d.GCPauses, herr = cur.GC.Pauses.Since(prev.GC.Pauses); herr != nil {
		err = herr
	}
	if err != nil {
		return Delta{}, err
	}
	d.SchedLatenciesSummary = Summarize(d.SchedLatencies)
	d.GCPausesSummary = Summarize(d.GCPauses)
	return d, nil
}
