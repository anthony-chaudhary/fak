package compute

import (
	"errors"
	"math"
	"testing"
)

// fakeCUDAClock stands in for the CUDA event pool: each record stamps the
// current device time, and tick advances it, so intervals are exact.
type fakeCUDAClock struct {
	now       float64
	stamps    map[cudaProfileMark]float64
	records   int
	waits     int
	capture   bool
	idle      bool // g_stream had no work left at the next record
	failNext  int  // fail the next N records
	failSpan  bool
	failPairs map[int]bool
	nanPair   int
}

func newFakeCUDAClock() *fakeCUDAClock {
	return &fakeCUDAClock{stamps: map[cudaProfileMark]float64{}, failPairs: map[int]bool{}, nanPair: -1}
}

func (c *fakeCUDAClock) tick(ns float64) { c.now += ns }

func (c *fakeCUDAClock) record(m cudaProfileMark) (bool, error) {
	c.records++
	if c.failNext > 0 {
		c.failNext--
		return false, errors.New("record refused")
	}
	c.stamps[m] = c.now
	return c.idle, nil
}

func (c *fakeCUDAClock) elapsedNS(from, to cudaProfileMark) (float64, error) {
	c.waits++
	if from.slot != to.slot && c.failSpan {
		return 0, errors.New("span refused")
	}
	if from.slot == to.slot && c.failPairs[from.slot] {
		return 0, errors.New("pair refused")
	}
	if from.slot == to.slot && from.slot == c.nanPair {
		return math.NaN(), nil
	}
	a, ok1 := c.stamps[from]
	b, ok2 := c.stamps[to]
	if !ok1 || !ok2 {
		return 0, errors.New("event never recorded")
	}
	return b - a, nil
}

func (c *fakeCUDAClock) capturing() bool { return c.capture }

func runCUDASection(p *cudaKernelProfiler, c *fakeCUDAClock, name string, ns float64) {
	p.begin(name)
	c.tick(ns)
	p.end()
}

func mustCUDASnapshot(t *testing.T, p *cudaKernelProfiler) KernelProfile {
	t.Helper()
	snap, ok := p.snapshot()
	if !ok {
		t.Fatal("snapshot reported the profile unusable")
	}
	return snap
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileEnvGrammar(t *testing.T) {
	for value, want := range map[string]bool{"1": true, "": false, "0": false, "true": false, "1 ": false, "yes": false} {
		if got := cudaTimestampProfileEnabled(value); got != want {
			t.Errorf("cudaTimestampProfileEnabled(%q) = %v, want %v", value, got, want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileFoldsSectionsAndSpan(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 8, true)

	zero := mustCUDASnapshot(t, p)
	if zero.Batches != 0 || zero.Ops != 0 || zero.GPUNS != 0 || len(zero.Rows) != 0 {
		t.Fatalf("fresh profile must be a zero baseline, got %+v", zero)
	}

	runCUDASection(p, c, "matvec", 100)
	c.tick(7) // device idle between sections: in the span, not in any row
	runCUDASection(p, c, "rmsnorm", 20)
	runCUDASection(p, c, "matvec", 50)

	snap := mustCUDASnapshot(t, p)
	if snap.Batches != 1 || snap.Ops != 3 || snap.Truncated != 0 || snap.UntimedCompute != 0 {
		t.Fatalf("counters = %+v", snap)
	}
	if snap.GPUNS != 170 || snap.WallNS != 177 {
		t.Fatalf("GPUNS=%v WallNS=%v, want 170 and 177", snap.GPUNS, snap.WallNS)
	}
	want := []KernelRow{{Kernel: "matvec", Calls: 2, NS: 150}, {Kernel: "rmsnorm", Calls: 1, NS: 20}}
	if len(snap.Rows) != len(want) {
		t.Fatalf("rows = %+v", snap.Rows)
	}
	var sum float64
	for i, r := range snap.Rows {
		if r != want[i] {
			t.Fatalf("row %d = %+v, want %+v", i, r, want[i])
		}
		sum += r.NS
	}
	if sum != snap.GPUNS {
		t.Fatalf("rows sum to %v, GPUNS %v", sum, snap.GPUNS)
	}

	// A second snapshot with no new work drains nothing and stays identical.
	again := mustCUDASnapshot(t, p)
	if again.Batches != 1 || again.GPUNS != 170 {
		t.Fatalf("idle snapshot changed the fold: %+v", again)
	}
	if d := KernelProfileDelta(&zero, &snap); d == nil || d.Ops != 3 || d.GPUNS != 170 {
		t.Fatalf("delta from zero baseline = %+v", d)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileBoundedPoolDrains(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 2, true)
	for i := 0; i < 5; i++ {
		runCUDASection(p, c, "swiglu", 10)
		if len(p.pending) > 2 {
			t.Fatalf("pending grew past the pool: %d", len(p.pending))
		}
	}
	snap := mustCUDASnapshot(t, p)
	if snap.Batches != 3 || snap.Ops != 5 || snap.Truncated != 0 || snap.GPUNS != 50 {
		t.Fatalf("bounded fold = %+v", snap)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileCaptureIsUntimed(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 8, true)

	c.capture = true
	runCUDASection(p, c, "rope", 30) // captured: runs at replay
	c.capture = false
	p.begin("graphbegin") // capture opens inside the section
	c.capture = true
	p.end()
	c.capture = false
	runCUDASection(p, c, "add", 5)

	snap := mustCUDASnapshot(t, p)
	if snap.UntimedCompute != 2 || snap.Ops != 1 || snap.Truncated != 0 || snap.GPUNS != 5 {
		t.Fatalf("capture accounting = %+v", snap)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileIdleBoundaryNeverBooksHostTime(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 8, true)

	// Begins on an idle stream: the start event fires at once, so the interval
	// would include the host's launch latency.
	c.idle = true
	p.begin("rmsnorm")
	c.idle = false
	c.tick(500)
	p.end()

	// Busy at the start, but the work finished before the end was recorded: the
	// interval would include host time after it.
	p.begin("swiglu")
	c.tick(500)
	c.idle = true
	p.end()
	c.idle = false

	runCUDASection(p, c, "matvec", 9)
	snap := mustCUDASnapshot(t, p)
	if snap.Truncated != 2 || snap.Ops != 1 || snap.GPUNS != 9 || len(snap.Rows) != 1 || snap.Rows[0].Kernel != "matvec" {
		t.Fatalf("idle-boundary sections must be excluded, not timed: %+v", snap)
	}

	// While g_stream is capturing, an event cannot be waited on: the read is
	// unavailable and the pending pair is kept for the next read.
	runCUDASection(p, c, "add", 4)
	c.capture = true
	if _, ok := p.snapshot(); ok {
		t.Fatal("a read during capture must be unavailable")
	}
	c.capture = false
	if snap = mustCUDASnapshot(t, p); snap.Ops != 2 || snap.GPUNS != 13 {
		t.Fatalf("pair pending across a capture read was lost: %+v", snap)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileFailuresAreTruncatedNotZero(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 8, true)

	c.failNext = 1 // start event refused
	runCUDASection(p, c, "attention", 40)
	p.begin("attention")
	c.tick(40)
	c.failNext = 1 // end event refused
	p.end()
	runCUDASection(p, c, "attention", 40)
	c.nanPair = 0 // the one recorded pair reads NaN
	snap := mustCUDASnapshot(t, p)
	if snap.Truncated != 3 || snap.Ops != 0 || snap.GPUNS != 0 || len(snap.Rows) != 0 {
		t.Fatalf("failed sections must be truncated, not timed: %+v", snap)
	}

	c.nanPair = -1
	runCUDASection(p, c, "argmax", 3)
	runCUDASection(p, c, "argmax", 3)
	c.failSpan = true // the whole batch is lost
	snap = mustCUDASnapshot(t, p)
	if snap.Truncated != 5 || snap.Ops != 0 || snap.Batches != 0 || snap.WallNS != 0 {
		t.Fatalf("lost batch = %+v", snap)
	}

	c.failSpan = false
	runCUDASection(p, c, "argmax", 3)
	runCUDASection(p, c, "norm", 4)
	c.failPairs[1] = true
	snap = mustCUDASnapshot(t, p)
	if snap.Truncated != 6 || snap.Ops != 1 || snap.GPUNS != 3 || snap.WallNS != 7 {
		t.Fatalf("partial batch = %+v", snap)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileUnavailableAndDisabled(t *testing.T) {
	c := newFakeCUDAClock()
	p := newCUDAKernelProfiler(c, 8, false)
	runCUDASection(p, c, "matvec", 10)
	if c.records != 0 {
		t.Fatalf("an unusable pool recorded %d events", c.records)
	}
	if _, ok := p.snapshot(); ok {
		t.Fatal("an unusable pool must read unavailable, not zero")
	}
	if _, ok := newCUDAKernelProfiler(nil, 8, true).snapshot(); ok {
		t.Fatal("a profiler without a clock must read unavailable")
	}
	var nilProfiler *cudaKernelProfiler
	if _, ok := nilProfiler.snapshot(); ok {
		t.Fatal("nil profiler must read unavailable")
	}

	var m cudaDeviceMutex
	if _, ok := m.readProfile(); ok {
		t.Fatal("a mutex with no profiler must read as off")
	}
	p.end() // an end with no open section is inert
}

// fak-test:runtime fast est=20ms lane=default
func TestCUDAKernelProfileMutexOffPathIsCheap(t *testing.T) {
	var m cudaDeviceMutex
	if allocs := testing.AllocsPerRun(1000, func() { m.Lock(); m.Unlock() }); allocs != 0 {
		t.Fatalf("profiling-off Lock/Unlock allocated %v per run", allocs)
	}
	if m.names != nil {
		t.Fatal("profiling-off path resolved section names")
	}
}

func cudaProfileTestSection(m *cudaDeviceMutex, c *fakeCUDAClock) {
	m.Lock()
	defer m.Unlock()
	c.tick(11)
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKernelProfileMutexTimesCallerSections(t *testing.T) {
	c := newFakeCUDAClock()
	var m cudaDeviceMutex
	m.profile.Store(newCUDAKernelProfiler(c, 4, true))
	for i := 0; i < 3; i++ {
		cudaProfileTestSection(&m, c)
	}
	func() {
		m.Lock()
		defer m.Unlock()
		c.tick(2)
	}()

	snap, ok := m.readProfile()
	if !ok {
		t.Fatal("installed profiler read unavailable")
	}
	rows := map[string]KernelRow{}
	for _, r := range snap.Rows {
		rows[r.Kernel] = r
	}
	if r := rows["cudaprofiletestsection"]; r.Calls != 3 || r.NS != 33 {
		t.Fatalf("named section row = %+v (rows %+v)", r, snap.Rows)
	}
	if r := rows["testcudakernelprofilemutextimescallersections"]; r.Calls != 1 || r.NS != 2 {
		t.Fatalf("closure must fold into its parent: %+v (rows %+v)", r, snap.Rows)
	}
	if c.records != 8 {
		t.Fatalf("records = %d, want one start and one end per section", c.records)
	}
	// The reader locks the embedded mutex, so reading never times itself.
	if again, _ := m.readProfile(); again.Ops != 4 {
		t.Fatalf("reader timed itself: %+v", again)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileSectionNames(t *testing.T) {
	for fn, want := range map[string]string{
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).RMSNorm":          "rmsnorm",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).MatMul.func1":     "matvec",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).BatchedMatMul":    "batchedmatmul",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).Read":             "readback",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).uploadClass":      "h2d_upload",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).CloneTensor":      "d2d_copy",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaKV).Append":                "cudakv_append",
		"github.com/anthony-chaudhary/fak/internal/compute.applyCUDATF32":                   "applycudatf32",
		"github.com/anthony-chaudhary/fak/internal/compute.applyCUDATF32.func2.1":           "applycudatf32",
		"github.com/anthony-chaudhary/fak/internal/compute.(*cudaBackend).functionalRotate": "functionalrotate",
		"": "unknown",
	} {
		if got := cudaProfileSectionName(fn); got != want {
			t.Errorf("cudaProfileSectionName(%q) = %q, want %q", fn, got, want)
		}
	}
	for name, want := range map[string]string{"matvec": KernelOpMatvec, "batchedmatmul": KernelOpMatmul, "rmsnorm": KernelOpNorm, "readback": KernelOpCopy, "h2d_upload": KernelOpCopy, "attention": KernelOpAttention} {
		if got := KernelOpClass(name); got != want {
			t.Errorf("KernelOpClass(%q) = %q, want %q", name, got, want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileRegistersThroughSeam(t *testing.T) {
	c := newFakeCUDAClock()
	var m cudaDeviceMutex
	m.profile.Store(newCUDAKernelProfiler(c, 4, true))
	unregister := RegisterKernelProfileReader("cuda", m.readProfile)
	defer unregister()
	if !KernelProfileReaderRegistered() {
		t.Fatal("reader not registered")
	}
	before := KernelProfileSnapshot()
	cudaProfileTestSection(&m, c)
	after := KernelProfileSnapshot()
	if before == nil || after == nil || after.Backend != "cuda" {
		t.Fatalf("snapshots before=%+v after=%+v", before, after)
	}
	d := KernelProfileDelta(before, after)
	if d == nil || d.Backend != "cuda" || d.Ops != 1 || d.GPUNS != 11 {
		t.Fatalf("delta = %+v", d)
	}

	// A reinstalled profiler (a reset counter) must not pass as a window.
	m.profile.Store(newCUDAKernelProfiler(c, 4, true))
	if reset := KernelProfileSnapshot(); KernelProfileDelta(after, reset) != nil {
		t.Fatalf("a reset fold produced a delta from %+v to %+v", after, reset)
	}

	// An unusable pool keeps the reader registered but reads unavailable.
	m.profile.Store(newCUDAKernelProfiler(c, 4, false))
	if KernelProfileSnapshot() != nil || !KernelProfileReaderRegistered() {
		t.Fatal("unavailable must stay registered and read nil")
	}
}
