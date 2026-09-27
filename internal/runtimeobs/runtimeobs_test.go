package runtimeobs

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// catalogWithout returns the live catalog minus the named metrics.
func catalogWithout(names ...string) Catalog {
	drop := map[string]bool{}
	for _, n := range names {
		drop[n] = true
	}
	return func() []metrics.Description {
		var out []metrics.Description
		for _, d := range metrics.All() {
			if !drop[d.Name] {
				out = append(out, d)
			}
		}
		return out
	}
}

func findUnavailable(r Receipt, field string) (Unavailable, bool) {
	for _, u := range r.Unavailable {
		if u.Field == field {
			return u, true
		}
	}
	return Unavailable{}, false
}

func TestLiveRuntimeFillsFullVector(t *testing.T) {
	c := New(WithRSSReader(func() (uint64, bool) { return 4096, true }))
	r := c.Sample()
	if r.Schema != Schema || r.GoVersion != runtime.Version() || r.Epoch == "" || r.Seq != 1 {
		t.Fatalf("header = %+v", r)
	}
	if len(r.Unavailable) != 0 || len(r.Aliased) != 0 {
		t.Fatalf("live %s runtime should resolve every metric canonically: unavailable=%+v aliased=%+v", runtime.Version(), r.Unavailable, r.Aliased)
	}
	// Every declared field must be observed: walk the spec table itself so a
	// new spec cannot silently ship unfilled.
	for i := range specs {
		s := &specs[i]
		ok := false
		switch {
		case s.u64 != nil:
			ok = s.u64(&r).OK
		case s.f64 != nil:
			ok = s.f64(&r).OK
		case s.hist != nil:
			ok = s.hist(&r).OK
		}
		if !ok {
			t.Errorf("%s not observed", s.field)
		}
	}
	if !r.Memory.ProcessRSSBytes.OK || r.Memory.ProcessRSSBytes.V != 4096 {
		t.Fatalf("rss = %+v", r.Memory.ProcessRSSBytes)
	}
	m := r.Memory
	if m.HeapObjectsBytes.V == 0 || m.HeapObjectsBytes.V > m.TotalMappedBytes.V {
		t.Fatalf("go heap %d must be nonzero and within total mapped %d", m.HeapObjectsBytes.V, m.TotalMappedBytes.V)
	}
	if got := r.Sched.GOMAXPROCS.V; got != uint64(runtime.GOMAXPROCS(0)) {
		t.Fatalf("gomaxprocs = %d, want %d", got, runtime.GOMAXPROCS(0))
	}
	if r.Sched.Threads.V == 0 || r.Sched.Goroutines.V == 0 || r.Sched.GoroutinesCreated.V == 0 {
		t.Fatalf("sched = %+v", r.Sched)
	}
	// The latency histogram samples 1-in-8 transitions and may still be empty
	// this early in a test binary; presence is the contract here.
	if !r.Sched.Latencies.OK || !r.GC.Pauses.OK {
		t.Fatalf("histograms unavailable: latencies=%+v pauses=%+v", r.Sched.Latencies.OK, r.GC.Pauses.OK)
	}
	if ls := r.Sched.LatenciesSummary; !ls.Count.OK || ls.Count.V != r.Sched.Latencies.Count {
		t.Fatalf("latency summary %+v does not digest the histogram (count %d)", ls, r.Sched.Latencies.Count)
	}
	if ps := r.GC.PausesSummary; !ps.Count.OK || ps.Count.V != r.GC.Pauses.Count {
		t.Fatalf("pause summary %+v does not digest the histogram (count %d)", ps, r.GC.Pauses.Count)
	}
	if r2 := c.Sample(); r2.Seq != 2 || r2.Epoch != r.Epoch {
		t.Fatalf("second sample seq=%d epoch=%q, want 2 / %q", r2.Seq, r2.Epoch, r.Epoch)
	}
}

func TestMemoryLimitUnlimitedIsExplicit(t *testing.T) {
	old := debug.SetMemoryLimit(-1)
	defer debug.SetMemoryLimit(old)

	debug.SetMemoryLimit(math.MaxInt64)
	r := New().Sample()
	if !r.GC.MemoryLimitBytes.OK || !r.GC.MemoryLimitUnlimited {
		t.Fatalf("unlimited GOMEMLIMIT not explicit: %+v", r.GC)
	}
	debug.SetMemoryLimit(8 << 30)
	r = New().Sample()
	if r.GC.MemoryLimitUnlimited || r.GC.MemoryLimitBytes.V != 8<<30 {
		t.Fatalf("set GOMEMLIMIT not reported: %+v", r.GC)
	}
}

func TestGOGCOffIsExplicit(t *testing.T) {
	old := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(old)
	r := New().Sample()
	if !r.GC.GOGCPercent.OK || !r.GC.GOGCOff {
		t.Fatalf("GOGC=off not explicit: percent=%+v off=%v", r.GC.GOGCPercent, r.GC.GOGCOff)
	}
	debug.SetGCPercent(150)
	r = New().Sample()
	if r.GC.GOGCOff || r.GC.GOGCPercent.V != 150 {
		t.Fatalf("GOGC=150 not reported: percent=%+v off=%v", r.GC.GOGCPercent, r.GC.GOGCOff)
	}
}

func TestSummarizeIsConservativeAndNeverInfinite(t *testing.T) {
	if s := Summarize(Hist{}); s.Count.OK || s.P50.OK || s.Max.OK {
		t.Fatalf("unavailable histogram summary = %+v, want all null", s)
	}
	if s := Summarize(Hist{OK: true}); !s.Count.OK || s.Count.V != 0 || s.P50.OK || s.Max.OK {
		t.Fatalf("empty histogram summary = %+v, want count 0 and null quantiles", s)
	}
	h := Hist{OK: true, Count: 100, Buckets: []Bucket{
		{0, 1e-6, 50}, {1e-6, 1e-3, 40}, {1e-3, 1e-2, 9}, {1e-2, math.Inf(1), 1},
	}}
	got := Summarize(h)
	want := Summary{
		Count: U64{V: 100, OK: true},
		P50:   F64{V: 1e-6, OK: true},
		P90:   F64{V: 1e-3, OK: true},
		P99:   F64{V: 1e-2, OK: true},
		Max:   F64{V: 1e-2, OK: true}, // open top bucket reports its finite lower bound
	}
	if got != want {
		t.Fatalf("summary = %+v\nwant      %+v", got, want)
	}
	unbounded := Hist{OK: true, Count: 1, Buckets: []Bucket{{math.Inf(-1), math.Inf(1), 1}}}
	if s := Summarize(unbounded); !s.Count.OK || s.P50.OK || s.Max.OK {
		t.Fatalf("unbounded-bucket summary = %+v, want count only", s)
	}
}

func TestMissingMetricIsTypedUnavailableNotZero(t *testing.T) {
	c := New(WithCatalog(catalogWithout("/sched/goroutines/runnable:goroutines")))
	r := c.Sample()
	if r.Sched.GoroutinesRunnable.OK {
		t.Fatal("missing metric reported as observed")
	}
	u, ok := findUnavailable(r, "sched.goroutines_runnable")
	if !ok || u.Reason != ReasonUnsupported || u.Metric != "/sched/goroutines/runnable:goroutines" {
		t.Fatalf("unavailable = %+v (found %v)", r.Unavailable, ok)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"goroutines_runnable":null`) {
		t.Fatalf("missing metric must marshal as null, got %s", b)
	}
	if u, ok := findUnavailable(r, rssField); !ok || u.Reason != ReasonNoReader {
		t.Fatalf("rss without a reader must be no_reader: %+v", r.Unavailable)
	}
}

func TestRenamedMetricResolvesThroughAlias(t *testing.T) {
	r := New(WithCatalog(catalogWithout("/sched/pauses/total/gc:seconds"))).Sample()
	if !r.GC.Pauses.OK {
		t.Fatalf("GC pauses not filled through the /gc/pauses:seconds alias; unavailable=%+v", r.Unavailable)
	}
	want := []Alias{{Field: "gc.pauses_seconds", Metric: "/gc/pauses:seconds"}}
	if !reflect.DeepEqual(r.Aliased, want) {
		t.Fatalf("aliased = %+v, want %+v", r.Aliased, want)
	}
}

func TestCatalogWrongKindIsRefusedAtResolve(t *testing.T) {
	cat := func() []metrics.Description {
		// metrics.All returns the runtime's own slice: copy before editing.
		all := append([]metrics.Description(nil), metrics.All()...)
		for i := range all {
			if all[i].Name == "/gc/cycles/total:gc-cycles" {
				all[i].Kind = metrics.KindFloat64
			}
		}
		return all
	}
	c := New(WithCatalog(cat))
	for _, s := range c.samples {
		if s.Name == "/gc/cycles/total:gc-cycles" {
			t.Fatal("wrong-kind metric must not enter the read vector")
		}
	}
	r := c.Sample()
	if u, ok := findUnavailable(r, "gc.cycles"); r.GC.Cycles.OK || !ok || u.Reason != ReasonWrongKind {
		t.Fatalf("gc.cycles = %+v unavailable=%+v", r.GC.Cycles, r.Unavailable)
	}
}

func TestReaderWrongKindAndBadValueDoNotPanic(t *testing.T) {
	// metrics.Value has no exported constructor, so the fixture borrows a
	// real float64 value to impersonate a wrong-kind reader.
	probe := []metrics.Sample{{Name: "/cpu/classes/total:cpu-seconds"}}
	metrics.Read(probe)
	floatVal := probe[0].Value
	reader := func(s []metrics.Sample) {
		metrics.Read(s)
		for i := range s {
			switch s[i].Name {
			case "/gc/cycles/total:gc-cycles":
				s[i].Value = floatVal // wrong kind
			case "/sched/goroutines:goroutines", "/sched/latencies:seconds":
				s[i].Value = metrics.Value{} // KindBad
			}
		}
	}
	r := New(WithReader(reader)).Sample()
	cases := map[string]Reason{
		"gc.cycles":               ReasonWrongKind,
		"sched.goroutines":        ReasonBadValue,
		"sched.latencies_seconds": ReasonBadValue,
	}
	for field, want := range cases {
		if u, ok := findUnavailable(r, field); !ok || u.Reason != want {
			t.Errorf("%s: unavailable=%+v found=%v, want %s", field, u, ok, want)
		}
	}
	if r.GC.Cycles.OK || r.Sched.Goroutines.OK || r.Sched.Latencies.OK {
		t.Fatalf("refused fields reported observed: cycles=%+v goroutines=%+v", r.GC.Cycles, r.Sched.Goroutines)
	}
	if !r.GC.HeapGoalBytes.OK {
		t.Fatal("an unrelated field was lost")
	}
}

// TestOlderToolchainFixture models a pre-1.26 runtime: the goroutine-state,
// created and thread metrics are absent and the pause metric only exists under
// its old name. Exactly those fields degrade; the rest of the vector survives.
func TestOlderToolchainFixture(t *testing.T) {
	gone := []string{
		"/sched/goroutines/running:goroutines",
		"/sched/goroutines/runnable:goroutines",
		"/sched/goroutines/waiting:goroutines",
		"/sched/goroutines/not-in-go:goroutines",
		"/sched/goroutines-created:goroutines",
		"/sched/threads/total:threads",
		"/sched/pauses/total/gc:seconds",
	}
	r := New(WithCatalog(catalogWithout(gone...)), WithRSSReader(func() (uint64, bool) { return 0, false })).Sample()
	var got []string
	for _, u := range r.Unavailable {
		got = append(got, u.Field+"="+string(u.Reason))
	}
	want := []string{
		"sched.goroutines_running=unsupported",
		"sched.goroutines_runnable=unsupported",
		"sched.goroutines_waiting=unsupported",
		"sched.goroutines_not_in_go=unsupported",
		"sched.goroutines_created=unsupported",
		"sched.threads=unsupported",
		"memory.process_rss_bytes=read_failed",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unavailable = %v\nwant          %v", got, want)
	}
	if !r.GC.Pauses.OK || len(r.Aliased) != 1 || !r.Sched.Latencies.OK || !r.Memory.TotalMappedBytes.OK {
		t.Fatalf("surviving vector degraded: pauses=%v aliased=%+v", r.GC.Pauses.OK, r.Aliased)
	}
}

func TestReceiptJSONRoundTrip(t *testing.T) {
	r := New(WithCatalog(catalogWithout("/sched/threads/total:threads"))).Sample()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal (histogram infinities must not break JSON): %v", err)
	}
	var back Receipt
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r, back) {
		t.Fatalf("round trip mismatch\n got %+v\nwant %+v", back, r)
	}
	if !strings.Contains(string(b), `"latencies_seconds":{"count":`) || !strings.Contains(string(b), `"latencies_summary":{"count":`) || !strings.Contains(string(b), `"threads":null`) {
		t.Fatalf("receipt JSON shape drifted: %s", b)
	}
	inf := Hist{OK: true, Count: 2, Buckets: []Bucket{{math.Inf(-1), 0, 1}, {1, math.Inf(1), 1}}}
	hb, err := json.Marshal(inf)
	if err != nil {
		t.Fatal(err)
	}
	if string(hb) != `{"count":2,"buckets":[{"lo":null,"hi":0,"n":1},{"lo":1,"hi":null,"n":1}]}` {
		t.Fatalf("infinite bounds = %s", hb)
	}
	var hback Hist
	if err := json.Unmarshal(hb, &hback); err != nil || !reflect.DeepEqual(hback, inf) {
		t.Fatalf("infinite bounds round trip = %+v, %v", hback, err)
	}
}

// TestReceiptOwnsHistogramStorage pins the #10182 fence: runtime/metrics.Read
// reuses histogram backing storage, so a receipt must not alias it.
func TestReceiptOwnsHistogramStorage(t *testing.T) {
	c := New()
	r1 := c.Sample()
	snapshot := append([]Bucket(nil), r1.Sched.Latencies.Buckets...)
	burst(64)
	_ = c.Sample()
	if !reflect.DeepEqual(snapshot, r1.Sched.Latencies.Buckets) {
		t.Fatal("a later Sample mutated an earlier receipt's histogram")
	}
}

func TestDiffRefusesAcrossBoundaries(t *testing.T) {
	c := New()
	a, b := c.Sample(), c.Sample()
	if _, err := Diff(a, b); err != nil {
		t.Fatalf("same-epoch diff: %v", err)
	}
	other := b
	other.Epoch = "1@1"
	if _, err := Diff(a, other); !errors.Is(err, ErrEpochChanged) {
		t.Fatalf("epoch change err = %v", err)
	}
	other = b
	other.GoVersion = "go1.25.0"
	if _, err := Diff(a, other); !errors.Is(err, ErrGoVersionChanged) {
		t.Fatalf("toolchain change err = %v", err)
	}
	if _, err := Diff(b, a); !errors.Is(err, ErrOutOfOrder) {
		t.Fatalf("reversed diff err = %v", err)
	}
	other = b
	other.GC.Cycles.V = a.GC.Cycles.V - 1
	if a.GC.Cycles.V > 0 {
		if _, err := Diff(a, other); !errors.Is(err, ErrCounterReset) {
			t.Fatalf("decreasing counter err = %v", err)
		}
	}
	other = b
	other.Schema = "fak.go-runtime.v0"
	if _, err := Diff(a, other); !errors.Is(err, ErrSchemaMismatch) {
		t.Fatalf("schema err = %v", err)
	}
}

func TestDiffUnavailableStaysUnavailable(t *testing.T) {
	c := New(WithCatalog(catalogWithout("/gc/cycles/total:gc-cycles")))
	d, err := Diff(c.Sample(), c.Sample())
	if err != nil {
		t.Fatal(err)
	}
	if d.GCCycles.OK {
		t.Fatal("delta of an unavailable counter must stay unavailable")
	}
	if !d.CPUTotalSeconds.OK {
		t.Fatal("available counter lost in delta")
	}
}

func TestHistSinceAndQuantile(t *testing.T) {
	inf := math.Inf(1)
	prev := Hist{OK: true, Count: 3, Buckets: []Bucket{{0, 1, 2}, {1, 2, 1}}}
	cur := Hist{OK: true, Count: 10, Buckets: []Bucket{{0, 1, 2}, {1, 2, 4}, {2, inf, 4}}}
	d, err := cur.Since(prev)
	if err != nil {
		t.Fatal(err)
	}
	want := Hist{OK: true, Count: 7, Buckets: []Bucket{{1, 2, 3}, {2, inf, 4}}}
	if !reflect.DeepEqual(d, want) {
		t.Fatalf("delta = %+v, want %+v", d, want)
	}
	if q, ok := d.Quantile(0.5); !ok || q != 2 {
		t.Fatalf("p50 = %v %v, want 2", q, ok)
	}
	if q, ok := d.Quantile(1); !ok || q != 2 {
		t.Fatalf("p100 in open bucket must report its finite lower bound, got %v", q)
	}
	if _, ok := (Hist{OK: true}).Quantile(0.5); ok {
		t.Fatal("empty histogram quantile must be unavailable")
	}
	shrunk := Hist{OK: true, Count: 2, Buckets: []Bucket{{0, 1, 1}, {1, 2, 1}}}
	if _, err := shrunk.Since(prev); !errors.Is(err, ErrCounterReset) {
		t.Fatalf("shrunk bucket err = %v", err)
	}
	vanished := Hist{OK: true, Count: 5, Buckets: []Bucket{{1, 2, 5}}}
	if _, err := vanished.Since(prev); !errors.Is(err, ErrCounterReset) {
		t.Fatalf("vanished bucket err = %v", err)
	}
}

// burst spawns n short goroutines.
func burst(n int) {
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() { defer wg.Done() }()
	}
	wg.Wait()
}

var gcSink [][]byte

// gcPressure churns a fixed allocation volume (not a wall-clock window, so a
// loaded host slows it down instead of shrinking it) through a small live set
// under a low GOGC: every ~200KB allocated completes a cycle.
func gcPressure(totalBytes int) {
	old := debug.SetGCPercent(5)
	defer debug.SetGCPercent(old)
	const piece, live = 1024, 256
	for n := 0; n < totalBytes; n += piece {
		gcSink = append(gcSink, make([]byte, piece))
		if len(gcSink) >= live {
			gcSink = gcSink[:0]
		}
	}
	gcSink = nil
}

var spinSink atomic.Uint64

// queueing oversubscribes one P with allocation-free spinners that yield
// often, so runnable goroutines wait in the run queue without GC work.
func queueing(d time.Duration, mid func()) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	deadline := time.Now().Add(d)
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x := uint64(1)
			for time.Now().Before(deadline) {
				for j := 0; j < 200000; j++ {
					x = x*6364136223846793005 + 1442695040888963407
				}
				runtime.Gosched()
			}
			spinSink.Add(x)
		}()
	}
	time.Sleep(d / 2)
	mid()
	wg.Wait()
}

// TestQueueingAndGCPressureSignaturesDiffer is the #10182 witness: a
// run-queue overload and a GC-pressure run produce distinguishable deltas.
func TestQueueingAndGCPressureSignaturesDiffer(t *testing.T) {
	if testing.Short() {
		t.Skip("controlled load fixture")
	}
	c := New()
	runtime.GC()

	q0 := c.Sample()
	var qMid Receipt
	queueing(300*time.Millisecond, func() { qMid = c.Sample() })
	q1 := c.Sample()
	qd, err := Diff(q0, q1)
	if err != nil {
		t.Fatal(err)
	}

	g0 := c.Sample()
	gcPressure(32 << 20)
	g1 := c.Sample()
	gd, err := Diff(g0, g1)
	if err != nil {
		t.Fatal(err)
	}

	qP50, qok := qd.SchedLatenciesSummary.P50.V, qd.SchedLatenciesSummary.P50.OK
	gP50, gok := gd.SchedLatenciesSummary.P50.V, gd.SchedLatenciesSummary.P50.OK
	t.Logf("queueing: gc_cycles=%d gc_cpu=%.4f sched_p50=%v sched_p99=%v runnable_mid=%d latency_samples=%d",
		qd.GCCycles.V, qd.GCCPUFraction.V, time.Duration(qP50*1e9), time.Duration(qd.SchedLatenciesSummary.P99.V*1e9),
		qMid.Sched.GoroutinesRunnable.V, qd.SchedLatencies.Count)
	t.Logf("gc:       gc_cycles=%d gc_cpu=%.4f sched_p50=%v assist_s=%.4f pause_p99=%v latency_samples=%d",
		gd.GCCycles.V, gd.GCCPUFraction.V, time.Duration(gP50*1e9), gd.CPUGCMarkAssistSeconds.V,
		time.Duration(gd.GCPausesSummary.P99.V*1e9), gd.SchedLatencies.Count)

	// GC signature: many cycles, the churned volume, recorded pauses, and a
	// real GC CPU share.
	if gd.GCCycles.V < 10 {
		t.Errorf("gc fixture ran %d GC cycles, want >= 10", gd.GCCycles.V)
	}
	if !gd.HeapAllocsBytes.OK || gd.HeapAllocsBytes.V < 32<<20 {
		t.Errorf("gc fixture heap allocs delta = %+v, want >= the 32MiB it churned", gd.HeapAllocsBytes)
	}
	if !gd.GCPausesSummary.Count.OK || gd.GCPausesSummary.Count.V == 0 {
		t.Errorf("gc fixture recorded no stop-the-world pauses: %+v", gd.GCPausesSummary)
	}
	if !gd.GCCPUFraction.OK || gd.GCCPUFraction.V <= qd.GCCPUFraction.V {
		t.Errorf("gc cpu fraction gc=%v queue=%v: GC fixture must dominate", gd.GCCPUFraction, qd.GCCPUFraction)
	}
	// Queueing signature: scheduler delay, not GC.
	if qd.GCCycles.V*4 > gd.GCCycles.V {
		t.Errorf("queueing fixture ran %d GC cycles vs gc fixture %d: not distinguishable", qd.GCCycles.V, gd.GCCycles.V)
	}
	if !qok || qP50 < 1e-3 {
		t.Errorf("queueing sched latency p50 = %v (ok=%v), want >= 1ms", qP50, qok)
	}
	if gok && gP50 >= qP50 {
		t.Errorf("gc fixture sched p50 %v >= queueing p50 %v", gP50, qP50)
	}
	if !qMid.Sched.GoroutinesRunnable.OK || qMid.Sched.GoroutinesRunnable.V == 0 {
		t.Errorf("mid-overload runnable gauge = %+v, want > 0", qMid.Sched.GoroutinesRunnable)
	}
}

// TestSampleCostWithinCadenceBudget checks elapsed collection time against a
// predeclared 1% cadence budget. Each Sample is synchronous, so wall time
// conservatively bounds the caller thread's CPU time at a 1s cadence. This
// witness excludes secondary runtime work and total process CPU consumption.
func TestSampleCostWithinCadenceBudget(t *testing.T) {
	c := New()
	const n = 200
	start := time.Now()
	for i := 0; i < n; i++ {
		_ = c.Sample()
	}
	per := time.Since(start) / n
	t.Logf("Sample cost %v per receipt", per)
	if !WithinBudget(per, time.Second) {
		t.Fatalf("Sample cost %v exceeds %.0f%% of a 1s cadence", per, CPUBudgetFraction*100)
	}
}

func BenchmarkSample(b *testing.B) {
	c := New()
	b.ReportAllocs()
	for b.Loop() {
		_ = c.Sample()
	}
}

func BenchmarkReadMemStatsBaseline(b *testing.B) {
	var ms runtime.MemStats
	b.ReportAllocs()
	for b.Loop() {
		runtime.ReadMemStats(&ms)
	}
}

func BenchmarkRawRead(b *testing.B) {
	c := New()
	b.ReportAllocs()
	for b.Loop() {
		metrics.Read(c.samples)
	}
}
