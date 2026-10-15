package model

import (
	"bytes"
	"errors"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// overlapReaderAt serves bytes like bytes.Reader, holds each read briefly and records the
// peak number of reads in flight, so a test can tell serial page-ins from overlapped ones.
type overlapReaderAt struct {
	r        *bytes.Reader
	mu       sync.Mutex
	inflight int
	peak     int
}

func (o *overlapReaderAt) ReadAt(p []byte, off int64) (int, error) {
	o.mu.Lock()
	o.inflight++
	if o.inflight > o.peak {
		o.peak = o.inflight
	}
	o.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
	n, err := o.r.ReadAt(p, off)
	o.mu.Lock()
	o.inflight--
	o.mu.Unlock()
	return n, err
}

func prefetchTestTier(t *testing.T) (*ExpertCheckpointTier, *overlapReaderAt, []byte, int64) {
	t.Helper()
	const H, E = 256, 4
	stride := int64(H * (H / qkK) * q4kBlockBytes)
	blob := make([]byte, 3*E*stride)
	rand.New(rand.NewSource(13)).Read(blob)
	var fused []FusedExpertTensor
	for i, proj := range []string{"gate", "up", "down"} {
		fused = append(fused, FusedExpertTensor{
			Name: "blk.0.ffn_" + proj + "_exps.weight", Layer: 0, Proj: proj + "_proj",
			Quant: ExpertCheckpointQ4K, Offset: int64(i) * E * stride, Experts: E, Rows: H, Cols: H,
		})
	}
	src := &overlapReaderAt{r: bytes.NewReader(blob)}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(src, int64(len(blob)), fused); err != nil {
		t.Fatal(err)
	}
	return tier, src, blob, stride
}

func prefetchTestNames(experts ...int) []string {
	var names []string
	for _, e := range experts {
		for _, proj := range []string{"gate", "up", "down"} {
			names = append(names, expertName(0, e, proj+"_proj.weight"))
		}
	}
	return names
}

func waitPrefetchSettled(t *testing.T, tier *ExpertCheckpointTier) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		tier.mu.Lock()
		n := len(tier.loading)
		tier.mu.Unlock()
		if n == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("prefetched reads did not finish")
}

// A layer's misses are read together, and every demand fault is served from those reads: the
// checkpoint is read once per projection, the bytes are the checkpoint's, and reads overlap.
// fak-test:runtime fast est=300ms lane=default
func TestExpertCheckpointPrefetchOverlapsReadsAndServesDemandOnce(t *testing.T) {
	tier, src, blob, stride := prefetchTestTier(t)
	names := prefetchTestNames(1, 3)
	if got := tier.Prefetch(names); got != len(names) {
		t.Fatalf("Prefetch started %d reads, want %d", got, len(names))
	}
	if again := tier.Prefetch(names); again != 0 {
		t.Fatalf("a second Prefetch of in-flight names started %d reads, want 0", again)
	}
	for i, name := range names {
		w, err := tier.fault(name)
		if err != nil {
			t.Fatal(err)
		}
		proj, e := i%3, []int{1, 3}[i/3]
		start := int64(proj)*4*stride + int64(e)*stride
		if !bytes.Equal(w.q4.raw, blob[start:start+stride]) {
			t.Fatalf("%s bytes differ from the checkpoint", name)
		}
	}
	st := tier.Stats()
	if st.Reads != len(names) || st.Prefetched != len(names) || st.PrefetchHits+st.Coalesced != len(names) {
		t.Fatalf("reads=%d prefetched=%d hits=%d coalesced=%d, want %d reads all served from the prefetch",
			st.Reads, st.Prefetched, st.PrefetchHits, st.Coalesced, len(names))
	}
	if src.peak < 2 {
		t.Fatalf("peak concurrent checkpoint reads = %d, want the layer's misses overlapped", src.peak)
	}
}

// A handoff nobody consumes is dropped two prefetch generations later, so prefetched bytes stay
// bounded, and a later demand for that projection reads the checkpoint again.
// fak-test:runtime fast est=300ms lane=default
func TestExpertCheckpointPrefetchDropsUnconsumedHandoffs(t *testing.T) {
	tier, _, _, _ := prefetchTestTier(t)
	stale := prefetchTestNames(0)
	tier.Prefetch(stale)
	waitPrefetchSettled(t, tier)
	tier.Prefetch(prefetchTestNames(1))
	waitPrefetchSettled(t, tier)
	tier.Prefetch(prefetchTestNames(2))
	waitPrefetchSettled(t, tier)
	st := tier.Stats()
	if st.PrefetchStale != len(stale) {
		t.Fatalf("stale handoffs dropped = %d, want %d", st.PrefetchStale, len(stale))
	}
	before := st.Reads
	if _, err := tier.fault(stale[0]); err != nil {
		t.Fatal(err)
	}
	if got := tier.Stats().Reads; got != before+1 {
		t.Fatalf("reads after faulting a dropped handoff = %d, want %d", got, before+1)
	}
	if n := tier.Prefetch([]string{"model.layers.0.mlp.experts.9.gate_proj.weight"}); n != 0 {
		t.Fatalf("Prefetch of an unindexed name started %d reads, want 0", n)
	}
}

// Unindexed hints must not allocate job storage proportional to caller input, even when
// nothing is admitted. No source or goroutine is needed to exercise the real Prefetch path.
// fak-test:runtime fast est=20ms lane=default
func TestExpertCheckpointPrefetchBoundsJobStorage(t *testing.T) {
	for _, tc := range []struct {
		name          string
		names, active int
		maxBytes      uint64
	}{
		{"small-input", 1, 0, 16 << 10},
		{"large-input", 4096, 0, 16 << 10},
		{"one-slot-left", 4096, fakExpertPrefetchJobs - 1, 16 << 10},
		{"exhausted", 4096, fakExpertPrefetchJobs, 16 << 10},
		{"over-cap", 4096, fakExpertPrefetchJobs + 1, 16 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Stay nonparallel: TotalAlloc is process-wide. Build the input and tier
			// before measuring so only Prefetch's bookkeeping is charged here.
			names := make([]string, tc.names)
			tier := NewExpertCheckpointTier(0)
			tier.prefetchJobs = tc.active
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			got := tier.Prefetch(names)
			runtime.ReadMemStats(&after)
			if got != 0 || tier.prefetchJobs != tc.active {
				t.Fatalf("unindexed hints admitted %d jobs, active=%d, want 0 and %d", got, tier.prefetchJobs, tc.active)
			}
			// Allow allocator rounding and process-wide runtime noise without
			// allowing job storage proportional to the large caller input.
			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > tc.maxBytes {
				t.Fatalf("Prefetch of %d names with %d active jobs allocated %d bytes, want <= %d", tc.names, tc.active, allocated, tc.maxBytes)
			}
		})
	}
}

// TestExpertCheckpointPrefetchBoundedOwnership pins the ownership contract Prefetch owes the
// checkpoint: speculative work is bounded in aggregate, declined outright on an allocation path
// whose true extent is not the projection stride, and settled exactly once even when the read
// panics. Each subcase is deterministic and channel/ledger gated -- no timing is the oracle.
// fak-test:runtime fast est=500ms lane=default
func TestExpertCheckpointPrefetchBoundedOwnership(t *testing.T) {
	// The aggregate bound is on ADMITTED jobs, not on one call's readers: a single call offering
	// more names than the cap admits exactly the cap and declines the rest, so the tier's
	// speculative read set cannot grow with the caller's input length.
	t.Run("aggregate-job-bound", func(t *testing.T) {
		const n = 2 * fakExpertPrefetchJobs
		tier, src, _, _ := oneCopyGateTier(t, n, 0, 17)
		names := make([]string, 0, n)
		for e := 0; e < n; e++ {
			names = append(names, expertName(0, e, "gate_proj.weight"))
		}
		if got := tier.Prefetch(names); got != fakExpertPrefetchJobs {
			t.Fatalf("Prefetch of %d names admitted %d jobs, want the aggregate cap %d", n, got, fakExpertPrefetchJobs)
		}
		waitPrefetchSettled(t, tier)
		offsets, _ := src.recorded()
		if len(offsets) != fakExpertPrefetchJobs {
			t.Fatalf("admitted %d reads, want %d", len(offsets), fakExpertPrefetchJobs)
		}
		// Exactly-once settlement: every admitted job released its share of the aggregate bound.
		tier.mu.Lock()
		jobs := tier.prefetchJobs
		tier.mu.Unlock()
		if jobs != 0 {
			t.Fatalf("aggregate prefetch jobs after settlement = %d, want 0", jobs)
		}
	})

	// A page-cache cold read allocates a page-aligned buffer and returns a stride-length subslice,
	// so charging the stride would undercount the backing allocation. Prefetch declines that path
	// rather than silently assuming its extent; demand serving is untouched.
	t.Run("declines-page-cache-accounting", func(t *testing.T) {
		tier, reader, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ4K, 0, true, false)
		if got := tier.Prefetch([]string{name}); got != 0 {
			t.Fatalf("Prefetch on a page-cache source admitted %d jobs, want 0", got)
		}
		if calls := reader.reads.Load(); calls != 0 {
			t.Fatalf("declined prefetch moved %d payload reads, want 0", calls)
		}
		w, err := tier.fault(name)
		if err != nil || !bytes.Equal(mappedCheckpointRaw(w), reader.data[:stride]) {
			t.Fatalf("declining the hint changed demand serving: %v", err)
		}
	})

	// A sparse overlay read has its own allocation path, so the same decline applies.
	t.Run("declines-sparse-overlay", func(t *testing.T) {
		tier, reader, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ4K, 0, false, true)
		if got := tier.Prefetch([]string{name}); got != 0 {
			t.Fatalf("Prefetch with a registered overlay admitted %d jobs, want 0", got)
		}
		if calls := reader.reads.Load(); calls != 0 {
			t.Fatalf("declined prefetch moved %d payload reads, want 0", calls)
		}
		w, err := tier.fault(name)
		if err != nil || !bytes.Equal(mappedCheckpointRaw(w), reader.data[:stride]) {
			t.Fatalf("declining the hint changed demand serving: %v", err)
		}
	})

	// A zero budget is the documented stream-through default, not a decline: the faulted slab is
	// handed to the caller and dropped, so a hint on a zero-budget tier still starts its read.
	t.Run("zero-budget-still-prefetches", func(t *testing.T) {
		tier, src, _, _ := oneCopyGateTier(t, 1, 0, 23)
		name := expertName(0, 0, "gate_proj.weight")
		if got := tier.Prefetch([]string{name}); got != 1 {
			t.Fatalf("Prefetch on a zero-budget tier admitted %d jobs, want 1", got)
		}
		waitPrefetchSettled(t, tier)
		if offsets, _ := src.recorded(); len(offsets) != 1 {
			t.Fatalf("zero-budget prefetch recorded %d reads, want 1", len(offsets))
		}
	})

	// A background panic must not terminate the process, must release the loading owner and the
	// aggregate job counter, and must leave the key retryable.
	t.Run("panic-settles-ownership-once", func(t *testing.T) {
		tier, reader, stride := checkpointFlightFixture(t, 0)
		name := expertName(0, 0, "gate_proj.weight")
		reader.panicValue = &struct{}{}
		reader.unblock()
		if got := tier.Prefetch([]string{name}); got != 1 {
			t.Fatalf("Prefetch admitted %d jobs, want 1", got)
		}
		waitPrefetchSettled(t, tier)
		tier.mu.Lock()
		jobs, _, loading := tier.prefetchJobs, len(tier.ready), len(tier.loading)
		tier.mu.Unlock()
		if jobs != 0 || loading != 0 {
			t.Fatalf("after a panicking prefetch: jobs=%d loading=%d, want 0 and 0", jobs, loading)
		}
		if st := tier.Stats(); st.Failures != 1 || st.Reads != 0 {
			t.Fatalf("panicking prefetch: failures=%d reads=%d, want 1 and 0", st.Failures, st.Reads)
		}
		reader.panicValue = nil
		w, err := tier.fault(name)
		if err != nil || w.q4 == nil || !bytes.Equal(w.q4.raw, reader.data[:stride]) {
			t.Fatalf("retry after a panicking prefetch: %v", err)
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestExpertCheckpointPrefetchDelayedGeneration(t *testing.T) {
	for _, tc := range []struct {
		name      string
		advance   int
		waiter    bool
		budget    int64
		wantReady bool
		wantStale int
	}{
		{name: "previous-generation", advance: 1, wantReady: true},
		{name: "obsolete-generation", advance: 2, wantStale: 1},
		{name: "obsolete-demand-waiter", advance: 2, waiter: true},
		{name: "obsolete-host-retention", advance: 2, budget: 3 * q4kBlockBytes},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const stride = q4kBlockBytes
			data := make([]byte, 3*stride)
			for i := range data {
				data[i] = byte(i%251 + 1)
			}
			reader := &checkpointFlightReader{data: data, entered: make(chan int64, 16), release: make(chan struct{})}
			t.Cleanup(reader.unblock)
			tier := NewExpertCheckpointTier(tc.budget)
			if err := tier.AddShard(reader, int64(len(data)), []FusedExpertTensor{{
				Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
				Quant: ExpertCheckpointQ4K, Experts: 3, Rows: 1, Cols: qkK,
			}}); err != nil {
				t.Fatal(err)
			}
			name := expertName(0, 0, "gate_proj.weight")
			id := polymodel.ModelID(name)
			var loads []*expertCheckpointLoad
			for generation := 0; generation <= tc.advance; generation++ {
				next := expertName(0, generation, "gate_proj.weight")
				if got := tier.Prefetch([]string{next}); got != 1 {
					t.Fatalf("generation %d started %d reads, want 1", generation+1, got)
				}
				tier.mu.Lock()
				load := tier.loading[polymodel.ModelID(next)]
				tier.mu.Unlock()
				if load == nil {
					t.Fatal("gated prefetch has no loading owner")
				}
				loads = append(loads, load)
				if generation == 0 {
					checkpointFlightWait(t, func() bool { return len(reader.entered) > 0 })
				}
			}
			var demand <-chan checkpointFlightResult
			if tc.waiter {
				demand = checkpointFlightFaults(tier, name)
				checkpointFlightWait(t, func() bool {
					tier.mu.Lock()
					defer tier.mu.Unlock()
					return loads[0].waiters == 1
				})
			}
			reader.unblock()
			for _, load := range loads {
				checkpointFlightWait(t, func() bool {
					select {
					case <-load.done:
						return true
					default:
						return false
					}
				})
			}
			if tc.waiter {
				checkpointFlightComplete(t, checkpointFlightCollect(t, demand, 1)[0], data[:stride])
			}
			tier.mu.Lock()
			_, ready := tier.ready[id]
			tier.mu.Unlock()
			if ready != tc.wantReady {
				t.Errorf("delayed handoff retained = %t, want %t", ready, tc.wantReady)
			}
			st := tier.Stats()
			if st.PrefetchStale != tc.wantStale {
				t.Errorf("stale handoffs = %d, want %d", st.PrefetchStale, tc.wantStale)
			}
			if calls := reader.calls.Load(); calls != int64(tc.advance+1) {
				t.Fatalf("prefetch and joined demand made %d reads, want %d", calls, tc.advance+1)
			}
			w, err := tier.fault(name)
			checkpointFlightComplete(t, checkpointFlightResult{weight: w, err: err}, data[:stride])
			wantReads := st.Reads
			if !tc.wantReady && tc.budget == 0 {
				wantReads++
			}
			if got := tier.Stats().Reads; got != wantReads {
				t.Errorf("reads after later demand = %d, want %d", got, wantReads)
			}
		})
	}
}

// checkpointPrefetchExitReader gates the first worker wave, then aborts selected
// offsets without returning. Its deferred active count runs before load settlement.
type checkpointPrefetchExitReader struct {
	data       []byte
	entered    chan int64
	release    chan struct{}
	releaseOne sync.Once
	mu         sync.Mutex
	abortBelow int64
	calls      int
	active     int
	peak       int
}

func (r *checkpointPrefetchExitReader) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	r.calls++
	r.active++
	r.peak = max(r.peak, r.active)
	abort := off < r.abortBelow
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		r.active--
		r.mu.Unlock()
	}()
	r.entered <- off
	<-r.release
	if abort {
		runtime.Goexit()
	}
	return copy(p, r.data[off:off+int64(len(p))]), nil
}

func (r *checkpointPrefetchExitReader) unblock() {
	r.releaseOne.Do(func() { close(r.release) })
}

// A load's Goexit finalizer is insufficient if it also retires the only worker
// that could drain queued owners. Both cases exhaust the initial worker wave;
// the all-aborted case also requires a finite succession through the entire queue.
// fak-test:runtime fast est=100ms lane=default
func TestExpertCheckpointPrefetchGoexitDrainsQueue(t *testing.T) {
	const jobs = fakExpertPrefetchJobs
	const stride = q4kBlockBytes
	for _, aborted := range []int{fakExpertPrefetchWorkers, jobs} {
		name := "first-wave"
		if aborted == jobs {
			name = "every-job"
		}
		t.Run(name, func(t *testing.T) {
			data := make([]byte, jobs*stride)
			for i := range data {
				data[i] = byte(i%251 + 1)
			}
			reader := &checkpointPrefetchExitReader{
				data: data, entered: make(chan int64, 2*jobs), release: make(chan struct{}),
				abortBelow: int64(aborted * stride),
			}
			t.Cleanup(reader.unblock)
			tier := NewExpertCheckpointTier(0)
			if err := tier.AddShard(reader, int64(len(data)), []FusedExpertTensor{{
				Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
				Quant: ExpertCheckpointQ4K, Experts: jobs, Rows: 1, Cols: qkK,
			}}); err != nil {
				t.Fatal(err)
			}
			names := make([]string, jobs)
			for i := range names {
				names[i] = expertName(0, i, "gate_proj.weight")
			}
			if got := tier.Prefetch(names); got != jobs {
				t.Fatalf("admitted %d jobs, want %d", got, jobs)
			}
			checkpointFlightWait(t, func() bool { return len(reader.entered) == fakExpertPrefetchWorkers })
			// Prefetch has returned with every reader slot blocked and further jobs
			// queued. All names already have owners before a demand can attach.
			tier.mu.Lock()
			loads := make([]*expertCheckpointLoad, jobs)
			for i, name := range names {
				loads[i] = tier.loading[polymodel.ModelID(name)]
			}
			pending := tier.prefetchJobs
			tier.mu.Unlock()
			if pending != jobs {
				t.Fatalf("pending jobs before release = %d, want %d", pending, jobs)
			}
			for i, load := range loads {
				if load == nil {
					t.Fatalf("admitted job %d has no loading owner", i)
				}
			}
			results := checkpointFlightFaults(tier, names[:fakExpertPrefetchWorkers]...)
			checkpointFlightWait(t, func() bool {
				tier.mu.Lock()
				defer tier.mu.Unlock()
				for _, load := range loads[:fakExpertPrefetchWorkers] {
					if load.waiters != 1 {
						return false
					}
				}
				return true
			})
			reader.unblock()
			// Only the first wave has test-owned demand goroutines. Collect them
			// before the tail watchdog, so the old broken queue fails without
			// leaving demand goroutines waiting on its stranded loading entries.
			for _, result := range checkpointFlightCollect(t, results, fakExpertPrefetchWorkers) {
				if !errors.Is(result.err, ErrExpertCheckpointLoadAborted) || result.panic != nil || result.weight.q4 != nil || result.weight.kq != nil {
					t.Fatalf("first-wave demand returned error=%v panic=%v or a weight", result.err, result.panic)
				}
			}
			deadline := time.NewTimer(5 * time.Second)
			defer deadline.Stop()
			for _, load := range loads {
				select {
				case <-load.done:
				case <-deadline.C:
					settled := 0
					for _, owner := range loads {
						select {
						case <-owner.done:
							settled++
						default:
						}
					}
					tier.mu.Lock()
					pending := tier.prefetchJobs
					tier.mu.Unlock()
					t.Fatalf("prefetch queue stranded: settled %d of %d loads after first-wave Goexit; pending=%d", settled, jobs, pending)
				}
			}
			for i, name := range names {
				result := checkpointFlightResult{weight: loads[i].weight, err: loads[i].err}
				if i < aborted {
					if !errors.Is(result.err, ErrExpertCheckpointLoadAborted) || result.panic != nil || result.weight.q4 != nil || result.weight.kq != nil {
						t.Fatalf("aborted job %d returned error=%v panic=%v or a weight", i, result.err, result.panic)
					}
				} else {
					checkpointFlightComplete(t, result, data[i*stride:(i+1)*stride])
					w, err := tier.fault(name)
					checkpointFlightComplete(t, checkpointFlightResult{weight: w, err: err}, data[i*stride:(i+1)*stride])
				}
			}
			tier.mu.Lock()
			pending, loading, ready := tier.prefetchJobs, len(tier.loading), len(tier.ready)
			tier.mu.Unlock()
			if pending != 0 || loading != 0 || ready != 0 {
				t.Fatalf("settled queue retained jobs=%d loading=%d ready=%d", pending, loading, ready)
			}
			reader.mu.Lock()
			calls, active, peak := reader.calls, reader.active, reader.peak
			reader.abortBelow = 0
			reader.mu.Unlock()
			if calls != jobs || active != 0 || peak != fakExpertPrefetchWorkers {
				t.Fatalf("source calls=%d active=%d peak=%d, want %d/0/%d", calls, active, peak, jobs, fakExpertPrefetchWorkers)
			}
			if st := tier.Stats(); st.Failures != aborted || st.Reads != jobs-aborted || st.Coalesced != fakExpertPrefetchWorkers {
				t.Fatalf("settled queue stats: %+v", st)
			}
			// Retry the aborted names through the real asynchronous API, proving
			// both loading ownership and the aggregate admission slots were released.
			if got := tier.Prefetch(names[:aborted]); got != aborted {
				t.Fatalf("retry admitted %d jobs, want %d", got, aborted)
			}
			waitPrefetchSettled(t, tier)
			for i, name := range names[:aborted] {
				w, err := tier.fault(name)
				checkpointFlightComplete(t, checkpointFlightResult{weight: w, err: err}, data[i*stride:(i+1)*stride])
			}
			reader.mu.Lock()
			calls, active, peak = reader.calls, reader.active, reader.peak
			reader.mu.Unlock()
			if calls != jobs+aborted || active != 0 || peak != fakExpertPrefetchWorkers {
				t.Fatalf("retry source calls=%d active=%d peak=%d, want %d/0/%d", calls, active, peak, jobs+aborted, fakExpertPrefetchWorkers)
			}
			if st := tier.Stats(); st.Failures != aborted || st.Reads != jobs {
				t.Fatalf("retry stats: %+v", st)
			}
		})
	}
}
