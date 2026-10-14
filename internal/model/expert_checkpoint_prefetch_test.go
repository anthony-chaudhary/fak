package model

import (
	"bytes"
	"math/rand"
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
