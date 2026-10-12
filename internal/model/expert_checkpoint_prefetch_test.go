package model

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
	"time"
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
