package model

import (
	"bytes"
	"runtime"
	"testing"
	"weak"
)

type checkpointJoinReachabilityReader struct {
	data  []byte
	probe chan weak.Pointer[byte]
}

func (r *checkpointJoinReachabilityReader) ReadAt(p []byte, off int64) (int, error) {
	// This is the source's actual independently allocated read destination,
	// not the fixture backing bytes. The weak observation owns no payload.
	r.probe <- weak.Make(&p[0])
	return bytes.NewReader(r.data).ReadAt(p, off)
}

// Keep all strong payload locals in this frame, which returns before collection.
// No finalizer or timing-sensitive callback is used. The allocation is 9216B,
// well above the runtime's tiny-object batching range.
func checkpointJoinReachabilityFixture(t *testing.T) (func(), weak.Pointer[byte]) {
	t.Helper()
	const rows = 64
	data := bytes.Repeat([]byte{0x35}, rows*q4kBlockBytes)
	r := &checkpointJoinReachabilityReader{data: data, probe: make(chan weak.Pointer[byte], 1)}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(r, int64(len(data)), []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ4K, Experts: 1, Rows: rows, Cols: qkK,
	}}); err != nil {
		t.Fatal(err)
	}
	name := expertName(0, 0, "gate_proj.weight")
	n, join := tier.prefetchJoined([]string{name})
	if n != 1 {
		t.Fatalf("started %d loads, want 1", n)
	}
	join()
	join() // Completion remains idempotent.
	p := <-r.probe
	w, err := tier.fault(name) // Consume the only ready handoff.
	if err != nil || w.q4 == nil || !bytes.Equal(w.q4.raw, data) {
		t.Fatalf("completed demand did not receive exact bytes: %v", err)
	}
	runtime.GC()
	if p.Value() == nil {
		t.Fatal("reachability control lost a still-owned demand payload")
	}
	runtime.KeepAlive(w)
	tier.mu.Lock()
	empty := len(tier.host) == 0 && len(tier.ready) == 0 && len(tier.loading) == 0 && tier.prefetchJobs == 0
	tier.mu.Unlock()
	if !empty {
		t.Fatal("fixture has remaining cache/job ownership; cannot isolate join retention")
	}
	return join, p
}

func checkpointJoinCollected(p weak.Pointer[byte]) bool {
	// Bounded repeated complete collections, not elapsed-time polling. The
	// dropped-join control below must prove this runtime actually reclaims the
	// same fixture; lack of collection is not interpreted as a device result.
	for i := 0; i < 32; i++ {
		runtime.GC()
		runtime.Gosched()
		if p.Value() == nil {
			return true
		}
	}
	return false
}

func checkpointJoinDroppedProbe(t *testing.T) weak.Pointer[byte] {
	t.Helper()
	_, p := checkpointJoinReachabilityFixture(t)
	return p
}

// fak-test:justify why=regression when=changed:internal/model/expert_checkpoint_tier.go
// fak-test:runtime fast est=500ms lane=default
func TestExpertCheckpointJoinedCompletionReleasesPayload(t *testing.T) {
	// First distinguish a reachability defect from a collector/fixture problem.
	if !checkpointJoinCollected(checkpointJoinDroppedProbe(t)) {
		t.Fatal("GC control did not reclaim dropped-join payload; retention witness is inconclusive")
	}
	join, p := checkpointJoinReachabilityFixture(t)
	collected := checkpointJoinCollected(p)
	join() // Still usable after demand consumption and collection.
	runtime.KeepAlive(join)
	if !collected {
		t.Fatal("completed join retains consumed checkpoint payload after cache/job ownership ended")
	}
}
