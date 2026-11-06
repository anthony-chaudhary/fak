package shard

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=50ms lane=race
// Estimate only: finite in-memory samples and snapshots; no shard worker or allocator.
func TestLatencySnapshotConcurrentSamples(t *testing.T) {
	t.Parallel()
	s := &Shard{latency: &opLatencyTrackers{}, ops: make(chan ShardOp, 1)}
	kinds := [...]OpType{OpGet, OpSet, OpTest}
	// Seed each category past its minimum so scrapes cannot all take the empty
	// fast path if this reader runs before the writer is first scheduled.
	for i := 0; i < 30; i++ {
		d := time.Duration(i+1) * time.Microsecond
		s.latency.record(kinds[i%len(kinds)], d)
		s.latency.queueWait.record(d)
		s.latency.allocDur.record(d)
	}
	start := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-start
		for i := 0; i < 512; i++ {
			d := time.Duration(i%64+1) * time.Microsecond
			s.latency.record(kinds[i%len(kinds)], d)
			s.latency.queueWait.record(d)
			s.latency.allocDur.record(d)
			if i%128 == 127 {
				// Reset is serialized with record on this sole writer.
				s.latency.all.reset()
			}
		}
	}()
	defer func() { <-done }()
	close(start)
	for i := 0; i < 128; i++ {
		all50, all99, get50, get99, set50, set99, exists50, exists99, wait50, wait99, alloc50, alloc99, depth, capacity := s.OpLatencySnapshot()
		// Independent samples may straddle reset or wrap. Zero means insufficient
		// samples; every published nonzero duration is in this fixed interval.
		for _, value := range []int64{all50, all99, get50, get99, set50, set99, exists50, exists99, wait50, wait99, alloc50, alloc99} {
			if value < 0 || value > 64 {
				t.Fatalf("snapshot duration %d us is outside the published range", value)
			}
		}
		if depth != 0 || capacity != 1 {
			t.Fatalf("queue snapshot=(%d,%d), want (0,1)", depth, capacity)
		}
	}
	<-done
	// Transfer writer ownership only after the goroutine has retired, then verify
	// quiescent reset and a fully replaced ring still have exact semantics.
	s.latency.all.reset()
	if s.latency.all.p50() != 0 || s.latency.all.p99() != 0 {
		t.Fatal("reset retained visible samples")
	}
	for i := 0; i < latencyRingSize; i++ {
		s.latency.all.record(17 * time.Microsecond)
	}
	if s.latency.all.p50() != 17*time.Microsecond || s.latency.all.p99() != 17*time.Microsecond {
		t.Fatal("quiescent constant window did not preserve percentiles")
	}
}
