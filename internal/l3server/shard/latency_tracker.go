package shard

import (
	"sort"
	"sync/atomic"
	"time"
)

const latencyRingSize = 256

// opLatencyTracker is a fixed-size ring buffer that tracks recent op latencies
// for computing recent operation percentiles. record and reset must
// remain serialized on the shard goroutine; concurrent readers load each sample
// atomically, but a percentile is an approximate, possibly mixed-window snapshot.
type opLatencyTracker struct {
	ring  [latencyRingSize]atomic.Int64 // nanoseconds
	pos   int                           // single writer only
	count atomic.Int64
}

// record inserts a duration sample into the ring buffer.
func (t *opLatencyTracker) record(d time.Duration) {
	t.ring[t.pos].Store(d.Nanoseconds())
	t.pos = (t.pos + 1) % latencyRingSize
	if count := t.count.Load(); count < latencyRingSize {
		t.count.Store(count + 1)
	}
}

// percentile returns the p-th percentile latency (p in 0..100).
// Returns 0 if fewer than 10 samples have been recorded.
func (t *opLatencyTracker) percentile(p int) time.Duration {
	n := int(t.count.Load())
	if n < 10 {
		return 0
	}

	// Copy to scratch array and sort
	var scratch [latencyRingSize]int64
	if n > latencyRingSize {
		n = latencyRingSize
	}

	// Copy samples individually; the writer may advance or reset meanwhile.
	for i := 0; i < n; i++ {
		scratch[i] = t.ring[i].Load()
	}

	sort.Slice(scratch[:n], func(i, j int) bool {
		return scratch[i] < scratch[j]
	})

	idx := (n * p) / 100
	if idx >= n {
		idx = n - 1
	}
	return time.Duration(scratch[idx])
}

// p50 returns the 50th percentile (median) latency.
func (t *opLatencyTracker) p50() time.Duration {
	return t.percentile(50)
}

// p99 returns the 99th percentile latency from the ring buffer.
func (t *opLatencyTracker) p99() time.Duration {
	return t.percentile(99)
}

// reset clears all tracked latencies.
func (t *opLatencyTracker) reset() {
	t.pos = 0
	t.count.Store(0)
}

// opLatencyTrackers tracks latency per operation category.
type opLatencyTrackers struct {
	all       opLatencyTracker // every op
	get       opLatencyTracker // OpGet + OpGetAlloc + OpMGet + OpMGetWithAlloc
	set       opLatencyTracker // OpSet + OpMSet
	exists    opLatencyTracker // OpTest
	queueWait opLatencyTracker // time from enqueue to dequeue
	allocDur  opLatencyTracker // time in allocWithEvictionIn (including retries/sweep)
}

// record routes a latency sample to the appropriate category tracker(s).
func (lt *opLatencyTrackers) record(opType OpType, d time.Duration) {
	lt.all.record(d)
	switch opType {
	case OpGet, OpGetAlloc, OpMGet, OpMGetWithAlloc:
		lt.get.record(d)
	case OpSet, OpMSet:
		lt.set.record(d)
	case OpTest:
		lt.exists.record(d)
	}
}
