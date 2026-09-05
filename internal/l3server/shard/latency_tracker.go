package shard

import (
	"sort"
	"time"
)

const latencyRingSize = 256

// opLatencyTracker is a fixed-size ring buffer that tracks recent op latencies
// for computing p99 during migration. Single-threaded only (shard goroutine).
type opLatencyTracker struct {
	ring  [latencyRingSize]int64 // nanoseconds
	pos   int
	count int
}

// record inserts a duration sample into the ring buffer.
func (t *opLatencyTracker) record(d time.Duration) {
	t.ring[t.pos] = d.Nanoseconds()
	t.pos = (t.pos + 1) % latencyRingSize
	if t.count < latencyRingSize {
		t.count++
	}
}

// percentile returns the p-th percentile latency (p in 0..100).
// Returns 0 if fewer than 10 samples have been recorded.
func (t *opLatencyTracker) percentile(p int) time.Duration {
	if t.count < 10 {
		return 0
	}

	// Copy to scratch array and sort
	var scratch [latencyRingSize]int64
	n := t.count
	if n > latencyRingSize {
		n = latencyRingSize
	}

	// Copy the most recent n entries
	if t.count >= latencyRingSize {
		copy(scratch[:], t.ring[:])
	} else {
		copy(scratch[:n], t.ring[:n])
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
	t.count = 0
}

// opLatencyTrackers tracks latency per operation category.
type opLatencyTrackers struct {
	all       opLatencyTracker // every op
	get       opLatencyTracker // OpGet + OpMGet + OpMGetWithAlloc
	set       opLatencyTracker // OpSet + OpMSet
	exists    opLatencyTracker // OpTest
	queueWait opLatencyTracker // time from enqueue to dequeue
	allocDur  opLatencyTracker // time in allocWithEvictionIn (including retries/sweep)
}

// record routes a latency sample to the appropriate category tracker(s).
func (lt *opLatencyTrackers) record(opType OpType, d time.Duration) {
	lt.all.record(d)
	switch opType {
	case OpGet, OpMGet, OpMGetWithAlloc:
		lt.get.record(d)
	case OpSet, OpMSet:
		lt.set.record(d)
	case OpTest:
		lt.exists.record(d)
	}
}
