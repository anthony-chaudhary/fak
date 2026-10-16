package compute

import "sync/atomic"

// transientPoolGauge mirrors a backend's request-transient reuse pool in
// atomics, so a metrics scrape reads it without taking the backend lock that a
// forward pass holds. Callers mutate the pool under that lock and record here.
type transientPoolGauge struct {
	buffers atomic.Int64
	bytes   atomic.Int64
	peak    atomic.Int64
}

func (g *transientPoolGauge) parked(nbytes int) {
	g.buffers.Add(1)
	now := g.bytes.Add(int64(nbytes))
	for {
		peak := g.peak.Load()
		if now <= peak || g.peak.CompareAndSwap(peak, now) {
			return
		}
	}
}

func (g *transientPoolGauge) reused(nbytes int) {
	g.buffers.Add(-1)
	g.bytes.Add(-int64(nbytes))
}

func (g *transientPoolGauge) drained() {
	g.buffers.Store(0)
	g.bytes.Store(0)
}

// TransientPoolStats reports pooled buffers, pooled bytes, and the pooled-bytes
// high-water mark since start.
func (g *transientPoolGauge) TransientPoolStats() (pooledBuffers, pooledBytes, peakPooledBytes int64) {
	return g.buffers.Load(), g.bytes.Load(), g.peak.Load()
}
