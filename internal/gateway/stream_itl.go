package gateway

import "time"

// streamITL measures REAL inter-token latency on one live streamed request: the
// wall gap between consecutive content-delta emissions, the fak analogue of vLLM's
// vllm:inter_token_latency_seconds (which observes every gap between consecutive
// output tokens, not a per-turn average). The per-turn mean (decode wall-clock /
// generated tokens) is TPOT and stays on its own histogram; aliasing it into ITL hid
// the tail stalls ITL exists to expose — e.g. another request's prefill interrupting
// this request's decode shows up here as one long gap, while the mean barely moves.
//
// It is a plain value embedded in the stream's own state: no allocation per token,
// just one time.Time compare and one histogram observe. The first mark only arms the
// tracker (the first-token boundary is TTFT, not an inter-token gap). Buffered
// (non-stream) turns never mark, so they contribute no ITL sample at all rather than
// a fabricated one.
type streamITL struct {
	last time.Time
	max  time.Duration
	gaps uint64
}

// mark records one content-delta emission at now and observes the gap since the
// previous emission. A non-monotonic now (clock skew) re-arms without observing.
func (t *streamITL) mark(m *gatewayMetrics, now time.Time) {
	if t == nil {
		return
	}
	if !t.last.IsZero() {
		if gap := now.Sub(t.last); gap >= 0 {
			m.observeITLGap(gap)
			if gap > t.max {
				t.max = gap
			}
			t.gaps++
		}
	}
	t.last = now
}

// finish folds this request's worst stall (max ITL) into the per-request max
// histogram, once, and resets the tracker. A request with fewer than two content
// deltas has no gap and contributes nothing.
func (t *streamITL) finish(m *gatewayMetrics) {
	if t == nil {
		return
	}
	if t.gaps > 0 {
		m.observeMaxITL(t.max)
	}
	*t = streamITL{}
}

// observeITLGap records one real inter-token gap.
func (m *gatewayMetrics) observeITLGap(gap time.Duration) {
	if m == nil || gap < 0 {
		return
	}
	m.itlMu.Lock()
	if m.inferITLHist == nil {
		m.inferITLHist = newLatencyCounter()
	}
	m.inferITLHist.observe(gap.Seconds())
	m.itlMu.Unlock()
}

// observeMaxITL records one request's worst inter-token gap.
func (m *gatewayMetrics) observeMaxITL(max time.Duration) {
	if m == nil || max < 0 {
		return
	}
	m.itlMu.Lock()
	if m.inferMaxITLHist == nil {
		m.inferMaxITLHist = newLatencyCounter()
	}
	m.inferMaxITLHist.observe(max.Seconds())
	m.itlMu.Unlock()
}

// itlSnapshot reads both ITL histograms under itlMu.
func (m *gatewayMetrics) itlSnapshot() (itl, maxITL latencySnapshot) {
	if m == nil {
		return histSnapshot(nil), histSnapshot(nil)
	}
	m.itlMu.Lock()
	defer m.itlMu.Unlock()
	return histSnapshot(m.inferITLHist), histSnapshot(m.inferMaxITLHist)
}
