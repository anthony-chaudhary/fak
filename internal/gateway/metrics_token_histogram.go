package gateway

import "math"

// tokenHistogramBucketCount bounds the token-count histogram ladder regardless of the
// deployment's ceiling: a fixed 100 edges keeps /metrics scrape size flat whether the
// model window is 4K or 1M tokens.
const tokenHistogramBucketCount = 100

// defaultTokenHistogramMax scales the registered input/output token histograms until the
// deployment's configured total-token ceiling lands (scaleTokenHistograms): 128K sits on
// the common context rungs (defaultBudgetContexts) so P50/P95/P99 stay resolvable out of
// the box instead of inheriting the seconds ladder's shape.
const defaultTokenHistogramMax = 131072

// tokenBuckets derives a strictly increasing histogram bucket ladder scaled to the
// deployment's maximum token count. Adapted from TGI's auto-scaled token histograms
// (huggingface/text-generation-inference router/src/server.rs:1948-1985 @
// instead of a fixed grid, edge i of 1..100 spans i/100 of the ceiling, so the top edge
// lands exactly on maxTokens and each bucket covers ~1% of the range the deployment can
// actually serve. Each edge is the correctly rounded value of maxTokens*i/100 (exact
// int64 arithmetic while it fits), which keeps the ladder strictly increasing for every
// positive maxTokens. A non-positive ceiling yields no ladder (nil).
func tokenBuckets(maxTokens int) []float64 {
	if maxTokens <= 0 {
		return nil
	}
	buckets := make([]float64, tokenHistogramBucketCount)
	if int64(maxTokens) > math.MaxInt64/tokenHistogramBucketCount {
		// Beyond int64-exact scaling (far past any real token ceiling): float-step ladder.
		step := float64(maxTokens) / float64(tokenHistogramBucketCount)
		for i := 1; i <= tokenHistogramBucketCount; i++ {
			buckets[i-1] = step * float64(i)
		}
	} else {
		for i := 1; i <= tokenHistogramBucketCount; i++ {
			buckets[i-1] = float64(int64(maxTokens)*int64(i)) / float64(tokenHistogramBucketCount)
		}
	}
	buckets[tokenHistogramBucketCount-1] = float64(maxTokens) // pin the top edge exactly
	return buckets
}

// tokenHistogram is a Prometheus-style cumulative histogram over per-turn TOKEN COUNTS
// (input prompt tokens / output completion tokens). Unlike latencyCounter — whose bucket
// positions pair with the fixed seconds ladder gatewayLatencyBuckets — it carries its own
// scaled edges (tokenBuckets), so percentile queries stay meaningful at whatever scale the
// deployment serves. Empty (count 0) until a turn is observed: no phantom distribution.
type tokenHistogram struct {
	edges   []float64
	count   uint64
	sum     uint64
	buckets []uint64 // cumulative: buckets[i] counts observations <= edges[i]
}

func newTokenHistogram(maxTokens int) *tokenHistogram {
	edges := tokenBuckets(maxTokens)
	return &tokenHistogram{edges: edges, buckets: make([]uint64, len(edges))}
}

// observeLocked records one turn's token count into the cumulative buckets. Callers hold
// inferenceMu (the same lock the infer* gauge folds take); a non-positive count records
// nothing, matching the inferPromptTokens/inferComplTokens convention.
func (h *tokenHistogram) observeLocked(tokens int) {
	if h == nil || tokens <= 0 {
		return
	}
	h.count++
	h.sum += uint64(tokens)
	v := float64(tokens)
	for i, le := range h.edges {
		if v <= le {
			h.buckets[i]++
		}
	}
}

// scaleLocked re-derives the bucket ladder from a newly learned deployment ceiling. It
// applies only while the histogram has observed no turn yet: re-bucketing live data would
// silently drop the accumulated distribution, so a populated histogram keeps its ladder
// for the process lifetime (TGI derives its ladders once at startup; this mirrors that).
// Callers hold inferenceMu.
func (h *tokenHistogram) scaleLocked(maxTokens int) {
	if h == nil || maxTokens <= 0 || h.count != 0 {
		return
	}
	h.edges = tokenBuckets(maxTokens)
	h.buckets = make([]uint64, len(h.edges))
}

// tokenHistogramSnapshot is the scrape-time copy of a tokenHistogram, carrying its own
// edges because they are NOT the shared gatewayLatencyBuckets ladder.
type tokenHistogramSnapshot struct {
	count   uint64
	sum     uint64
	buckets []uint64
	edges   []float64
}

// snapshot copies the histogram for rendering. Callers hold inferenceMu.
func (h *tokenHistogram) snapshot() tokenHistogramSnapshot {
	if h == nil {
		return tokenHistogramSnapshot{}
	}
	out := tokenHistogramSnapshot{
		count:   h.count,
		sum:     h.sum,
		edges:   append([]float64(nil), h.edges...),
		buckets: make([]uint64, len(h.buckets)),
	}
	copy(out.buckets, h.buckets)
	return out
}

// scaleTokenHistograms re-derives the input/output token bucket ladders from the
// deployment's configured total-token ceiling — the TGI-adapted auto-scale the fixed
// seconds ladder cannot express. Populated histograms keep their existing ladder (see
// scaleLocked); a non-positive ceiling is a no-op.
func (m *gatewayMetrics) scaleTokenHistograms(maxTokens int) {
	if m == nil || maxTokens <= 0 {
		return
	}
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	m.inferInputTokenHist.scaleLocked(maxTokens)
	m.inferOutputTokenHist.scaleLocked(maxTokens)
}

// observeInferenceTokenCountsLocked folds one served turn's provider-reported token counts
// into the input/output token distributions. Callers hold inferenceMu; the production
// accounting fold calls it in the same critical section as inferPromptTokens /
// inferComplTokens, using the same normalized uncached input and output axes.
func (m *gatewayMetrics) observeInferenceTokenCountsLocked(promptTok, complTok int) {
	if m == nil {
		return
	}
	if m.inferInputTokenHist == nil {
		m.inferInputTokenHist = newTokenHistogram(defaultTokenHistogramMax)
	}
	if m.inferOutputTokenHist == nil {
		m.inferOutputTokenHist = newTokenHistogram(defaultTokenHistogramMax)
	}
	m.inferInputTokenHist.observeLocked(promptTok)
	m.inferOutputTokenHist.observeLocked(complTok)
}
