package gateway

import (
	"sort"
	"time"
)

// inflightSnapshot derives live signals from the set of requests running right
// now: how many are in flight per route, and the age of the oldest one. This is
// computed at scrape time precisely because the completion-time histograms can't
// see a request that has not finished; maxAge is the hung-request detector.
func (m *gatewayMetrics) inflightSnapshot(now time.Time) (byRoute map[string]int, maxAge float64) {
	byRoute = map[string]int{}
	if m == nil {
		return byRoute, 0
	}
	m.inflightMu.Lock()
	for _, e := range m.inflightReq {
		byRoute[e.route]++
		if age := now.Sub(e.start).Seconds(); age > maxAge {
			maxAge = age
		}
	}
	m.inflightMu.Unlock()
	return byRoute, maxAge
}

type latencySnapshot struct {
	count   uint64
	sum     float64
	buckets []uint64
}

func (c *latencyCounter) snapshot() latencySnapshot {
	return latencySnapshot{
		count:   c.count,
		sum:     c.sum,
		buckets: append([]uint64(nil), c.buckets...),
	}
}

type inferenceSnapshot struct {
	reqs           map[string]uint64
	promptTok      uint64
	complTok       uint64
	cachedTok      uint64
	cacheCreateTok uint64
	cachedHits     uint64
	decodeSecs     float64
	// prefillSecs is the cumulative TTFT wall-clock over the ttftTurns that measured
	// it; prefillPromptTok is the prompt-token sum over those same turns. ttftTurns is
	// the denominator that keeps the prefill rate honest on a mixed workload.
	prefillSecs      float64
	ttftTurns        uint64
	prefillPromptTok uint64
	// measuredDecodeSecs / measuredComplTok are the decode-rate pair over ttftTurns
	// only (see the accumulator doc) — kept separate so a mixed workload never blends
	// measured and unmeasured denominators.
	measuredDecodeSecs float64
	measuredComplTok   uint64
	// Latency-distribution snapshots (see the inferTTFTHist/TPOT/E2E accumulators).
	ttftHist latencySnapshot
	tpotHist latencySnapshot
	e2eHist  latencySnapshot
}

type compactionSnapshot struct {
	attempts            map[string]uint64
	bailReasons         map[string]uint64
	dropped             uint64
	restored            uint64
	shed                uint64
	cacheReads          uint64
	lastCacheRd         float64
	anchorStarved       uint64
	thrashSessions      uint64
	solvencyForced      uint64
	lastSuffixTokens    uint64
	peakSuffixTokens    uint64
	uncachedTrimResults uint64
	uncachedTrimShed    uint64
	ttlUpgrades         map[string]uint64
	placementAttempts   map[string]uint64
}

type ctxViewRewriteSnapshot struct {
	events  uint64
	dropped uint64
	shed    uint64
}

type requestMemoryAggregateSnapshot struct {
	observed map[string]uint64
	plans    []requestMemoryPlanSnapshot
	tokens   []requestMemoryTokenSnapshot
	fits     []requestMemoryFitSnapshot
}

type requestMemoryPlanSnapshot struct {
	key            requestMemoryMetricKey
	observations   uint64
	totalBytes     uint64
	highWaterBytes int64
}

type requestMemoryTokenSnapshot struct {
	key          requestMemoryTokenKey
	observations uint64
	total        uint64
	highWater    int
}

type requestMemoryFitSnapshot struct {
	key            requestMemoryFitKey
	observations   uint64
	wantHighWater  int64
	marginLowWater int64
	marginKnown    bool
}

type inKernelOOMSnapshot struct {
	class           string
	count           uint64
	failedBytes     uint64
	lastFailedBytes uint64
	lastSite        string
}

func (m *gatewayMetrics) inferenceSnapshotData() inferenceSnapshot {
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	reqs := make(map[string]uint64, len(m.inferReqs))
	for k, v := range m.inferReqs {
		reqs[k] = v
	}
	return inferenceSnapshot{
		reqs:               reqs,
		promptTok:          m.inferPromptTokens,
		complTok:           m.inferComplTokens,
		cachedTok:          m.inferCachedTokens,
		cacheCreateTok:     m.inferCacheCreationTokens,
		cachedHits:         m.inferCachedHits,
		decodeSecs:         m.inferDecodeSecs,
		prefillSecs:        m.inferPrefillSecs,
		ttftTurns:          m.inferTTFTTurns,
		prefillPromptTok:   m.inferPrefillPromptTokens,
		measuredDecodeSecs: m.inferMeasuredDecodeSecs,
		measuredComplTok:   m.inferMeasuredComplTokens,
		ttftHist:           histSnapshot(m.inferTTFTHist),
		tpotHist:           histSnapshot(m.inferTPOTHist),
		e2eHist:            histSnapshot(m.inferE2EHist),
	}
}

// histSnapshot reads a latency histogram, tolerating a nil counter (a
// directly-constructed gatewayMetrics that rendered before any turn) by returning an
// empty but correctly-sized snapshot so writeHistogram never indexes a nil bucket slice.
func histSnapshot(c *latencyCounter) latencySnapshot {
	if c == nil {
		return latencySnapshot{buckets: make([]uint64, len(gatewayLatencyBuckets))}
	}
	return c.snapshot()
}

func (m *gatewayMetrics) compactionSnapshotData() compactionSnapshot {
	if m == nil {
		return compactionSnapshot{
			attempts:    map[string]uint64{},
			bailReasons: map[string]uint64{},
		}
	}
	m.compactMu.Lock()
	defer m.compactMu.Unlock()
	attempts := map[string]uint64{}
	for k, v := range m.compactAttempts {
		attempts[k] = v
	}
	bailReasons := map[string]uint64{}
	for k, v := range m.compactBailReasons {
		bailReasons[k] = v
	}
	ttlUpgrades := map[string]uint64{}
	for k, v := range m.ttlUpgrades {
		ttlUpgrades[k] = v
	}
	placementAttempts := map[string]uint64{}
	for k, v := range m.placementAttempts {
		placementAttempts[k] = v
	}
	return compactionSnapshot{
		attempts:            attempts,
		bailReasons:         bailReasons,
		ttlUpgrades:         ttlUpgrades,
		placementAttempts:   placementAttempts,
		dropped:             m.compactDropped,
		restored:            m.compactRestored,
		shed:                m.compactShed,
		cacheReads:          m.compactCacheReads,
		lastCacheRd:         m.compactLastCacheRd,
		anchorStarved:       m.compactAnchorStarved,
		thrashSessions:      m.compactThrashSessions,
		solvencyForced:      m.compactSolvencyForced,
		lastSuffixTokens:    m.compactLastSuffixTokens,
		peakSuffixTokens:    m.compactPeakSuffixTokens,
		uncachedTrimResults: m.uncachedTrimResults,
		uncachedTrimShed:    m.uncachedTrimShed,
	}
}

func (m *gatewayMetrics) ctxViewRewriteSnapshotData() ctxViewRewriteSnapshot {
	if m == nil {
		return ctxViewRewriteSnapshot{}
	}
	m.ctxViewMu.Lock()
	defer m.ctxViewMu.Unlock()
	return ctxViewRewriteSnapshot{
		events:  m.ctxViewEvents,
		dropped: m.ctxViewDropped,
		shed:    m.ctxViewShed,
	}
}

func (m *gatewayMetrics) requestMemoryAggregateSnapshotData() requestMemoryAggregateSnapshot {
	if m == nil {
		return requestMemoryAggregateSnapshot{observed: map[string]uint64{}}
	}
	m.reqMemoryMu.Lock()
	defer m.reqMemoryMu.Unlock()
	out := requestMemoryAggregateSnapshot{observed: map[string]uint64{}}
	for backend, n := range m.reqMemoryObserved {
		out.observed[backend] = n
	}
	for key, st := range m.reqMemoryPlan {
		if st == nil {
			continue
		}
		out.plans = append(out.plans, requestMemoryPlanSnapshot{
			key:            key,
			observations:   st.observations,
			totalBytes:     st.totalBytes,
			highWaterBytes: st.highWaterBytes,
		})
	}
	for key, st := range m.reqMemoryTokens {
		if st == nil {
			continue
		}
		out.tokens = append(out.tokens, requestMemoryTokenSnapshot{
			key:          key,
			observations: st.observations,
			total:        st.total,
			highWater:    st.highWater,
		})
	}
	for key, st := range m.reqMemoryFit {
		if st == nil {
			continue
		}
		out.fits = append(out.fits, requestMemoryFitSnapshot{
			key:            key,
			observations:   st.observations,
			wantHighWater:  st.wantHighWater,
			marginLowWater: st.marginLowWater,
			marginKnown:    st.marginKnown,
		})
	}
	sort.SliceStable(out.plans, func(i, j int) bool {
		a, b := out.plans[i].key, out.plans[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		if a.scope != b.scope {
			return a.scope < b.scope
		}
		if a.class != b.class {
			return a.class < b.class
		}
		return a.dtype < b.dtype
	})
	sort.SliceStable(out.tokens, func(i, j int) bool {
		a, b := out.tokens[i].key, out.tokens[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		return a.kind < b.kind
	})
	sort.SliceStable(out.fits, func(i, j int) bool {
		a, b := out.fits[i].key, out.fits[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		return a.scope < b.scope
	})
	return out
}

func (m *gatewayMetrics) inKernelOOMSnapshotData() []inKernelOOMSnapshot {
	if m == nil {
		return nil
	}
	m.oomMu.Lock()
	defer m.oomMu.Unlock()
	out := make([]inKernelOOMSnapshot, 0, len(m.inKernelOOM))
	for class, st := range m.inKernelOOM {
		if st == nil {
			continue
		}
		out = append(out, inKernelOOMSnapshot{
			class:           class,
			count:           st.count,
			failedBytes:     st.failedBytes,
			lastFailedBytes: st.lastFailedBytes,
			lastSite:        st.lastSite,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].class < out[j].class })
	return out
}

func (m *gatewayMetrics) snapshot() ([]httpMetricSnapshot, []operationMetricSnapshot) {
	m.mu.Lock()
	defer m.mu.Unlock()
	httpRows := make([]httpMetricSnapshot, 0, len(m.http))
	for k, v := range m.http {
		httpRows = append(httpRows, httpMetricSnapshot{key: k, val: v.snapshot()})
	}
	sort.Slice(httpRows, func(i, j int) bool {
		a, b := httpRows[i].key, httpRows[j].key
		if a.route != b.route {
			return a.route < b.route
		}
		if a.method != b.method {
			return a.method < b.method
		}
		return a.status < b.status
	})
	opRows := make([]operationMetricSnapshot, 0, len(m.operations))
	for k, v := range m.operations {
		opRows = append(opRows, operationMetricSnapshot{key: k, val: v.snapshot()})
	}
	sort.Slice(opRows, func(i, j int) bool {
		a, b := opRows[i].key, opRows[j].key
		if a.operation != b.operation {
			return a.operation < b.operation
		}
		if a.verdict != b.verdict {
			return a.verdict < b.verdict
		}
		if a.reason != b.reason {
			return a.reason < b.reason
		}
		if a.refusalSubtype != b.refusalSubtype {
			return a.refusalSubtype < b.refusalSubtype
		}
		if a.disposition != b.disposition {
			return a.disposition < b.disposition
		}
		return a.by < b.by
	})
	return httpRows, opRows
}

// resetShadowSnapshot is a lock-free copy of the resetScore SHADOW accumulators for rendering.
type resetShadowSnapshot struct {
	reasons   map[string]uint64
	recommend uint64
	lastScore float64
}

func (m *gatewayMetrics) resetShadowSnapshotData() resetShadowSnapshot {
	out := resetShadowSnapshot{reasons: map[string]uint64{}}
	if m == nil {
		return out
	}
	m.resetShadowMu.Lock()
	for k, v := range m.resetShadowReasons {
		out.reasons[k] = v
	}
	out.recommend = m.resetShadowRecommend
	out.lastScore = m.resetShadowLastScore
	m.resetShadowMu.Unlock()
	return out
}
