// Package cachemeta provides metadata, tiering, and operational governance for the KV cache.
package cachemeta

import (
	"sync"
)

// DefaultSlidingWindowSize is the default number of requests held in the sliding window (vLLM CachingMetrics default 1000).
const DefaultSlidingWindowSize = 1000

// CacheSample captures cache activity for a single request or scheduling pass.
type CacheSample struct {
	Queries int64 `json:"queries"`
	Hits    int64 `json:"hits"`
	Tokens  int64 `json:"tokens"`
}

// IsEmpty reports whether the sample represents zero activity.
func (s CacheSample) IsEmpty() bool {
	return s.Queries == 0 && s.Hits == 0 && s.Tokens == 0
}

// SlidingMetricsSnapshot holds a point-in-time view of sliding and lifetime cache metrics.
type SlidingMetricsSnapshot struct {
	RecentHitRate   float64 `json:"recent_hit_rate"`
	LifetimeHitRate float64 `json:"lifetime_hit_rate"`
	WindowRequests  int     `json:"window_requests"`
	WindowCapacity  int     `json:"window_capacity"`
	WindowQueries   int64   `json:"window_queries"`
	WindowHits      int64   `json:"window_hits"`
	WindowTokens    int64   `json:"window_tokens"`
	TotalRequests   int64   `json:"total_requests"`
	TotalQueries    int64   `json:"total_queries"`
	TotalHits       int64   `json:"total_hits"`
	TotalTokens     int64   `json:"total_tokens"`
}

// SlidingCacheMetrics maintains an O(1) bounded sliding-window hit-rate metric with
// running sums and empty-update suppression (#10728, borrowed from vLLM CachingMetrics).
// It prevents long-running sessions from diluting acute cache degradation while ensuring
// idle periods never evict operational evidence.
type SlidingCacheMetrics struct {
	mu sync.Mutex

	capacity int
	ring     []CacheSample
	head     int
	count    int

	// Running window sums (maintained in O(1) time)
	windowQueries int64
	windowHits    int64
	windowTokens  int64

	// Cumulative lifetime sums
	totalRequests int64
	totalQueries  int64
	totalHits     int64
	totalTokens   int64
}

// NewSlidingCacheMetrics constructs a new sliding-window cache metrics tracker.
// If capacity <= 0, DefaultSlidingWindowSize is used.
func NewSlidingCacheMetrics(capacity int) *SlidingCacheMetrics {
	if capacity <= 0 {
		capacity = DefaultSlidingWindowSize
	}
	return &SlidingCacheMetrics{
		capacity: capacity,
		ring:     make([]CacheSample, capacity),
	}
}

// Record observes one request's cache activity. Empty updates (queries == 0 && hits == 0 && tokens == 0)
// are suppressed so idle scheduler passes do not push useful samples out of the window.
func (m *SlidingCacheMetrics) Record(queries, hits, tokens int64) {
	if m == nil {
		return
	}
	sample := CacheSample{Queries: queries, Hits: hits, Tokens: tokens}
	if sample.IsEmpty() {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Update cumulative lifetime counters
	m.totalRequests++
	m.totalQueries += queries
	m.totalHits += hits
	m.totalTokens += tokens

	// If ring slot has an existing sample, subtract it from running window sums
	if m.count == m.capacity {
		old := m.ring[m.head]
		m.windowQueries -= old.Queries
		m.windowHits -= old.Hits
		m.windowTokens -= old.Tokens
	} else {
		m.count++
	}

	// Store new sample and update running window sums
	m.ring[m.head] = sample
	m.windowQueries += queries
	m.windowHits += hits
	m.windowTokens += tokens

	m.head = (m.head + 1) % m.capacity
}

// RecordRequest records a single request outcome (hit or miss) with token count.
func (m *SlidingCacheMetrics) RecordRequest(hit bool, tokens int64) {
	hits := int64(0)
	if hit {
		hits = 1
	}
	m.Record(1, hits, tokens)
}

// RecentHitRate returns the hit rate over the sliding window [0.0, 1.0].
func (m *SlidingCacheMetrics) RecentHitRate() float64 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.windowQueries <= 0 {
		return 0
	}
	return float64(m.windowHits) / float64(m.windowQueries)
}

// LifetimeHitRate returns the hit rate across all observed non-empty requests [0.0, 1.0].
func (m *SlidingCacheMetrics) LifetimeHitRate() float64 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.totalQueries <= 0 {
		return 0
	}
	return float64(m.totalHits) / float64(m.totalQueries)
}

// Snapshot returns a point-in-time copy of all metrics.
func (m *SlidingCacheMetrics) Snapshot() SlidingMetricsSnapshot {
	if m == nil {
		return SlidingMetricsSnapshot{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	recentRate := 0.0
	if m.windowQueries > 0 {
		recentRate = float64(m.windowHits) / float64(m.windowQueries)
	}
	lifetimeRate := 0.0
	if m.totalQueries > 0 {
		lifetimeRate = float64(m.totalHits) / float64(m.totalQueries)
	}

	return SlidingMetricsSnapshot{
		RecentHitRate:   recentRate,
		LifetimeHitRate: lifetimeRate,
		WindowRequests:  m.count,
		WindowCapacity:  m.capacity,
		WindowQueries:   m.windowQueries,
		WindowHits:      m.windowHits,
		WindowTokens:    m.windowTokens,
		TotalRequests:   m.totalRequests,
		TotalQueries:    m.totalQueries,
		TotalHits:       m.totalHits,
		TotalTokens:     m.totalTokens,
	}
}

// Reset clears all windowed and cumulative metrics.
func (m *SlidingCacheMetrics) Reset() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	m.head = 0
	m.count = 0
	m.windowQueries = 0
	m.windowHits = 0
	m.windowTokens = 0
	m.totalRequests = 0
	m.totalQueries = 0
	m.totalHits = 0
	m.totalTokens = 0
	for i := range m.ring {
		m.ring[i] = CacheSample{}
	}
}
