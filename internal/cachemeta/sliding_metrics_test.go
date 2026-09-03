package cachemeta

import (
	"math"
	"sync"
	"testing"
)

// TestSlidingCacheMetricsRecentVsLifetime verifies the core requirement of #10728:
// recording 100 requests with 100% hit rate followed by 50 requests with 0% hit rate in
// a window of N=100 yields RecentHitRate() == 0.50 while LifetimeHitRate() == 0.6667.
func TestSlidingCacheMetricsRecentVsLifetime(t *testing.T) {
	m := NewSlidingCacheMetrics(100)

	// 100 requests with 100% hit rate (1 hit, 1 query each)
	for i := 0; i < 100; i++ {
		m.RecordRequest(true, 10)
	}

	snap1 := m.Snapshot()
	if snap1.WindowRequests != 100 || snap1.RecentHitRate != 1.0 || snap1.LifetimeHitRate != 1.0 {
		t.Fatalf("after 100 hits: windowRequests=%d recent=%v lifetime=%v, want 100/1.0/1.0",
			snap1.WindowRequests, snap1.RecentHitRate, snap1.LifetimeHitRate)
	}

	// 50 requests with 0% hit rate (0 hits, 1 query each)
	for i := 0; i < 50; i++ {
		m.RecordRequest(false, 10)
	}

	snap2 := m.Snapshot()
	if snap2.WindowRequests != 100 {
		t.Fatalf("after 150 total requests: windowRequests=%d, want 100", snap2.WindowRequests)
	}
	if snap2.TotalRequests != 150 {
		t.Fatalf("after 150 total requests: totalRequests=%d, want 150", snap2.TotalRequests)
	}

	// Recent hit rate must be exactly 50/100 = 0.50
	const eps = 1e-6
	if math.Abs(snap2.RecentHitRate-0.50) > eps {
		t.Fatalf("RecentHitRate = %v, want 0.50", snap2.RecentHitRate)
	}

	// Lifetime hit rate must be 100/150 = ~0.6667
	expectedLifetime := 100.0 / 150.0
	if math.Abs(snap2.LifetimeHitRate-expectedLifetime) > eps {
		t.Fatalf("LifetimeHitRate = %v, want %v", snap2.LifetimeHitRate, expectedLifetime)
	}
}

// TestSlidingCacheMetricsIdleUpdateSuppression verifies that empty observation passes
// (queries=0, hits=0, tokens=0) do not evict operational evidence or alter metrics.
func TestSlidingCacheMetricsIdleUpdateSuppression(t *testing.T) {
	m := NewSlidingCacheMetrics(10)

	// Record 5 active requests
	for i := 0; i < 5; i++ {
		m.Record(1, 1, 10)
	}
	initialRate := m.RecentHitRate()
	if initialRate != 1.0 {
		t.Fatalf("initial RecentHitRate = %v, want 1.0", initialRate)
	}

	// Simulate 100 empty/idle ticks
	for i := 0; i < 100; i++ {
		m.Record(0, 0, 0)
	}

	snap := m.Snapshot()
	if snap.WindowRequests != 5 {
		t.Fatalf("after idle ticks: windowRequests=%d, want 5 (idle updates must not evict)", snap.WindowRequests)
	}
	if snap.TotalRequests != 5 {
		t.Fatalf("after idle ticks: totalRequests=%d, want 5", snap.TotalRequests)
	}
	if snap.RecentHitRate != 1.0 {
		t.Fatalf("after idle ticks: RecentHitRate = %v, want 1.0", snap.RecentHitRate)
	}
}

// TestSlidingCacheMetricsReset verifies reset behavior.
func TestSlidingCacheMetricsReset(t *testing.T) {
	m := NewSlidingCacheMetrics(10)
	m.RecordRequest(true, 50)
	m.RecordRequest(false, 50)

	m.Reset()
	snap := m.Snapshot()
	if snap.WindowRequests != 0 || snap.TotalRequests != 0 || snap.RecentHitRate != 0.0 || snap.LifetimeHitRate != 0.0 {
		t.Fatalf("after reset: %+v, want all zero", snap)
	}
}

// TestSlidingCacheMetricsConcurrent ensures thread safety under concurrent writers and readers.
func TestSlidingCacheMetricsConcurrent(t *testing.T) {
	m := NewSlidingCacheMetrics(50)
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				m.RecordRequest(j%2 == 0, int64(workerID*10))
				_ = m.RecentHitRate()
				_ = m.Snapshot()
			}
		}(i)
	}
	wg.Wait()

	snap := m.Snapshot()
	if snap.TotalRequests != 1000 {
		t.Fatalf("totalRequests = %d, want 1000", snap.TotalRequests)
	}
}
