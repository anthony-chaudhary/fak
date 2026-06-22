package cachemeta

import (
	"strings"
	"testing"
)

// entry builds a minimal Entry for a given plane/tier/bytes — enough to exercise the
// StreamMetrics fold without constructing a full adapter payload.
func streamEntry(plane Plane, tier ResidencyTier, bytesMoved int64) Entry {
	return Entry{
		Plane:     plane,
		Residency: Residency{Tier: tier},
		Metrics:   Metrics{BytesTransferred: bytesMoved},
	}
}

// TestStreamMetricsFoldsByPlaneTierKind asserts the unified fold buckets events by
// (plane, tier, kind), keeps honest headline totals, and sums bytes-moved.
func TestStreamMetricsFoldsByPlaneTierKind(t *testing.T) {
	m := NewStreamMetrics()

	// Two vDSO tier-2 tool-cache lifecycle events (no bytes moved — a local cache).
	m.Observe("fill", streamEntry(PlaneToolResult, TierDRAM, 0))
	m.Observe("hit", streamEntry(PlaneToolResult, TierDRAM, 0))
	m.Observe("hit", streamEntry(PlaneToolResult, TierDRAM, 0))
	// A verdict-shaped KV-transfer event that DID move bytes across tiers.
	m.Observe("fault", streamEntry(PlaneKVTransfer, TierRemote, 4096))

	s := m.Snapshot()
	if s.Events != 4 {
		t.Fatalf("Events = %d, want 4", s.Events)
	}
	if s.BytesMoved != 4096 {
		t.Fatalf("BytesMoved = %d, want 4096", s.BytesMoved)
	}

	// The two tool-cache hits collapse into one (plane,tier,kind) bucket with count 2.
	got := map[string]uint64{}
	for _, r := range s.Rows {
		got[r.Plane+"/"+r.Tier+"/"+r.Kind] = r.Count
	}
	for key, want := range map[string]uint64{
		"tool_result/dram/fill":   1,
		"tool_result/dram/hit":    2,
		"kv_transfer/remote/fault": 1,
	} {
		if got[key] != want {
			t.Fatalf("bucket %q count = %d, want %d (rows=%+v)", key, got[key], want, s.Rows)
		}
	}

	// Rows are sorted by (plane, tier, kind) for a stable scrape.
	for i := 1; i < len(s.Rows); i++ {
		a, b := s.Rows[i-1], s.Rows[i]
		if a.Plane > b.Plane || (a.Plane == b.Plane && a.Tier > b.Tier) ||
			(a.Plane == b.Plane && a.Tier == b.Tier && a.Kind > b.Kind) {
			t.Fatalf("rows not sorted at %d: %+v then %+v", i, a, b)
		}
	}
}

// TestStreamMetricsPrometheus asserts the exposition text carries the unified family
// with its own HELP/TYPE headers and the labeled breakdown.
func TestStreamMetricsPrometheus(t *testing.T) {
	m := NewStreamMetrics()
	m.Observe("fill", streamEntry(PlaneToolResult, TierDRAM, 0))
	m.Observe("evict", streamEntry(PlaneToolResult, TierDRAM, 0))

	prom := m.Snapshot().Prometheus()
	for _, want := range []string{
		"# TYPE fak_cache_events_total counter",
		"fak_cache_events_total 2",
		"# TYPE fak_cache_bytes_moved_total counter",
		"fak_cache_bytes_moved_total 0",
		"# TYPE fak_cache_event_breakdown_total counter",
		`fak_cache_event_breakdown_total{plane="tool_result",tier="dram",kind="fill"} 1`,
		`fak_cache_event_breakdown_total{plane="tool_result",tier="dram",kind="evict"} 1`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("prometheus missing %q\n--- prom ---\n%s", want, prom)
		}
	}
}

// TestStreamMetricsEmptyKindAndNilSafe confirms an empty kind is recorded as "unknown"
// and a nil surface is a no-op (an unwired producer never panics).
func TestStreamMetricsEmptyKindAndNilSafe(t *testing.T) {
	var nilM *StreamMetrics
	nilM.Observe("hit", streamEntry(PlaneBlob, TierDisk, 0)) // must not panic
	if got := nilM.Snapshot(); got.Events != 0 || len(got.Rows) != 0 {
		t.Fatalf("nil snapshot not empty: %+v", got)
	}

	m := NewStreamMetrics()
	m.Observe("", streamEntry(PlaneBlob, TierDisk, 0))
	s := m.Snapshot()
	if len(s.Rows) != 1 || s.Rows[0].Kind != "unknown" {
		t.Fatalf("empty kind not normalized to unknown: %+v", s.Rows)
	}
	// A missing plane/tier renders as the stable "unknown" label, not an empty string.
	prom := s.Prometheus()
	if !strings.Contains(prom, `kind="unknown"`) {
		t.Fatalf("prometheus missing unknown-kind row:\n%s", prom)
	}
}
