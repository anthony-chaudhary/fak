package cachemeta

// stream_metrics.go is the aggregate metric surface OVER the cachemeta.Entry stream
// — the one fold every cache plane lowers into. Each per-entry producer already
// keeps its own counters (the vDSO Stats(), the engine CacheEventMetrics, the bench
// Arm's provider split), but until now nothing summed the SHARED Entry shape across
// planes, so a scrape could see one cache's view but never "across ALL planes, what
// is the per-plane/per-residency-tier cache event mix." StreamMetrics is that
// missing fold: a producer calls Observe(kind, entry) on every cache lifecycle or
// lookup-verdict event and the surface buckets it by (plane, tier, kind).
//
// It is deliberately kind-agnostic — "fill"/"hit"/"evict"/"revoke" for the vDSO
// tier-2 tool cache (the first live producer, wired in the gateway), and later
// "hit"/"miss"/"fault" for a verdict-shaped producer such as the live-engine KV
// recorder — so the same surface unifies every level instead of inventing one
// counter family per cache. The package stays pure (abi only); the gateway owns the
// instance, subscribes it to the live vDSO cache-event sink, and renders it on
// /metrics alongside the fak_vdso_*/fak_engine_cache_* families.

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// StreamMetrics aggregates cachemeta.Entry events across every cache plane. The
// zero value is not ready; construct with NewStreamMetrics. Safe for concurrent
// use — producers call Observe from the cache hot path (outside their own locks).
type StreamMetrics struct {
	mu    sync.Mutex
	byKey map[streamKey]*streamAgg

	// headline totals, kept so a scrape need not re-sum the keyed map.
	events     uint64
	bytesMoved int64
}

type streamKey struct {
	plane string
	tier  string
	kind  string
}

type streamAgg struct {
	count uint64
	bytes int64
}

// NewStreamMetrics returns an empty, ready surface.
func NewStreamMetrics() *StreamMetrics {
	return &StreamMetrics{byKey: map[streamKey]*streamAgg{}}
}

// Observe folds one entry into the surface under a lifecycle/verdict kind (e.g.
// "fill"/"hit"/"evict"/"revoke" from the vDSO tier-2 cache, or "hit"/"miss"/"fault"
// from a verdict-shaped producer). An empty kind is recorded as "unknown" so a
// producer that forgets to label an event never silently drops it. A nil receiver
// is a no-op, so an unwired surface (no producer set) is always safe to call.
func (m *StreamMetrics) Observe(kind string, e Entry) {
	if m == nil {
		return
	}
	if kind == "" {
		kind = "unknown"
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.byKey == nil {
		m.byKey = map[streamKey]*streamAgg{}
	}
	key := streamKey{plane: string(e.Plane), tier: string(e.Residency.Tier), kind: kind}
	agg := m.byKey[key]
	if agg == nil {
		agg = &streamAgg{}
		m.byKey[key] = agg
	}
	agg.count++
	agg.bytes += e.Metrics.BytesTransferred
	m.events++
	m.bytesMoved += e.Metrics.BytesTransferred
}

// StreamSnapshot is an immutable read of the surface, safe to render without
// holding the lock.
type StreamSnapshot struct {
	Events     uint64
	BytesMoved int64
	Rows       []StreamRow
}

// StreamRow is one (plane, tier, kind) bucket.
type StreamRow struct {
	Plane string
	Tier  string
	Kind  string
	Count uint64
	Bytes int64
}

// Snapshot copies the current counters into a stable, sorted read.
func (m *StreamMetrics) Snapshot() StreamSnapshot {
	s := StreamSnapshot{}
	if m == nil {
		return s
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	s.Events = m.events
	s.BytesMoved = m.bytesMoved
	s.Rows = make([]StreamRow, 0, len(m.byKey))
	for k, agg := range m.byKey {
		s.Rows = append(s.Rows, StreamRow{
			Plane: k.plane,
			Tier:  k.tier,
			Kind:  k.kind,
			Count: agg.count,
			Bytes: agg.bytes,
		})
	}
	sort.Slice(s.Rows, func(i, j int) bool {
		a, b := s.Rows[i], s.Rows[j]
		if a.Plane != b.Plane {
			return a.Plane < b.Plane
		}
		if a.Tier != b.Tier {
			return a.Tier < b.Tier
		}
		return a.Kind < b.Kind
	})
	return s
}

// Prometheus renders the snapshot as Prometheus exposition text. Metric names are
// namespaced fak_cache_* so the unified-stream family sits alongside the per-cache
// fak_vdso_*/fak_engine_cache_* families. The block carries its own HELP/TYPE lines
// so it concatenates cleanly into a larger /metrics body. A tier label is emitted as
// "unknown" when an entry carries no residency tier, so the series shape is stable.
func (s StreamSnapshot) Prometheus() string {
	var b strings.Builder
	help := func(name, h, typ string) {
		b.WriteString("# HELP " + name + " " + h + "\n")
		b.WriteString("# TYPE " + name + " " + typ + "\n")
	}
	help("fak_cache_events_total", "cachemeta cache-entry lifecycle/verdict events folded across every cache plane (the unified-stream fold; vDSO tier-2 tool cache is the live producer today).", "counter")
	b.WriteString("fak_cache_events_total " + strconv.FormatUint(s.Events, 10) + "\n")
	help("fak_cache_bytes_moved_total", "Bytes moved across residency tiers by cache events (0 until a byte-moving producer, e.g. the live-engine KV recorder, feeds the stream; the local vDSO cache moves no bytes).", "counter")
	b.WriteString("fak_cache_bytes_moved_total " + strconv.FormatInt(s.BytesMoved, 10) + "\n")
	help("fak_cache_event_breakdown_total", "Cache events by plane, residency tier, and kind (fill/hit/evict/revoke for the local tool cache; hit/miss/fault for verdict-shaped producers).", "counter")
	for _, r := range s.Rows {
		plane := r.Plane
		if plane == "" {
			plane = "unknown"
		}
		tier := r.Tier
		if tier == "" {
			tier = "unknown"
		}
		b.WriteString("fak_cache_event_breakdown_total{plane=\"" + promStreamLabel(plane) +
			"\",tier=\"" + promStreamLabel(tier) + "\",kind=\"" + promStreamLabel(r.Kind) + "\"} " +
			strconv.FormatUint(r.Count, 10) + "\n")
	}
	return b.String()
}

func promStreamLabel(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}
