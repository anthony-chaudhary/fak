package cachemeta

// shadowmap.go reconstructs the REALIZED per-entry lifecycle from a stream of
// cache events — the post-hoc counterpart to lifecycle.go's live forward state
// machine. lifecycle.go answers "what state is this entry in NOW"; the existing
// stream_metrics.go fold answers "how many events of each kind happened" but
// discards per-entry identity, so nothing in the package could answer "how long
// did entries actually LIVE, how long did they sit IDLE before eviction, and how
// far apart were their reuses." (Technique borrowed clean-room from LMCache's
// l0_lifecycle shadow map; see issue #3392.)
//
// ShadowMap is that reconstruction: feed it admit/access/evict events (entry id
// + injected millisecond timestamp — the same wall-clock-free posture as
// lifecycle.go's nowMillis) and it pairs them per entry, folding three realized
// distributions into bucketed histograms:
//
//   - lifetime         = evict_ts - admit_ts
//   - idle-before-evict = evict_ts - last_access_ts (admit counts as the first
//     "access" for a never-touched entry, mirroring NewLifecycle's
//     LastAccessMillis = nowMillis)
//   - reuse-gap        = gap between CONSECUTIVE accesses of the same entry
//     (the admit->first-access span is a fill-to-first-use latency, not a reuse
//     gap, so it is deliberately NOT folded here)
//
// Memory stays bounded by the LIVE population: an evict releases the entry's
// shadow state and returns its realized ShadowRecord to the caller instead of
// retaining it. Events that cannot be paired (an access/evict for an unknown or
// empty id, an unknown kind, a re-admit that supersedes a live shadow) are never
// silently dropped — they are counted in the snapshot's Orphans, the same
// no-silent-drop posture as StreamMetrics' "unknown" kind.
//
// The Event struct is deliberately local and minimal (kind + id + millis): the
// richer engine.CacheEvent lives in a higher tier and this package is tier-1
// foundation (see doc.go), so producers adapt DOWN to ShadowEvent rather than
// this package importing up.

import (
	"sort"
	"sync"
)

// ShadowEventKind is the lifecycle edge a ShadowEvent carries. The set is the
// minimal alphabet a realized-lifecycle reconstruction needs; producers with a
// richer vocabulary (fill/hit/revoke, demote/promote) map onto it (fill->admit,
// hit->access, revoke/evict->evict) before feeding the shadow map.
type ShadowEventKind string

const (
	// ShadowAdmit — the entry became resident (its realized lifetime clock starts).
	ShadowAdmit ShadowEventKind = "admit"
	// ShadowAccess — the entry served a reuse hit.
	ShadowAccess ShadowEventKind = "access"
	// ShadowEvict — the entry was removed (its realized lifetime clock stops).
	ShadowEvict ShadowEventKind = "evict"
)

// ShadowEvent is one cache event as the shadow map consumes it: which edge, for
// which entry, at what injected millisecond clock. Timestamps are caller-supplied
// (never sampled here) so a recorded workload replays deterministically.
type ShadowEvent struct {
	Kind    ShadowEventKind
	EntryID string
	Millis  int64
}

// ShadowRecord is one entry's REALIZED lifecycle, produced when its evict pairs
// with its admit. Durations are clamped at zero so an out-of-order stream can
// only understate, never fabricate a negative bucket.
type ShadowRecord struct {
	EntryID        string
	AdmitMillis    int64
	EvictMillis    int64
	LifetimeMillis int64 // EvictMillis - AdmitMillis
	// IdleMillis is evict minus the last access (or minus admit when the entry
	// was never accessed — idle since admission).
	IdleMillis int64
	Accesses   uint64
	// ReuseGapsMillis are the realized gaps between consecutive accesses, in
	// stream order. Empty for an entry accessed fewer than twice.
	ReuseGapsMillis []int64
}

// ShadowHistogram is a simple bucketed histogram over millisecond durations.
// BoundsMillis are ascending inclusive upper bounds; Counts has one extra
// trailing bucket for values above the last bound. A value v lands in the first
// bucket i with v <= BoundsMillis[i] (Prometheus "le" semantics).
type ShadowHistogram struct {
	BoundsMillis []int64
	Counts       []uint64
}

// Total sums the histogram's bucket counts (the number of durations folded).
func (h ShadowHistogram) Total() uint64 {
	var n uint64
	for _, c := range h.Counts {
		n += c
	}
	return n
}

// ShadowSnapshot is an immutable read of the three realized-lifecycle
// histograms plus the pairing bookkeeping, safe to render without the lock.
type ShadowSnapshot struct {
	Lifetime        ShadowHistogram
	IdleBeforeEvict ShadowHistogram
	ReuseGap        ShadowHistogram
	// Completed counts entries whose admit->evict span was fully realized (one
	// per ShadowRecord returned).
	Completed uint64
	// Live counts entries admitted but not yet evicted (their lifetime/idle are
	// not yet folded; their reuse gaps already are, realized at access time).
	Live int
	// Orphans counts events that could not be paired into any entry's realized
	// lifecycle: an access/evict for an unknown or empty id, an unknown kind,
	// or a live shadow superseded by a re-admit.
	Orphans uint64
}

// defaultShadowBoundsMillis spans sub-millisecond-adjacent tool caches up
// through hour-scale KV residency: 1ms, 10ms, 100ms, 1s, 10s, 1min, 5min, 1h.
var defaultShadowBoundsMillis = []int64{1, 10, 100, 1_000, 10_000, 60_000, 300_000, 3_600_000}

// shadowEntry is the live per-entry shadow state between admit and evict.
type shadowEntry struct {
	admitMillis      int64
	lastAccessMillis int64
	accesses         uint64
	reuseGapsMillis  []int64
}

// ShadowMap pairs admit/access/evict events per entry and folds the realized
// durations into three histograms. The zero value is not ready; construct with
// NewShadowMap or NewShadowMapWithBounds. Safe for concurrent use; a nil
// receiver is a no-op, so an unwired consumer is always safe to call.
type ShadowMap struct {
	mu     sync.Mutex
	bounds []int64
	live   map[string]*shadowEntry

	lifetime []uint64
	idle     []uint64
	reuseGap []uint64

	completed uint64
	orphans   uint64
}

// NewShadowMap returns an empty shadow map over the default millisecond bucket
// boundaries (1ms .. 1h; see defaultShadowBoundsMillis).
func NewShadowMap() *ShadowMap {
	return NewShadowMapWithBounds(nil)
}

// NewShadowMapWithBounds returns an empty shadow map over the given ascending
// inclusive upper bounds in milliseconds. The slice is copied, sorted, and
// deduplicated, and non-positive bounds are dropped (a duration is never
// negative after clamping, so they could never match). An empty result falls
// back to the default boundaries.
func NewShadowMapWithBounds(boundsMillis []int64) *ShadowMap {
	bounds := make([]int64, 0, len(boundsMillis))
	for _, b := range boundsMillis {
		if b > 0 {
			bounds = append(bounds, b)
		}
	}
	sort.Slice(bounds, func(i, j int) bool { return bounds[i] < bounds[j] })
	dedup := bounds[:0]
	for i, b := range bounds {
		if i == 0 || b != bounds[i-1] {
			dedup = append(dedup, b)
		}
	}
	bounds = dedup
	if len(bounds) == 0 {
		bounds = append([]int64(nil), defaultShadowBoundsMillis...)
	}
	return &ShadowMap{
		bounds:   bounds,
		live:     map[string]*shadowEntry{},
		lifetime: make([]uint64, len(bounds)+1),
		idle:     make([]uint64, len(bounds)+1),
		reuseGap: make([]uint64, len(bounds)+1),
	}
}

// Observe folds one event into the shadow map. When the event is an evict that
// pairs with a live admit, it returns the entry's realized ShadowRecord and
// true — the caller owns the record (the map retains nothing for completed
// entries, keeping memory bounded by the live population). Every other event
// returns a zero record and false. Unpairable events are counted as orphans,
// never silently dropped. A nil receiver is a no-op.
func (m *ShadowMap) Observe(ev ShadowEvent) (ShadowRecord, bool) {
	if m == nil {
		return ShadowRecord{}, false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if ev.EntryID == "" {
		m.orphans++
		return ShadowRecord{}, false
	}
	switch ev.Kind {
	case ShadowAdmit:
		if _, alreadyLive := m.live[ev.EntryID]; alreadyLive {
			// A re-admit supersedes the live shadow: the old span never realized
			// an evict, so it is counted as unpaired rather than fabricated.
			m.orphans++
		}
		m.live[ev.EntryID] = &shadowEntry{
			admitMillis: ev.Millis,
			// Admit counts as the first "access" for idle purposes, mirroring
			// NewLifecycle's LastAccessMillis = nowMillis.
			lastAccessMillis: ev.Millis,
		}
	case ShadowAccess:
		se := m.live[ev.EntryID]
		if se == nil {
			m.orphans++
			return ShadowRecord{}, false
		}
		if se.accesses > 0 {
			// Only a gap between two ACCESSES is a reuse gap; admit->first-access
			// is fill-to-first-use latency, deliberately not folded here.
			gap := clampNonNegative(ev.Millis - se.lastAccessMillis)
			se.reuseGapsMillis = append(se.reuseGapsMillis, gap)
			m.reuseGap[shadowBucket(m.bounds, gap)]++
		}
		se.accesses++
		se.lastAccessMillis = ev.Millis
	case ShadowEvict:
		se := m.live[ev.EntryID]
		if se == nil {
			m.orphans++
			return ShadowRecord{}, false
		}
		delete(m.live, ev.EntryID)
		rec := ShadowRecord{
			EntryID:         ev.EntryID,
			AdmitMillis:     se.admitMillis,
			EvictMillis:     ev.Millis,
			LifetimeMillis:  clampNonNegative(ev.Millis - se.admitMillis),
			IdleMillis:      clampNonNegative(ev.Millis - se.lastAccessMillis),
			Accesses:        se.accesses,
			ReuseGapsMillis: se.reuseGapsMillis,
		}
		m.lifetime[shadowBucket(m.bounds, rec.LifetimeMillis)]++
		m.idle[shadowBucket(m.bounds, rec.IdleMillis)]++
		m.completed++
		return rec, true
	default:
		m.orphans++
	}
	return ShadowRecord{}, false
}

// Histograms copies the three realized-lifecycle histograms and the pairing
// bookkeeping into a stable snapshot.
func (m *ShadowMap) Histograms() ShadowSnapshot {
	s := ShadowSnapshot{}
	if m == nil {
		return s
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	s.Lifetime = ShadowHistogram{BoundsMillis: append([]int64(nil), m.bounds...), Counts: append([]uint64(nil), m.lifetime...)}
	s.IdleBeforeEvict = ShadowHistogram{BoundsMillis: append([]int64(nil), m.bounds...), Counts: append([]uint64(nil), m.idle...)}
	s.ReuseGap = ShadowHistogram{BoundsMillis: append([]int64(nil), m.bounds...), Counts: append([]uint64(nil), m.reuseGap...)}
	s.Completed = m.completed
	s.Live = len(m.live)
	s.Orphans = m.orphans
	return s
}

// shadowBucket returns the index of the first bound >= v, or the trailing
// overflow bucket when v exceeds every bound. Bounds are few and sorted; a
// linear scan keeps the fold allocation-free and obvious.
func shadowBucket(bounds []int64, v int64) int {
	for i, b := range bounds {
		if v <= b {
			return i
		}
	}
	return len(bounds)
}

// clampNonNegative floors a duration at zero: an out-of-order stream can only
// understate a realized duration, never fold a negative one.
func clampNonNegative(d int64) int64 {
	if d < 0 {
		return 0
	}
	return d
}
