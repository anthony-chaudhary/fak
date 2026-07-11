package cachemeta

import (
	"reflect"
	"testing"
)

// TestShadowMapRealizedHistograms feeds a small synthetic event stream with known
// admit/access/evict timings and asserts both the derived per-entry durations and
// the exact histogram bucket counts.
func TestShadowMapRealizedHistograms(t *testing.T) {
	// Buckets: (<=100], (100,1000], (1000,2000], (2000,+inf).
	m := NewShadowMapWithBounds([]int64{100, 1000, 2000})

	// Entry "a": lives 3000ms, last touched 1600ms before evict, one reuse gap of
	// 1000ms (accesses at 1400 and 2400; admit->first-access is NOT a reuse gap).
	if _, done := m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "a", Millis: 1000}); done {
		t.Fatalf("admit must not complete a record")
	}
	if _, done := m.Observe(ShadowEvent{Kind: ShadowAccess, EntryID: "a", Millis: 1400}); done {
		t.Fatalf("access must not complete a record")
	}
	m.Observe(ShadowEvent{Kind: ShadowAccess, EntryID: "a", Millis: 2400})

	// Entry "b": admitted at 2000, never accessed, evicted 50ms later — idle since
	// admission, so idle == lifetime == 50.
	m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "b", Millis: 2000})
	recB, done := m.Observe(ShadowEvent{Kind: ShadowEvict, EntryID: "b", Millis: 2050})
	if !done {
		t.Fatalf("evict of live entry b did not complete a record")
	}
	wantB := ShadowRecord{EntryID: "b", AdmitMillis: 2000, EvictMillis: 2050, LifetimeMillis: 50, IdleMillis: 50}
	if !reflect.DeepEqual(recB, wantB) {
		t.Fatalf("record b = %+v, want %+v", recB, wantB)
	}

	recA, done := m.Observe(ShadowEvent{Kind: ShadowEvict, EntryID: "a", Millis: 4000})
	if !done {
		t.Fatalf("evict of live entry a did not complete a record")
	}
	wantA := ShadowRecord{
		EntryID:         "a",
		AdmitMillis:     1000,
		EvictMillis:     4000,
		LifetimeMillis:  3000,
		IdleMillis:      1600,
		Accesses:        2,
		ReuseGapsMillis: []int64{1000},
	}
	if !reflect.DeepEqual(recA, wantA) {
		t.Fatalf("record a = %+v, want %+v", recA, wantA)
	}

	s := m.Histograms()
	if s.Completed != 2 || s.Live != 0 || s.Orphans != 0 {
		t.Fatalf("bookkeeping = completed %d live %d orphans %d, want 2/0/0", s.Completed, s.Live, s.Orphans)
	}
	wantBounds := []int64{100, 1000, 2000}
	for name, h := range map[string]ShadowHistogram{
		"lifetime": s.Lifetime, "idle": s.IdleBeforeEvict, "reuse-gap": s.ReuseGap,
	} {
		if !reflect.DeepEqual(h.BoundsMillis, wantBounds) {
			t.Fatalf("%s bounds = %v, want %v", name, h.BoundsMillis, wantBounds)
		}
	}
	// Lifetimes: b=50 -> bucket 0 (<=100); a=3000 -> overflow bucket 3 (>2000).
	if want := []uint64{1, 0, 0, 1}; !reflect.DeepEqual(s.Lifetime.Counts, want) {
		t.Fatalf("lifetime counts = %v, want %v", s.Lifetime.Counts, want)
	}
	// Idle: b=50 -> bucket 0; a=1600 -> bucket 2 (1000 < 1600 <= 2000).
	if want := []uint64{1, 0, 1, 0}; !reflect.DeepEqual(s.IdleBeforeEvict.Counts, want) {
		t.Fatalf("idle counts = %v, want %v", s.IdleBeforeEvict.Counts, want)
	}
	// Reuse gaps: the single 1000ms gap sits exactly ON a bound -> bucket 1 (le).
	if want := []uint64{0, 1, 0, 0}; !reflect.DeepEqual(s.ReuseGap.Counts, want) {
		t.Fatalf("reuse-gap counts = %v, want %v", s.ReuseGap.Counts, want)
	}
	if s.Lifetime.Total() != 2 || s.IdleBeforeEvict.Total() != 2 || s.ReuseGap.Total() != 1 {
		t.Fatalf("totals = %d/%d/%d, want 2/2/1",
			s.Lifetime.Total(), s.IdleBeforeEvict.Total(), s.ReuseGap.Total())
	}
}

// TestShadowMapLiveEntryFoldTiming asserts a still-live entry contributes its
// reuse gaps (realized at access time) but no lifetime/idle (not yet realized),
// and is counted in Live.
func TestShadowMapLiveEntryFoldTiming(t *testing.T) {
	m := NewShadowMapWithBounds([]int64{10, 100})
	m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "x", Millis: 0})
	m.Observe(ShadowEvent{Kind: ShadowAccess, EntryID: "x", Millis: 5})
	m.Observe(ShadowEvent{Kind: ShadowAccess, EntryID: "x", Millis: 12}) // gap 7 -> bucket 0

	s := m.Histograms()
	if s.Live != 1 || s.Completed != 0 {
		t.Fatalf("live %d completed %d, want 1/0", s.Live, s.Completed)
	}
	if s.Lifetime.Total() != 0 || s.IdleBeforeEvict.Total() != 0 {
		t.Fatalf("live entry folded lifetime/idle early: %v / %v", s.Lifetime.Counts, s.IdleBeforeEvict.Counts)
	}
	if want := []uint64{1, 0, 0}; !reflect.DeepEqual(s.ReuseGap.Counts, want) {
		t.Fatalf("reuse-gap counts = %v, want %v", s.ReuseGap.Counts, want)
	}
}

// TestShadowMapOrphansAndClamp asserts unpairable events are counted, never
// folded, and an out-of-order evict clamps to a zero duration instead of going
// negative.
func TestShadowMapOrphansAndClamp(t *testing.T) {
	m := NewShadowMapWithBounds([]int64{10})

	m.Observe(ShadowEvent{Kind: ShadowAccess, EntryID: "ghost", Millis: 1}) // unknown id
	m.Observe(ShadowEvent{Kind: ShadowEvict, EntryID: "ghost", Millis: 2})  // unknown id
	m.Observe(ShadowEvent{Kind: "demote", EntryID: "ghost", Millis: 3})     // unknown kind
	m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "", Millis: 4})       // empty id
	m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "r", Millis: 100})
	m.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "r", Millis: 200}) // re-admit supersedes

	// The superseding admit restarts r's shadow: an evict BEFORE the new admit
	// timestamp clamps to a zero lifetime in the first bucket.
	rec, done := m.Observe(ShadowEvent{Kind: ShadowEvict, EntryID: "r", Millis: 150})
	if !done || rec.LifetimeMillis != 0 || rec.IdleMillis != 0 || rec.AdmitMillis != 200 {
		t.Fatalf("clamped record = %+v done=%v, want zero durations from admit 200", rec, done)
	}

	s := m.Histograms()
	if s.Orphans != 5 {
		t.Fatalf("orphans = %d, want 5", s.Orphans)
	}
	if s.Completed != 1 || s.Live != 0 {
		t.Fatalf("completed %d live %d, want 1/0", s.Completed, s.Live)
	}
	if want := []uint64{1, 0}; !reflect.DeepEqual(s.Lifetime.Counts, want) {
		t.Fatalf("lifetime counts = %v, want %v", s.Lifetime.Counts, want)
	}
}

// TestShadowMapBoundsNormalizationAndNilSafe asserts constructor bound hygiene
// (sort, dedup, drop non-positive, default fallback) and the package's standard
// nil-receiver no-op posture.
func TestShadowMapBoundsNormalizationAndNilSafe(t *testing.T) {
	m := NewShadowMapWithBounds([]int64{500, -3, 10, 500, 0})
	s := m.Histograms()
	if want := []int64{10, 500}; !reflect.DeepEqual(s.Lifetime.BoundsMillis, want) {
		t.Fatalf("normalized bounds = %v, want %v", s.Lifetime.BoundsMillis, want)
	}
	if len(s.Lifetime.Counts) != 3 {
		t.Fatalf("counts len = %d, want bounds+1 = 3", len(s.Lifetime.Counts))
	}

	d := NewShadowMap()
	if got := d.Histograms().Lifetime.BoundsMillis; !reflect.DeepEqual(got, defaultShadowBoundsMillis) {
		t.Fatalf("default bounds = %v, want %v", got, defaultShadowBoundsMillis)
	}
	if got := NewShadowMapWithBounds(nil).Histograms().Lifetime.BoundsMillis; !reflect.DeepEqual(got, defaultShadowBoundsMillis) {
		t.Fatalf("nil-bounds fallback = %v, want %v", got, defaultShadowBoundsMillis)
	}

	var nilM *ShadowMap
	if _, done := nilM.Observe(ShadowEvent{Kind: ShadowAdmit, EntryID: "x", Millis: 1}); done {
		t.Fatalf("nil Observe must be a no-op")
	}
	if got := nilM.Histograms(); got.Completed != 0 || got.Live != 0 || got.Lifetime.Counts != nil {
		t.Fatalf("nil Histograms not empty: %+v", got)
	}
}
