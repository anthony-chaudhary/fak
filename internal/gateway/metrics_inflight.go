package gateway

import (
	"sort"
	"sync"
	"time"
)

// ProgressRegistry is the per-gateway live-heartbeat table keyed by the
// beginInflight request id (#12760): attachProgress registers the heartbeat of
// a stream at start, endInflight unregisters it at completion or cancel, so
// the observation plane can always resolve a live row to its tick source.
// Guarded by its own mutex, never held across a registry fold.
type ProgressRegistry struct {
	mu sync.Mutex
	hb map[uint64]*heartbeatConfig
}

// RegisterProgress binds hb to id for the life of the stream. Nil-safe: a nil
// receiver, a zero id, or a nil heartbeat is a no-op. Re-registering an id
// replaces the prior heartbeat.
func (m *gatewayMetrics) RegisterProgress(id uint64, hb *heartbeatConfig) {
	if m == nil || id == 0 || hb == nil {
		return
	}
	m.progressReg.mu.Lock()
	if m.progressReg.hb == nil {
		m.progressReg.hb = map[uint64]*heartbeatConfig{}
	}
	m.progressReg.hb[id] = hb
	m.progressReg.mu.Unlock()
}

// UnregisterProgress drops the heartbeat bound to id. Always safe to call,
// including for ids that were never registered or already retired.
func (m *gatewayMetrics) UnregisterProgress(id uint64) {
	if m == nil || id == 0 {
		return
	}
	m.progressReg.mu.Lock()
	delete(m.progressReg.hb, id)
	m.progressReg.mu.Unlock()
}

// ProgressFor resolves the live heartbeat registered for id, reporting false
// for unknown or already retired ids.
func (m *gatewayMetrics) ProgressFor(id uint64) (*heartbeatConfig, bool) {
	if m == nil || id == 0 {
		return nil, false
	}
	m.progressReg.mu.Lock()
	hb, ok := m.progressReg.hb[id]
	m.progressReg.mu.Unlock()
	return hb, ok
}

// beginInflight records a request as live and returns a token to release it with.
// The returned id is 0 when m is nil so endInflight is always safe to defer.
func (m *gatewayMetrics) beginInflight(route string, start time.Time) uint64 {
	if m == nil {
		return 0
	}
	m.inflightMu.Lock()
	m.inflightSeq++
	id := m.inflightSeq
	m.inflightReq[id] = inflightEntry{route: route, start: start}
	m.inflightMu.Unlock()
	return id
}

func (m *gatewayMetrics) endInflight(id uint64) {
	if m == nil || id == 0 {
		return
	}
	m.inflightMu.Lock()
	delete(m.inflightReq, id)
	m.inflightMu.Unlock()
	m.UnregisterProgress(id)
}

// SetInflightProgress folds the per-request progress tick pair (prefill ticks,
// decode ticks) into the live registry entry keyed by the beginInflight id.
// Sourced from the same ProgressHeartbeat path (stream_proxy.go hb.recordEvent /
// messages_stream_passthrough.go content-delta relay) that reports live-stream
// progress fate-side. A nil receiver, a zero id, or an id already retired by
// endInflight is a no-op; the wire never resurrects a row for a dead id.
// The per-phase stamps (prefTouched, decTouched) advance only when that phase
// carries a nonzero count, so progress=1 can tell prefill-stalled from
// decode-stalled.
func (m *gatewayMetrics) SetInflightProgress(id uint64, pref, dec uint64) {
	if m == nil || id == 0 {
		return
	}
	now := time.Now().UnixNano()
	m.inflightMu.Lock()
	if e, ok := m.inflightReq[id]; ok {
		e.prefProg = pref
		e.decProg = dec
		e.touched = now
		if pref > 0 {
			e.prefTouched = now
		}
		if dec > 0 {
			e.decTouched = now
		}
		m.inflightReq[id] = e
	}
	m.inflightMu.Unlock()
}

// TouchInflightProgress marks a live request as still moving without new tick
// counts: the prefill/decode phase boundary (markStreamStart) and silent
// keep-alive ticks ride through here so last-progress-age stays truthful even
// when a phase delivers no new counts. Counts are preserved; only the
// liveness markers advance.
func (m *gatewayMetrics) TouchInflightProgress(id uint64) {
	if m == nil || id == 0 {
		return
	}
	now := time.Now().UnixNano()
	m.inflightMu.Lock()
	if e, ok := m.inflightReq[id]; ok {
		e.touched = now
		e.prefTouched = now
		e.decTouched = now
		m.inflightReq[id] = e
	}
	m.inflightMu.Unlock()
}

// RequestSnapshot is one live in-flight request's observability record, the
// fak.observation.requests.v1 row shape: registry id, serving route, live
// state, start age, and (progress opt-in) the prefill/decode tick counts plus
// last-progress age. Payload-free by construction (route + timing only, the
// privacy floor the /v1/fak observation family holds), so a watcher can answer
// "what is being served right now, for how long, still moving or wedged".
type RequestSnapshot struct {
	// ID is the beginInflight registry token; unique among LIVE entries but
	// reused across the process lifetime once a request retires.
	ID uint64 `json:"id"`
	// Route is the metrics label (routeForMetrics), not a raw URL.
	Route string `json:"route"`
	// State is the per-request lifecycle position visible to the registry:
	// "active" from beginInflight until endInflight - the only state v1 emits;
	// a retired request is removed by endInflight, never projected.
	State string `json:"state"`
	// StartAgeMs is milliseconds since the request entered the registry,
	// floored at 100ms so a just-registered row never reports a flaky
	// sub-resolution age (two snapshots one tick apart must agree it started).
	StartAgeMs int64 `json:"start_age_ms"`
	// Progress fields ride the progress=1 opt-in when the handler merges them;
	// without opt-in they stay zero and the handler omits them entirely.
	// PrefProg / DecProg are the ProgressHeartbeat tick counts; TouchedUnixNanos
	// is the last-progress wall clock where zero means the wire never ticked
	// this request; LastProgressAgeMs is milliseconds since that tick (a nil
	// pointer encodes the never-ticked null the schema calls for).
	PrefProg          uint64 `json:"prefill_progress,omitempty"`
	DecProg           uint64 `json:"decode_progress,omitempty"`
	TouchedUnixNanos  int64  `json:"last_progress_unix_nanos,omitempty"`
	LastProgressAgeMs *int64 `json:"last_progress_age_ms,omitempty"`
	// PrefillChunks and DecodeChunks mirror the folded per-phase tick counts
	// while LastPrefillAgeMs and LastDecodeAgeMs measure per-phase liveness,
	// so a watcher can tell prefill-stalled from decode-stalled. Zero and nil
	// mean unknown; the progress=0 projection drops them so the T1 envelope
	// stays byte-identical.
	PrefillChunks    uint64 `json:"prefill_chunks,omitempty"`
	DecodeChunks     uint64 `json:"decode_chunks,omitempty"`
	LastPrefillAgeMs *int64 `json:"last_prefill_age_ms,omitempty"`
	LastDecodeAgeMs  *int64 `json:"last_decode_age_ms,omitempty"`
}

// inflightAgeFloorMs is the snapshot age floor: a beginInflight-to-snapshot
// window below this (and a negative one - clock skew) reads as the floor so
// the "for how long" answer is never a flaky 0.
const inflightAgeFloorMs = 100

// inflightAgeMsFloor clamps a raw registry age to the reporting floor.
func inflightAgeMsFloor(ageMs int64) int64 {
	if ageMs < inflightAgeFloorMs {
		return inflightAgeFloorMs
	}
	return ageMs
}

// SnapshotInflight copies the live-request registry under inflightMu and
// projects it oldest-first (start age descending; ties by ascending id, which
// regains registry order since beginInflight mints monotonic ids for
// monotonically non-decreasing starts). StartAgeMs is measured against now,
// so a caller pacing two snapshots sees it grow. A nil receiver, the same
// default beginInflight/endInflight honor, yields an empty (non-nil) slice,
// never a panic.
func (m *gatewayMetrics) SnapshotInflight(now time.Time) []RequestSnapshot {
	out := []RequestSnapshot{}
	if m == nil {
		return out
	}
	m.inflightMu.Lock()
	out = make([]RequestSnapshot, 0, len(m.inflightReq))
	for id, e := range m.inflightReq {
		row := RequestSnapshot{
			ID:         id,
			Route:      e.route,
			State:      "active",
			StartAgeMs: inflightAgeMsFloor(now.Sub(e.start).Milliseconds()),
		}
		if e.touched != 0 {
			row.PrefProg = e.prefProg
			row.DecProg = e.decProg
			row.TouchedUnixNanos = e.touched
			age := now.UnixNano() - e.touched
			if age < 0 {
				age = 0
			}
			lp := age / int64(time.Millisecond)
			row.LastProgressAgeMs = &lp
			if e.prefTouched != 0 {
				row.PrefillChunks = e.prefProg
				pa := now.UnixNano() - e.prefTouched
				if pa < 0 {
					pa = 0
				}
				pam := pa / int64(time.Millisecond)
				row.LastPrefillAgeMs = &pam
			}
			if e.decTouched != 0 {
				row.DecodeChunks = e.decProg
				da := now.UnixNano() - e.decTouched
				if da < 0 {
					da = 0
				}
				dam := da / int64(time.Millisecond)
				row.LastDecodeAgeMs = &dam
			}
		}
		out = append(out, row)
	}
	m.inflightMu.Unlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].StartAgeMs != out[j].StartAgeMs {
			return out[i].StartAgeMs > out[j].StartAgeMs
		}
		return out[i].ID < out[j].ID
	})
	return out
}
