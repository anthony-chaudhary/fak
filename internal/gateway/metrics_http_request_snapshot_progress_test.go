package gateway

// metrics_http_request_snapshot_progress_test.go — #12760: per-request
// compute progress heartbeats on the observation plane. The heartbeat path
// already folds prefill and decode ticks onto the live-registry row; this
// file witnesses that the folds carry per-phase chunk counts and ages to
// GET /v1/fak/observation/requests?progress=1, that the default envelope
// stays byte-identical to the T1 schema, and that concurrent ticks racing
// snapshot reads stay clean under the race detector.

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

// progressPhaseRow decodes one observation row with the #12760 per-phase
// fields alongside the T1 aliases.
type progressPhaseRow struct {
	ID              uint64 `json:"id"`
	Route           string `json:"route"`
	State           string `json:"state"`
	StartAgeMs      int64  `json:"start_age_ms"`
	PrefillProgress uint64 `json:"prefill_progress"`
	DecodeProgress  uint64 `json:"decode_progress"`
	PrefillChunks   uint64 `json:"prefill_chunks"`
	DecodeChunks    uint64 `json:"decode_chunks"`
	LastProgressAge *int64 `json:"last_progress_age_ms"`
	LastPrefillAge  *int64 `json:"last_prefill_age_ms"`
	LastDecodeAge   *int64 `json:"last_decode_age_ms"`
}

type progressPhaseWire struct {
	Schema   string             `json:"schema"`
	Count    int                `json:"count"`
	Requests []progressPhaseRow `json:"requests"`
}

func getProgressPhaseWire(t *testing.T, srv *Server) progressPhaseWire {
	t.Helper()
	_, raw := getObservationRequestsQuery(t, srv, "progress=1")
	var wire progressPhaseWire
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("decode progress=1 envelope: %v\n%s", err, raw)
	}
	return wire
}

func findProgressPhaseRow(t *testing.T, wire progressPhaseWire, id uint64) progressPhaseRow {
	t.Helper()
	for _, row := range wire.Requests {
		if row.ID == id {
			return row
		}
	}
	t.Fatalf("request id %d missing from progress snapshot (%d rows)", id, len(wire.Requests))
	return progressPhaseRow{}
}

func TestProgressHeartbeatSnapshot(t *testing.T) {
	t.Run("concurrent ticks vs snapshot reads", func(t *testing.T) {
		srv := newObservationTestServer(t)
		m := srv.metrics
		id := m.beginInflight("/v1/chat/completions", time.Now())
		defer m.endInflight(id)
		hb := newHeartbeatConfig()
		hb.attachProgress(m, id)
		if _, ok := m.ProgressFor(id); !ok {
			t.Fatal("attachProgress did not register the heartbeat")
		}
		hb.markStreamStart()
		var wg sync.WaitGroup
		for g := 0; g < 4; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				for i := 0; i < 60; i++ {
					switch (i + g) % 3 {
					case 0:
						hb.TickPrefillChunk()
					case 1:
						hb.TickDecode()
					default:
						hb.Tick()
					}
					if i%9 == 0 {
						hb.recordEvent(4)
					}
				}
			}(g)
		}
		for r := 0; r < 4; r++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < 60; i++ {
					_ = m.SnapshotInflight(time.Now())
					_ = srv.observationRequestsWithProgress(time.Now(), true)
				}
			}()
		}
		wg.Wait()
		wire := getProgressPhaseWire(t, srv)
		row := findProgressPhaseRow(t, wire, id)
		if row.PrefillChunks == 0 || row.DecodeChunks == 0 {
			t.Fatalf("post-race chunks = prefill %d decode %d, want both > 0", row.PrefillChunks, row.DecodeChunks)
		}
		if row.LastPrefillAge == nil || row.LastDecodeAge == nil {
			t.Fatalf("post-race per-phase ages missing: prefill %v decode %v", row.LastPrefillAge, row.LastDecodeAge)
		}
		if *row.LastPrefillAge < 0 || *row.LastDecodeAge < 0 {
			t.Fatalf("negative per-phase age: prefill %d decode %d", *row.LastPrefillAge, *row.LastDecodeAge)
		}
		pref, dec, preNanos, decNanos := hb.progressSnapshot()
		if pref == 0 || dec == 0 || preNanos == 0 || decNanos == 0 {
			t.Fatalf("heartbeat atomics = prefill %d decode %d nanos %d/%d, want all > 0", pref, dec, preNanos, decNanos)
		}
		if row.PrefillProgress == 0 || row.DecodeProgress == 0 || row.LastProgressAge == nil {
			t.Fatalf("T1 progress fields lost under concurrency: %+v", row)
		}
	})

	t.Run("progress=0 envelope omits every progress key", func(t *testing.T) {
		srv := newObservationTestServer(t)
		m := srv.metrics
		id := m.beginInflight("/v1/chat/completions", time.Now().Add(-time.Second))
		defer m.endInflight(id)
		hb := newHeartbeatConfig()
		hb.attachProgress(m, id)
		hb.markStreamStart()
		hb.recordEvent(8)
		_, raw := getObservationRequests(t, srv)
		for _, key := range []string{"prefill_progress", "decode_progress", "last_progress_age_ms", "last_progress_unix_nanos", "prefill_chunks", "decode_chunks", "last_prefill_age_ms", "last_decode_age_ms"} {
			if strings.Contains(string(raw), key) {
				t.Fatalf("default envelope leaked %q:\n%s", key, raw)
			}
		}
	})

	t.Run("progress=1 populates per-phase fields and completion unregisters", func(t *testing.T) {
		srv := newObservationTestServer(t)
		m := srv.metrics
		id := m.beginInflight("/v1/chat/completions", time.Now().Add(-time.Second))
		hb := newHeartbeatConfig()
		hb.attachProgress(m, id)
		hb.markStreamStart()
		hb.recordEvent(5)
		hb.recordEvent(7)
		wire := getProgressPhaseWire(t, srv)
		if wire.Count != 1 {
			t.Fatalf("progress snapshot count = %d, want 1", wire.Count)
		}
		row := findProgressPhaseRow(t, wire, id)
		if row.PrefillChunks != 1 || row.DecodeChunks != 2 {
			t.Fatalf("chunks = prefill %d decode %d, want 1/2", row.PrefillChunks, row.DecodeChunks)
		}
		if row.LastPrefillAge == nil || row.LastDecodeAge == nil {
			t.Fatalf("per-phase ages missing after ticks: %+v", row)
		}
		if *row.LastPrefillAge < 0 || *row.LastDecodeAge < 0 {
			t.Fatalf("negative per-phase age: prefill %d decode %d", *row.LastPrefillAge, *row.LastDecodeAge)
		}
		m.endInflight(id)
		if _, ok := m.ProgressFor(id); ok {
			t.Fatal("endInflight did not unregister the heartbeat")
		}
		drained := getProgressPhaseWire(t, srv)
		if drained.Count != 0 {
			t.Fatalf("post-completion count = %d, want 0", drained.Count)
		}
	})
}
