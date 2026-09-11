package gateway

// metrics_http_request_snapshot_test.go — #12759: the per-request read surface
// GET /v1/fak/observation/requests. Mirrors the observation_test.go posture:
// auth via the read-scoped bearer off loopback, wire decoded into a local
// struct so the JSON contract (schema / count / requests) is pinned, and the
// live-request registry driven directly through beginInflight/endInflight.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type observationRequestsWireTest struct {
	Schema    string `json:"schema"`
	Count     int    `json:"count"`
	Truncated int    `json:"truncated"`
	Requests  []struct {
		ID         uint64 `json:"id"`
		Route      string `json:"route"`
		State      string `json:"state"`
		StartAgeMs int64  `json:"start_age_ms"`
		// progress=1 opt-in fields; decoded but only asserted on progress reads.
		PrefillProgress   int64  `json:"prefill_progress"`
		DecodeProgress    int64  `json:"decode_progress"`
		LastProgressAgeMs *int64 `json:"last_progress_age_ms"`
	} `json:"requests"`
}

func getObservationRequests(t *testing.T, srv *Server) (observationRequestsWireTest, []byte) {
	t.Helper()
	return getObservationRequestsQuery(t, srv, "")
}

func getObservationRequestsQuery(t *testing.T, srv *Server, query string) (observationRequestsWireTest, []byte) {
	t.Helper()
	h := srv.Handler()
	path := "/v1/fak/observation/requests"
	if query != "" {
		path += "?" + query
	}

	unauthorized := httptest.NewRequest(http.MethodGet, path, nil)
	unauthorized.RemoteAddr = "203.0.113.9:40010"
	unauthorizedRec := httptest.NewRecorder()
	h.ServeHTTP(unauthorizedRec, unauthorized)
	if unauthorizedRec.Code != http.StatusUnauthorized {
		t.Fatalf("remote requests read without credentials = %d, want 401", unauthorizedRec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.RemoteAddr = "203.0.113.9:40011"
	req.Header.Set("Authorization", "Bearer snapshot-read-secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/fak/observation/requests with read bearer = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("content-type = %q, want application/json", got)
	}
	raw := append([]byte(nil), rec.Body.Bytes()...)
	var wire observationRequestsWireTest
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("decode observation requests: %v\n%s", err, raw)
	}
	if wire.Schema != "fak.observation.requests.v1" {
		t.Fatalf("schema = %q, want fak.observation.requests.v1", wire.Schema)
	}
	if wire.Count != len(wire.Requests) {
		t.Fatalf("count = %d, want len(requests) = %d (a clean zero is data, not a null)", wire.Count, len(wire.Requests))
	}
	return wire, raw
}

func TestObservationRequestsSnapshot(t *testing.T) {
	t.Run("empty registry is an explicit empty array", func(t *testing.T) {
		srv := newObservationTestServer(t)
		wire, raw := getObservationRequests(t, srv)
		if wire.Count != 0 || wire.Requests == nil {
			t.Fatalf("idle snapshot = count %d requests %v, want count 0 and []\n%s", wire.Count, wire.Requests, raw)
		}
		t.Logf("idle snapshot: %s", raw)
	})

	t.Run("two inflight entries carry their routes", func(t *testing.T) {
		srv := newObservationTestServer(t)
		base := time.Now()
		// Registry order = start order: id1 started before id2.
		id1 := srv.metrics.beginInflight("/v1/chat/completions", base.Add(-5*time.Second))
		id2 := srv.metrics.beginInflight("/v1/fak/syscall", base.Add(-1*time.Second))
		defer func() {
			srv.metrics.endInflight(id1)
			srv.metrics.endInflight(id2)
		}()

		wire, _ := getObservationRequests(t, srv)
		if wire.Count != 2 || len(wire.Requests) != 2 {
			t.Fatalf("snapshot count = %d with %d rows, want 2/2: %+v", wire.Count, len(wire.Requests), wire.Requests)
		}
		if got := wire.Requests[0].Route; got != "/v1/chat/completions" {
			t.Fatalf("oldest row route = %q, want /v1/chat/completions (sorted oldest-first)", got)
		}
		if got := wire.Requests[1].Route; got != "/v1/fak/syscall" {
			t.Fatalf("newest row route = %q, want /v1/fak/syscall", got)
		}
		if got := wire.Requests[0].StartAgeMs; got < 4990 {
			t.Fatalf("oldest start_age_ms = %d, want >= ~5000 (started 5s ago)", got)
		}
		if got := wire.Requests[1].StartAgeMs; got < 990 {
			t.Fatalf("newest start_age_ms = %d, want >= ~1000 (started 1s ago)", got)
		}
		// Payload-free floor: identity + route + state + timing, nothing else,
		// and never prompt/PII content on the wire.
		for i, row := range wire.Requests {
			if row.Route == "" || row.ID == 0 || row.State != "active" {
				t.Fatalf("row %d missing identity/route/state: %+v", i, row)
			}
			if row.PrefillProgress != 0 || row.DecodeProgress != 0 || row.LastProgressAgeMs != nil {
				t.Fatalf("row %d leaked progress fields without progress=1: %+v", i, row)
			}
		}
		if wire.Truncated != 0 {
			t.Fatalf("truncated = %d, want 0 under cap", wire.Truncated)
		}
	})

	t.Run("elapsed grows and registry drains on completion", func(t *testing.T) {
		srv := newObservationTestServer(t)
		base := time.Now()
		id := srv.metrics.beginInflight("/v1/chat/completions", base)

		first, _ := getObservationRequests(t, srv)
		if first.Count != 1 {
			t.Fatalf("live snapshot count = %d, want 1", first.Count)
		}
		if first.Requests[0].State != "active" {
			t.Fatalf("state = %q, want active", first.Requests[0].State)
		}

		time.Sleep(120 * time.Millisecond)
		second, _ := getObservationRequests(t, srv)
		if second.Count != 1 {
			t.Fatalf("paced snapshot count = %d, want 1", second.Count)
		}
		if second.Requests[0].StartAgeMs <= first.Requests[0].StartAgeMs {
			t.Fatalf("start_age_ms did not grow: first %d, paced %d", first.Requests[0].StartAgeMs, second.Requests[0].StartAgeMs)
		}

		srv.metrics.endInflight(id)
		drained, _ := getObservationRequests(t, srv)
		if drained.Count != 0 || drained.Requests == nil {
			t.Fatalf("post-completion snapshot = %+v, want count 0 and []", drained)
		}
	})

	t.Run("cap keeps the newest 256 records", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		base := time.Now()
		for i := 0; i < maxObservationRequests+40; i++ {
			id := m.beginInflight("/v1/chat/completions", base.Add(time.Duration(i)*time.Millisecond))
			if id == 0 {
				t.Fatal("beginInflight returned 0")
			}
		}
		srvCapped := &Server{metrics: m}
		env := srvCapped.observationRequestsWithProgress(base.Add(time.Duration(maxObservationRequests+40)*time.Millisecond), false)
		if env.Count != maxObservationRequests+40 {
			t.Fatalf("capped count = %d, want %d (count is the untruncated live total)", env.Count, maxObservationRequests+40)
		}
		if len(env.Requests) != maxObservationRequests {
			t.Fatalf("capped rows = %d, want %d", len(env.Requests), maxObservationRequests)
		}
		if env.Truncated != 40 {
			t.Fatalf("truncated = %d, want 40", env.Truncated)
		}
		// Newest-kept: first row (oldest survivor) younger than the last row.
		if env.Requests[0].StartAgeMs >= env.Requests[len(env.Requests)-1].StartAgeMs && env.Requests[0].StartAgeMs <= 100 {
			t.Fatalf("capped snapshot is not newest-kept: first age %d, last age %d", env.Requests[0].StartAgeMs, env.Requests[len(env.Requests)-1].StartAgeMs)
		}
	})

	t.Run("method and nil-registry posture", func(t *testing.T) {
		srv := newObservationTestServer(t)
		h := srv.Handler()
		// Loopback peer: the read-scoped exemption admits the request so the
		// handler's own method gate (not auth) produces the 405.
		reqPost := httptest.NewRequest(http.MethodPost, "/v1/fak/observation/requests", nil)
		reqPost.RemoteAddr = "127.0.0.1:54321"
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, reqPost)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("POST observation/requests = %d, want 405", rec.Code)
		}

		var nilMetrics *gatewayMetrics
		got := nilMetrics.SnapshotInflight(time.Now())
		if got == nil || len(got) != 0 {
			t.Fatalf("nil-receiver snapshot = %v, want empty non-nil", got)
		}
	})
}

// TestObservationRequestsProgressMerge is the progress=1 opt-in witness: the
// ProgressHeartbeat ticks (Touch at the prefill/decode boundary, decode ticks
// per content event) merge onto the SAME row the poll reads, and the
// last_progress_age refreshes instead of freezing.
func TestObservationRequestsProgressMerge(t *testing.T) {
	srv := newObservationTestServer(t)
	m := srv.metrics
	id := m.beginInflight("/v1/chat/completions", time.Now().Add(-time.Second))
	defer m.endInflight(id)

	// Direct-set ticks provenance equivalent to what the heartbeat path folds:
	m.SetInflightProgress(id, 1, 3)

	wire, _ := getObservationRequestsQuery(t, srv, "progress=1")
	if wire.Count != 1 {
		t.Fatalf("progress snapshot count = %d, want 1", wire.Count)
	}
	row := wire.Requests[0]
	if row.PrefillProgress != 1 || row.DecodeProgress != 3 {
		t.Fatalf("progress merge = prefill %d decode %d, want 1/3", row.PrefillProgress, row.DecodeProgress)
	}
	if row.LastProgressAgeMs == nil {
		t.Fatal("last_progress_age_ms missing after a tick, want a number")
	}
	if *row.LastProgressAgeMs < 0 {
		t.Fatalf("last_progress_age_ms = %d, want >= 0", *row.LastProgressAgeMs)
	}

	// No progress=1 -> the fields vanish entirely (wire-shape parity).
	plain, raw := getObservationRequests(t, srv)
	for i, r2 := range plain.Requests {
		if r2.PrefillProgress != 0 || r2.DecodeProgress != 0 || r2.LastProgressAgeMs != nil {
			t.Fatalf("row %d leaked progress fields without progress=1: %+v\n%s", i, r2, raw)
		}
	}
	if !strings.Contains(string(raw), "\"state\"") || strings.Contains(string(raw), "prefill_progress") || strings.Contains(string(raw), "decode_progress") || strings.Contains(string(raw), "last_progress_age_ms") {
		t.Fatalf("default envelope must omit progress keys entirely:\n%s", raw)
	}

	// Touch refreshes liveness without tick counts.
	m.TouchInflightProgress(id)
	touched, _ := getObservationRequestsQuery(t, srv, "progress=1")
	if touched.Requests[0].LastProgressAgeMs == nil || *touched.Requests[0].LastProgressAgeMs > *row.LastProgressAgeMs+1 {
		t.Fatalf("touch did not refresh last_progress_age: first %v, after %v", *row.LastProgressAgeMs, touched.Requests[0].LastProgressAgeMs)
	}
}

// TestObservationRequestsSchemaAndShape pins the exact schema string, the
// byte-stable top-level key order, and the sub-100ms age floor.
func TestObservationRequestsSchemaAndShape(t *testing.T) {
	srv := newObservationTestServer(t)
	m := srv.metrics
	id := m.beginInflight("/v1/chat/completions", time.Now())
	defer m.endInflight(id)

	_, raw := getObservationRequests(t, srv)
	if !strings.HasPrefix(string(raw), "{\"schema\":\"fak.observation.requests.v1\",\"count\":") {
		t.Fatalf("schema/count prefix wrong or reordered:\n%s", raw)
	}
	// Byte-stable field order inside a row: id, route, state, start_age_ms.
	if !strings.Contains(string(raw), "\"requests\":[{\"id\":") || !strings.Contains(string(raw), ",\"route\":\"/v1/chat/completions\",\"state\":\"active\",\"start_age_ms\":") {
		t.Fatalf("row key order drifted from id,route,state,start_age_ms:\n%s", raw)
	}

	wire, _ := getObservationRequests(t, srv)
	// Age floor: the row was registered microseconds ago; the floor keeps the
	// answer off the flaky 0 while still being honest order-of-magnitude.
	if wire.Requests[0].StartAgeMs < 100 {
		t.Fatalf("start_age_ms = %d, want >= 100 floor", wire.Requests[0].StartAgeMs)
	}
}

// TestObservationRequestsDeadID pins endInflight/removal semantics: progress
// folded onto a retired id never resurrects a row.
func TestObservationRequestsDeadID(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	id := m.beginInflight("/v1/chat/completions", time.Now())
	m.endInflight(id)
	m.SetInflightProgress(id, 9, 9)
	m.TouchInflightProgress(id)
	srv := &Server{metrics: m}
	wire := srv.observationRequests(time.Now())
	if wire.Count != 0 || len(wire.Requests) != 0 || wire.Truncated != 0 {
		t.Fatalf("dead-id fold resurrected a row: %+v", wire)
	}
}
