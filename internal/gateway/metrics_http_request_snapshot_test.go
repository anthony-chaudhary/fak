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
	"testing"
	"time"
)

type observationRequestsWireTest struct {
	Schema   string `json:"schema"`
	Count    int    `json:"count"`
	Requests []struct {
		ID             uint64 `json:"id"`
		Route          string `json:"route"`
		StartUnixNanos int64  `json:"start_unix_nanos"`
		ElapsedMs      int64  `json:"elapsed_ms"`
	} `json:"requests"`
}

func getObservationRequests(t *testing.T, srv *Server) (observationRequestsWireTest, []byte) {
	t.Helper()
	h := srv.Handler()

	unauthorized := httptest.NewRequest(http.MethodGet, "/v1/fak/observation/requests", nil)
	unauthorized.RemoteAddr = "203.0.113.9:40010"
	unauthorizedRec := httptest.NewRecorder()
	h.ServeHTTP(unauthorizedRec, unauthorized)
	if unauthorizedRec.Code != http.StatusUnauthorized {
		t.Fatalf("remote requests read without credentials = %d, want 401", unauthorizedRec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/fak/observation/requests", nil)
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
			t.Fatalf("oldest row route = %q, want /v1/chat/completions (sorted by start ascending)", got)
		}
		if got := wire.Requests[1].Route; got != "/v1/fak/syscall" {
			t.Fatalf("newest row route = %q, want /v1/fak/syscall", got)
		}
		if got := wire.Requests[0].ElapsedMs; got < 4990 || got > 5010 {
			t.Fatalf("oldest elapsed_ms = %d, want ~5000 (started 5s ago)", got)
		}
		if got := wire.Requests[1].ElapsedMs; got < 990 || got > 1010 {
			t.Fatalf("newest elapsed_ms = %d, want ~1000 (started 1s ago)", got)
		}
		// Payload-free floor: only route + timing surface, nothing else.
		for i, row := range wire.Requests {
			if row.Route == "" || row.ID == 0 || row.StartUnixNanos == 0 {
				t.Fatalf("row %d missing identity/route/start: %+v", i, row)
			}
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
		if elapsed := first.Requests[0].ElapsedMs; elapsed < 0 {
			t.Fatalf("elapsed_ms = %d, want >= 0", elapsed)
		}

		time.Sleep(20 * time.Millisecond)
		second, _ := getObservationRequests(t, srv)
		if second.Count != 1 {
			t.Fatalf("paced snapshot count = %d, want 1", second.Count)
		}
		if second.Requests[0].ElapsedMs <= first.Requests[0].ElapsedMs {
			t.Fatalf("elapsed_ms did not grow: first %d, paced %d", first.Requests[0].ElapsedMs, second.Requests[0].ElapsedMs)
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
		env := srvCapped.observationRequests(base.Add(time.Duration(maxObservationRequests+40) * time.Millisecond))
		if env.Count != maxObservationRequests {
			t.Fatalf("capped count = %d, want %d", env.Count, maxObservationRequests)
		}
		if env.Requests[0].ElapsedMs <= env.Requests[len(env.Requests)-1].ElapsedMs {
			t.Fatalf("capped snapshot is not newest-kept: first elapsed %d, last elapsed %d", env.Requests[0].ElapsedMs, env.Requests[len(env.Requests)-1].ElapsedMs)
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
