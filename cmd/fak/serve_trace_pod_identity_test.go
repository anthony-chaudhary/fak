package main

// serve_trace_pod_identity_test.go â€” witness for anthony-chaudhary/fak#14445's
// done condition: two replicas NEVER mint the same trace id. Each gateway.Server
// embeds its own per-replica identity in the gw-* ids it mints for callers that
// omit X-Trace-Id, so two live replicas behind one service stay disjoint while a
// single replica stays unique within itself.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestTraceIDPodIdentity(t *testing.T) {
	replicas := make([]*gateway.Server, 0, 2)
	for i := 0; i < 2; i++ {
		srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "test-model"})
		if err != nil {
			t.Fatalf("replica %d: New: %v", i, err)
		}
		t.Cleanup(srv.Close)
		replicas = append(replicas, srv)
	}
	a, b := replicas[0], replicas[1]

	if a.TraceIdentity() == "" || b.TraceIdentity() == "" {
		t.Fatalf("both replicas must carry a non-empty trace identity: a=%q b=%q", a.TraceIdentity(), b.TraceIdentity())
	}
	if a.TraceIdentity() == b.TraceIdentity() {
		t.Fatalf("two replicas on one host share the host name; the per-process nonce must still separate them: %q", a.TraceIdentity())
	}

	const n = 8
	mint := func(srv *gateway.Server) []string {
		ids := make([]string, 0, n)
		for i := 0; i < n; i++ {
			req := httptest.NewRequest(http.MethodPost, "/v1/fak/adjudicate", strings.NewReader(`{"tool":"read_file","arguments":{}}`))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, req)
			if rec.Code < 200 || rec.Code >= 300 {
				t.Fatalf("adjudicate status=%d body=%s", rec.Code, rec.Body.String())
			}
			var out struct {
				TraceID string `json:"trace_id"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
				t.Fatalf("adjudicate body: %v (%s)", err, rec.Body.String())
			}
			ids = append(ids, out.TraceID)
		}
		return ids
	}

	check := func(name string, ids []string, wantIdentity string) {
		seen := map[string]bool{}
		for _, id := range ids {
			if !strings.HasPrefix(id, "gw-") {
				t.Fatalf("%s: minted id %q lost the gw- prefix", name, id)
			}
			if !strings.Contains(id, wantIdentity) {
				t.Fatalf("%s: minted id %q does not carry this replica's identity %q", name, id, wantIdentity)
			}
			if seen[id] {
				t.Fatalf("%s: replica minted the same id twice: %q", name, id)
			}
			seen[id] = true
		}
	}
	idsA, idsB := mint(a), mint(b)
	check("replica-a", idsA, a.TraceIdentity())
	check("replica-b", idsB, b.TraceIdentity())
	for _, idA := range idsA {
		for _, idB := range idsB {
			if idA == idB {
				t.Fatalf("two replicas minted the same trace id: %q", idA)
			}
		}
	}
}
