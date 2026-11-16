package gateway

// tenant_observability_test.go — the #14452 two-tenant route witnesses:
// /v1/fak/events drains filtered by caller ownership, and the operator-grade
// observability routes refused to tenant keys while the metrics bearer and the
// full operator key keep their doors.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// twoTenantServer arms a keyset gateway with two tenant keys plus both operator
// doors (#14452): the full-strength RequireKey bearer and the read-scoped
// metrics bearer. AllowLAN is off and every request below presents an off-box
// peer address, so no loopback/LAN exemption can shortcut the doors under test.
func twoTenantServer(t *testing.T) *Server {
	t.Helper()
	s, err := New(Config{
		EngineID:      "mock",
		Model:         "m",
		Provider:      "openai",
		RequireKey:    "op-key",
		ReadBearer:    "metrics-token",
		KeyPrincipals: map[string]string{"tenant-a-key": "org-a", "tenant-b-key": "org-b"},
		AllowLAN:      false,
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// getOffBox issues one GET through the full handler chain from an off-box peer
// (TEST-NET-3: neither private nor loopback, so no exemption applies) with the
// given bearer ("" = none).
func getOffBox(t *testing.T, s *Server, path, bearer string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.RemoteAddr = "203.0.113.7:1234"
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	s.Handler().ServeHTTP(rec, req)
	return rec
}

// TestEventsTenantIsolation is the #14452 two-tenant route witness: tenant A's
// /v1/fak/events drain carries no tenant-B rows, tool names or trace ids, the
// cursor still advances over B's walked rows, and the single-tenant drain keeps
// its exact historical shape (whole tail + attribution map).
func TestEventsTenantIsolation(t *testing.T) {
	s := twoTenantServer(t)
	j := journal.OpenMemory()
	j.Emit(denyEvent("send_email", "trace-a")) // Seq 1, owned by org-a
	j.Emit(denyEvent("Bash", "trace-b"))       // Seq 2, owned by org-b
	j.Emit(vdsoHitEvent("fetch_url", ""))      // Seq 3, unbound trace
	s.bindTraceOwner("trace-a", "org-a")
	s.bindTraceOwner("trace-b", "org-b")
	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	defer func() { activeJournal = prev }()

	drain := func(bearer string) (EventsResponse, string) {
		t.Helper()
		rec := getOffBox(t, s, "/v1/fak/events", bearer)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status = %d, want 200; body=%s", bearer, rec.Code, rec.Body.String())
		}
		var resp EventsResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: unmarshal: %v", bearer, err)
		}
		return resp, rec.Body.String()
	}

	t.Run("tenant A sees only its own rows", func(t *testing.T) {
		resp, body := drain("tenant-a-key")
		if len(resp.Events) != 1 || resp.Events[0].TraceID != "trace-a" {
			t.Fatalf("events = %+v, want exactly the trace-a row", resp.Events)
		}
		for _, leaked := range []string{"trace-b", "org-b", "Bash", "org-a"} {
			if strings.Contains(body, leaked) {
				t.Errorf("tenant-A drain leaks %q: %s", leaked, body)
			}
		}
		if len(resp.Principals) != 0 {
			t.Errorf("principals = %v, want absent: the trace->tenant map is stripped for tenant callers (#14452)", resp.Principals)
		}
		if resp.Cursor != 3 {
			t.Errorf("cursor = %d, want 3 (advanced over every walked row)", resp.Cursor)
		}
	})

	t.Run("tenant B is symmetric", func(t *testing.T) {
		resp, body := drain("tenant-b-key")
		if len(resp.Events) != 1 || resp.Events[0].TraceID != "trace-b" {
			t.Fatalf("events = %+v, want exactly the trace-b row", resp.Events)
		}
		for _, leaked := range []string{"trace-a", "org-a", "send_email", "org-b"} {
			if strings.Contains(body, leaked) {
				t.Errorf("tenant-B drain leaks %q: %s", leaked, body)
			}
		}
		if resp.Cursor != 3 {
			t.Errorf("cursor = %d, want 3", resp.Cursor)
		}
	})

	t.Run("single-tenant drain keeps the historical shape", func(t *testing.T) {
		solo, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "solo-key", AllowLAN: false})
		if err != nil {
			t.Fatal(err)
		}
		solo.bindTraceOwner("trace-a", "org-a")
		solo.bindTraceOwner("trace-b", "org-b")
		rec := getOffBox(t, solo, "/v1/fak/events", "solo-key")
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
		}
		var resp EventsResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(resp.Events) != 3 {
			t.Fatalf("events = %+v, want all three rows", resp.Events)
		}
		if len(resp.Principals) != 2 || resp.Principals["trace-a"] != "org-a" || resp.Principals["trace-b"] != "org-b" {
			t.Errorf("principals = %v, want {trace-a: org-a, trace-b: org-b}", resp.Principals)
		}
	})
}

// TestMetricsScopeTenantDenied witnesses the #14452 operator-scope floor:
// tenant keys get 403 on every operator-grade observability route whatever
// their source address, while the metrics bearer and the full operator key keep
// scraping, and tenants retain their own scoped reads.
func TestMetricsScopeTenantDenied(t *testing.T) {
	s := twoTenantServer(t)
	prev := activeJournal
	activeJournal = func() *journal.Journal { return journal.OpenMemory() }
	defer func() { activeJournal = prev }()

	operatorRoutes := []string{
		"/metrics",
		"/debug/vars",
		"/v1/fak/observation",
		"/v1/fak/observation/requests",
		"/v1/fak/observation/engine",
		"/slots",
	}
	for _, path := range operatorRoutes {
		for _, tenant := range []string{"tenant-a-key", "tenant-b-key"} {
			rec := getOffBox(t, s, path, tenant)
			if rec.Code != http.StatusForbidden {
				t.Errorf("%s with %s: status = %d, want 403", path, tenant, rec.Code)
			}
		}
	}

	if rec := getOffBox(t, s, "/metrics", "metrics-token"); rec.Code != http.StatusOK {
		t.Errorf("/metrics with metrics bearer: status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if rec := getOffBox(t, s, "/metrics", "op-key"); rec.Code != http.StatusOK {
		t.Errorf("/metrics with operator key: status = %d, want 200", rec.Code)
	}
	if rec := getOffBox(t, s, "/v1/fak/events", "tenant-a-key"); rec.Code != http.StatusOK {
		t.Errorf("/v1/fak/events with tenant key: status = %d, want 200 (filtered drain, not a refusal)", rec.Code)
	}
}
