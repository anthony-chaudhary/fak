package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFakFleetAggregatesBoundedSessionCards(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(_ context.Context) []SessionState {
		return []SessionState{{TraceID: "a", Run: "running", Budget: SessionBudget{TurnsLeft: 5, TokensLeft: 100}, Priority: 1, Rev: 2}, {TraceID: "b", Run: "paused", Reason: "budget", Budget: SessionBudget{TurnsLeft: 0, TokensLeft: 10}, Priority: 2, Rev: 3}, {TraceID: "c", Run: "throttled", Budget: SessionBudget{TurnsLeft: 2, TokensLeft: 20}, Priority: 3, Rev: 4}}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/v1/fak/fleet")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got FleetStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != FleetResponseSchema || got.Rollup != (FleetRollup{Sessions: 3, Running: 1, Blocked: 1, Throttled: 1, BudgetPressure: 1}) {
		t.Fatalf("response=%+v", got)
	}
	if len(got.Sessions) != 3 || got.Sessions[1].TraceID != "b" {
		t.Fatalf("cards=%+v", got.Sessions)
	}
}

func TestFakFleetUsesExistingBearerGate(t *testing.T) {
	srv := newReadBearerServer(t)
	srv.listSessions = func(context.Context) []SessionState { return nil }
	h := srv.Handler()
	req := httptest.NewRequest(http.MethodGet, "/v1/fak/fleet", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("no credential=%d", rec.Code)
	}
	req = httptest.NewRequest(http.MethodGet, "/v1/fak/fleet", nil)
	req.Header.Set("Authorization", "Bearer sekret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("bearer=%d body=%s", rec.Code, rec.Body.String())
	}
}
