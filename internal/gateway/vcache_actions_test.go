package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestHandleFakVCacheActionsIdleIsExplicit(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakVCacheActions(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/actions", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var plan vcacheobserve.ProviderActionPlan
	if err := json.Unmarshal(rec.Body.Bytes(), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if plan.Schema != vcacheobserve.ProviderActionSchema || plan.Turns != 0 || len(plan.Actions) != 0 {
		t.Fatalf("idle action plan = %+v", plan)
	}
	if plan.Transport.Ready || plan.Transport.Mode != "decision_only" {
		t.Fatalf("transport = %+v, want decision-only", plan.Transport)
	}
}

func TestHandleFakVCacheActionsReportsObservedFamilies(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("head", 1, 40000, 0, 40000)
	m.observeVCacheTurn("head", 2, 50, 40000, 500)
	s := &Server{metrics: m}

	rec := httptest.NewRecorder()
	s.handleFakVCacheActions(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/actions", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var plan vcacheobserve.ProviderActionPlan
	if err := json.Unmarshal(rec.Body.Bytes(), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if plan.Schema != vcacheobserve.ProviderActionSchema || plan.Turns != 2 || plan.FamilyCount != 1 {
		t.Fatalf("plan header = %+v", plan)
	}
	if plan.Counts.Noop != 1 || plan.Counts.Gated != 0 || plan.Counts.Ready != 0 {
		t.Fatalf("counts = %+v, want one ride-natural no-op", plan.Counts)
	}
	row := plan.Actions[0]
	if row.Action != "ride_natural" || row.State != vcacheobserve.ActionNoop || row.CacheReadTokens != 40000 {
		t.Fatalf("action row = %+v, want observed ride-natural provider row", row)
	}
}

func TestHandleFakVCacheActionsAcceptsTransportWitness(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("bursty", 1, 40000, 0, 40000)
	m.observeVCacheTurn("bursty", 700001, 50, 40000, 500)
	s := &Server{metrics: m}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/actions?heartbeat_transport=1&prefix_witness=1&transport_source=test", nil)
	s.handleFakVCacheActions(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var plan vcacheobserve.ProviderActionPlan
	if err := json.Unmarshal(rec.Body.Bytes(), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if plan.Transport.Mode != "witnessed_transport" || !plan.Transport.Ready || plan.Transport.Witness == nil {
		t.Fatalf("transport = %+v, want witnessed ready", plan.Transport)
	}
	if plan.Counts.Ready != 1 || len(plan.Actions) != 1 {
		t.Fatalf("plan = %+v, want one ready action", plan)
	}
	row := plan.Actions[0]
	if row.Action != "heartbeat_pin" || row.State != vcacheobserve.ActionReady {
		t.Fatalf("row = %+v, want ready heartbeat pin", row)
	}
}

func TestHandleFakVCacheActionsRejectsNonGet(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakVCacheActions(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/vcache/actions", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status=%d want 405", rec.Code)
	}
}
