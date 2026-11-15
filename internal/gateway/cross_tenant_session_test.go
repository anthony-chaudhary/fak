package gateway

// cross_tenant_session_test.go — the two-key gateway witness for #14442
// (multi-tenant isolation epic #3263): tenant B cannot observe/stop/steer/
// budget/reset tenant A's trace, list routes are filtered to the caller's rows,
// A2A ignores a forged tenant_id, and the operator key keeps full visibility —
// while single-key deployments keep today's behaviour byte-for-byte.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// crossTenantSeams records what reached the injected session-plane seams, so a
// refused request can be proven to have stopped at the ownership floor rather
// than merely drawing a 404 after acting.
type crossTenantSeams struct {
	controlCalls []string // "trace/verb"
	resetCalls   []string
}

// newCrossTenantServer builds the hosted deployment #14442 targets: an operator
// key plus (optionally) two tenant keys, with the session-plane seams injected
// and two traces bound to their driving tenants.
func newCrossTenantServer(t *testing.T, keyPrincipals map[string]string) (*Server, *crossTenantSeams) {
	t.Helper()
	cfg := Config{EngineID: "mock", Model: "m", VDSO: true, RequireKey: "op-key"}
	if keyPrincipals != nil {
		cfg.KeyPrincipals = keyPrincipals
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	seams := &crossTenantSeams{}
	srv.observeSession = func(ctx context.Context, traceID string) SessionState {
		return SessionState{TraceID: traceID, Run: "running"}
	}
	srv.controlSession = func(ctx context.Context, traceID, verb string, req SessionControlRequest) (SessionState, bool, error) {
		seams.controlCalls = append(seams.controlCalls, traceID+"/"+verb)
		return SessionState{TraceID: traceID, Run: "stopped"}, true, nil
	}
	srv.listSessions = func(ctx context.Context) []SessionState {
		return []SessionState{
			{TraceID: "trace-acme", Run: "running"},
			{TraceID: "trace-globex", Run: "running"},
		}
	}
	srv.resetTrace = func(ctx context.Context, traceID string) error {
		seams.resetCalls = append(seams.resetCalls, traceID)
		return nil
	}
	srv.observeTrace = func(ctx context.Context, traceID string) (string, bool) { return "clean", false }
	srv.bindTraceOwner("trace-acme", "acme")
	srv.bindTraceOwner("trace-globex", "globex")
	return srv, seams
}

func crossTenantDo(t *testing.T, h http.Handler, method, path, key string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req := httptest.NewRequest(method, path, reader)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func sessionIDs(t *testing.T, rec *httptest.ResponseRecorder) []string {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/fak/sessions = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	var resp SessionListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode session list: %v", err)
	}
	ids := make([]string, 0, len(resp.Sessions))
	for _, s := range resp.Sessions {
		ids = append(ids, s.TraceID)
	}
	return ids
}

func containsTaskID(t *testing.T, rec *httptest.ResponseRecorder, taskID string) bool {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /a2a/v1/tasks = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	var resp struct {
		Tasks []struct {
			TaskID string `json:"task_id"`
		} `json:"tasks"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode task list: %v", err)
	}
	for _, task := range resp.Tasks {
		if task.TaskID == taskID {
			return true
		}
	}
	return false
}

// TestCrossTenant_SessionRouteRefusesForeignTenant is the headline #14442 bite:
// tenant B draws 404 on tenant A's session (observe AND control verbs) without
// reaching the drive seam, while tenant A and the operator key act freely.
func TestCrossTenant_SessionRouteRefusesForeignTenant(t *testing.T) {
	srv, seams := newCrossTenantServer(t, map[string]string{"acme-key": "acme", "globex-key": "globex"})
	h := srv.Handler()

	if rec := crossTenantDo(t, h, http.MethodGet, "/v1/fak/session/trace-acme", "globex-key", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("globex observing acme's session = %d (%s), want 404", rec.Code, rec.Body.String())
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/session/trace-acme/stop", "globex-key", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("globex stopping acme's session = %d (%s), want 404", rec.Code, rec.Body.String())
	}
	if len(seams.controlCalls) != 0 {
		t.Fatalf("cross-tenant control reached the drive seam: %v", seams.controlCalls)
	}

	if rec := crossTenantDo(t, h, http.MethodGet, "/v1/fak/session/trace-acme", "acme-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("acme observing its own session = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/session/trace-acme/stop", "acme-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("acme stopping its own session = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if got := len(seams.controlCalls); got != 1 || seams.controlCalls[0] != "trace-acme/stop" {
		t.Fatalf("owner control seam calls = %v, want [trace-acme/stop]", seams.controlCalls)
	}

	// The operator scope keeps full visibility across tenants.
	if rec := crossTenantDo(t, h, http.MethodGet, "/v1/fak/session/trace-acme", "op-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("operator observing a tenant session = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/session/trace-globex/stop", "op-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("operator stopping a tenant session = %d (%s), want 200", rec.Code, rec.Body.String())
	}
}

// TestCrossTenant_SessionListFilteredByTenant pins the plural route: each
// tenant sees only the sessions it drove; the operator sees the whole table.
func TestCrossTenant_SessionListFilteredByTenant(t *testing.T) {
	srv, _ := newCrossTenantServer(t, map[string]string{"acme-key": "acme", "globex-key": "globex"})
	h := srv.Handler()

	if ids := sessionIDs(t, crossTenantDo(t, h, http.MethodGet, "/v1/fak/sessions", "acme-key", nil)); len(ids) != 1 || ids[0] != "trace-acme" {
		t.Fatalf("acme session list = %v, want [trace-acme]", ids)
	}
	if ids := sessionIDs(t, crossTenantDo(t, h, http.MethodGet, "/v1/fak/sessions", "globex-key", nil)); len(ids) != 1 || ids[0] != "trace-globex" {
		t.Fatalf("globex session list = %v, want [trace-globex]", ids)
	}
	if ids := sessionIDs(t, crossTenantDo(t, h, http.MethodGet, "/v1/fak/sessions", "op-key", nil)); len(ids) != 2 {
		t.Fatalf("operator session list = %v, want both rows", ids)
	}
}

// TestCrossTenant_TraceResetRequiresOwnerOrOperator: a foreign tenant cannot
// lower another tenant's guard floor (or read its taint level); owner and
// operator can.
func TestCrossTenant_TraceResetRequiresOwnerOrOperator(t *testing.T) {
	srv, seams := newCrossTenantServer(t, map[string]string{"acme-key": "acme", "globex-key": "globex"})
	h := srv.Handler()

	body := []byte(`{"trace_id":"trace-acme"}`)
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/trace/reset", "globex-key", body); rec.Code != http.StatusNotFound {
		t.Fatalf("globex resetting acme's trace = %d (%s), want 404", rec.Code, rec.Body.String())
	}
	if len(seams.resetCalls) != 0 {
		t.Fatalf("cross-tenant reset reached the taint seam: %v", seams.resetCalls)
	}
	if rec := crossTenantDo(t, h, http.MethodGet, "/v1/fak/trace/trace-acme", "globex-key", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("globex observing acme's taint = %d (%s), want 404", rec.Code, rec.Body.String())
	}

	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/trace/reset", "acme-key", body); rec.Code != http.StatusOK {
		t.Fatalf("acme resetting its own trace = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if got := len(seams.resetCalls); got != 1 || seams.resetCalls[0] != "trace-acme" {
		t.Fatalf("reset seam calls = %v, want [trace-acme]", seams.resetCalls)
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/trace/reset", "op-key", body); rec.Code != http.StatusOK {
		t.Fatalf("operator resetting a tenant trace = %d (%s), want 200", rec.Code, rec.Body.String())
	}
}

// TestCrossTenant_A2ATasksScopedToAuthenticatedTenant: a forged X-Tenant-ID
// header cannot file a task under a foreign tenant; list/get/cancel are scoped
// to the authenticated principal; the operator sees everything.
func TestCrossTenant_A2ATasksScopedToAuthenticatedTenant(t *testing.T) {
	srv, seams := newCrossTenantServer(t, map[string]string{"acme-key": "acme", "globex-key": "globex"})
	h := srv.Handler()

	msg := `{"message_id":"m-ct-1","from":"acme-agent","to":"globex-agent","content":{"method":"agent.ping","session_id":"trace-acme"}}`
	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/messages", strings.NewReader(msg))
	req.Header.Set("Authorization", "Bearer acme-key")
	req.Header.Set("X-Tenant-ID", "globex") // forged: must be ignored under an armed keyset
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("acme sending an A2A message = %d (%s), want 201", rec.Code, rec.Body.String())
	}
	var created struct {
		TaskID string `json:"task_id"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created task: %v", err)
	}
	store := getA2AStore()
	store.mu.RLock()
	task := store.tasks[created.TaskID]
	store.mu.RUnlock()
	if task.TenantID != "acme" {
		t.Fatalf("task filed under TenantID %q, want acme (forged X-Tenant-ID ignored)", task.TenantID)
	}

	// A forged tenant_id query parameter cannot widen the listing.
	if containsTaskID(t, crossTenantDo(t, h, http.MethodGet, "/a2a/v1/tasks?tenant_id=acme", "globex-key", nil), created.TaskID) {
		t.Fatalf("globex listing leaked acme's task despite the forged tenant_id parameter")
	}
	if !containsTaskID(t, crossTenantDo(t, h, http.MethodGet, "/a2a/v1/tasks", "acme-key", nil), created.TaskID) {
		t.Fatalf("acme listing missing its own task")
	}
	if !containsTaskID(t, crossTenantDo(t, h, http.MethodGet, "/a2a/v1/tasks", "op-key", nil), created.TaskID) {
		t.Fatalf("operator listing missing a tenant task")
	}

	if rec := crossTenantDo(t, h, http.MethodGet, "/a2a/v1/tasks/"+created.TaskID, "globex-key", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("globex reading acme's task = %d, want 404", rec.Code)
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/a2a/v1/tasks/"+created.TaskID+"/cancel", "globex-key", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("globex canceling acme's task = %d, want 404", rec.Code)
	}
	if len(seams.controlCalls) != 0 {
		t.Fatalf("cross-tenant cancel reached the session control seam: %v", seams.controlCalls)
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/a2a/v1/tasks/"+created.TaskID+"/cancel", "acme-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("acme canceling its own task = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if got := len(seams.controlCalls); got != 1 || seams.controlCalls[0] != "trace-acme/run" {
		t.Fatalf("owner cancel seam calls = %v, want [trace-acme/run]", seams.controlCalls)
	}
}

// TestCrossTenant_SingleKeySessionPlaneUnchanged pins the out-of-scope half of
// #14442: without a keyset (single-key / loopback deployment) the session plane
// keeps today's behaviour — every caller shares the "" principal and nothing is
// filtered, even when traces carry header-derived owner attributions.
func TestCrossTenant_SingleKeySessionPlaneUnchanged(t *testing.T) {
	srv, _ := newCrossTenantServer(t, nil)
	h := srv.Handler()

	if rec := crossTenantDo(t, h, http.MethodGet, "/v1/fak/session/trace-acme", "op-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("single-key observe = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/session/trace-acme/stop", "op-key", nil); rec.Code != http.StatusOK {
		t.Fatalf("single-key stop = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if ids := sessionIDs(t, crossTenantDo(t, h, http.MethodGet, "/v1/fak/sessions", "op-key", nil)); len(ids) != 2 {
		t.Fatalf("single-key session list = %v, want both rows", ids)
	}
	if rec := crossTenantDo(t, h, http.MethodPost, "/v1/fak/trace/reset", "op-key", []byte(`{"trace_id":"trace-acme"}`)); rec.Code != http.StatusOK {
		t.Fatalf("single-key trace reset = %d (%s), want 200", rec.Code, rec.Body.String())
	}
}
