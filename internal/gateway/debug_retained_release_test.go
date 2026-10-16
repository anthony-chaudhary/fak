package gateway

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type retainedDebugProbe struct {
	agent.Planner
	calls int
	text  string
}

func (p *retainedDebugProbe) RetainedReleaseSummary() (string, bool) { p.calls++; return p.text, true }

// fak-test:runtime fast est=5ms lane=default
func TestDebugRetainedReleaseIsOnDemandAndPartial(t *testing.T) {
	p := &retainedDebugProbe{text: "status=ok truncated=true returned_bindings=0"}
	if p.calls != 0 {
		t.Fatal("eager query")
	}
	got := debugRetainedRelease(p)
	if got == nil || p.calls != 1 || got.Summary != p.text || got.StartedAt.IsZero() || got.CompletedAt.Before(got.StartedAt) {
		t.Fatalf("snapshot=%+v calls=%d", got, p.calls)
	}
	for _, label := range []string{"partial", "not atomic", "independent of request completion", "never add"} {
		if !strings.Contains(got.Comparison, label) {
			t.Fatal(got.Comparison)
		}
	}
	if debugRetainedRelease(nil) != nil {
		t.Fatal("unsupported source")
	}
	proxy := &retainedDebugProbe{text: "wrong proxy"}
	dual := &DualPlanner{local: p, proxy: proxy}
	got = debugRetainedRelease(dual)
	if got == nil || got.Summary != p.text || p.calls != 2 || proxy.calls != 0 {
		t.Fatal("wrong dual source")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDebugRetainedReleaseUsesExistingAuthenticatedRoute(t *testing.T) {
	srv, err := New(Config{EngineID: "mock", Model: "m", RequireKey: "secret-for-test"})
	if err != nil {
		t.Fatal(err)
	}
	p := &retainedDebugProbe{Planner: srv.planner, text: "status=stale-generation"}
	srv.planner = p
	denied := httptest.NewRecorder()
	srv.Handler().ServeHTTP(denied, httptest.NewRequest(http.MethodGet, "/debug/vars?retained_release=1", nil))
	if denied.Code != http.StatusUnauthorized || p.calls != 0 {
		t.Fatalf("status=%d calls=%d", denied.Code, p.calls)
	}
	// Existing UI/default polling and explicit disable do not query native state.
	for _, path := range []string{"/debug/vars", "/debug/vars?retained_release=0"} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		request.Header.Set("Authorization", "Bearer secret-for-test")
		response := httptest.NewRecorder()
		srv.Handler().ServeHTTP(response, request)
		if response.Code != http.StatusOK || p.calls != 0 || strings.Contains(response.Body.String(), `"retained_release"`) {
			t.Fatalf("default/disabled scrape queried backend: status=%d calls=%d", response.Code, p.calls)
		}
	}
	request := httptest.NewRequest(http.MethodGet, "/debug/vars?retained_release=1", nil)
	request.Header.Set("Authorization", "Bearer secret-for-test")
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || p.calls != 1 {
		t.Fatalf("status=%d calls=%d body=%s", response.Code, p.calls, response.Body.String())
	}
	var body struct {
		RetainedRelease *debugRetainedReleaseVars `json:"retained_release"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.RetainedRelease == nil || body.RetainedRelease.Summary != p.text {
		t.Fatalf("missing failure evidence: %+v", body)
	}
}
