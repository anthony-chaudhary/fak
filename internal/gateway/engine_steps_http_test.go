package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func newEngineStepNativeServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = agent.NewInKernelPlanner(model.NewSynthetic(model.Config{}), nil, "native", false, nil, false)
	return srv
}

type engineStepsWireTest struct {
	Schema  string              `json:"schema"`
	Planner string              `json:"planner"`
	Native  bool                `json:"native"`
	Engine  enginestep.Snapshot `json:"engine"`
}

func getObservationEngine(t *testing.T, srv *Server, method, query string) *httptest.ResponseRecorder {
	t.Helper()
	path := "/v1/fak/observation/engine"
	if query != "" {
		path += "?" + query
	}
	req := httptest.NewRequest(method, path, nil)
	req.RemoteAddr = "127.0.0.1:50001"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

// fak-test:runtime fast est=200ms lane=default
func TestEngineStepMetricsRenderOnlyForNativePlanner(t *testing.T) {
	srv := newEngineStepNativeServer(t)
	out := srv.renderMetrics()
	if !strings.Contains(out, "# TYPE "+enginestep.MetricDecodeStepSeconds+" histogram") {
		t.Fatalf("native /metrics lacks %s histogram", enginestep.MetricDecodeStepSeconds)
	}
	for _, fam := range enginestep.MetricFamilies {
		if !strings.Contains(out, "# TYPE "+fam+" ") {
			t.Fatalf("native /metrics lacks family %s", fam)
		}
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.RemoteAddr = "127.0.0.1:50002"
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "# TYPE "+enginestep.MetricDecodeStepSeconds+" histogram") {
		t.Fatalf("GET /metrics = %d, want 200 carrying fak_engine_*", rec.Code)
	}

	dual, err := NewDualPlanner(&agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"}, srv.planner.(*agent.InKernelPlanner), "native")
	if err != nil {
		t.Fatal(err)
	}
	srv.planner = dual
	if !strings.Contains(srv.renderMetrics(), "# TYPE "+enginestep.MetricDecodeStepSeconds+" histogram") {
		t.Fatal("dual planner /metrics lacks fak_engine_*")
	}

	for name, p := range map[string]agent.Planner{
		"mock":  &agent.MockPlanner{},
		"proxy": &agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"},
	} {
		srv.planner = p
		// fak_engine_cache_* is exempt: it is the live-engine KV cache-event stream
		// (engine.DefaultCacheEvents), fed by proxy adapters such as vLLM/SGLang and
		// rendered on every serve with an explicit observed gauge, not the native
		// continuous-batching cycle this test guards.
		for _, line := range strings.Split(srv.renderMetrics(), "\n") {
			if strings.Contains(line, "fak_engine_") && !strings.Contains(line, "fak_engine_cache_") {
				t.Fatalf("%s planner /metrics carries a phantom fak_engine_* family: %s", name, line)
			}
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestObservationEngineEnvelope(t *testing.T) {
	srv := newEngineStepNativeServer(t)
	for i := 0; i < 3; i++ {
		enginestep.Default.ObserveCohort(1)
	}
	rec := getObservationEngine(t, srv, http.MethodGet, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET engine = %d body=%s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q", ct)
	}
	var wire engineStepsWireTest
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if wire.Schema != "fak-observation-engine/1" || wire.Planner != "inkernel" || !wire.Native || wire.Engine.Schema != "fak-engine-steps/1" {
		t.Fatalf("envelope = schema %q planner %q native %v engine.schema %q", wire.Schema, wire.Planner, wire.Native, wire.Engine.Schema)
	}
	if len(wire.Engine.Recent) < 3 || len(wire.Engine.Recent) > 32 {
		t.Fatalf("default recent = %d, want 3..32", len(wire.Engine.Recent))
	}

	rec = getObservationEngine(t, srv, http.MethodGet, "n=2")
	wire = engineStepsWireTest{}
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("n=2: code=%d err=%v", rec.Code, err)
	}
	if len(wire.Engine.Recent) != 2 {
		t.Fatalf("n=2 recent = %d, want 2", len(wire.Engine.Recent))
	}

	rec = getObservationEngine(t, srv, http.MethodGet, "n=0&kind=cohort")
	wire = engineStepsWireTest{}
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil || rec.Code != http.StatusOK || len(wire.Engine.Recent) != 0 {
		t.Fatalf("n=0: code=%d err=%v recent=%d", rec.Code, err, len(wire.Engine.Recent))
	}
	rec = getObservationEngine(t, srv, http.MethodGet, "kind=cohort&n=5")
	wire = engineStepsWireTest{}
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("kind=cohort: code=%d err=%v", rec.Code, err)
	}
	for _, r := range wire.Engine.Recent {
		if r.Kind != enginestep.KindCohort {
			t.Fatalf("kind=cohort returned %+v", r)
		}
	}

	srv.planner = &agent.MockPlanner{}
	rec = getObservationEngine(t, srv, http.MethodGet, "")
	wire = engineStepsWireTest{}
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("mock: code=%d err=%v", rec.Code, err)
	}
	if wire.Native || wire.Planner != "mock" {
		t.Fatalf("mock planner envelope native=%v planner=%q, want false/mock", wire.Native, wire.Planner)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestObservationEngineRejectsBadInput(t *testing.T) {
	srv := newEngineStepNativeServer(t)
	for _, q := range []string{"n=-1", "n=abc", "kind=bogus"} {
		if rec := getObservationEngine(t, srv, http.MethodGet, q); rec.Code != http.StatusBadRequest {
			t.Fatalf("?%s = %d, want 400", q, rec.Code)
		}
	}
	rec := getObservationEngine(t, srv, http.MethodPost, "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST = %d, want 405", rec.Code)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestObservationEngineCompactFormat(t *testing.T) {
	srv := newEngineStepNativeServer(t)
	rec := getObservationEngine(t, srv, http.MethodGet, "format=compact")
	if rec.Code != http.StatusOK {
		t.Fatalf("compact = %d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.HasPrefix(body, "ENGINE ") || strings.Count(body, "\n") != 1 || !strings.HasSuffix(body, "\n") {
		t.Fatalf("compact body = %q, want one line starting ENGINE", body)
	}
	if strings.Contains(body, "not native") {
		t.Fatalf("native compact body = %q", body)
	}
	srv.planner = &agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"}
	body = getObservationEngine(t, srv, http.MethodGet, "format=compact").Body.String()
	if !strings.HasPrefix(body, "ENGINE not native (planner=proxy)") || strings.Count(body, "\n") != 1 {
		t.Fatalf("proxy compact body = %q", body)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestObservationEngineRemoteRequiresReadScope(t *testing.T) {
	srv, err := New(Config{EngineID: "test", Model: "m", RequireKey: "sekret"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	h := srv.Handler()
	rem := httptest.NewRecorder()
	rr := httptest.NewRequest(http.MethodGet, "/v1/fak/observation/engine", nil)
	rr.RemoteAddr = "203.0.113.7:40000"
	h.ServeHTTP(rem, rr)
	if rem.Code != http.StatusUnauthorized {
		t.Fatalf("remote without bearer = %d, want 401", rem.Code)
	}
	ok := httptest.NewRecorder()
	ra := httptest.NewRequest(http.MethodGet, "/v1/fak/observation/engine", nil)
	ra.RemoteAddr = "203.0.113.7:40001"
	ra.Header.Set("Authorization", "Bearer sekret")
	h.ServeHTTP(ok, ra)
	if ok.Code != http.StatusOK {
		t.Fatalf("remote with bearer = %d, want 200", ok.Code)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestEngineStepAdmissionAcquireObservesAdmissionWait(t *testing.T) {
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 4, TokenBudget: 1000, MaxWaiting: 4})
	count := func() uint64 {
		return enginestep.Default.Snapshot(0, "").Phases[string(enginestep.PhaseAdmissionWait)].Count
	}
	before := count()
	lease, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "enginestep-admit", Tokens: 10})
	if err != nil || lease == nil {
		t.Fatalf("Acquire = (%v, %v), want lease", lease, err)
	}
	defer lease.Release()
	if got := count() - before; got != 1 {
		t.Fatalf("admission_wait delta = %d, want 1", got)
	}
}
