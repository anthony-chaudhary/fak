package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// These HTTP witnesses use only parent-existing APIs and wire JSON. The separate
// internal test file uses candidate-only symbols and is NOT a parent RED witness.
// Each fixture owns its model/server. No New/newTestServer, ABI installation,
// Setenv, singleton reset, or global-counter assertion is safe here in parallel.
// Session admission/cache observers do use synchronized process singletons. An
// earlier serial test may already have selected a test-owned session ledger;
// neither its directory nor its contents are assumptions of these tests.
func requestOutcomeTestServer(t *testing.T, planner agent.Planner) *Server {
	t.Helper()
	requestOutcomeTestEnvironment(t)
	return &Server{
		model: "outcome-synth", planner: planner, requireKey: "outcome-test-key",
		logf: func(string, ...any) {}, metrics: newGatewayMetrics(time.Now()),
		activity: newSessionActivity(),
	}
}

func requestOutcomeTestEnvironment(t *testing.T) {
	t.Helper()
	for _, name := range []string{"FAK_EP_FANOUT_ADDRS", "FAK_NEGFRAME_JOURNAL", "FAK_SESSION_LEDGER_DIR", "FAK_CA_BUNDLE"} {
		if os.Getenv(name) != "" {
			t.Fatalf("%s must be empty for the request-local outcome witness", name)
		}
	}
}

func requestOutcomeNativePlanner(t *testing.T) *agent.InKernelPlanner {
	t.Helper()
	requestOutcomeTestEnvironment(t)
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	p := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "outcome-synth", false, nil, false)
	// No artifact identity is installed; additionally opt out explicitly.
	p.SetWarmDiskConfig(agent.WarmDiskConfig{Disabled: true})
	return p
}

type requestOutcomeHTTPBody struct {
	Schema       string `json:"schema"`
	Availability string `json:"availability"`
	Scope        string `json:"scope"`
	Persistence  string `json:"persistence"`
	Record       *struct {
		TraceID     string         `json:"trace_id"`
		Traceparent string         `json:"traceparent"`
		Outcome     string         `json:"outcome"`
		Source      string         `json:"terminal_source"`
		Causal      map[string]any `json:"causal"`
	} `json:"record"`
}

func requestOutcomeGet(t *testing.T, client *http.Client, base, trace, parent, credential string) (int, requestOutcomeHTTPBody) {
	t.Helper()
	query := url.Values{"trace_id": {trace}, "traceparent": {parent}}
	req, err := http.NewRequest(http.MethodGet, base+"/v1/fak/observation/requests?"+query.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+credential)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body requestOutcomeHTTPBody
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, body
}

func requestOutcomePost(t *testing.T, client *http.Client, base, trace, parent, credential string) http.Header {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, base+"/v1/completions",
		strings.NewReader(`{"model":"outcome-synth","prompt":"hi","max_tokens":1,"temperature":0}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+credential)
	req.Header.Set("X-Trace-Id", trace)
	// An attribution header must neither override a keyset principal nor turn
	// the shared-key domain into an arbitrary caller-selected owner.
	req.Header.Set("X-Fak-Principal", "spoofed-outcome-owner")
	if parent != "" {
		req.Header.Set("traceparent", parent)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("completion: status=%d err=%v body=%s", resp.StatusCode, err, body)
	}
	return resp.Header.Clone()
}

// fak-test:runtime medium est=5s lane=default
func TestRequestOutcomeNativeAttemptHTTP(t *testing.T) {
	t.Parallel()
	s := requestOutcomeTestServer(t, requestOutcomeNativePlanner(t))
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	client := &http.Client{Timeout: 10 * time.Second}
	trace := "outcome-" + randomHex(16)
	first := requestOutcomePost(t, client, ts.URL, trace, "", "outcome-test-key")
	code, body := requestOutcomeGet(t, client, ts.URL, first.Get("X-Trace-Id"), first.Get("traceparent"), "outcome-test-key")
	if code != 200 || body.Schema != "fak.request.outcome.v1" || body.Record == nil ||
		body.Record.Outcome != "succeeded" || body.Record.Source != "native_generate_return" {
		t.Fatalf("real CPU native generation outcome: status=%d body=%+v", code, body)
	}
	if body.Scope != "buffered_generation" || body.Persistence != "memory_only" ||
		body.Record.TraceID != first.Get("X-Trace-Id") || body.Record.Traceparent != first.Get("traceparent") {
		t.Fatalf("attempt binding: %+v", body)
	}
	causal := body.Record.Causal
	if causal["schema"] != "fak-causal-envelope/1" || causal["parent_status"] != "unknown" ||
		causal["mapping_status"] != "unknown" || causal["span_id"] != strings.Split(first.Get("traceparent"), "-")[2] {
		t.Fatalf("causal evidence: %+v", causal)
	}
	// Reusing BOTH caller identities must still mint a distinct server span.
	second := requestOutcomePost(t, client, ts.URL, trace, first.Get("traceparent"), "outcome-test-key")
	if second.Get("traceparent") == first.Get("traceparent") {
		t.Fatal("server attempt identity reused")
	}
	code, body = requestOutcomeGet(t, client, ts.URL, second.Get("X-Trace-Id"), second.Get("traceparent"), "outcome-test-key")
	if code != 200 || body.Record == nil || body.Record.Outcome != "succeeded" {
		t.Fatalf("second attempt: %d %+v", code, body)
	}

	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", nil)
	req.Header.Set("X-Trace-Id", trace)
	req.Header.Set("Authorization", "Bearer outcome-test-key")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	code, body = requestOutcomeGet(t, client, ts.URL, resp.Header.Get("X-Trace-Id"), resp.Header.Get("traceparent"), "outcome-test-key")
	if code != 404 || body.Availability != "not_found" || body.Record != nil {
		t.Fatalf("uncovered route reused success: %d %+v", code, body)
	}

	// Legacy reads still show live requests only, never retired completions.
	resp, err = client.Get(ts.URL + "/v1/fak/observation/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var live struct {
		Schema   string
		Requests []struct{ Route string }
	}
	if err := json.NewDecoder(resp.Body).Decode(&live); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 || live.Schema != "fak.observation.requests.v1" {
		t.Fatalf("legacy read: %+v", live)
	}
	for _, row := range live.Requests {
		if row.Route == "/v1/completions" {
			t.Fatal("retired completion remained live")
		}
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestRequestOutcomeAuthAndQueryHTTP(t *testing.T) {
	t.Parallel()
	s := requestOutcomeTestServer(t, agent.NewMockPlanner("outcome-synth"))
	s.allowLAN = true
	handler := s.Handler()
	valid := "trace_id=t&traceparent=00-11111111111111111111111111111111-2222222222222222-00"
	for _, address := range []string{"127.0.0.1:1234", "192.168.1.2:1234"} {
		for _, query := range []string{valid, "trace_id=", "traceparent=", "trace_id=%ZZ", "other=%ZZ"} {
			req := httptest.NewRequest("GET", "/v1/fak/observation/requests?"+query, nil)
			req.RemoteAddr = address
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != 401 {
				t.Fatalf("anonymous outcome query %q from %s: %d", query, address, rec.Code)
			}
		}
	}
	for _, query := range []string{"trace_id=", "traceparent=", valid + "&trace_id=t", valid + "&traceparent=x", "trace_id=%ZZ", "other=%ZZ",
		"trace_id=t&traceparent=00-11111111111111111111111111111111-2222222222222222-FF"} {
		req := httptest.NewRequest("GET", "/v1/fak/observation/requests?"+query, nil)
		req.Header.Set("Authorization", "Bearer outcome-test-key")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != 400 {
			t.Fatalf("malformed identity %q: %d", query, rec.Code)
		}
		var refusal struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &refusal); err != nil || refusal.Error.Code != "outcome_identity_invalid" {
			t.Fatalf("untyped identity refusal: err=%v body=%s", err, rec.Body.String())
		}
	}
	noKeys := requestOutcomeTestServer(t, agent.NewMockPlanner("outcome-synth"))
	noKeys.requireKey = ""
	req := httptest.NewRequest("GET", "/v1/fak/observation/requests?"+valid, nil)
	req.Header.Set("Authorization", "Bearer outcome-test-key")
	rec := httptest.NewRecorder()
	noKeys.Handler().ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatalf("unconfigured reader admitted: %d", rec.Code)
	}
	legacy := httptest.NewRecorder()
	legacyReq := httptest.NewRequest("GET", "/v1/fak/observation/requests?progress=1", nil)
	legacyReq.RemoteAddr = "127.0.0.1:1234"
	handler.ServeHTTP(legacy, legacyReq)
	if legacy.Code != 200 {
		t.Fatalf("legacy local read changed: %d", legacy.Code)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestRequestOutcomePrincipalAndCustomPlannerHTTP(t *testing.T) {
	t.Parallel()
	// A fabricated NativeDecode on a custom planner must never prove success.
	s := requestOutcomeTestServer(t, stubPlanner{comp: &agent.Completion{
		FinishReason: "stop", NativeDecode: &agent.NativeDecodeSummary{},
		Message: agent.Message{Role: agent.RoleAssistant, Content: "untrusted text"},
	}})
	s.keyset = newKeyset(map[string]string{"tenant-a-key": "tenant-a", "tenant-b-key": "tenant-b"})
	s.readBearer = "reader"
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	client := &http.Client{Timeout: 10 * time.Second}
	headers := requestOutcomePost(t, client, ts.URL, "tenant-"+randomHex(16), "", "tenant-a-key")
	for _, key := range []string{"tenant-a-key", "tenant-b-key", "reader", "outcome-test-key"} {
		code, body := requestOutcomeGet(t, client, ts.URL, headers.Get("X-Trace-Id"), headers.Get("traceparent"), key)
		if key == "tenant-a-key" {
			if code != 200 || body.Record == nil || body.Record.Outcome != "completion_unknown" {
				t.Fatalf("owner custom evidence: %d %+v", code, body)
			}
		} else if code != 404 || body.Record != nil {
			t.Fatalf("cross-principal read via %s: %d %+v", key, code, body)
		}
	}
	// Empty principal is one explicit shared-key domain; a read bearer is not
	// an administrator over keyset principals.
	shared := requestOutcomePost(t, client, ts.URL, "shared-"+randomHex(16), "", "outcome-test-key")
	code, body := requestOutcomeGet(t, client, ts.URL, shared.Get("X-Trace-Id"), shared.Get("traceparent"), "reader")
	if code != 200 || body.Record == nil {
		t.Fatalf("shared-key reader: %d %+v", code, body)
	}
}

// fak-test:runtime medium est=5s lane=default
func TestRequestOutcomeReusedTraceInflightAndProxyCancelHTTP(t *testing.T) {
	t.Parallel()
	entered, release, upstreamDone, servedDone := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		defer close(upstreamDone)
		<-release // deliberately independent of downstream cancellation
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"late"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(up.Close)
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) })
	s := requestOutcomeTestServer(t, requestOutcomeNativePlanner(t))
	proxy := agent.NewHTTPPlanner(up.URL+"/v1", "outcome-synth", "")
	handler := s.Handler()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Outcome-Test-Proxy") == "1" {
			// Existing request-local roster binding: the default remains native.
			r = r.WithContext(context.WithValue(r.Context(), chatRouteContextKey{}, &chatRouteBinding{Planner: proxy}))
			defer close(servedDone)
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)
	client := &http.Client{Timeout: 10 * time.Second}
	first := requestOutcomePost(t, client, ts.URL, "reuse-"+randomHex(16), "", "outcome-test-key")
	code, body := requestOutcomeGet(t, client, ts.URL, first.Get("X-Trace-Id"), first.Get("traceparent"), "outcome-test-key")
	if code != 200 || body.Record == nil || body.Record.Outcome != "succeeded" {
		t.Fatalf("first native attempt: %d %+v", code, body)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1,"stream":true}`))
	req.Header.Set("X-Trace-Id", first.Get("X-Trace-Id"))
	req.Header.Set("X-Outcome-Test-Proxy", "1")
	req.Header.Set("Authorization", "Bearer outcome-test-key")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("proxy did not enter")
	}
	if resp.Header.Get("traceparent") == first.Get("traceparent") {
		t.Fatal("inflight attempt reused span")
	}
	code, body = requestOutcomeGet(t, client, ts.URL, resp.Header.Get("X-Trace-Id"), resp.Header.Get("traceparent"), "outcome-test-key")
	if code != 404 || body.Record != nil {
		t.Fatalf("inflight borrowed previous success: %d %+v", code, body)
	}
	cancel()
	select {
	case <-servedDone:
	case <-time.After(5 * time.Second):
		t.Fatal("gateway did not return after cancellation")
	}
	code, body = requestOutcomeGet(t, client, ts.URL, resp.Header.Get("X-Trace-Id"), resp.Header.Get("traceparent"), "outcome-test-key")
	if code != 200 || body.Record == nil || body.Record.Outcome != "completion_unknown" {
		t.Fatalf("cancel incorrectly proved remote stop: %d %+v", code, body)
	}
	select {
	case <-upstreamDone:
		t.Fatal("test upstream unexpectedly stopped")
	default:
	}
	releaseOnce.Do(func() { close(release) })
	select {
	case <-upstreamDone:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream failed to drain")
	}
}
