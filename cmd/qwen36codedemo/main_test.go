package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBrowserRenderWitness(t *testing.T) {
	s := &server{model: "Qwen3.6-27B-Q4_K_M", kernelRev: "test", hardware: "test"}
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	s.page(w, r)
	body := w.Body.String()
	for _, want := range []string{"fak UltraCode", "Qwen3.6-27B", "Agent event transcript", "Shared cache telemetry", "browser holds no gateway bearer token", "/api/fanout?n=4"} {
		if !strings.Contains(body, want) {
			t.Fatalf("captured render missing %q", want)
		}
	}
	if strings.Contains(body, "FAK_DEMO_GATEWAY_KEY") || strings.Contains(body, "Authorization: Bearer") {
		t.Fatal("captured browser render exposes gateway secret contract")
	}
}

func TestRunResultEventContractUsesLowercaseJSON(t *testing.T) {
	w := httptest.NewRecorder()
	writeJSON(w, runResult{OK: true, Agent: "primary", Events: []event{{Kind: "tool", Title: "read", Body: "counter.go"}}})
	body := w.Body.String()
	for _, want := range []string{`"kind":"tool"`, `"title":"read"`, `"body":"counter.go"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("event JSON missing %s: %s", want, body)
		}
	}
	for _, legacy := range []string{`"Kind"`, `"Title"`, `"Body"`} {
		if strings.Contains(body, legacy) {
			t.Fatalf("event JSON leaked legacy key %s: %s", legacy, body)
		}
	}
}

func TestGatewayBearerStaysServerSide(t *testing.T) {
	const secret = "server-only-test-secret"
	var gotAuth string
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"total_tokens":1}}`)
	}))
	defer gateway.Close()
	s := &server{gateway: gateway.URL, key: secret, model: "Qwen3.6-27B-Q4_K_M", client: gateway.Client()}
	got, _, _, err := s.complete(t.Context(), []msg{{Role: "user", Content: "hi"}}, 8)
	if err != nil || got != "ok" {
		t.Fatalf("complete = %q, %v", got, err)
	}
	if gotAuth != "Bearer "+secret {
		t.Fatalf("gateway auth = %q", gotAuth)
	}
	w := httptest.NewRecorder()
	s.page(w, httptest.NewRequest(http.MethodGet, "/", nil))
	if strings.Contains(w.Body.String(), secret) {
		t.Fatal("secret leaked into browser response")
	}
}

func TestHealthUsesCanonicalGatewayRouteAndRequiresOK(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		live   string
	}{
		{name: "ready", status: http.StatusOK, live: `"live":true`},
		{name: "not found is unavailable", status: http.StatusNotFound, live: `"live":false`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.WriteHeader(tc.status)
			}))
			defer gateway.Close()
			s := &server{gateway: gateway.URL, key: "server-only", model: "Qwen3.6-27B-Q4_K_M", client: gateway.Client()}
			w := httptest.NewRecorder()
			s.health(w, httptest.NewRequest(http.MethodGet, "/api/health", nil))
			if gotPath != "/healthz" {
				t.Fatalf("gateway path = %q, want /healthz", gotPath)
			}
			if !strings.Contains(w.Body.String(), tc.live) {
				t.Fatalf("status=%d body=%s, want %s", tc.status, w.Body.String(), tc.live)
			}
		})
	}
}

func TestEdgeProxyKeepsBothSecretsOutOfBrowser(t *testing.T) {
	const edgeSecret = "edge-server-only"
	var got string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("X-Fak-Demo-Edge")
		io.WriteString(w, `{"live":true}`)
	}))
	defer backend.Close()
	t.Setenv("FAK_DEMO_EDGE_KEY", edgeSecret)
	edge := httptest.NewServer(edgeHandler(backend.URL))
	defer edge.Close()
	resp, err := edge.Client().Get(edge.URL + "/api/health")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || got != edgeSecret {
		t.Fatalf("status=%d header=%q body=%s", resp.StatusCode, got, body)
	}
	page, _ := edge.Client().Get(edge.URL + "/")
	html, _ := io.ReadAll(page.Body)
	page.Body.Close()
	if strings.Contains(string(html), edgeSecret) {
		t.Fatal("edge secret leaked into browser page")
	}
}

func TestBackendRefusesDirectTraffic(t *testing.T) {
	const edgeSecret = "edge-server-only"
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) })
	h := requireEdgeKey(edgeSecret, next)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/health", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("direct status=%d want 404", w.Code)
	}
	r := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	r.Header.Set("X-Fak-Demo-Edge", edgeSecret)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusNoContent {
		t.Fatalf("edge status=%d want 204", w.Code)
	}
}
