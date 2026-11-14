package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// proxyEmbeddingsServer builds a bare Server whose /v1/embeddings forwards to
// the named upstream (the #14444 seam), bypassing New's ABI/engine requirements.
func proxyEmbeddingsServer(t *testing.T, baseURL, apiKey string) *Server {
	t.Helper()
	return &Server{
		model: "chat-model",
		embeddingsUpstream: &embeddingsUpstream{
			baseURL: baseURL,
			apiKey:  apiKey,
			client:  &http.Client{Timeout: 5 * time.Second},
		},
		metrics: newGatewayMetrics(time.Now()),
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestEmbeddingsProxyForwardsToUpstream(t *testing.T) {
	var gotAuth, gotPath, gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":3,"total_tokens":3}}`))
	}))
	t.Cleanup(upstream.Close)
	s := proxyEmbeddingsServer(t, upstream.URL+"/v1", "sk-test")

	w := httptest.NewRecorder()
	s.handleEmbeddings(w, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":"hello world"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var resp EmbeddingsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "text-embedding-3-small" {
		t.Errorf("model = %q, want the UPSTREAM model (not the configured chat model)", resp.Model)
	}
	if len(resp.Data) != 1 || resp.Usage.TotalTokens != 3 {
		t.Errorf("envelope not passed through: %+v", resp)
	}
	if gotPath != "/v1/embeddings" {
		t.Errorf("upstream path = %q, want /v1/embeddings", gotPath)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("upstream auth = %q, want Bearer sk-test", gotAuth)
	}
	if !strings.Contains(gotBody, `"input":"hello world"`) {
		t.Errorf("forwarded body lost the input: %s", gotBody)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestEmbeddingsProxyRefusesWhenUpstreamCannotServe(t *testing.T) {
	newProvider404 := func() *Server {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"message":"secret-upstream-detail"}}`))
		}))
		t.Cleanup(upstream.Close)
		return proxyEmbeddingsServer(t, upstream.URL+"/v1", "")
	}
	cases := map[string]*Server{
		"provider-404":      newProvider404(),
		"transport-failure": proxyEmbeddingsServer(t, "http://127.0.0.1:1/v1", ""),
	}
	for name, s := range cases {
		w := httptest.NewRecorder()
		s.handleEmbeddings(w, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"input":"hello"}`)))
		if w.Code != http.StatusNotImplemented {
			t.Fatalf("%s: status = %d, want 501", name, w.Code)
		}
		var env struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &env); err != nil {
			t.Fatalf("%s: decode refusal: %v (%s)", name, err, w.Body.String())
		}
		if env.Error.Code != "embeddings_backend_unwired" {
			t.Errorf("%s: code = %q, want embeddings_backend_unwired", name, env.Error.Code)
		}
		if strings.Contains(w.Body.String(), "secret-upstream-detail") {
			t.Errorf("%s: upstream error detail leaked: %s", name, w.Body.String())
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestEmbeddingsHashOptInSelfLabels(t *testing.T) {
	s := &Server{model: "chat-model", embeddingsHashOptIn: true, metrics: newGatewayMetrics(time.Now())}
	w := httptest.NewRecorder()
	s.handleEmbeddings(w, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"input":"a b","dimensions":3}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("X-Fak-Embedding-Mode"); got != "hash" {
		t.Errorf("header = %q, want hash", got)
	}
	var resp EmbeddingsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "fak-hash-embed" {
		t.Errorf("model = %q, want fak-hash-embed", resp.Model)
	}
	want, n := embedText("a b", 3)
	got, _ := json.Marshal(resp.Data[0].Embedding)
	expected, _ := json.Marshal(want)
	if string(got) != string(expected) || resp.Usage.TotalTokens != n {
		t.Fatal("hash output changed")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNewEmbeddingsBackendPosture(t *testing.T) {
	// Mirror newTestServer's isolated ABI chain (this file sorts before
	// gateway_test.go, so restore the package-init invariant it relies on).
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(upstream.Close)

	cases := []struct {
		name          string
		backend       string
		baseURL       string
		wantErr       bool
		wantUpstream  bool
		wantHashOptIn bool
	}{
		{"proxy-default", "", upstream.URL + "/v1", false, true, false},
		{"proxy-hash-optin", "hash", upstream.URL + "/v1", false, false, true},
		{"offline-default", "", "", false, false, false},
		{"offline-hash-optin", "hash", "", false, false, true},
		{"bogus-refused", "bogus", "", true, false, false},
	}
	for _, tc := range cases {
		srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true, BaseURL: tc.baseURL, Provider: "openai-compatible", EmbeddingsBackend: tc.backend})
		if tc.wantErr {
			if err == nil {
				t.Fatalf("%s: New accepted unknown backend", tc.name)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: New: %v", tc.name, err)
		}
		if got := srv.embeddingsUpstream != nil; got != tc.wantUpstream {
			t.Errorf("%s: embeddingsUpstream armed = %v, want %v", tc.name, got, tc.wantUpstream)
		}
		if srv.embeddingsHashOptIn != tc.wantHashOptIn {
			t.Errorf("%s: embeddingsHashOptIn = %v, want %v", tc.name, srv.embeddingsHashOptIn, tc.wantHashOptIn)
		}
		srv.Close()
	}
}
