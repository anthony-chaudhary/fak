package enginecache

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func sampleDirective(provider string) cachemeta.ExternalInvalidationDirective {
	kv := cachemeta.FromKVPrefix(
		cachemeta.KVPrefix{Tokens: []int{1, 2, 3}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer"},
		cachemeta.WithResidency(cachemeta.TierProvider, provider, "lease-1"),
		cachemeta.WithLabel("provider", provider),
		cachemeta.WithLabel("engine", provider),
	)
	return cachemeta.ExternalInvalidationDirective{
		Kind:      cachemeta.ExternalInvalidateKVSpan,
		Entry:     kv.ID,
		Plane:     kv.Plane,
		Residency: kv.Residency,
		Provider:  provider,
		Engine:    provider,
		Reason:    "poisoned_kv",
	}
}

func TestInvalidateSGLangFlushesRadixCache(t *testing.T) {
	var calls int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/flush_cache" {
			t.Fatalf("path = %q, want /flush_cache", r.URL.Path)
		}
		if got := r.URL.Query().Get("timeout"); got != "30" {
			t.Fatalf("timeout query = %q, want 30", got)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer admin-secret" {
			t.Fatalf("auth = %q, want bearer admin-secret", got)
		}
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer ts.Close()

	res, err := Client{
		BaseURL:     ts.URL + "/v1",
		AdminAPIKey: "admin-secret",
		IdleTimeout: 30 * time.Second,
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("sglang")})
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
	if res.Engine != EngineSGLang || res.Scope != ScopeWholePrefixCache || res.ExactSpanSupported || res.Directives != 1 || res.StatusCode != http.StatusOK {
		t.Fatalf("bad result: %+v", res)
	}
}

func TestInvalidateVLLMResetsPrefixCache(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/reset_prefix_cache" {
			t.Fatalf("path = %q, want /reset_prefix_cache", r.URL.Path)
		}
		if r.URL.RawQuery != "" {
			t.Fatalf("unexpected query: %s", r.URL.RawQuery)
		}
		_, _ = w.Write([]byte(`true`))
	}))
	defer ts.Close()

	res, err := Client{
		Engine:  EngineVLLM,
		BaseURL: ts.URL + "/v1",
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("ignored")})
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if res.Engine != EngineVLLM || res.Scope != ScopeWholePrefixCache || res.ExactSpanSupported || res.Directives != 1 {
		t.Fatalf("bad result: %+v", res)
	}
}

func TestInvalidateExactSpanUnsupportedUsesWholePrefixReset(t *testing.T) {
	var calls int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/flush_cache" {
			t.Fatalf("path = %q, want /flush_cache", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		if len(body) != 0 {
			t.Fatalf("body = %q, want empty whole-cache reset request", string(body))
		}
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer ts.Close()

	res, err := Client{Engine: EngineSGLang, BaseURL: ts.URL}.Invalidate(
		context.Background(),
		[]cachemeta.ExternalInvalidationDirective{
			sampleDirective("sglang"),
			sampleDirective("sglang"),
		},
	)
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want one whole-cache reset for multiple span directives", calls)
	}
	if res.Scope != ScopeWholePrefixCache || res.ExactSpanSupported || res.Directives != 2 {
		t.Fatalf("exact-span fallback not witnessed: %+v", res)
	}
}

func TestInvalidateExactSpanRequiredFailsBeforeWholeCacheReset(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("server should not be called when exact-span eviction is required but unsupported")
	}))
	defer ts.Close()

	res, err := Client{
		Engine:        EngineSGLang,
		BaseURL:       ts.URL,
		RequiredScope: ScopeExactSpan,
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("sglang")})
	if err == nil {
		t.Fatal("expected exact-span required failure")
	}
	if !strings.Contains(err.Error(), "exact-span eviction required") {
		t.Fatalf("error = %q, want exact-span required detail", err)
	}
	if res.Engine != EngineSGLang || res.Scope != ScopeWholePrefixCache || res.ExactSpanSupported || res.Directives != 1 {
		t.Fatalf("bad failure witness: %+v", res)
	}
}

func TestInvalidateExactSpanEvictsNamedSpansWhenEndpointConfigured(t *testing.T) {
	var calls int
	var gotBody exactSpanRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/evict_span" {
			t.Fatalf("path = %q, want /evict_span", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("content-type = %q, want application/json", ct)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode exact-span body: %v", err)
		}
		_, _ = w.Write([]byte(`{"evicted":true}`))
	}))
	defer ts.Close()

	kv := sampleDirective("sglang")
	idx := sampleDirective("sglang")
	idx.Kind = cachemeta.ExternalInvalidateAttentionIndex
	idx.Reason = "parent_kv_poisoned"

	res, err := Client{
		Engine:            EngineSGLang,
		BaseURL:           ts.URL,
		ExactSpanEndpoint: ts.URL + "/evict_span",
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{kv, idx})
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want one exact-span eviction", calls)
	}
	if res.Scope != ScopeExactSpan || !res.ExactSpanSupported || res.Directives != 2 || res.StatusCode != http.StatusOK {
		t.Fatalf("exact-span eviction not witnessed: %+v", res)
	}
	if len(gotBody.Spans) != 2 {
		t.Fatalf("exact-span body carried %d spans, want 2: %+v", len(gotBody.Spans), gotBody)
	}
	if gotBody.Spans[0].Digest == "" || gotBody.Spans[0].Kind != cachemeta.ExternalInvalidateKVSpan {
		t.Fatalf("first span lost K/V identity: %+v", gotBody.Spans[0])
	}
	if gotBody.Spans[1].Kind != cachemeta.ExternalInvalidateAttentionIndex {
		t.Fatalf("dependent DSA attention_index span not carried: %+v", gotBody.Spans[1])
	}
}

func TestInvalidateExactSpanRequiredSucceedsWhenEndpointConfigured(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/evict_span" {
			t.Fatalf("path = %q, want /evict_span", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	res, err := Client{
		Engine:            EngineSGLang,
		BaseURL:           ts.URL,
		ExactSpanEndpoint: ts.URL + "/evict_span",
		RequiredScope:     ScopeExactSpan,
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("sglang")})
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if res.Scope != ScopeExactSpan || !res.ExactSpanSupported {
		t.Fatalf("required exact-span eviction not satisfied by configured endpoint: %+v", res)
	}
}

func TestInvalidateExactSpanFailsClosedOnEndpointError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no such span", http.StatusConflict)
	}))
	defer ts.Close()

	res, err := Client{
		Engine:            EngineSGLang,
		BaseURL:           ts.URL,
		ExactSpanEndpoint: ts.URL + "/evict_span",
		RequiredScope:     ScopeExactSpan,
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("sglang")})
	if err == nil {
		t.Fatal("expected fail-closed error when exact-span eviction cannot be confirmed")
	}
	if res.Scope != ScopeExactSpan || res.StatusCode != http.StatusConflict {
		t.Fatalf("fail-closed witness missing exact-span status: %+v", res)
	}
}

func TestInvalidateExactSpanRequiredFailsClosedWithoutNamedSpan(t *testing.T) {
	// An exact-span endpoint is configured, but the directive carries no span
	// identity (the coarse proxy-quarantine shape). Requiring exact-span eviction
	// must refuse rather than POST a precise eviction over an empty target set.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("server must not be called when no named span can be evicted exactly")
	}))
	defer ts.Close()

	res, err := Client{
		Engine:            EngineSGLang,
		BaseURL:           ts.URL,
		ExactSpanEndpoint: ts.URL + "/evict_span",
		RequiredScope:     ScopeExactSpan,
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{{
		Kind:   cachemeta.ExternalInvalidateKVSpan,
		Reason: "proxy_tool_result_quarantine",
	}})
	if err == nil {
		t.Fatal("expected fail-closed error when no span identity is available")
	}
	if res.Scope != ScopeExactSpan || res.ExactSpanSupported != true {
		t.Fatalf("bad fail-closed witness: %+v", res)
	}
}

func TestInvalidateExactSpanEndpointFallsBackToWholeResetWhenNotRequired(t *testing.T) {
	// Endpoint configured (capable) but the directive names no span and exact-span
	// is NOT required: degrade to the safe whole-prefix reset superset.
	var path string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	res, err := Client{
		Engine:            EngineSGLang,
		BaseURL:           ts.URL,
		ExactSpanEndpoint: ts.URL + "/evict_span",
	}.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{{
		Kind:   cachemeta.ExternalInvalidateKVSpan,
		Reason: "proxy_tool_result_quarantine",
	}})
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if path != "/flush_cache" {
		t.Fatalf("path = %q, want whole-prefix /flush_cache fallback", path)
	}
	if res.Scope != ScopeWholePrefixCache {
		t.Fatalf("expected whole-prefix fallback, got %+v", res)
	}
}

func TestSupportsExactSpanIsFalseForCurrentPublicEngines(t *testing.T) {
	for _, engine := range []Engine{EngineSGLang, EngineVLLM} {
		if SupportsExactSpan(engine) {
			t.Fatalf("%s exact-span eviction should stay false until a documented public endpoint exists", engine)
		}
	}
}

func TestInvalidateNoopsWithoutDirectives(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("server should not be called for empty directive set")
	}))
	defer ts.Close()

	res, err := Client{Engine: EngineSGLang, BaseURL: ts.URL}.Invalidate(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invalidate: %v", err)
	}
	if res != (Result{}) {
		t.Fatalf("empty directive set should return zero result, got %+v", res)
	}
}

func TestInvalidateReportsHTTPFailure(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "busy", http.StatusBadRequest)
	}))
	defer ts.Close()

	res, err := Client{Engine: EngineSGLang, BaseURL: ts.URL}.Invalidate(
		context.Background(),
		[]cachemeta.ExternalInvalidationDirective{sampleDirective("sglang")},
	)
	if err == nil {
		t.Fatal("expected HTTP failure")
	}
	if res.StatusCode != http.StatusBadRequest || res.BodySummary == "" {
		t.Fatalf("failure result missing status/body: %+v", res)
	}
}
