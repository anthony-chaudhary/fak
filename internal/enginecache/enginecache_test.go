package enginecache

import (
	"context"
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
