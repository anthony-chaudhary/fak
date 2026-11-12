package servechain

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	billingkeystore "github.com/anthony-chaudhary/fak/pkg/billing/keystore"
	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
	billingpricebook "github.com/anthony-chaudhary/fak/pkg/billing/pricebook"
)

// fixture wires one chain against one httptest upstream, one in-memory
// ledger, one temp-file price book, and one temp-dir key store. Every test
// gets its own fixture; nothing is shared across tests.
type fixture struct {
	t *testing.T

	upstream       *httptest.Server
	upstreamMu     sync.Mutex
	upstreamStatus int
	upstreamBody   string
	upstreamCT     string
	upstreamHits   atomic.Int64
	lastPath       atomic.Value // string
	lastAuth       atomic.Value // string
	lastModel      atomic.Value // string

	chain  *Chain
	ledger *billingledger.Ledger
	keys   *billingkeystore.Store

	keyByTenant map[string]string // tenant -> full minted key

	now      atomic.Int64 // unix seconds the chain clock returns
	closeAll sync.Once
}

// tenant deposits, in ledger units. tenant-b carries exactly one default
// hold so its second request must be refused at reserve.
const (
	depositTenantA = 10_000_000
	depositTenantB = 1_000_000
)

// newFixture builds the standard two-model fixture: model-a (fresh 2.0,
// output 4.0 per million USD) and model-b (fresh 1.0, output 2.0), both
// served by one upstream whose usage reports prompt 10 / completion 5.
func newFixture(t *testing.T, limits billingmanifest.Limits) *fixture {
	t.Helper()
	f := &fixture{t: t, upstreamStatus: http.StatusOK, upstreamBody: "", upstreamCT: "application/json"}
	f.now.Store(1_700_000_000)

	f.upstream = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.upstreamMu.Lock()
		status, body, ct := f.upstreamStatus, f.upstreamBody, f.upstreamCT
		f.upstreamMu.Unlock()
		f.upstreamHits.Add(1)
		f.lastPath.Store(r.URL.Path)
		f.lastAuth.Store(r.Header.Get("Authorization"))
		var req struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		f.lastModel.Store(req.Model)
		if body == "" {
			// Derive per-model usage: internal-a reports (10, 5),
			// internal-b reports (100, 50) — the two settlement fixtures.
			body = completionBody(10, 5)
			if req.Model == "internal-b" {
				body = completionBody(100, 50)
			}
		}
		w.Header().Set("Content-Type", ct)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	f.start(t, limits, f.upstream.URL+"/v1")
	return f
}

// newClosedUpstreamFixture builds the same fixture against a base URL with
// no listener: every proxied request fails at transport, exercising the
// release path.
func newClosedUpstreamFixture(t *testing.T, limits billingmanifest.Limits) *fixture {
	t.Helper()
	f := &fixture{t: t, upstreamStatus: http.StatusOK, upstreamBody: completionBody(10, 5)}
	f.now.Store(1_700_000_000)
	f.start(t, limits, "http://127.0.0.1:1/v1")
	return f
}

// start resolves the price book, manifest, ledger, key store, and chain
// against the given upstream base URL.
func (f *fixture) start(t *testing.T, limits billingmanifest.Limits, baseURL string) {
	t.Helper()

	book := `{
	  "schema_version": 1,
	  "models": [
	    {"id": "model-a", "version": 1, "source_revision": "test", "source_time": "2026-01-01T00:00:00Z",
	     "currency": "USD", "units": "per_million_tokens",
	     "fresh_input_rate": 2.0, "cache_read_rate": null, "cache_write_rate": null, "output_rate": 4.0},
	    {"id": "model-b", "version": 1, "source_revision": "test", "source_time": "2026-01-01T00:00:00Z",
	     "currency": "USD", "units": "per_million_tokens",
	     "fresh_input_rate": 1.0, "cache_read_rate": null, "cache_write_rate": null, "output_rate": 2.0}
	  ]
	}`
	bookPath := filepath.Join(t.TempDir(), "prices.json")
	if err := os.WriteFile(bookPath, []byte(book), 0o600); err != nil {
		t.Fatalf("write price book: %v", err)
	}

	manifestJSON := fmt.Sprintf(`{
	  "version": "fak-billing-manifest/v1",
	  "state_dir": "state",
	  "prices_path": %q,
	  "upstreams": [{"name": "up", "base_url": %q}],
	  "models": [
	    {"name": "model-a", "upstream": "up", "upstream_model": "internal-a"},
	    {"name": "model-b", "upstream": "up", "upstream_model": "internal-b"},
	    {"name": "model-c", "upstream": "up", "upstream_model": "internal-c"}
	  ],
	  "limits": {"requests_per_minute": %d, "tokens_per_minute": %d,
	             "daily_token_cap": %d, "session_token_cap": %d}
	}`, bookPath, baseURL, limits.RequestsPerMinute, limits.TokensPerMinute,
		limits.DailyTokenCap, limits.SessionTokenCap)
	m, err := billingmanifest.Load([]byte(manifestJSON))
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	f.ledger = billingledger.New()
	if err := f.ledger.Deposit("dep-a", "tenant-a", depositTenantA); err != nil {
		t.Fatalf("deposit tenant-a: %v", err)
	}
	if err := f.ledger.Deposit("dep-b", "tenant-b", depositTenantB); err != nil {
		t.Fatalf("deposit tenant-b: %v", err)
	}

	ks, err := billingkeystore.Open(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatalf("open key store: %v", err)
	}
	f.keys = ks
	f.keyByTenant = map[string]string{}
	for _, tenant := range []string{"tenant-a", "tenant-b"} {
		key, _, err := ks.Mint(tenant)
		if err != nil {
			t.Fatalf("mint key for %s: %v", tenant, err)
		}
		f.keyByTenant[tenant] = key
	}

	chain, err := New(Config{
		Manifest: m,
		Prices:   mustPriceBook(t, bookPath),
		Ledger:   f.ledger,
		Keys:     ks,
		Now:      func() time.Time { return time.Unix(f.now.Load(), 0).UTC() },
	})
	if err != nil {
		t.Fatalf("new chain: %v", err)
	}
	f.chain = chain
	t.Cleanup(func() { f.finish() })
}

// finish closes the chain, which drains the settle queue, so every ledger
// assertion after it is deterministic. Idempotent.
func (f *fixture) finish() {
	f.closeAll.Do(func() { f.chain.Close() })
}

// keyFor returns the minted key for tenant.
func (f *fixture) keyFor(t *testing.T, tenant string) string {
	t.Helper()
	key, ok := f.keyByTenant[tenant]
	if !ok {
		t.Fatalf("no key minted for %s", tenant)
	}
	return key
}

// mustPriceBook loads the price book the fixture wrote.
func mustPriceBook(t *testing.T, path string) *billingpricebook.PriceBook {
	t.Helper()
	pb, err := billingpricebook.LoadFile(path)
	if err != nil {
		t.Fatalf("load price book: %v", err)
	}
	return pb
}

// post sends one POST as tenant's key would, with the given extra headers.
func (f *fixture) post(t *testing.T, path, key, session, body string) *http.Response {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	if session != "" {
		req.Header.Set("X-Fak-Session", session)
	}
	rec := httptest.NewRecorder()
	f.chain.ServeHTTP(rec, req)
	return rec.Result()
}

// completionBody is the upstream's OpenAI-shaped completion with usage.
func completionBody(prompt, completion int64) string {
	return fmt.Sprintf(`{"id":"cmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d}}`, prompt, completion)
}

// TestServeChainRouteRefusals proves the closed served-path set and the
// POST-only method rule: a foreign path is 404, a GET on a served path is
// 405 with Allow, and neither reaches the upstream.
//
// fak-test:runtime fast est=1s lane=default
func TestServeChainRouteRefusals(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	resp := f.post(t, "/v1/other", f.keyFor(t, "tenant-a"), "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("foreign path: got %d, want 404", resp.StatusCode)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer "+f.keyFor(t, "tenant-a"))
	rec := httptest.NewRecorder()
	f.chain.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET: got %d, want 405", rec.Code)
	}
	if allow := rec.Header().Get("Allow"); allow != http.MethodPost {
		t.Fatalf("GET Allow header: got %q, want POST", allow)
	}
	if f.upstreamHits.Load() != 0 {
		t.Fatalf("upstream hit %d times on refusals, want 0", f.upstreamHits.Load())
	}
}

// TestServeChainAuth proves the bearer gate: no key, an unknown key, and a
// revoked key are refused 401/401/403 before anything else moves.
//
// fak-test:runtime fast est=1s lane=default
func TestServeChainAuth(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	for _, tc := range []struct {
		name string
		key  string
		want int
	}{
		{"no key", "", http.StatusUnauthorized},
		{"unknown key", "fak-00000000000000000000000000000000", http.StatusUnauthorized},
	} {
		resp := f.post(t, "/v1/chat/completions", tc.key, "", `{"model":"model-a"}`)
		if resp.StatusCode != tc.want {
			t.Fatalf("%s: got %d, want %d", tc.name, resp.StatusCode, tc.want)
		}
	}

	// A revoked key: mint one, revoke it by its full key, present it.
	revokedFull, _, err := f.keys.Mint("tenant-revoked")
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	if err := f.keys.Revoke(revokedFull); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	resp := f.post(t, "/v1/chat/completions", revokedFull, "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("revoked key: got %d, want 403", resp.StatusCode)
	}
	if f.upstreamHits.Load() != 0 {
		t.Fatalf("upstream hit %d times on auth refusals, want 0", f.upstreamHits.Load())
	}
}

// TestServeChainScrub proves credential headers never reach the upstream:
// the chain's bearer key, the api key, cookies, and the session header are
// all stripped, while the billing request id is stamped on the response.
//
// fak-test:runtime fast est=1s lane=default
func TestServeChainScrub(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model-a"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+f.keyFor(t, "tenant-a"))
	req.Header.Set("X-Api-Key", "secondary-secret")
	req.Header.Set("Cookie", "session=secret")
	req.Header.Set("X-Fak-Session", "sess-1")
	req.Header.Set("X-Custom-Keep", "keepme")
	rec := httptest.NewRecorder()
	f.chain.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("scrubbed request: got %d, want 200", rec.Code)
	}
	if got := f.lastAuth.Load(); got != "" {
		t.Fatalf("upstream saw Authorization %q, want none", got)
	}
	if got := rec.Header().Get("X-Fak-Billing-Request-Id"); got == "" {
		t.Fatal("response carries no X-Fak-Billing-Request-Id")
	}
}

// TestServeChainBodyShape proves the bounded-body and shape stages: an
// oversized body is 413, non-JSON is 400, a body without a model is 400,
// and an unserved model name is 404 — none reaching the upstream.
//
// fak-test:runtime fast est=1s lane=default
func TestServeChainBodyShape(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})
	f.chain.cfg.MaxBodyBytes = 64 // tighten the bound for a cheap oversize case

	key := f.keyFor(t, "tenant-a")
	cases := []struct {
		name string
		body string
		want int
	}{
		{"oversized", `{"model":"model-a","pad":"` + strings.Repeat("a", 128) + `"}`, http.StatusRequestEntityTooLarge},
		{"not json", "hello", http.StatusBadRequest},
		{"no model", `{"messages":[]}`, http.StatusBadRequest},
		{"unserved model", `{"model":"model-z"}`, http.StatusNotFound},
	}
	for _, tc := range cases {
		resp := f.post(t, "/v1/chat/completions", key, "", tc.body)
		if resp.StatusCode != tc.want {
			t.Fatalf("%s: got %d, want %d", tc.name, resp.StatusCode, tc.want)
		}
	}
	if f.upstreamHits.Load() != 0 {
		t.Fatalf("upstream hit %d times on shape refusals, want 0", f.upstreamHits.Load())
	}
}
