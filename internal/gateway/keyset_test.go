package gateway

// keyset_test.go — org-scoped multi-key auth + principal attribution (#5332).
//
// Coverage:
//   - newKeyset / keyset.lookup: half-specified bindings are dropped, keys are matched
//     by digest (never held raw), empty/nil never match.
//   - withAuth: a keyset key authenticates AND stamps its tenant principal; the single
//     RequireKey stays anonymous ("" principal); a keyset alone still gates auth; the
//     RequireKey-only path is unchanged; a spoofed key still 401s.
//   - principalFor: the door-authenticated principal outranks a spoofed X-Fak-Principal
//     header and body field.
//   - handleFakEvents / the access log: each row/turn is attributed to the tenant that
//     owns its trace, joinable by trace_id, without changing the persisted row schema.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestNewKeysetDropsHalfSpecifiedBindings(t *testing.T) {
	if ks := newKeyset(nil); ks != nil {
		t.Fatalf("nil map must yield a nil keyset (RequireKey-only path unchanged), got %+v", ks)
	}
	if ks := newKeyset(map[string]string{}); ks != nil {
		t.Fatalf("empty map must yield a nil keyset, got %+v", ks)
	}
	// A key with no principal, or a principal with no key, must NEVER authenticate — a
	// half-specified binding falling back to the empty single-tenant principal would be
	// a silent auth hole.
	for name, m := range map[string]map[string]string{
		"empty-principal": {"k": ""},
		"empty-key":       {"": "org"},
		"blank-both":      {"   ": "   "},
	} {
		if ks := newKeyset(m); ks != nil {
			t.Fatalf("%s must yield a nil keyset, got %+v", name, ks)
		}
	}

	// A mix keeps only the fully-specified bindings.
	ks := newKeyset(map[string]string{
		"k1":  "org-a",
		"  ":  "skipped-empty-key",
		"k2":  "  ", // empty principal -> skipped
		"k3 ": " org-c ",
	})
	if ks == nil {
		t.Fatal("a map with a usable binding must yield a non-nil keyset")
	}
	if p, ok := ks.lookup("k1"); !ok || p != "org-a" {
		t.Fatalf("lookup k1 = (%q,%v), want (org-a,true)", p, ok)
	}
	// Keys/principals are trimmed at construction.
	if p, ok := ks.lookup("k3"); !ok || p != "org-c" {
		t.Fatalf("lookup k3 (trimmed) = (%q,%v), want (org-c,true)", p, ok)
	}
	if _, ok := ks.lookup("k2"); ok {
		t.Fatal("k2 had an empty principal and must not authenticate")
	}
}

func TestKeysetLookupMatchesByDigest(t *testing.T) {
	var nilKS *keyset
	if p, ok := nilKS.lookup("anything"); ok || p != "" {
		t.Fatalf("nil keyset lookup = (%q,%v), want (\"\",false)", p, ok)
	}

	ks := newKeyset(map[string]string{"key-a": "org-a", "key-b": "org-b"})
	if p, ok := ks.lookup("key-a"); !ok || p != "org-a" {
		t.Fatalf("lookup key-a = (%q,%v), want (org-a,true)", p, ok)
	}
	if p, ok := ks.lookup("key-b"); !ok || p != "org-b" {
		t.Fatalf("lookup key-b = (%q,%v), want (org-b,true)", p, ok)
	}
	// An empty credential is not the empty-key tenant.
	if _, ok := ks.lookup(""); ok {
		t.Fatal("empty presented key must never match")
	}
	// A non-member key is a miss with no principal leaked.
	if p, ok := ks.lookup("key-c"); ok || p != "" {
		t.Fatalf("lookup unknown = (%q,%v), want (\"\",false)", p, ok)
	}
	// The raw key is hashed, not stored: presenting the digest's own text is not the key.
	if _, ok := ks.lookup("6b3a55e0261b0304143f805a24924d0c1c44524821305f31d9277843b8a10f4e"); ok {
		t.Fatal("a digest-shaped string must not authenticate as any key")
	}
}

// principalEchoHandler is the terminal handler under withAuth: it echoes the
// door-stamped isolation principal so a test can assert attribution.
func principalEchoHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(principalFromContext(r.Context())))
	})
}

func serveAuth(t *testing.T, h http.Handler, setKey func(*http.Request)) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	if setKey != nil {
		setKey(req)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestWithAuthKeysetAuthenticatesAndAttributes(t *testing.T) {
	srv, err := New(Config{
		EngineID:      "mock",
		Model:         "m",
		VDSO:          true,
		RequireKey:    "single-secret",
		KeyPrincipals: map[string]string{"org-a-key": "org-a", "org-b-key": "org-b"},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h := srv.withAuth(principalEchoHandler())

	cases := []struct {
		name    string
		set     func(*http.Request)
		code    int
		princip string
	}{
		{"keyset via x-api-key", func(r *http.Request) { r.Header.Set("X-Api-Key", "org-a-key") }, http.StatusOK, "org-a"},
		{"keyset via bearer", func(r *http.Request) { r.Header.Set("Authorization", "Bearer org-b-key") }, http.StatusOK, "org-b"},
		{"single requirekey stays anonymous", func(r *http.Request) { r.Header.Set("X-Api-Key", "single-secret") }, http.StatusOK, ""},
		{"wrong key", func(r *http.Request) { r.Header.Set("X-Api-Key", "nope") }, http.StatusUnauthorized, ""},
		{"no credential", nil, http.StatusUnauthorized, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveAuth(t, h, tc.set)
			if rec.Code != tc.code {
				t.Fatalf("status = %d, want %d (%s)", rec.Code, tc.code, rec.Body.String())
			}
			if tc.code == http.StatusOK && rec.Body.String() != tc.princip {
				t.Fatalf("attributed principal = %q, want %q", rec.Body.String(), tc.princip)
			}
		})
	}
}

func TestWithAuthKeysetAloneGatesAuth(t *testing.T) {
	// No single RequireKey — the keyset is the ONLY credential source, and it must still
	// gate every non-exempt route.
	srv, err := New(Config{
		EngineID:      "mock",
		Model:         "m",
		VDSO:          true,
		KeyPrincipals: map[string]string{"org-a-key": "org-a"},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h := srv.withAuth(principalEchoHandler())

	if rec := serveAuth(t, h, nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("no credential must 401 under a keyset-only gate, got %d", rec.Code)
	}
	rec := serveAuth(t, h, func(r *http.Request) { r.Header.Set("X-Api-Key", "org-a-key") })
	if rec.Code != http.StatusOK || rec.Body.String() != "org-a" {
		t.Fatalf("bound key = (%d,%q), want (200,org-a)", rec.Code, rec.Body.String())
	}
	if rec := serveAuth(t, h, func(r *http.Request) { r.Header.Set("X-Api-Key", "wrong") }); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key must 401, got %d", rec.Code)
	}
}

func TestWithAuthSingleKeyPathUnchanged(t *testing.T) {
	// keyset == nil: behavior is exactly the prior single-RequireKey gate.
	guarded, err := New(Config{EngineID: "mock", Model: "m", VDSO: true, RequireKey: "secret"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h := guarded.withAuth(principalEchoHandler())
	if rec := serveAuth(t, h, func(r *http.Request) { r.Header.Set("X-Api-Key", "secret") }); rec.Code != http.StatusOK || rec.Body.String() != "" {
		t.Fatalf("single key accepted = (%d,%q), want (200,\"\")", rec.Code, rec.Body.String())
	}
	if rec := serveAuth(t, h, func(r *http.Request) { r.Header.Set("X-Api-Key", "wrong") }); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong single key must 401, got %d", rec.Code)
	}
	if rec := serveAuth(t, h, nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing single key must 401, got %d", rec.Code)
	}

	// No RequireKey and no keyset: open gate, every request reaches next.
	open, err := New(Config{EngineID: "mock", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	oh := open.withAuth(principalEchoHandler())
	if rec := serveAuth(t, oh, nil); rec.Code != http.StatusOK {
		t.Fatalf("open gate must reach next without a credential, got %d", rec.Code)
	}
}

func TestPrincipalForDoorPrincipalOutranksSpoof(t *testing.T) {
	// A door-authenticated (keyset) principal in the context is authoritative over a
	// client-supplied header/body, so a caller cannot present one key and claim another
	// tenant.
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/changes", nil)
	req.Header.Set("X-Fak-Principal", "victim-org")
	req = req.WithContext(WithPrincipal(req.Context(), "org-a"))
	if got := principalFor(req, "body-org"); got != "org-a" {
		t.Fatalf("door principal must win over spoofed header/body, got %q", got)
	}

	// Without a door principal the header wins, then the body — the prior precedence.
	hdrOnly := httptest.NewRequest(http.MethodPost, "/v1/fak/changes", nil)
	hdrOnly.Header.Set("X-Fak-Principal", "hdr-org")
	if got := principalFor(hdrOnly, "body-org"); got != "hdr-org" {
		t.Fatalf("header should win when no door principal, got %q", got)
	}
	bodyOnly := httptest.NewRequest(http.MethodPost, "/v1/fak/changes", nil)
	if got := principalFor(bodyOnly, "body-org"); got != "body-org" {
		t.Fatalf("body should win when no door principal or header, got %q", got)
	}
}

func TestHandleFakEventsAttributesPrincipalsByTrace(t *testing.T) {
	j := journal.OpenMemory()
	j.Emit(denyEvent("send_email", "trace-a"))
	j.Emit(denyEvent("Bash", "trace-b"))
	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	defer func() { activeJournal = prev }()

	s := &Server{}
	s.bindTraceOwner("trace-a", "org-a") // trace-b stays unbound (single-tenant "")

	rec := httptest.NewRecorder()
	s.handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var resp EventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := resp.Principals["trace-a"]; got != "org-a" {
		t.Fatalf("trace-a principal = %q, want org-a", got)
	}
	if _, ok := resp.Principals["trace-b"]; ok {
		t.Fatal("an unbound (single-tenant) trace must not appear in Principals")
	}
	// The event rows themselves are unchanged (schema untouched).
	if len(resp.Events) != 2 {
		t.Fatalf("want 2 drained rows, got %d", len(resp.Events))
	}
}

func TestHandleFakEventsOmitsPrincipalsWhenSingleTenant(t *testing.T) {
	j := journal.OpenMemory()
	j.Emit(denyEvent("send_email", "trace-a"))
	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	defer func() { activeJournal = prev }()

	rec := httptest.NewRecorder()
	(&Server{}).handleFakEvents(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/events", nil))
	if strings.Contains(rec.Body.String(), "principals") {
		t.Fatalf("no bound owner => principals must be omitted, got %s", rec.Body.String())
	}
}

func TestAccessLogAttributesPrincipalByTrace(t *testing.T) {
	srv, err := New(Config{EngineID: "mock", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var lines []string
	srv.logf = func(format string, a ...any) {
		if len(a) == 1 {
			if b, ok := a[0].([]byte); ok {
				lines = append(lines, string(b))
				return
			}
		}
		lines = append(lines, format)
	}
	srv.bindTraceOwner("trace-a", "org-a")

	lastEvent := func() map[string]any {
		if len(lines) == 0 {
			t.Fatal("no access-log line emitted")
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(lines[len(lines)-1]), &ev); err != nil {
			t.Fatalf("decode access log %q: %v", lines[len(lines)-1], err)
		}
		return ev
	}

	srv.logInferenceTurnWithContextEvent("trace-a", "anthropic", false, agent.Usage{}, "stop", time.Millisecond, false, false)
	ev := lastEvent()
	if ev["trace_id"] != "trace-a" {
		t.Fatalf("trace_id = %v, want trace-a", ev["trace_id"])
	}
	if ev["principal"] != "org-a" {
		t.Fatalf("access-log principal = %v, want org-a", ev["principal"])
	}

	// An unbound trace leaves the line's shape unchanged (no principal key).
	srv.logInferenceTurnWithContextEvent("trace-unbound", "anthropic", false, agent.Usage{}, "stop", time.Millisecond, false, false)
	if _, ok := lastEvent()["principal"]; ok {
		t.Fatal("an unbound trace must not carry a principal in the access log")
	}
}
