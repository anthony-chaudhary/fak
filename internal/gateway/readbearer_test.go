package gateway

// readbearer_test.go — the #3461 read-scoped bearer: accepted ONLY on the read-only
// observability endpoints, never on a mutating route. httptest.NewRequest's default
// RemoteAddr (192.0.2.1:1234) is deliberately NON-loopback, so these requests cannot
// ride the loopback exemption — the bearer alone must decide.

import (
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func newReadBearerServer(t *testing.T) *Server {
	t.Helper()
	// "test" is not a package-init engine — it exists only because some other helper
	// registered it. Register it here so these cases stand on their own rather than on
	// whichever test happened to run first.
	abi.RegisterEngine("test", echoEngine{})
	srv, err := New(Config{EngineID: "test", Model: "m", RequireKey: "sekret", ReadBearer: "read-tok"})
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

func TestReadBearerGrantsReadEndpointsOffLoopback(t *testing.T) {
	h := newReadBearerServer(t).Handler()
	for _, path := range []string{"/debug/vars", "/metrics", "/v1/fak/observation"} {
		// Without ANY credential a non-loopback caller is rejected.
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		if rec.Code != 401 {
			t.Fatalf("GET %s with no credential = %d, want 401", path, rec.Code)
		}
		// The read-scoped bearer grants the read endpoint.
		req := httptest.NewRequest("GET", path, nil)
		req.Header.Set("Authorization", "Bearer read-tok")
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("GET %s with read bearer = %d, want 200", path, rec.Code)
		}
		// A WRONG bearer stays rejected.
		req = httptest.NewRequest("GET", path, nil)
		req.Header.Set("Authorization", "Bearer wrong")
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 401 {
			t.Fatalf("GET %s with wrong bearer = %d, want 401", path, rec.Code)
		}
	}
}

func TestReadBearerDoesNotGrantMutatingEndpoints(t *testing.T) {
	h := newReadBearerServer(t).Handler()
	for _, tc := range []struct{ method, path string }{
		{"POST", "/v1/fak/policy/reload"},
		{"POST", "/v1/messages"},
		{"POST", "/v1/fak/session/sess-1/run"},
	} {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		req.Header.Set("Authorization", "Bearer read-tok")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 401 {
			t.Fatalf("%s %s with read bearer = %d, want 401 (read scope must not extend)", tc.method, tc.path, rec.Code)
		}
	}
}

func TestReadBearerEmptyConfigAuthorizesNothing(t *testing.T) {
	abi.RegisterEngine("test", echoEngine{})
	srv, err := New(Config{EngineID: "test", Model: "m", RequireKey: "sekret"})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/debug/vars", nil)
	req.Header.Set("Authorization", "Bearer ")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatalf("empty ReadBearer + empty presented bearer = %d, want 401 (fail closed)", rec.Code)
	}
}
