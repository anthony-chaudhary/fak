package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAuthGateMissingCredentialsJSON(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "secret-key-123"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if got := rec.Header().Get("WWW-Authenticate"); got != `Bearer realm="fak-gateway"` {
		t.Errorf("WWW-Authenticate = %q, want Bearer realm=\"fak-gateway\"", got)
	}
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("json unmarshal failed: %v", err)
	}
	if env.Error.Code != "missing_credentials" {
		t.Errorf("error.code = %q, want missing_credentials", env.Error.Code)
	}
	if env.Error.Type != "authentication_error" {
		t.Errorf("error.type = %q, want authentication_error", env.Error.Type)
	}
	if !strings.HasPrefix(env.Error.Message, "missing or invalid credentials") {
		t.Errorf("error.message = %q, want prefix 'missing or invalid credentials'", env.Error.Message)
	}
}

func TestAuthGateInvalidCredentialsJSON(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "secret-key-123"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	req.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if got := rec.Header().Get("WWW-Authenticate"); got != `Bearer realm="fak-gateway"` {
		t.Errorf("WWW-Authenticate = %q, want Bearer realm=\"fak-gateway\"", got)
	}
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("json unmarshal failed: %v", err)
	}
	if env.Error.Code != "invalid_credentials" {
		t.Errorf("error.code = %q, want invalid_credentials", env.Error.Code)
	}
	if env.Error.Type != "authentication_error" {
		t.Errorf("error.type = %q, want authentication_error", env.Error.Type)
	}
	if !strings.Contains(env.Error.Message, "provided gateway token is incorrect") {
		t.Errorf("error.message = %q, want 'provided gateway token is incorrect'", env.Error.Message)
	}
}

func TestAuthGateBrowserHTMLGate(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "secret-key-123"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if !strings.Contains(rec.Header().Get("Content-Type"), "text/html") {
		t.Errorf("Content-Type = %q, want text/html", rec.Header().Get("Content-Type"))
	}
	body := rec.Body.String()
	if !strings.Contains(body, "Authentication Required") {
		t.Errorf("body missing 'Authentication Required'")
	}
	if !strings.Contains(body, `<input type="password" id="key-input" name="key"`) {
		t.Errorf("body missing key input field")
	}
	if !strings.Contains(body, "/etc/fak/gateway.env") {
		t.Errorf("body missing path to gateway.env")
	}
	if !strings.Contains(body, "fak-strix key") {
		t.Errorf("body missing reference to fak-strix key")
	}
}

func TestAuthGateBrowserHTMLGateInvalidKey(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "secret-key-123"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/?key=wrong-key", nil)
	req.Header.Set("Accept", "text/html")
	req.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "Invalid gateway key") {
		t.Errorf("body missing 'Invalid gateway key' error banner")
	}
}

func TestAuthGateValidQueryKeyAdmitsOffBox(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "secret-key-123"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/?key=secret-key-123", nil)
	req.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	raw, err := io.ReadAll(rec.Result().Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "fak gateway") {
		t.Errorf("expected gateway home page, got: %s", raw)
	}
}

func TestAuthGateAllowLAN(t *testing.T) {
	s, err := New(Config{
		EngineID:   "mock",
		Model:      "m",
		Provider:   "openai",
		RequireKey: "secret-key-123",
		AllowLAN:   true,
	})
	if err != nil {
		t.Fatal(err)
	}

	// 1. LAN caller (192.168.1.100) without token -> admitted (200 OK)
	recLAN := httptest.NewRecorder()
	reqLAN := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	reqLAN.RemoteAddr = "192.168.1.100:12345"
	s.Handler().ServeHTTP(recLAN, reqLAN)
	if recLAN.Code != http.StatusOK {
		t.Errorf("LAN caller without token = %d, want 200 OK", recLAN.Code)
	}

	// 2. LAN caller (10.0.0.5) to root / without token -> admitted (200 OK)
	recLANHome := httptest.NewRecorder()
	reqLANHome := httptest.NewRequest(http.MethodGet, "/", nil)
	reqLANHome.RemoteAddr = "10.0.0.5:12345"
	s.Handler().ServeHTTP(recLANHome, reqLANHome)
	if recLANHome.Code != http.StatusOK {
		t.Errorf("LAN caller to / without token = %d, want 200 OK", recLANHome.Code)
	}

	// 3. WAN caller (203.0.113.8) without token -> rejected (401 Unauthorized)
	recWAN := httptest.NewRecorder()
	reqWAN := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	reqWAN.RemoteAddr = "203.0.113.8:12345"
	s.Handler().ServeHTTP(recWAN, reqWAN)
	if recWAN.Code != http.StatusUnauthorized {
		t.Errorf("WAN caller without token = %d, want 401 Unauthorized", recWAN.Code)
	}

	// 4. WAN caller (203.0.113.8) WITH token -> admitted (200 OK)
	recWANAuthed := httptest.NewRecorder()
	reqWANAuthed := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	reqWANAuthed.Header.Set("Authorization", "Bearer secret-key-123")
	reqWANAuthed.RemoteAddr = "203.0.113.8:12345"
	s.Handler().ServeHTTP(recWANAuthed, reqWANAuthed)
	if recWANAuthed.Code != http.StatusOK {
		t.Errorf("WAN caller with valid token = %d, want 200 OK", recWANAuthed.Code)
	}
}
