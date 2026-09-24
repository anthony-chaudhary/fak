package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const (
	controlIngressCollectionPath = "/v1/fak/control/directives"
	controlIngressItemPath       = "/v1/fak/control/directives/directive-1"
)

type recordingControlIngress struct {
	calls   int
	method  string
	urlPath string
}

func (h *recordingControlIngress) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.calls++
	h.method = r.Method
	h.urlPath = r.URL.Path
	w.WriteHeader(http.StatusNoContent)
}

func newControlIngressServer(t *testing.T, ingress ControlIngress, requireKey string, keyPrincipals map[string]string, allowLAN bool) *Server {
	t.Helper()
	srv, err := New(Config{
		EngineID:       "mock",
		Model:          "m",
		Provider:       "openai",
		ControlIngress: ingress,
		RequireKey:     requireKey,
		KeyPrincipals:  keyPrincipals,
		AllowLAN:       allowLAN,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func serveControlIngress(h http.Handler, method, path, remoteAddr, credential string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, nil)
	req.RemoteAddr = remoteAddr
	if credential != "" {
		req.Header.Set("X-Api-Key", credential)
	}
	h.ServeHTTP(rec, req)
	return rec
}

func assertControlIngressUnavailable(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if contentType := rec.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", contentType)
	}
	var receipt struct {
		State string `json:"state"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &receipt); err != nil {
		t.Fatalf("decode unavailable receipt: %v (body %q)", err, rec.Body.String())
	}
	if receipt.State != "unavailable" {
		t.Fatalf("receipt state = %q, want unavailable (body %q)", receipt.State, rec.Body.String())
	}
}

func TestControlIngressWithoutCredentialDoorIsUnavailable(t *testing.T) {
	for _, methodPath := range []struct {
		method string
		path   string
	}{
		{method: http.MethodPost, path: controlIngressCollectionPath},
		{method: http.MethodGet, path: controlIngressItemPath},
	} {
		for _, ingressPresent := range []bool{false, true} {
			name := methodPath.method + "/nil_ingress"
			var ingress *recordingControlIngress
			var configured ControlIngress
			if ingressPresent {
				name = methodPath.method + "/configured_ingress"
				ingress = &recordingControlIngress{}
				configured = ingress
			}
			t.Run(name, func(t *testing.T) {
				srv := newControlIngressServer(t, configured, "", nil, false)
				rec := serveControlIngress(srv.Handler(), methodPath.method, methodPath.path, "203.0.113.8:12345", "")
				assertControlIngressUnavailable(t, rec)
				if ingress != nil && ingress.calls != 0 {
					t.Fatalf("ingress calls = %d, want 0", ingress.calls)
				}
			})
		}
	}
}

func TestControlIngressCredentialDoorsRequireConfiguredIngress(t *testing.T) {
	doors := []struct {
		name          string
		requireKey    string
		keyPrincipals map[string]string
		validKey      string
	}{
		{name: "require_key", requireKey: "operator-secret", validKey: "operator-secret"},
		{name: "keyset", keyPrincipals: map[string]string{"tenant-secret": "tenant-a"}, validKey: "tenant-secret"},
	}
	for _, door := range doors {
		for _, methodPath := range []struct {
			method string
			path   string
		}{
			{method: http.MethodPost, path: controlIngressCollectionPath},
			{method: http.MethodGet, path: controlIngressItemPath},
		} {
			t.Run(door.name+"/"+methodPath.method, func(t *testing.T) {
				srv := newControlIngressServer(t, nil, door.requireKey, door.keyPrincipals, false)
				rec := serveControlIngress(srv.Handler(), methodPath.method, methodPath.path, "203.0.113.8:12345", door.validKey)
				assertControlIngressUnavailable(t, rec)
			})
		}
	}
}

func TestControlIngressCredentialDoorsAuthenticateBeforeDispatch(t *testing.T) {
	doors := []struct {
		name          string
		requireKey    string
		keyPrincipals map[string]string
		validKey      string
	}{
		{name: "require_key", requireKey: "operator-secret", validKey: "operator-secret"},
		{name: "keyset", keyPrincipals: map[string]string{"tenant-secret": "tenant-a"}, validKey: "tenant-secret"},
	}
	methods := []struct {
		method string
		path   string
	}{
		{method: http.MethodPost, path: controlIngressCollectionPath},
		{method: http.MethodGet, path: controlIngressItemPath},
	}
	for _, door := range doors {
		for _, methodPath := range methods {
			t.Run(door.name+"/"+methodPath.method, func(t *testing.T) {
				ingress := &recordingControlIngress{}
				srv := newControlIngressServer(t, ingress, door.requireKey, door.keyPrincipals, false)
				h := srv.Handler()

				if rec := serveControlIngress(h, methodPath.method, methodPath.path, "203.0.113.8:12345", ""); rec.Code != http.StatusUnauthorized {
					t.Fatalf("missing credential status = %d, want 401", rec.Code)
				}
				if rec := serveControlIngress(h, methodPath.method, methodPath.path, "203.0.113.8:12345", "wrong-secret"); rec.Code != http.StatusUnauthorized {
					t.Fatalf("invalid credential status = %d, want 401", rec.Code)
				}
				if ingress.calls != 0 {
					t.Fatalf("ingress calls before authentication = %d, want 0", ingress.calls)
				}

				rec := serveControlIngress(h, methodPath.method, methodPath.path, "203.0.113.8:12345", door.validKey)
				if rec.Code != http.StatusNoContent {
					t.Fatalf("valid credential status = %d, want 204", rec.Code)
				}
				if ingress.calls != 1 || ingress.method != methodPath.method || ingress.urlPath != methodPath.path {
					t.Fatalf("dispatch = (%d,%q,%q), want (1,%q,%q)", ingress.calls, ingress.method, ingress.urlPath, methodPath.method, methodPath.path)
				}
			})
		}
	}
}

func TestControlIngressAllowLANDoesNotBypassCredential(t *testing.T) {
	for _, methodPath := range []struct {
		method string
		path   string
	}{
		{method: http.MethodPost, path: controlIngressCollectionPath},
		{method: http.MethodGet, path: controlIngressItemPath},
	} {
		t.Run(methodPath.method, func(t *testing.T) {
			ingress := &recordingControlIngress{}
			srv := newControlIngressServer(t, ingress, "operator-secret", nil, true)
			h := srv.Handler()

			if rec := serveControlIngress(h, methodPath.method, methodPath.path, "192.168.1.100:12345", ""); rec.Code != http.StatusUnauthorized {
				t.Fatalf("LAN caller without credential status = %d, want 401", rec.Code)
			}
			if rec := serveControlIngress(h, methodPath.method, methodPath.path, "192.168.1.100:12345", "wrong-secret"); rec.Code != http.StatusUnauthorized {
				t.Fatalf("LAN caller with invalid credential status = %d, want 401", rec.Code)
			}
			if ingress.calls != 0 {
				t.Fatalf("ingress calls before authentication = %d, want 0", ingress.calls)
			}
			if rec := serveControlIngress(h, methodPath.method, methodPath.path, "192.168.1.100:12345", "operator-secret"); rec.Code != http.StatusNoContent {
				t.Fatalf("LAN caller with valid credential status = %d, want 204", rec.Code)
			}
			if ingress.calls != 1 {
				t.Fatalf("ingress calls = %d, want 1", ingress.calls)
			}
		})
	}
}

func TestControlIngressAcceptsOnlyDeclaredCredentialSchemes(t *testing.T) {
	const configuredSecret = "operator-secret"
	methods := []struct {
		method string
		path   string
	}{
		{method: http.MethodPost, path: controlIngressCollectionPath},
		{method: http.MethodGet, path: controlIngressItemPath},
	}
	credentials := []struct {
		name         string
		configure    func(*http.Request)
		wantStatus   int
		wantDispatch int
	}{
		{
			name: "valid_authorization_bearer",
			configure: func(r *http.Request) {
				r.Header.Set("Authorization", "Bearer "+configuredSecret)
			},
			wantStatus:   http.StatusNoContent,
			wantDispatch: 1,
		},
		{
			name: "invalid_authorization_bearer",
			configure: func(r *http.Request) {
				r.Header.Set("Authorization", "Bearer wrong-secret")
			},
			wantStatus: http.StatusUnauthorized,
		},
		{
			name: "x_goog_api_key",
			configure: func(r *http.Request) {
				r.Header.Set("X-Goog-Api-Key", configuredSecret)
			},
			wantStatus: http.StatusUnauthorized,
		},
		{
			name: "url_query_key",
			configure: func(r *http.Request) {
				query := r.URL.Query()
				query.Set("key", configuredSecret)
				r.URL.RawQuery = query.Encode()
			},
			wantStatus: http.StatusUnauthorized,
		},
	}

	for _, methodPath := range methods {
		for _, credential := range credentials {
			t.Run(methodPath.method+"/"+credential.name, func(t *testing.T) {
				ingress := &recordingControlIngress{}
				srv := newControlIngressServer(t, ingress, configuredSecret, nil, true)
				rec := httptest.NewRecorder()
				req := httptest.NewRequest(methodPath.method, methodPath.path, nil)
				req.RemoteAddr = "192.168.1.100:12345"
				credential.configure(req)

				srv.Handler().ServeHTTP(rec, req)

				if rec.Code != credential.wantStatus {
					t.Fatalf("status = %d, want %d", rec.Code, credential.wantStatus)
				}
				if ingress.calls != credential.wantDispatch {
					t.Fatalf("ingress calls = %d, want %d", ingress.calls, credential.wantDispatch)
				}
			})
		}
	}
}
