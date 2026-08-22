package harnessweb

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestStatusModeTracksGatewayReachability(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer gateway.Close()

	unreachable := &liveAdapter{
		baseURL:         "http://127.0.0.1:1",
		offlineFallback: true,
		client:          &http.Client{Timeout: 100 * time.Millisecond},
	}
	reachable := &liveAdapter{baseURL: gateway.URL, offlineFallback: true, client: gateway.Client()}
	for _, tt := range []struct {
		name       string
		live       *liveAdapter
		mode       string
		configured bool
		reachable  bool
	}{
		{name: "configured but unreachable", live: unreachable, mode: "offline", configured: true},
		{name: "reachable", live: reachable, mode: "live", configured: true, reachable: true},
		{name: "explicitly offline", mode: "offline"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handlerWithLive(newStore(), tt.live).ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/status", nil))
			var status struct {
				Mode    string          `json:"mode"`
				Gateway gatewayOverview `json:"gateway"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil {
				t.Fatal(err)
			}
			if status.Mode != tt.mode || status.Gateway.Configured != tt.configured || status.Gateway.Reachable != tt.reachable {
				t.Fatalf("status = mode %q gateway %+v, want mode %q configured=%v reachable=%v", status.Mode, status.Gateway, tt.mode, tt.configured, tt.reachable)
			}
		})
	}

	if !strings.Contains(page, `status.textContent=data.mode==="live"?"live - gateway connected":"offline - gateway unavailable"`) {
		t.Fatal("browser headline does not derive from the API mode contract")
	}
}
func TestDefaultGatewayConnectsWithoutFlag(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.WriteHeader(http.StatusOK)
		case "/debug/vars":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer gateway.Close()

	live := &liveAdapter{baseURL: gateway.URL, offlineFallback: true, client: gateway.Client()}
	response := httptest.NewRecorder()
	handlerWithLive(newStore(), live).ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/status", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.Code)
	}
	var status struct {
		Gateway gatewayOverview `json:"gateway"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if !status.Gateway.Reachable || status.Gateway.URL != gateway.URL || status.Gateway.Target != gateway.URL {
		t.Fatalf("gateway = %+v, want reachable default target", status.Gateway)
	}
}

func TestDefaultGatewayUnavailableKeepsOfflineDemoUseful(t *testing.T) {
	live := &liveAdapter{
		baseURL:         "http://127.0.0.1:1",
		offlineFallback: true,
		client:          &http.Client{Timeout: 100 * time.Millisecond},
	}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/runs", strings.NewReader(`{"message":"book a meeting"}`))
	handlerWithLive(newStore(), live).ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "LIVE_FAK_ERROR") {
		t.Fatalf("default gateway outage should fall back to the offline demo: %s", response.Body.String())
	}
}

func TestGatewaySetupCopyNamesAutomaticDefault(t *testing.T) {
	for _, unwanted := range []string{"shown now; connect with -fak-url to enable", "add -fak-url for gateway state"} {
		if strings.Contains(page, unwanted) {
			t.Fatalf("page retains manual setup copy %q", unwanted)
		}
	}
	for _, want := range []string{"Start fak serve", "connects automatically"} {
		if !strings.Contains(page, want) {
			t.Fatalf("page missing automatic setup copy %q", want)
		}
	}
	if defaultFakURL != "http://127.0.0.1:8080" {
		t.Fatalf("default gateway = %q, want fak serve default", defaultFakURL)
	}
}
