package harnessweb

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

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
