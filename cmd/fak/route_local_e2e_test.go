package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

type routerStatusTargetView struct {
	Account             string `json:"account"`
	Kind                string `json:"kind"`
	UpstreamModel       string `json:"upstream_model"`
	Locality            string `json:"locality"`
	State               string `json:"state"`
	CooldownRemainingMs int64  `json:"cooldown_remaining_ms"`
}

type routerStatusRouteView struct {
	Model          string                   `json:"model"`
	ResidencyFloor string                   `json:"residency_floor,omitempty"`
	Primary        routerStatusTargetView   `json:"primary"`
	Fallbacks      []routerStatusTargetView `json:"fallbacks"`
}

type routerStatusView struct {
	Schema         string                  `json:"schema"`
	RoutingEnabled bool                    `json:"routing_enabled"`
	DefaultModel   string                  `json:"default_model"`
	Routes         []routerStatusRouteView `json:"routes"`
	Totals         struct {
		Attempts  int64 `json:"attempts"`
		Successes int64 `json:"successes"`
		Failovers int64 `json:"failovers"`
	} `json:"totals"`
}

func newMockOpenAIUpstream(t *testing.T, status int, reply string, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		var reqBody struct {
			Model string `json:"model"`
		}
		if raw, err := io.ReadAll(r.Body); err == nil {
			_ = json.Unmarshal(raw, &reqBody)
		}
		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":{"message":"upstream unavailable"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-e2e",
			"object":  "chat.completion",
			"model":   reqBody.Model,
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": reply}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
		})
	}))
	t.Cleanup(ts.Close)
	return ts
}

func postGatewayChat(t *testing.T, baseURL, model string) (int, string) {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]any{{"role": "user", "content": "ping"}},
	})
	if err != nil {
		t.Fatalf("marshal chat request: %v", err)
	}
	resp, err := http.Post(baseURL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions (%s): %v", model, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read chat response (%s): %v", model, err)
	}
	return resp.StatusCode, string(raw)
}

// fak-test:runtime fast est=2s
func TestRouteLocalE2E(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	t.Setenv("OPENAI_API_KEY", "sk-e2e-openai")
	t.Setenv("ANTHROPIC_API_KEY", "sk-e2e-anthropic")

	var localHits, cloudHits atomic.Int32
	localMock := newMockOpenAIUpstream(t, http.StatusServiceUnavailable, "", &localHits)
	cloudMock := newMockOpenAIUpstream(t, http.StatusOK, "ok-from-openai", &cloudHits)

	outDir := filepath.Join(t.TempDir(), "router")
	var stdout, stderr bytes.Buffer
	if rc := runRouteInit(&stdout, &stderr, []string{"--profile", "hybrid", "--out-dir", outDir, "--json"}); rc != 0 {
		t.Fatalf("runRouteInit failed (rc=%d): %s", rc, stderr.String())
	}

	var receipt routeInitReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode routeInitReceipt: %v\n%s", err, stdout.String())
	}

	roster, err := modelroute.LoadRoster(receipt.AccountsPath)
	if err != nil {
		t.Fatalf("LoadRoster: %v", err)
	}
	manifest, err := modelroute.LoadManifest(receipt.ManifestPath)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}

	for i := range roster.Accounts {
		switch roster.Accounts[i].ID {
		case "local-gpu":
			roster.Accounts[i].BaseURL = localMock.URL + "/v1"
		case "openai":
			roster.Accounts[i].BaseURL = cloudMock.URL + "/v1"
		}
	}

	srv, err := gateway.New(gateway.Config{
		ExposeProfile: "headless",
		RouteManifest: &manifest,
		RouteAccounts: &roster,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(srv.Close)

	gwTS := httptest.NewServer(srv.Handler())
	t.Cleanup(gwTS.Close)

	// (a) 503 failover on the `default` binding: local-gpu returns 503, fails over to openai.
	code, chatBody := postGatewayChat(t, gwTS.URL, "default")
	if code != http.StatusOK {
		t.Fatalf("default chat status=%d, want 200; body=%s", code, chatBody)
	}
	if !strings.Contains(chatBody, "ok-from-openai") {
		t.Fatalf("default chat body=%s, want ok-from-openai", chatBody)
	}
	if localHits.Load() != 1 || cloudHits.Load() != 1 {
		t.Fatalf("after default failover: localHits=%d cloudHits=%d, want 1 and 1", localHits.Load(), cloudHits.Load())
	}

	// (b) fail-closed refusal on the `private` (`local_only`) binding when local primary is unavailable.
	privCode, privBody := postGatewayChat(t, gwTS.URL, "private")
	if privCode == http.StatusOK {
		t.Fatalf("private chat unexpectedly succeeded: body=%s", privBody)
	}
	if cloudHits.Load() != 1 {
		t.Fatalf("cloud fallback was contacted during private local_only refusal: cloudHits=%d, want 1", cloudHits.Load())
	}

	// (c) GET /v1/router/status reflects cooldown and failover counters.
	statusResp, err := http.Get(gwTS.URL + "/v1/router/status")
	if err != nil {
		t.Fatalf("GET /v1/router/status: %v", err)
	}
	defer statusResp.Body.Close()
	if statusResp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(statusResp.Body)
		t.Fatalf("GET /v1/router/status status=%d, want 200; body=%s", statusResp.StatusCode, string(raw))
	}

	var status routerStatusView
	if err := json.NewDecoder(statusResp.Body).Decode(&status); err != nil {
		t.Fatalf("decode /v1/router/status: %v", err)
	}
	if status.Schema != "fak-router-status/v1" || !status.RoutingEnabled {
		t.Fatalf("unexpected router status header: schema=%q routing_enabled=%v", status.Schema, status.RoutingEnabled)
	}
	if status.Totals.Failovers < 1 || status.Totals.Successes < 1 || status.Totals.Attempts < 2 {
		t.Fatalf("unexpected router totals: %+v, want failovers>=1 successes>=1 attempts>=2", status.Totals)
	}

	routesByModel := make(map[string]routerStatusRouteView, len(status.Routes))
	for _, r := range status.Routes {
		routesByModel[r.Model] = r
	}

	defRoute, ok := routesByModel["default"]
	if !ok {
		t.Fatalf("missing default route in /v1/router/status: %+v", status.Routes)
	}
	if defRoute.Primary.State != "cooldown" || defRoute.Primary.CooldownRemainingMs <= 0 {
		t.Fatalf("default primary=%+v, want state=cooldown cooldown_remaining_ms>0", defRoute.Primary)
	}
	if len(defRoute.Fallbacks) != 1 || defRoute.Fallbacks[0].State != "healthy" {
		t.Fatalf("default fallbacks=%+v, want 1 healthy fallback", defRoute.Fallbacks)
	}

	privRoute, ok := routesByModel["private"]
	if !ok {
		t.Fatalf("missing private route in /v1/router/status: %+v", status.Routes)
	}
	if privRoute.ResidencyFloor != "local_only" || privRoute.Primary.State != "cooldown" {
		t.Fatalf("private route=%+v, want residency_floor=local_only and primary.state=cooldown", privRoute)
	}
}
