package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// TestGuardManagedCachePassthroughWire binds the guard's billing decision to the
// actual outbound body (#5295). It reuses the stable breakpoint from gateway's
// TestMaybeUpgradeCacheTTL1HGate, with a loopback provider and synthetic credentials.
// fak-test:runtime fast est=100ms
func TestGuardManagedCachePassthroughWire(t *testing.T) {
	// The deliberate ablation bypass is independent of the managed-cache posture.
	t.Setenv("FAK_ABLATE_TTL_1H", "")
	raw := []byte(`{"model":"claude","max_tokens":1024,` +
		`"system":[{"type":"text","text":"stable policy","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	for _, tc := range []struct {
		name        string
		mode        string
		apiKey      string
		wantUpgrade bool
	}{
		{name: "on unknown passthrough", mode: "on"},
		{name: "auto unknown passthrough", mode: "auto"},
		{name: "on proven api key", mode: "on", apiKey: "test-api-key", wantUpgrade: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			posture, err := resolveGuardManagedCache(tc.mode, guardManagedCacheInputs{
				provider: "anthropic", apiKey: tc.apiKey,
			})
			if err != nil {
				t.Fatal(err)
			}
			bodies := make(chan []byte, 1)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				bodies <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			}))
			defer upstream.Close()
			srv, err := gateway.New(gateway.Config{
				EngineID: "mock", Model: "claude", BaseURL: upstream.URL,
				Provider: "anthropic", APIKey: tc.apiKey, CacheTTL1H: posture.active,
			})
			if err != nil {
				t.Fatalf("gateway.New: %v", err)
			}
			defer srv.Close()
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			if tc.apiKey == "" {
				// Client-forwarded OAuth does not populate the guard's oauthSource.
				req.Header.Set("Authorization", "Bearer sk-ant-oat-not-a-real-token")
			}
			client := ts.Client()
			client.Timeout = 5 * time.Second
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
			}
			var body []byte
			select {
			case body = <-bodies:
			default:
				t.Fatal("request never reached the loopback upstream")
			}
			if got := bytes.Contains(body, []byte(`"ttl":"1h"`)); got != tc.wantUpgrade {
				t.Fatalf("outbound 1h TTL = %v, want %v; posture = %+v; body = %s", got, tc.wantUpgrade, posture, body)
			}
			if !tc.wantUpgrade && !bytes.Equal(body, raw) {
				t.Fatalf("passive managed cache changed the client's body:\nwant %s\ngot  %s", raw, body)
			}
		})
	}
}
