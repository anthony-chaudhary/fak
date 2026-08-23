package gateway

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestCachePostureFlipAppliesAtNextRequestBoundary(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "claude", BaseURL: "https://api.anthropic.com", Provider: string(agent.ProviderAnthropic)})
	var logs bytes.Buffer
	priorOutput := log.Writer()
	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(priorOutput) })

	post := httptest.NewRequest(http.MethodPost, "/v1/fak/cache/posture", strings.NewReader(`{"mode":"on"}`))
	w := httptest.NewRecorder()
	srv.handleFakCachePosture(w, post)
	if w.Code != http.StatusOK {
		t.Fatalf("POST status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if !srv.cacheTTL1H.Load() {
		t.Fatal("POST on left managed-cache posture off")
	}
	if !strings.Contains(logs.String(), "managed-cache posture off->on") {
		t.Fatalf("flip log = %q, want prior->new posture", logs.String())
	}

	raw := []byte(`{"model":"claude","max_tokens":1024,"system":[{"type":"text","text":"stable policy","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !srv.maybeUpgradeAnthropicCacheTTL1H(req) || !bytes.Contains(req.Raw, []byte(`"ttl":"1h"`)) {
		t.Fatalf("next request did not reflect live posture: %s", req.Raw)
	}
	if got := srv.debugVars(time.Now()).ManagedCache; got == nil || !got.Active {
		t.Fatalf("debug managed-cache = %+v, want active", got)
	}

	get := httptest.NewRequest(http.MethodGet, "/v1/fak/cache/posture", nil)
	w = httptest.NewRecorder()
	srv.handleFakCachePosture(w, get)
	var posture cachePostureResponse
	if err := json.NewDecoder(w.Body).Decode(&posture); err != nil || posture.Mode != "on" {
		t.Fatalf("GET posture = %+v, err=%v", posture, err)
	}
}

func TestCachePostureRefusesUnresolvedModesWithoutChangingPosture(t *testing.T) {
	for _, mode := range []string{"auto", "unknown"} {
		t.Run(mode, func(t *testing.T) {
			srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "claude", BaseURL: "https://api.anthropic.com", Provider: string(agent.ProviderAnthropic)})
			srv.cacheTTL1H.Store(true)
			w := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, "/v1/fak/cache/posture", strings.NewReader(`{"mode":"`+mode+`"}`))
			srv.handleFakCachePosture(w, r)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", w.Code)
			}
			if !srv.cacheTTL1H.Load() {
				t.Fatal("refused posture changed the live lever")
			}
			if mode == "auto" && !strings.Contains(w.Body.String(), "launch-resolved") {
				t.Fatalf("auto refusal = %q, want launch reason", w.Body.String())
			}
		})
	}
}
