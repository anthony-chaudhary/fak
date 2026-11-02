package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestTurnkeyRefusesExplicitNativeBeamBeforeMockOrPlanner(t *testing.T) {
	s := &turnkeyServer{mock: true}
	for _, stream := range []string{"false", "true"} {
		body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":` + stream + `,"fak":{"beam_search":{"width":2}}}`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		w := httptest.NewRecorder()
		s.handleChatCompletions(w, r)
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "native beam search is not supported") || s.requestCount != 0 || s.activeRequests != 0 {
			t.Fatalf("turnkey ignored beam selection: %d %s", w.Code, w.Body.String())
		}
	}
}
