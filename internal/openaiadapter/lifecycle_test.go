package openaiadapter

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Invariant: OpenAI adapter wire servers must enforce bearer authentication and validate chat completions schema.
// Guard: Handler returns unauthorized on invalid tokens and refuses general-purpose LAN bindings.

func TestOpenAIAdapterLifecycle(t *testing.T) {
	t.Parallel()

	s := server()
	h := s.Handler()

	// Unauthorized request
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected status unauthorized (401), got %d", w.Code)
	}
}
