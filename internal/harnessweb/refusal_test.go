package harnessweb

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name, method, target, body, want string
		code                             int
	}{
		{"missing message", http.MethodPost, "/api/runs", `{}`, "message is required", http.StatusBadRequest},
		{"invalid cursor", http.MethodGet, "/api/events?run=x&after=not-a-number", "", "invalid cursor", http.StatusBadRequest},
		{"invalid approval", http.MethodPost, "/api/approvals", `{}`, "invalid approval", http.StatusBadRequest},
	}
	h := handler(newStore())
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			r := httptest.NewRequest(tc.method, tc.target, strings.NewReader(tc.body))
			if tc.body != "" {
				r.Header.Set("content-type", "application/json")
			}
			h.ServeHTTP(w, r)
			if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.want) {
				t.Fatalf("status=%d body=%q, want %d containing %q", w.Code, w.Body.String(), tc.code, tc.want)
			}
		})
	}
}
