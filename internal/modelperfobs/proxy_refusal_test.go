package modelperfobs

import (
	"bufio"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

type refusingRoundTripper struct{ err error }

func (rt refusingRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, rt.err
}

func TestProxyEarlyFailureErrorsNameRecovery(t *testing.T) {
	tests := []struct {
		name       string
		wantStatus int
		wantCue    string
		request    func() *http.Request
		client     *http.Client
	}{
		{
			name:       "inbound body read",
			wantStatus: http.StatusBadRequest,
			wantCue:    "retry with a readable request body",
			request: func() *http.Request {
				r := httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", nil)
				r.Body = failingRequestBody{err: errors.New("forced inbound body read failure")}
				return r
			},
		},
		{
			name:       "outbound request construction",
			wantStatus: http.StatusBadGateway,
			wantCue:    "retry with a valid HTTP method and request target",
			request: func() *http.Request {
				r := httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", strings.NewReader(`{"model":"test"}`))
				r.Method = "invalid method"
				return r
			},
		},
		{
			name:       "backend transport",
			wantStatus: http.StatusBadGateway,
			wantCue:    "verify the backend URL and availability, then retry",
			request: func() *http.Request {
				return httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", strings.NewReader(`{"model":"test"}`))
			},
			client: &http.Client{Transport: refusingRoundTripper{err: errors.New("forced backend failure")}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend, err := ParseBackend("http://backend.test")
			if err != nil {
				t.Fatal(err)
			}
			ledger := t.TempDir() + "/observations.jsonl"
			proxy := &Proxy{Backend: backend, Ledger: ledger, Client: tt.client}
			w := httptest.NewRecorder()

			proxy.ServeHTTP(w, tt.request())

			if w.Code != tt.wantStatus {
				t.Fatalf("response status=%d, want %d; body=%q", w.Code, tt.wantStatus, w.Body.String())
			}
			if !strings.Contains(w.Body.String(), tt.wantCue) {
				t.Errorf("response %q does not name recovery %q", w.Body.String(), tt.wantCue)
			}
			f, err := os.Open(ledger)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := ReadObservations(bufio.NewReader(f))
			closeErr := f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if closeErr != nil {
				t.Fatal(closeErr)
			}
			if len(rows) != 1 {
				t.Fatalf("ledger rows=%d, want exactly 1: %+v", len(rows), rows)
			}
			if !strings.Contains(rows[0].Error, tt.wantCue) {
				t.Errorf("observation error %q does not name recovery %q", rows[0].Error, tt.wantCue)
			}
		})
	}
}
