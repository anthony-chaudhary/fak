package modelperfobs

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

type adversarialProxyBody struct {
	data []byte
	err  error
	read bool
}

func (b *adversarialProxyBody) Read(p []byte) (int, error) {
	if b.read {
		return 0, b.err
	}
	b.read = true
	n := copy(p, b.data)
	b.data = b.data[n:]
	if len(b.data) > 0 {
		b.read = false
		return n, nil
	}
	return n, b.err
}

func (*adversarialProxyBody) Close() error { return nil }

func TestProxyEarlyFailureEdgeAndAdversarialInputs(t *testing.T) {
	oversized := bytes.Repeat([]byte("x"), 16*1024*1024+1)
	tests := []struct {
		name       string
		body       func() io.ReadCloser
		method     string
		wantStatus int
		wantError  string
	}{
		{
			name:       "Edge/empty_body_read_failure",
			body:       func() io.ReadCloser { return &adversarialProxyBody{err: errors.New("empty body read failure")} },
			method:     http.MethodPost,
			wantStatus: http.StatusBadRequest,
			wantError:  "empty body read failure",
		},
		{
			name: "Edge/oversized_partial_body_read_failure",
			body: func() io.ReadCloser {
				return &adversarialProxyBody{data: append([]byte(nil), oversized...), err: errors.New("oversized body read failure")}
			},
			method:     http.MethodPost,
			wantStatus: http.StatusBadRequest,
			wantError:  "oversized body read failure",
		},
		{
			name:       "Edge/malformed_method_request_construction_failure",
			body:       func() io.ReadCloser { return io.NopCloser(strings.NewReader("{")) },
			method:     "malformed method",
			wantStatus: http.StatusBadGateway,
			wantError:  "invalid method",
		},
		{
			name:       "Adversarial/hostile_method_request_construction_failure",
			body:       func() io.ReadCloser { return io.NopCloser(strings.NewReader(`{"model":"hostile","stream":true}`)) },
			method:     "POST\r\nX-Injected: true",
			wantStatus: http.StatusBadGateway,
			wantError:  "invalid method",
		},
		{
			name: "Adversarial/hostile_body_read_error_is_json_safe",
			body: func() io.ReadCloser {
				return &adversarialProxyBody{data: []byte(`{"model":"ignored"}`), err: errors.New("hostile\nerror\t\u0000")}
			},
			method:     http.MethodPost,
			wantStatus: http.StatusBadRequest,
			wantError:  "hostile\nerror\t\u0000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend, err := ParseBackend("http://backend.test")
			if err != nil {
				t.Fatal(err)
			}
			ledger := t.TempDir() + "/observations.jsonl"
			proxy := &Proxy{Backend: backend, Ledger: ledger}
			req := httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", nil)
			req.Method = tt.method
			req.Body = tt.body()
			w := httptest.NewRecorder()

			proxy.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Fatalf("response status=%d, want %d; body=%q", w.Code, tt.wantStatus, w.Body.String())
			}
			rows := readAdversarialProxyObservations(t, ledger)
			if len(rows) != 1 {
				t.Fatalf("ledger rows=%d, want exactly 1: %+v", len(rows), rows)
			}
			got := rows[0]
			if got.Schema != Schema || got.RequestID == "" || got.Timestamp.IsZero() || got.CompletedAt.IsZero() {
				t.Fatalf("observation lacks required identity/completion fields: %+v", got)
			}
			if got.Backend != backend.String() || got.Status != 0 {
				t.Fatalf("observation backend/status=(%q,%d), want (%q,0): %+v", got.Backend, got.Status, backend.String(), got)
			}
			if !strings.Contains(got.Error, tt.wantError) {
				t.Fatalf("observation error=%q, want substring %q", got.Error, tt.wantError)
			}
			proxy.mu.Lock()
			defer proxy.mu.Unlock()
			if len(proxy.active) != 0 || len(proxy.overlaps) != 0 {
				t.Fatalf("active request state leaked: active=%v overlaps=%v", proxy.active, proxy.overlaps)
			}
		})
	}
}

func readAdversarialProxyObservations(t *testing.T, ledger string) []Observation {
	t.Helper()
	f, err := os.Open(ledger)
	if err != nil {
		t.Fatalf("early failure did not create observation ledger: %v", err)
	}
	rows, readErr := ReadObservations(bufio.NewReader(f))
	closeErr := f.Close()
	if readErr != nil {
		t.Fatal(readErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	return rows
}
