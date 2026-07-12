package zaitask

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRunReturnsContentAndReceipt(t *testing.T) {
	var gotStream bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("authorization = %q", r.Header.Get("Authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		gotStream = strings.Contains(string(body), `"stream":false`)
		if !strings.Contains(string(body), `"thinking":{"type":"disabled"}`) {
			t.Errorf("request did not disable thinking: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"done"}}],"model":"glm-5.2","request_id":"req-1","usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer srv.Close()
	got, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).Run(context.Background(), "task", "", 20)
	if err != nil {
		t.Fatal(err)
	}
	if got.Content != "done" || got.RequestID != "req-1" || got.Usage.TotalTokens != 5 {
		t.Fatalf("result = %+v", got)
	}
	if !gotStream {
		t.Fatal("request did not explicitly disable streaming")
	}
}
func TestRunReportsAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":{"message":"resource exhausted"}}`))
	}))
	defer srv.Close()
	_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).Run(context.Background(), "task", "", 20)
	if err == nil || !strings.Contains(err.Error(), "resource exhausted") {
		t.Fatalf("error = %v", err)
	}
}
func TestRunRequiresAPIKey(t *testing.T) {
	if _, err := (Client{}).Run(context.Background(), "task", "", 1); err == nil {
		t.Fatal("expected error")
	}
}
func TestRunHonorsContextTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { time.Sleep(200 * time.Millisecond) }))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).Run(ctx, "task", "", 1)
	if err == nil || !strings.Contains(err.Error(), "deadline exceeded") {
		t.Fatalf("error = %v", err)
	}
}

func TestRunRejectsOversizeResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat("x", (8<<20)+1))
	}))
	defer srv.Close()
	_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).Run(context.Background(), "task", "", 20)
	if err == nil || !strings.Contains(err.Error(), "exceeds 8 MiB") {
		t.Fatalf("error = %v", err)
	}
}
