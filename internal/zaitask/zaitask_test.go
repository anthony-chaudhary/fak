package zaitask

import (
	"context"
	"errors"
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

func TestClassifySuitableWork(t *testing.T) {
	for _, tc := range []struct{ name, prompt, class string }{
		{"explicit light", "review these names", "light"},
		{"gardening", "deduplicate this list", "gardening"},
		{"short trivial", "reply with OK", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.prompt, tc.class); !got.Suitable {
				t.Fatalf("Classify() = %+v", got)
			}
		})
	}
}

func TestClassifyRefusesFrontierWork(t *testing.T) {
	for _, class := range []string{"hard", "engineering", "apex"} {
		if got := Classify("implement a security-critical scheduler", class); got.Suitable {
			t.Fatalf("class %q = %+v, want refusal", class, got)
		}
	}
}

func TestRunErrorMessagesNameRecovery(t *testing.T) {
	tests := []struct {
		name   string
		client Client
		prompt string
		want   string
	}{
		{name: "requires API key", client: Client{}, prompt: "task", want: "set Client.APIKey"},
		{name: "requires prompt", client: Client{APIKey: "secret"}, prompt: "  ", want: "provide a non-empty task prompt"},
		{name: "invalid base URL", client: Client{APIKey: "secret", BaseURL: ":"}, prompt: "task", want: "valid HTTP(S) URL"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.client.Run(context.Background(), tc.prompt, "", 1)
			if err == nil || !strings.Contains(err.Error(), "recovery:") || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want recovery naming %q", err, tc.want)
			}
		})
	}

	t.Run("request transport error", func(t *testing.T) {
		client := Client{APIKey: "secret", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("offline")
		})}}
		_, err := client.Run(context.Background(), "task", "", 1)
		assertRecovery(t, err, "check endpoint/network availability")
	})

	t.Run("response read error", func(t *testing.T) {
		client := Client{APIKey: "secret", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: failingBody{}}, nil
		})}}
		_, err := client.Run(context.Background(), "task", "", 1)
		assertRecovery(t, err, "inspect the endpoint transport")
	})

	for _, tc := range []struct {
		name, body, want string
		status           int
	}{
		{name: "oversize response", body: strings.Repeat("x", (8<<20)+1), status: http.StatusOK, want: "smaller max_tokens"},
		{name: "invalid JSON", body: "not-json", status: http.StatusOK, want: "provider compatibility"},
		{name: "provider refusal", body: `{"error":{"message":"resource exhausted"}}`, status: http.StatusTooManyRequests, want: "quota"},
		{name: "missing choices", body: `{}`, status: http.StatusOK, want: "provider response compatibility"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()
			_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).Run(context.Background(), "task", "", 1)
			assertRecovery(t, err, tc.want)
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (failingBody) Close() error             { return nil }

func assertRecovery(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), "recovery:") || !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %v, want recovery naming %q", err, want)
	}
}
