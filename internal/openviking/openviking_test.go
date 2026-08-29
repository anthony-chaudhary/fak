package openviking

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestCapturedWireSpine(t *testing.T) {
	t.Parallel()
	const (
		key     = "ov-user-secret"
		account = "account-a"
		user    = "user-a"
		peer    = "fak-agent"
		session = "fak-session-42"
	)
	content := "remember the typed boundary"
	call := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		wantHeaders(t, r, map[string]string{"X-API-Key": key, "X-OpenViking-Account": account, "X-OpenViking-User": user, "X-OpenViking-Actor-Peer": peer})
		switch call {
		case 1:
			wantRequest(t, r, http.MethodGet, "/health")
			writeJSON(t, w, http.StatusOK, map[string]any{"status": "ok"})
		case 2:
			wantRequest(t, r, http.MethodPost, "/api/v1/search/search")
			var body map[string]any
			decodeJSON(t, r, &body)
			if body["mode"] != "context" || body["query"] != "typed boundary" || body["purpose"] != "coding" || body["max_tokens"] != float64(2048) {
				t.Fatalf("search body = %#v", body)
			}
			writeJSON(t, w, http.StatusOK, map[string]any{"status": "ok", "result": map[string]any{"entries": []map[string]any{{"uri": "viking://resources/proof.md", "category": "resource", "score": 0.91, "detail": "full", "text": "contract", "origin": "resource"}}, "rendered": "bounded context", "digest": "sha256:context", "stats": map[string]any{"tokens": 17}}, "telemetry": map[string]any{"request_id": "search-1"}, "profile": []string{"retrieval=3ms"}})
		case 3:
			wantRequest(t, r, http.MethodPost, "/api/v1/sessions/"+session+"/messages/batch")
			var body BatchMessagesRequest
			decodeJSON(t, r, &body)
			if len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.Messages[0].Content == nil || *body.Messages[0].Content != content {
				t.Fatalf("batch body = %#v", body)
			}
			writeJSON(t, w, http.StatusOK, map[string]any{"status": "ok", "result": map[string]any{"session_id": session, "message_count": 1, "added": 1, "pending_tokens": 6}})
		case 4:
			wantRequest(t, r, http.MethodPost, "/api/v1/sessions/"+session+"/commit")
			var body CommitRequest
			decodeJSON(t, r, &body)
			if body.KeepRecentCount != 4 {
				t.Fatalf("commit body = %#v", body)
			}
			writeJSON(t, w, http.StatusAccepted, map[string]any{"status": "ok", "result": map[string]any{"task_id": "task-7", "session_id": session, "status": "queued"}})
		default:
			t.Fatalf("unexpected call %d: %s %s", call, r.Method, r.URL.Path)
		}
	}))
	defer server.Close()
	client, err := NewClient(Config{BaseURL: server.URL + "/", APIKey: key, Account: account, User: user, ActorPeer: peer})
	if err != nil {
		t.Fatal(err)
	}
	health, err := client.Health(context.Background())
	if err != nil || health.Status != "ok" {
		t.Fatalf("health = %#v, %v", health, err)
	}
	search, err := client.SearchContext(context.Background(), SearchContextRequest{Query: "typed boundary", Purpose: "coding", MaxTokens: 2048})
	if err != nil {
		t.Fatal(err)
	}
	if search.Digest != "sha256:context" || len(search.Entries) != 1 || len(search.Meta.Profile) != 1 || !strings.Contains(string(search.Meta.Telemetry), "search-1") {
		t.Fatalf("search = %#v", search)
	}
	batch, err := client.BatchMessages(context.Background(), session, BatchMessagesRequest{Messages: []Message{{Role: "user", Content: &content, TurnID: "turn-1", MessageKind: "user_query"}}})
	if err != nil || batch.Added != 1 || batch.PendingTokens != 6 {
		t.Fatalf("batch = %#v, %v", batch, err)
	}
	commit, err := client.Commit(context.Background(), session, CommitRequest{KeepRecentCount: 4})
	if err != nil || commit.TaskID != "task-7" || commit.Status != "queued" {
		t.Fatalf("commit = %#v, %v", commit, err)
	}
	if call != 4 {
		t.Fatalf("calls = %d, want 4", call)
	}
}

func TestNewClientDefaultHTTPTimeoutIsBounded(t *testing.T) {
	client, err := NewClient(Config{BaseURL: "https://example.test"})
	if err != nil {
		t.Fatal(err)
	}
	if got := client.http.Timeout; got != defaultHTTPTimeout || got <= 0 {
		t.Fatalf("default OpenViking client timeout = %s, want %s", got, defaultHTTPTimeout)
	}
}

func TestErrorsRetainHTTPAndBusinessCodesWithoutSecrets(t *testing.T) {
	t.Parallel()
	const secret = "ov-secret-must-not-leak"
	for _, tc := range []struct {
		name       string
		httpStatus int
		code       string
	}{{"business error on HTTP success", http.StatusOK, "PERMISSION_DENIED"}, {"HTTP service error", http.StatusServiceUnavailable, "UNAVAILABLE"}} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, tc.httpStatus, map[string]any{"status": "error", "error": map[string]any{"code": tc.code, "message": "rejected " + secret, "details": map[string]any{"credential": secret}}})
			}))
			defer server.Close()
			client, err := NewClient(Config{BaseURL: server.URL, APIKey: secret})
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.SearchContext(context.Background(), SearchContextRequest{Query: "x"})
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %T %v, want *APIError", err, err)
			}
			if apiErr.HTTPStatus != tc.httpStatus || apiErr.Code != tc.code {
				t.Fatalf("error = %#v", apiErr)
			}
			if strings.Contains(apiErr.Error(), secret) || strings.Contains(apiErr.Message, secret) || strings.Contains(string(apiErr.Details), secret) {
				t.Fatalf("secret leaked through error: %#v", apiErr)
			}
		})
	}
}

func TestValidationRejectsBeforeIO(t *testing.T) {
	t.Parallel()
	for _, raw := range []string{"", "ftp://example.test", "https://", "https://user:pass@example.test", "https://example.test/base", "https://example.test?x=1"} {
		if _, err := NewClient(Config{BaseURL: raw}); err == nil {
			t.Errorf("NewClient(%q) succeeded", raw)
		}
	}
	var calls atomic.Int32
	client, err := NewClient(Config{BaseURL: "https://example.test", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("network should not be reached")
	})}})
	if err != nil {
		t.Fatal(err)
	}
	content := "x"
	for _, sessionID := range []string{"", "../other", `folder\other`, "a?b", "a/b"} {
		if _, err := client.BatchMessages(context.Background(), sessionID, BatchMessagesRequest{Messages: []Message{{Role: "user", Content: &content}}}); err == nil {
			t.Errorf("BatchMessages(%q) succeeded", sessionID)
		}
		if _, err := client.Commit(context.Background(), sessionID, CommitRequest{}); err == nil {
			t.Errorf("Commit(%q) succeeded", sessionID)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("validation made %d network calls", calls.Load())
	}
}

func TestDeadlineAndOfflineAreExplicitUnavailableErrors(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { <-r.Context().Done() }))
	defer server.Close()
	client, err := NewClient(Config{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err = client.Health(ctx)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != CodeUnavailable || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline error = %T %v", err, err)
	}
	const secret = "offline-secret"
	var calls atomic.Int32
	offline, err := NewClient(Config{BaseURL: "https://offline.example", APIKey: secret, HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, fmt.Errorf("dial failed with %s", secret)
	})}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = offline.SearchContext(context.Background(), SearchContextRequest{Query: "x"})
	if !errors.As(err, &apiErr) || apiErr.Code != CodeUnavailable || strings.Contains(err.Error(), secret) {
		t.Fatalf("offline error = %T %v", err, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("offline calls = %d, want one request and no fallback", calls.Load())
	}
}

func TestEmptyIdentityFieldsAreOmitted(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, name := range []string{"X-API-Key", "X-OpenViking-Account", "X-OpenViking-User", "X-OpenViking-Actor-Peer"} {
			if _, ok := r.Header[http.CanonicalHeaderKey(name)]; ok {
				t.Errorf("unexpected %s header", name)
			}
		}
		writeJSON(t, w, http.StatusOK, map[string]any{"status": "ok"})
	}))
	defer server.Close()
	client, err := NewClient(Config{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Health(context.Background()); err != nil {
		t.Fatal(err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func wantRequest(t *testing.T, r *http.Request, method, path string) {
	t.Helper()
	if r.Method != method || r.URL.Path != path {
		t.Fatalf("request = %s %s, want %s %s", r.Method, r.URL.Path, method, path)
	}
}
func wantHeaders(t *testing.T, r *http.Request, expected map[string]string) {
	t.Helper()
	for name, want := range expected {
		if got := r.Header.Get(name); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
	if got := r.Header.Get("Authorization"); got != "" {
		t.Errorf("unexpected Authorization = %q", got)
	}
}
func decodeJSON(t *testing.T, r *http.Request, out any) {
	t.Helper()
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(out); err != nil {
		t.Fatal(err)
	}
}
func writeJSON(t *testing.T, w http.ResponseWriter, status int, value any) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Fatal(err)
	}
}
