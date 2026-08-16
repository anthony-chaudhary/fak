package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLiveToolPromptHasClosedReadOnlyClasses(t *testing.T) {
	p := liveToolPrompt(semanticRecord{Title: "x", Body: "y"}, `{"state":"open"}`)
	if strings.Contains(p, "write") || strings.Contains(p, "effect") {
		t.Fatal("effect authority leaked into prompt")
	}
	for _, x := range []string{"read_only|current_state", "TOOL_RECEIPT"} {
		if !strings.Contains(p, x) {
			t.Fatalf("missing %s", x)
		}
	}
}
func TestFetchIssueReceiptIsBoundedRead(t *testing.T) { // Exercise response shape without depending on GitHub rate state.
	old := http.DefaultClient
	defer func() { http.DefaultClient = old }()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"state":"open","updated_at":"2026-08-10T00:00:00Z","locked":false,"body":"secret large body"}`))
	}))
	defer srv.Close()
	// Transport rewrites only the authority, retaining the requested canonical path.
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		r.URL.Scheme = "http"
		r.URL.Host = strings.TrimPrefix(srv.URL, "http://")
		return http.DefaultTransport.RoundTrip(r)
	})}
	out, _, err := fetchIssueReceipt(context.Background(), semanticRecord{Number: 1})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "body") || !strings.Contains(out, "state") {
		t.Fatalf("unbounded receipt %s", out)
	}
}

func TestFetchIssueReceiptHonorsTimeoutWithInjectedTransport(t *testing.T) {
	old := http.DefaultClient
	defer func() { http.DefaultClient = old }()

	called := make(chan struct{}, 1)
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		called <- struct{}{}
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}

	start := time.Now()
	_, _, err := fetchIssueReceiptWithin(context.Background(), semanticRecord{Number: 1}, 20*time.Millisecond)
	if err == nil {
		t.Fatal("expected bounded request to time out")
	}
	select {
	case <-called:
	default:
		t.Fatal("injected default transport was bypassed")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("request exceeded bounded timeout: %s", elapsed)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestGitHubFallbackIsConfiguredBeforeOutput(t *testing.T) {
	p := "live_filter_tool_scheduler.go"
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	configure := strings.Index(s, "windowgate.ConfigureBackgroundCommand(cmd)")
	output := strings.Index(s, "out, ge := cmd.Output()")
	if configure < 0 || output < 0 || configure > output {
		t.Fatal("GitHub gh fallback must be configured windowlessly before Output")
	}
}
func TestVerifyLiveFilterToolArtifact(t *testing.T) {
	p := filepath.Join("..", "..", "experiments", "microcontext", "s8o-live-filter-tool-2026-08-10.json")
	if _, e := os.Stat(p); e != nil {
		t.Fatal(e)
	}
	if e := verifyLiveFilterToolMatrix(p); e != nil {
		t.Fatal(e)
	}
}
