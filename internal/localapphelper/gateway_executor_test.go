package localapphelper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeClientBindsNativeArtifactAndNoFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatal(r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"{\"score\":1}"}}]}`))
	}))
	defer srv.Close()
	e := NativeClient{Endpoint: srv.URL, Model: "qwen", Artifact: "sha256:abc", Revision: "r1", Complete: agent.CompleteLocalAppChat}
	got, err := e.Execute(context.Background(), TaskRequest{TaskID: "t", Task: "job-apply", Payload: json.RawMessage(`{"resume":"x"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if got.Receipt.Engine != "fak-native" || !strings.Contains(got.Receipt.Reason, "artifact=sha256:abc;fallback=none") {
		t.Fatalf("receipt=%+v", got.Receipt)
	}
	if len(got.Events) != 3 {
		t.Fatal("events")
	}
}
func TestNativeClientFailsClosedOnBadResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "bad", 500) }))
	defer srv.Close()
	_, err := (NativeClient{Endpoint: srv.URL, Model: "q", Artifact: "a", Revision: "r", Complete: agent.CompleteLocalAppChat}).Execute(context.Background(), TaskRequest{TaskID: "t", Payload: json.RawMessage(`{}`)})
	if err == nil {
		t.Fatal("bad gateway accepted")
	}
}

func TestNativeClientInjectsTransportWithDefaultTimeout(t *testing.T) {
	var called int
	complete := func(_ context.Context, client *http.Client, endpoint, model, system, user string) (string, error) {
		called++
		if client.Timeout != 10*time.Minute {
			t.Fatalf("timeout=%s, want 10m", client.Timeout)
		}
		if endpoint != "http://gateway" || model != "qwen" {
			t.Fatalf("identity=%q %q", endpoint, model)
		}
		if system != "Return only valid JSON for the requested app task." || user != `{"resume":"x"}` {
			t.Fatalf("messages=%q %q", system, user)
		}
		return `{"score":1}`, nil
	}
	e := NativeClient{Endpoint: "http://gateway", Model: "qwen", Artifact: "sha256:abc", Revision: "r1", Complete: complete}
	got, err := e.Execute(context.Background(), TaskRequest{TaskID: "t", Payload: json.RawMessage(`{"resume":"x"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || got.Receipt.Attempts != 1 || !strings.Contains(got.Receipt.Reason, "fallback=none") {
		t.Fatalf("called=%d receipt=%+v", called, got.Receipt)
	}
}

func TestNativeClientRequiresInjectedTransport(t *testing.T) {
	_, err := (NativeClient{Endpoint: "http://gateway", Model: "q", Artifact: "a", Revision: "r"}).Execute(context.Background(), TaskRequest{TaskID: "t", Payload: json.RawMessage(`{}`)})
	if err == nil || !strings.Contains(err.Error(), "no native chat transport configured") {
		t.Fatalf("err=%v", err)
	}
}
