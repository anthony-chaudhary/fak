package localapphelper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
	e := NativeClient{Endpoint: srv.URL, Model: "qwen", Artifact: "sha256:abc", Revision: "r1"}
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
	_, err := (NativeClient{Endpoint: srv.URL, Model: "q", Artifact: "a", Revision: "r"}).Execute(context.Background(), TaskRequest{TaskID: "t", Payload: json.RawMessage(`{}`)})
	if err == nil {
		t.Fatal("bad gateway accepted")
	}
}
