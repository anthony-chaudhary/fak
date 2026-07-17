package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestEPFanoutURLsFromEnv(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "127.0.0.1:8001, http://127.0.0.1:8002/ ;https://rank3.example")
	got := epFanoutURLsFromEnv()
	want := []string{
		"http://127.0.0.1:8001/v1/chat/completions",
		"http://127.0.0.1:8002/v1/chat/completions",
		"https://rank3.example/v1/chat/completions",
	}
	if len(got) != len(want) {
		t.Fatalf("urls len = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("urls[%d] = %q, want %q (all=%v)", i, got[i], want[i], got)
		}
	}
}

func TestStartEPFanoutFollowersMirrorsBodyAndMarksFollower(t *testing.T) {
	body := `{"model":"glm-5.2","messages":[{"role":"user","content":"ok"}],"max_tokens":1}`
	seen := make(chan string, 1)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get(epFollowerHeader); got != "1" {
			t.Errorf("%s = %q, want 1", epFollowerHeader, got)
		}
		b, _ := io.ReadAll(r.Body)
		seen <- string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer peer.Close()
	t.Setenv("FAK_EP_FANOUT_ADDRS", peer.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rr := httptest.NewRecorder()
	srv := &Server{logf: t.Logf}
	wait, ok := srv.startEPFanoutFollowers(rr, req)
	if !ok {
		t.Fatalf("startEPFanoutFollowers refused: %d %s", rr.Code, rr.Body.String())
	}
	wait()
	if got := <-seen; got != body {
		t.Fatalf("follower body = %q, want %q", got, body)
	}
	restored, _ := io.ReadAll(req.Body)
	if string(restored) != body {
		t.Fatalf("front-rank body after fanout = %q, want original %q", restored, body)
	}
}

// TestStartEPFanoutFollowersMirrorsStreamingRequest captures the #4855 regression:
// a stream:true request used to return before starting any follower, so ranks 1-7
// never entered the shared forward pass and the front rank stalled on its first
// collective. The follower must now receive the identical streaming body, and the
// helper must drain the whole SSE stream (not a fixed prefix) without hanging the
// test process. Against the pre-fix code the follower is never called and this fails.
func TestStartEPFanoutFollowersMirrorsStreamingRequest(t *testing.T) {
	body := `{"model":"glm-5.2","messages":[{"role":"user","content":"ok"}],"max_tokens":32,"stream":true}`
	seen := make(chan string, 1)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get(epFollowerHeader); got != "1" {
			t.Errorf("%s = %q, want 1", epFollowerHeader, got)
		}
		b, _ := io.ReadAll(r.Body)
		seen <- string(b)
		// Emit a small SSE stream and terminate, mirroring a streaming follower's
		// per-step chunks so the helper's full-body drain returns.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer peer.Close()
	t.Setenv("FAK_EP_FANOUT_ADDRS", peer.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rr := httptest.NewRecorder()
	srv := &Server{logf: t.Logf}
	wait, ok := srv.startEPFanoutFollowers(rr, req)
	if !ok {
		t.Fatalf("startEPFanoutFollowers refused streaming request: %d %s", rr.Code, rr.Body.String())
	}
	waited := make(chan struct{})
	go func() { wait(); close(waited) }()
	select {
	case got := <-seen:
		if got != body {
			t.Fatalf("streaming follower body = %q, want %q", got, body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("streaming request did not fan out to the follower rank (#4855 stall)")
	}
	select {
	case <-waited:
	case <-time.After(5 * time.Second):
		t.Fatal("streaming fanout helper hung draining the follower response")
	}
	restored, _ := io.ReadAll(req.Body)
	if string(restored) != body {
		t.Fatalf("front-rank body after streaming fanout = %q, want original %q", restored, body)
	}
}

func TestStartEPFanoutFollowersSkipsFollowerRequests(t *testing.T) {
	called := make(chan struct{}, 1)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called <- struct{}{}
	}))
	defer peer.Close()
	t.Setenv("FAK_EP_FANOUT_ADDRS", peer.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[]}`))
	req.Header.Set(epFollowerHeader, "1")
	wait, ok := (&Server{}).startEPFanoutFollowers(httptest.NewRecorder(), req)
	if !ok {
		t.Fatal("follower request should bypass fanout, not refuse")
	}
	wait()
	select {
	case <-called:
		t.Fatal("follower request recursively fanned out")
	default:
	}
}

func TestEPFanoutClientInheritsRequestCancellation(t *testing.T) {
	if epFanoutClient.Timeout != 0 {
		t.Fatalf("epFanoutClient.Timeout = %s, want 0 so the inbound request owns the deadline", epFanoutClient.Timeout)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
	}))
	defer peer.Close()
	t.Setenv("FAK_EP_FANOUT_ADDRS", peer.URL)

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"slow"}]}`)).WithContext(ctx)
	wait, ok := (&Server{logf: t.Logf}).startEPFanoutFollowers(httptest.NewRecorder(), req)
	if !ok {
		t.Fatal("fanout refused")
	}
	<-started
	cancel()
	done := make(chan struct{})
	go func() { wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("fanout did not inherit inbound request cancellation")
	}
	close(release)
}
