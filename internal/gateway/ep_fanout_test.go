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
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-r.Context().Done()
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
		t.Fatal("fanout did not inherit inbound request cancellation")
	}
}
