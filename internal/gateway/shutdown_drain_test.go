package gateway

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
)

func TestShutdownDrainTimeoutConfiguration(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want time.Duration
	}{
		{"", 5 * time.Second}, {"1", time.Second}, {" 37 ", 37 * time.Second},
		{"0", 5 * time.Second}, {"-1", 5 * time.Second}, {"1.5", 5 * time.Second},
		{"invalid", 5 * time.Second}, {"9223372037", 5 * time.Second},
		{"9223372036854775807", 5 * time.Second},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			if got := parseHTTPDrainTimeout(tc.raw); got != tc.want {
				t.Fatalf("drain timeout for %q = %s, want %s", tc.raw, got, tc.want)
			}
		})
	}
}

func TestShutdownReadinessPrecedesLoopJoinAndDrainsRequest(t *testing.T) {
	t.Setenv("FAK_HTTP_DRAIN_TIMEOUT_S", "2")
	srv := newTestServer(t)
	planner := newBlockingAdmissionPlanner()
	srv.planner = planner
	loopEntered, releaseLoop := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseLoop) }) }
	srv.loops = bgloop.New()
	if err := srv.loops.Register(bgloop.Loop{Name: "shutdown-witness", Tick: func(ctx context.Context) error {
		close(loopEntered)
		<-ctx.Done()
		<-releaseLoop
		return nil
	}}); err != nil {
		t.Fatal(err)
	}
	shutdownLog := make(chan string, 1)
	srv.logf = func(format string, args ...any) {
		line := fmt.Sprintf(format, args...)
		if strings.HasPrefix(line, "fak gateway shutdown:") {
			select {
			case shutdownLog <- line:
			default:
			}
		}
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx, ln) }()
	t.Cleanup(func() { cancel(); release(); planner.Release(); ln.Close() })
	client := &http.Client{Timeout: 4 * time.Second}
	base := "http://" + ln.Addr().String()
	waitServing(t, client, base+"/readyz")
	readyAt := srv.startup.snapshot().ready
	select {
	case <-loopEntered:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not start")
	}
	requestDone := make(chan error, 1)
	go func() {
		resp, err := client.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err == nil {
			defer resp.Body.Close()
			body, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				err = readErr
			} else if resp.StatusCode != 200 || !strings.Contains(string(body), `"content":"ok"`) {
				err = fmt.Errorf("drained response = %d %s", resp.StatusCode, body)
			}
		}
		requestDone <- err
	}()
	select {
	case <-planner.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("request did not reach planner")
	}
	cancel()
	select {
	case line := <-shutdownLog:
		if !strings.Contains(line, "drain_timeout=2s") || !strings.Contains(line, "inflight_requests=1") {
			t.Fatalf("shutdown log lacks configured window/request count: %s", line)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown did not log before loop join")
	}
	if code, body := smokeGet(t, client, base+"/readyz"); code != 503 || body["ok"] != false || body["startup_ready"] != true || body["stopping"] != true {
		t.Fatalf("readiness during loop join = %d %v", code, body)
	}
	if code, body := smokeGet(t, client, base+"/healthz"); code != 200 || body["ok"] != true {
		t.Fatalf("liveness changed during shutdown = %d %v", code, body)
	}
	if !srv.startup.snapshot().ready.Equal(readyAt) {
		t.Error("shutdown changed startup readiness timestamp")
	}
	release()
	planner.Release()
	select {
	case err := <-requestDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("inflight request did not drain")
	}
	select {
	case err := <-served:
		if err != nil {
			t.Fatalf("Serve shutdown: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Serve did not finish")
	}
}

func TestShutdownHTTPDrainHonorsConfiguredBound(t *testing.T) {
	t.Setenv("FAK_HTTP_DRAIN_TIMEOUT_S", "1")
	srv := newTestServer(t)
	planner := newBlockingAdmissionPlanner()
	srv.planner = planner
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx, ln) }()
	t.Cleanup(func() { cancel(); planner.Release(); ln.Close() })
	client := &http.Client{Timeout: 4 * time.Second}
	base := "http://" + ln.Addr().String()
	waitServing(t, client, base+"/readyz")
	requestDone := make(chan struct{})
	go func() {
		defer close(requestDone)
		resp, err := client.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}()
	select {
	case <-planner.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("request did not reach planner")
	}
	cancel()
	select {
	case err := <-served:
		if err != context.DeadlineExceeded {
			t.Fatalf("bounded shutdown = %v, want deadline exceeded", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("one-second HTTP drain override was not honored")
	}
	planner.Release()
	select {
	case <-requestDone:
	case <-time.After(3 * time.Second):
		t.Fatal("request cleanup did not finish")
	}
}
