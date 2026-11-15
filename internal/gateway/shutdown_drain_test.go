package gateway

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/bgloop"
)

func TestShutdownDrainTimeoutConfiguration(t *testing.T) {
	def := defaultHTTPDrainTimeout(90 * time.Second) // proxy write timeout -> derived default
	for _, tc := range []struct {
		raw  string
		want time.Duration
	}{
		{"", def}, {"1", time.Second}, {" 37 ", 37 * time.Second},
		{"0", def}, {"-1", def}, {"1.5", def},
		{"invalid", def}, {"9223372037", def}, {"9223372036854775807", def},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			if got := parseHTTPDrainTimeout(tc.raw, def); got != tc.want {
				t.Fatalf("drain timeout for %q = %s, want %s", tc.raw, got, tc.want)
			}
		})
	}
}

// TestDefaultHTTPDrainTimeoutCoversWriteTimeout pins the #14454 derivation: the
// default drain must cover the handler write timeout so a stream that legally
// runs up to WriteTimeout survives SIGTERM; a zero write timeout (in-kernel
// local model, whose turn can run for minutes) gets the longer floor instead.
func TestDefaultHTTPDrainTimeoutCoversWriteTimeout(t *testing.T) {
	for _, tc := range []struct {
		write time.Duration
		want  time.Duration
	}{
		{90 * time.Second, 100 * time.Second},  // proxy default covers the 90s write timeout
		{600 * time.Second, 610 * time.Second}, // raised inkernel manifest timeouts
		{0, 120 * time.Second},                 // no write timeout: the floor applies
	} {
		if got := defaultHTTPDrainTimeout(tc.write); got != tc.want {
			t.Errorf("defaultHTTPDrainTimeout(%s) = %s, want %s", tc.write, got, tc.want)
		}
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

// fak-test:runtime medium est=10s lane=default
// TestDrainInFlightStream witnesses the #14454 fix end-to-end: a streaming
// chat-completion request in flight when SIGTERM arrives (ctx cancel) completes
// under the DEFAULT drain window - no FAK_HTTP_DRAIN_TIMEOUT_S set - while
// /readyz reports stopping so new traffic routes away first. The upstream emits
// 8 chunks x 800ms (~6.4s total) and the cancel lands ~1s in, leaving more than
// the old fixed 5s default, which cut this exact shape mid-body on every
// rolling update.
func TestDrainInFlightStream(t *testing.T) {
	// The "test" engine id is a test-local stand-in (the same seam the other
	// gateway tests use); register it before New validates the Config.
	abi.RegisterEngine("test", echoEngine{})
	const chunks = 8
	const perChunk = 800 * time.Millisecond
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for i := 0; i < chunks; i++ {
			time.Sleep(perChunk)
			fmt.Fprintf(w, "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"chunk%d\"},\"finish_reason\":null}]}\n\n", i)
			fl.Flush()
		}
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":8,\"total_tokens\":9}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	loopEntered, releaseLoop := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseLoop) }) }
	srv.loops = bgloop.New()
	if err := srv.loops.Register(bgloop.Loop{Name: "drain-witness", Tick: func(ctx context.Context) error {
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
	t.Cleanup(func() { cancel(); release() })
	client := &http.Client{Timeout: 30 * time.Second}
	base := "http://" + ln.Addr().String()
	waitServing(t, client, base+"/readyz")
	select {
	case <-loopEntered:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not start")
	}
	firstChunk := make(chan struct{})
	var bodyMu sync.Mutex
	bodyBuf := &strings.Builder{}
	reqDone := make(chan error, 1)
	go func() {
		resp, err := client.Post(base+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`))
		if err != nil {
			reqDone <- err
			return
		}
		defer resp.Body.Close()
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			bodyMu.Lock()
			bodyBuf.WriteString(line + "\n")
			bodyMu.Unlock()
			if strings.Contains(line, "chunk0") {
				close(firstChunk)
			}
		}
		if err := scanner.Err(); err != nil {
			reqDone <- fmt.Errorf("stream read: %w", err)
			return
		}
		if resp.StatusCode != http.StatusOK {
			reqDone <- fmt.Errorf("stream status = %d, want 200", resp.StatusCode)
			return
		}
		reqDone <- nil
	}()
	select {
	case <-firstChunk:
	case <-time.After(5 * time.Second):
		t.Fatal("first stream chunk never arrived")
	}
	cancel() // SIGTERM: readiness flips and the in-flight stream must survive
	select {
	case line := <-shutdownLog:
		if !strings.Contains(line, "drain_timeout=1m40s") || !strings.Contains(line, "inflight_requests=1") {
			t.Fatalf("default drain window not derived from the 90s write timeout: %s", line)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown did not log before loop join")
	}
	if code, b := smokeGet(t, client, base+"/readyz"); code != http.StatusServiceUnavailable || b["ok"] != false || b["stopping"] != true {
		t.Fatalf("/readyz during drain = %d %v, want 503 ok=false stopping=true", code, b)
	}
	release()
	select {
	case err := <-reqDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("in-flight stream did not complete within the default drain window")
	}
	bodyMu.Lock()
	full := bodyBuf.String()
	bodyMu.Unlock()
	for i := 0; i < chunks; i++ {
		if !strings.Contains(full, "chunk"+strconv.Itoa(i)) {
			t.Fatalf("stream lost chunk%d after SIGTERM:\n%s", i, full)
		}
	}
	if !strings.Contains(full, "\"finish_reason\":\"stop\"") || !strings.Contains(full, "data: [DONE]") {
		t.Fatalf("stream did not finish cleanly after SIGTERM:\n%s", full)
	}
	select {
	case err := <-served:
		if err != nil {
			t.Fatalf("Serve returned %v, want a clean drain (no deadline exceeded)", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return after the in-flight stream drained")
	}
}
