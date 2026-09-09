package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func TestReadyzRequiresStartupAndReusesHealthState(t *testing.T) {
	srv := newAgentSessionsTestServer(t)

	before := httptest.NewRecorder()
	srv.Handler().ServeHTTP(before, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if before.Code != http.StatusServiceUnavailable {
		t.Fatalf("before MarkReady status = %d, want 503; body=%s", before.Code, before.Body.String())
	}
	var notReady map[string]any
	if err := json.Unmarshal(before.Body.Bytes(), &notReady); err != nil {
		t.Fatalf("decode not-ready response: %v", err)
	}
	if notReady["startup_ready"] != false || notReady["ok"] != false {
		t.Fatalf("not-ready response = %#v", notReady)
	}

	srv.MarkReady()
	after := httptest.NewRecorder()
	srv.Handler().ServeHTTP(after, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if after.Code != http.StatusOK {
		t.Fatalf("after MarkReady status = %d, want 200; body=%s", after.Code, after.Body.String())
	}
	var ready map[string]any
	if err := json.Unmarshal(after.Body.Bytes(), &ready); err != nil {
		t.Fatalf("decode ready response: %v", err)
	}
	if ready["startup_ready"] != true || ready["ok"] != true || ready["engine"] != "mock" {
		t.Fatalf("ready response = %#v", ready)
	}
}

func TestGatewayReadyLogging(t *testing.T) {
	abi.RegisterEngine("mock", engine.MockEngine)
	var logged []string
	logf := func(format string, args ...any) {
		logged = append(logged, fmt.Sprintf(format, args...))
	}
	cfg := Config{
		EngineID: "mock",
		Model:    "mock",
		Logf:     logf,
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	serveErr := make(chan error, 1)
	go func() {
		serveErr <- srv.Serve(ctx, ln)
	}()

	deadline := time.Now().Add(3 * time.Second)
	foundReady := false
	for time.Now().Before(deadline) {
		for _, line := range logged {
			if strings.Contains(line, "[READY]") && strings.Contains(line, ln.Addr().String()) {
				foundReady = true
				break
			}
		}
		if foundReady {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	_ = <-serveErr

	if !foundReady {
		t.Fatalf("expected [READY] in logs for %s, got: %v", ln.Addr().String(), logged)
	}
}

func TestGatewayWarmupReadyLogging(t *testing.T) {
	var logged []string
	logf := func(format string, args ...any) {
		logged = append(logged, fmt.Sprintf(format, args...))
	}
	srv := &Server{
		logf: logf,
	}
	addr := "127.0.0.1:9999"
	srv.boundAddr.Store(&addr)

	srv.ArmWarmupGate()
	srv.MarkWarmupComplete(150 * time.Millisecond)

	foundWarmupReady := false
	for _, line := range logged {
		if strings.Contains(line, "[READY]") && strings.Contains(line, "server warmup completed") && strings.Contains(line, addr) {
			foundWarmupReady = true
			break
		}
	}
	if !foundWarmupReady {
		t.Fatalf("expected [READY] warmup completion log, got: %v", logged)
	}
}
