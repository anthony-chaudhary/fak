package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func newTestControlGateway(t *testing.T, cfg Config) *Server {
	t.Helper()
	if cfg.EngineID == "" {
		cfg.EngineID = "mock"
	}
	if cfg.Model == "" {
		cfg.Model = "mock"
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func TestControlConfigGet(t *testing.T) {
	srv := newTestControlGateway(t, Config{
		CompactHistoryBudget: 32000,
		CompactAnchorHead:    true,
	})
	handler := srv.Handler()

	for _, path := range []string{"/v1/control/config", "/v1/fak/control/config"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		if rr.Code != http.StatusOK {
			t.Fatalf("%s returned status %d, want 200 (body: %s)", path, rr.Code, rr.Body.String())
		}
		epochHeader := rr.Header().Get("X-Fak-Config-Epoch")
		if epochHeader != "1" {
			t.Fatalf("expected X-Fak-Config-Epoch: 1, got %q", epochHeader)
		}

		var resp ControlConfigResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}

		if resp.ConfigEpoch != 1 {
			t.Errorf("ConfigEpoch = %d, want 1", resp.ConfigEpoch)
		}
		if resp.Config.CompactHistoryBudget != 32000 {
			t.Errorf("CompactHistoryBudget = %d, want 32000", resp.Config.CompactHistoryBudget)
		}
		if resp.Config.CompactAnchorHead != 1 {
			t.Errorf("CompactAnchorHead = %d, want 1", resp.Config.CompactAnchorHead)
		}
		if resp.Config.MaxWaitingSeqs != 1024 {
			t.Errorf("MaxWaitingSeqs = %d, want 1024", resp.Config.MaxWaitingSeqs)
		}
		if resp.Config.LogLevel != "info" {
			t.Errorf("LogLevel = %q, want 'info'", resp.Config.LogLevel)
		}
	}
}

func TestControlConfigPatch(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	handler := srv.Handler()

	deadline := uint32(15000)
	streamTimeout := uint32(120000)
	maxWait := uint32(512)
	compactBudget := 64000
	compactAnchor := 1
	logLvl := "debug"

	patch := ScalarConfigPatch{
		CompletionDeadlineMs:    &deadline,
		StreamProgressTimeoutMs: &streamTimeout,
		MaxWaitingSeqs:          &maxWait,
		CompactHistoryBudget:    &compactBudget,
		CompactAnchorHead:       &compactAnchor,
		LogLevel:                &logLvl,
	}

	bodyBytes, _ := json.Marshal(patch)
	req := httptest.NewRequest(http.MethodPatch, "/v1/control/config", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("PATCH returned status %d, want 200 (body: %s)", rr.Code, rr.Body.String())
	}
	if got := rr.Header().Get("X-Fak-Config-Epoch"); got != "2" {
		t.Errorf("X-Fak-Config-Epoch = %q, want '2'", got)
	}
	if got := rr.Header().Get("X-Fak-Witness"); got != "verified-atomic-swap" {
		t.Errorf("X-Fak-Witness = %q, want 'verified-atomic-swap'", got)
	}

	var resp ControlConfigResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Status != "applied" {
		t.Errorf("Status = %q, want 'applied'", resp.Status)
	}
	if resp.ConfigEpoch != 2 {
		t.Errorf("ConfigEpoch = %d, want 2", resp.ConfigEpoch)
	}
	if resp.Config.CompletionDeadlineMs != 15000 {
		t.Errorf("CompletionDeadlineMs = %d, want 15000", resp.Config.CompletionDeadlineMs)
	}
	if resp.Config.StreamProgressTimeoutMs != 120000 {
		t.Errorf("StreamProgressTimeoutMs = %d, want 120000", resp.Config.StreamProgressTimeoutMs)
	}
	if resp.Config.MaxWaitingSeqs != 512 {
		t.Errorf("MaxWaitingSeqs = %d, want 512", resp.Config.MaxWaitingSeqs)
	}
	if resp.Config.CompactHistoryBudget != 64000 {
		t.Errorf("CompactHistoryBudget = %d, want 64000", resp.Config.CompactHistoryBudget)
	}
	if resp.Config.CompactAnchorHead != 1 {
		t.Errorf("CompactAnchorHead = %d, want 1", resp.Config.CompactAnchorHead)
	}
	if resp.Config.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want 'debug'", resp.Config.LogLevel)
	}

	// Verify subsequent GET reflects epoch 2 and updated values
	getReq := httptest.NewRequest(http.MethodGet, "/v1/control/config", nil)
	getRR := httptest.NewRecorder()
	handler.ServeHTTP(getRR, getReq)

	var getResp ControlConfigResponse
	_ = json.Unmarshal(getRR.Body.Bytes(), &getResp)
	if getResp.ConfigEpoch != 2 {
		t.Errorf("GET ConfigEpoch = %d, want 2", getResp.ConfigEpoch)
	}
	if getResp.Config.MaxWaitingSeqs != 512 {
		t.Errorf("GET MaxWaitingSeqs = %d, want 512", getResp.Config.MaxWaitingSeqs)
	}
}

func TestControlConfigValidation(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	handler := srv.Handler()

	tests := []struct {
		name  string
		patch ScalarConfigPatch
	}{
		{
			name: "completion deadline exceeding ceiling",
			patch: func() ScalarConfigPatch {
				bad := uint32(5000000)
				return ScalarConfigPatch{CompletionDeadlineMs: &bad}
			}(),
		},
		{
			name: "stream progress timeout exceeding ceiling",
			patch: func() ScalarConfigPatch {
				bad := uint32(5000000)
				return ScalarConfigPatch{StreamProgressTimeoutMs: &bad}
			}(),
		},
		{
			name: "negative compact history budget",
			patch: func() ScalarConfigPatch {
				bad := -5
				return ScalarConfigPatch{CompactHistoryBudget: &bad}
			}(),
		},
		{
			name: "negative compact anchor head",
			patch: func() ScalarConfigPatch {
				bad := -1
				return ScalarConfigPatch{CompactAnchorHead: &bad}
			}(),
		},
		{
			name: "compact anchor head exceeding 1",
			patch: func() ScalarConfigPatch {
				bad := 2
				return ScalarConfigPatch{CompactAnchorHead: &bad}
			}(),
		},
		{
			name: "stream progress timeout below minimum floor",
			patch: func() ScalarConfigPatch {
				bad := uint32(2000)
				return ScalarConfigPatch{StreamProgressTimeoutMs: &bad}
			}(),
		},
		{
			name: "invalid log level",
			patch: func() ScalarConfigPatch {
				bad := "nonexistent"
				return ScalarConfigPatch{LogLevel: &bad}
			}(),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bodyBytes, _ := json.Marshal(tc.patch)
			req := httptest.NewRequest(http.MethodPatch, "/v1/control/config", bytes.NewReader(bodyBytes))
			rr := httptest.NewRecorder()
			handler.ServeHTTP(rr, req)

			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400 Bad Request for %s, got %d (body: %s)", tc.name, rr.Code, rr.Body.String())
			}
		})
	}

	// Verify disallowed method returns 405
	postReq := httptest.NewRequest(http.MethodPost, "/v1/control/config", bytes.NewReader([]byte("{}")))
	postRR := httptest.NewRecorder()
	handler.ServeHTTP(postRR, postReq)
	if postRR.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 Method Not Allowed for POST, got %d", postRR.Code)
	}
}

func TestControlConfigConcurrentReadWriteRaceFree(t *testing.T) {
	srv := newTestControlGateway(t, Config{})

	const readers = 30
	const iterations = 500

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Concurrently read scalar config lock-free
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					sc := srv.ScalarConfig()
					_ = sc.CompletionDeadlineMs
					_ = sc.StreamProgressTimeoutMs
					_ = sc.MaxWaitingSeqs
					_ = sc.CompactHistoryBudget
					_ = sc.CompactAnchorHead
					_ = sc.LogLevel
					_ = srv.ConfigEpoch()
				}
			}
		}()
	}

	// Concurrently mutate configuration
	for i := 1; i <= iterations; i++ {
		budget := 1000 + i
		wait := uint32(100 + (i % 50))
		_, epoch, err := srv.PatchScalarConfig(ScalarConfigPatch{
			CompactHistoryBudget: &budget,
			MaxWaitingSeqs:       &wait,
		})
		if err != nil {
			t.Fatalf("PatchScalarConfig error: %v", err)
		}
		if epoch <= 1 {
			t.Fatalf("expected monotonic epoch > 1, got %d", epoch)
		}
	}

	close(stop)
	wg.Wait()
}

func TestControlConfigMaxWaitingAdmissionEnforcement(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	ctl := NewAdmissionController(AdmissionPolicy{
		MaxNumSeqs: 10,
		MaxWaiting: 10,
	})
	srv.SetAdmissionController(ctl)

	if ctl.MaxWaiting() != 10 {
		t.Fatalf("expected MaxWaiting 10, got %d", ctl.MaxWaiting())
	}

	// Patch MaxWaitingSeqs to 3
	newMax := uint32(3)
	_, _, err := srv.PatchScalarConfig(ScalarConfigPatch{
		MaxWaitingSeqs: &newMax,
	})
	if err != nil {
		t.Fatalf("PatchScalarConfig failed: %v", err)
	}

	if ctl.MaxWaiting() != 3 {
		t.Fatalf("expected admission controller MaxWaiting updated to 3, got %d", ctl.MaxWaiting())
	}
}

type delayedMockPlanner struct {
	delay time.Duration
}

func (p *delayedMockPlanner) Model() string {
	return "mock"
}

func (p *delayedMockPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	select {
	case <-time.After(p.delay):
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "delayed-response"}}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func TestControlConfigCompletionDeadlineEnforcement(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	srv.planner = &delayedMockPlanner{delay: 40 * time.Millisecond}
	handler := srv.Handler()

	// Initial state: CompletionDeadlineMs is 0 (disabled). Request finishes successfully.
	reqBody := `{"model":"mock","messages":[{"role":"user","content":"ping"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(reqBody)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("initial request should succeed, got %d (body: %s)", rr.Code, rr.Body.String())
	}

	// Hot-swap completion deadline down to 10ms.
	shortDeadline := uint32(10)
	_, _, err := srv.PatchScalarConfig(ScalarConfigPatch{
		CompletionDeadlineMs: &shortDeadline,
	})
	if err != nil {
		t.Fatalf("PatchScalarConfig: %v", err)
	}

	// New request takes 40ms, which exceeds 10ms deadline -> should be timed out.
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(reqBody)))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	handler.ServeHTTP(rr2, req2)

	if rr2.Code == http.StatusOK {
		t.Fatalf("expected request with exceeded deadline to fail, but got 200 OK")
	}
}

func TestControlConfigUnixDomainSocketIPC(t *testing.T) {
	srv := newTestControlGateway(t, Config{
		CompactHistoryBudget: 40000,
	})

	socketPath := filepath.Join(t.TempDir(), "control.sock")
	cs, err := srv.StartControlSocket(socketPath)
	if err != nil {
		t.Fatalf("StartControlSocket: %v", err)
	}
	defer cs.Close()

	// 1. Test HTTP over Unix socket
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("unix", socketPath)
			},
		},
		Timeout: 3 * time.Second,
	}

	resp, err := client.Get("http://unix/v1/control/config")
	if err != nil {
		t.Fatalf("HTTP over UDS GET: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("HTTP over UDS status = %d, want 200", resp.StatusCode)
	}
	var getResp ControlConfigResponse
	if err := json.NewDecoder(resp.Body).Decode(&getResp); err != nil {
		t.Fatalf("decode HTTP over UDS response: %v", err)
	}
	if getResp.Config.CompactHistoryBudget != 40000 {
		t.Errorf("got budget %d, want 40000", getResp.Config.CompactHistoryBudget)
	}

	// 2. Test raw line-delimited JSON IPC over Unix socket
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		t.Fatalf("net.Dial unix: %v", err)
	}
	defer conn.Close()

	// Send raw JSON get_config command
	ipcReq := `{"operation":"get_config"}` + "\n"
	if _, err := conn.Write([]byte(ipcReq)); err != nil {
		t.Fatalf("conn.Write get_config: %v", err)
	}

	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		t.Fatalf("read IPC response: %v", err)
	}

	var ipcResp controlIPCResponse
	if err := json.Unmarshal(line, &ipcResp); err != nil {
		t.Fatalf("unmarshal IPC response: %v", err)
	}
	if ipcResp.Status != "ok" || ipcResp.Config == nil || ipcResp.Config.CompactHistoryBudget != 40000 {
		t.Fatalf("unexpected IPC get_config response: %+v", ipcResp)
	}

	// Send raw JSON patch_config command
	newWait := uint32(999)
	patchReq, _ := json.Marshal(map[string]any{
		"operation": "patch_config",
		"patch": map[string]any{
			"max_waiting_seqs": newWait,
		},
	})
	conn2, err := net.Dial("unix", socketPath)
	if err != nil {
		t.Fatalf("net.Dial unix for patch: %v", err)
	}
	defer conn2.Close()

	if _, err := conn2.Write(append(patchReq, '\n')); err != nil {
		t.Fatalf("conn2.Write patch_config: %v", err)
	}
	line2, err := bufio.NewReader(conn2).ReadBytes('\n')
	if err != nil {
		t.Fatalf("read IPC patch response: %v", err)
	}

	var ipcPatchResp controlIPCResponse
	if err := json.Unmarshal(line2, &ipcPatchResp); err != nil {
		t.Fatalf("unmarshal IPC patch response: %v", err)
	}
	if ipcPatchResp.Status != "applied" || ipcPatchResp.Config == nil || ipcPatchResp.Config.MaxWaitingSeqs != 999 {
		t.Fatalf("unexpected IPC patch response: %+v", ipcPatchResp)
	}

	// Verify server state was mutated
	if srv.ScalarConfig().MaxWaitingSeqs != 999 {
		t.Errorf("server ScalarConfig().MaxWaitingSeqs = %d, want 999", srv.ScalarConfig().MaxWaitingSeqs)
	}
}
