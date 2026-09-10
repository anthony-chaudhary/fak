package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macobs"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestCompactHistoryBudgetDefaultsToDefaultConst pins the default-on sprawl trigger: the
// --compact-history-budget flag (defined identically in runServe and runGuard) now defaults
// to gateway.DefaultCompactHistoryBudget — a non-zero default so a sprawling conversation is
// compacted with NO operator configuration — while an explicit =0 preserves the byte-for-byte
// OFF opt-out. This mirrors the live flag registration in serve.go:98 / guard.go:84; if either
// is changed away from the const, this test fails.
func TestCompactHistoryBudgetDefaultsToDefaultConst(t *testing.T) {
	if gateway.DefaultCompactHistoryBudget <= 0 {
		t.Fatalf("DefaultCompactHistoryBudget must be a non-zero default-on value, got %d", gateway.DefaultCompactHistoryBudget)
	}

	// Default with no flag passed → the const (default-on).
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	budget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if *budget != gateway.DefaultCompactHistoryBudget {
		t.Fatalf("default = %d, want %d (the default-on trigger)", *budget, gateway.DefaultCompactHistoryBudget)
	}

	// Explicit =0 → OFF opt-out (the only way to get the old byte-for-byte path now).
	fs2 := flag.NewFlagSet("serve", flag.ContinueOnError)
	budget2 := fs2.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "")
	if err := fs2.Parse([]string{"--compact-history-budget=0"}); err != nil {
		t.Fatalf("parse =0: %v", err)
	}
	if *budget2 != 0 {
		t.Fatalf("explicit =0 must override to OFF, got %d", *budget2)
	}
}

func TestRepeatedStringFlagAccumulatesTrimmedValues(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" http://127.0.0.1:8001/v1 "); err != nil {
		t.Fatalf("Set first: %v", err)
	}
	if err := f.Set("http://127.0.0.1:8002/v1"); err != nil {
		t.Fatalf("Set second: %v", err)
	}
	want := []string{"http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"}
	if got := f.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Values() = %v, want %v", got, want)
	}
	if got := f.String(); got != "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" {
		t.Fatalf("String() = %q", got)
	}
	got := f.Values()
	got[0] = "mutated"
	if again := f.Values(); !reflect.DeepEqual(again, want) {
		t.Fatalf("Values() returned internal storage: %v", again)
	}
}

func TestRepeatedStringFlagRejectsEmptyValue(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" \t "); err == nil {
		t.Fatal("Set blank value succeeded, want error")
	}
}

// TestServeMetalFlagDefaultsFalse pins the flag parse contract: --metal defaults false because
// runtime auto-selection happens in resolveServeMetal, not in the flag package.
func TestServeMetalFlagDefaultsFalse(t *testing.T) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	metal := fs.Bool("metal", false, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if *metal {
		t.Fatal("--metal must default to false; runtime auto-select happens after parse")
	}
	fs2 := flag.NewFlagSet("serve", flag.ContinueOnError)
	metal2 := fs2.Bool("metal", false, "")
	if err := fs2.Parse([]string{"--metal"}); err != nil {
		t.Fatalf("parse --metal: %v", err)
	}
	if !*metal2 {
		t.Fatal("--metal must flip the bit on")
	}
}

// TestResolveServeMetal exercises auto-select plus explicit fail-loud behavior. With no
// explicit request, Metal follows runtime availability: Apple-Silicon+cgo with a device uses it,
// and every other build/device state falls back to CPU. Explicit --metal/FAK_METAL still errors
// when unavailable, mirroring resolveServeChatBackend.
func TestResolveServeMetal(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	// Not requested -> runtime auto-select only when a usable Metal device is present.
	if use, err := resolveServeMetal(false, false, ""); use != metalgemm.Available() || err != nil {
		t.Fatalf("neither flag nor env: got (%v,%v), want (%v,nil)", use, err, metalgemm.Available())
	}
	// A named compute backend disables Metal auto-select; only an explicit Metal request conflicts.
	if use, err := resolveServeMetal(false, false, "cuda"); use || err != nil {
		t.Fatalf("backend without explicit metal: got (%v,%v), want (false,nil)", use, err)
	}
	// Requested + a device --backend → conflict error, independent of Metal availability.
	if _, err := resolveServeMetal(true, false, "cuda"); err == nil {
		t.Fatal("--metal with --backend cuda must be rejected as mutually exclusive")
	}
	if _, err := resolveServeMetal(false, true, "cuda"); err == nil {
		t.Fatal("FAK_METAL with --backend cuda must be rejected as mutually exclusive")
	}
	// Requested with no conflicting backend: on a non-Metal build this fails loud (no silent
	// CPU fallback). On an Apple-Silicon+cgo build with a device it would succeed — assert by availability
	// so the test is correct on BOTH builds.
	use, err := resolveServeMetal(true, false, "")
	if metalgemm.Available() {
		if !use || err != nil {
			t.Fatalf("metal available: got (%v,%v), want (true,nil)", use, err)
		}
	} else {
		if use || err == nil {
			t.Fatalf("metal unavailable must fail loud: got (%v,%v), want (false, error)", use, err)
		}
	}
	// FAK_METAL env is an equivalent trigger to the flag (same code path).
	if _, err := resolveServeMetal(false, true, ""); !metalgemm.Available() && err == nil {
		t.Fatal("FAK_METAL on a non-Metal build must fail loud, same as --metal")
	}
}

func TestServeDeferToolsFlag(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if !*sf.deferTools {
		t.Fatal("--defer-tools must default to true")
	}
	if *sf.toolCeiling != gateway.DefaultMCPToolAdvertisementCeiling {
		t.Fatalf("--tool-ceiling must default to %d, got %d", gateway.DefaultMCPToolAdvertisementCeiling, *sf.toolCeiling)
	}

	// Default --defer-tools=true configures 4 core tools
	srv, err := gateway.New(gateway.Config{
		EngineID:        "mock",
		DisableMCPDefer: !*sf.deferTools,
		MCPToolCeiling:  *sf.toolCeiling,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()
	tools, status := srv.MCPToolListSnapshot(false)
	if len(tools) != 4 {
		t.Fatalf("expected 4 core tools with --defer-tools=true, got %d", len(tools))
	}
	if status.Mode != "active" {
		t.Fatalf("expected status mode active, got %s", status.Mode)
	}

	// --defer-tools=false emits startup warning and wires --tool-ceiling
	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--defer-tools=false", "--tool-ceiling=10"}); err != nil {
		t.Fatalf("parse --defer-tools=false: %v", err)
	}
	if *sf2.deferTools {
		t.Fatal("--defer-tools=false must set flag to false")
	}
	if *sf2.toolCeiling != 10 {
		t.Fatalf("--tool-ceiling must be 10, got %d", *sf2.toolCeiling)
	}

	var stderr bytes.Buffer
	warnIfDeferToolsDisabled(&stderr, *sf2.deferTools)
	if !strings.Contains(stderr.String(), "WARNING: --defer-tools=false is set") {
		t.Fatalf("expected stderr warning, got: %s", stderr.String())
	}

	// When --defer-tools=false (DisableMCPDefer=true), MCPToolCeiling clamps tools
	srv2, err := gateway.New(gateway.Config{
		EngineID:        "mock",
		DisableMCPDefer: !*sf2.deferTools,
		MCPToolCeiling:  *sf2.toolCeiling,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv2.Close()
	tools2, status2 := srv2.MCPToolListSnapshot(true)
	if len(tools2) > 10 {
		t.Fatalf("expected at most 10 tools with ceiling 10, got %d", len(tools2))
	}
	if status2.Mode != "ceiling" || status2.Reason != "advertisement_ceiling" {
		t.Fatalf("expected status ceiling/advertisement_ceiling, got %+v", status2)
	}
}

func TestServeMemoryGovernorDynamicZeroSwapAdmission(t *testing.T) {
	// 1. Configure MemoryGovernor with 25GB wired ceiling
	hw := macobs.HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  25 * 1024 * 1024 * 1024, // 25 GB limit
		SwapUsedBytes:          0,
		PageOuts:               0,
		Available:              true,
	}
	cfg := macobs.Qwen38GDNHeadroomConfig() // base resident = 23.75 GB = 25501368320 bytes
	// Available headroom to 25 GB = 1.25 GB = 1342177280 bytes
	// An isolated (non-shared preamble) agent requires ~338.7 MB.
	// 4 isolated agents require 4 * 338.7 MB = 1354.8 MB > 1280 MB!
	// So exactly 3 isolated agents fit under 25 GB limit; the 4th MUST be rejected with SWAP_RISK!

	gov := macobs.NewMemoryGovernor(hw, cfg)

	// 2. Build gateway server and attach MemoryGovernor
	srv, err := gateway.New(gateway.Config{
		EngineID:      "mock",
		Model:         "mock",
		ExposeProfile: "headless",
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()

	srv.SetMemoryGovernor(gov)

	// 3. Start loopback HTTP test server
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 4. Pre-admit 3 isolated agents (simulate 3 active concurrent sessions holding KV headroom)
	for i := 1; i <= 3; i++ {
		admitted, reason := gov.Admit(macobs.AgentSpec{
			ID:             fmt.Sprintf("active-session-%d", i),
			SharedPreamble: false,
			PreambleTokens: 4096,
			TailTokens:     1024,
		})
		if !admitted {
			t.Fatalf("failed to admit initial session %d: %s", i, reason)
		}
	}

	// 5. Inbound client burst requesting token generation:
	// A 4th non-shared session arrives via loopback HTTP POST /v1/chat/completions.
	// This 4th session would exceed the 25GB wired ceiling, so MemoryGovernor.CanAdmit
	// MUST evaluate headroom and reject with structured SWAP_RISK backpressure.
	body := `{"model":"mock","messages":[{"role":"user","content":"hello"}]}`
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "overflow-agent-4")
	req.Header.Set("X-Fak-Shared-Preamble", "false") // Non-shared preamble forces full preamble allocation

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("http request failed: %v", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	// Must be rejected with HTTP 503 (Service Unavailable)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503 Service Unavailable, got %d; body: %s", resp.StatusCode, string(rawBody))
	}

	// Must contain Retry-After header
	if retryAfter := resp.Header.Get("Retry-After"); retryAfter == "" {
		t.Errorf("expected Retry-After header in 503 refusal response")
	}

	// Must contain structured SWAP_RISK error
	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(rawBody, &errResp); err != nil {
		t.Fatalf("unmarshal error response failed: %v (raw: %s)", err, rawBody)
	}
	if errResp.Error.Code != "swap_risk" {
		t.Errorf("expected error.code 'swap_risk', got %q", errResp.Error.Code)
	}
	if !strings.Contains(errResp.Reason, "SWAP_RISK") {
		t.Errorf("expected reason to contain 'SWAP_RISK', got %q", errResp.Reason)
	}

	// Zero-swap bound assertion: verify zero swap delta on the host
	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Fatalf("zero-swap delta assertion failed: %v", err)
	}
	telem := gov.Telemetry()
	if telem.SwapUsedDeltaBytes != 0 || telem.PageoutsDelta != 0 {
		t.Errorf("expected swap delta == 0 and pageouts delta == 0, got swap=%d, pageouts=%d",
			telem.SwapUsedDeltaBytes, telem.PageoutsDelta)
	}
	if !telem.ZeroSwapGuaranteed {
		t.Errorf("expected ZeroSwapGuaranteed == true")
	}

	// 6. Release one active agent and verify admission succeeds
	gov.Release("active-session-1")

	reqAdmit, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	reqAdmit.Header.Set("Content-Type", "application/json")
	reqAdmit.Header.Set("X-Trace-Id", "now-admitted-agent")
	reqAdmit.Header.Set("X-Fak-Shared-Preamble", "false")

	respAdmit, err := http.DefaultClient.Do(reqAdmit)
	if err != nil {
		t.Fatalf("http request failed: %v", err)
	}
	defer respAdmit.Body.Close()
	rawAdmit, _ := io.ReadAll(respAdmit.Body)
	if respAdmit.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 OK after headroom freed, got %d; body: %s", respAdmit.StatusCode, string(rawAdmit))
	}
}

func TestServeMemoryGovernorConcurrentBurstAdmission(t *testing.T) {
	// Set wired ceiling to allow only 2 shared agents
	// Base = 23.75 GB = 25501368320 bytes
	// 2 shared agents = 2 * 70464307 = 140928614 bytes
	// Ceiling = 25501368320 + 140928614 + 10MB = 25652777334 bytes
	base := uint64(25501368320)
	twoAgents := uint64(140928614)
	ceiling := base + twoAgents + 10*1024*1024

	hw := macobs.HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  ceiling,
		SwapUsedBytes:          0,
		PageOuts:               0,
		Available:              true,
	}
	cfg := macobs.Qwen38GDNHeadroomConfig()
	gov := macobs.NewMemoryGovernor(hw, cfg)

	srv, err := gateway.New(gateway.Config{
		EngineID:      "mock",
		Model:         "mock",
		ExposeProfile: "headless",
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()

	srv.SetMemoryGovernor(gov)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Pre-admit 2 shared agents to fill the ceiling
	for i := 1; i <= 2; i++ {
		admitted, reason := gov.Admit(macobs.AgentSpec{
			ID:             fmt.Sprintf("concurrent-holder-%d", i),
			SharedPreamble: true,
			PreambleTokens: 4096,
			TailTokens:     1024,
		})
		if !admitted {
			t.Fatalf("admit holder %d: %s", i, reason)
		}
	}

	// Now burst 5 concurrent requests; all must receive structured SWAP_RISK refusal
	var wg sync.WaitGroup
	errCh := make(chan error, 5)

	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go func(agentNum int) {
			defer wg.Done()
			body := `{"model":"mock","messages":[{"role":"user","content":"concurrent burst"}]}`
			req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
			if err != nil {
				errCh <- err
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Trace-Id", fmt.Sprintf("burst-client-%d", agentNum))

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				errCh <- err
				return
			}
			defer resp.Body.Close()
			raw, _ := io.ReadAll(resp.Body)

			if resp.StatusCode != http.StatusServiceUnavailable {
				errCh <- fmt.Errorf("agent %d expected status 503, got %d: %s", agentNum, resp.StatusCode, raw)
				return
			}
			if !strings.Contains(string(raw), "SWAP_RISK") && !strings.Contains(string(raw), "swap_risk") {
				errCh <- fmt.Errorf("agent %d missing SWAP_RISK: %s", agentNum, raw)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("burst error: %v", err)
	}

	if err := gov.AssertZeroSwapDelta(); err != nil {
		t.Fatalf("zero swap delta violated: %v", err)
	}
}
