package macbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLoadDriver_OptionsDefaults(t *testing.T) {
	opts := DefaultLoadDriverOptions()
	if opts.Concurrency != 24 {
		t.Fatalf("expected default concurrency 24, got %d", opts.Concurrency)
	}
	if opts.TargetTokens != 300 {
		t.Fatalf("expected default target tokens 300, got %d", opts.TargetTokens)
	}
	if opts.SharedPrefixTokens != 4096 {
		t.Fatalf("expected default shared prefix tokens 4096, got %d", opts.SharedPrefixTokens)
	}
	if opts.TurnDeltaTokens != 128 {
		t.Fatalf("expected default turn delta tokens 128, got %d", opts.TurnDeltaTokens)
	}

	driver := NewLoadDriver(opts)
	if driver == nil {
		t.Fatal("expected NewLoadDriver to return non-nil")
	}
}

func TestLoadDriver_ConcurrentStreams24(t *testing.T) {
	var activeConnections int32
	var peakActiveConnections int32
	var totalRequests int32

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		cur := atomic.AddInt32(&activeConnections, 1)
		for {
			peak := atomic.LoadInt32(&peakActiveConnections)
			if cur <= peak || atomic.CompareAndSwapInt32(&peakActiveConnections, peak, cur) {
				break
			}
		}
		defer atomic.AddInt32(&activeConnections, -1)
		atomic.AddInt32(&totalRequests, 1)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		// Emit 8 tokens with micro-delays to produce measurable ITL
		for i := 0; i < 8; i++ {
			chunk := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"tok%d \"}}]}\n\n", i)
			_, _ = w.Write([]byte(chunk))
			flusher.Flush()
			time.Sleep(100 * time.Microsecond)
		}
		usageChunk := "data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":4224,\"completion_tokens\":8,\"total_tokens\":4232}}\n\n"
		_, _ = w.Write([]byte(usageChunk))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer ts.Close()

	opts := LoadDriverOptions{
		Gateway:            ts.URL,
		Model:              "qwen3.8-27b",
		Key:                "test-key",
		Concurrency:        24,
		Duration:           0,
		TargetTokens:       8,
		SharedPrefixTokens: 4096,
		TurnDeltaTokens:    128,
		Horizon:            1,
		HTTPClient:         ts.Client(),
		Now: func() time.Time {
			return time.Date(2026, 9, 8, 17, 0, 0, 0, time.UTC)
		},
	}

	driver := NewLoadDriver(opts)
	result, err := driver.Run(context.Background())
	if err != nil {
		t.Fatalf("driver.Run failed: %v", err)
	}

	if result.Schema != AgenticMTPSchema {
		t.Fatalf("expected schema %q, got %q", AgenticMTPSchema, result.Schema)
	}
	if result.Summary.Concurrency != 24 {
		t.Fatalf("expected summary concurrency 24, got %d", result.Summary.Concurrency)
	}
	if len(result.Streams) != 24 {
		t.Fatalf("expected 24 streams, got %d", len(result.Streams))
	}
	if len(result.RawSamples.Streams) != 24 {
		t.Fatalf("expected 24 raw stream samples, got %d", len(result.RawSamples.Streams))
	}

	// Verify peak concurrency proved concurrent in-flight streams
	peak := atomic.LoadInt32(&peakActiveConnections)
	if peak < 2 {
		t.Fatalf("expected concurrent connections, peak was %d", peak)
	}

	// Verify TTFT and ITL metrics are populated and sensible
	if result.Metrics.Prefill.TTFTMS.P50 <= 0 {
		t.Fatalf("expected positive P50 TTFT, got %f", result.Metrics.Prefill.TTFTMS.P50)
	}
	if result.Metrics.Prefill.TTFTMS.P95 < result.Metrics.Prefill.TTFTMS.P50 {
		t.Fatalf("expected P95 TTFT >= P50 TTFT, got %f < %f",
			result.Metrics.Prefill.TTFTMS.P95, result.Metrics.Prefill.TTFTMS.P50)
	}
	if result.Summary.P50ITLMS <= 0 {
		t.Fatalf("expected positive P50 ITL, got %f", result.Summary.P50ITLMS)
	}
	if result.Summary.P95ITLMS < result.Summary.P50ITLMS {
		t.Fatalf("expected P95 ITL >= P50 ITL, got %f < %f",
			result.Summary.P95ITLMS, result.Summary.P50ITLMS)
	}

	// Verify aggregate decode tok/s
	if result.Summary.AggregateDecodeTokS <= 0 {
		t.Fatalf("expected positive aggregate decode tok/s, got %f", result.Summary.AggregateDecodeTokS)
	}
	if result.Summary.PerAgentDecodeTokS <= 0 {
		t.Fatalf("expected positive per-agent decode tok/s, got %f", result.Summary.PerAgentDecodeTokS)
	}

	// Validate against fail-closed AgenticMTPPacket validator
	if err := ValidateAgenticMTPPacket(result.AgenticMTPPacket); err != nil {
		t.Fatalf("ValidateAgenticMTPPacket failed on load driver result: %v", err)
	}
}

func TestLoadDriver_DurationLoop(t *testing.T) {
	var totalRequests int32
	var mu sync.Mutex
	agentsSeen := make(map[int]int)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&totalRequests, 1)
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		content := ""
		if msgs, ok := body["messages"].([]any); ok && len(msgs) > 0 {
			if m, ok := msgs[0].(map[string]any); ok {
				content, _ = m["content"].(string)
			}
		}
		var agentID int
		if strings.HasPrefix(content, "Agent ") {
			_, _ = fmt.Sscanf(content, "Agent %d", &agentID)
			mu.Lock()
			agentsSeen[agentID]++
			mu.Unlock()
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":1024,\"completion_tokens\":1,\"total_tokens\":1025}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer ts.Close()

	opts := LoadDriverOptions{
		Gateway:            ts.URL,
		Model:              "qwen3.8-27b",
		Concurrency:        24,
		Duration:           80 * time.Millisecond,
		TargetTokens:       4,
		SharedPrefixTokens: 1024,
		TurnDeltaTokens:    32,
		HTTPClient:         ts.Client(),
	}

	driver := NewLoadDriver(opts)
	result, err := driver.Run(context.Background())
	if err != nil {
		t.Fatalf("driver.Run failed: %v", err)
	}

	reqCount := atomic.LoadInt32(&totalRequests)
	if reqCount < 24 {
		t.Fatalf("expected at least 24 requests in duration loop, got %d", reqCount)
	}
	if len(result.Streams) != 24 {
		t.Fatalf("expected 24 streams, got %d", len(result.Streams))
	}
}

func TestLoadDriver_EvidenceFilesAndOutDir(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; i < 4; i++ {
			_, _ = w.Write([]byte(fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"t%d\"}}]}\n\n", i)))
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":4224,\"completion_tokens\":4,\"total_tokens\":4228}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer ts.Close()

	tempDir := t.TempDir()
	opts := LoadDriverOptions{
		Gateway:            ts.URL,
		Model:              "qwen3.8-27b",
		Concurrency:        24,
		TargetTokens:       4,
		SharedPrefixTokens: 4096,
		TurnDeltaTokens:    128,
		Horizon:            1,
		OutDir:             tempDir,
		HTTPClient:         ts.Client(),
	}

	driver := NewLoadDriver(opts)
	result, err := driver.Run(context.Background())
	if err != nil {
		t.Fatalf("driver.Run failed: %v", err)
	}

	packetPath := filepath.Join(tempDir, "packet.json")
	if _, err := os.Stat(packetPath); err != nil {
		t.Fatalf("packet.json missing in %s: %v", tempDir, err)
	}
	rawPath := filepath.Join(tempDir, result.RawResult.Path)
	if _, err := os.Stat(rawPath); err != nil {
		t.Fatalf("raw result missing at %s: %v", rawPath, err)
	}
	qualityPath := filepath.Join(tempDir, result.Quality.ResultPath)
	if _, err := os.Stat(qualityPath); err != nil {
		t.Fatalf("quality result missing at %s: %v", qualityPath, err)
	}

	// Validate evidence files integrity using strict validator
	if err := ValidateAgenticMTPEvidence(result.AgenticMTPPacket, packetPath); err != nil {
		t.Fatalf("ValidateAgenticMTPEvidence failed: %v", err)
	}
}

func TestLoadDriver_ErrorHandlingOnConnectionFailure(t *testing.T) {
	// Point driver at an unreachable port
	opts := LoadDriverOptions{
		Gateway:      "http://127.0.0.1:59999",
		Model:        "qwen3.8-27b",
		Concurrency:  24,
		TargetTokens: 4,
		Horizon:      1,
		HTTPClient:   &http.Client{Timeout: 100 * time.Millisecond},
	}

	driver := NewLoadDriver(opts)
	_, err := driver.Run(context.Background())
	if err == nil {
		t.Fatal("expected error connecting to dead port, got nil")
	}
	if !strings.Contains(err.Error(), "load driver: all 24 streams failed") {
		t.Fatalf("expected 'all 24 streams failed' error, got: %v", err)
	}
}

func TestLoadDriver_JSONSerialization(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":4224,\"completion_tokens\":1,\"total_tokens\":4225}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer ts.Close()

	opts := LoadDriverOptions{
		Gateway:            ts.URL,
		Model:              "qwen3.8-27b",
		Concurrency:        24,
		TargetTokens:       1,
		SharedPrefixTokens: 4096,
		TurnDeltaTokens:    128,
		Horizon:            1,
		HTTPClient:         ts.Client(),
	}

	result, err := RunLoadDriver(context.Background(), opts)
	if err != nil {
		t.Fatalf("RunLoadDriver failed: %v", err)
	}

	rawJSON, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal result failed: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(rawJSON, &parsed); err != nil {
		t.Fatalf("unmarshal json failed: %v", err)
	}

	if parsed["schema"] != AgenticMTPSchema {
		t.Fatalf("expected top-level schema %q, got %v", AgenticMTPSchema, parsed["schema"])
	}

	summary, ok := parsed["summary"].(map[string]any)
	if !ok {
		t.Fatalf("missing summary in JSON: %s", rawJSON)
	}
	if concurrency, ok := summary["concurrency"].(float64); !ok || int(concurrency) != 24 {
		t.Fatalf("expected summary concurrency 24, got %v", summary["concurrency"])
	}

	metrics, ok := parsed["metrics"].(map[string]any)
	if !ok {
		t.Fatalf("missing metrics in JSON")
	}
	prefill, ok := metrics["prefill"].(map[string]any)
	if !ok {
		t.Fatalf("missing prefill in metrics")
	}
	ttft, ok := prefill["ttft_ms"].(map[string]any)
	if !ok || ttft["p50"].(float64) <= 0 {
		t.Fatalf("invalid ttft_ms in prefill metrics: %v", prefill)
	}
}
