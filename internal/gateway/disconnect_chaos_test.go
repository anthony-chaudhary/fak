// fak-test:runtime fast est=1s lane=default
package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func newDisconnectChaosTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for ch := byte(32); ch < 127; ch++ {
		vocab[string(ch)] = int(ch)
	}
	added := []map[string]any{
		{"id": 200, "content": "<|message_model|>", "special": true},
		{"id": 201, "content": "<|content_text|>", "special": true},
		{"id": 202, "content": "<|end_message|>", "special": true},
		{"id": 203, "content": agent.Inkling2EndOfTurnMarker, "special": true},
		{"id": 204, "content": "<|message_user|>", "special": true},
		{"id": 205, "content": "<|message_system|>", "special": true},
	}
	raw, err := json.Marshal(map[string]any{
		"model":        map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder":      map[string]any{"type": "ByteLevel"},
		"added_tokens": added,
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatalf("parse tokenizer: %v", err)
	}
	return tok
}

type disconnectChaosGen struct {
	prefixIDs   []int
	tokenID     int
	emitted     atomic.Int64
	mu          sync.Mutex
	done        chan struct{}
	finishedErr error
}

func (g *disconnectChaosGen) Generate(ctx context.Context, _ []int, maxNew int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	defer close(g.done)
	for _, id := range g.prefixIDs {
		if err := ctx.Err(); err != nil {
			g.mu.Lock()
			g.finishedErr = err
			g.mu.Unlock()
			return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, err
		}
		g.emitted.Add(1)
		if !emit(id) {
			return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load()), Stopped: true}, nil
		}
	}
	for int(g.emitted.Load()) < maxNew {
		select {
		case <-ctx.Done():
			err := ctx.Err()
			g.mu.Lock()
			g.finishedErr = err
			g.mu.Unlock()
			return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, err
		case <-time.After(5 * time.Millisecond):
		}
		g.emitted.Add(1)
		if !emit(g.tokenID) {
			return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load()), Stopped: true}, nil
		}
	}
	return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, nil
}

// TestDisconnectChaosAbruptSSECloseDrainsInflight verifies Gate 3 (Mid-Stream
// Disconnect / Cancellation Chaos): an abrupt client socket close on an active
// SSE stream without an explicit client-side context cancel terminates the
// decode loop promptly and drains fak_gateway_inflight_requests to 0 within <2s.
func TestDisconnectChaosAbruptSSECloseDrainsInflight(t *testing.T) {
	tok := newDisconnectChaosTokenizer(t)
	prefixIDs, err := tok.Encode("<|message_model|><|content_text|>")
	if err != nil {
		t.Fatalf("encode prefix: %v", err)
	}
	gen := &disconnectChaosGen{
		prefixIDs: prefixIDs,
		tokenID:   65, // "a"
		done:      make(chan struct{}),
	}
	planner, err := agent.NewInkling2LoopPlanner(gen, tok, "inkling2-mini", model.Config{ModelType: "inkling2_loop"}, 8192)
	if err != nil {
		t.Fatalf("NewInkling2LoopPlanner: %v", err)
	}

	srv := newTestServerWithConfig(t, Config{
		EngineID:          "test",
		NativeChatPlanner: planner,
		Model:             "inkling2-mini",
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"inkling2-mini","stream":true,"max_tokens":4096,"messages":[{"role":"user","content":"stream forever"}]}`
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	deltas := 0
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var frame struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			continue
		}
		for _, c := range frame.Choices {
			if c.Delta.Content != "" {
				deltas++
			}
		}
		if deltas >= 10 {
			break
		}
	}
	if deltas < 10 {
		resp.Body.Close()
		t.Fatalf("received only %d deltas before stream ended (scan err=%v)", deltas, scanner.Err())
	}

	emittedAtDisconnect := gen.emitted.Load()
	_ = resp.Body.Close()
	client.CloseIdleConnections()

	select {
	case <-gen.done:
	case <-time.After(2 * time.Second):
		t.Fatal("engine decode loop did not terminate within 2s of abrupt socket close")
	}

	gen.mu.Lock()
	genErr := gen.finishedErr
	gen.mu.Unlock()
	if !errors.Is(genErr, context.Canceled) {
		t.Fatalf("generator error = %v, want context.Canceled", genErr)
	}
	if extra := gen.emitted.Load() - emittedAtDisconnect; extra > 5 {
		t.Fatalf("decode loop emitted %d extra tokens after disconnect, want <= 5", extra)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		metrics := srv.renderMetrics()
		inflightZero := strings.Contains(metrics, "fak_gateway_inflight_requests 0\n")
		maxAgeZero := strings.Contains(metrics, "fak_gateway_inflight_max_age_seconds 0\n")
		if inflightZero && maxAgeZero {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("fak_gateway_inflight metrics did not drain to 0 within 2s:\n%s", metrics)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
