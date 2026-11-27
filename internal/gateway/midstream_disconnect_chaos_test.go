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

type chaosDisconnectGen struct {
	prefixIDs   []int
	tokenID     int
	emitted     atomic.Int64
	mu          sync.Mutex
	done        chan struct{}
	finishedErr error
}

func (g *chaosDisconnectGen) Generate(ctx context.Context, _ []int, maxNew int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
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
			if err := ctx.Err(); err != nil {
				g.mu.Lock()
				g.finishedErr = err
				g.mu.Unlock()
				return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, err
			}
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
			if err := ctx.Err(); err != nil {
				g.mu.Lock()
				g.finishedErr = err
				g.mu.Unlock()
				return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, err
			}
			return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load()), Stopped: true}, nil
		}
	}
	return agent.Inkling2LoopGeneration{Generated: int(g.emitted.Load())}, nil
}

func newChaosInkling2Tokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := 33; c < 127; c++ {
		vocab[string(rune(c))] = c
	}
	var added []map[string]any
	for i, s := range []string{
		"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
		"<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|content_xml|>",
		"<|end_message|>", agent.Inkling2EndOfTurnMarker,
	} {
		added = append(added, map[string]any{"id": 200 + i, "content": s, "special": true})
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

func TestMidstreamDisconnectCancelsDecodeAndDrainsInflight(t *testing.T) {
	tok := newChaosInkling2Tokenizer(t)
	prefixIDs, err := tok.Encode("<|message_model|><|content_text|>")
	if err != nil {
		t.Fatalf("encode prefix: %v", err)
	}
	gen := &chaosDisconnectGen{
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

	reqCtx, cancel := context.WithCancel(t.Context())
	defer cancel()
	body := `{"model":"inkling2-mini","stream":true,"max_tokens":4096,"messages":[{"role":"user","content":"stream forever"}]}`
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
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

	// Read 20 content deltas, then abruptly close the client socket and cancel context.
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
		if deltas >= 20 {
			break
		}
	}
	if deltas < 20 {
		resp.Body.Close()
		t.Fatalf("received only %d deltas before stream ended (scan err=%v)", deltas, scanner.Err())
	}

	emittedAtDisconnect := gen.emitted.Load()
	_ = resp.Body.Close()
	cancel()
	client.CloseIdleConnections()

	select {
	case <-gen.done:
	case <-time.After(2 * time.Second):
		t.Fatal("engine decode loop did not terminate within 2s of client disconnect")
	}

	gen.mu.Lock()
	genErr := gen.finishedErr
	gen.mu.Unlock()
	if !errors.Is(genErr, context.Canceled) {
		t.Fatalf("generator error = %v, want context.Canceled", genErr)
	}
	extraTokens := gen.emitted.Load() - emittedAtDisconnect
	if extraTokens > 5 {
		t.Fatalf("decode loop emitted %d extra tokens after disconnect, want <= 5", extraTokens)
	}

	// Assert fak_gateway_inflight_requests and fak_gateway_inflight_max_age_seconds return to 0 within < 2s.
	deadline := time.Now().Add(2 * time.Second)
	for {
		metrics := srv.renderMetrics()
		inflightZero := strings.Contains(metrics, "fak_gateway_inflight_requests 0\n")
		maxAgeZero := strings.Contains(metrics, "fak_gateway_inflight_max_age_seconds 0\n")
		if inflightZero && maxAgeZero {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("inflight metrics did not return to 0 within 2s:\n%s", metrics)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
