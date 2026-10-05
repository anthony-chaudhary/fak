package agent

// concurrent_upstream_test.go — the gateway's proxy planner must not collapse
// CONCURRENT turns onto one upstream slot.
//
// [HW-WITNESSED] 2026-10-05T16:45Z on strix1, Qwen3.8-27B-UD-Q2_K_XL: four
// concurrent 128-token requests straight at the upstream llama.cpp reference
// server (:8091) finished at 41.65 tok/s aggregate with all four slots inside
// 77 ms of each other (true continuous batching). The same four requests through
// the fak gateway front door (:8080, `fak serve --engine mock --provider openai
// --base-url http://127.0.0.1:8091/v1 --model Qwen3.8-27B-UD-Q2_K_XL`) came back
// at 13.89 tok/s aggregate — BELOW the 18.23 tok/s single-request rate — with a
// per-request staircase 7.37 -> 22.86 -> 29.85 -> 36.86 s: one full serial
// quantum added per request, i.e. one upstream slot processed them one after
// another.
//
// The upstream here reproduces llama.cpp's slot contract — one KV cache per slot,
// one request at a time per slot, /props advertising total_slots — and holds every
// request at a four-way barrier of simultaneously in-flight work. A planner that
// stamps the SAME id_slot onto every concurrent turn cannot clear that barrier, so
// the regression fails in seconds instead of on the appliance.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// TestGatewayProxyConcurrentUpstreamIsNotSerialized proves four concurrent proxy
// turns reach the upstream four-at-a-time instead of queueing behind one slot.
//
// fak-test:runtime fast est=50ms
func TestGatewayProxyConcurrentUpstreamIsNotSerialized(t *testing.T) {
	const (
		concurrency = 4
		totalSlots  = 4
	)

	var (
		mu          sync.Mutex
		armed       bool
		inFlight    int
		maxInFlight int
		sawHint     bool
		handlerErrs []error
		arrived     = make(chan struct{}, 4*concurrency)
	)
	// noteHandlerErr records a handler failure under mu: the handler goroutine must
	// never call t.*, because a failure path can outlive the test body.
	noteHandlerErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		handlerErrs = append(handlerErrs, err)
	}
	// slotLocks mirrors llama.cpp's slot semantics: a pinned id_slot admits ONE
	// request at a time (that slot's single KV cache); an absent hint runs free.
	var slotMu sync.Mutex
	slotLocks := map[int]*sync.Mutex{}
	slotGate := func(slot int) *sync.Mutex {
		slotMu.Lock()
		defer slotMu.Unlock()
		g, ok := slotLocks[slot]
		if !ok {
			g = &sync.Mutex{}
			slotLocks[slot] = g
		}
		return g
	}

	// release is closed on cleanup so a broken barrier can never park an upstream
	// handler (or the package) forever.
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/props" {
			// llama-server's slot advertisement; probeLlamaTotalSlots reads this.
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"total_slots": totalSlots})
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			noteHandlerErr(fmt.Errorf("read upstream body: %w", err))
			return
		}
		var wire struct {
			IDSlot      *int  `json:"id_slot"`
			CachePrompt *bool `json:"cache_prompt"`
		}
		_ = json.Unmarshal(body, &wire)
		if wire.IDSlot != nil {
			gate := slotGate(*wire.IDSlot)
			gate.Lock()
			defer gate.Unlock()
		}
		mu.Lock()
		sawHint = sawHint || wire.CachePrompt != nil
		inFlight++
		if inFlight > maxInFlight {
			maxInFlight = inFlight
		}
		mu.Unlock()
		select {
		case arrived <- struct{}{}:
		default:
		}
		// Hold this request at the barrier until every concurrent request is in
		// flight at once. Before the barrier is armed (slot discovery warm-up) the
		// upstream answers immediately.
		for {
			mu.Lock()
			open := !armed || maxInFlight >= concurrency
			mu.Unlock()
			if open {
				break
			}
			select {
			case <-arrived:
			case <-release:
			case <-time.After(10 * time.Millisecond):
			case <-r.Context().Done():
				mu.Lock()
				open = !armed || maxInFlight >= concurrency
				mu.Unlock()
				if open {
					break
				}
				return // client gone: stop counting this slot as occupied
			}
		}
		mu.Lock()
		inFlight--
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-1",
			"object":  "chat.completion",
			"model":   "test-model",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
		}); err != nil {
			noteHandlerErr(fmt.Errorf("encode upstream response: %w", err))
		}
	}))
	t.Cleanup(upstream.Close)

	p, err := NewProviderHTTPPlanner("openai", upstream.URL+"/v1", "test-model", "")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	// `fak serve` defaults this ON (--llama-slot-affinity, cmd/fak/serve.go), so the
	// drop-in front door carries the hint without the operator asking for it.
	p.LlamaSlotAffinity = true

	messages := []Message{{Role: RoleUser, Content: "concurrent turn"}}
	turn := func(ctx context.Context) error {
		_, err := p.Complete(ctx, messages, nil)
		return err
	}

	// Slot discovery is a BACKGROUND probe (llama_slot_affinity.go:maybeProbe), so
	// the first turns carry no hint. Drive single turns until the hint is live, then
	// arm the barrier and measure the concurrent burst.
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	deadline := time.Now().Add(15 * time.Second)
	for {
		one, oneCancel := context.WithTimeout(ctx, 5*time.Second)
		err := turn(one)
		oneCancel()
		mu.Lock()
		hinted := sawHint
		mu.Unlock()
		if hinted {
			if err != nil {
				t.Fatalf("warm-up turn with the slot hint live failed: %v", err)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("upstream never saw the llama-slot hint (cache_prompt); slot discovery never completed")
		}
	}
	mu.Lock()
	armed = true
	maxInFlight = 0
	mu.Unlock()

	// burstCtx is separate so a FAILED barrier cancels the still-queued turns instead
	// of draining them one serial quantum at a time.
	burstCtx, burstCancel := context.WithCancel(ctx)
	defer burstCancel()
	var wg sync.WaitGroup
	errs := make([]error, concurrency)
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = turn(burstCtx)
		}(i)
	}

	barrierDeadline := time.After(10 * time.Second)
	for {
		mu.Lock()
		reached := maxInFlight >= concurrency
		peak := maxInFlight
		mu.Unlock()
		if reached {
			break
		}
		select {
		case <-barrierDeadline:
			burstCancel()
			unblock()
			wg.Wait()
			t.Fatalf("proxy planner serialized concurrent upstream requests: the upstream processed at most %d of %d at once under llama.cpp slot semantics (one request per slot), so the four-way barrier was never reached", peak, concurrency)
		case <-time.After(10 * time.Millisecond):
		}
	}
	unblock()
	wg.Wait()
	mu.Lock()
	handlerErr := errors.Join(handlerErrs...)
	mu.Unlock()
	if handlerErr != nil {
		t.Fatalf("upstream handler: %v", handlerErr)
	}
	for i, err := range errs {
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
}
