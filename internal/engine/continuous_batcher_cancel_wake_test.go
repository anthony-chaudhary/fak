package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=10ms lane=default
func TestContinuousBatcherCancellationWakesBeforeReclamation(t *testing.T) {
	t.Parallel()
	for _, parentCancel := range []bool{false, true} {
		name := "handle"
		if parentCancel {
			name = "parent_context"
		}
		t.Run(name, func(t *testing.T) {
			cfg := DefaultBatchingEngineConfig()
			cfg.MaxSteps = 1
			cfg.Batcher.MaxSlots = 1
			cfg.Batcher.MaxKVCacheBytes = 1
			cfg.Batcher.KVBytesPerToken = 1
			cb, err := NewContinuousBatcher(cfg.Batcher)
			if err != nil {
				t.Fatal(err)
			}
			loopCtx, stopLoop := context.WithCancel(context.Background())
			// Hold the real scheduler's start, rather than guess when a goroutine
			// parked. Admit/Cancel, the wake channel, and run/pump are production
			// code; no fake producer or copied cleanup logic participates.
			e := &BatchingEngine{
				cfg: cfg, cb: cb, requests: make(map[string]*batcherRequest),
				ctx: loopCtx, cancel: stopLoop, done: make(chan struct{}),
				wake: make(chan struct{}, 1),
			}
			started := false
			defer func() {
				if started {
					_ = e.Close()
				} else {
					stopLoop()
					_ = cb.Close()
				}
			}()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			req, err := e.Admit(ctx, &abi.ToolCall{
				Tool: "test", Args: abi.Ref{Kind: abi.RefInline,
					Inline: []byte(`{"session_id":"queued","target_tokens":2}`)},
			})
			if err != nil {
				t.Fatal(err)
			}
			if cb.WaitingQueueLength() != 1 || cb.ActiveSlotCount() != 0 {
				t.Fatal("request must be queued without a model producer")
			}
			select {
			case <-e.wake: // Consume the separate admission notification.
			default:
				t.Fatal("Admit did not notify the scheduler")
			}
			if parentCancel {
				cancel()
			} else {
				req.Cancel()
			}
			select {
			case <-e.wake:
			case <-time.After(time.Second):
				t.Fatal("cancellation did not wake the parked scheduler")
			}
			// Notification is not completion: only the scheduler may remove the
			// queued request or publish its terminal state at a step boundary.
			if cb.WaitingQueueLength() != 1 {
				t.Fatal("cancellation callback reclaimed scheduler-owned work")
			}
			select {
			case <-req.Tokens():
				t.Fatal("cancellation published before the scheduler advanced")
			default:
			}
			started = true
			go e.run()
			select {
			case _, open := <-req.Tokens():
				if open {
					t.Fatal("canceled queued request produced a token")
				}
			case <-time.After(time.Second):
				t.Fatal("scheduler did not reclaim the canceled request")
			}
			if result, err := req.Result(); result != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("Result = %v, %v; want nil, context.Canceled", result, err)
			}
			if cb.WaitingQueueLength() != 0 {
				t.Fatal("terminal request still owns queued work")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if e.outstanding() != 0 {
				t.Fatal("scheduler retained terminal request metadata")
			}
		})
	}
}
