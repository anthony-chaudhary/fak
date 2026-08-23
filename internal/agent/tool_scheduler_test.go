package agent

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"
)

func TestParallelToolCancellationDrainsStartedAndSkipsQueuedInModelOrder(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := make(chan string, 2)
	finished := make(chan string, 2)
	releaseFirst := make(chan struct{})
	releaseSecond := make(chan struct{})
	var thirdRan atomic.Bool

	blockingCall := func(id, content string, release <-chan struct{}) scheduledToolCall {
		return scheduledToolCall{
			call:   ToolCall{ID: id, Function: Func{Name: "read"}},
			effect: toolEffectSafe,
			run: func(context.Context) (string, error) {
				started <- id
				<-release
				finished <- id
				return content, nil
			},
		}
	}
	calls := []scheduledToolCall{
		blockingCall("call-1", "first result", releaseFirst),
		blockingCall("call-2", "second result", releaseSecond),
		{
			call: ToolCall{ID: "call-3", Function: Func{Name: "read"}},
			run: func(context.Context) (string, error) {
				thirdRan.Store(true)
				return "third result", nil
			},
		},
	}

	done := make(chan []scheduledToolResult, 1)
	go func() {
		done <- runScheduledToolCalls(ctx, 2, calls)
	}()

	wantStarted := map[string]bool{"call-1": true, "call-2": true}
	for range 2 {
		select {
		case id := <-started:
			if !wantStarted[id] {
				t.Fatalf("unexpected started call %q", id)
			}
			delete(wantStarted, id)
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for the two admitted calls to start")
		}
	}
	if len(wantStarted) != 0 {
		t.Fatalf("calls never started: %v", wantStarted)
	}

	cancel()
	close(releaseSecond)
	if id := <-finished; id != "call-2" {
		t.Fatalf("first completed body = %q, want call-2", id)
	}
	close(releaseFirst)

	var results []scheduledToolResult
	select {
	case results = <-done:
	case <-time.After(time.Second):
		t.Fatal("scheduler did not drain the already-started calls")
	}

	if thirdRan.Load() {
		t.Fatal("call-3 executed after cancellation froze the queued set")
	}
	if len(results) != len(calls) {
		t.Fatalf("results = %d, want one terminal result for each of %d calls", len(results), len(calls))
	}
	for i, wantID := range []string{"call-1", "call-2", "call-3"} {
		if results[i].call.ID != wantID {
			t.Fatalf("result[%d] call id = %q, want model-order id %q", i, results[i].call.ID, wantID)
		}
	}
	if !results[0].started || !results[1].started {
		t.Fatalf("started outcomes lost: first=%+v second=%+v", results[0], results[1])
	}
	if results[0].content != "first result" || results[1].content != "second result" {
		t.Fatalf("started outcomes = %q, %q", results[0].content, results[1].content)
	}
	if results[2].started {
		t.Fatal("call-3 is marked started despite never executing")
	}
	var receipt ToolReceipt
	if err := json.Unmarshal([]byte(results[2].content), &receipt); err != nil {
		t.Fatalf("call-3 result is not a typed receipt: %v (%q)", err, results[2].content)
	}
	if receipt.Status != ToolResultSkipped || receipt.Reason != toolCallSkippedByCancellation {
		t.Fatalf("call-3 receipt = %+v, want status=%q reason=%q", receipt, ToolResultSkipped, toolCallSkippedByCancellation)
	}
}

func TestToolSchedulerOverlapsSafeBodiesAndPreservesModelOrder(t *testing.T) {
	release := make(chan struct{})
	started := make(chan string, 2)
	call := func(id string) scheduledToolCall {
		return scheduledToolCall{call: ToolCall{ID: id}, effect: toolEffectSafe, run: func(context.Context) (string, error) {
			started <- id
			<-release
			return id + " result", nil
		}}
	}
	done := make(chan []scheduledToolResult, 1)
	go func() {
		done <- runScheduledToolCalls(context.Background(), 2, []scheduledToolCall{call("first"), call("second")})
	}()
	for range 2 {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("safe tool bodies ran serially")
		}
	}
	close(release)
	results := <-done
	if results[0].content != "first result" || results[1].content != "second result" {
		t.Fatalf("results lost model order: %+v", results)
	}
}

func TestToolSchedulerExclusiveCallIsBarrier(t *testing.T) {
	var active atomic.Int32
	var crossed atomic.Bool
	mk := func(id string, effect toolEffectClass) scheduledToolCall {
		return scheduledToolCall{call: ToolCall{ID: id}, effect: effect, run: func(context.Context) (string, error) {
			if active.Add(1) != 1 {
				crossed.Store(true)
			}
			time.Sleep(20 * time.Millisecond)
			active.Add(-1)
			return id, nil
		}}
	}
	calls := []scheduledToolCall{mk("read-1", toolEffectSafe), mk("write", toolEffectExclusive), mk("read-2", toolEffectSafe)}
	results := runScheduledToolCalls(context.Background(), 3, calls)
	if crossed.Load() {
		t.Fatal("an exclusive call overlapped across its barrier")
	}
	for i, want := range []string{"read-1", "write", "read-2"} {
		if results[i].content != want {
			t.Fatalf("result[%d]=%q, want %q", i, results[i].content, want)
		}
	}
}

func TestToolEffectForUnknownDefaultsExclusive(t *testing.T) {
	if got := toolEffectFor("unknown_tool"); got != toolEffectExclusive {
		t.Fatalf("unknown tool effect = %v, want exclusive", got)
	}
	if got := toolEffectFor(toolSearch); got != toolEffectSafe {
		t.Fatalf("declared read effect = %v, want safe", got)
	}
}
