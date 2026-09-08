package agent

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestDispatchToolCallsClosesCancelledBatchInModelOrder(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var events []ProgressEvent
	var envelopes []harnesskit.Envelope
	var trace []traceEvent
	const traceID = "cancelled-batch"
	table := session.NewTable()
	table.SetBudget(traceID, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ToolCallsLeft: 1})
	audit := journal.OpenMemory()
	runner := &armRunner{
		cfg: &runConfig{trace: traceID, table: table, auditJournal: audit, observer: func(ev ProgressEvent) {
			events = append(events, ev)
		}},
		metrics:        &ArmMetrics{Arm: "baseline"},
		log:            &trace,
		stopTerminated: func() bool { return false },
		envelopeSink: func(env harnesskit.Envelope) {
			envelopes = append(envelopes, env)
		},
	}
	calls := []ToolCall{
		{ID: "read", Function: Func{Name: toolSearch, Arguments: `{}`}},
		{ID: "wait", Function: Func{Name: ToolTaskWait, Arguments: `{}`}},
		{ID: "todo", Function: Func{Name: ToolTodoRead, Arguments: `{}`}},
	}

	stopped, err := runner.dispatchToolCalls(ctx, 0, Message{Role: RoleAssistant, ToolCalls: calls})
	if err != nil {
		t.Fatalf("dispatchToolCalls() error = %v", err)
	}
	if stopped {
		t.Fatal("a cancelled tool batch terminated the arm instead of closing the turn")
	}
	if runner.metrics.ToolCalls != 0 || runner.metrics.ToolCallsSafe != 0 || runner.metrics.ToolCallsExclusive != 0 {
		t.Fatalf("cancelled calls consumed metrics: total:%d safe:%d exclusive:%d", runner.metrics.ToolCalls, runner.metrics.ToolCallsSafe, runner.metrics.ToolCallsExclusive)
	}
	if v := table.DebitToolCall(traceID); !v.Proceed {
		t.Fatalf("cancelled calls consumed the tool budget: %+v", v)
	}
	if len(runner.messages) != len(calls) || len(trace) != len(calls) {
		t.Fatalf("closed surfaces = messages:%d trace:%d, want %d each", len(runner.messages), len(trace), len(calls))
	}
	for i, call := range calls {
		msg := runner.messages[i]
		if msg.ToolCallID != call.ID || msg.Name != call.Function.Name {
			t.Fatalf("message[%d] = call:%q tool:%q, want call:%q tool:%q", i, msg.ToolCallID, msg.Name, call.ID, call.Function.Name)
		}
		var receipt ToolReceipt
		if err := json.Unmarshal([]byte(msg.Content), &receipt); err != nil {
			t.Fatalf("message[%d] is not a typed receipt: %v (%q)", i, err, msg.Content)
		}
		if receipt.Status != ToolResultSkipped || receipt.Reason != toolCallSkippedByCancellation {
			t.Fatalf("message[%d] receipt = %+v, want status=%q reason=%q", i, receipt, ToolResultSkipped, toolCallSkippedByCancellation)
		}
		if trace[i].Tool != call.Function.Name || trace[i].Verdict != "DROPPED" || trace[i].Reason != toolCallSkippedByCancellation {
			t.Fatalf("trace[%d] = %+v, want ordered DROPPED/%s", i, trace[i], toolCallSkippedByCancellation)
		}
	}

	var lifecycle []ProgressEvent
	for _, ev := range events {
		if ev.Kind == ProgressToolStarted {
			t.Fatalf("cancelled-before-dispatch call emitted tool_started: %+v", ev)
		}
		if ev.Kind == ProgressCallAdjudicated || ev.Kind == ProgressResultAdmitted {
			lifecycle = append(lifecycle, ev)
		}
	}
	if len(lifecycle) != len(calls)*2 {
		t.Fatalf("terminal lifecycle events = %d, want %d", len(lifecycle), len(calls)*2)
	}
	for i, call := range calls {
		adjudicated, admitted := lifecycle[i*2], lifecycle[i*2+1]
		if adjudicated.Kind != ProgressCallAdjudicated || admitted.Kind != ProgressResultAdmitted || adjudicated.CallID != call.ID || admitted.CallID != call.ID {
			t.Fatalf("lifecycle pair[%d] = %+v / %+v, want ordered adjudicated/admitted for %q", i, adjudicated, admitted, call.ID)
		}
	}
	if len(envelopes) != len(calls) {
		t.Fatalf("terminal envelopes = %d, want %d", len(envelopes), len(calls))
	}
	for i, env := range envelopes {
		if env.Type != harnesskit.EventToolCompleted {
			t.Fatalf("envelope[%d].Type = %q, want %q", i, env.Type, harnesskit.EventToolCompleted)
		}
		var payload harnesskit.ToolPayload
		if err := json.Unmarshal(env.Payload, &payload); err != nil {
			t.Fatalf("envelope[%d] payload: %v", i, err)
		}
		if payload.CallID != calls[i].ID || payload.Status != "denied" {
			t.Fatalf("envelope[%d] = %+v, want ordered denied completion for %q", i, payload, calls[i].ID)
		}
	}
	rows := audit.Recent(0)
	if len(rows) != len(calls) {
		t.Fatalf("audit decisions = %d, want %d", len(rows), len(calls))
	}
	for i, row := range rows {
		if row.By != "tool-scheduler/interruption" || row.Reason == "POLICY_BLOCK" {
			t.Fatalf("audit[%d] provenance = by:%q reason:%q, want scheduler/interruption and not policy", i, row.By, row.Reason)
		}
	}
}

func TestDispatchToolCallsBudgetFreezeAccountsOnlyStartedCall(t *testing.T) {
	const traceID = "budget-freeze"
	table := session.NewTable()
	table.SetBudget(traceID, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ToolCallsLeft: 1})
	var events []ProgressEvent
	var trace []traceEvent
	runner := &armRunner{
		cfg:            &runConfig{trace: traceID, table: table, observer: func(ev ProgressEvent) { events = append(events, ev) }},
		metrics:        &ArmMetrics{Arm: "baseline"},
		log:            &trace,
		stopTerminated: func() bool { return false },
	}
	calls := []ToolCall{
		{ID: "read-1", Function: Func{Name: toolSearch, Arguments: `{}`}},
		{ID: "read-2", Function: Func{Name: toolSearch, Arguments: `{}`}},
		{ID: "mutation", Function: Func{Name: toolDelete, Arguments: `{}`}},
	}

	stopped, err := runner.dispatchToolCalls(context.Background(), 0, Message{Role: RoleAssistant, ToolCalls: calls})
	if err != nil {
		t.Fatalf("dispatchToolCalls: %v", err)
	}
	if !stopped || runner.metrics.StoppedBySession != session.ReasonBudgetToolCalls {
		t.Fatalf("stop = %v/%q, want true/%q", stopped, runner.metrics.StoppedBySession, session.ReasonBudgetToolCalls)
	}
	if runner.metrics.ToolCalls != 1 || runner.metrics.ToolCallsSafe != 1 || runner.metrics.ToolCallsExclusive != 0 {
		t.Fatalf("admitted metrics = total:%d safe:%d exclusive:%d, want 1/1/0", runner.metrics.ToolCalls, runner.metrics.ToolCallsSafe, runner.metrics.ToolCallsExclusive)
	}
	started := 0
	for _, ev := range events {
		if ev.Kind == ProgressToolStarted {
			started++
			if ev.CallID != "read-1" {
				t.Fatalf("unexpected started call: %+v", ev)
			}
		}
	}
	if started != 1 || len(runner.messages) != len(calls) {
		t.Fatalf("started/messages = %d/%d, want 1/%d", started, len(runner.messages), len(calls))
	}
	for i := 1; i < len(calls); i++ {
		var receipt ToolReceipt
		if err := json.Unmarshal([]byte(runner.messages[i].Content), &receipt); err != nil {
			t.Fatalf("message[%d] receipt: %v", i, err)
		}
		if receipt.Status != ToolResultSkipped || receipt.Reason != session.ReasonBudgetToolCalls || receipt.Disposition != "TERMINAL" {
			t.Fatalf("message[%d] = %+v, want terminal budget skip", i, receipt)
		}
	}
	if len(trace) != 1 || trace[0].Tool != toolSearch || trace[0].Verdict == "DROPPED" {
		t.Fatalf("trace = %+v, want exactly 1 non-DROPPED trace event", trace)
	}
}

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
		results, err := runScheduledToolCalls(ctx, 2, calls)
		if err != nil {
			t.Errorf("runScheduledToolCalls: %v", err)
		}
		done <- results
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
		results, err := runScheduledToolCalls(context.Background(), 2, []scheduledToolCall{call("first"), call("second")})
		if err != nil {
			t.Errorf("runScheduledToolCalls: %v", err)
		}
		done <- results
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

func TestToolSchedulerExclusiveCallWaitsForReadCommit(t *testing.T) {
	var active atomic.Int32
	var crossed atomic.Bool
	var readCommitted atomic.Bool
	mk := func(id string, effect toolEffectClass) scheduledToolCall {
		call := scheduledToolCall{call: ToolCall{ID: id}, effect: effect, run: func(context.Context) (string, error) {
			if effect == toolEffectExclusive && !readCommitted.Load() {
				crossed.Store(true)
			}
			if active.Add(1) != 1 {
				crossed.Store(true)
			}
			time.Sleep(20 * time.Millisecond)
			active.Add(-1)
			return id, nil
		}}
		if id == "read-1" {
			call.commit = func(scheduledToolResult) error {
				readCommitted.Store(true)
				return nil
			}
		}
		return call
	}
	calls := []scheduledToolCall{mk("read-1", toolEffectSafe), mk("write", toolEffectExclusive), mk("read-2", toolEffectSafe)}
	results, err := runScheduledToolCalls(context.Background(), 3, calls)
	if err != nil {
		t.Fatalf("runScheduledToolCalls: %v", err)
	}
	if crossed.Load() {
		t.Fatal("an exclusive call overlapped across its barrier")
	}
	for i, want := range []string{"read-1", "write", "read-2"} {
		if results[i].content != want {
			t.Fatalf("result[%d]=%q, want %q", i, results[i].content, want)
		}
	}
}

func TestToolSchedulerCommitFailurePreventsLaterMutation(t *testing.T) {
	var mutationRan atomic.Bool
	wantErr := errors.New("read admission failed")
	calls := []scheduledToolCall{
		{
			call: ToolCall{ID: "read"}, effect: toolEffectSafe,
			run:    func(context.Context) (string, error) { return "read", nil },
			commit: func(scheduledToolResult) error { return wantErr },
		},
		{
			call: ToolCall{ID: "mutation"}, effect: toolEffectExclusive,
			run: func(context.Context) (string, error) {
				mutationRan.Store(true)
				return "mutation", nil
			},
		},
	}
	if _, err := runScheduledToolCalls(context.Background(), 2, calls); !errors.Is(err, wantErr) {
		t.Fatalf("runScheduledToolCalls error = %v, want %v", err, wantErr)
	}
	if mutationRan.Load() {
		t.Fatal("exclusive mutation ran after the preceding read failed to commit")
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
