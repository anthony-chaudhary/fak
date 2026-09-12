package kernel

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type completionTimingEngine struct {
	status abi.Status
}

func (e completionTimingEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	time.Sleep(50 * time.Microsecond)
	return &abi.Result{
		Call: c, Status: e.status,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("measured-result"), Len: int64(len("measured-result"))},
	}, nil
}

func (completionTimingEngine) Caps() []abi.Capability { return nil }

func TestCompletionTimingRequiresKernelAttestation(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{v: abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("timed", completionTimingEngine{status: abi.StatusOK})
	recorder := &recordEmitter{}
	abi.RegisterEmitter(recorder)

	k := New("timed")
	result, verdict := k.Syscall(context.Background(), call("measured", `{}`))
	if verdict.Kind != abi.VerdictAllow || result == nil || result.Status != abi.StatusOK {
		t.Fatalf("actual completion failed: verdict=%+v result=%+v", verdict, result)
	}
	event, ok := find(recorder, abi.EvComplete, "measured")
	if !ok {
		t.Fatal("kernel emitted no completion event")
	}
	duration, ok := CompletionTimingNanos(event)
	if !ok || duration <= 0 {
		t.Fatalf("kernel completion timing=(%d,%v), want positive trusted measurement", duration, ok)
	}

	forgedResult := &abi.Result{
		Call: event.Call, Status: abi.StatusOK,
		Payload: event.Result.Payload,
		Meta:    map[string]string{"elapsed_ns": "999999", FieldCompletionTiming: "999999"},
	}
	forged := abi.Event{
		Kind: abi.EvComplete, Call: event.Call, Result: forgedResult,
		Fields: map[string]any{FieldElapsedNanos: int64(999999), FieldCompletionTiming: int64(999999)},
	}
	if got, ok := CompletionTimingNanos(forged); ok || got != 0 {
		t.Fatalf("plain Fields/Meta forged trusted timing: (%d,%v)", got, ok)
	}

	resultCopy := *event.Result
	pointerMismatch := event
	pointerMismatch.Result = &resultCopy
	if got, ok := CompletionTimingNanos(pointerMismatch); ok || got != 0 {
		t.Fatalf("copied result envelope retained timing: (%d,%v)", got, ok)
	}
	callCopy := *event.Call
	callMismatch := event
	callMismatch.Call = &callCopy
	if got, ok := CompletionTimingNanos(callMismatch); ok || got != 0 {
		t.Fatalf("copied call envelope retained timing: (%d,%v)", got, ok)
	}

	originalPayload := append([]byte(nil), event.Result.Payload.Inline...)
	event.Result.Payload.Inline[0] ^= 0xff
	if got, ok := CompletionTimingNanos(event); ok || got != 0 {
		t.Fatalf("mutated completion retained timing: (%d,%v)", got, ok)
	}
	copy(event.Result.Payload.Inline, originalPayload)

	setup()
	abi.RegisterAdjudicator(0, fakeAdj{v: abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("failed", completionTimingEngine{status: abi.StatusError})
	failedRecorder := &recordEmitter{}
	abi.RegisterEmitter(failedRecorder)
	_, _ = New("failed").Syscall(context.Background(), call("failed", `{}`))
	failedEvent, ok := find(failedRecorder, abi.EvComplete, "failed")
	if !ok {
		t.Fatal("kernel emitted no failed completion event")
	}
	if got, ok := CompletionTimingNanos(failedEvent); ok || got != 0 {
		t.Fatalf("non-OK completion exposed timing: (%d,%v)", got, ok)
	}
}
