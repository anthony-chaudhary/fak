package vdso

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

type lookupTimingAdjudicator struct{}

func (lookupTimingAdjudicator) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "timing-test"}
}
func (lookupTimingAdjudicator) Caps() []abi.Capability { return nil }

type lookupTimingEngine struct{ fixtureID string }

func (e lookupTimingEngine) Complete(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
	body := []byte("engine-result-for-" + call.Tool)
	return &abi.Result{Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
		Meta:    map[string]string{"lookup_timing_fixture": e.fixtureID}}, nil
}
func (lookupTimingEngine) Caps() []abi.Capability { return nil }

type lookupTimingEmitter struct {
	fixtureID string
	v         *VDSO
}

func (e lookupTimingEmitter) Emit(ev abi.Event) {
	if ev.Result != nil && ev.Result.Meta["lookup_timing_fixture"] == e.fixtureID {
		e.v.Emit(ev)
	}
}

func (lookupTimingEmitter) Subscriptions() []abi.EventKind { return []abi.EventKind{abi.EvComplete} }

func timedReadCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}}
}

func newTimedResidentVDSO(t *testing.T, capacity int) (*VDSO, *kernel.Kernel) {
	t.Helper()
	fixtureID := "lookup-timing/" + t.Name()
	abi.RegisterEngine(fixtureID, lookupTimingEngine{fixtureID: fixtureID})
	v := New(capacity)
	abi.RegisterEmitter(lookupTimingEmitter{fixtureID: fixtureID, v: v})
	k := kernel.New(fixtureID, kernel.WithAdjudicators([]abi.Adjudicator{lookupTimingAdjudicator{}}))
	k.SetVDSO(false)
	return v, k
}

func TestLookupReceiptTimingRequiresMeasuredResidentHit(t *testing.T) {
	v, k := newTimedResidentVDSO(t, 1)
	first := timedReadCall("get_first", `{"id":"first"}`)
	if result, verdict := k.Syscall(context.Background(), first); verdict.Kind != abi.VerdictAllow || result == nil {
		t.Fatalf("cold engine completion failed: verdict=%+v result=%+v", verdict, result)
	}

	ctx, receipt := WithLookupReceipt(context.Background())
	warm, hit := (tier{v: v, n: 1}).Lookup(ctx, timedReadCall("get_first", `{"id":"first"}`))
	if !hit || warm == nil {
		t.Fatal("measured resident entry did not hit")
	}
	historical, lookup, ok := receipt.Timing(warm)
	if !ok || historical <= 0 || lookup < 0 {
		t.Fatalf("resident timing=(historical=%d lookup=%d ok=%v), want measured spans", historical, lookup, ok)
	}
	copyResult := *warm
	if h, l, ok := receipt.Timing(&copyResult); ok || h != 0 || l != 0 {
		t.Fatalf("copied result matched receipt timing: (%d,%d,%v)", h, l, ok)
	}
	forged := &abi.Result{Call: warm.Call, Status: abi.StatusOK, Payload: warm.Payload,
		Meta: map[string]string{"served_by": "vdso", "tier": "2", "elapsed_ns": "999999"}}
	if h, l, ok := receipt.Timing(forged); ok || h != 0 || l != 0 {
		t.Fatalf("forged result metadata matched receipt timing: (%d,%d,%v)", h, l, ok)
	}
	v.RegisterStatic("replacement", []byte(`{"answer":true}`))
	if replacement, hit := v.Lookup(ctx, timedReadCall("replacement", `{}`)); !hit || replacement == nil {
		t.Fatal("replacement lookup did not hit")
	}
	if h, l, ok := receipt.Timing(warm); ok || h != 0 || l != 0 {
		t.Fatalf("later lookup retained overwritten timing observation: (%d,%d,%v)", h, l, ok)
	}

	// A later cold completion evicts the capacity-one entry. Looking up the
	// evicted key on the same context must clear the prior timing observation.
	ctx, receipt = WithLookupReceipt(context.Background())
	warm, hit = (tier{v: v, n: 1}).Lookup(ctx, timedReadCall("get_first", `{"id":"first"}`))
	if !hit || warm == nil {
		t.Fatal("measured resident entry did not re-hit before eviction")
	}
	second := timedReadCall("get_second", `{"id":"second"}`)
	if result, verdict := k.Syscall(context.Background(), second); verdict.Kind != abi.VerdictAllow || result == nil {
		t.Fatalf("second cold completion failed: verdict=%+v result=%+v", verdict, result)
	}
	if stale, hit := v.Lookup(ctx, timedReadCall("get_first", `{"id":"first"}`)); hit || stale != nil {
		t.Fatalf("evicted entry still hit: result=%+v hit=%v", stale, hit)
	}
	if h, l, ok := receipt.Timing(warm); ok || h != 0 || l != 0 {
		t.Fatalf("miss reused evicted timing: (%d,%d,%v)", h, l, ok)
	}
}

func TestLookupReceiptTimingUnavailableForUnmeasuredTiers(t *testing.T) {
	v, _ := newTimedResidentVDSO(t, 4)

	v.RegisterStatic("static_answer", []byte(`{"answer":true}`))
	ctx, receipt := WithLookupReceipt(context.Background())
	static, hit := v.Lookup(ctx, timedReadCall("static_answer", `{}`))
	if !hit || static == nil {
		t.Fatal("static fixture did not hit")
	}
	if h, l, ok := receipt.Timing(static); ok || h != 0 || l != 0 {
		t.Fatalf("static result exposed historical timing: (%d,%d,%v)", h, l, ok)
	}

	unmeasuredCall := timedReadCall("unmeasured", `{"id":"manual"}`)
	v.Emit(completeEvent(unmeasuredCall, `{"value":1}`))
	ctx, receipt = WithLookupReceipt(context.Background())
	unmeasured, hit := v.Lookup(ctx, timedReadCall("unmeasured", `{"id":"manual"}`))
	if !hit || unmeasured == nil {
		t.Fatal("unmeasured resident fixture did not hit")
	}
	if h, l, ok := receipt.Timing(unmeasured); ok || h != 0 || l != 0 {
		t.Fatalf("unmeasured resident result exposed timing: (%d,%d,%v)", h, l, ok)
	}
}
