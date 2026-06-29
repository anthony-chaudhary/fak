package kernel

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---- test doubles for the span-cost witness ---------------------------------

// slowAdj sleeps a fixed minimum so the EvSubmit->EvDecide adjudication span is
// reliably non-zero on a coarse monotonic clock, and returns a TRANSFORM for
// "redact_me" whose rewritten args are strictly LONGER than the originals (a token
// cost ADDED), an ALLOW otherwise.
type slowAdj struct{}

func (slowAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	time.Sleep(20 * time.Microsecond)
	if c.Tool == "redact_me" {
		ref, _ := abi.ActiveResolver().Put(ctx, []byte(`{"field":"[REDACTED-LONGER-THAN-THE-ORIGINAL-ARGS]"}`))
		return abi.Verdict{Kind: abi.VerdictTransform, By: "test", Payload: abi.TransformPayload{NewArgs: ref}}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}
func (slowAdj) Caps() []abi.Capability { return nil }

// slowEngine sleeps a fixed minimum so the EvDispatch->EvComplete engine span is
// reliably non-zero.
type slowEngine struct{}

func (slowEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	time.Sleep(50 * time.Microsecond)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (slowEngine) Caps() []abi.Capability { return nil }

// vdsoPayloadFP serves "vdso_me" locally with a non-empty payload, so the SAVED
// token-delta on the EvVDSOHit is strictly negative.
type vdsoPayloadFP struct{}

func (vdsoPayloadFP) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	if c.Tool == "vdso_me" {
		body := []byte("served-locally-no-engine-round-trip")
		return &abi.Result{Call: c, Status: abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}}, true
	}
	return nil, false
}
func (vdsoPayloadFP) Caps() []abi.Capability { return nil }

// fieldInt reads an int64 telemetry field the kernel stamped on Event.Fields.
func fieldInt(ev abi.Event, key string) (int64, bool) {
	if ev.Fields == nil {
		return 0, false
	}
	v, ok := ev.Fields[key].(int64)
	return v, ok
}

// find returns the FIRST recorded event matching kind+tool, or false.
func find(rec *recordEmitter, kind abi.EventKind, tool string) (abi.Event, bool) {
	for _, ev := range rec.events {
		if ev.Kind == kind && ev.Call != nil && ev.Call.Tool == tool {
			return ev, true
		}
	}
	return abi.Event{}, false
}

// TestSpanCostAcrossThreeSpans is the #1149 L0 witness: the kernel stamps non-zero,
// correctly-bucketed COST on the OPEN Event.Fields channel across the three lifecycle
// spans —
//
//	span 1  EvSubmit->EvDecide   adjudication tax   (elapsed_ns > 0)
//	span 2  EvDispatch->EvComplete  engine cost     (elapsed_ns > 0)
//	span 3  token-delta            transform ADDED (>0) vs vDSO SAVED (<0)
//
// so an observer can fold cost, not just verdict, and the offline rungstats read-out
// reconciles with the live fak_gateway_operation_duration_seconds twin.
func TestSpanCostAcrossThreeSpans(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, slowAdj{})
	abi.RegisterEngine("e", slowEngine{})
	abi.RegisterFastPath(1, vdsoPayloadFP{})

	rec := &recordEmitter{}
	abi.RegisterEmitter(rec)

	k := New("e")
	ctx := context.Background()

	// Allowed call: exercises the adjudication span (EvDecide) AND the engine span
	// (EvComplete) through the full Submit->Reap lifecycle.
	if _, v := k.Syscall(ctx, call("allow_me", "{}")); v.Kind != abi.VerdictAllow {
		t.Fatalf("allow_me: got %v, want ALLOW", v.Kind)
	}
	// Transform: re-emits longer args (token ADDED), then dispatches.
	if _, v := k.Syscall(ctx, call("redact_me", `{"f":"x"}`)); v.Kind != abi.VerdictTransform {
		t.Fatalf("redact_me: got %v, want TRANSFORM", v.Kind)
	}
	// vDSO hit: served locally (token SAVED).
	if _, v := k.Syscall(ctx, call("vdso_me", "{}")); v.By != "vdso" {
		t.Fatalf("vdso_me: verdict.By=%q, want vdso", v.By)
	}

	// Span 1 — adjudication tax on EvDecide.
	dec, ok := find(rec, abi.EvDecide, "allow_me")
	if !ok {
		t.Fatal("no EvDecide recorded for allow_me")
	}
	if adj, ok := fieldInt(dec, FieldElapsedNanos); !ok || adj <= 0 {
		t.Fatalf("span 1 adjudication tax: EvDecide[%s]=%d (ok=%v), want > 0", FieldElapsedNanos, adj, ok)
	}

	// Span 2 — engine cost on EvComplete.
	comp, ok := find(rec, abi.EvComplete, "allow_me")
	if !ok {
		t.Fatal("no EvComplete recorded for allow_me")
	}
	if eng, ok := fieldInt(comp, FieldElapsedNanos); !ok || eng <= 0 {
		t.Fatalf("span 2 engine cost: EvComplete[%s]=%d (ok=%v), want > 0", FieldElapsedNanos, eng, ok)
	}

	// Span 3a — token ADDED by the transform (EvDecide carries the original-vs-rewrite delta).
	tdec, ok := find(rec, abi.EvDecide, "redact_me")
	if !ok {
		t.Fatal("no EvDecide recorded for redact_me")
	}
	if td, ok := fieldInt(tdec, FieldTokenDelta); !ok || td <= 0 {
		t.Fatalf("span 3a transform token-delta: EvDecide[%s]=%d (ok=%v), want > 0 (added)", FieldTokenDelta, td, ok)
	}

	// Span 3b — token SAVED by the vDSO hit (negative delta).
	vh, ok := find(rec, abi.EvVDSOHit, "vdso_me")
	if !ok {
		t.Fatal("no EvVDSOHit recorded for vdso_me")
	}
	if td, ok := fieldInt(vh, FieldTokenDelta); !ok || td >= 0 {
		t.Fatalf("span 3b vDSO token-delta: EvVDSOHit[%s]=%d (ok=%v), want < 0 (saved)", FieldTokenDelta, td, ok)
	}
}
