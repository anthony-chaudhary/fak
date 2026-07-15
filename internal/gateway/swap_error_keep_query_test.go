package gateway

import (
	"strings"
	"testing"
)

func TestSwapErrorKeepQuery(t *testing.T) {
	t.Setenv(swapErrorKeepQueryEnv, "true")
	before := ErrorQueryTurn{
		Content:    "POLICY_BLOCK: do not retry refund_payment",
		Correction: "Keep POLICY_BLOCK. Use search_kb with the customer reference.",
		RestoreID:  "origin-task-deadbeef",
		Failed:     true,
	}
	after := SwapErrorKeepQuery(before)
	if !after.Replacement || after.Failed {
		t.Fatalf("after=%+v", after)
	}
	if after.Content != "Keep POLICY_BLOCK. Use search_kb with the customer reference." {
		t.Fatalf("golden after=%q", after.Content)
	}
	if after.RestoreID != before.RestoreID {
		t.Fatalf("pin changed: before=%q after=%q", before.RestoreID, after.RestoreID)
	}
}

func TestSwapErrorKeepQueryReframeFailSafe(t *testing.T) {
	t.Setenv(swapErrorKeepQueryEnv, "true")
	in := ErrorQueryTurn{
		Content:   "POLICY_BLOCK: do not retry refund_payment; use search_kb.",
		RestoreID: "pin-1", Failed: true,
	}
	once := SwapErrorKeepQuery(in)
	if !strings.Contains(once.Content, "POLICY_BLOCK") || !strings.Contains(once.Content, "refund_payment") {
		t.Fatalf("must-keep reason token dropped: %q", once.Content)
	}
	twice := SwapErrorKeepQuery(once)
	if twice != once {
		t.Fatalf("not idempotent: once=%+v twice=%+v", once, twice)
	}
}

func TestSwapErrorKeepQueryPreservesPin(t *testing.T) {
	t.Setenv(swapErrorKeepQueryEnv, "true")
	srv := newTestServer(t)
	const trace = "swap-error"
	task := []byte(`{"role":"user","content":"resolve the customer request"}`)
	id := "pin-byte-identical"
	srv.stashRestore(trace, id, "resolve the customer request", task)

	after := SwapErrorKeepQuery(ErrorQueryTurn{
		Content: "tool failed", Correction: "Use search_kb with the customer reference.",
		RestoreID: id, Failed: true,
	})
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: after.RestoreID, TraceID: trace})
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != id || got.Bytes != string(task) {
		t.Fatalf("restored=%+v", got)
	}
}

func TestSwapErrorKeepQueryDefaultOff(t *testing.T) {
	t.Setenv(swapErrorKeepQueryEnv, "")
	before := ErrorQueryTurn{Content: "tool failed", Correction: "Use search_kb.", RestoreID: "pin", Failed: true}
	if after := SwapErrorKeepQuery(before); after != before {
		t.Fatalf("default-off changed turn: before=%+v after=%+v", before, after)
	}

	t.Setenv(swapErrorKeepQueryEnv, "true")
	start := SwapErrorKeepQueryTotal()
	_ = SwapErrorKeepQuery(before)
	if got := SwapErrorKeepQueryTotal(); got != start+1 {
		t.Fatalf("counter=%d start=%d", got, start)
	}
	if line := SwapErrorKeepQueryPrometheus(); !strings.Contains(line, "fak_swap_error_keep_query_total ") {
		t.Fatalf("metrics=%q", line)
	}
}
