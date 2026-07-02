package toolprocgate

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

const liveBody = `{"stdout":"secret-adjacent bytes the revoked call produced"}`

func call(trace string) *abi.ToolCall {
	return &abi.ToolCall{Tool: "bg_dump", TraceID: trace}
}

func result(body string) *abi.Result {
	return &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body), Len: int64(len(body))}}
}

// TestKernelPathQuarantinesPostKillResult is the seam-2 witness THROUGH THE
// REAL KERNEL FOLD: a completion for a revoked call, run through
// kernel.AdmitResult (the same admitResult the Reap path and the gateway's
// served path use), is quarantined citing TOOL_RESULT_AFTER_KILL, its payload
// stubbed before it can enter context.
func TestKernelPathQuarantinesPostKillResult(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	Kill("t-revoked", toolproc.ReasonToolDeadlineExceededName)

	c := call("t-revoked")
	r := result(liveBody)
	v := kernel.New("").AdmitResult(context.Background(), c, r)

	if v.Kind != abi.VerdictQuarantine || v.Reason != toolproc.ReasonToolResultAfterKill {
		t.Fatalf("want Quarantine/TOOL_RESULT_AFTER_KILL, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if r.Meta["admit"] != "quarantined" {
		t.Errorf("kernel admit meta = %q, want quarantined", r.Meta["admit"])
	}
	if r.Meta["kill_reason"] != toolproc.ReasonToolDeadlineExceededName {
		t.Errorf("kill_reason meta = %q, want %s", r.Meta["kill_reason"], toolproc.ReasonToolDeadlineExceededName)
	}
	after := string(r.Payload.Inline)
	if strings.Contains(after, "secret-adjacent") {
		t.Errorf("revoked call's payload leaked through the stub: %q", after)
	}
	var stub map[string]any
	if err := json.Unmarshal(r.Payload.Inline, &stub); err != nil {
		t.Fatalf("stub is not structured JSON: %v (%q)", err, after)
	}
	if stub["_quarantined"] != true || stub["reason"] != toolproc.ReasonToolResultAfterKillName {
		t.Errorf("stub = %v, want _quarantined + TOOL_RESULT_AFTER_KILL", stub)
	}
	if r.Payload.Taint != abi.TaintQuarantined {
		t.Errorf("stub taint = %v, want TaintQuarantined", r.Payload.Taint)
	}
}

// TestKernelPathAdmitsLiveResultUnchanged: with no revocation, the gate Defers
// and the kernel admits the result byte-identical — registered-but-inert. (A
// winning Defer is the kernel's admit-as-is default branch; secretgate's clean
// path asserts the same kind.)
func TestKernelPathAdmitsLiveResultUnchanged(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	c := call("t-live")
	r := result(liveBody)
	v := kernel.New("").AdmitResult(context.Background(), c, r)

	if v.Kind != abi.VerdictDefer || v.By != "toolprocgate" {
		t.Fatalf("live result must Defer through the inert gate, got %v (%s)", v.Kind, v.By)
	}
	if r.Meta["admit"] == "quarantined" || r.Meta["admit"] == "denied" {
		t.Fatalf("live result must be admitted, got admit=%q", r.Meta["admit"])
	}
	if got := string(r.Payload.Inline); got != liveBody {
		t.Errorf("live payload was altered: %q", got)
	}
	if r.Meta["toolprocgate"] != "" {
		t.Errorf("live result must carry no gate meta, got %q", r.Meta["toolprocgate"])
	}
}

// TestDirectAdmitEdges: nil call / empty trace / nil result never panic and
// always Defer.
func TestDirectAdmitEdges(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	g := Gate{}
	ctx := context.Background()
	if v := g.Admit(ctx, nil, result("x")); v.Kind != abi.VerdictDefer {
		t.Errorf("nil call: want Defer, got %v", v.Kind)
	}
	if v := g.Admit(ctx, call(""), result("x")); v.Kind != abi.VerdictDefer {
		t.Errorf("empty trace: want Defer, got %v", v.Kind)
	}
	if v := g.Admit(ctx, call("t"), nil); v.Kind != abi.VerdictDefer {
		t.Errorf("nil result: want Defer, got %v", v.Kind)
	}
}

// TestKillTableSemantics: idempotent first-reason-wins, empty id ignored,
// Reset clears, default reason applied.
func TestKillTableSemantics(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	Kill("", "IGNORED")
	if _, ok := KilledReason(""); ok {
		t.Error("empty call id must not enter the table")
	}
	Kill("c1", toolproc.ReasonToolOrphanedName)
	Kill("c1", "LATER_REASON")
	if r, _ := KilledReason("c1"); r != toolproc.ReasonToolOrphanedName {
		t.Errorf("first reason must win, got %q", r)
	}
	Kill("c2", "")
	if r, _ := KilledReason("c2"); r != toolproc.ReasonToolResultAfterKillName {
		t.Errorf("empty reason must default to the closed token, got %q", r)
	}
	Reset()
	if _, ok := KilledReason("c1"); ok {
		t.Error("Reset must clear the table")
	}
}

// TestKillTableBounded: the table evicts FIFO past maxKills — the oldest
// revocation ages out, the newest stays.
func TestKillTableBounded(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	for i := 0; i <= maxKills; i++ {
		Kill(fmt.Sprintf("c%d", i), "X")
	}
	if _, ok := KilledReason("c0"); ok {
		t.Error("oldest entry must have evicted")
	}
	if _, ok := KilledReason(fmt.Sprintf("c%d", maxKills)); !ok {
		t.Error("newest entry must remain")
	}
}

// TestReasonNamesRegistered: this leaf's init is the in-kernel consumer that
// registers toolproc's out-of-tree vocabulary — the names must round-trip.
func TestReasonNamesRegistered(t *testing.T) {
	for _, pr := range toolproc.ReasonPairs() {
		if got := abi.ReasonName(pr.Code); got != pr.Name {
			t.Errorf("ReasonName(%d) = %q, want %q", pr.Code, got, pr.Name)
		}
		if code, ok := abi.ReasonByName(pr.Name); !ok || code != pr.Code {
			t.Errorf("ReasonByName(%s) = %d,%t, want %d", pr.Name, code, ok, pr.Code)
		}
	}
}
