package kernel

import (
	"context"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---- test doubles -----------------------------------------------------------

type inlineRes struct{}

func (inlineRes) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) { return r.Inline, nil }
func (inlineRes) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...), Len: int64(len(b))}, nil
}

type inlineBackend struct{}

func (inlineBackend) Resolver() abi.Resolver { return inlineRes{} }
func (inlineBackend) Caps() []abi.Capability { return nil }

type fakeAdj struct{ v abi.Verdict }

func (f fakeAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict { return f.v }
func (f fakeAdj) Caps() []abi.Capability                                      { return nil }

type countEngine struct{ n int64 }

func (e *countEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	atomic.AddInt64(&e.n, 1)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (e *countEngine) Caps() []abi.Capability { return nil }

type namedEngine struct {
	id string
	n  int64
}

func (e *namedEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	atomic.AddInt64(&e.n, 1)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args, Meta: map[string]string{"engine": e.id}}, nil
}
func (e *namedEngine) Caps() []abi.Capability { return nil }

type fakeFP struct{ hit bool }

func (f fakeFP) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	if f.hit {
		return &abi.Result{Call: c, Status: abi.StatusOK, Meta: map[string]string{"served_by": "fp"}}, true
	}
	return nil, false
}
func (f fakeFP) Caps() []abi.Capability { return nil }

type recordEmitter struct{ events []abi.Event }

func (r *recordEmitter) Emit(ev abi.Event) { r.events = append(r.events, ev) }

func (r *recordEmitter) has(kind abi.EventKind) bool {
	for _, ev := range r.events {
		if ev.Kind == kind {
			return true
		}
	}
	return false
}

type quarantineAdmitter struct{}

func (quarantineAdmitter) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictQuarantine, By: "test"}
}
func (quarantineAdmitter) Caps() []abi.Capability { return nil }

func setup() { abi.ResetForTest(); abi.RegisterRegionBackend(inlineBackend{}) }

func call(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

// ---- unit 15: default-deny ---------------------------------------------------

func TestFoldDefaultDenyEmptyPolicy(t *testing.T) {
	if v := Fold(context.Background(), nil, call("x", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("empty chain must DENY, got %v", v.Kind)
	}
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictDefer}}}
	if v := Fold(context.Background(), chain, call("x", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("all-defer must DENY, got %v", v.Kind)
	}
}

func TestFoldMostRestrictiveWins(t *testing.T) {
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}},
	}
	if v := Fold(context.Background(), chain, call("x", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("deny must beat allow in the lattice, got %v", v.Kind)
	}
}

// ---- units 16/17/18: allow dispatches, deny doesn't, transform mutates -------

func TestAllowReachesDispatch(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")
	r, _ := k.Syscall(context.Background(), call("read_x", "{}"))
	if r.Status != abi.StatusOK || atomic.LoadInt64(&eng.n) != 1 {
		t.Fatalf("allow must reach dispatch exactly once (engine n=%d)", eng.n)
	}
}

func TestPerCallEngineRouteOverridesKernelDefault(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	local := &namedEngine{id: "local"}
	remote := &namedEngine{id: "remote"}
	abi.RegisterEngine("local", local)
	abi.RegisterEngine("remote", remote)
	k := New("local")

	r, v := k.Syscall(context.Background(), &abi.ToolCall{
		Tool:   "read_x",
		Engine: "remote",
		Args:   abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	})
	if v.Kind != abi.VerdictAllow || r.Meta["engine"] != "remote" {
		t.Fatalf("per-call route should dispatch to remote engine, verdict=%v meta=%v", v.Kind, r.Meta)
	}
	if atomic.LoadInt64(&remote.n) != 1 || atomic.LoadInt64(&local.n) != 0 {
		t.Fatalf("route counts: local=%d remote=%d, want local=0 remote=1", local.n, remote.n)
	}

	r, v = k.Syscall(context.Background(), call("read_default", "{}"))
	if v.Kind != abi.VerdictAllow || r.Meta["engine"] != "local" {
		t.Fatalf("empty route should dispatch to kernel default, verdict=%v meta=%v", v.Kind, r.Meta)
	}
	if atomic.LoadInt64(&local.n) != 1 {
		t.Fatalf("default route did not hit local engine, local=%d", local.n)
	}
}

func TestDenyNeverReachesDispatch(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")
	r, v := k.Syscall(context.Background(), call("danger", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("want DENY verdict, got %v", v.Kind)
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("deny must NEVER reach dispatch, engine n=%d", eng.n)
	}
	if r.Meta["disposition"] == "" {
		t.Fatal("deny result must carry a disposition (deny-as-value, unit 74)")
	}
}

func TestTransformMutatesArgs(t *testing.T) {
	setup()
	newArgs := abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"redacted":true}`)}
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictTransform,
		Payload: abi.TransformPayload{NewArgs: newArgs}}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")
	r, _ := k.Syscall(context.Background(), call("write_x", `{"secret":"x"}`))
	if string(r.Payload.Inline) != `{"redacted":true}` {
		t.Fatalf("transform must mutate args before dispatch, engine saw %q", r.Payload.Inline)
	}
}

// ---- unit 30: vDSO consulted first -------------------------------------------

func TestVDSOConsultedBeforeAdjudicator(t *testing.T) {
	setup()
	abi.RegisterFastPath(1, fakeFP{hit: true})
	// a DENY adjudicator that must NOT run because the fast path short-circuits
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictDeny}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")
	r, v := k.Syscall(context.Background(), call("read_x", "{}"))
	if v.By != "vdso" || r.Meta["served_by"] != "fp" {
		t.Fatalf("vDSO must be consulted first and short-circuit, got verdict.by=%q", v.By)
	}
	if atomic.LoadInt64(&eng.n) != 0 || k.Counters().VDSOHits != 1 {
		t.Fatalf("a vDSO hit must skip the engine (n=%d hits=%d)", eng.n, k.Counters().VDSOHits)
	}
}

// ---- unit 73: verdict routing ; unit 75: batch ; ctxmmu quarantine ----------

func TestResultAdmitQuarantineCounted(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("e", &countEngine{})
	abi.RegisterResultAdmitter(0, quarantineAdmitter{})
	k := New("e")
	k.Syscall(context.Background(), call("read_x", "{}"))
	if k.Counters().Quarantines != 1 {
		t.Fatalf("quarantine admitter must increment the quarantine counter, got %d", k.Counters().Quarantines)
	}
}

func TestBatchEqualsSerial(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	k := New("")
	calls := []*abi.ToolCall{call("a", "{}"), call("b", "{}"), call("c", "{}")}
	batch := k.BatchDecide(context.Background(), calls)
	for i, c := range calls {
		if serial := k.Decide(context.Background(), c); serial.Kind != batch[i].Kind {
			t.Fatalf("batch[%d]=%v != serial=%v", i, batch[i].Kind, serial.Kind)
		}
	}
}

func TestDirectDecideEmitsDecisionAndDeny(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}})
	rec := &recordEmitter{}
	abi.RegisterEmitter(rec)
	k := New("")

	v := k.Decide(context.Background(), call("deny_x", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Decide = %v, want Deny", v.Kind)
	}
	if !rec.has(abi.EvDecide) {
		t.Fatalf("direct Decide did not emit EvDecide: %+v", rec.events)
	}
	if !rec.has(abi.EvDeny) {
		t.Fatalf("direct Decide deny did not emit EvDeny: %+v", rec.events)
	}
}

func TestDirectAdmitResultEmitsQuarantine(t *testing.T) {
	setup()
	abi.RegisterResultAdmitter(0, quarantineAdmitter{})
	rec := &recordEmitter{}
	abi.RegisterEmitter(rec)
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("AdmitResult = %v, want Quarantine", v.Kind)
	}
	if !rec.has(abi.EvQuarantine) {
		t.Fatalf("direct AdmitResult quarantine did not emit EvQuarantine: %+v", rec.events)
	}
}

// ---- unit 74: disposition mapping --------------------------------------------

func TestDispositionMapping(t *testing.T) {
	cases := map[abi.ReasonCode]string{
		abi.ReasonMisroute:    "RETRYABLE",
		abi.ReasonRateLimited: "WAIT",
		abi.ReasonSelfModify:  "ESCALATE",
		abi.ReasonPolicyBlock: "TERMINAL",
	}
	for r, want := range cases {
		if got := Disposition(r); got != want {
			t.Fatalf("Disposition(%s)=%s want %s", abi.ReasonName(r), got, want)
		}
	}
}

// ---- unit 72: no os/exec on the dispatch hot path (ABSENCE proof) ------------

func TestNoOsExecOnHotPath(t *testing.T) {
	for _, f := range []string{"kernel.go"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), `"os/exec"`) {
			t.Fatalf("%s imports os/exec — forbidden on the dispatch hot path", f)
		}
	}
}

// ---- unit 76: race-clean dispatcher ------------------------------------------

func TestDispatchRaceClean(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("e", &countEngine{})
	k := New("e")
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			k.Syscall(context.Background(), call("read_x", "{}"))
		}()
	}
	wg.Wait()
	if k.Counters().Submits != 50 {
		t.Fatalf("want 50 submits, got %d", k.Counters().Submits)
	}
}
