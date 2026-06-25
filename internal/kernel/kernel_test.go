package kernel

import (
	"context"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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

type verdictAdmitter struct{ v abi.Verdict }

func (a verdictAdmitter) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	return a.v
}
func (a verdictAdmitter) Caps() []abi.Capability { return nil }

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

func TestAdmitResultDenyHardRefuses(t *testing.T) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "result-deny-test",
	}})
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("AdmitResult = %v, want Deny", v.Kind)
	}
	if r.Status != abi.StatusError || r.Outcome != abi.OutcomeCommitted {
		t.Fatalf("denied result status/outcome = %v/%v, want Error/Committed", r.Status, r.Outcome)
	}
	if r.Meta["admit"] != "denied" || r.Meta["reason"] != "TRUST_VIOLATION" || r.Meta["disposition"] != "ESCALATE" {
		t.Fatalf("denied result meta = %+v, want admit=denied reason=TRUST_VIOLATION disposition=ESCALATE", r.Meta)
	}
	if len(r.Payload.Inline) != 0 || r.Payload.Kind != 0 {
		t.Fatalf("denied result must not carry the original payload, got %+v", r.Payload)
	}
	if c := k.Counters(); c.ResultDenies != 1 || c.Admitted != 0 {
		t.Fatalf("counters = %+v, want ResultDenies=1 and Admitted=0", c)
	}
}

func TestAdmitResultRequireWitnessFailsClosed(t *testing.T) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{requireWitness("origin:trusted-read")})
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonUnwitnessed {
		t.Fatalf("unwitnessed result must DENY/UNWITNESSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if r.Status != abi.StatusError || r.Meta["admit"] != "denied" || r.Meta["reason"] != "UNWITNESSED" {
		t.Fatalf("unwitnessed result was not denied-as-value: status=%v meta=%+v", r.Status, r.Meta)
	}
	if c := k.Counters(); c.ResultDenies != 1 || c.Admitted != 0 {
		t.Fatalf("counters = %+v, want ResultDenies=1 and Admitted=0", c)
	}
}

func TestAdmitResultRequireWitnessConfirmedAdmits(t *testing.T) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{requireWitness("origin:trusted-read")})
	abi.RegisterWitnessResolver("test", fakeWitness{abi.WitnessConfirmed})
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictAllow || v.Meta["witness"] != "confirmed" {
		t.Fatalf("confirmed result witness must admit with witness metadata, got %+v", v)
	}
	if r.Status != abi.StatusOK {
		t.Fatalf("confirmed result status = %v, want OK", r.Status)
	}
	if r.Meta != nil && r.Meta["admit"] == "denied" {
		t.Fatalf("confirmed result must not be denied: %+v", r.Meta)
	}
	if c := k.Counters(); c.ResultDenies != 0 || c.Admitted != 1 {
		t.Fatalf("counters = %+v, want ResultDenies=0 and Admitted=1", c)
	}
}

func TestAdmitResultDenyEmitsResultDeny(t *testing.T) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
		By:     "result-deny-test",
	}})
	rec := &recordEmitter{}
	abi.RegisterEmitter(rec)
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	k.AdmitResult(context.Background(), c, r)
	if !rec.has(abi.EvResultDeny) {
		t.Fatalf("direct AdmitResult deny did not emit EvResultDeny: %+v", rec.events)
	}
	if rec.has(abi.EvQuarantine) {
		t.Fatalf("result deny must not be reported as quarantine: %+v", rec.events)
	}
}

func TestAdmitResultDefaultAdmitUnchanged(t *testing.T) {
	setup()
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictAllow || v.By != "default-admit" {
		t.Fatalf("empty result-admitter chain = %+v, want Allow/default-admit", v)
	}
	if r.Status != abi.StatusOK || string(r.Payload.Inline) != "{}" {
		t.Fatalf("empty chain mutated result: %+v", r)
	}
	if v = k.AdmitResult(context.Background(), c, nil); v.Kind != abi.VerdictAllow || v.By != "default-admit" {
		t.Fatalf("nil result = %+v, want Allow/default-admit", v)
	}
	if c := k.Counters(); c.Admitted != 2 || c.Quarantines != 0 || c.ResultDenies != 0 {
		t.Fatalf("default-admit counters = %+v, want Admitted=2 only", c)
	}
}

func TestAdmitResultTransformTallyUnchanged(t *testing.T) {
	setup()
	rewritten := abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"rewritten":true}`)}
	abi.RegisterResultAdmitter(0, verdictAdmitter{abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "transform-test",
		Payload: abi.TransformPayload{NewArgs: rewritten},
	}})
	k := New("")
	c := call("read_x", "{}")
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}

	v := k.AdmitResult(context.Background(), c, r)
	if v.Kind != abi.VerdictTransform || string(r.Payload.Inline) != `{"rewritten":true}` {
		t.Fatalf("transform result = verdict %+v payload %q, want transformed payload", v, string(r.Payload.Inline))
	}
	if r.Meta["admit"] != "transformed" {
		t.Fatalf("transform admit meta = %+v, want admit=transformed", r.Meta)
	}
	if c := k.Counters(); c.Admitted != 1 || c.Quarantines != 0 || c.ResultDenies != 0 {
		t.Fatalf("transform counters = %+v, want Admitted=1 only", c)
	}
}

type resultEffect struct {
	Status  abi.Status
	Outcome abi.Outcome
	Payload abi.Ref
	Meta    map[string]string
}

func effectOf(r *abi.Result) resultEffect {
	if r == nil {
		return resultEffect{}
	}
	return resultEffect{Status: r.Status, Outcome: r.Outcome, Payload: r.Payload, Meta: r.Meta}
}

func TestResultLadderParity(t *testing.T) {
	rewritten := abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"rewritten":true}`)}
	cases := []struct {
		name string
		v    abi.Verdict
	}{
		{"allow", abi.Verdict{Kind: abi.VerdictAllow, By: "allow-test"}},
		{"quarantine", abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonSecretExfil, By: "quarantine-test"}},
		{"transform", abi.Verdict{Kind: abi.VerdictTransform, By: "transform-test", Payload: abi.TransformPayload{NewArgs: rewritten}}},
		{"require-witness", requireWitness("origin:trusted-read")},
		{"deny", abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny-test"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setup()
			abi.RegisterResultAdmitter(0, verdictAdmitter{tc.v})
			c := call("read_x", "{}")
			direct := &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}
			New("").AdmitResult(context.Background(), c, direct)
			directEffect := effectOf(direct)

			setup()
			abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "allow-test"}})
			abi.RegisterResultAdmitter(0, verdictAdmitter{tc.v})
			abi.RegisterEngine("e", &countEngine{})
			reaped, _ := New("e").Syscall(context.Background(), call("read_x", "{}"))
			reapEffect := effectOf(reaped)

			if !reflect.DeepEqual(directEffect, reapEffect) {
				t.Fatalf("AdmitResult/Reap result effect mismatch\n direct=%+v\n reap=%+v", directEffect, reapEffect)
			}
		})
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

// TestDenyResultWaitCarriesRetryAfter is the issue-#699 acceptance witness
// (criterion 4): an over-cap WAIT deny's DenyResult meta carries a non-empty
// retry_after that parses as a Go time.Duration (and retry_after_ms as integer ms),
// while a sub-cap (Allow) call carries none. The closed Disposition switch is
// unchanged (criterion 5) — the hint rides the existing WAIT path only.
func TestDenyResultWaitCarriesRetryAfter(t *testing.T) {
	c := &abi.ToolCall{Tool: "rate_limited_tool", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	v := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonRateLimited,
		By:     "ratelimit",
		Meta:   map[string]string{"retry_after": "500ms", "retry_after_ms": "500"},
	}
	if got := Disposition(v.Reason); got != "WAIT" {
		t.Fatalf("Disposition = %q, want WAIT (no regression to the closed switch)", got)
	}
	r := DenyResult(c, v)
	if r.Meta["disposition"] != "WAIT" {
		t.Fatalf("disposition = %q, want WAIT", r.Meta["disposition"])
	}
	ra, err := time.ParseDuration(r.Meta["retry_after"])
	if err != nil || ra != 500*time.Millisecond {
		t.Fatalf("retry_after = %q (%v), want 500ms", r.Meta["retry_after"], err)
	}
	if ms, err := strconv.Atoi(r.Meta["retry_after_ms"]); err != nil || ms != 500 {
		t.Fatalf("retry_after_ms = %q (%v), want 500", r.Meta["retry_after_ms"], err)
	}
}

// TestDenyResultNonWaitNoRetryAfter proves the WAIT guard (criterion 4): a
// TERMINAL deny never carries retry_after even if its verdict meta happens to hold
// one, and a WAIT deny whose verdict carries no hint degrades to today's bare
// token. So a loop that ignores retry_after is byte-for-byte on the old behavior.
func TestDenyResultNonWaitNoRetryAfter(t *testing.T) {
	c := &abi.ToolCall{Tool: "x", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	// A TERMINAL deny (POLICY_BLOCK) with a stray retry_after in meta: must NOT surface.
	terminal := DenyResult(c, abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
		Meta:   map[string]string{"retry_after": "1s", "retry_after_ms": "1000"},
	})
	if terminal.Meta["disposition"] != "TERMINAL" {
		t.Fatalf("disposition = %q, want TERMINAL", terminal.Meta["disposition"])
	}
	if _, ok := terminal.Meta["retry_after"]; ok {
		t.Fatal("TERMINAL deny must not surface retry_after (WAIT guard)")
	}
	if _, ok := terminal.Meta["retry_after_ms"]; ok {
		t.Fatal("TERMINAL deny must not surface retry_after_ms (WAIT guard)")
	}
	// A WAIT deny with NO hint in its verdict degrades to today's bare-token deny.
	bare := DenyResult(c, abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonRateLimited})
	if bare.Meta["disposition"] != "WAIT" {
		t.Fatalf("disposition = %q, want WAIT", bare.Meta["disposition"])
	}
	if _, ok := bare.Meta["retry_after"]; ok {
		t.Fatal("a WAIT deny with no verdict hint must not invent retry_after")
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
