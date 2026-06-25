package rungobs

import (
	"context"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// ---- test doubles (self-contained: no cross-package test helpers) -----------

type inlineRes struct{}

func (inlineRes) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) { return r.Inline, nil }
func (inlineRes) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...), Len: int64(len(b))}, nil
}

type inlineBackend struct{}

func (inlineBackend) Resolver() abi.Resolver { return inlineRes{} }
func (inlineBackend) Caps() []abi.Capability { return nil }

// mixedAdj produces a different verdict per tool name so one stream exercises the
// allow/deny/transform buckets (and the all-defer → default-deny path).
type mixedAdj struct{}

func (mixedAdj) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	switch c.Tool {
	case "allow_it":
		return abi.Verdict{Kind: abi.VerdictAllow, By: "mixed"}
	case "deny_it":
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "mixed"}
	case "transform_it":
		return abi.Verdict{Kind: abi.VerdictTransform, By: "mixed",
			Payload: abi.TransformPayload{NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"redacted":true}`)}}}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "mixed"}
}
func (mixedAdj) Caps() []abi.Capability { return nil }

type echoEngine struct{ n int64 }

func (e *echoEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	atomic.AddInt64(&e.n, 1)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (*echoEngine) Caps() []abi.Capability { return nil }

// hitFP serves exactly one tool name from the vDSO so the rest fall through to
// adjudication.
type hitFP struct{ tool string }

func (f hitFP) Lookup(_ context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	if c.Tool == f.tool {
		return &abi.Result{Call: c, Status: abi.StatusOK, Meta: map[string]string{"served_by": "vdso"}}, true
	}
	return nil, false
}
func (hitFP) Caps() []abi.Capability { return nil }

func call(tool string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
}

// registerChain wires an isolated adjudication chain on the global ABI registry.
func registerChain() {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, mixedAdj{})
	abi.RegisterEngine("e", &echoEngine{})
	abi.RegisterFastPath(1, hitFP{tool: "vdso_hit"})
}

// sumByKind totals the observer's buckets whose Kind == kind.
func sumByKind(rows []DecisionRow, kind string) int64 {
	var t int64
	for _, r := range rows {
		if r.Kind == kind {
			t += r.Count
		}
	}
	return t
}

func sumNonVDSOByKind(rows []DecisionRow, kind string) int64 {
	var t int64
	for _, r := range rows {
		if r.Rung != "vdso" && r.Kind == kind {
			t += r.Count
		}
	}
	return t
}

func vdsoCount(rows []DecisionRow) int64 {
	for _, r := range rows {
		if r.Rung == "vdso" {
			return r.Count
		}
	}
	return 0
}

// AC #1 — the per-rung histogram reconciles EXACTLY with kernel.Counters() for the
// same stream: grand total == Submits, the non-vDSO buckets sum to the adjudicated
// calls (Denies + Transforms + allows == Submits - VDSOHits), and each verdict kind
// reconciles with its flat counter. No double-count, no drop.
func TestRungObsReconcilesWithCounters(t *testing.T) {
	registerChain()
	obs := New()
	abi.RegisterEmitter(obs)

	k := kernel.New("e")
	// 3 allows, 2 denies, 1 transform, 1 vDSO-served (not adjudicated) = 7 submits.
	seq := []string{"allow_it", "allow_it", "allow_it", "deny_it", "deny_it", "transform_it", "vdso_hit"}
	for _, tool := range seq {
		k.Syscall(context.Background(), call(tool))
	}

	c := k.Counters()
	rows := obs.Snapshot()
	got := obs.Total()

	// Grand total counts every submit, including the vDSO-served one.
	if got != int64(c.Submits) {
		t.Fatalf("Total = %d, want %d (Submits)", got, c.Submits)
	}
	// vDSO bucket == VDSOHits.
	if v := vdsoCount(rows); v != c.VDSOHits {
		t.Errorf("vdso bucket = %d, want VDSOHits %d", v, c.VDSOHits)
	}
	// Non-vDSO decisions reconcile: Submits - VDSOHits == Denies + Transforms + allows.
	nonVdso := got - vdsoCount(rows)
	if nonVdso != int64(c.Submits-c.VDSOHits) {
		t.Errorf("non-vdso total = %d, want %d", nonVdso, c.Submits-c.VDSOHits)
	}
	// Each adjudicated verdict kind reconciles with its flat counter.
	if d := sumByKind(rows, "DENY"); d != c.Denies {
		t.Errorf("DENY buckets sum = %d, want Denies %d", d, c.Denies)
	}
	if tr := sumByKind(rows, "TRANSFORM"); tr != c.Transforms {
		t.Errorf("TRANSFORM buckets sum = %d, want Transforms %d", tr, c.Transforms)
	}
	wantAllow := int64(c.Submits - c.VDSOHits - c.Denies - c.Transforms)
	// Structural allows exclude the rung="vdso" bucket: a vDSO hit is bucketed
	// kind="ALLOW" (the vDSO verdict is an allow) but is not an adjudicated allow.
	allowStructural := sumByKind(rows, "ALLOW") - vdsoCount(rows)
	if allowStructural != wantAllow {
		t.Errorf("ALLOW buckets sum = %d, want allows %d", allowStructural, wantAllow)
	}
	if a := sumByKind(rows, "ALLOW"); a != wantAllow+c.VDSOHits {
		t.Errorf("all ALLOW buckets sum = %d, want allows+VDSOHits %d", a, wantAllow+c.VDSOHits)
	}
}

// AC #2 — for a call the observer attributes a winning rung to, that rung equals
// kernel.FoldExplain(...).Rungs[winner].Rung for the same call (the observer agrees
// with the canonical trace, because it re-folds the same global chain).
func TestRungObsAttributesWinningRung(t *testing.T) {
	registerChain()
	obs := New()
	abi.RegisterEmitter(obs)

	k := kernel.New("e")
	k.Syscall(context.Background(), call("allow_it"))

	tc := call("allow_it")
	_, d := kernel.FoldExplain(context.Background(), abi.AdjudicatorsFor(tc), tc)
	wantRung := ""
	for _, r := range d.Rungs {
		if r.Winner {
			wantRung = r.Rung
		}
	}
	if wantRung == "" {
		t.Fatalf("FoldExplain produced no winning rung for allow_it")
	}

	// The observer's only ALLOW row must carry exactly that winning rung.
	for _, r := range obs.Snapshot() {
		if r.Kind == "ALLOW" && r.Rung != wantRung {
			t.Errorf("observer attributed ALLOW to rung %q, FoldExplain winner is %q", r.Rung, wantRung)
		}
	}
}

// AC #3 — a vDSO-served call (EvVDSOHit, no adjudication ran) is counted under a
// distinct rung="vdso" bucket and never misattributed to a structural rung.
func TestRungObsVDSOBucketDistinct(t *testing.T) {
	registerChain()
	obs := New()
	abi.RegisterEmitter(obs)

	k := kernel.New("e")
	r, v := k.Syscall(context.Background(), call("vdso_hit"))
	if v.By != "vdso" {
		t.Fatalf("vdso_hit must be served by the fast path, got verdict.By=%q", v.By)
	}
	if r.Meta["served_by"] != "vdso" {
		t.Fatalf("vdso_hit was not served by the fast path: served_by=%q", r.Meta["served_by"])
	}
	if k.Counters().VDSOHits != 1 {
		t.Fatalf("VDSOHits = %d, want 1", k.Counters().VDSOHits)
	}

	rows := obs.Snapshot()
	if v := vdsoCount(rows); v != 1 {
		t.Errorf("vdso bucket = %d, want 1", v)
	}
	// The mixedAdj structural rung type must never carry the vDSO call.
	for _, row := range rows {
		if row.Rung != "vdso" {
			t.Errorf("vDSO-served call leaked into structural rung %q (kind=%s)", row.Rung, row.Kind)
		}
	}
}

// AC #4 — registering the observer adds 0 allocations to EmittersFor for an event
// kind it does NOT subscribe to (it scopes itself via Subscriptions to Dec/Deny/
// VDSOHit only), mirroring TestEmittersForZeroAlloc. It also proves the selective
// fan-out: the observer receives its subscribed kinds and is excluded from the rest.
func TestRungObsZeroAllocOnUnsubscribedKind(t *testing.T) {
	abi.ResetForTest()
	obs := New()
	abi.RegisterEmitter(obs)

	for _, kind := range []abi.EventKind{abi.EvDecide, abi.EvDeny, abi.EvVDSOHit} {
		if !contains(abi.EmittersFor(kind), obs) {
			t.Errorf("observer missing from EmittersFor(subscribed kind %d)", kind)
		}
	}
	for _, kind := range []abi.EventKind{abi.EvSubmit, abi.EvDispatch, abi.EvComplete} {
		if contains(abi.EmittersFor(kind), obs) {
			t.Errorf("observer must NOT appear in EmittersFor(unsubscribed kind %d)", kind)
		}
	}
	// The fan-out itself stays allocation-free for an unsubscribed kind.
	if a := testing.AllocsPerRun(200, func() { _ = len(abi.EmittersFor(abi.EvSubmit)) }); a != 0 {
		t.Errorf("EmittersFor(EvSubmit) allocates %.2f/op; want 0", a)
	}
}

// AC #6 — the decide/deny hot path is byte-for-byte unchanged with the observer
// present vs absent: identical kernel.Counters() and identical per-call verdicts.
// The observer is passive by construction (it only reads); this proves it.
func TestObserverIsPassive(t *testing.T) {
	seq := []string{"allow_it", "deny_it", "transform_it", "allow_it", "vdso_hit"}

	// Phase A: observer registered.
	registerChain()
	obsA := New()
	abi.RegisterEmitter(obsA)
	cntA, vrdA := runSeq(kernel.New("e"), seq)

	// Phase B: same chain, NO observer.
	registerChain()
	cntB, vrdB := runSeq(kernel.New("e"), seq)

	if cntA != cntB {
		t.Errorf("Counters differ with observer present:\n  with:    %+v\n  without: %+v", cntA, cntB)
	}
	if len(vrdA) != len(vrdB) {
		t.Fatalf("verdict count differs: %d vs %d", len(vrdA), len(vrdB))
	}
	for i := range vrdA {
		if !reflect.DeepEqual(vrdA[i], vrdB[i]) {
			t.Errorf("verdict[%d] differs with observer:\n  with:    %+v\n  without: %+v", i, vrdA[i], vrdB[i])
		}
	}
	// And the observer still saw the stream (sanity: it counted every submit).
	if got, want := obsA.Total(), int64(len(seq)); got != want {
		t.Errorf("observer Total = %d, want %d", got, want)
	}
}

func runSeq(k *kernel.Kernel, seq []string) (kernel.Counters, []abi.Verdict) {
	vers := make([]abi.Verdict, 0, len(seq))
	for _, tool := range seq {
		_, v := k.Syscall(context.Background(), call(tool))
		vers = append(vers, v)
	}
	return k.Counters(), vers
}

func contains(es []abi.Emitter, want abi.Emitter) bool {
	for _, e := range es {
		if e == want {
			return true
		}
	}
	return false
}
