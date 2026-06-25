package kernel

import (
	"context"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type spyAdj struct {
	v abi.Verdict
	n *int32
}

func (s spyAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	atomic.AddInt32(s.n, 1)
	return s.v
}
func (s spyAdj) Caps() []abi.Capability { return nil }

func legacyFoldNoIndeterminate(ctx context.Context, chain []abi.Adjudicator, c *abi.ToolCall) abi.Verdict {
	if len(chain) == 0 {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "empty-policy"}
	}
	best := abi.Verdict{Kind: abi.VerdictDefer, By: "no-link"}
	bestRank := -1
	sawNonDefer := false
	for _, a := range chain {
		v := a.Adjudicate(ctx, c)
		if v.Kind == abi.VerdictDefer {
			continue
		}
		sawNonDefer = true
		if r := abi.FoldRank(v.Kind); r > bestRank {
			bestRank, best = r, v
		}
	}
	if !sawNonDefer {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "all-defer"}
	}
	return best
}

func TestFoldNoIndeterminateMatchesLegacy(t *testing.T) {
	ctx := context.Background()
	c := call("x", "{}")
	verdicts := []abi.Verdict{
		{Kind: abi.VerdictAllow, By: "allow"},
		{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny"},
		{Kind: abi.VerdictTransform, Payload: abi.TransformPayload{NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"ok":true}`)}}, By: "transform"},
		{Kind: abi.VerdictQuarantine, By: "quarantine"},
		{Kind: abi.VerdictRequireWitness, By: "witness"},
		{Kind: abi.VerdictDefer, By: "defer"},
	}

	var check func(depth, target int, chain []abi.Adjudicator, names []string)
	check = func(depth, target int, chain []abi.Adjudicator, names []string) {
		if depth == target {
			got := Fold(ctx, chain, c)
			want := legacyFoldNoIndeterminate(ctx, chain, c)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("chain %v: Fold = %+v, legacy = %+v", names, got, want)
			}
			return
		}
		for _, v := range verdicts {
			check(depth+1, target, append(chain, fakeAdj{v: v}), append(names, kindName(v.Kind)))
		}
	}
	for n := 0; n <= 5; n++ {
		check(0, n, nil, nil)
	}
}

func TestFoldDenyShortCircuitsTail(t *testing.T) {
	var allowN, denyN, tailN int32
	chain := []abi.Adjudicator{
		spyAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "allow"}, n: &allowN},
		spyAdj{v: abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny"}, n: &denyN},
		spyAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "tail"}, n: &tailN},
	}
	v := Fold(context.Background(), chain, call("x", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Fold kind = %v, want Deny", v.Kind)
	}
	if atomic.LoadInt32(&allowN) != 1 || atomic.LoadInt32(&denyN) != 1 || atomic.LoadInt32(&tailN) != 0 {
		t.Fatalf("call counts allow=%d deny=%d tail=%d, want 1/1/0", allowN, denyN, tailN)
	}
}

func TestFoldExplainShortCircuitMatchesFold(t *testing.T) {
	var tailN int32
	explainChain := []abi.Adjudicator{
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "allow"}},
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny"}},
		spyAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "tail"}, n: &tailN},
	}
	foldChain := []abi.Adjudicator{
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "allow"}},
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny"}},
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "tail"}},
	}

	v, d := FoldExplain(context.Background(), explainChain, callInline("x", "{}"))
	want := Fold(context.Background(), foldChain, call("x", "{}"))
	if !reflect.DeepEqual(v, want) {
		t.Fatalf("FoldExplain verdict = %+v, Fold = %+v", v, want)
	}
	// The short-circuited tail is RECORDED as elided (#668) — it still was NOT
	// evaluated (tailN stays 0), so the trace shows all three rungs: 2 run + 1 elided.
	if len(d.Rungs) != 3 {
		t.Fatalf("FoldExplain rungs = %d, want 3 (2 run + 1 elided)", len(d.Rungs))
	}
	if atomic.LoadInt32(&tailN) != 0 {
		t.Fatalf("short-circuited tail was called %d times", tailN)
	}
	if !d.Rungs[1].Winner || d.Rungs[1].Kind != "DENY" {
		t.Fatalf("winning rung = %+v, want rung[1] DENY", d.Rungs[1])
	}
	if !d.Rungs[2].Elided {
		t.Fatalf("short-circuited tail rung should be marked Elided, got %+v", d.Rungs[2])
	}
}

func TestFoldResidualIndeterminateFailsClosed(t *testing.T) {
	chain := []abi.Adjudicator{
		fakeAdj{v: abi.Verdict{Kind: abi.VerdictIndeterminate, By: "cheap"}},
	}
	v := Fold(context.Background(), chain, call("x", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("residual Indeterminate folded to %v, want Deny", v.Kind)
	}
	if v.Meta["fold"] != "indeterminate" {
		t.Fatalf("fold meta = %v, want indeterminate marker", v.Meta)
	}
}

func TestFoldIndeterminateLosesToConclusiveVerdict(t *testing.T) {
	for name, conclusive := range map[string]abi.Verdict{
		"allow":     {Kind: abi.VerdictAllow, By: "allow"},
		"deny":      {Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny"},
		"transform": {Kind: abi.VerdictTransform, Payload: abi.TransformPayload{NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"redacted":true}`)}}, By: "transform"},
	} {
		t.Run(name, func(t *testing.T) {
			chain := []abi.Adjudicator{
				fakeAdj{v: abi.Verdict{Kind: abi.VerdictIndeterminate, By: "cheap"}},
				fakeAdj{v: conclusive},
			}
			v := Fold(context.Background(), chain, call("x", "{}"))
			if v.Kind != conclusive.Kind {
				t.Fatalf("Fold kind = %v, want %v", v.Kind, conclusive.Kind)
			}
		})
	}
}

func TestSubmitHoldsResidualIndeterminate(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{v: abi.Verdict{Kind: abi.VerdictIndeterminate, By: "cheap"}})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")

	r, v := k.Syscall(context.Background(), call("x", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Submit verdict = %v, want Deny", v.Kind)
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("residual Indeterminate must not dispatch (engine n=%d)", eng.n)
	}
	if r == nil || r.Status != abi.StatusError {
		t.Fatalf("denied call result = %+v, want StatusError", r)
	}
}
