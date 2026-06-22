package kernel

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// callInline builds a ToolCall whose args ride INLINE, so FoldExplain's digest +
// transform-diff resolve without a registered RegionBackend (the explain tests
// are pure, no kernel/resolver wiring).
func callInline(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Len: int64(len(args))}}
}

// transformAdj rewrites Args to a fixed inline payload (a stand-in for the
// monitor's secret-redaction TRANSFORM), so the changedKeys diff is exercised.
type transformAdj struct{ newArgs string }

func (t transformAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictTransform, By: "monitor",
		Payload: abi.TransformPayload{NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(t.newArgs), Len: int64(len(t.newArgs))}}}
}
func (transformAdj) Caps() []abi.Capability { return nil }

// TestFoldExplainParity is the load-bearing invariant: FoldExplain MUST resolve
// to the byte-identical verdict Fold does — the trace is forensic surplus, never
// a behavior change. We assert it across the lattice corners.
func TestFoldExplainParity(t *testing.T) {
	ctx := context.Background()
	deny := abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}
	allow := abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
	defer_ := abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}
	xform := abi.Verdict{Kind: abi.VerdictTransform, By: "monitor"}

	cases := map[string][]abi.Adjudicator{
		"empty":          nil,
		"all-defer":      {fakeAdj{defer_}, fakeAdj{defer_}},
		"allow":          {fakeAdj{defer_}, fakeAdj{allow}},
		"deny-wins":      {fakeAdj{allow}, fakeAdj{deny}},  // deny (100) beats allow (0)
		"deny-tie":       {fakeAdj{deny}, fakeAdj{deny}},   // first at max rank wins
		"xform-vs-deny":  {fakeAdj{xform}, fakeAdj{deny}},  // deny (100) beats transform (2)
		"xform-vs-allow": {fakeAdj{allow}, fakeAdj{xform}}, // transform (2) beats allow (0)
	}
	for name, chain := range cases {
		want := Fold(ctx, chain, callInline("t", "{}"))
		got, _ := FoldExplain(ctx, chain, callInline("t", "{}"))
		if got.Kind != want.Kind || got.Reason != want.Reason || got.By != want.By {
			t.Errorf("%s: FoldExplain verdict {%s,%s,%s} != Fold {%s,%s,%s}",
				name, kindName(got.Kind), abi.ReasonName(got.Reason), got.By,
				kindName(want.Kind), abi.ReasonName(want.Reason), want.By)
		}
	}
}

// TestFoldExplainRungCapture checks the trace records every rung, flags the ones
// that deferred, and marks exactly the winner.
func TestFoldExplainRungCapture(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "preflight"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}},
	}
	v, d := FoldExplain(ctx, chain, callInline("refund", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("verdict = %s, want DENY", kindName(v.Kind))
	}
	if len(d.Rungs) != 3 {
		t.Fatalf("rungs = %d, want 3", len(d.Rungs))
	}
	if !d.Rungs[0].Deferred || !d.Rungs[1].Deferred {
		t.Errorf("first two rungs should be Deferred, got %+v / %+v", d.Rungs[0], d.Rungs[1])
	}
	if d.Rungs[0].Winner || d.Rungs[1].Winner || !d.Rungs[2].Winner {
		t.Errorf("winner should be rung[2], got winners %v/%v/%v", d.Rungs[0].Winner, d.Rungs[1].Winner, d.Rungs[2].Winner)
	}
	if d.Rungs[2].Reason != "POLICY_BLOCK" || d.By != "monitor" || d.Disposition == "" {
		t.Errorf("decision summary wrong: reason=%q by=%q disp=%q", d.Reason, d.By, d.Disposition)
	}
}

// TestFoldExplainTieFirstWins mirrors Fold's tie-break: the FIRST rung to reach
// the max rank wins, so an earlier deny owns the verdict even when a later rung
// also denies (this is what makes a preflight MALFORMED deny shadow a later
// monitor DEFAULT_DENY — the exact nuance the trace exists to reveal).
func TestFoldExplainTieFirstWins(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMalformed, By: "preflight"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "monitor"}},
	}
	_, d := FoldExplain(ctx, chain, callInline("t", "{}"))
	if !d.Rungs[0].Winner || d.Rungs[1].Winner {
		t.Fatalf("first deny should win the tie, got winners %v/%v", d.Rungs[0].Winner, d.Rungs[1].Winner)
	}
	if d.By != "preflight" || d.Reason != "MALFORMED" {
		t.Errorf("winner should be preflight/MALFORMED, got %s/%s", d.By, d.Reason)
	}
}

// TestFoldExplainEmptyAndAllDefer covers the two synthesized fail-closed denies
// (no rung produced the verdict, so none is marked winner).
func TestFoldExplainEmptyAndAllDefer(t *testing.T) {
	ctx := context.Background()

	_, empty := FoldExplain(ctx, nil, callInline("t", "{}"))
	if empty.Verdict != "DENY" || empty.By != "empty-policy" || len(empty.Rungs) != 0 {
		t.Errorf("empty chain: got verdict=%s by=%s rungs=%d", empty.Verdict, empty.By, len(empty.Rungs))
	}
	if !strings.Contains(empty.Explanation, "default deny") {
		t.Errorf("empty chain explanation should mention default deny: %q", empty.Explanation)
	}

	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "a"}}}
	_, ad := FoldExplain(ctx, chain, callInline("t", "{}"))
	if ad.Verdict != "DENY" || ad.By != "all-defer" {
		t.Errorf("all-defer: got verdict=%s by=%s", ad.Verdict, ad.By)
	}
	for _, r := range ad.Rungs {
		if r.Winner {
			t.Errorf("all-defer should mark no rung winner, got %+v", r)
		}
	}
}

// TestFoldExplainArgsNoLeak is the safe-to-log guarantee: the Decision carries a
// digest, never the raw args — a secret in the args must NOT appear in the JSON
// or text rendering.
func TestFoldExplainArgsNoLeak(t *testing.T) {
	ctx := context.Background()
	const secret = "supersecret-DO-NOT-LEAK"
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}}}
	_, d := FoldExplain(ctx, chain, callInline("t", `{"password":"`+secret+`"}`))
	if d.ArgsDigest == "" || d.ArgsBytes == 0 {
		t.Fatalf("digest/bytes not captured: %+v", d)
	}
	if strings.Contains(d.JSON(), secret) || strings.Contains(d.Text(), secret) {
		t.Errorf("Decision leaked the raw args secret into its rendering")
	}
}

// TestFoldExplainTransformRedaction checks the TRANSFORM path reports which arg
// keys the rung rewrote (the redaction diff), without leaking values.
func TestFoldExplainTransformRedaction(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{transformAdj{newArgs: `{"q":"NYC","password":"[REDACTED]"}`}}
	_, d := FoldExplain(ctx, chain, callInline("search", `{"q":"NYC","password":"hunter2"}`))
	if d.Verdict != "TRANSFORM" {
		t.Fatalf("verdict = %s, want TRANSFORM", d.Verdict)
	}
	if len(d.Redacted) != 1 || d.Redacted[0] != "password" {
		t.Errorf("redacted = %v, want [password]", d.Redacted)
	}
	if !strings.Contains(d.Explanation, "password") {
		t.Errorf("explanation should name the rewritten key: %q", d.Explanation)
	}
	if strings.Contains(d.JSON(), "hunter2") {
		t.Errorf("transform diff leaked the original secret value")
	}
}

// TestFoldExplainClaim checks the bounded-disclosure witness rides through to the
// Decision summary and the rung row.
func TestFoldExplainClaim(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{
		Kind: abi.VerdictDeny, Reason: abi.ReasonSelfModify, By: "monitor",
		Payload: abi.WitnessPayload{Claim: "internal/abi/"},
	}}}
	_, d := FoldExplain(ctx, chain, callInline("write_file", `{"path":"internal/abi/x.go"}`))
	if d.Claim != "internal/abi/" {
		t.Errorf("decision claim = %q, want internal/abi/", d.Claim)
	}
	if d.Rungs[0].Claim != "internal/abi/" {
		t.Errorf("rung claim = %q, want internal/abi/", d.Rungs[0].Claim)
	}
	if !strings.Contains(d.Text(), "internal/abi/") {
		t.Errorf("text rendering should disclose the offending claim")
	}
}

// TestDecisionTextSmoke checks the human renderer includes the chain and marks
// the winner.
func TestDecisionTextSmoke(t *testing.T) {
	ctx := context.Background()
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}},
	}
	_, d := FoldExplain(ctx, chain, callInline("refund", "{}"))
	txt := d.Text()
	for _, want := range []string{"decision chain", "=>", "winner", "POLICY_BLOCK", "explanation:"} {
		if !strings.Contains(txt, want) {
			t.Errorf("Text() missing %q\n%s", want, txt)
		}
	}
}
