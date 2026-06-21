package kernel

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// scopedAdj is a CORRECT abi.CallScope rung: it returns its verdict only for the
// tools it claims and Defers (no opinion) for any other tool. That self-Defer is
// exactly the contract CallScope promises, so folding the full chain and folding
// only the per-tool chain must produce the same verdict.
type scopedAdj struct {
	v     abi.Verdict
	tools []string
}

func (s scopedAdj) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	for _, t := range s.tools {
		if t == c.Tool {
			return s.v
		}
	}
	return abi.Verdict{Kind: abi.VerdictDefer}
}
func (s scopedAdj) Caps() []abi.Capability { return nil }
func (s scopedAdj) Tools() []string        { return s.tools }

// TestScopedRungRoutesByTool is the fail-closed routing proof: a rung scoped to
// "write" with a DENY must deny a write call and must NOT affect a read call
// (which only the unconditional allow rung opines on). This proves the per-tool
// chain selection routes a restrictive rung to its tool without leaking it to
// others, and without dropping it.
func TestScopedRungRoutesByTool(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}) // unconditional
	abi.RegisterAdjudicator(50, scopedAdj{
		v:     abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock},
		tools: []string{"write"},
	})
	k := New("")

	if _, v := k.Syscall(context.Background(), call("write", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("write call: got %v, want Deny (scoped deny must route to its tool)", v.Kind)
	}
	if _, v := k.Syscall(context.Background(), call("read", "{}")); v.Kind != abi.VerdictAllow {
		t.Fatalf("read call: got %v, want Allow (scoped deny must NOT leak to other tools)", v.Kind)
	}
}

// scopedAdmitter is a CORRECT result-side CallScope gate: it returns its verdict
// only for its tools and Allow (admit-as-is, the fold identity) otherwise — the
// strict contract that makes skipping it for an unlisted tool verdict-equivalent.
type scopedAdmitter struct {
	v     abi.Verdict
	tools []string
}

func (s scopedAdmitter) Admit(_ context.Context, c *abi.ToolCall, _ *abi.Result) abi.Verdict {
	for _, t := range s.tools {
		if t == c.Tool {
			return s.v
		}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
}
func (s scopedAdmitter) Caps() []abi.Capability { return nil }
func (s scopedAdmitter) Tools() []string        { return s.tools }

// TestScopedResultAdmitterRoutesByTool is the result-side fail-closed routing proof
// (Fix D): a gate scoped to "fetch" with a QUARANTINE must quarantine a fetch
// result and must NOT touch a read result. The exfil/quarantine floor routes to its
// tool without leaking to others and without being dropped.
func TestScopedResultAdmitterRoutesByTool(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}) // allow so calls dispatch
	abi.RegisterEngine("", &countEngine{})
	abi.RegisterResultAdmitter(10, scopedAdmitter{
		v:     abi.Verdict{Kind: abi.VerdictQuarantine, By: "test"},
		tools: []string{"fetch"},
	})
	k := New("")

	r, _ := k.Syscall(context.Background(), call("fetch", "{}"))
	if r.Meta["admit"] != "quarantined" {
		t.Fatalf("fetch result admit=%q, want quarantined (scoped gate must route to its tool)", r.Meta["admit"])
	}
	r2, _ := k.Syscall(context.Background(), call("read", "{}"))
	if r2.Meta["admit"] == "quarantined" {
		t.Fatalf("read result was quarantined; scoped gate must NOT leak to other tools")
	}
}

// TestScopedFoldEquivalentToFullChain is the core safety proof: for a set of
// correctly-scoped rungs, the per-tool chain (what the kernel now folds) yields the
// IDENTICAL verdict to the full chain (what it folded before) for every tool. The
// optimization can never change a security decision.
func TestScopedFoldEquivalentToFullChain(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}) // unconditional allow
	abi.RegisterAdjudicator(10, scopedAdj{
		v:     abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock},
		tools: []string{"write", "delete"},
	})
	abi.RegisterAdjudicator(20, scopedAdj{
		v:     abi.Verdict{Kind: abi.VerdictQuarantine},
		tools: []string{"fetch"},
	})

	for _, tool := range []string{"read", "write", "delete", "fetch", "exec", ""} {
		c := call(tool, "{}")
		full := Fold(context.Background(), abi.Adjudicators(), c)
		scoped := Fold(context.Background(), abi.AdjudicatorsFor(c), c)
		if full.Kind != scoped.Kind {
			t.Errorf("tool %q: full-chain verdict=%v, per-tool verdict=%v (must be identical)",
				tool, full.Kind, scoped.Kind)
		}
	}
}
