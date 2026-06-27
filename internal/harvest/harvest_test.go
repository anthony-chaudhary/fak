package harvest

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// ---- test doubles (a minimal kernel harness, local to this leaf) -------------

type inlineRes struct{}

func (inlineRes) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) { return r.Inline, nil }
func (inlineRes) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...), Len: int64(len(b))}, nil
}

type inlineBackend struct{}

func (inlineBackend) Resolver() abi.Resolver { return inlineRes{} }
func (inlineBackend) Caps() []abi.Capability { return nil }

type countEngine struct{ n int64 }

func (e *countEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.n++
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (e *countEngine) Caps() []abi.Capability { return nil }

// denyTool denies one named tool (POLICY_BLOCK) and allows everything else.
type denyTool struct{ deny string }

func (d denyTool) Caps() []abi.Capability { return nil }
func (d denyTool) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c.Tool == d.deny {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}

func inlineArgs(s string) abi.Ref { return abi.Ref{Kind: abi.RefInline, Inline: []byte(s)} }
func newKernel(t *testing.T, id string) *kernel.Kernel {
	t.Helper()
	return kernel.New(id)
}

// TestHarvestsDeniesAsLabels — a Deny event becomes a positive LabelRow carrying the
// verdict kind + reason; an Allow does not (it would be a negative, only collected
// on an explicit decide event).
func TestHarvestsDeniesAsLabels(t *testing.T) {
	c := NewCorpus()
	h := New(c)
	call := &abi.ToolCall{Tool: "send_email", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"x@evil.com"}`)}}
	deny := &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "ifc-sink"}
	h.Emit(abi.Event{Kind: abi.EvDeny, Call: call, Verdict: deny})

	rows := c.Positives()
	if len(rows) != 1 {
		t.Fatalf("expected 1 positive label, got %d", len(rows))
	}
	if rows[0].Verdict != abi.VerdictDeny || rows[0].Reason != abi.ReasonTrustViolation {
		t.Fatalf("label did not carry the verdict/reason: %+v", rows[0])
	}
	if rows[0].CallHash == "" {
		t.Fatal("label must carry a stable call identity")
	}
}

// A denied call must contribute ONE training row, not two. The kernel pairs an
// EvDecide(DENY) with a dedicated EvDeny for the same call (see kernel.Decide /
// kernel.Submit); deriving a LabelRow from both would double every catch in the
// corpus. This reproduces the exact emit pair and asserts a single positive.
func TestDeniedCallHarvestedOnce(t *testing.T) {
	c := NewCorpus()
	h := New(c)
	call := &abi.ToolCall{Tool: "send_email", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"x@evil.com"}`)}}
	deny := &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "ifc-sink"}
	h.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: deny})
	h.Emit(abi.Event{Kind: abi.EvDeny, Call: call, Verdict: deny})

	if rows := c.Positives(); len(rows) != 1 {
		t.Fatalf("denied call harvested %d rows, want exactly 1: %+v", len(rows), rows)
	}
}

func TestHarvestsResultDeniesAsLabels(t *testing.T) {
	c := NewCorpus()
	h := New(c)
	call := &abi.ToolCall{Tool: "read_webpage", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.com"}`)}}
	h.Emit(abi.Event{
		Kind:    abi.EvResultDeny,
		Call:    call,
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonUnwitnessed, By: "result-admit"},
	})

	rows := c.Positives()
	if len(rows) != 1 {
		t.Fatalf("expected 1 positive result-deny label, got %d", len(rows))
	}
	if rows[0].Verdict != abi.VerdictDeny || rows[0].Reason != abi.ReasonUnwitnessed {
		t.Fatalf("result-deny label did not carry the verdict/reason: %+v", rows[0])
	}
}

// TestExplicitLabelRowTakenVerbatim — the pre-flight ladder's typed EvRungLabel
// (RungPassed/RungFailed) is collected verbatim, and surfaces as a HARD NEGATIVE.
func TestExplicitLabelRowTakenVerbatim(t *testing.T) {
	c := NewCorpus()
	h := New(c)
	lr := &abi.LabelRow{CallHash: "k", RungPassed: 2, RungFailed: 4,
		Verdict: abi.VerdictDeny, Reason: abi.ReasonMalformed}
	h.Emit(abi.Event{Kind: abi.EvRungLabel, Label: lr})

	if c.Len() != 1 {
		t.Fatalf("explicit label must be collected, got %d rows", c.Len())
	}
	hn := c.HardNegatives()
	if len(hn) != 1 || hn[0].RungPassed != 2 || hn[0].RungFailed != 4 {
		t.Fatalf("a passed-cheap-failed-expensive row must be a hard negative, got %+v", hn)
	}
}

// TestCompiledLoopDataPath is the integration: drive the kernel over a benign call
// and an attacker exfil, with the harvester attached as an Emitter. The corpus must
// capture the exfil DENY as a labeled training example — the defender-side loop's
// data path (attack -> kernel verdict -> labeled corpus) end to end.
func TestCompiledLoopDataPath(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})

	corpus := NewCorpus()
	abi.RegisterEmitter(New(corpus))

	// a monitor that allows the planned read but denies the exfil tool.
	abi.RegisterAdjudicator(100, denyTool{deny: "send_email"})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)

	k := newKernel(t, "e")
	ctx := context.Background()

	k.Syscall(ctx, &abi.ToolCall{Tool: "search_flights", Args: inlineArgs(`{}`)})     // allowed
	k.Syscall(ctx, &abi.ToolCall{Tool: "send_email", Args: inlineArgs(`{"to":"a"}`)}) // denied

	pos := corpus.Positives()
	if len(pos) == 0 {
		t.Fatal("the exfil deny must be harvested as a positive label")
	}
	found := false
	for _, r := range pos {
		if r.Reason == abi.ReasonPolicyBlock || r.Verdict == abi.VerdictDeny {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a deny label in the corpus, got %+v", pos)
	}
	if by := corpus.ByReason(); len(by) == 0 {
		t.Fatal("the corpus must tally catches by reason")
	}
}
