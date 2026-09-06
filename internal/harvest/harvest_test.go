package harvest

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

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

func TestCompiledLoopDataPath(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})

	corpus := NewCorpus()
	abi.RegisterEmitter(New(corpus))

	abi.RegisterAdjudicator(100, denyTool{deny: "send_email"})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)

	k := newKernel(t, "e")
	ctx := context.Background()

	k.Syscall(ctx, &abi.ToolCall{Tool: "search_flights", Args: inlineArgs(`{}`)})
	k.Syscall(ctx, &abi.ToolCall{Tool: "send_email", Args: inlineArgs(`{"to":"a"}`)})

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

func TestCorpusRetentionBounded(t *testing.T) {
	c := NewCorpus()
	c.SetMaxRows(4)

	for i := 0; i < 100; i++ {
		c.add(abi.LabelRow{CallHash: fmt.Sprintf("call-%d", i), Verdict: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock})
	}
	if c.Len() != 4 {
		t.Fatalf("Len() = %d, want capped at 4 (the corpus grew unbounded)", c.Len())
	}
	rows := c.Rows()
	if rows[0].CallHash != "call-96" || rows[len(rows)-1].CallHash != "call-99" {
		t.Fatalf("retained window = [%s .. %s], want [call-96 .. call-99]", rows[0].CallHash, rows[len(rows)-1].CallHash)
	}
}

func TestCorpusUnboundedOptOut(t *testing.T) {
	c := NewCorpus()
	c.SetMaxRows(-1)
	const n = defaultMaxCorpusRows + 25
	for i := 0; i < n; i++ {
		c.add(abi.LabelRow{CallHash: fmt.Sprintf("call-%d", i), Verdict: abi.VerdictAllow})
	}
	if c.Len() != n {
		t.Fatalf("unbounded corpus retained %d, want all %d", c.Len(), n)
	}
}

func TestCorpusRetentionDynamicTrimAndRestoreDefault(t *testing.T) {
	c := NewCorpus()
	for i := 0; i < 10; i++ {
		c.add(abi.LabelRow{CallHash: fmt.Sprintf("call-%d", i), Verdict: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock})
	}
	if c.Len() != 10 {
		t.Fatalf("expected 10 rows, got %d", c.Len())
	}

	c.SetMaxRows(4)
	if c.Len() != 4 {
		t.Fatalf("expected 4 rows after downsize, got %d", c.Len())
	}
	rows := c.Rows()
	if rows[0].CallHash != "call-6" || rows[len(rows)-1].CallHash != "call-9" {
		t.Fatalf("expected [call-6..call-9], got [%s..%s]", rows[0].CallHash, rows[len(rows)-1].CallHash)
	}

	c.SetMaxRows(0)
	for i := 10; i < defaultMaxCorpusRows+50; i++ {
		c.add(abi.LabelRow{CallHash: fmt.Sprintf("call-%d", i), Verdict: abi.VerdictAllow})
	}
	if c.Len() != defaultMaxCorpusRows {
		t.Fatalf("expected default cap %d, got %d", defaultMaxCorpusRows, c.Len())
	}
}

func TestCorpusSnapshotIsolation(t *testing.T) {
	c := NewCorpus()
	c.add(abi.LabelRow{CallHash: "call-1", Verdict: abi.VerdictAllow})
	c.add(abi.LabelRow{CallHash: "call-2", Verdict: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock})

	snap1 := c.Rows()
	if len(snap1) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(snap1))
	}
	snap1[0].CallHash = "mutated"

	snap2 := c.Rows()
	if snap2[0].CallHash != "call-1" {
		t.Fatalf("corpus internal state mutated via returned slice: got %s", snap2[0].CallHash)
	}
}

func TestHardNegativesInvariant(t *testing.T) {
	c := NewCorpus()
	cases := []struct {
		passed int
		failed int
		wantHN bool
	}{
		{passed: -1, failed: -1, wantHN: false},
		{passed: 2, failed: 2, wantHN: false},
		{passed: 3, failed: 1, wantHN: false},
		{passed: 0, failed: 1, wantHN: true},
		{passed: 1, failed: 3, wantHN: true},
	}
	for i, tc := range cases {
		c.add(abi.LabelRow{
			CallHash:   fmt.Sprintf("call-%d", i),
			RungPassed: tc.passed,
			RungFailed: tc.failed,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonMalformed,
		})
	}

	hn := c.HardNegatives()
	if len(hn) != 2 {
		t.Fatalf("expected 2 hard negatives, got %d", len(hn))
	}
	for _, r := range hn {
		if r.RungPassed < 0 || r.RungFailed <= r.RungPassed {
			t.Fatalf("violates hard negative invariant: passed=%d failed=%d", r.RungPassed, r.RungFailed)
		}
	}
}

func TestEmitIgnoredEvents(t *testing.T) {
	c := NewCorpus()
	h := New(c)

	h.Emit(abi.Event{Kind: abi.EvDecide})
	h.Emit(abi.Event{Kind: abi.EvComplete})
	if c.Len() != 0 {
		t.Fatalf("expected 0 rows for unhandled events, got %d", c.Len())
	}

	h.Emit(abi.Event{
		Kind:    abi.EvDeny,
		Call:    nil,
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock},
	})
	if c.Len() != 1 {
		t.Fatalf("expected 1 row, got %d", c.Len())
	}
	if c.Rows()[0].CallHash != "" {
		t.Fatalf("expected empty CallHash for nil call, got %q", c.Rows()[0].CallHash)
	}
}

func TestCorpusConcurrentSafety(t *testing.T) {
	c := NewCorpus()
	h := New(c)
	const goroutines = 8
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(goroutines * 3)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				h.Emit(abi.Event{
					Kind:    abi.EvDeny,
					Call:    &abi.ToolCall{Tool: "tool", Args: abi.Ref{Kind: abi.RefInline, Digest: fmt.Sprintf("d-%d-%d", id, i)}},
					Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation},
				})
			}
		}(g)

		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				_ = c.Rows()
				_ = c.Positives()
				_ = c.HardNegatives()
				_ = c.ByReason()
				_ = c.Len()
			}
		}()

		go func(id int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				if i%20 == 0 {
					c.SetMaxRows(50 + (i % 100))
				}
			}
		}(g)
	}

	wg.Wait()
	if c.Len() == 0 {
		t.Fatal("expected non-empty corpus after concurrent run")
	}
}
