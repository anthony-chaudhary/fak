package gateway

// The served-inline seams (vDSO probe and toolproc reuse) answer a proposed call
// locally. A cached answer warmed under an earlier policy, or by another server's
// chain, must never reach the model unless THIS server's current adjudicator
// chain admits the call now.

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// toggleDenyAdj models a policy reload: it allows until deny is set, then
// refuses one tool.
type toggleDenyAdj struct {
	tool string
	deny *atomic.Bool
}

func (toggleDenyAdj) Caps() []abi.Capability { return nil }
func (a toggleDenyAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if a.deny.Load() && c != nil && c.Tool == a.tool {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "test-deny-reload"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test-allow"}
}

func assertServedRefused(t *testing.T, label string, kept []agent.ToolCall, adjs []ToolAdjudication, dropped, servedHits int, servedText string) {
	t.Helper()
	if servedHits != 0 || servedText != "" {
		t.Fatalf("%s: denied call was served inline (servedHits=%d text=%q)", label, servedHits, servedText)
	}
	if len(kept) != 0 || dropped != 1 {
		t.Fatalf("%s: denied call must be dropped, got kept=%d dropped=%d", label, len(kept), dropped)
	}
	for _, a := range adjs {
		if a.Admitted || a.Verdict.Kind != "DENY" {
			t.Fatalf("%s: adjudication must be a DENY, got %+v", label, a)
		}
	}
}

func newVDSODenyReloadServer(t *testing.T, tool string, deny *atomic.Bool) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toggleDenyAdj{tool: tool, deny: deny})
	v := vdso.New(vdso.DefaultCacheSize)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func TestServedInlineVDSOHitRefusedAfterDenyReload(t *testing.T) {
	const tool, args = "get_doc", `{"id":"refund-policy"}`
	var deny atomic.Bool
	srv := newVDSODenyReloadServer(t, tool, &deny)
	ctx := context.Background()

	servedRead(t, srv, tool, args, "agent-A")
	call := []agent.ToolCall{{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: args}}}
	if _, _, _, _, hits := srv.adjudicateProposedServed(ctx, call, "agent-B"); hits != 1 {
		t.Fatalf("positive control: warmed read must be served inline under the allowing policy, servedHits=%d", hits)
	}

	deny.Store(true)
	kept, adjs, dropped, text, hits := srv.adjudicateProposedServed(ctx, call, "agent-B")
	assertServedRefused(t, "vdso after deny reload", kept, adjs, dropped, hits, text)
}

func TestServedInlineVDSOHitHonorsPerServerAdjudicators(t *testing.T) {
	const tool, args = "get_doc", `{"id":"refund-policy"}`
	var never, always atomic.Bool
	always.Store(true)
	srv := newVDSODenyReloadServer(t, tool, &never)
	ctx := context.Background()
	servedRead(t, srv, tool, args, "agent-A")

	denying, err := New(Config{EngineID: "test", Model: "test", VDSO: true,
		Adjudicators: []abi.Adjudicator{toggleDenyAdj{tool: tool, deny: &always}}})
	if err != nil {
		t.Fatalf("New denying server: %v", err)
	}
	t.Cleanup(denying.Close)

	call := []agent.ToolCall{{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: args}}}
	kept, adjs, dropped, text, hits := denying.adjudicateProposedServed(ctx, call, "agent-B")
	assertServedRefused(t, "vdso on per-server denying chain", kept, adjs, dropped, hits, text)
}

func TestToolprocReuseHitRefusedAfterDenyReload(t *testing.T) {
	dir, name := writeReuseFile(t, "skill.md", "cached body\n")
	var deny atomic.Bool
	srv := newBareReuseServer(t)
	abi.RegisterAdjudicator(1, toggleDenyAdj{tool: "Read", deny: &deny})
	srv.SetToolprocReuse(toolproc.ArmedConfig{}, toolproc.FileDigest(dir))
	t.Cleanup(srv.DisarmToolprocReuse)
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`
	const body = `{"content":"cached body"}`

	if served, _ := reuseTurn(t, srv, ctx, "tr", "c1", "Read", args, body); served {
		t.Fatal("first read must not be served")
	}
	if served, _ := reuseTurn(t, srv, ctx, "tr", "c2", "Read", args, body); !served {
		t.Fatal("positive control: re-read must be served under the allowing policy")
	}

	deny.Store(true)
	call := []agent.ToolCall{{ID: "c3", Type: "function", Function: agent.Func{Name: "Read", Arguments: args}}}
	kept, adjs, dropped, text, hits := srv.adjudicateProposedServed(ctx, call, "tr")
	assertServedRefused(t, "reuse after deny reload", kept, adjs, dropped, hits, text)
}
