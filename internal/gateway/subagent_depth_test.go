package gateway

// subagent_depth_test.go — the #12631 witness: the depth-1 leaf subagent recursion
// cap is enforced at the gateway adjudication seam. A spawn-shaped tool proposed by
// a session whose child would exceed policy.SubagentDepthRule is refused with a
// structured SUBAGENT_DEPTH_EXCEEDED denial BEFORE any spend or backend forwarding.
//
// The tests drive the REAL adjudicateProposed seam and the REAL policy rule (not
// doubles), and assert the depth gate specifically (Reason == ReasonSubagentDepthExceeded)
// rather than the whole verdict, since the kernel may deny `task` for unrelated policy
// reasons.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// depthServer builds a server whose subagent_depth cap is 1, wiring the same abi
// fixture chain other gateway tests use.
func depthServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	rt := policy.Runtime{SubagentDepth: &policy.SubagentDepthRule{MaxDepth: 1}}
	abi.RegisterAdjudicator(0, adjudicator.New(adjudicator.Policy{}))
	srv, err := New(Config{EngineID: "mock", Model: "fixture", PolicyRuntime: &rt})
	if err != nil {
		t.Fatalf("New server: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func depthCall(id, tool string) agent.ToolCall {
	return agent.ToolCall{ID: id, Type: "function", Function: agent.Func{Name: tool, Arguments: `{}`}}
}

// TestGatewaySubagentDepthAdjudicationDepthZeroAllowsChild: a root coordinator
// (depth 0) may spawn a child under a cap of 1; the depth gate must not fire.
func TestGatewaySubagentDepthAdjudicationDepthZeroAllowsChild(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 0)
	_, adjs, _ := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("c1", "task")}, "trace-depth0")
	if len(adjs) != 1 {
		t.Fatalf("adjs = %d, want 1", len(adjs))
	}
	if adjs[0].Verdict.Reason == ReasonSubagentDepthExceeded {
		t.Fatalf("depth-0 child was refused by the depth gate: %+v", adjs[0].Verdict)
	}
}

// TestGatewaySubagentDepthAdjudicationDepthOneRefusesChild: a depth-1 worker's child
// would be depth 2 > cap 1, so the spawn must be refused and dropped.
func TestGatewaySubagentDepthAdjudicationDepthOneRefusesChild(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 1)
	kept, adjs, dropped := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("c1", "task")}, "trace-depth1")
	if len(kept) != 0 {
		t.Fatalf("kept = %d, want 0 (refused spawn must not reach the wire)", len(kept))
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1", dropped)
	}
	if len(adjs) != 1 {
		t.Fatalf("adjs = %d, want 1", len(adjs))
	}
	v := adjs[0].Verdict
	if adjs[0].Admitted {
		t.Fatalf("refused spawn marked admitted: %+v", adjs[0])
	}
	if v.Kind != "DENY" || v.Reason != ReasonSubagentDepthExceeded {
		t.Fatalf("verdict = %+v, want DENY/%s", v, ReasonSubagentDepthExceeded)
	}
	if v.By != "subagent-depth-cap" {
		t.Fatalf("By = %q, want subagent-depth-cap", v.By)
	}
	if v.Detail["parent_depth"] != "1" || v.Detail["cap"] != "1" {
		t.Fatalf("Detail = %v, want parent_depth=1 cap=1", v.Detail)
	}
}

// TestGatewaySubagentDepthAdjudicationRefusalPayloadStructure: the refusal must be a
// RETRYABLE, inspectable per-tool denial, not a bare string.
func TestGatewaySubagentDepthAdjudicationRefusalPayloadStructure(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 1)
	_, adjs, _ := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("c1", "task")}, "trace-payload")
	if len(adjs) != 1 || adjs[0].Verdict.Reason != ReasonSubagentDepthExceeded {
		t.Fatalf("adjs = %+v, want one SUBAGENT_DEPTH_EXCEEDED refusal", adjs)
	}
	v := adjs[0].Verdict
	if v.Disposition != "RETRYABLE" {
		t.Fatalf("Disposition = %q, want RETRYABLE", v.Disposition)
	}
	if len(v.Detail) == 0 {
		t.Fatalf("Detail is empty; refusal must be inspectable")
	}
	if _, ok := v.Detail["parent_depth"]; !ok {
		t.Fatalf("Detail missing parent_depth: %v", v.Detail)
	}
	if _, ok := v.Detail["cap"]; !ok {
		t.Fatalf("Detail missing cap: %v", v.Detail)
	}
}

// TestGatewaySubagentDepthAdjudicationNonSpawnToolUnaffected: the gate only fires on
// spawn-shaped tools; a normal tool at depth 5 is not refused for depth.
func TestGatewaySubagentDepthAdjudicationNonSpawnToolUnaffected(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 5)
	_, adjs, _ := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("c1", "fak_read")}, "trace-nonspawn")
	if len(adjs) != 1 {
		t.Fatalf("adjs = %d, want 1", len(adjs))
	}
	if adjs[0].Verdict.Reason == ReasonSubagentDepthExceeded {
		t.Fatalf("non-spawn tool was refused by the depth gate: %+v", adjs[0].Verdict)
	}
}

// TestGatewaySubagentDepthAdjudicationSpawnShapes: every recognized spawn shape is
// refused at depth 1 under a cap of 1.
func TestGatewaySubagentDepthAdjudicationSpawnShapes(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 1)
	for _, tool := range []string{"task", "spawn_agent", "subagent", "dispatch", "task_spawn"} {
		_, adjs, _ := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("c1", tool)}, "trace-shape-"+tool)
		if len(adjs) != 1 || adjs[0].Verdict.Reason != ReasonSubagentDepthExceeded {
			t.Fatalf("tool %q: verdict = %+v, want %s", tool, adjs, ReasonSubagentDepthExceeded)
		}
	}
}
