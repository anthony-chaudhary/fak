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
	"github.com/anthony-chaudhary/fak/internal/vdso"
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

// Admission distinguishes child creation from child inspection; the fold witness
// deliberately remains conservative for both families' unwitnessed done claims.
// fak-test:runtime fast est=100ms lane=default
func TestGatewaySubagentDepthAdjudicationOperationNames(t *testing.T) {
	srv := depthServer(t)
	ctx := WithSessionDepth(context.Background(), 1)
	for _, tc := range []struct {
		tool  string
		spawn bool
	}{
		{"Task", true},
		{"Agent", true},
		{"Workflow", true},
		{"subagent", true},
		{"dispatch", true},
		{"spawn", true},
		{"task_spawn", true},
		{"spawn_task", true},
		{"spawn_agent", true},
		{"spawn_subagent", true},
		{"agent_spawn", true},
		{"subagent_spawn", true},
		{"create_agent", true},
		{"create_subagent", true},
		{"functions.spawn_agent", true},
		{"agent.task_spawn", true},
		{"mcp__workers__spawn_agent", true},
		{"mcp:workers:create_subagent", true},
		{"get_worker.spawn_agent", true},
		{" TASK ", true},
		{"list_agents", false},
		{"get_task", false},
		{"get_subagent_result", false},
		{"search_dispatches", false},
		{"dispatching_help", false},
		{"task_wait", false},
		{"task_status", false},
		{"task_cancel", false},
		{"TaskCreate", false},
		{"TaskGet", false},
		{"wait_agent", false},
		{"close_agent", false},
		{"respawn_status", false},
		{"functions.get_task", false},
		{"mcp__agents__get_config", false},
		{"mcp:tasks:get_config", false},
		{"agents.get_config", false},
	} {
		t.Run(tc.tool, func(t *testing.T) {
			kept, adjs, dropped := srv.adjudicateProposed(ctx, []agent.ToolCall{depthCall("op", tc.tool)}, "depth-operation-"+tc.tool)
			if len(adjs) != 1 {
				t.Fatalf("adjudications = %+v, want one", adjs)
			}
			v := adjs[0].Verdict
			if refused := v.Reason == ReasonSubagentDepthExceeded; refused != tc.spawn {
				t.Fatalf("depth refusal = %v, want %v; verdict = %+v", refused, tc.spawn, v)
			}
			if tc.spawn && (len(kept) != 0 || dropped != 1 || adjs[0].Admitted || v.Kind != "DENY" || v.Disposition != "RETRYABLE") {
				t.Fatalf("spawn escaped depth admission: kept=%v dropped=%d adjudication=%+v", kept, dropped, adjs[0])
			}
		})
	}
	if _, tainted := subagentDoneVerdict("list_agents", "I shipped the change.", WireVerdict{Kind: "ALLOW"}); !tainted {
		t.Fatal("the conservative fold witness must still taint an unbacked look-alike claim")
	}
	// A policy-backed single-underscore alias must retain the same classification
	// as buildCall. An explicitly declared get_task is not an alias for task.
	srv.SetPolicyRuntime(&policy.Runtime{
		SubagentDepth: &policy.SubagentDepthRule{MaxDepth: 1},
		Adjudicator: adjudicator.Policy{Allow: map[string]bool{
			"task": true, "spawn_agent": true, "get_task": true,
		}},
	})
	for tool, want := range map[string]bool{"worker_task": true, "get_spawn_agent": true, "get_task": false} {
		if verdict, refused := srv.subagentDepthRefusal(ctx, tool); refused != want {
			t.Errorf("policy alias %q: depth refusal=%v, want %v; verdict=%+v", tool, refused, want, verdict)
		}
	}
}

// Exercise a real vDSO hit rather than only the served path's miss fall-through.
// A read-shaped namespace must not hide a spawn, but an actual read stays usable
// at the cap. Warming uses the fixture echo engine, never a child-task executor.
// fak-test:runtime fast est=100ms lane=default
func TestGatewaySubagentDepthAdjudicationServedOperationNames(t *testing.T) {
	installEmptyTestLeaseAuthority(t)
	for _, tc := range []struct {
		tool  string
		spawn bool
	}{
		{"list_agents", false},
		{"get_task", false},
		{"get_worker.spawn_agent", true},
		{"get_spawn_agent", true},
	} {
		t.Run(tc.tool, func(t *testing.T) {
			srv, _ := newSharingServer(t, vdso.Global)
			srv.SetPolicyRuntime(&policy.Runtime{
				SubagentDepth: &policy.SubagentDepthRule{MaxDepth: 1},
				Adjudicator:   adjudicator.Policy{Allow: map[string]bool{"spawn_agent": true}},
			})
			const args = `{}`
			warmServedRead(t, srv, tc.tool, args)
			ctx := WithSessionDepth(context.Background(), 1)
			call, err := srv.buildCall(ctx, tc.tool, args, true, "", "served-depth")
			if err != nil {
				t.Fatal(err)
			}
			if result, hit, _ := fastPathLookupReceipt(ctx, call); !hit || result == nil {
				t.Fatal("fixture must have a real vDSO hit before testing depth admission")
			}
			before := srv.k.Counters().EngineCalls
			if tc.spawn {
				rootKept, rootAdjs, rootDropped, rootText, rootHits := srv.adjudicateProposedServed(context.Background(), []agent.ToolCall{depthCall("root-op", tc.tool)}, "served-root")
				if len(rootKept) != 1 || len(rootAdjs) != 1 || !rootAdjs[0].Admitted || rootDropped != 0 || rootText != "" || rootHits != 0 {
					t.Fatalf("root spawn must reach ordinary adjudication: kept=%v adjudications=%+v dropped=%d text=%q hits=%d", rootKept, rootAdjs, rootDropped, rootText, rootHits)
				}
			}
			kept, adjs, dropped, text, hits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{depthCall("served-op", tc.tool)}, "served-depth")
			if len(adjs) != 1 || len(kept) != 0 {
				t.Fatalf("kept=%v adjudications=%+v, want exactly one resolved call", kept, adjs)
			}
			if tc.spawn {
				if hits != 0 || text != "" || dropped != 1 || adjs[0].Admitted || adjs[0].Verdict.Reason != ReasonSubagentDepthExceeded {
					t.Fatalf("cached spawn escaped depth admission: hits=%d text=%q dropped=%d adjudication=%+v", hits, text, dropped, adjs[0])
				}
			} else if hits != 1 || text == "" || dropped != 0 || !adjs[0].Admitted || adjs[0].Verdict.Reason != "SERVED_INLINE" {
				t.Fatalf("cached read refused at cap: hits=%d text=%q dropped=%d adjudication=%+v", hits, text, dropped, adjs[0])
			}
			if after := srv.k.Counters().EngineCalls; after != before {
				t.Fatalf("admission dispatched the fixture engine: calls %d -> %d", before, after)
			}
		})
	}
}
