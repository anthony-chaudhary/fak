package main

import (
	"strings"
	"testing"
)

// TestRenderInfoAgentsViewTreeHierarchy witnesses the hierarchical tree view and per-agent
// token reuse breakdown in the expanded agents view (Issue #12311).
func TestRenderInfoAgentsViewTreeHierarchy(t *testing.T) {
	v := guardInfoVars{
		Sessions: []guardInfoSession{
			{
				TraceID:        "trace-a1b2",
				Role:           "coord",
				ElapsedSeconds: 720,
				PromptTokens:   80000,
				LastTool:       "task_spawn",
				SpawnCount:     3,
			},
			{
				TraceID:         "trace-c3d4",
				ParentSessionID: "trace-a1b2",
				SubagentType:    "researcher",
				ElapsedSeconds:  120,
				PromptTokens:    24000,
				SharedTokens:    21000,
				ReuseRate:       0.88,
				LastTool:        "read",
				InflightSeconds: 5,
			},
			{
				TraceID:         "trace-e5f6",
				ParentSessionID: "trace-a1b2",
				SubagentType:    "worker",
				ElapsedSeconds:  60,
				PromptTokens:    32000,
				SharedTokens:    25000,
				ReuseRate:       0.79,
				LastTool:        "edit",
				InflightSeconds: 2,
			},
			{
				TraceID:         "trace-789a",
				ParentSessionID: "trace-a1b2",
				SubagentType:    "tester",
				ElapsedSeconds:  30,
				PromptTokens:    18000,
				SharedTokens:    16000,
				ReuseRate:       0.92,
				LastTool:        "bash",
			},
		},
	}

	lines := renderInfoAgentsView(v)
	rendered := strings.Join(lines, "\n")

	// 1. Verify summary row has active counts, subagents count, in-flight, and cross-agent token reuse rate
	summaryWants := []string{"4 active", "3 subagents", "2 in-flight", "84% x-agent reuse"}
	for _, want := range summaryWants {
		if !strings.Contains(rendered, want) {
			t.Errorf("summary row missing %q in rendered output:\n%s", want, rendered)
		}
	}

	// 2. Verify coordinator row rendering
	coordWants := []string{"trace-a1b2", "[coord]", "up 12m", "80k tok", "tool task_spawn"}
	for _, want := range coordWants {
		if !strings.Contains(rendered, want) {
			t.Errorf("coordinator row missing %q in rendered output:\n%s", want, rendered)
		}
	}

	// 3. Verify Unicode branch connectors for subagents
	if !strings.Contains(rendered, "├─ ") {
		t.Errorf("rendered output missing branch connector ├─ :\n%s", rendered)
	}
	if !strings.Contains(rendered, "└─ ") {
		t.Errorf("rendered output missing leaf connector └─ :\n%s", rendered)
	}

	// 4. Verify subagent 1 (researcher)
	sub1Wants := []string{"trace-c3d4", "[sub:researcher]", "up 2m", "24k tok", "88% reuse (21k shared)", "tool read"}
	for _, want := range sub1Wants {
		if !strings.Contains(rendered, want) {
			t.Errorf("subagent 1 row missing %q in rendered output:\n%s", want, rendered)
		}
	}

	// 5. Verify subagent 2 (worker)
	sub2Wants := []string{"trace-e5f6", "[sub:worker]", "up 1m", "32k tok", "79% reuse (25k shared)", "tool edit"}
	for _, want := range sub2Wants {
		if !strings.Contains(rendered, want) {
			t.Errorf("subagent 2 row missing %q in rendered output:\n%s", want, rendered)
		}
	}

	// 6. Verify subagent 3 (tester) as the last child
	sub3Wants := []string{"trace-789a", "[sub:tester]", "up 30s", "18k tok", "92% reuse (16k shared)", "tool bash"}
	for _, want := range sub3Wants {
		if !strings.Contains(rendered, want) {
			t.Errorf("subagent 3 row missing %q in rendered output:\n%s", want, rendered)
		}
	}

	// 7. Verify tree structure ordering: parent appears before children, branch prefixes present
	coordIdx := strings.Index(rendered, "trace-a1b2")
	sub1Idx := strings.Index(rendered, "trace-c3d4")
	sub2Idx := strings.Index(rendered, "trace-e5f6")
	sub3Idx := strings.Index(rendered, "trace-789a")
	if !(coordIdx < sub1Idx && sub1Idx < sub2Idx && sub2Idx < sub3Idx) {
		t.Errorf("expected tree order coord < sub1 < sub2 < sub3, got indices: %d, %d, %d, %d",
			coordIdx, sub1Idx, sub2Idx, sub3Idx)
	}
}

// TestGuardInfoAgentTextRoleAndReuse pins guardInfoAgentText's role and reuse chips.
func TestGuardInfoAgentTextRoleAndReuse(t *testing.T) {
	// Subagent row with explicit SubagentType, PromptTokens, SharedTokens, and ReuseRate
	sub := guardInfoAgentText(guardInfoSession{
		TraceID:         "sub-lead",
		SubagentType:    "researcher",
		ElapsedSeconds:  120,
		PromptTokens:    24000,
		SharedTokens:    21000,
		ReuseRate:       0.88,
		LastTool:        "read",
		InflightSeconds: 5,
	})

	for _, want := range []string{
		"sub-lead",
		"[sub:researcher]",
		"up 2m",
		"24k tok",
		"88% reuse (21k shared)",
		"tool read",
		"in-flight 5s",
	} {
		if !strings.Contains(sub, want) {
			t.Errorf("subagent text missing %q: %q", want, sub)
		}
	}

	// Coordinator row with explicit Role
	coord := guardInfoAgentText(guardInfoSession{
		TraceID:        "coord-root",
		Role:           "coord",
		ElapsedSeconds: 720,
		PromptTokens:   80000,
		LastTool:       "task_spawn",
	})

	for _, want := range []string{
		"coord-root",
		"[coord]",
		"up 12m",
		"80k tok",
		"tool task_spawn",
	} {
		if !strings.Contains(coord, want) {
			t.Errorf("coordinator text missing %q: %q", want, coord)
		}
	}
}

// TestRenderInfoAgentsViewTreeFallback verifies that flat sessions with no subagents
// render without tree branches and retain backward compatibility.
func TestRenderInfoAgentsViewTreeFallback(t *testing.T) {
	v := guardInfoVars{
		Sessions: []guardInfoSession{
			{TraceID: "session-1", Run: "running"},
			{TraceID: "session-2", Run: "running"},
		},
	}
	lines := renderInfoAgentsView(v)
	rendered := strings.Join(lines, "\n")

	if strings.Contains(rendered, "├─") || strings.Contains(rendered, "└─") {
		t.Errorf("flat sessions must not render tree connectors:\n%s", rendered)
	}
	if !strings.Contains(rendered, "2 active") {
		t.Errorf("flat sessions summary missing '2 active':\n%s", rendered)
	}
}
