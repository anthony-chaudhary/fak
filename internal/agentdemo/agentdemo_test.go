package agentdemo_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentdemo"

	// Wire the full adjudicator chain (resolver, vDSO, adjudicator, ctx-MMU,
	// normgate, IFC, witness, engines) so kernel.Fold folds the REAL floor — the
	// same one-line requirement every on-box demo main carries. Excluded from the
	// architest upward-import gate (it skips _test.go files).
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// fixedTools is a tiny toolset mirroring cmd/timewolfdemo's shape: a read-only
// get_ tool the floor allows, and a destructive sink the floor refuses.
func fixedTools() *agentdemo.Toolset {
	return agentdemo.NewToolset(
		agentdemo.Floor{
			AllowPrefix: []string{"get_"},
			Deny:        []string{"delete_calendar"},
		},
		agentdemo.Tool{
			Name:    "get_time",
			Summary: "report the (injected, deterministic) clock",
			Handler: func(args json.RawMessage) string { return "it is 11:58 — one minute to dinner" },
		},
		agentdemo.Tool{
			Name:    "delete_calendar",
			Summary: "wipe the calendar (the injection's payload)",
			Handler: func(args json.RawMessage) string { return "calendar wiped" },
		},
	)
}

func TestRun_AllowsReadDeniesDestructive(t *testing.T) {
	ts := fixedTools()
	plan := []agentdemo.Step{
		{Tool: "get_time", Note: "the user's actual question"},
		{Tool: "delete_calendar", Note: "the injection's payload"},
		{Tool: "wipe_disk", Note: "an off-floor tool — falls to DEFAULT_DENY"},
	}
	tr, err := ts.Run(context.Background(), "mr-wolf", "what time is it, Mr. Wolf?", plan)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(tr.Turns) != 3 {
		t.Fatalf("turns = %d, want 3", len(tr.Turns))
	}

	// get_time: allowed, handler ran, result captured.
	if got := tr.Turns[0]; got.Verdict != "ALLOW" || !got.Allowed {
		t.Errorf("get_time verdict = %s allowed=%v, want ALLOW/true (by=%s reason=%s)", got.Verdict, got.Allowed, got.By, got.Reason)
	}
	if !strings.Contains(tr.Turns[0].Result, "dinner") {
		t.Errorf("get_time result = %q, want the handler output", tr.Turns[0].Result)
	}

	// delete_calendar: explicit deny, POLICY_BLOCK, no result.
	if got := tr.Turns[1]; got.Verdict != "DENY" || got.Allowed {
		t.Errorf("delete_calendar verdict = %s allowed=%v, want DENY/false", got.Verdict, got.Allowed)
	}
	if tr.Turns[1].Reason != "POLICY_BLOCK" {
		t.Errorf("delete_calendar reason = %s, want POLICY_BLOCK", tr.Turns[1].Reason)
	}
	if tr.Turns[1].Result != "" {
		t.Errorf("delete_calendar result = %q, want empty (call never dispatched)", tr.Turns[1].Result)
	}

	// wipe_disk: not on the floor -> fail-closed DEFAULT_DENY.
	if got := tr.Turns[2]; got.Verdict != "DENY" || got.Reason != "DEFAULT_DENY" {
		t.Errorf("wipe_disk verdict=%s reason=%s, want DENY/DEFAULT_DENY", got.Verdict, got.Reason)
	}

	// Tally + answer.
	if tr.Allowed != 1 || tr.Denied != 2 {
		t.Errorf("tally = %d allowed / %d denied, want 1/2", tr.Allowed, tr.Denied)
	}
	if !strings.Contains(tr.Answer, "dinner") {
		t.Errorf("answer = %q, want the allowed get_time result", tr.Answer)
	}
}

func TestRun_AllExplicitAllow(t *testing.T) {
	ts := agentdemo.NewToolset(
		agentdemo.Floor{Allow: []string{"get_date"}},
		agentdemo.Tool{Name: "get_date", Handler: func(json.RawMessage) string { return "2026-06-28" }},
	)
	tr, err := ts.Run(context.Background(), "date", "what's the date?", []agentdemo.Step{{Tool: "get_date"}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Allowed != 1 || tr.Denied != 0 {
		t.Fatalf("tally = %d/%d, want 1/0", tr.Allowed, tr.Denied)
	}
	if tr.Answer != "2026-06-28" {
		t.Errorf("answer = %q, want 2026-06-28", tr.Answer)
	}
}

func TestPlan_DeterministicPlanner(t *testing.T) {
	ts := fixedTools()
	planner := func(prompt string) []agentdemo.Step {
		if strings.Contains(strings.ToLower(prompt), "time") {
			return []agentdemo.Step{{Tool: "get_time"}}
		}
		return nil
	}
	tr, err := ts.Plan(context.Background(), "mr-wolf", "what time is it?", planner)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if len(tr.Turns) != 1 || tr.Turns[0].Tool != "get_time" {
		t.Fatalf("planner did not route 'time' to get_time: %+v", tr.Turns)
	}
}

func TestTranscript_RenderTextAndJSON(t *testing.T) {
	ts := fixedTools()
	tr, err := ts.Run(context.Background(), "mr-wolf", "what time is it, Mr. Wolf?",
		[]agentdemo.Step{{Tool: "get_time"}, {Tool: "delete_calendar"}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	var sb strings.Builder
	tr.RenderText(&sb)
	out := sb.String()
	for _, want := range []string{"ALLOW", "DENY", "get_time", "REFUSED (POLICY_BLOCK)", "1 allowed · 1 refused"} {
		if !strings.Contains(out, want) {
			t.Errorf("RenderText missing %q in:\n%s", want, out)
		}
	}
	if js := tr.JSON(); !strings.Contains(js, "\"verdict\": \"ALLOW\"") {
		t.Errorf("JSON missing verdict field:\n%s", js)
	}
}
