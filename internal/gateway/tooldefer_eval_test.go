package gateway

import "testing"

// tooldefer_eval_test.go witnesses the #3533 held-accuracy fault-in-recall eval: the gate
// holds on the production transform (deferral loses no capability), the fixed eval set only
// requires genuinely-cold tools, and the per-tool recall check catches every way a tool
// could become unreachable (dropped, unsearchable, or schema-mangled).

func TestDeferHeldAccuracyGateHolds(t *testing.T) {
	rep := DeferHeldAccuracyReport()
	if rep.Total != len(defaultHeldAccuracyTasks) {
		t.Fatalf("Total=%d, want %d", rep.Total, len(defaultHeldAccuracyTasks))
	}
	// Every cold tool must survive deferral faultable-in, so armed matches ablated matches total.
	if rep.AblatedPass != rep.Total {
		t.Fatalf("ablated pass=%d, want %d (every tool is eagerly present in its own body)", rep.AblatedPass, rep.Total)
	}
	if rep.ArmedPass != rep.Total {
		t.Fatalf("armed pass=%d, want %d — deferral lost a capability; offenders=%v; results=%+v",
			rep.ArmedPass, rep.Total, rep.Offenders, rep.Results)
	}
	if !rep.GateHolds {
		t.Fatalf("gate must hold (armed %d >= ablated %d); offenders=%v", rep.ArmedPass, rep.AblatedPass, rep.Offenders)
	}
	if len(rep.Offenders) != 0 {
		t.Fatalf("no offenders expected when the gate holds, got %v", rep.Offenders)
	}
	// Honesty: this is a mechanical recall witness, never a live-model accuracy claim.
	if rep.LiveAccuracyClaimAllowed {
		t.Fatalf("held-accuracy eval must NOT allow a live-model accuracy claim (that is #3536)")
	}
	if rep.Mode != "deterministic-faultin-sim" {
		t.Fatalf("Mode=%q, want deterministic-faultin-sim", rep.Mode)
	}
}

func TestHeldAccuracyTasksAreCold(t *testing.T) {
	if len(defaultHeldAccuracyTasks) < 3 {
		t.Fatalf("eval set has %d tasks; want >=3 (one per representative cold category)", len(defaultHeldAccuracyTasks))
	}
	for _, task := range defaultHeldAccuracyTasks {
		if defaultHotToolSet[task.Tool] {
			t.Fatalf("required tool %q is in the hot set — the eval must require a COLD tool (else deferral never touches it)", task.Tool)
		}
	}
}

func TestToolFaultableInCatchesRegressions(t *testing.T) {
	schema := []byte(`{"type":"object"}`)
	search := `{"type":"` + toolSearchToolType + `","name":"` + toolSearchToolName + `"}`

	// Valid: present + defer_loading + a tool_search_tool + schema intact -> faultable-in.
	valid := `{"tools":[{"name":"mcp__x__t","input_schema":{"type":"object"},"defer_loading":true},` + search + `]}`
	if ok, why := toolFaultableIn([]byte(valid), "mcp__x__t", schema); !ok {
		t.Fatalf("a valid deferred tool must be faultable-in, got failure %q", why)
	}

	cases := []struct {
		name, body string
	}{
		{"dropped", `{"tools":[` + search + `]}`}, // required tool absent
		{"unsearchable", `{"tools":[{"name":"mcp__x__t","input_schema":{"type":"object"},"defer_loading":true}]}`},                  // deferred, no search tool
		{"schema-mangled", `{"tools":[{"name":"mcp__x__t","input_schema":{"type":"string"},"defer_loading":true},` + search + `]}`}, // schema changed
		{"neither-deferred-nor-hot", `{"tools":[{"name":"mcp__x__t","input_schema":{"type":"object"}},` + search + `]}`},            // cold but not deferred
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if ok, _ := toolFaultableIn([]byte(c.body), "mcp__x__t", schema); ok {
				t.Fatalf("%s must fail the recall check (a silent-loss shape must be caught)", c.name)
			}
		})
	}
}

// TestHeldAccuracyNamesOffenderOnDrop drives the fold's offender path: a task whose required
// tool the transform cannot keep faultable-in (here: a tool already carrying defer_loading in
// the input, which makes the transform stand down so the tool ends up neither freshly deferred
// with a search tool nor hot) must red the gate and NAME the tool.
func TestHeldAccuracyNamesOffenderOnDrop(t *testing.T) {
	// A hand-built scorer scenario: reuse heldAccuracy over a task whose body we cannot control,
	// so assert the fold semantics directly on a synthetic result set instead.
	rep := foldHeldAccuracyResults([]HeldAccuracyResult{
		{Task: HeldAccuracyTask{Tool: "mcp__ok__a"}, AblatedPass: true, ArmedPass: true},
		{Task: HeldAccuracyTask{Tool: "mcp__lost__b"}, AblatedPass: true, ArmedPass: false, Reason: "dropped"},
	})
	if rep.GateHolds {
		t.Fatalf("gate must NOT hold when a tool is lost under deferral")
	}
	if len(rep.Offenders) != 1 || rep.Offenders[0] != "mcp__lost__b" {
		t.Fatalf("offenders=%v, want [mcp__lost__b] named on the drop", rep.Offenders)
	}
}
