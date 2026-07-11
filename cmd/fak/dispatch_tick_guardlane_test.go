package main

import (
	"encoding/json"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchTickGuardedAutoPickSkipsSelfSourceLane is the un-stall half of #1397: a
// GUARDED auto-pick (no explicit lane) must SKIP fak's TRUST-CRITICAL lanes (the
// adjudicator/policy/kernel the referee binds to) and land on a shippable lane (docs) --
// even when the held lane carries a far larger step budget. Before the fix the picker
// chose the busiest lane, then refused with SELF_MODIFY_HOLD every tick and surfaced
// NOTHING; now it skips the held lane BEFORE the busiest-pick so the tick surfaces real
// shippable work. NOTE the busiest lane here is the ADJUDICATOR (trust-critical), not
// gateway -- gateway is guard-shippable now and would NOT be skipped. The default guard
// is left ON (no FLEET_DOGFOOD_GUARD override) so this exercises the real guarded path.
func TestDispatchTickGuardedAutoPickSkipsSelfSourceLane(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				// The busiest lane by step budget is TRUST-CRITICAL (the referee's own
				// adjudicator tree) -- the only backlog shape that still stalls the surface.
				"adjudicator": {Tree: []string{"internal/adjudicator/**"}, Issues: []int{20, 21}, Count: 2, StepBudget: 9},
				"docs":        {Tree: []string{"docs/**"}, Issues: []int{12}, Count: 1, StepBudget: 3},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	// Pure picker: the trust-critical busiest lane is skipped, docs is chosen.
	pick, err := pickDispatchLane(t.TempDir(), io.Discard, "", nil, false, "", dispatchGoalProfileThroughput, 0)
	if err != nil {
		t.Fatalf("pickDispatchLane: %v", err)
	}
	if pick.Lane != "docs" {
		t.Fatalf("lane = %q, want docs (the busier adjudicator lane is trust-critical and skipped under guard)", pick.Lane)
	}
	if len(pick.SelfSourceHeld) != 1 || pick.SelfSourceHeld[0] != "adjudicator" {
		t.Fatalf("self-source held = %+v, want [adjudicator]", pick.SelfSourceHeld)
	}

	// End-to-end tick: the surface is the shippable docs lane, not an empty/held one.
	root := t.TempDir()
	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a guarded auto-pick must surface the shippable docs lane) (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["verdict"] != "WOULD_SPAWN" || got["lane"] != "docs" || got["target_issue"] != float64(12) {
		t.Fatalf("guarded auto-pick = verdict %v lane %v target %v, want WOULD_SPAWN/docs/12", got["verdict"], got["lane"], got["target_issue"])
	}
}

// TestDispatchTickGuardedAllSelfSourceBacklogExplainsHold is the honest-empty half of
// #1397: when the ENTIRE eligible backlog is TRUST-CRITICAL under guard, the auto-pick
// finds no shippable lane -- but the tick must EXPLAIN why (SELF_MODIFY_HOLD over the held
// set) rather than render a silent/empty NO_LANE that falsely reads as "router
// empty/error". The backlog here is two referee-own lanes only (adjudicator + policy);
// both are held, so the surface names them and routes the operator to an
// unguarded/worktree-isolated path (#1334).
func TestDispatchTickGuardedAllSelfSourceBacklogExplainsHold(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"adjudicator": {Tree: []string{"internal/adjudicator/**"}, Issues: []int{20}, Count: 1, StepBudget: 5},
				"policy":      {Tree: []string{"internal/policy/**"}, Issues: []int{30}, Count: 1, StepBudget: 3},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (self-source hold is reported in the payload) (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "self_modify_hold" || got["verdict"] != "SELF_MODIFY_HOLD" || got["ok"] != false {
		t.Fatalf("all-trust-critical surface = action %v verdict %v ok %v, want self_modify_hold/SELF_MODIFY_HOLD/false", got["action"], got["verdict"], got["ok"])
	}
	if got["verdict"] == "NO_LANE" {
		t.Fatalf("all-trust-critical backlog must not render as the misleading NO_LANE router-empty surface: %#v", got)
	}
	held := stringAnySlice(got["self_modify_held_lanes"])
	if len(held) != 2 || held[0] != "adjudicator" || held[1] != "policy" {
		t.Fatalf("self_modify_held_lanes = %v, want sorted [adjudicator policy] (every candidate lane held as trust-critical)", held)
	}
}
