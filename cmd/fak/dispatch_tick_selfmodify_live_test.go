package main

import (
	"encoding/json"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// The adjudicator lane is named EXPLICITLY (--lane adjudicator): the #1397 fix skips
// trust-critical lanes from the guarded AUTO-pick (a default live tick lands on a
// shippable lane), but an operator who explicitly names a trust-critical lane -- the
// referee's own trees -- must still hit the SELF_MODIFY hold BEFORE any lease/spawn --
// the live safety-net this test pins. (A merely-self-source lane like cmd/gateway is NOT
// held: the worker guard permits shipping it.)
func TestDispatchTickLiveHoldsGuardedSelfModifyBeforeSpawn(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"adjudicator": {Tree: []string{"internal/adjudicator/**"}, Issues: []int{2409}, Count: 1},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--live", "--lane", "adjudicator", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (live self-modify hold is reported in the payload) (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "self_modify_hold" || got["verdict"] != "SELF_MODIFY_HOLD" || got["ok"] != false {
		t.Fatalf("live dispatch tick result = action %v verdict %v ok %v, want self_modify_hold/SELF_MODIFY_HOLD/false", got["action"], got["verdict"], got["ok"])
	}
	if got["live"] != true || got["lane"] != "adjudicator" || got["self_modify_tree"] != "internal/adjudicator/**" {
		t.Fatalf("live/lane/tree = %v/%v/%v, want true/adjudicator/internal/adjudicator/**", got["live"], got["lane"], got["self_modify_tree"])
	}
	if _, ok := got["lease"]; ok {
		t.Fatalf("self-modify hold must happen before lease acquisition: %#v", got["lease"])
	}
	if _, ok := got["spawned"]; ok {
		t.Fatalf("self-modify hold must happen before spawning a worker: %#v", got["spawned"])
	}
}
