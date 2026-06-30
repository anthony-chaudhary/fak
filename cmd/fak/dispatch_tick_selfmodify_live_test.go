package main

import (
	"encoding/json"
	"testing"
)

// The cmd lane is named EXPLICITLY (--lane cmd): the #1397 fix skips self-source lanes from
// the guarded AUTO-pick (a default live tick on this fixture lands on the shippable docs
// lane), but an operator who explicitly names a self-source lane must still hit the
// SELF_MODIFY hold BEFORE any lease/spawn -- the live safety-net this test pins.
func TestDispatchTickLiveHoldsGuardedSelfModifyBeforeSpawn(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--live", "--lane", "cmd", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (a live self-modify hold is a refuse) (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "self_modify_hold" || got["verdict"] != "SELF_MODIFY_HOLD" || got["ok"] != false {
		t.Fatalf("live dispatch tick result = action %v verdict %v ok %v, want self_modify_hold/SELF_MODIFY_HOLD/false", got["action"], got["verdict"], got["ok"])
	}
	if got["live"] != true || got["lane"] != "cmd" || got["self_modify_tree"] != "cmd/**" {
		t.Fatalf("live/lane/tree = %v/%v/%v, want true/cmd/cmd/**", got["live"], got["lane"], got["self_modify_tree"])
	}
	if _, ok := got["lease"]; ok {
		t.Fatalf("self-modify hold must happen before lease acquisition: %#v", got["lease"])
	}
	if _, ok := got["spawned"]; ok {
		t.Fatalf("self-modify hold must happen before spawning a worker: %#v", got["spawned"])
	}
}
