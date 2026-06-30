package main

import (
	"encoding/json"
	"testing"
)

func TestDispatchTickLiveHoldsGuardedSelfModifyBeforeSpawn(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--live", "--no-refresh", "--no-loop-ledger", "--json")
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
