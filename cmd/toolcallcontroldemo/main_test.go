package main

import (
	"encoding/json"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestSelfcheckCapturedOutput(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "-selfcheck", "-pretty")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("selfcheck: %v", err)
	}
	var got output
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("json: %v", err)
	}
	if got.SelfCheck != "PASS" || len(got.Decisions) != 5 || len(got.Ablation) != 2 {
		t.Fatalf("output=%s", out)
	}
	actions := map[string]toolAction{}
	for _, d := range got.Decisions {
		actions[d.ID] = toolAction(d.Action)
	}
	if actions["repeat"] != "reuse" || actions["search-a"] != "allow" || actions["search-b"] != "batch" || actions["browse"] != "defer" || actions["write"] != "allow" {
		t.Fatalf("actions=%v", actions)
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", out); err != nil {
		t.Fatal(err)
	}
}

type toolAction string
