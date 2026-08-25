package dispatchtick

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

type codexSandboxWitness struct {
	Issue    int    `json:"issue"`
	Decision string `json:"decision"`
	Arms     []struct {
		Name          string   `json:"name"`
		Argv          []string `json:"argv"`
		EditCompleted bool     `json:"edit_completed"`
		TestsGreen    bool     `json:"tests_green"`
		TimedOut      bool     `json:"timed_out"`
	} `json:"arms"`
}

func TestCodexWorkerCommandMatchesIssue9007LiveSandboxWitness(t *testing.T) {
	data, err := os.ReadFile("testdata/issue-9007-codex-sandbox-witness.json")
	if err != nil {
		t.Fatal(err)
	}
	var witness codexSandboxWitness
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatal(err)
	}
	if witness.Issue != 9007 || witness.Decision != "retain-full-bypass" || len(witness.Arms) != 2 {
		t.Fatalf("unexpected witness decision: %+v", witness)
	}
	baseline, narrowed := witness.Arms[0], witness.Arms[1]
	if !baseline.EditCompleted || !baseline.TestsGreen || baseline.TimedOut {
		t.Fatalf("full-bypass control did not complete the bounded task: %+v", baseline)
	}
	if narrowed.EditCompleted || narrowed.TestsGreen || !narrowed.TimedOut {
		t.Fatalf("workspace-write arm unexpectedly satisfied the bounded task: %+v", narrowed)
	}
	got, err := BuildWorkerCommand("codex", "task", WorkerLaunch{})
	if err != nil {
		t.Fatal(err)
	}
	want := append(slices.Clone(baseline.Argv), "-")
	if !slices.Equal(got, want) {
		t.Fatalf("Codex command = %q, live-witnessed command = %q", got, want)
	}
}
