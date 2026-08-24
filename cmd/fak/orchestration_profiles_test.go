package main

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestOrchestrationWorkerArgsPropagateProfiles(t *testing.T) {
	req := orchestrationWorkerLaunchRequest{OutputProfile: "caveman:native:high", WorkProfile: "ponytail:native:low"}
	args := orchestrationWorkerArgs(req, "audit.jsonl")
	for _, pair := range [][2]string{{"--output-profile", "caveman:native:high"}, {"--work-profile", "ponytail:native:low"}} {
		i := slices.Index(args, pair[0])
		if i < 0 || i+1 >= len(args) || args[i+1] != pair[1] {
			t.Fatalf("args missing %q %q: %q", pair[0], pair[1], args)
		}
	}
}

func TestOrchestrationProfilesRejectUnknownBeforePlanning(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{name: "output", args: []string{"plan", "--task-text", "x", "--output-profile", "mystery"}, want: "invalid --output-profile"},
		{name: "work", args: []string{"plan", "--task-text", "x", "--work-profile", "mystery"}, want: "invalid --work-profile"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runOrchestration(&stdout, &stderr, tc.args); code != 2 || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("code=%d stderr=%q", code, stderr.String())
			}
		})
	}
}

func TestOrchestrationProfileReceiptAndStatusReadback(t *testing.T) {
	receipt := codexOrchestrationLaunchReceipt{
		SessionID: "s", RunID: "r", RequestedProfile: "ultracode", ResolvedProfile: "ultracode",
		OutputProfile: "full", WorkProfile: "standard", ProfileSource: "cli",
		Workers: []codexOrchestrationWorkerLaunch{{RoleID: "writer", OutputProfile: "full", WorkProfile: "standard"}},
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"output_profile":"full"`, `"work_profile":"standard"`, `"profile_source":"cli"`} {
		if !bytes.Contains(data, []byte(want)) {
			t.Fatalf("receipt missing %s: %s", want, data)
		}
	}
	status := inspectOrchestrationRun(t.TempDir(), receipt)
	if status.OutputProfile != "full" || status.WorkProfile != "standard" || status.ProfileSource != "cli" || len(status.Workers) != 1 || status.Workers[0].OutputProfile != "full" || status.Workers[0].WorkProfile != "standard" {
		t.Fatalf("status readback = %+v", status)
	}
}
