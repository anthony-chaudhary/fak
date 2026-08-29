package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestOrchestrationPlanAcceptsCurrentTaskText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fan out independent checks in parallel", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Requested.Name != orchestration.ProfileAuto || got.Resolved.Profile != orchestration.ProfileUltracode || got.Resolved.TaskID == "" || got.Resolved.SOLRoute.Mode != orchestration.SOLUltra {
		t.Fatalf("resolved=%+v", got)
	}
	for _, role := range got.Resolved.Roles {
		if role.Access.Mode != orchestration.ChildAccessObserve || len(role.Access.WriteSet) != 0 {
			t.Fatalf("--task-text inferred worker authority: %+v", role)
		}
	}
	if strings.Contains(stdout.String(), "fan out independent checks") {
		t.Fatal("raw task text leaked into receipt")
	}
}

func TestOrchestrationPlanAcceptsTypedWorkerAccessFixture(t *testing.T) {
	fixture := t.TempDir() + "/task.json"
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"typed-access",
		"work_class":"grind",
		"max_workers":2,
		"worker_access":[{
			"role_id":"worker-1",
			"access":{
				"mode":"effect",
				"read_set":["internal/orchestration"],
				"write_set":["internal/orchestration"],
				"tools":["Read","Write"]
			}
		}]
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "ultracode", "--task", fixture, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Resolved.Roles) != 2 || got.Resolved.Roles[0].Access.Mode != orchestration.ChildAccessObserve {
		t.Fatalf("roles=%+v", got.Resolved.Roles)
	}
	worker := got.Resolved.Roles[1]
	if worker.ID != "worker-1" || worker.Access.Mode != orchestration.ChildAccessEffect ||
		len(worker.Access.WriteSet) != 1 || worker.Access.WriteSet[0] != "internal/orchestration" {
		t.Fatalf("worker access=%+v", worker.Access)
	}
}

func TestOrchestrationPlanTaskTextKeepsTinyWorkDirect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fix typo", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Resolved.Profile != orchestration.ProfileOff || got.Resolved.SOLRoute.Mode != orchestration.SOLStandard {
		t.Fatalf("resolved=%+v, want off/standard", got.Resolved)
	}
}

func TestOrchestrationPlanRequiresExactlyOneTaskSource(t *testing.T) {
	for _, args := range [][]string{{"plan"}, {"plan", "--task", "x", "--task-text", "y"}} {
		var stdout, stderr bytes.Buffer
		if code := runOrchestration(&stdout, &stderr, args); code != 2 {
			t.Fatalf("args=%v code=%d stderr=%q", args, code, stderr.String())
		}
	}
}
