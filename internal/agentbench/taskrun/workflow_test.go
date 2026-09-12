package taskrun

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

func TestAgentBenchWorkflowEvidence(t *testing.T) {
	cases := taskfixture.Cases(false)
	kinds := map[string]int{}
	areas := map[string]int{}
	for _, fixture := range cases {
		kinds[fixture.Workflow.Kind]++
		if fixture.Workflow.Kind == "shared_area_followup" {
			if fixture.Workflow.Area != "labels" || fixture.Workflow.PriorTaskID == "" || fixture.Workflow.PriorStateSHA256 != digestString(fixture.BrokenSource) {
				t.Fatalf("shared follow-up %s lacks an exact frozen labels prior state: %+v", fixture.ID, fixture.Workflow)
			}
			if !reflect.DeepEqual(fixture.Workflow.PriorSteps, []string{"inspect", "edit", "test_pass"}) {
				t.Fatalf("shared follow-up %s prior trace = %#v", fixture.ID, fixture.Workflow.PriorSteps)
			}
			if fixture.Workflow.PriorBeforeSource == "" || fixture.Workflow.PriorAfterSource != fixture.BrokenSource || fixture.Workflow.PriorTestSource == "" {
				t.Fatalf("shared follow-up %s lacks replayable frozen source/test bytes", fixture.ID)
			}
			reconstructed, err := applyFrozenPriorDiff(fixture.Workflow.PriorBeforeSource, fixture.Workflow.PriorDiff)
			if err != nil || reconstructed != fixture.Workflow.PriorAfterSource {
				t.Fatalf("shared follow-up %s prior diff does not reconstruct after state: err=%v", fixture.ID, err)
			}
			if !strings.Contains(strings.ToLower(fixture.Prompt), "frozen prehistory") {
				t.Fatalf("shared follow-up %s prompt does not disclose supplied frozen prehistory: %q", fixture.ID, fixture.Prompt)
			}
			for _, supplied := range []string{fixture.Workflow.PriorAfterSource, fixture.Workflow.PriorDiff, fixture.Workflow.PriorTestSource} {
				if !strings.Contains(fixture.Prompt, supplied) {
					t.Fatalf("shared follow-up %s prompt omits supplied frozen prehistory bytes", fixture.ID)
				}
			}
			areas[fixture.Workflow.Area]++
		}
		for _, step := range fixture.Workflow.RequiredSteps {
			if !strings.Contains(strings.ToLower(fixture.Prompt), strings.ReplaceAll(step, "_", " ")) {
				t.Fatalf("fixture %s prompt does not explicitly request %q sequence: %q", fixture.ID, step, fixture.Prompt)
			}
		}
	}
	if fmt.Sprint(kinds) == "" || kinds["behavior_fix"] != 2 || kinds["shared_area_followup"] != 2 || kinds["tdd"] != 1 || kinds["stale_reread"] != 1 || len(areas) != 1 {
		t.Fatalf("six-task workflow mix = kinds %#v shared areas %#v", kinds, areas)
	}

	testCommand := "go test ./... -count=1"
	fixed := cases[0].FixedSource
	happy := workflowTurns(testCommand, fixed, fixed)
	contracts := []taskfixture.Fixture{
		{Workflow: taskfixture.Workflow{Kind: "behavior_fix", RequiredSteps: []string{"inspect", "edit", "test_pass"}}, TargetFile: "target.go", FixedSource: fixed},
		{Workflow: taskfixture.Workflow{Kind: "tdd", RequiredSteps: []string{"inspect", "test_fail", "edit", "test_pass"}}, TargetFile: "target.go", FixedSource: fixed},
		{Workflow: taskfixture.Workflow{Kind: "stale_reread", RequiredSteps: []string{"inspect", "edit", "reread", "test_pass"}}, TargetFile: "target.go", BrokenSource: "old source", FixedSource: fixed},
	}
	for _, fixture := range contracts {
		receipt, err := validateWorkflow(fixture, happy, testCommand)
		if err != nil || !receipt.Passed || receipt.Kind != fixture.Workflow.Kind {
			t.Fatalf("%s happy workflow = %+v err=%v", fixture.Workflow.Kind, receipt, err)
		}
	}

	tdd := contracts[1]
	for name, turns := range map[string][]PlannerTurn{
		"unlinked result": workflowTurns(testCommand, fixed, fixed)[:1],
		"out of order":    append(workflowTurns(testCommand, fixed, fixed)[4:], workflowTurns(testCommand, fixed, fixed)[:4]...),
		"false tdd":       withoutFailure(workflowTurns(testCommand, fixed, fixed)),
	} {
		if receipt, err := validateWorkflow(tdd, turns, testCommand); err == nil || receipt.Passed {
			t.Fatalf("%s qualified TDD: %+v err=%v", name, receipt, err)
		}
	}
	stale := contracts[2]
	if receipt, err := validateWorkflow(stale, workflowTurns(testCommand, fixed, stale.BrokenSource), testCommand); err == nil || receipt.Passed {
		t.Fatalf("stale reread of old content qualified: %+v err=%v", receipt, err)
	}
	if receipt, err := validateWorkflow(stale, workflowTurns(testCommand, "candidate differs from golden", stale.FixedSource), testCommand); err == nil || receipt.Passed {
		t.Fatalf("reread matching golden but not the actual edit qualified: %+v err=%v", receipt, err)
	}
}

func TestAgentBenchFrozenPrehistoryVerifiedBeforeModel(t *testing.T) {
	if err := SandboxAvailable(); err != nil {
		t.Skip(err)
	}
	for _, fixture := range taskfixture.Cases(false) {
		if fixture.Workflow.Kind != "shared_area_followup" {
			continue
		}
		artifact := filepath.Join(t.TempDir(), "prehistory.json")
		receipt, err := validateFrozenPrehistory(context.Background(), fixture, artifact)
		if err != nil || !receipt.Passed || receipt.PriorTaskID != fixture.Workflow.PriorTaskID || receipt.BeforeSHA256 != digestString(fixture.Workflow.PriorBeforeSource) || receipt.AfterSHA256 != digestString(fixture.Workflow.PriorAfterSource) || receipt.DiffSHA256 != digestString(fixture.Workflow.PriorDiff) || receipt.TestSHA256 != digestString(fixture.Workflow.PriorTestSource) {
			t.Fatalf("frozen prehistory verification = %+v err=%v", receipt, err)
		}
		body, readErr := os.ReadFile(artifact)
		var persisted PrehistoryReceipt
		decodeErr := json.Unmarshal(body, &persisted)
		if readErr != nil || decodeErr != nil || !reflect.DeepEqual(receipt, persisted) {
			t.Fatalf("pre-model prehistory receipt was not durably persisted: read=%v decode=%v got=%+v", readErr, decodeErr, persisted)
		}
	}
}

func workflowTurns(testCommand, written, reread string) []PlannerTurn {
	call := func(id, name, args string) PlannerTurn {
		return PlannerTurn{ToolCalls: []agent.ToolCall{{ID: id, Type: "function", Function: agent.Func{Name: name, Arguments: args}}}}
	}
	result := func(id, content string) PlannerTurn {
		return PlannerTurn{Messages: []agent.Message{{Role: agent.RoleTool, ToolCallID: id, Content: content}}}
	}
	return []PlannerTurn{
		call("read-before", "Read", `{"file_path":"target.go"}`), result("read-before", `{"content":"old source"}`),
		call("test-red", "Bash", fmt.Sprintf(`{"command":%q}`, testCommand)), result("test-red", `{"exit_code":1,"output":"assertion failed"}`),
		call("edit", "Write", fmt.Sprintf(`{"file_path":"target.go","content":%q}`, written)), result("edit", `{"ok":true}`),
		call("read-after", "Read", `{"file_path":"target.go"}`), result("read-after", fmt.Sprintf(`{"content":%q}`, reread)),
		call("test-green", "Bash", fmt.Sprintf(`{"command":%q}`, testCommand)), result("test-green", `{"exit_code":0,"output":"ok"}`),
	}
}

func withoutFailure(turns []PlannerTurn) []PlannerTurn {
	copyTurns := append([]PlannerTurn(nil), turns...)
	copyTurns[3] = PlannerTurn{Messages: []agent.Message{{Role: agent.RoleTool, ToolCallID: "test-red", Content: `{"exit_code":0,"output":"ok"}`}}}
	return copyTurns
}
