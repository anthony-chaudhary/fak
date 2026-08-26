package orchestration

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/witnessprocess"
)

func TestTaskFromTextClassifiesWorkShape(t *testing.T) {
	tests := []struct {
		name string
		text string
		want WorkClass
	}{
		{"tiny", "fix typo", WorkDefault},
		{"bounded", "implement the feature and verify its behavior", WorkDefault},
		{"multi-step", "implement the multi-step feature, add observability, dogfood it, and ship it", WorkGrind},
		{"serial-action-list", "inspect the workflow default, verify its receipt, and summarize the result", WorkGrind},
		{"comma-prose", "summarize the current behavior, including caveats, for the operator", WorkDefault},
		{"parallel", "fan out independent checks in parallel", WorkGrind},
		{"long", "drain the backlog unattended overnight", WorkRigor},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task, err := TaskFromText(tt.text)
			if err != nil {
				t.Fatal(err)
			}
			if task.Schema != "fak-orchestration-task/1" || task.WorkClass != tt.want || task.ID == "" {
				t.Fatalf("task=%+v want class=%s", task, tt.want)
			}
		})
	}
}

func TestTaskFromTextRejectsEmptyAndDoesNotRetainPrompt(t *testing.T) {
	if _, err := TaskFromText("   "); err == nil {
		t.Fatal("empty task accepted")
	}
	task, err := TaskFromText("Implement Secret_Widget and verify it")
	if err != nil {
		t.Fatal(err)
	}
	if task.ID == "Implement Secret_Widget and verify it" || len(task.ID) != len("task-")+16 {
		t.Fatalf("task id %q is not a bounded digest", task.ID)
	}
}

func TestResolveWitnessFirstWarnsOrEnforcesBeforeDispatch(t *testing.T) {
	incomplete := &witnessprocess.Block{Context: witnessprocess.Logic, Envelope: "fixed input", Lever: "one parser change", CandidateArtifact: "candidate", DurableWitness: "test"}
	task := TaskSpec{Schema: "fak-orchestration-task/1", ID: "task-witness", WorkClass: WorkGrind, WitnessFirst: incomplete}
	got, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, task, HarnessCapabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Resolved.Warnings) != 1 {
		t.Fatalf("warnings=%v", got.Resolved.Warnings)
	}
	incomplete.Policy = witnessprocess.Enforce
	if _, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, task, HarnessCapabilities{}); err == nil {
		t.Fatal("enforced lane accepted incomplete witness block")
	}
}
