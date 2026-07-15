package operatorquestion

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

func TestEquivalentHarnessQuestionsNormalizeAndTriageIdentically(t *testing.T) {
	claude, err := Normalize(NativeGate{
		HarnessCommand: "claude",
		Tool:           "AskUserQuestion",
		Payload:        []byte(`{"questions":[{"header":"Isolation","multiSelect":false,"question":"Which isolation should I use?","options":[{"label":"Explicit paths","description":"Commit only owned files"},{"label":"Wait","description":"Wait for peer edits"}]}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	codex, err := Normalize(NativeGate{
		HarnessCommand: "codex",
		Tool:           "functions.request_user_input",
		Payload:        []byte(`{"questions":[{"id":"isolation","header":"Isolation","question":"Which isolation should I use?","options":[{"label":"Explicit paths","description":"Commit only owned files"},{"label":"Wait","description":"Wait for peer edits"}]}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if claude.Harness != "claude" || codex.Harness != "codex" {
		t.Fatalf("harness identities: claude=%q codex=%q", claude.Harness, codex.Harness)
	}
	if claude.Kind != ChooseApproach || codex.Kind != ChooseApproach {
		t.Fatalf("kinds: claude=%q codex=%q", claude.Kind, codex.Kind)
	}
	if !reflect.DeepEqual(claude.Options, codex.Options) || claude.Question != codex.Question {
		t.Fatalf("normalization drift:\nclaude=%+v\ncodex=%+v", claude, codex)
	}
	cv, xv := choicetriage.Triage(claude.ToSignal()), choicetriage.Triage(codex.ToSignal())
	if cv != xv || cv.Disposition != choicetriage.FreshContext {
		t.Fatalf("triage drift: claude=%+v codex=%+v", cv, xv)
	}
}

func TestPlanGatesNormalizeToPlanApproval(t *testing.T) {
	tests := []struct {
		name      string
		gate      NativeGate
		wantSteps int
	}{
		{"claude", NativeGate{HarnessCommand: "claude-code", Tool: "ExitPlanMode", Payload: []byte(`{"plan":"inspect then edit","file_tree":["internal/x/**"],"steps":[{"text":"inspect","tool":"Read","args":{"path":"x"}}],"done_criterion":"dos verify plan phase"}`)}, 1},
		{"codex", NativeGate{HarnessCommand: "codex", Tool: "update_plan", Payload: []byte(`{"explanation":"safe sequence","file_tree":["internal/x/**"],"done_criterion":"dos verify plan phase","plan":[{"step":"inspect","status":"pending","tool":"Read","args":{"path":"x"}},{"step":"edit","status":"pending"}]}`)}, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Normalize(tc.gate)
			if err != nil {
				t.Fatal(err)
			}
			if got.Kind != PlanApproval || got.Provenance.NativeTool != tc.gate.Tool || got.Detail == "" || got.Plan == nil || len(got.Plan.FileTree) != 1 || len(got.Plan.Steps) != tc.wantSteps || got.Plan.DoneCriterion == "" {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestNormalizeRejectsUnknownFieldsAndHarnesses(t *testing.T) {
	if _, err := Normalize(NativeGate{HarnessCommand: "claude", Tool: "AskUserQuestion", Payload: []byte(`{"questions":[],"surprise":true}`)}); err == nil {
		t.Fatal("unknown payload field was accepted")
	}
	if _, err := Normalize(NativeGate{HarnessCommand: "mystery-agent", Tool: "ask", Payload: []byte(`{}`)}); err == nil {
		t.Fatal("unknown harness was accepted")
	}
}

func TestLastFromTranscriptNormalizesLastOperatorGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "claude.jsonl")
	content := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"AskUserQuestion","input":{"questions":[{"question":"Which isolation?","options":[{"label":"Explicit paths","description":"owned files"},{"label":"Wait","description":"peer completion"}]}]}}]}}` + "\n" +
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"path":"README.md"}}]}}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	got, found, err := LastFromTranscript(path, "claude")
	if err != nil {
		t.Fatal(err)
	}
	if !found || got.Kind != ChooseApproach || got.Question != "Which isolation?" || len(got.Options) != 2 {
		t.Fatalf("found=%v got=%+v", found, got)
	}
}

func TestLastFromTranscriptNormalizesCodexGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "codex.jsonl")
	content := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"functions.request_user_input","input":{"questions":[{"id":"choice","header":"Choice","question":"Which isolation?","options":[{"label":"Explicit paths","description":"owned files"},{"label":"Wait","description":"peer completion"}]}]}}]}}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	got, found, err := LastFromTranscript(path, "codex")
	if err != nil {
		t.Fatal(err)
	}
	if !found || got.Harness != "codex" || got.Kind != ChooseApproach || got.Question != "Which isolation?" || len(got.Options) != 2 {
		t.Fatalf("found=%v got=%+v", found, got)
	}
}
