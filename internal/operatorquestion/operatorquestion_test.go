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

func TestLastFromTranscriptAnyPreservesCrossHarnessOrder(t *testing.T) {
	claude := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"AskUserQuestion","input":{"questions":[{"header":"Old","multiSelect":false,"question":"Claude old?","options":[]}]}}]}}`
	codex := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"functions.request_user_input","input":{"questions":[{"id":"new","header":"New","question":"Codex new?","options":[]}]}}]}}`
	invalid := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"UnknownGate","input":{"bad":true}}]}}`
	for _, tc := range []struct {
		name, content, wantHarness, wantQuestion string
	}{
		{"codex last", claude + "\n" + invalid + "\n" + codex + "\n", "codex", "Codex new?"},
		{"claude last", codex + "\n" + invalid + "\n" + claude + "\n", "claude", "Claude old?"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "mixed.jsonl")
			if err := os.WriteFile(path, []byte(tc.content), 0o600); err != nil {
				t.Fatal(err)
			}
			got, found, err := LastFromTranscriptAny(path)
			if err != nil {
				t.Fatal(err)
			}
			if !found || got.Harness != tc.wantHarness || got.Question != tc.wantQuestion {
				t.Fatalf("found=%v got=%+v", found, got)
			}
		})
	}
}

func TestKindValidity(t *testing.T) {
	valid := []Kind{Clarify, ChooseApproach, PlanApproval, Permission, ConfirmIrreversible}
	for _, k := range valid {
		if !k.Valid() {
			t.Errorf("expected Kind %q to be valid", k)
		}
	}
	invalid := []Kind{"", "INVALID", "UNKNOWN"}
	for _, k := range invalid {
		if k.Valid() {
			t.Errorf("expected Kind %q to be invalid", k)
		}
	}
}

func TestToSignalSeverity(t *testing.T) {
	tests := []struct {
		kind         Kind
		wantSeverity string
	}{
		{Permission, "decision"},
		{ConfirmIrreversible, "decision"},
		{PlanApproval, "decision"},
		{Clarify, "action"},
		{ChooseApproach, "action"},
		{Kind("OTHER"), "action"},
	}
	for _, tc := range tests {
		q := OperatorQuestion{Kind: tc.kind, Harness: "claude", Question: "test"}
		sig := q.ToSignal()
		if sig.Severity != tc.wantSeverity {
			t.Errorf("kind %s: want severity %q, got %q", tc.kind, tc.wantSeverity, sig.Severity)
		}
		if sig.Source != "operator-question:claude" {
			t.Errorf("unexpected source: %q", sig.Source)
		}
	}
}

func TestNormalizeValidationAndRejection(t *testing.T) {
	t.Run("unsupported tool on Claude", func(t *testing.T) {
		_, err := Normalize(NativeGate{HarnessCommand: "claude", Tool: "Bash", Payload: []byte(`{}`)})
		if err == nil {
			t.Fatal("expected error for non-gate tool")
		}
	})

	t.Run("unsupported tool on Codex", func(t *testing.T) {
		_, err := Normalize(NativeGate{HarnessCommand: "codex", Tool: "execute_command", Payload: []byte(`{}`)})
		if err == nil {
			t.Fatal("expected error for non-gate tool")
		}
	})

	t.Run("multiple questions rejected in Claude", func(t *testing.T) {
		payload := []byte(`{"questions":[{"question":"Q1"},{"question":"Q2"}]}`)
		_, err := Normalize(NativeGate{HarnessCommand: "claude", Tool: "AskUserQuestion", Payload: payload})
		if err == nil {
			t.Fatal("expected error for multiple questions")
		}
	})

	t.Run("multiple questions rejected in Codex", func(t *testing.T) {
		payload := []byte(`{"questions":[{"id":"1","header":"h","question":"Q1"},{"id":"2","header":"h","question":"Q2"}]}`)
		_, err := Normalize(NativeGate{HarnessCommand: "codex", Tool: "request_user_input", Payload: payload})
		if err == nil {
			t.Fatal("expected error for multiple questions")
		}
	})

	t.Run("single option maps to Clarify", func(t *testing.T) {
		payload := []byte(`{"questions":[{"question":"Confirm single?","options":[{"label":"OK","description":"Proceed"}]}]}`)
		q, err := Normalize(NativeGate{HarnessCommand: "claude", Tool: "AskUserQuestion", Payload: payload})
		if err != nil {
			t.Fatal(err)
		}
		if q.Kind != Clarify {
			t.Fatalf("want Clarify for single option, got %s", q.Kind)
		}
	})

	t.Run("multiple trailing JSON rejected", func(t *testing.T) {
		payload := []byte(`{"questions":[{"question":"Valid?"}]}{"extra":true}`)
		_, err := Normalize(NativeGate{HarnessCommand: "claude", Tool: "AskUserQuestion", Payload: payload})
		if err == nil {
			t.Fatal("expected error for trailing JSON tokens")
		}
	})
}

func TestTranscriptErrorHandling(t *testing.T) {
	t.Run("nonexistent file", func(t *testing.T) {
		_, _, err := LastFromTranscript("nonexistent-path.jsonl", "claude")
		if err == nil {
			t.Fatal("expected error on nonexistent file")
		}
		_, _, err = LastFromTranscriptAny("nonexistent-path.jsonl")
		if err == nil {
			t.Fatal("expected error on nonexistent file")
		}
	})

	t.Run("corrupt JSON", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "corrupt.jsonl")
		if err := os.WriteFile(path, []byte(`{not valid json}`), 0o600); err != nil {
			t.Fatal(err)
		}
		_, _, err := LastFromTranscript(path, "claude")
		if err == nil {
			t.Fatal("expected error on corrupt json line")
		}
		_, _, err = LastFromTranscriptAny(path)
		if err == nil {
			t.Fatal("expected error on corrupt json line")
		}
	})
}
