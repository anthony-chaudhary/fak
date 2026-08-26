package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSessionsCodexLoopHookInjectsWorkflowDefaultOnceForGuardedSession(t *testing.T) {
	home, sessionID := writeCodexHookSession(t, "fak")
	if err := writeCodexGuardWitness(home, sessionID); err != nil {
		t.Fatalf("write guard witness: %v", err)
	}
	input := `{"hook_event_name":"UserPromptSubmit","session_id":"` + sessionID + `","prompt":"implement the multi-step feature and verify it"}`

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader(input), []string{"--hardened", "--codex-home", home}); code != 0 {
		t.Fatalf("hook code=%d stderr=%q", code, stderr.String())
	}
	var output codexLoopHookOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatalf("decode output %q: %v", stdout.String(), err)
	}
	if output.additionalContext() == "" || !strings.Contains(output.additionalContext(), "ultracode-style workflow generation") {
		t.Fatalf("additionalContext=%q, want guarded workflow default", output.additionalContext())
	}
	if output.Continue == nil || !*output.Continue {
		t.Fatalf("continue=%v, want true", output.Continue)
	}
	witnessPath := filepath.Join(home, "fak-workflow-defaults", sessionID+".json")
	raw, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("read workflow witness: %v", err)
	}
	var witness codexWorkflowDefaultWitness
	if err := json.Unmarshal(raw, &witness); err != nil {
		t.Fatalf("decode workflow witness: %v", err)
	}
	if witness.Classification != "consider-workflow" || witness.Decision != "inject" {
		t.Fatalf("witness=%+v", witness)
	}

	stdout.Reset()
	stderr.Reset()
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader(input), []string{"--hardened", "--codex-home", home}); code != 0 {
		t.Fatalf("second hook code=%d stderr=%q", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("second prompt output=%q, want no repeated context", stdout.String())
	}
}

func TestSessionsCodexLoopHookInjectsWorkflowDefaultOnActiveGuardFastPath(t *testing.T) {
	home := t.TempDir()
	sessionID := "active-guard-workflow-default"
	t.Setenv(guardActiveEnv, "1")
	input := `{"hook_event_name":"UserPromptSubmit","session_id":"` + sessionID + `","prompt":"implement the multi-step feature and verify it"}`

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader(input), []string{"--codex-home", home}); code != 0 {
		t.Fatalf("hook code=%d stderr=%q", code, stderr.String())
	}
	var output codexLoopHookOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatalf("decode output %q: %v", stdout.String(), err)
	}
	if output.additionalContext() == "" {
		t.Fatal("active fak guard fast path did not inject workflow default")
	}
	if !codexGuardWitnessExists(home, sessionID) || !codexWorkflowDefaultWitnessExists(home, sessionID) {
		t.Fatal("active guard fast path did not persist both guard and workflow witnesses")
	}
}

func TestCodexWorkflowDefaultClassifiesShortPromptAsLikelyDirect(t *testing.T) {
	home := t.TempDir()
	output, ok := codexWorkflowDefaultOutput(codexLoopHookInput{SessionID: "short-prompt", Prompt: "fix typo"}, home)
	if !ok || output.additionalContext() == "" {
		t.Fatalf("output=%+v ok=%v, want injected first-turn default", output, ok)
	}
	raw, err := os.ReadFile(filepath.Join(home, "fak-workflow-defaults", "short-prompt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var witness codexWorkflowDefaultWitness
	if err := json.Unmarshal(raw, &witness); err != nil {
		t.Fatal(err)
	}
	if witness.Classification != "likely-direct" {
		t.Fatalf("classification=%q, want likely-direct", witness.Classification)
	}
}

func TestSessionsCodexLoopHookKeepsWorkflowDefaultOffHardenedDirectSession(t *testing.T) {
	home, sessionID := writeCodexHookSession(t, "openai")
	input := `{"hook_event_name":"UserPromptSubmit","session_id":"` + sessionID + `","prompt":"implement the multi-step feature"}`

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader(input), []string{"--hardened", "--codex-home", home}); code != 0 {
		t.Fatalf("hook code=%d stderr=%q", code, stderr.String())
	}
	var output codexLoopHookOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatalf("decode output %q: %v", stdout.String(), err)
	}
	if output.Decision != "block" || output.additionalContext() != "" {
		t.Fatalf("output=%+v, want direct-provider block without workflow injection", output)
	}
}
