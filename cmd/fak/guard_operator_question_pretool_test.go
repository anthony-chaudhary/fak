package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestGoalQuestionDecision(t *testing.T) {
	t.Run("active goal suppresses nonblocking question", func(t *testing.T) {
		got := decideGoalQuestion("goal-1", "AskUserQuestion", map[string]any{"question": "Which naming option do you prefer?"})
		if got.Action != "continue" || got.Reason != "ACTIVE_GOAL_QUESTION_SUPPRESSED" {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("genuine authority blocker escalates", func(t *testing.T) {
		got := decideGoalQuestion("goal-1", "AskUserQuestion", map[string]any{"question": "Authorize destructive production deletion?"})
		if got.Action != "escalate" || got.Reason != guardGoalQuestionNeedsHuman {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("interactive session passes through", func(t *testing.T) {
		if got := decideGoalQuestion("", "AskUserQuestion", map[string]any{"question": "Choose"}); got.Action != "allow" {
			t.Fatalf("got %+v", got)
		}
	})
}

func TestRunGuardGoalQuestionHookEmitsTypedDeny(t *testing.T) {
	payload := `{"tool_name":"AskUserQuestion","tool_input":{"question":"Which harmless name?"}}`
	var stdout bytes.Buffer
	if code := runGuardGoalQuestionHook(&stdout, &bytes.Buffer{}, strings.NewReader(payload), []string{"--goal", "goal-1"}); code != 0 {
		t.Fatalf("code=%d", code)
	}
	var got guardGoalQuestionHookOutput
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.HookSpecificOutput.PermissionDecision != "deny" || !strings.Contains(got.HookSpecificOutput.PermissionDecisionReason, "continue autonomously") {
		t.Fatalf("output=%s", stdout.String())
	}
}

func TestGuardGoalQuestionMatcherWiring(t *testing.T) {
	m := guardGoalQuestionMatchers("fak")
	if m.Matcher != "AskUserQuestion" || len(m.Hooks) != 1 {
		t.Fatalf("matcher=%+v", m)
	}
	if got := m.Hooks[0].Args; len(got) != 1 || got[0] != "guard-goal-question" {
		t.Fatalf("args=%v", got)
	}
}
