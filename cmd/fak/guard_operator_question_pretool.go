package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

const guardGoalQuestionNeedsHuman = "GOAL_QUESTION_NEEDS_HUMAN"

type guardGoalQuestionHookInput struct {
	ToolName  string         `json:"tool_name"`
	ToolInput map[string]any `json:"tool_input"`
}

type guardGoalQuestionDecision struct {
	Action string
	Reason string
}

type guardGoalQuestionHookOutput struct {
	HookSpecificOutput struct {
		HookEventName            string `json:"hookEventName"`
		PermissionDecision       string `json:"permissionDecision"`
		PermissionDecisionReason string `json:"permissionDecisionReason"`
	} `json:"hookSpecificOutput"`
}

func decideGoalQuestion(activeGoal, toolName string, input map[string]any) guardGoalQuestionDecision {
	if strings.TrimSpace(activeGoal) == "" || !strings.EqualFold(strings.TrimSpace(toolName), "AskUserQuestion") {
		return guardGoalQuestionDecision{Action: "allow"}
	}
	text := strings.ToLower(fmt.Sprint(input))
	for _, marker := range []string{"credential", "secret", "permission", "authorize", "destructive", "legal", "safety", "human must", "cannot proceed"} {
		if strings.Contains(text, marker) {
			return guardGoalQuestionDecision{Action: "escalate", Reason: guardGoalQuestionNeedsHuman}
		}
	}
	return guardGoalQuestionDecision{Action: "continue", Reason: "ACTIVE_GOAL_QUESTION_SUPPRESSED"}
}

func runGuardGoalQuestionHook(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("guard-goal-question", flag.ContinueOnError)
	fs.SetOutput(stderr)
	goal := fs.String("goal", os.Getenv("FAK_GOAL_ID"), "active session goal id")
	if fs.Parse(argv) != nil {
		return 0
	}
	payload, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil {
		return 0
	}
	var input guardGoalQuestionHookInput
	if json.Unmarshal(payload, &input) != nil {
		return 0
	}
	decision := decideGoalQuestion(*goal, input.ToolName, input.ToolInput)
	if decision.Action == "allow" {
		return 0
	}
	var output guardGoalQuestionHookOutput
	output.HookSpecificOutput.HookEventName = "PreToolUse"
	output.HookSpecificOutput.PermissionDecision = "deny"
	output.HookSpecificOutput.PermissionDecisionReason = decision.Reason
	if decision.Action == "continue" {
		output.HookSpecificOutput.PermissionDecisionReason += ": continue autonomously toward the active goal; log assumptions instead of pausing"
	} else {
		output.HookSpecificOutput.PermissionDecisionReason += ": emit the typed blocker and stop; do not open an interactive prompt"
	}
	_ = json.NewEncoder(stdout).Encode(output)
	return 0
}

func cmdGuardGoalQuestionHook(argv []string) {
	os.Exit(runGuardGoalQuestionHook(os.Stdout, os.Stderr, os.Stdin, argv))
}
