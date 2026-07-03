package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestToolFailureNotesDetectsBashGitExit143(t *testing.T) {
	messages := []agent.Message{
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{
					ID: "call_ok", Type: "function",
					Function: agent.Func{Name: "Bash", Arguments: `{"command":"git branch --show-current"}`},
				},
				{
					ID: "call_git", Type: "function",
					Function: agent.Func{Name: "Bash", Arguments: `{"command":"git status"}`},
				},
			},
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "call_ok",
			Content:    "main",
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "call_git",
			Content:    "Exit code 143\nCommand timed out after 2m 0s",
		},
	}

	notes := toolFailureNotes(messages)
	if len(notes) != 1 {
		t.Fatalf("notes = %d, want 1: %+v", len(notes), notes)
	}
	if notes[0].Command != "git status" {
		t.Fatalf("command = %q, want git status", notes[0].Command)
	}
	text := toolFailureNoteText(notes)
	for _, want := range []string{"TOOL_HANG_SHELL_MISMATCH", "git status", "git/gh also succeeded through Bash", ".git/*.lock"} {
		if !strings.Contains(text, want) {
			t.Fatalf("note missing %q: %s", want, text)
		}
	}
	if strings.Contains(text, "powershell -NoProfile") {
		t.Fatalf("timeout without shell-mismatch evidence should not advise PowerShell: %s", text)
	}
}

func TestToolFailureNotesDetectsWrappedGhCommand(t *testing.T) {
	messages := []agent.Message{
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{{
				ID: "call_gh", Type: "function",
				Function: agent.Func{Name: "Bash", Arguments: `{"command":"bash -lc 'gh issue view 2128 --json state'"}`},
			}},
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "call_gh",
			Content:    "terminated with 143",
		},
	}

	notes := toolFailureNotes(messages)
	if len(notes) != 1 {
		t.Fatalf("notes = %d, want 1: %+v", len(notes), notes)
	}
	if notes[0].Command != "gh issue view 2128 --json state" {
		t.Fatalf("command = %q", notes[0].Command)
	}
	if !strings.Contains(notes[0].Recovery, `gh issue view 2128 --json state`) {
		t.Fatalf("recovery command not pre-filled: %s", notes[0].Recovery)
	}
}

func TestToolFailureNotesPowerShellOnlyWithShellMismatchEvidence(t *testing.T) {
	messages := []agent.Message{
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{{
				ID: "call_git", Type: "function",
				Function: agent.Func{Name: "Bash", Arguments: `{"command":"git status"}`},
			}},
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "call_git",
			Content:    "shell mismatch: git command exited with exit status 143",
		},
	}

	text := toolFailureNoteText(toolFailureNotes(messages))
	if !strings.Contains(text, `powershell -NoProfile -Command "git status"`) {
		t.Fatalf("shell mismatch should advise native PowerShell: %s", text)
	}
}

func TestToolFailureNotesIgnoreNonMatchingFailures(t *testing.T) {
	cases := []struct {
		name string
		msg  agent.Message
	}{
		{
			name: "non shell tool",
			msg:  agent.Message{Role: agent.RoleTool, Name: "Read", Content: "Command: git status\nexit status 143"},
		},
		{
			name: "non git command",
			msg:  agent.Message{Role: agent.RoleTool, Name: "Bash", Content: "Command: npm test\nexit status 143"},
		},
		{
			name: "non hang",
			msg:  agent.Message{Role: agent.RoleTool, Name: "Bash", Content: "Command: git status\nexit status 1"},
		},
		{
			name: "quoted transcript without paired tool call",
			msg:  agent.Message{Role: agent.RoleTool, Name: "Bash", Content: "forensics dump\nCMD: git status\nExit code 143\nCommand timed out after 2m 0s"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if notes := toolFailureNotes([]agent.Message{tc.msg}); len(notes) != 0 {
				t.Fatalf("unexpected notes: %+v", notes)
			}
		})
	}
}

func TestToolFailureNoteOnceDedup(t *testing.T) {
	s := &Server{}
	messages := []agent.Message{
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{{
				ID: "call_gh", Type: "function",
				Function: agent.Func{Name: "Bash", Arguments: `{"command":"gh issue list"}`},
			}},
		},
		{
			Role:       agent.RoleTool,
			ToolCallID: "call_gh",
			Content:    "exit code 143",
		},
	}

	if got := s.toolFailureNoteOnce("trace-a", messages); got == "" {
		t.Fatal("first note should emit")
	}
	if got := s.toolFailureNoteOnce("trace-a", messages); got != "" {
		t.Fatalf("same replayed failure should be deduped, got: %s", got)
	}
	if got := s.toolFailureNoteOnce("trace-b", messages); got == "" {
		t.Fatal("different trace should emit independently")
	}
}
