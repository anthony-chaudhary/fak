package gateway

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/auditreason"
)

type toolFailureNote struct {
	ToolCallID string
	Tool       string
	Command    string
	Recovery   string
	Token      auditreason.ToolFailure
}

// toolFailureNotes detects known tool-executor failures in replayed tool results.
// A result body is only the failure witness: the command is recovered from the
// paired assistant tool call by ToolCallID, so quoted transcripts cannot trigger
// a recovery note.
func toolFailureNotes(messages []agent.Message) []toolFailureNote {
	callByID := toolCallsByID(messages)
	gitGhShellSucceeded := hasSuccessfulGitGhShellCall(messages, callByID)
	out := make([]toolFailureNote, 0, 1)
	for _, m := range messages {
		if m.Role != agent.RoleTool {
			continue
		}
		spec, ok := auditreason.ToolFailureFromMessage(m.Content)
		if !ok || spec.Token != auditreason.ToolFailureHangShellMismatch {
			continue
		}
		call, ok := callByID[m.ToolCallID]
		if !ok {
			continue
		}
		tool := call.Tool
		if tool == "" {
			tool = m.Name
		}
		if tool != "" && !isShellToolName(tool) {
			continue
		}
		command, ok := normalizeGitGhCommand(call.Command)
		if !ok {
			continue
		}
		out = append(out, toolFailureNote{
			ToolCallID: m.ToolCallID,
			Tool:       tool,
			Command:    command,
			Recovery:   exit143Recovery(command, m.Content, gitGhShellSucceeded),
			Token:      spec.Token,
		})
	}
	return out
}

func toolFailureNoteText(notes []toolFailureNote) string {
	if len(notes) == 0 {
		return ""
	}
	token := string(notes[0].Token)
	if len(notes) == 1 {
		return "[fak] " + token + ": Bash git/gh command ended with exit 143; " + notes[0].Recovery
	}
	recoveries := make([]string, 0, len(notes))
	for _, n := range notes {
		recoveries = append(recoveries, n.Recovery)
	}
	return "[fak] " + token + ": " + itoa(uint64(len(notes))) + " Bash git/gh commands ended with exit 143; recovery: " + strings.Join(recoveries, " ; ")
}

func (s *Server) toolFailureNoteOnce(trace string, messages []agent.Message) string {
	notes := toolFailureNotes(messages)
	if s == nil || trace == "" {
		return toolFailureNoteText(notes)
	}
	fresh := make([]toolFailureNote, 0, len(notes))
	s.notedToolFailuresMu.Lock()
	if s.notedToolFailures == nil {
		s.notedToolFailures = map[string]map[string]struct{}{}
	}
	if len(s.notedToolFailures) >= maxResetHealthSessions {
		for k := range s.notedToolFailures {
			delete(s.notedToolFailures, k)
			break
		}
	}
	seen := s.notedToolFailures[trace]
	if seen == nil {
		seen = map[string]struct{}{}
		s.notedToolFailures[trace] = seen
	}
	for _, n := range notes {
		key := toolFailureNoteKey(n)
		if _, already := seen[key]; already {
			continue
		}
		seen[key] = struct{}{}
		fresh = append(fresh, n)
	}
	s.notedToolFailuresMu.Unlock()
	return toolFailureNoteText(fresh)
}

type toolCallFailureContext struct {
	Tool    string
	Command string
}

func toolCallsByID(messages []agent.Message) map[string]toolCallFailureContext {
	calls := map[string]toolCallFailureContext{}
	for _, m := range messages {
		if m.Role != agent.RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.ID == "" {
				continue
			}
			command, _ := toolCallCommand(tc.Function.Arguments)
			calls[tc.ID] = toolCallFailureContext{Tool: tc.Function.Name, Command: command}
		}
	}
	return calls
}

func toolCallCommand(args string) (string, bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(args), &obj); err != nil {
		return "", false
	}
	for _, key := range []string{"command", "cmd"} {
		raw := obj[key]
		if len(raw) == 0 {
			continue
		}
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s), true
		}
	}
	return "", false
}

func hasSuccessfulGitGhShellCall(messages []agent.Message, callByID map[string]toolCallFailureContext) bool {
	for _, m := range messages {
		if m.Role != agent.RoleTool || m.ToolCallID == "" {
			continue
		}
		call, ok := callByID[m.ToolCallID]
		if !ok {
			continue
		}
		tool := m.Name
		if tool == "" {
			tool = call.Tool
		}
		if !isShellToolName(tool) {
			continue
		}
		if _, ok := normalizeGitGhCommand(call.Command); !ok {
			continue
		}
		if _, failed := auditreason.ToolFailureFromMessage(m.Content); failed {
			continue
		}
		return true
	}
	return false
}

func isShellToolName(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return name == "bash" || name == "shell"
}

func normalizeGitGhCommand(candidate string) (string, bool) {
	cmd := strings.TrimSpace(candidate)
	cmd = strings.Trim(cmd, "`")
	cmd = strings.TrimSpace(strings.Trim(cmd, `"'`))
	if inner, ok := unwrapBashCommand(cmd); ok {
		cmd = inner
	}
	cmd = strings.TrimSpace(strings.Trim(cmd, `"'`))
	cmd = trimTrailingExitProse(cmd)
	if startsGitGh(cmd) {
		return cmd, true
	}
	return "", false
}

func unwrapBashCommand(cmd string) (string, bool) {
	low := strings.ToLower(cmd)
	for _, prefix := range []string{"bash -lc ", "bash -c ", "sh -lc ", "sh -c "} {
		if strings.HasPrefix(low, prefix) {
			inner := strings.TrimSpace(cmd[len(prefix):])
			inner = strings.Trim(inner, `"'`)
			return inner, true
		}
	}
	return "", false
}

func trimTrailingExitProse(cmd string) string {
	for _, marker := range []string{" exited with ", " failed with ", " terminated with "} {
		if i := strings.Index(strings.ToLower(cmd), marker); i >= 0 {
			return strings.TrimSpace(cmd[:i])
		}
	}
	return cmd
}

func startsGitGh(cmd string) bool {
	cmd = strings.TrimSpace(cmd)
	return cmd == "git" || cmd == "gh" || strings.HasPrefix(cmd, "git ") || strings.HasPrefix(cmd, "gh ")
}

func powershellRecoveryCommand(command string) string {
	return `powershell -NoProfile -Command "` + escapePowerShellDoubleQuoted(command) + `"`
}

func exit143Recovery(command, result string, gitGhShellSucceeded bool) string {
	if hasShellMismatchEvidence(result) && !gitGhShellSucceeded {
		return "retry from native PowerShell: " + powershellRecoveryCommand(command)
	}
	return lockLoadRecovery(command, gitGhShellSucceeded)
}

func hasShellMismatchEvidence(result string) bool {
	low := strings.ToLower(result)
	for _, needle := range []string{
		"shell mismatch",
		"syntax error near unexpected token",
		"is not recognized as",
		"running scripts is disabled",
	} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

func lockLoadRecovery(command string, gitGhShellSucceeded bool) string {
	prefix := "check .git/*.lock and host load, stop wedged git/gh processes, then retry with a bounded timeout"
	if gitGhShellSucceeded {
		prefix = "git/gh also succeeded through Bash in this transcript; check .git/*.lock and host load, stop wedged git/gh processes, then retry with a bounded timeout"
	}
	return prefix + ": " + command
}

func escapePowerShellDoubleQuoted(s string) string {
	s = strings.ReplaceAll(s, "`", "``")
	s = strings.ReplaceAll(s, `"`, "`\"")
	return s
}

func toolFailureNoteKey(n toolFailureNote) string {
	if n.ToolCallID != "" {
		return n.ToolCallID
	}
	return string(n.Token) + "|" + n.Command
}
