package gateway

import (
	"regexp"
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

var runningCommandPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bwhile\s+running\s+(.+)$`),
	regexp.MustCompile(`(?i)\brunning\s+command:?\s+(.+)$`),
	regexp.MustCompile(`(?i)\bcommand\s+["']([^"']+)["']\s+(?:exited|failed|terminated)`),
}

// toolFailureNotes detects known tool-executor failures in replayed tool results. The
// first shipped detector is intentionally narrow: a Bash/shell git or gh command that
// came back with the exit-143 hang signature this Windows host sees when git/gh is routed
// through the Bash tool instead of native PowerShell.
func toolFailureNotes(messages []agent.Message) []toolFailureNote {
	nameByID := toolCallNamesByID(messages)
	out := make([]toolFailureNote, 0, 1)
	for _, m := range messages {
		if m.Role != agent.RoleTool {
			continue
		}
		spec, ok := auditreason.ToolFailureFromMessage(m.Content)
		if !ok || spec.Token != auditreason.ToolFailureHangShellMismatch {
			continue
		}
		command, ok := extractGitGhCommand(m.Content)
		if !ok {
			continue
		}
		tool := m.Name
		if tool == "" {
			tool = nameByID[m.ToolCallID]
		}
		if tool != "" && !isShellToolName(tool) {
			continue
		}
		out = append(out, toolFailureNote{
			ToolCallID: m.ToolCallID,
			Tool:       tool,
			Command:    command,
			Recovery:   powershellRecoveryCommand(command),
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
		return "[fak] " + token + ": Bash git/gh command ended with exit 143; retry from native PowerShell: " + notes[0].Recovery
	}
	recoveries := make([]string, 0, len(notes))
	for _, n := range notes {
		recoveries = append(recoveries, n.Recovery)
	}
	return "[fak] " + token + ": " + itoa(uint64(len(notes))) + " Bash git/gh commands ended with exit 143; retry from native PowerShell: " + strings.Join(recoveries, " ; ")
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

func toolCallNamesByID(messages []agent.Message) map[string]string {
	names := map[string]string{}
	for _, m := range messages {
		if m.Role != agent.RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.ID != "" && tc.Function.Name != "" {
				names[tc.ID] = tc.Function.Name
			}
		}
	}
	return names
}

func isShellToolName(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	return name == "bash" || name == "shell"
}

func extractGitGhCommand(text string) (string, bool) {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		for _, candidate := range commandCandidates(line) {
			if cmd, ok := normalizeGitGhCommand(candidate); ok {
				return cmd, true
			}
		}
	}
	return "", false
}

func commandCandidates(line string) []string {
	candidates := []string{line}
	trimmed := strings.TrimLeft(line, "-* ")
	for _, prefix := range []string{"command:", "cmd:", "$", ">"} {
		if rest, ok := stripCasePrefix(trimmed, prefix); ok {
			candidates = append(candidates, rest)
		}
	}
	for _, re := range runningCommandPatterns {
		if m := re.FindStringSubmatch(line); len(m) == 2 {
			candidates = append(candidates, m[1])
		}
	}
	return candidates
}

func stripCasePrefix(s, prefix string) (string, bool) {
	if strings.HasPrefix(strings.ToLower(s), prefix) {
		return strings.TrimSpace(s[len(prefix):]), true
	}
	return "", false
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
