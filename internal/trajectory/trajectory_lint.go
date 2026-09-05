package trajectory

import (
	"fmt"
	"strings"
	"testing"
)

// FileReadToolRatio audits the ratio of structured file read tools (e.g. fak_read, read)
// versus ad-hoc shell inspection commands (e.g. cat, head, tail) executed via shell tools.
type FileReadToolRatio struct {
	FakReadCalls       int     `json:"fak_read_calls"`
	ExecCommandCalls   int     `json:"exec_command_calls"`
	ExecReadCalls      int     `json:"exec_read_calls"`
	EffectiveRatio     float64 `json:"effective_ratio"`
	MinAcceptableRatio float64 `json:"min_acceptable_ratio"`
	Passed             bool    `json:"passed"`
	Defect             string  `json:"defect,omitempty"`
}

// ToolCallRecord records a single tool invocation with its optional command payload.
type ToolCallRecord struct {
	Tool    string `json:"tool"`
	Command string `json:"command,omitempty"`
}

var readCommandVerbs = map[string]bool{
	"cat":           true,
	"head":          true,
	"tail":          true,
	"get-content":   true,
	"gc":            true,
	"type":          true,
	"get-childitem": true,
	"gci":           true,
	"dir":           true,
}

func isStructuredReadTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	switch t {
	case "fak_read", "read", "fak_fak_read", "mcp__fak__fak_read", "mcp__fak_guard__fak_read":
		return true
	default:
		return strings.HasSuffix(t, "fak_read") || strings.HasSuffix(t, ".read") || strings.HasSuffix(t, "__read")
	}
}

func isShellTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	switch t {
	case "exec_command", "bash", "sh", "powershell", "pwsh", "cmd", "shell_command", "functions.shell_command":
		return true
	default:
		return strings.HasSuffix(t, "exec_command") || strings.HasSuffix(t, "shell_command")
	}
}

func isReadCommand(cmd string) bool {
	segments := strings.FieldsFunc(cmd, func(r rune) bool {
		return r == ';' || r == '|' || r == '&' || r == '\n'
	})
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" || strings.Contains(seg, ">") {
			continue
		}
		words := strings.Fields(seg)
		if len(words) == 0 {
			continue
		}
		for _, w := range words {
			if strings.Contains(w, "=") && !strings.HasPrefix(w, "-") {
				continue
			}
			clean := strings.Trim(w, "\"'`")
			if idx := strings.LastIndexAny(clean, "/\\"); idx >= 0 {
				clean = clean[idx+1:]
			}
			clean = strings.TrimSuffix(strings.ToLower(clean), ".exe")
			if clean == "sudo" || clean == "nohup" || clean == "env" || clean == "time" {
				continue
			}
			if clean == "cmd" || clean == "powershell" || clean == "pwsh" || clean == "bash" || clean == "sh" {
				continue
			}
			if strings.HasPrefix(clean, "-") {
				continue
			}
			if readCommandVerbs[clean] {
				return true
			}
			break
		}
	}
	return false
}

// LintFileReadToolSelection audits tool invocations to evaluate whether file inspections
// use structured tools (fak_read, read) or shell commands.
func LintFileReadToolSelection(calls []ToolCallRecord, minRatio float64) FileReadToolRatio {
	ratio := FileReadToolRatio{
		MinAcceptableRatio: minRatio,
	}

	for _, c := range calls {
		if isStructuredReadTool(c.Tool) {
			ratio.FakReadCalls++
		} else if isShellTool(c.Tool) {
			ratio.ExecCommandCalls++
			if isReadCommand(c.Command) {
				ratio.ExecReadCalls++
			}
		}
	}

	totalReads := ratio.FakReadCalls + ratio.ExecReadCalls
	if totalReads == 0 {
		ratio.EffectiveRatio = 1.0
	} else {
		ratio.EffectiveRatio = float64(ratio.FakReadCalls) / float64(totalReads)
	}

	if ratio.EffectiveRatio >= minRatio {
		ratio.Passed = true
	} else {
		ratio.Passed = false
		ratio.Defect = fmt.Sprintf("file read tool ratio %.2f is below minimum acceptable ratio %.2f (fak_read: %d, exec_read: %d, exec_command: %d)",
			ratio.EffectiveRatio, minRatio, ratio.FakReadCalls, ratio.ExecReadCalls, ratio.ExecCommandCalls)
	}

	return ratio
}

// AssertFileReadToolSelectionRatio fails t if ratio does not pass.
func AssertFileReadToolSelectionRatio(t *testing.T, ratio FileReadToolRatio) {
	t.Helper()
	if !ratio.Passed {
		t.Fatalf("file read tool selection ratio check failed: %s", ratio.Defect)
	}
}

// TimeoutRecoveryAudit records metrics on subagent wait timeouts and recovery behavior.
type TimeoutRecoveryAudit struct {
	WaitAgentCalls            int    `json:"wait_agent_calls"`
	Timeouts                  int    `json:"timeouts"`
	StalledWithoutCancel      int    `json:"stalled_without_cancel"`
	FallbackToMonolithicShell int    `json:"fallback_to_monolithic_shell"`
	Passed                    bool   `json:"passed"`
	Defect                    string `json:"defect,omitempty"`
}

func isWaitAgentEvent(e string) bool {
	lower := strings.ToLower(e)
	return strings.Contains(lower, "wait_agent") || strings.Contains(lower, "waitagent")
}

func isTimeoutEvent(e string) bool {
	lower := strings.ToLower(e)
	return strings.Contains(lower, "timeout")
}

func isCloseAgentEvent(e string) bool {
	lower := strings.ToLower(e)
	return strings.Contains(lower, "close_agent") || strings.Contains(lower, "closeagent")
}

func isSpawnAgentEvent(e string) bool {
	lower := strings.ToLower(e)
	return strings.Contains(lower, "spawn_agent") || strings.Contains(lower, "spawnagent")
}

func isShellExecEvent(e string) bool {
	lower := strings.ToLower(e)
	return strings.Contains(lower, "exec_command") || strings.Contains(lower, "shell_command") || lower == "bash" || lower == "sh" || lower == "powershell"
}

// LintSubagentTimeoutRecovery audits event sequences to ensure wait_agent timeouts
// are recovered by close_agent and re-dispatch rather than monolithic shell loops.
func LintSubagentTimeoutRecovery(events []string) TimeoutRecoveryAudit {
	var audit TimeoutRecoveryAudit
	audit.Passed = true

	for i, ev := range events {
		if isWaitAgentEvent(ev) {
			audit.WaitAgentCalls++
		}
		if isTimeoutEvent(ev) {
			audit.Timeouts++
			if audit.WaitAgentCalls < audit.Timeouts {
				audit.WaitAgentCalls = audit.Timeouts
			}

			closed := false
			spawned := false
			shellCalls := 0

			for j := i + 1; j < len(events); j++ {
				next := events[j]
				if isTimeoutEvent(next) {
					break
				}
				if isCloseAgentEvent(next) {
					closed = true
				}
				if isSpawnAgentEvent(next) {
					spawned = true
					break
				}
				if isShellExecEvent(next) {
					shellCalls++
				}
			}

			if !closed {
				audit.StalledWithoutCancel++
			}
			if shellCalls >= 5 || (!spawned && shellCalls > 0 && !closed) {
				audit.FallbackToMonolithicShell++
			}
		}
	}

	if audit.StalledWithoutCancel > 0 || audit.FallbackToMonolithicShell > 0 {
		audit.Passed = false
		var defects []string
		if audit.StalledWithoutCancel > 0 {
			defects = append(defects, fmt.Sprintf("stalled subagent was not canceled with close_agent (%d time(s))", audit.StalledWithoutCancel))
		}
		if audit.FallbackToMonolithicShell > 0 {
			defects = append(defects, fmt.Sprintf("fallback to monolithic shell execution detected (%d time(s))", audit.FallbackToMonolithicShell))
		}
		audit.Defect = strings.Join(defects, "; ")
	}

	return audit
}
