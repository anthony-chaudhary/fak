package auditreason

import (
	"sort"
	"strings"
)

// ToolFailure is the closed vocabulary for non-guard tool failures. These are not
// policy refusals and do not belong in abi.ReasonCode or dos.toml [reasons.*];
// they describe failures of the tool transport/executor itself.
type ToolFailure string

const (
	ToolFailureHang              ToolFailure = "TOOL_HANG"
	ToolFailureTimeout           ToolFailure = "TOOL_TIMEOUT"
	ToolFailureShellMismatch     ToolFailure = "TOOL_SHELL_MISMATCH"
	ToolFailureHangShellMismatch ToolFailure = "TOOL_HANG_SHELL_MISMATCH"
	ToolFailurePartialApply      ToolFailure = "TOOL_PARTIAL_APPLY"
)

// ToolFailureSpec is the check-reason-like row for one non-guard tool failure.
// NextCommand is the pre-filled, runnable recovery the agent should try next, so
// it can branch on Token/NextCommand instead of regexing the failure prose. For
// the shell-mismatch classes it is a PowerShell template whose placeholder is
// filled with the exact failing command by ToolFailurePayloadForCommand.
type ToolFailureSpec struct {
	Token       ToolFailure `json:"token"`
	Summary     string      `json:"summary"`
	Fix         string      `json:"fix"`
	Retryable   bool        `json:"retryable"`
	NextCommand string      `json:"next_command"`
}

var toolFailureSpecs = []ToolFailureSpec{
	{
		Token:       ToolFailureHang,
		Summary:     "tool process stopped making progress and produced no terminal result",
		Fix:         "reap or interrupt the stuck process, verify any affected state from disk or the durable service, then rerun with a bounded timeout or narrower command",
		Retryable:   true,
		NextCommand: "Get-Process -Name git,gh -ErrorAction SilentlyContinue | Stop-Process -Force",
	},
	{
		Token:       ToolFailureTimeout,
		Summary:     "tool exceeded an explicit wall-clock budget before returning a complete result",
		Fix:         "read back the operation state, raise the budget only when the workload is expected to exceed it, otherwise narrow the command and rerun",
		Retryable:   true,
		NextCommand: "git status --short",
	},
	{
		Token:       ToolFailureShellMismatch,
		Summary:     "command syntax was routed to an incompatible shell or operating-system environment",
		Fix:         "rerun with the matching shell and syntax; on Windows use PowerShell syntax natively or invoke WSL bash explicitly for POSIX syntax",
		Retryable:   true,
		NextCommand: "powershell -NoProfile -Command '<rerun the command in PowerShell syntax>'",
	},
	{
		Token:       ToolFailureHangShellMismatch,
		Summary:     "a shell/environment mismatch presented as a hung or externally-terminated tool, commonly git or gh under the wrong Windows shell",
		Fix:         "stop the wedged process, rerun git/gh from native PowerShell or an explicit WSL shell, and avoid mixed shell quoting for the retry",
		Retryable:   true,
		NextCommand: "powershell -NoProfile -Command '<rerun the git/gh command in PowerShell>'",
	},
	{
		Token:       ToolFailurePartialApply,
		Summary:     "a mutating tool failed after applying only part of the requested change",
		Fix:         "read back the affected files or service state, keep only verified successful effects, then reapply the missing change idempotently; do not claim success from the transcript alone",
		Retryable:   false,
		NextCommand: "git status --short",
	},
}

// ToolFailures returns the complete closed table in deterministic token order.
func ToolFailures() []ToolFailureSpec {
	out := append([]ToolFailureSpec(nil), toolFailureSpecs...)
	sort.Slice(out, func(i, j int) bool { return out[i].Token < out[j].Token })
	return out
}

// LookupToolFailure returns the metadata row for token. Token matching is case-insensitive
// and accepts hyphen/space spellings by normalizing them to underscores.
func LookupToolFailure(token string) (ToolFailureSpec, bool) {
	norm := normalizeToolFailureToken(token)
	for _, spec := range toolFailureSpecs {
		if string(spec.Token) == norm {
			return spec, true
		}
	}
	return ToolFailureSpec{}, false
}

func normalizeToolFailureToken(token string) string {
	token = strings.TrimSpace(token)
	token = strings.ReplaceAll(token, "-", "_")
	token = strings.ReplaceAll(token, " ", "_")
	return strings.ToUpper(token)
}

type toolFailureSignature struct {
	needles []string
	token   ToolFailure
}

var toolFailureSignatures = []toolFailureSignature{
	{[]string{"tool_hang_shell_mismatch", "exit status 143", "exit code 143", "terminated with 143", "signal: terminated"}, ToolFailureHangShellMismatch},
	{[]string{"tool_timeout", "context deadline exceeded", "timed out waiting for", "timeout exceeded", "command timed out"}, ToolFailureTimeout},
	{[]string{"tool_shell_mismatch", "shell mismatch", "syntax error near unexpected token", "is not recognized as", "cannot be loaded because running scripts is disabled", "exit status 127", "exit code 127", "command not found"}, ToolFailureShellMismatch},
	{[]string{"tool_partial_apply", "partial apply", "partially applied", "partial mutation", "applied only part"}, ToolFailurePartialApply},
	{[]string{"tool_hang", "tool hung", "hang detected", "no output for"}, ToolFailureHang},
}

// ToolFailureFromMessage classifies raw tool-failure prose into the closed non-guard
// vocabulary. The bool is false when no known signature matched.
func ToolFailureFromMessage(msg string) (ToolFailureSpec, bool) {
	low := strings.ToLower(msg)
	for _, sig := range toolFailureSignatures {
		for _, needle := range sig.needles {
			if strings.Contains(low, needle) {
				return LookupToolFailure(string(sig.token))
			}
		}
	}
	return ToolFailureSpec{}, false
}

// ToolFailurePayload is the structured, actionable outcome for a non-guard tool
// failure: the closed-vocabulary Code the agent branches on, the Cause and human
// Fix from that row, the observed Evidence that classified this instance, whether
// a retry is safe, and a pre-filled runnable NextCommand. It extends the same
// closed contract the guard refuse-reasons use (summary+fix) to EVERY tool
// outcome, so a hang or a generic failure is as branchable as a policy denial —
// the agent keys on Code and runs NextCommand instead of regexing English.
type ToolFailurePayload struct {
	Code        ToolFailure `json:"code"`
	Cause       string      `json:"cause"`
	Evidence    string      `json:"evidence"`
	Fix         string      `json:"fix"`
	Retryable   bool        `json:"retryable"`
	NextCommand string      `json:"next_command"`
}

// Payload projects a vocabulary row into a full outcome payload, attaching the
// observed evidence for this instance. The Cause is the row Summary; the
// NextCommand is the row's pre-filled recovery.
func (s ToolFailureSpec) Payload(evidence string) ToolFailurePayload {
	return ToolFailurePayload{
		Code:        s.Token,
		Cause:       s.Summary,
		Evidence:    strings.TrimSpace(evidence),
		Fix:         s.Fix,
		Retryable:   s.Retryable,
		NextCommand: s.NextCommand,
	}
}

// ToolFailurePayloadFromMessage classifies a raw tool-failure message into the
// closed vocabulary and returns the full structured payload, using the message as
// the observed evidence. The bool is false when no known signature matched, so a
// caller never fabricates a payload for an unrecognized failure.
func ToolFailurePayloadFromMessage(msg string) (ToolFailurePayload, bool) {
	spec, ok := ToolFailureFromMessage(msg)
	if !ok {
		return ToolFailurePayload{}, false
	}
	return spec.Payload(msg), true
}

// ToolFailurePayloadForCommand is ToolFailurePayloadFromMessage plus the exact
// failing command: for the shell-mismatch classes (where the recovery is "rerun
// this in PowerShell"), a non-empty cmd is folded into a literally-runnable
// PowerShell NextCommand instead of the placeholder template — the exact recovery
// pre-filled. Non-shell classes keep their static NextCommand.
func ToolFailurePayloadForCommand(msg, cmd string) (ToolFailurePayload, bool) {
	payload, ok := ToolFailurePayloadFromMessage(msg)
	if !ok {
		return ToolFailurePayload{}, false
	}
	if strings.TrimSpace(cmd) == "" {
		return payload, true
	}
	switch payload.Code {
	case ToolFailureShellMismatch, ToolFailureHangShellMismatch:
		payload.NextCommand = powershellRecovery(cmd)
	}
	return payload, true
}

// powershellRecovery wraps a failing shell command so it reruns natively under
// PowerShell — the canonical recovery for a git/gh command that hung or was
// mis-routed under Bash on this Windows host. Single quotes in cmd are doubled
// per PowerShell single-quoted-string escaping so the recovery stays runnable.
func powershellRecovery(cmd string) string {
	escaped := strings.ReplaceAll(strings.TrimSpace(cmd), "'", "''")
	return "powershell -NoProfile -Command '" + escaped + "'"
}
