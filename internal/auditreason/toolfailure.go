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
type ToolFailureSpec struct {
	Token     ToolFailure `json:"token"`
	Summary   string      `json:"summary"`
	Fix       string      `json:"fix"`
	Retryable bool        `json:"retryable"`
}

var toolFailureSpecs = []ToolFailureSpec{
	{
		Token:     ToolFailureHang,
		Summary:   "tool process stopped making progress and produced no terminal result",
		Fix:       "reap or interrupt the stuck process, verify any affected state from disk or the durable service, then rerun with a bounded timeout or narrower command",
		Retryable: true,
	},
	{
		Token:     ToolFailureTimeout,
		Summary:   "tool exceeded an explicit wall-clock budget before returning a complete result",
		Fix:       "read back the operation state, raise the budget only when the workload is expected to exceed it, otherwise narrow the command and rerun",
		Retryable: true,
	},
	{
		Token:     ToolFailureShellMismatch,
		Summary:   "command syntax was routed to an incompatible shell or operating-system environment",
		Fix:       "rerun with the matching shell and syntax; on Windows use PowerShell syntax natively or invoke WSL bash explicitly for POSIX syntax",
		Retryable: true,
	},
	{
		Token:     ToolFailureHangShellMismatch,
		Summary:   "a shell/environment mismatch presented as a hung or externally-terminated tool, commonly git or gh under the wrong Windows shell",
		Fix:       "stop the wedged process, rerun git/gh from native PowerShell or an explicit WSL shell, and avoid mixed shell quoting for the retry",
		Retryable: true,
	},
	{
		Token:     ToolFailurePartialApply,
		Summary:   "a mutating tool failed after applying only part of the requested change",
		Fix:       "read back the affected files or service state, keep only verified successful effects, then reapply the missing change idempotently; do not claim success from the transcript alone",
		Retryable: false,
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
	{[]string{"tool_timeout", "context deadline exceeded", "timed out", "timeout exceeded", "command timed out"}, ToolFailureTimeout},
	{[]string{"tool_shell_mismatch", "shell mismatch", "syntax error near unexpected token", "is not recognized as", "cannot be loaded because running scripts is disabled"}, ToolFailureShellMismatch},
	{[]string{"tool_partial_apply", "partial apply", "partially applied", "partial mutation", "applied only part"}, ToolFailurePartialApply},
	{[]string{"tool_hang", "hung", "hang detected", "no output for", "stopped making progress"}, ToolFailureHang},
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
