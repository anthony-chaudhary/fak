package loopgate

import (
	"context"
	"regexp"
	"strings"
)

// ReasonUnwitnessedNarrationClaim is returned when a turn claims test
// completion/success without matching execution receipts in the trajectory.
const ReasonUnwitnessedNarrationClaim = "UNWITNESSED_NARRATION_CLAIM"

// ReceiptRecord represents an execution receipt for an action or tool call.
type ReceiptRecord struct {
	Tool         string   `json:"tool"`
	Command      []string `json:"command,omitempty"`
	Verdict      string   `json:"verdict,omitempty"` // "ALLOW", etc.
	ExitCode     int      `json:"exit_code"`
	OutputSHA256 string   `json:"output_sha256,omitempty"`
	Timestamp    int64    `json:"timestamp,omitempty"`
}

var narrationPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\ball\s+(\d+\s+)?tests?\s+(are\s+|is\s+)?(pass(es|ed|ing)?|succeed(ed|s)?|green)\b`),
	regexp.MustCompile(`(?i)\btests?\s+(are\s+|is\s+)?(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\btest\s+suite\s+(is\s+)?(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\b(all\s+)?(tests?|test\s+suite|all\s+checks|unit\s+tests?)\s+(is|are)\s+green\b`),
	regexp.MustCompile(`(?i)\bverified\s+green\b`),
	regexp.MustCompile(`(?i)\bunit\s+tests?\s+(are\s+|is\s+)?(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\ball\s+checks\s+(are\s+|is\s+)?(pass(es|ed|ing)?|succeed(ed|s)?|green)\b`),
}

// IsNarrationTestClaim inspects model claim text for phrases asserting that
// tests passed or verification succeeded.
func IsNarrationTestClaim(claimText string) bool {
	claim := strings.TrimSpace(claimText)
	if claim == "" {
		return false
	}
	for _, pat := range narrationPatterns {
		if pat.MatchString(claim) {
			return true
		}
	}
	return false
}

// HasMatchingExecutionReceipt checks if receipts contains at least one green
// execution receipt (e.g. ExitCode == 0, tool indicates test execution such as
// go test, test, bash, cmd, exec, or test runner).
func HasMatchingExecutionReceipt(claimText string, receipts []ReceiptRecord) bool {
	if len(receipts) == 0 {
		return false
	}
	for _, r := range receipts {
		if r.ExitCode != 0 {
			continue
		}
		if r.Verdict != "" && (strings.EqualFold(r.Verdict, "DENY") || strings.EqualFold(r.Verdict, "REFUSED") || strings.EqualFold(r.Verdict, "BLOCK")) {
			continue
		}
		if isTestExecutionReceipt(r) {
			return true
		}
	}
	return false
}

func isGenericShellTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	if idx := strings.LastIndex(t, ":"); idx != -1 {
		t = t[idx+1:]
	}
	if idx := strings.LastIndex(t, "/"); idx != -1 {
		t = t[idx+1:]
	}
	if idx := strings.LastIndex(t, "\\"); idx != -1 {
		t = t[idx+1:]
	}
	t = strings.TrimSuffix(t, ".exe")
	switch t {
	case "bash", "sh", "zsh", "cmd", "exec", "powershell", "pwsh", "terminal", "shell", "command", "subprocess", "wsl":
		return true
	}
	for _, sh := range []string{"bash", "sh", "zsh", "cmd", "powershell", "pwsh"} {
		if strings.HasPrefix(t, sh+" ") || strings.HasPrefix(t, sh+"-") || strings.HasPrefix(t, sh+"/") {
			return true
		}
	}
	return false
}

func isNonTestTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	if idx := strings.LastIndex(t, ":"); idx != -1 {
		t = t[idx+1:]
	}
	switch t {
	case "git status", "git diff", "git log", "git commit", "git push", "git branch", "git checkout", "git add", "git",
		"ls", "dir", "cd", "pwd", "cat", "echo", "head", "tail", "read", "write", "edit", "glob", "grep":
		return true
	}
	return false
}

func isNonTestCommand(fullCmd string) bool {
	trimmed := strings.TrimSpace(fullCmd)
	for _, pfx := range []string{"bash -c ", "sh -c ", "cmd /c ", "powershell -command ", "pwsh -command "} {
		if strings.HasPrefix(trimmed, pfx) {
			trimmed = strings.TrimSpace(strings.Trim(trimmed[len(pfx):], "\"'"))
		}
	}
	parts := strings.Fields(trimmed)
	if len(parts) == 0 {
		return true
	}
	base := parts[0]
	if idx := strings.LastIndex(base, "/"); idx != -1 {
		base = base[idx+1:]
	}
	if idx := strings.LastIndex(base, "\\"); idx != -1 {
		base = base[idx+1:]
	}
	base = strings.TrimSuffix(base, ".exe")

	if base == "git" {
		return true
	}
	switch base {
	case "ls", "dir", "cd", "pwd", "cat", "echo", "head", "tail", "type",
		"read", "write", "edit", "glob", "grep", "find", "cp", "mv", "rm",
		"mkdir", "touch", "diff", "which", "where", "stat":
		return true
	}
	return false
}

func commandIndicatesTest(cmd []string, fullCmd string) bool {
	trimmed := strings.TrimSpace(fullCmd)
	for _, pfx := range []string{"bash -c ", "sh -c ", "cmd /c ", "powershell -command ", "pwsh -command "} {
		if strings.HasPrefix(trimmed, pfx) {
			trimmed = strings.TrimSpace(strings.Trim(trimmed[len(pfx):], "\"'"))
		}
	}
	if trimmed == "test" || strings.HasPrefix(trimmed, "test ") {
		return true
	}
	testPatterns := []string{
		"go test",
		"pytest",
		"cargo test",
		"npm test",
		"pnpm test",
		"yarn test",
		"make test",
		"make test-fast",
		"fak test",
		"test.ps1",
		"test.sh",
		"python -m unittest",
		"python -m pytest",
		"unittest",
		"vitest",
		"jest",
		"ctest",
		"rspec",
	}
	for _, p := range testPatterns {
		if strings.Contains(trimmed, p) {
			return true
		}
	}
	for _, arg := range cmd {
		argLower := strings.ToLower(strings.TrimSpace(arg))
		for _, p := range testPatterns {
			if strings.Contains(argLower, p) {
				return true
			}
		}
	}
	return false
}

func isTestExecutionReceipt(r ReceiptRecord) bool {
	tool := strings.ToLower(strings.TrimSpace(r.Tool))
	if isNonTestTool(tool) {
		return false
	}

	isGenericShell := isGenericShellTool(tool)

	if !isGenericShell {
		if tool == "go test" || tool == "test" || tool == "test_runner" || tool == "test-runner" || tool == "test runner" || tool == "pytest" || tool == "jest" || tool == "cargo test" {
			return true
		}
		if strings.Contains(tool, "test") || strings.Contains(tool, "runner") {
			return true
		}
	}

	if len(r.Command) == 0 {
		return false
	}

	fullCmd := strings.ToLower(strings.TrimSpace(strings.Join(r.Command, " ")))
	if isNonTestCommand(fullCmd) {
		return false
	}

	return commandIndicatesTest(r.Command, fullCmd)
}

// AdjudicateTurnWithReceipts evaluates a turn's done claim against both trajectory
// receipts and external witness verification. If the turn asserts test completion
// without matching execution receipts, it is refused with ReasonUnwitnessedNarrationClaim.
func AdjudicateTurnWithReceipts(ctx context.Context, turn Turn, receipts []ReceiptRecord, witness WitnessFunc) Decision {
	if len(turn.Receipts) == 0 && len(receipts) > 0 {
		turn.Receipts = receipts
	} else if len(receipts) == 0 && len(turn.Receipts) > 0 {
		receipts = turn.Receipts
	}
	if turn.ClaimedDone && IsNarrationTestClaim(turn.Claim) {
		if len(turn.Receipts) == 0 || !HasMatchingExecutionReceipt(turn.Claim, turn.Receipts) {
			return Decision{
				Verdict: VerdictRefused,
				Reason:  ReasonUnwitnessedNarrationClaim,
				Summary: "turn asserted test completion without matching execution receipts in trajectory",
			}
		}
	}
	return Adjudicate(ctx, turn, witness)
}
