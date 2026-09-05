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
	regexp.MustCompile(`(?i)\ball\s+(\d+\s+)?tests?\s+(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\btests?\s+(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\btest\s+suite\s+(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\b(tests?|test\s+suite|all\s+checks)\s+(is|are)\s+green\b`),
	regexp.MustCompile(`(?i)\bverified\s+green\b`),
	regexp.MustCompile(`(?i)\bunit\s+tests?\s+(pass(es|ed|ing)?|succeed(ed|s)?)\b`),
	regexp.MustCompile(`(?i)\ball\s+checks\s+(pass(es|ed|ing)?|succeed(ed|s)?|green)\b`),
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

func isTestExecutionReceipt(r ReceiptRecord) bool {
	tool := strings.ToLower(strings.TrimSpace(r.Tool))
	isGenericShell := false
	switch tool {
	case "bash", "cmd", "exec", "sh", "powershell", "pwsh", "terminal":
		isGenericShell = true
	}
	if strings.HasSuffix(tool, ":bash") || strings.HasSuffix(tool, ":cmd") || strings.HasSuffix(tool, ":exec") {
		isGenericShell = true
	}

	if !isGenericShell {
		if tool == "go test" || tool == "test" || tool == "test_runner" || tool == "test-runner" || tool == "test runner" || tool == "pytest" || tool == "jest" || tool == "cargo test" {
			return true
		}
		if strings.Contains(tool, "test") || strings.Contains(tool, "runner") {
			return true
		}
	}

	// For shell commands or explicit commands, verify command arguments actually invoke tests
	for _, arg := range r.Command {
		argLower := strings.ToLower(arg)
		if strings.Contains(argLower, "go test") ||
			strings.Contains(argLower, "pytest") ||
			strings.Contains(argLower, "npm test") ||
			strings.Contains(argLower, "cargo test") ||
			strings.Contains(argLower, "make test") ||
			strings.Contains(argLower, "make test-fast") ||
			strings.Contains(argLower, "test.ps1") ||
			strings.Contains(argLower, "test.sh") ||
			strings.Contains(argLower, "fak test") {
			return true
		}
	}
	return false
}

// AdjudicateTurnWithReceipts evaluates a turn's done claim against both trajectory
// receipts and external witness verification. If the turn asserts test completion
// without matching execution receipts, it is refused with ReasonUnwitnessedNarrationClaim.
func AdjudicateTurnWithReceipts(ctx context.Context, turn Turn, receipts []ReceiptRecord, witness WitnessFunc) Decision {
	if turn.ClaimedDone && IsNarrationTestClaim(turn.Claim) {
		if !HasMatchingExecutionReceipt(turn.Claim, receipts) {
			return Decision{
				Verdict: VerdictRefused,
				Reason:  ReasonUnwitnessedNarrationClaim,
				Summary: "turn asserted test completion without matching execution receipts in trajectory",
			}
		}
	}
	if turn.Receipts == nil && receipts != nil {
		turn.Receipts = receipts
	}
	return Adjudicate(ctx, turn, witness)
}
