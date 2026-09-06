package auditreason

import (
	"encoding/json"
	"testing"
)

func TestGrepPatternNotFound(t *testing.T) {
	// Proves grep -q or grep returning 1 is qualified as benign pattern not found, not a fatal crash.
	commands := []string{
		"grep -q pattern file.txt",
		"grep pattern file.txt",
		"grep -rn 'search term' .",
		"egrep -q foo bar",
		"fgrep -q hello world.txt",
		"rg -q pattern file.txt",
		"git grep pattern",
		"/usr/bin/grep -q needle haystack",
		`C:\bin\grep.exe -q foo bar`,
	}

	for _, cmd := range commands {
		status := QualifyExitCode(cmd, 1)
		if status != StatusPatternNotFound {
			t.Errorf("QualifyExitCode(%q, 1) = %q, want %q", cmd, status, StatusPatternNotFound)
		}
		if !status.IsBenign() {
			t.Errorf("status for %q should be benign", cmd)
		}
		if status.IsFatal() {
			t.Errorf("status for %q should not be fatal", cmd)
		}

		qual := QualifyExit(cmd, 1)
		if !qual.Benign || qual.Fatal {
			t.Errorf("QualifyExit(%q, 1) = %+v, want benign and not fatal", cmd, qual)
		}
		if qual.Status != StatusPatternNotFound {
			t.Errorf("QualifyExit(%q, 1).Status = %q, want %q", cmd, qual.Status, StatusPatternNotFound)
		}
	}
}

func TestGitDiffQuiet(t *testing.T) {
	// Proves git diff --quiet returning 1 is qualified as benign diff present, not a fatal crash.
	commands := []string{
		"git diff --quiet",
		"git diff --quiet HEAD~1",
		"git --no-pager diff --quiet",
		"git -C /path/to/repo diff --quiet",
		"git diff-files --quiet",
		"git diff-index --quiet HEAD",
		"git diff --exit-code",
		"diff -q a.txt b.txt",
		"cmp -s file1 file2",
	}

	for _, cmd := range commands {
		status := QualifyExitCode(cmd, 1)
		if status != StatusDiffPresent {
			t.Errorf("QualifyExitCode(%q, 1) = %q, want %q", cmd, status, StatusDiffPresent)
		}
		if !status.IsBenign() {
			t.Errorf("status for %q should be benign", cmd)
		}
		if status.IsFatal() {
			t.Errorf("status for %q should not be fatal", cmd)
		}

		qual := QualifyExit(cmd, 1)
		if !qual.Benign || qual.Fatal {
			t.Errorf("QualifyExit(%q, 1) = %+v, want benign and not fatal", cmd, qual)
		}
	}
}

func TestGeneralToolsExitCode1AndCrash(t *testing.T) {
	// Proves general exit code 1 or exit 127/143 on other tools is classified as an actual error/crash.
	errorCommands := []string{
		"cat nonexistent.txt",
		"go build ./...",
		"npm test",
		"curl https://invalid.endpoint",
		"fake_tool --run",
	}

	for _, cmd := range errorCommands {
		status := QualifyExitCode(cmd, 1)
		if status != StatusError {
			t.Errorf("QualifyExitCode(%q, 1) = %q, want %q", cmd, status, StatusError)
		}
		if status.IsBenign() {
			t.Errorf("general error %q should not be benign", cmd)
		}
		if !status.IsFatal() {
			t.Errorf("general error %q should be fatal", cmd)
		}
	}

	// Exit 127: Command not found
	cmdNotFound := QualifyExitCode("nonexistent_binary", 127)
	if cmdNotFound != StatusCommandNotFound {
		t.Errorf("QualifyExitCode(..., 127) = %q, want %q", cmdNotFound, StatusCommandNotFound)
	}
	if !cmdNotFound.IsFatal() || cmdNotFound.IsBenign() {
		t.Errorf("exit 127 must be fatal and not benign")
	}

	// Exit 143: Terminated / SIGTERM
	crash143 := QualifyExitCode("go test ./...", 143)
	if crash143 != StatusProcessCrash {
		t.Errorf("QualifyExitCode(..., 143) = %q, want %q", crash143, StatusProcessCrash)
	}
	if !crash143.IsFatal() || crash143.IsBenign() {
		t.Errorf("exit 143 must be fatal and not benign")
	}

	// Exit 137: SIGKILL
	crash137 := QualifyExitCode("process", 137)
	if crash137 != StatusProcessCrash {
		t.Errorf("QualifyExitCode(..., 137) = %q, want %q", crash137, StatusProcessCrash)
	}

	// Exit 2 on grep (syntax error)
	syntaxErr := QualifyExitCode("grep -[ invalid", 2)
	if syntaxErr != StatusSyntaxError {
		t.Errorf("QualifyExitCode(grep, 2) = %q, want %q", syntaxErr, StatusSyntaxError)
	}
	if !syntaxErr.IsFatal() {
		t.Errorf("syntax error must be fatal")
	}
}

func TestPredicateCommandsExitZero(t *testing.T) {
	// Proves predicate commands returning exit 0 are successful matches.
	commands := []string{
		"grep -q pattern file.txt",
		"grep pattern file.txt",
		"git diff --quiet",
		"test -e file.go",
		"[ -f file.go ]",
		"diff -q a.txt b.txt",
		"which go",
	}

	for _, cmd := range commands {
		status := QualifyExitCode(cmd, 0)
		if status != StatusSuccess {
			t.Errorf("QualifyExitCode(%q, 0) = %q, want %q", cmd, status, StatusSuccess)
		}
		if !status.IsBenign() {
			t.Errorf("exit 0 for %q must be benign", cmd)
		}
		if status.IsFatal() {
			t.Errorf("exit 0 for %q must not be fatal", cmd)
		}
	}
}

func TestFileExistenceAndPredicates(t *testing.T) {
	cases := []struct {
		cmd  string
		want SemanticExitStatus
	}{
		{"test -e missing.txt", StatusFileNotFound},
		{"test -f missing.txt", StatusFileNotFound},
		{"test -d missing_dir", StatusFileNotFound},
		{"[ -e missing.txt ]", StatusFileNotFound},
		{"[[ -f missing.txt ]]", StatusFileNotFound},
		{"test 1 -eq 2", StatusPredicateFalse},
		{"false", StatusPredicateFalse},
		{"git merge-base --is-ancestor revA revB", StatusPredicateFalse},
		{"which nonexistent_tool", StatusMatchNotFound},
		{"command -v missing_cmd", StatusMatchNotFound},
	}

	for _, tc := range cases {
		status := QualifyExitCode(tc.cmd, 1)
		if status != tc.want {
			t.Errorf("QualifyExitCode(%q, 1) = %q, want %q", tc.cmd, status, tc.want)
		}
		if !status.IsBenign() {
			t.Errorf("QualifyExitCode(%q, 1) should be benign", tc.cmd)
		}
	}
}

func TestShellUnwrapping(t *testing.T) {
	cases := []struct {
		cmd  string
		want SemanticExitStatus
	}{
		{`bash -c "grep -q foo bar"`, StatusPatternNotFound},
		{`sh -c 'git diff --quiet'`, StatusDiffPresent},
		{`powershell -NoProfile -Command 'grep -q foo bar'`, StatusPatternNotFound},
		{`VAR=1 FOO=2 grep -q foo bar`, StatusPatternNotFound},
	}

	for _, tc := range cases {
		status := QualifyExitCode(tc.cmd, 1)
		if status != tc.want {
			t.Errorf("QualifyExitCode(%q, 1) = %q, want %q", tc.cmd, status, tc.want)
		}
	}
}

func TestFailureCounterAndEscalation(t *testing.T) {
	fc := &FailureCounter{}

	// Benign queries: must NOT increment failure counter
	if fc.Record("grep -q pattern file.txt", 1) {
		t.Error("grep -q exit 1 should not record as failure escalation")
	}
	if fc.Record("git diff --quiet", 1) {
		t.Error("git diff --quiet exit 1 should not record as failure escalation")
	}
	if fc.Record("test -e missing.txt", 1) {
		t.Error("test -e exit 1 should not record as failure escalation")
	}
	if fc.Record("grep -q pattern file.txt", 0) {
		t.Error("exit 0 should not record as failure escalation")
	}
	if fc.Count() != 0 {
		t.Errorf("FailureCounter.Count() = %d, want 0", fc.Count())
	}

	// Genuine errors: must increment failure counter
	if !fc.Record("cat missing.txt", 1) {
		t.Error("cat missing.txt exit 1 should record as failure escalation")
	}
	if !fc.Record("go build ./...", 1) {
		t.Error("go build exit 1 should record as failure escalation")
	}
	if !fc.Record("crashed_tool", 143) {
		t.Error("exit 143 should record as failure escalation")
	}
	if fc.Count() != 3 {
		t.Errorf("FailureCounter.Count() = %d, want 3", fc.Count())
	}

	fc.Reset()
	if fc.Count() != 0 {
		t.Errorf("FailureCounter.Count() after Reset() = %d, want 0", fc.Count())
	}
}

func TestToolFailurePayloadForCommandBenignExit(t *testing.T) {
	// A message with error-like keywords (e.g. "command not found" or "exit code 127")
	// should NOT trigger a tool failure when the command itself was a benign query returning exit 1.
	msg := "exit code 127: command not found"
	payload, ok := ToolFailurePayloadForCommand(msg, "grep -q pattern file.txt", 1)
	if ok {
		t.Fatalf("benign grep -q exit 1 should not resolve to tool failure, got %+v", payload)
	}

	payloadDiff, okDiff := ToolFailurePayloadForCommand(msg, "git diff --quiet", 1)
	if okDiff {
		t.Fatalf("benign git diff --quiet exit 1 should not resolve to tool failure, got %+v", payloadDiff)
	}

	spec, okSpec := ToolFailureForCommand(msg, "grep -q pattern file.txt", 1)
	if okSpec {
		t.Fatalf("ToolFailureForCommand should return false for benign grep, got %+v", spec)
	}

	// Non-benign command with exit 127 DOES resolve to ToolFailure
	payloadFail, okFail := ToolFailurePayloadForCommand(msg, "some_missing_tool", 127)
	if !okFail {
		t.Fatal("missing tool with exit 127 should resolve to tool failure")
	}
	if payloadFail.Code != ToolFailureShellMismatch {
		t.Fatalf("payload.Code = %q, want %q", payloadFail.Code, ToolFailureShellMismatch)
	}
}

func TestJSONHostBoundarySerialization(t *testing.T) {
	boundary := QualifyHostBoundary("grep -q foo bar.txt", 1, "/work/repo", "", "")
	if !boundary.Benign || boundary.Fatal {
		t.Errorf("boundary should be benign and not fatal: %+v", boundary)
	}
	if boundary.Status != StatusPatternNotFound {
		t.Errorf("boundary.Status = %q, want %q", boundary.Status, StatusPatternNotFound)
	}

	data, err := boundary.JSON()
	if err != nil {
		t.Fatalf("boundary.JSON() failed: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if parsed["status"] != "STATUS_PATTERN_NOT_FOUND" {
		t.Errorf("JSON status = %v, want STATUS_PATTERN_NOT_FOUND", parsed["status"])
	}
	if parsed["benign"] != true {
		t.Errorf("JSON benign = %v, want true", parsed["benign"])
	}
	if parsed["fatal"] != false {
		t.Errorf("JSON fatal = %v, want false", parsed["fatal"])
	}
	if parsed["cwd"] != "/work/repo" {
		t.Errorf("JSON cwd = %v, want /work/repo", parsed["cwd"])
	}
}
