package main

import (
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"strings"
)

const (
	testRepairPacketSchema = "fak.test_repair_packet.v1"
	testOutputTailRunes    = 4000
)

type testRepairPacket struct {
	Schema          string              `json:"schema"`
	OK              bool                `json:"ok"`
	Verdict         string              `json:"verdict"`
	Finding         string              `json:"finding"`
	Reason          string              `json:"reason,omitempty"`
	NextAction      string              `json:"next_action"`
	Tier            string              `json:"tier,omitempty"`
	Command         []string            `json:"command,omitempty"`
	GoArgs          []string            `json:"go_args,omitempty"`
	ViaWSL          bool                `json:"via_wsl,omitempty"`
	DryRun          bool                `json:"dry_run,omitempty"`
	ExitCode        int                 `json:"exit_code"`
	StdoutTail      string              `json:"stdout_tail,omitempty"`
	StderrTail      string              `json:"stderr_tail,omitempty"`
	OutputTruncated bool                `json:"output_truncated,omitempty"`
	Diagnostics     []testDiagnostic    `json:"diagnostics,omitempty"`
	Findings        []testRepairFinding `json:"findings"`
}

type testDiagnostic struct {
	Tool     string `json:"tool"`
	Code     string `json:"code"`
	File     string `json:"file,omitempty"`
	Line     int    `json:"line,omitempty"`
	Col      int    `json:"col,omitempty"`
	Severity string `json:"severity"`
	Detail   string `json:"detail"`
}

type testRepairFinding struct {
	Class      string `json:"class"`
	Severity   string `json:"severity"`
	Finding    string `json:"finding"`
	Reason     string `json:"reason,omitempty"`
	NextAction string `json:"next_action"`
	ExitCode   int    `json:"exit_code,omitempty"`
}

type testCommandResult struct {
	ExitCode   int
	SpawnError string
}

var runTestCommand = execTestCommand

func execTestCommand(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
	if len(argv) == 0 {
		msg := "empty test command"
		fmt.Fprintf(stderr, "fak test: %s\n", msg)
		return testCommandResult{ExitCode: 1, SpawnError: msg}
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = stdout, stderr, stdin
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return testCommandResult{ExitCode: ee.ExitCode()}
		}
		fmt.Fprintf(stderr, "fak test: %v\n", err)
		return testCommandResult{ExitCode: 1, SpawnError: err.Error()}
	}
	return testCommandResult{ExitCode: 0}
}

func runTestRepairJSON(stdout, stderr io.Writer, stdin io.Reader, p testPlan) int {
	var childOut, childErr bytes.Buffer
	result := runTestCommand(p.Argv, stdin, &childOut, &childErr)
	packet := newTestFinishedRepairPacket(p, result, childOut.String(), childErr.String())
	return writeTestRepairJSON(stdout, stderr, packet)
}

func runTestPlan(stdout, stderr io.Writer, stdin io.Reader, p testPlan) int {
	if !p.FailOnStdout {
		return runTestCommand(p.Argv, stdin, stdout, stderr).ExitCode
	}
	var childOut, childErr bytes.Buffer
	result := runTestCommand(p.Argv, stdin, &childOut, &childErr)
	_, _ = io.WriteString(stdout, childOut.String())
	_, _ = io.WriteString(stderr, childErr.String())
	if result.ExitCode == 0 && strings.TrimSpace(childOut.String()) != "" {
		return 1
	}
	return result.ExitCode
}

func writeTestRepairJSON(stdout, stderr io.Writer, packet testRepairPacket) int {
	if err := writeIndentedJSONNoEscape(stdout, packet); err != nil {
		fmt.Fprintf(stderr, "fak test: encode json: %v\n", err)
		return 1
	}
	return packet.ExitCode
}

func newTestUsageRepairPacket(err error) testRepairPacket {
	finding := fmt.Sprintf("fak test resolver rejected the request: %v", err)
	return testRepairPacket{
		Schema:     testRepairPacketSchema,
		OK:         false,
		Verdict:    "USAGE",
		Finding:    finding,
		Reason:     err.Error(),
		NextAction: "choose one of fast, full, race, affected, durations, shards, or a ./package/... target; rerun `fak test --list` for the menu",
		ExitCode:   2,
		Findings: []testRepairFinding{{
			Class:      "usage",
			Severity:   "error",
			Finding:    finding,
			Reason:     err.Error(),
			NextAction: "correct the tier or package argument, then rerun fak test",
			ExitCode:   2,
		}},
	}
}

func newTestResolvedRepairPacket(p testPlan) testRepairPacket {
	finding := fmt.Sprintf("resolved fak test %s command", p.Tier)
	return testRepairPacket{
		Schema:     testRepairPacketSchema,
		OK:         true,
		Verdict:    "READY",
		Finding:    finding,
		NextAction: "run the same command without -n/--print to execute this tier",
		Tier:       p.Tier,
		Command:    append([]string(nil), p.Argv...),
		GoArgs:     append([]string(nil), p.GoArgs...),
		ViaWSL:     p.ViaWSL,
		DryRun:     true,
		ExitCode:   0,
		Findings: []testRepairFinding{{
			Class:      "resolved_command",
			Severity:   "info",
			Finding:    finding,
			NextAction: "execute this exact command when ready",
		}},
	}
}

func newTestFinishedRepairPacket(p testPlan, result testCommandResult, stdout, stderr string) testRepairPacket {
	unexpectedStdout := p.FailOnStdout && result.ExitCode == 0 && strings.TrimSpace(stdout) != ""
	if unexpectedStdout {
		result.ExitCode = 1
	}
	stdoutTail, stdoutTruncated := tailRunes(stdout, testOutputTailRunes)
	stderrTail, stderrTruncated := tailRunes(stderr, testOutputTailRunes)
	packet := testRepairPacket{
		Schema:          testRepairPacketSchema,
		Tier:            p.Tier,
		Command:         append([]string(nil), p.Argv...),
		GoArgs:          append([]string(nil), p.GoArgs...),
		ViaWSL:          p.ViaWSL,
		ExitCode:        result.ExitCode,
		StdoutTail:      stdoutTail,
		StderrTail:      stderrTail,
		OutputTruncated: stdoutTruncated || stderrTruncated,
	}
	if result.ExitCode == 0 && result.SpawnError == "" {
		packet.OK = true
		packet.Verdict = "PASS"
		packet.Finding = fmt.Sprintf("fak test %s passed", p.Tier)
		packet.NextAction = "continue with the next required gate for this change"
		packet.Findings = []testRepairFinding{{
			Class:      "test_passed",
			Severity:   "info",
			Finding:    packet.Finding,
			NextAction: packet.NextAction,
		}}
		return packet
	}

	if result.SpawnError != "" {
		packet.OK = false
		packet.Verdict = "ERROR"
		packet.Finding = "fak test command did not start"
		packet.Reason = result.SpawnError
		packet.NextAction = "confirm the required executable exists and the host routing is valid; on Windows this normally means running through test.ps1/WSL"
		packet.Findings = []testRepairFinding{{
			Class:      "spawn_error",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     result.SpawnError,
			NextAction: packet.NextAction,
			ExitCode:   result.ExitCode,
		}}
		return packet
	}

	class, next := classifyTestFailure(p.Tier, stdout+"\n"+stderr)
	packet.OK = false
	packet.Verdict = "FAIL"
	packet.Finding = fmt.Sprintf("fak test %s failed with exit %d", p.Tier, result.ExitCode)
	if unexpectedStdout {
		packet.Reason = "command produced output"
	} else {
		packet.Reason = fmt.Sprintf("command exited %d", result.ExitCode)
	}
	packet.NextAction = next
	packet.Findings = []testRepairFinding{{
		Class:      class,
		Severity:   "error",
		Finding:    packet.Finding,
		Reason:     packet.Reason,
		NextAction: next,
		ExitCode:   result.ExitCode,
	}}
	return packet
}

func classifyTestFailure(tier, output string) (class, nextAction string) {
	lower := strings.ToLower(output)
	switch {
	case tier == "build":
		return "go_build_failure", "fix the compile/build error named in stdout_tail or stderr_tail, then rerun `fak test build`"
	case tier == "vet":
		return "go_vet_failure", "fix the go vet diagnostic named in stdout_tail or stderr_tail, then rerun `fak test vet`"
	case tier == "gofmt":
		return "gofmt_failure", "run gofmt -w on the files listed in stdout_tail, then rerun `fak test gofmt`"
	case strings.Contains(output, "--- FAIL:") || strings.Contains(output, "\nFAIL\t"):
		return "go_test_failure", "inspect stdout_tail for the first failing test/package, fix it, then rerun the exact command"
	case strings.Contains(lower, "[build failed]") || strings.Contains(lower, "build failed") || strings.Contains(lower, "syntax error:"):
		return "go_build_failure", "fix the compile/build error named in stdout_tail or stderr_tail, then rerun the exact command"
	case strings.Contains(lower, "cannot find main module") || strings.Contains(lower, "no go files"):
		return "go_setup_failure", "run from the Go module root or pass a valid package target, then rerun fak test"
	default:
		return "test_command_failed", "read stdout_tail and stderr_tail for the failing command output, repair the named issue, then rerun the exact command"
	}
}

func tailRunes(s string, limit int) (string, bool) {
	if limit <= 0 {
		return "", s != ""
	}
	r := []rune(s)
	if len(r) <= limit {
		return s, false
	}
	return string(r[len(r)-limit:]), true
}

type testListReport struct {
	Schema string         `json:"schema"`
	Tiers  []testTierInfo `json:"tiers"`
}

type testTierInfo struct {
	Name    string `json:"name"`
	Command string `json:"command"`
	When    string `json:"when"`
}

func writeTestListJSON(stdout, stderr io.Writer) int {
	rep := testListReport{
		Schema: "fak.test_tiers.v1",
		Tiers: []testTierInfo{
			{Name: "fast", Command: "go test -short ./...", When: "default pre-commit smoke tier"},
			{Name: "build", Command: "go build ./...", When: "compile/build gate"},
			{Name: "vet", Command: "go vet ./...", When: "static Go analyzer gate"},
			{Name: "gofmt", Command: "gofmt -l .", When: "formatting gate"},
			{Name: "codelint", Command: "fak codelint ...", When: "agent-written code lint packs"},
			{Name: "full", Command: "go test ./...", When: "authoritative suite"},
			{Name: "race", Command: "go test -short -race ./...", When: "local race gate"},
			{Name: "affected", Command: "fak affected ...", When: "changed packages plus transitive importers"},
			{Name: "durations", Command: "fak test durations ...", When: "fold go test -json into a duration ledger"},
			{Name: "shards", Command: "fak test shards ...", When: "balance packages from a duration ledger"},
			{Name: "<pkg>", Command: "go test <pkg>", When: "single package or package pattern"},
		},
	}
	if err := writeIndentedJSONNoEscape(stdout, rep); err != nil {
		fmt.Fprintf(stderr, "fak test: encode json: %v\n", err)
		return 1
	}
	return 0
}

func hasFlag(args []string, name string) bool {
	long := "--" + name
	short := "-" + name
	for _, a := range args {
		if a == "--" {
			return false
		}
		if a == long || a == short {
			return true
		}
	}
	return false
}
