package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
)

var ruffLookPath = exec.LookPath

type ruffJSONDiagnostic struct {
	Code     string `json:"code"`
	Filename string `json:"filename"`
	Message  string `json:"message"`
	Location struct {
		Row    int `json:"row"`
		Column int `json:"column"`
	} `json:"location"`
}

func runTestRuff(stdout, stderr io.Writer, stdin io.Reader, args []string, asJSON, dry bool) int {
	targets := defaultArgs(args, ".")
	humanCommand := append([]string{"ruff", "check"}, targets...)
	jsonCommand := append([]string{"ruff", "check", "--output-format", "json"}, targets...)
	if !asJSON {
		if _, err := ruffLookPath("ruff"); err != nil {
			fmt.Fprintln(stderr, "fak test ruff: SKIP (ruff not on PATH)")
			return 0
		}
		return runTestCommand(humanCommand, stdin, stdout, stderr).ExitCode
	}
	if dry {
		return writeTestRepairJSON(stdout, stderr, newTestResolvedRepairPacket(testPlan{
			Tier: "ruff",
			Argv: jsonCommand,
			Note: "running ruff check",
		}))
	}
	if _, err := ruffLookPath("ruff"); err != nil {
		return writeTestRepairJSON(stdout, stderr, newRuffUnavailablePacket(jsonCommand, err))
	}

	var childOut, childErr bytes.Buffer
	result := runTestCommand(jsonCommand, stdin, &childOut, &childErr)
	rawOut, rawErr := childOut.String(), childErr.String()
	var diagnostics []ruffJSONDiagnostic
	var parseErr error
	if strings.TrimSpace(rawOut) != "" {
		parseErr = json.Unmarshal(childOut.Bytes(), &diagnostics)
	}
	packet := newRuffRepairPacket(jsonCommand, result.ExitCode, rawOut, rawErr, diagnostics, parseErr)
	return writeTestRepairJSON(stdout, stderr, packet)
}

func newRuffUnavailablePacket(command []string, err error) testRepairPacket {
	reason := "ruff executable not found on PATH"
	if err != nil && !errors.Is(err, exec.ErrNotFound) {
		reason = err.Error()
	}
	return testRepairPacket{
		Schema:     testRepairPacketSchema,
		OK:         true,
		Verdict:    "SKIP",
		Finding:    "ruff lint is unmeasured on this host",
		Reason:     reason,
		NextAction: "install ruff or run on a host that declares ruff support; until then Python lint is explicitly skipped, not clean",
		Tier:       "ruff",
		Command:    append([]string(nil), command...),
		ExitCode:   0,
		Findings: []testRepairFinding{{
			Class:      "ruff_unavailable",
			Severity:   "warning",
			Finding:    "ruff not available",
			Reason:     reason,
			NextAction: "install ruff or run this gate on a ruff-capable host",
		}},
	}
}

func newRuffRepairPacket(command []string, rc int, rawOut, rawErr string, diagnostics []ruffJSONDiagnostic, parseErr error) testRepairPacket {
	stdoutTail, stdoutTruncated := tailRunes(rawOut, testOutputTailRunes)
	stderrTail, stderrTruncated := tailRunes(rawErr, testOutputTailRunes)
	packet := testRepairPacket{
		Schema:          testRepairPacketSchema,
		Tier:            "ruff",
		Command:         append([]string(nil), command...),
		ExitCode:        rc,
		StdoutTail:      stdoutTail,
		StderrTail:      stderrTail,
		OutputTruncated: stdoutTruncated || stderrTruncated,
		Diagnostics:     ruffDiagnostics(diagnostics),
	}
	if parseErr != nil {
		packet.OK = false
		packet.Verdict = "ERROR"
		packet.Finding = "ruff returned non-JSON output"
		packet.Reason = parseErr.Error()
		packet.NextAction = "inspect stdout_tail/stderr_tail, fix the ruff JSON invocation, then rerun `fak test --json ruff ...`"
		packet.ExitCode = 1
		packet.Findings = []testRepairFinding{{
			Class:      "ruff_output_error",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   packet.ExitCode,
		}}
		return packet
	}
	if rc == 0 {
		packet.OK = true
		packet.Verdict = "PASS"
		packet.Finding = fmt.Sprintf("ruff passed with %d diagnostic(s)", len(diagnostics))
		packet.NextAction = "continue with the next required gate for this change"
		packet.Findings = []testRepairFinding{{
			Class:      "ruff_clean",
			Severity:   "info",
			Finding:    packet.Finding,
			NextAction: packet.NextAction,
		}}
		return packet
	}
	if rc == 1 {
		packet.OK = false
		packet.Verdict = "FAIL"
		packet.Finding = fmt.Sprintf("ruff found %d diagnostic(s)", len(diagnostics))
		packet.Reason = "ruff exited 1"
		packet.NextAction = "fix the first ruff diagnostic, then rerun `fak test --json ruff ...`"
		packet.Findings = []testRepairFinding{{
			Class:      "ruff_failure",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   rc,
		}}
		return packet
	}
	packet.OK = false
	packet.Verdict = "ERROR"
	packet.Finding = fmt.Sprintf("ruff exited %d", rc)
	packet.Reason = fmt.Sprintf("ruff exited %d", rc)
	packet.NextAction = "inspect stdout_tail/stderr_tail, repair the ruff invocation, then rerun"
	packet.Findings = []testRepairFinding{{
		Class:      "ruff_error",
		Severity:   "error",
		Finding:    packet.Finding,
		Reason:     packet.Reason,
		NextAction: packet.NextAction,
		ExitCode:   rc,
	}}
	return packet
}

func ruffDiagnostics(diagnostics []ruffJSONDiagnostic) []testDiagnostic {
	if len(diagnostics) == 0 {
		return nil
	}
	out := make([]testDiagnostic, 0, len(diagnostics))
	for _, d := range diagnostics {
		out = append(out, testDiagnostic{
			Tool:     "ruff",
			Code:     d.Code,
			File:     d.Filename,
			Line:     d.Location.Row,
			Col:      d.Location.Column,
			Severity: "error",
			Detail:   d.Message,
		})
	}
	return out
}
