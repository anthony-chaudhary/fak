package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type codelintJSONFinding struct {
	Pack     string `json:"pack"`
	Code     string `json:"code"`
	File     string `json:"file"`
	Line     int    `json:"line"`
	Col      int    `json:"col"`
	Severity string `json:"severity"`
	Detail   string `json:"detail"`
}

func runTestCodelint(stdout, stderr io.Writer, args []string, asJSON, dry bool) int {
	if !asJSON {
		return runCodelint(stdout, stderr, args)
	}
	command := append([]string{"fak", "codelint"}, args...)
	if dry {
		return writeTestRepairJSON(stdout, stderr, newTestResolvedRepairPacket(testPlan{
			Tier: "codelint",
			Argv: command,
			Note: "running codelint packs",
		}))
	}

	codelintArgs := append([]string(nil), args...)
	if !hasFlag(codelintArgs, "json") {
		codelintArgs = append([]string{"--json"}, codelintArgs...)
	}
	var childOut, childErr bytes.Buffer
	rc := runCodelint(&childOut, &childErr, codelintArgs)
	rawOut, rawErr := childOut.String(), childErr.String()

	var findings []codelintJSONFinding
	var parseErr error
	if strings.TrimSpace(rawOut) != "" {
		parseErr = json.Unmarshal(childOut.Bytes(), &findings)
	}
	packet := newCodelintRepairPacket(command, rc, rawOut, rawErr, findings, parseErr)
	return writeTestRepairJSON(stdout, stderr, packet)
}

func newCodelintRepairPacket(command []string, rc int, rawOut, rawErr string, findings []codelintJSONFinding, parseErr error) testRepairPacket {
	stdoutTail, stdoutTruncated := tailRunes(rawOut, testOutputTailRunes)
	stderrTail, stderrTruncated := tailRunes(rawErr, testOutputTailRunes)
	packet := testRepairPacket{
		Schema:          testRepairPacketSchema,
		Tier:            "codelint",
		Command:         append([]string(nil), command...),
		ExitCode:        rc,
		StdoutTail:      stdoutTail,
		StderrTail:      stderrTail,
		OutputTruncated: stdoutTruncated || stderrTruncated,
		Diagnostics:     codelintDiagnostics(findings),
	}
	if parseErr != nil {
		packet.OK = false
		packet.Verdict = "ERROR"
		packet.Finding = "fak codelint returned non-JSON output"
		packet.Reason = parseErr.Error()
		packet.NextAction = "inspect stdout_tail/stderr_tail, fix the codelint JSON path, then rerun `fak test --json codelint ...`"
		packet.ExitCode = 1
		packet.Findings = []testRepairFinding{{
			Class:      "codelint_output_error",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   packet.ExitCode,
		}}
		return packet
	}

	switch rc {
	case 0:
		packet.OK = true
		packet.Verdict = "PASS"
		packet.Finding = fmt.Sprintf("fak codelint passed with %d finding(s)", len(findings))
		packet.NextAction = "continue with the next required gate for this change"
		class, severity := "codelint_clean", "info"
		if len(findings) > 0 {
			class, severity = "codelint_warning", "warning"
			packet.NextAction = "review warning diagnostics if relevant, then continue; hard codelint errors are absent"
		}
		packet.Findings = []testRepairFinding{{
			Class:      class,
			Severity:   severity,
			Finding:    packet.Finding,
			NextAction: packet.NextAction,
		}}
	case 1:
		packet.OK = false
		packet.Verdict = "FAIL"
		packet.Finding = fmt.Sprintf("fak codelint found %d diagnostic(s)", len(findings))
		packet.Reason = "codelint exited 1"
		packet.NextAction = "fix the first error diagnostic, then rerun `fak test --json codelint ...`"
		packet.Findings = []testRepairFinding{{
			Class:      "codelint_failure",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   rc,
		}}
	case 2:
		packet.OK = false
		packet.Verdict = "USAGE"
		packet.Finding = "fak codelint usage error"
		packet.Reason = strings.TrimSpace(rawErr)
		packet.NextAction = "pass one or more files/directories to lint, for example `fak test --json codelint cmd/fak/test.go`"
		packet.Findings = []testRepairFinding{{
			Class:      "codelint_usage",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   rc,
		}}
	default:
		packet.OK = false
		packet.Verdict = "ERROR"
		packet.Finding = fmt.Sprintf("fak codelint exited %d", rc)
		packet.Reason = fmt.Sprintf("codelint exited %d", rc)
		packet.NextAction = "inspect stdout_tail/stderr_tail, repair the codelint invocation, then rerun"
		packet.Findings = []testRepairFinding{{
			Class:      "codelint_error",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   rc,
		}}
	}
	return packet
}

func codelintDiagnostics(findings []codelintJSONFinding) []testDiagnostic {
	if len(findings) == 0 {
		return nil
	}
	out := make([]testDiagnostic, 0, len(findings))
	for _, f := range findings {
		out = append(out, testDiagnostic{
			Tool:     f.Pack,
			Code:     f.Code,
			File:     f.File,
			Line:     f.Line,
			Col:      f.Col,
			Severity: f.Severity,
			Detail:   f.Detail,
		})
	}
	return out
}
