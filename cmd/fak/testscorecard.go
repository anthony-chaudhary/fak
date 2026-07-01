package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

var runScorecardControlPaneForTest = runScorecardControlPane

func runTestScorecard(stdout, stderr io.Writer, args []string, asJSON, dry bool) int {
	p, controlArgs, err := planTestScorecard(args, asJSON)
	if err != nil {
		if asJSON {
			return writeTestRepairJSON(stdout, stderr, newTestUsageRepairPacket(err))
		}
		fmt.Fprintf(stderr, "fak test scorecard: %v\n", err)
		return 2
	}
	if asJSON {
		if dry {
			return writeTestRepairJSON(stdout, stderr, newTestResolvedRepairPacket(p))
		}
		var childOut, childErr bytes.Buffer
		rc := runScorecardControlPaneForTest(&childOut, &childErr, controlArgs)
		rawOut, rawErr := childOut.String(), childErr.String()
		var payload scorecardpane.Payload
		var parseErr error
		if strings.TrimSpace(rawOut) == "" {
			parseErr = errors.New("empty scorecard control-pane JSON output")
		} else {
			parseErr = json.Unmarshal(childOut.Bytes(), &payload)
		}
		packet := newScorecardRepairPacket(p.Argv, rc, rawOut, rawErr, payload, parseErr)
		return writeTestRepairJSON(stdout, stderr, packet)
	}

	fmt.Fprintf(stdout, "# %s\n# tier=%s -> %s\n", p.Note, p.Tier, strings.Join(p.Argv, " "))
	if dry {
		return 0
	}
	return runScorecardControlPaneForTest(stdout, stderr, controlArgs)
}

func planTestScorecard(args []string, asJSON bool) (testPlan, []string, error) {
	if hasScorecardFlag(args, "pin") {
		return testPlan{}, nil, errors.New("fak test scorecard is read-only; use `fak scorecard control-pane --pin` to update the baseline")
	}
	if hasFalseScorecardBoolFlag(args, "check") {
		return testPlan{}, nil, errors.New("fak test scorecard always runs the ratchet check; use `fak scorecard control-pane --check=false` for a non-gating snapshot")
	}
	if asJSON && hasFalseScorecardBoolFlag(args, "json") {
		return testPlan{}, nil, errors.New("fak test --json scorecard requires scorecard JSON output; drop --json=false")
	}
	controlArgs := make([]string, 0, len(args)+2)
	if !hasScorecardFlag(args, "check") {
		controlArgs = append(controlArgs, "--check")
	}
	if asJSON && !hasScorecardFlag(args, "json") {
		controlArgs = append(controlArgs, "--json")
	}
	controlArgs = append(controlArgs, args...)
	command := append([]string{"fak", "scorecard", "control-pane"}, controlArgs...)
	return testPlan{
		Tier: "scorecard",
		Argv: command,
		Note: "running scorecard ratchet gate",
	}, controlArgs, nil
}

func hasScorecardFlag(args []string, name string) bool {
	long := "--" + name
	for _, arg := range args {
		if arg == "--" {
			return false
		}
		if arg == long || strings.HasPrefix(arg, long+"=") {
			return true
		}
	}
	return false
}

func hasFalseScorecardBoolFlag(args []string, name string) bool {
	long := "--" + name + "="
	for _, arg := range args {
		if arg == "--" {
			return false
		}
		if strings.HasPrefix(arg, long) {
			v := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(arg, long)))
			return v == "false" || v == "0"
		}
	}
	return false
}

func newScorecardRepairPacket(command []string, rc int, rawOut, rawErr string, payload scorecardpane.Payload, parseErr error) testRepairPacket {
	stdoutTail, stdoutTruncated := tailRunes(rawOut, testOutputTailRunes)
	stderrTail, stderrTruncated := tailRunes(rawErr, testOutputTailRunes)
	packet := testRepairPacket{
		Schema:          testRepairPacketSchema,
		Tier:            "scorecard",
		Command:         append([]string(nil), command...),
		ExitCode:        rc,
		StdoutTail:      stdoutTail,
		StderrTail:      stderrTail,
		OutputTruncated: stdoutTruncated || stderrTruncated,
		Diagnostics:     scorecardDiagnostics(payload),
	}
	if parseErr != nil {
		packet.OK = false
		packet.Verdict = "ERROR"
		packet.Finding = "scorecard ratchet returned non-JSON output"
		packet.Reason = parseErr.Error()
		packet.NextAction = "inspect stdout_tail/stderr_tail, repair `fak scorecard control-pane --check --json`, then rerun `fak test --json scorecard`"
		if packet.ExitCode == 0 {
			packet.ExitCode = 1
		}
		packet.Findings = []testRepairFinding{{
			Class:      "scorecard_output_error",
			Severity:   "error",
			Finding:    packet.Finding,
			Reason:     packet.Reason,
			NextAction: packet.NextAction,
			ExitCode:   packet.ExitCode,
		}}
		return packet
	}

	class := scorecardFindingClass(payload, rc)
	severity := "info"
	if rc != 0 {
		severity = "error"
	} else if len(payload.Trend.EarlyWarning) > 0 || len(payload.EarlyWarning) > 0 {
		severity = "warning"
	}
	packet.OK = rc == 0 && payload.OK
	if packet.OK {
		packet.Verdict = "PASS"
		packet.Finding = fmt.Sprintf("scorecard ratchet passed (%s)", nonEmpty(payload.Finding, "all_clear"))
		packet.NextAction = "continue with the next required gate for this change; review gate_message if it contains an advisory"
		packet.Findings = []testRepairFinding{{
			Class:      class,
			Severity:   severity,
			Finding:    packet.Finding,
			Reason:     payload.GateMessage,
			NextAction: packet.NextAction,
		}}
		return packet
	}

	packet.Verdict = "FAIL"
	packet.Finding = fmt.Sprintf("scorecard ratchet failed (%s)", nonEmpty(payload.Finding, "unknown"))
	packet.Reason = scorecardReason(payload, rc)
	packet.NextAction = nonEmpty(payload.NextAction, "repair the scorecard ratchet finding, then rerun `fak test --json scorecard`")
	packet.Findings = []testRepairFinding{{
		Class:      class,
		Severity:   severity,
		Finding:    packet.Finding,
		Reason:     packet.Reason,
		NextAction: packet.NextAction,
		ExitCode:   rc,
	}}
	return packet
}

func scorecardFindingClass(payload scorecardpane.Payload, rc int) string {
	lowerGate := strings.ToLower(payload.GateMessage)
	switch {
	case rc == 0 && payload.Finding == "all_clear":
		return "scorecard_clean"
	case rc == 0:
		return "scorecard_ratchet_held"
	case payload.Errored > 0 || payload.Finding == "scorecard_unmeasured":
		return "scorecard_unmeasured"
	case payload.Trend.Direction == "unpinned" || strings.Contains(lowerGate, "unpinned"):
		return "scorecard_unpinned"
	case payload.Trend.Direction == "regressed" || strings.Contains(lowerGate, "ratchet fail") || strings.Contains(lowerGate, "grade-ratchet fail"):
		return "scorecard_regression"
	case payload.Finding != "":
		return payload.Finding
	default:
		return "scorecard_failure"
	}
}

func scorecardReason(payload scorecardpane.Payload, rc int) string {
	switch {
	case strings.TrimSpace(payload.GateMessage) != "":
		return payload.GateMessage
	case strings.TrimSpace(payload.Reason) != "":
		return payload.Reason
	default:
		return fmt.Sprintf("scorecard ratchet exited %d", rc)
	}
}

func scorecardDiagnostics(payload scorecardpane.Payload) []testDiagnostic {
	var out []testDiagnostic
	worsened := make(map[string]bool, len(payload.Trend.Worsened))
	for _, key := range payload.Trend.Worsened {
		worsened[key] = true
	}
	for _, m := range payload.Metrics {
		label := nonEmpty(m.Label, m.Key)
		if m.Debt == nil || strings.TrimSpace(m.Error) != "" {
			detail := nonEmpty(m.Error, "scorecard did not report a debt integer")
			out = append(out, testDiagnostic{
				Tool:     "scorecard",
				Code:     "SCORECARD_UNMEASURED",
				Severity: "error",
				Detail:   fmt.Sprintf("%s: %s", label, detail),
			})
			continue
		}
		if worsened[m.Key] || worsened[m.Label] {
			out = append(out, testDiagnostic{
				Tool:     "scorecard",
				Code:     "SCORECARD_REGRESSED",
				Severity: "error",
				Detail:   fmt.Sprintf("%s regressed to debt %d", label, *m.Debt),
			})
		}
	}
	for _, warning := range payload.Trend.EarlyWarning {
		out = append(out, testDiagnostic{
			Tool:     "scorecard",
			Code:     "SCORECARD_EARLY_WARNING",
			Severity: "warning",
			Detail:   fmt.Sprintf("%s rose %+d vs baseline (%d -> %d)", warning.Label, warning.Delta, warning.From, warning.To),
		})
	}
	return out
}

func nonEmpty(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}
