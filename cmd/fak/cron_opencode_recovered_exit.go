package main

import (
	"encoding/json"
	"strings"
)

// cronOpenCodeRecoveredExitCode is the only nonzero child exit a scheduled
// OpenCode run may be re-scored from: OpenCode v2 `run` exits 1 whenever any
// step of the session finished with an error, even when it retried past that
// error and finished the session normally.
const cronOpenCodeRecoveredExitCode = 1

// cronOpenCodeRecoveredExit reports whether a failed child exit is OpenCode v2's
// "a step errored earlier" exit after a session that demonstrably recovered.
// Observed 2026-11-03: one mid-stream provider.transport ECONNRESET from the
// router, OpenCode retried, the session finished with outcome succeeded and a
// final step_finish reason "stop", yet `opencode run` exited 1, so every such
// scheduled run was scored failed and drove routine failure streaks.
//
// Evidence-gated: exit code exactly 1, a session id was seen, the attempt
// ended on its own (not a soft/hard timeout or start error), and the LAST
// JSON event on the child's stdout is step_finish with reason "stop". Any
// other final event (an error step, tool-calls, plain text) stays failed.
func cronOpenCodeRecoveredExit(exitCode int, sessionID, stopReason, stdout string) bool {
	if exitCode != cronOpenCodeRecoveredExitCode || sessionID == "" || stopReason != "completed" {
		return false
	}
	return cronOpenCodeFinalStepStop(stdout)
}

// cronOpenCodeFinalStepStop reports whether the last non-empty stdout line is
// an OpenCode JSON step_finish event whose part.reason is "stop".
func cronOpenCodeFinalStepStop(stdout string) bool {
	s := strings.TrimRight(stdout, " \t\r\n")
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSpace(s)
	if s == "" || s[0] != '{' {
		return false
	}
	var ev struct {
		Type string `json:"type"`
		Part struct {
			Type   string `json:"type"`
			Reason string `json:"reason"`
		} `json:"part"`
	}
	if err := json.Unmarshal([]byte(s), &ev); err != nil {
		return false
	}
	return ev.Type == "step_finish" && ev.Part.Reason == "stop"
}
