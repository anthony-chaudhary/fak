package main

import (
	"strings"
	"testing"
)

// #3055: a budget restart must REATTACH the existing transcript, not boot the child cold. The
// carryover seed is captured correctly and the FAK_RESET_* env vars are set, but Claude Code reads
// none of them, so continuity has to come from the wrapped agent's own resume flag. These are the
// fail-before/pass-after boundary for guardRestartRelaunchCommand: they fail against the old restart
// path (which relaunched with the same command and only inert env vars) and pass once the resume
// flag is appended on relaunch.

func TestGuardRestartRelaunchReattachesClaudeTranscript(t *testing.T) {
	cmd := []string{"claude", "-p"}
	got := guardRestartRelaunchCommand(cmd, "claude")
	if len(got) == 0 || got[len(got)-1] != "--continue" {
		t.Fatalf("budget restart must append --continue to reattach the transcript, got %v", got)
	}
	if strings.Join(got, " ") != "claude -p --continue" {
		t.Fatalf("want [claude -p --continue], got %v", got)
	}
	// The input command must not be mutated in place.
	if strings.Join(cmd, " ") != "claude -p" {
		t.Fatalf("input command was mutated: %v", cmd)
	}
}

func TestGuardRestartRelaunchIsIdempotent(t *testing.T) {
	// A second restart in the same session must not stack --continue twice.
	once := guardRestartRelaunchCommand([]string{"claude", "-p"}, "claude")
	twice := guardRestartRelaunchCommand(once, "claude")
	n := 0
	for _, a := range twice {
		if a == "--continue" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("--continue must appear exactly once across repeated restarts, got %d in %v", n, twice)
	}
}

func TestGuardRestartRelaunchLeavesUnknownAgentCold(t *testing.T) {
	// fak cannot guess a foreign tool's resume syntax, so an unrecognized agent is relaunched
	// with its command unchanged — the headless/no-continue seed-prompt handback is the separate
	// #3056 rung, not this one.
	cmd := []string{"codex", "run"}
	got := guardRestartRelaunchCommand(cmd, "codex")
	if strings.Join(got, " ") != strings.Join(cmd, " ") {
		t.Fatalf("unrecognized agent must relaunch unchanged, got %v", got)
	}
}
