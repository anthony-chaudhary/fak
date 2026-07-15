package main

import "testing"

func TestGuardCommitGateInstalledAndSurvivesToolprocMerge(t *testing.T) {
	settings := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{}}
	settings.Hooks["PreToolUse"] = guardCommitGateMatchers("fak")
	guardToolprocSetHooks(&settings, "fak", "journal.jsonl")
	matchers := settings.Hooks["PreToolUse"]
	if len(matchers) != 2 {
		t.Fatalf("PreToolUse matchers=%d, want commit gate + toolproc", len(matchers))
	}
	if matchers[0].Matcher != "Bash" || len(matchers[0].Hooks) != 1 || len(matchers[0].Hooks[0].Args) == 0 || matchers[0].Hooks[0].Args[0] != "guard-commit-gate" {
		t.Fatalf("commit gate not preserved first: %+v", matchers)
	}
	if len(matchers[1].Hooks) != 1 || len(matchers[1].Hooks[0].Args) < 3 || matchers[1].Hooks[0].Args[2] != "pre" {
		t.Fatalf("toolproc observer missing: %+v", matchers)
	}
}
