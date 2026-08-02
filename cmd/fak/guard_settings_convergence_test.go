package main

import (
	"path/filepath"
	"testing"
)

// guardConvergenceGatewayURL is a throwaway loopback URL: every installer exercised here is
// either off or only writes hook argv, so nothing ever dials it.
const guardConvergenceGatewayURL = "http://127.0.0.1:4567"

// TestGuardHookSettingsConvergeWhenPreCompactAndStopDisabled is the #5526 witness.
//
// The four guard settings installers run in a fixed order (PreCompact, Stop, toolproc,
// SessionStart) and are meant to converge on ONE --settings file: the first to run creates it
// and injects `--settings`, and every later one read-modify-writes that same file. Each
// installer therefore has to be handed the path an earlier installer CREATED — which its
// install record reports — not the path that earlier installer was OFFERED.
//
// With PreCompact and Stop both OFF, toolproc is the first to create the file. Handing
// SessionStart toolproc's INPUT path (empty, since Stop and PreCompact created nothing) makes
// SessionStart write a SECOND settings file and append a second `--settings`. Claude resolves
// `--settings` last-wins, so the #5510 fold in appendClaudeSettingsArg strips the earlier
// occurrence and refuses its `hooks` key with SETTINGS_HOOKS_DROPPED: the argv keeps exactly
// one `--settings`, but it names SessionStart's file and toolproc's three observation hooks are
// gone from the settings the child actually reads.
func TestGuardHookSettingsConvergeWhenPreCompactAndStopDisabled(t *testing.T) {
	command := []string{"claude", "--dangerously-skip-permissions", "-p", "go"}

	// 1. PreCompact OFF: creates no file, injects no --settings, reports an empty SettingsPath.
	command, _, preCompactInstall, err := installGuardPreCompactHook(command, guardPreCompactModeOff, guardConvergenceGatewayURL)
	if err != nil {
		t.Fatalf("install PreCompact hook: %v", err)
	}
	if preCompactInstall.SettingsPath != "" {
		t.Fatalf("PreCompact is off but reported a settings file: %+v", preCompactInstall)
	}

	// 2. Stop OFF, offered PreCompact's (empty) path — the same chain guard.go builds.
	command, _, stopInstall, err := installGuardStopHook(command, guardPreCompactModeOff, guardConvergenceGatewayURL,
		preCompactInstall.SettingsPath, 0, 0, 0, 0, guardOperatorDirectedModeWarn)
	if err != nil {
		t.Fatalf("install Stop hook: %v", err)
	}
	if stopInstall.SettingsPath != "" {
		t.Fatalf("Stop is off but reported a settings file: %+v", stopInstall)
	}

	// 3. toolproc ON, offered Stop's then PreCompact's path (both empty), so it is the FIRST
	//    installer to create the file and the one that injects the argv's only --settings.
	toolprocOffered := stopInstall.SettingsPath
	if toolprocOffered == "" {
		toolprocOffered = preCompactInstall.SettingsPath
	}
	command, _, toolprocInstall, err := installGuardToolprocHooksAt(command, guardToolprocModeObserve, toolprocOffered,
		"fak", t.TempDir(), filepath.Join(t.TempDir(), "journal.jsonl"))
	if err != nil {
		t.Fatalf("install toolproc hooks: %v", err)
	}
	if !toolprocInstall.Applied || toolprocInstall.SettingsPath == "" {
		t.Fatalf("toolproc did not create a settings file: %+v", toolprocInstall)
	}
	if toolprocInstall.SettingsPath == toolprocOffered {
		t.Fatalf("test precondition lost: toolproc was expected to CREATE a file, not merge into %q", toolprocOffered)
	}
	if n := countSettingsFlags(command); n != 1 {
		t.Fatalf("after toolproc the argv carries %d %s occurrences, want exactly 1: %v", n, guardClaudeSettingsFlag, command)
	}

	// 4. SessionStart, handed the shared path guard.go resolves. This is the line under test.
	sessionStartSettings := guardSharedHookSettingsPath(toolprocInstall, stopInstall, preCompactInstall)
	command, sessionStartInstall, err := installGuardSessionStartHookAt(command, guardSessionStartModeOn, true,
		"fak", t.TempDir(), sessionStartSettings, "trace-5526")
	if err != nil {
		t.Fatalf("install SessionStart hook: %v", err)
	}
	if !sessionStartInstall.Applied {
		t.Fatalf("SessionStart hook not applied: %+v", sessionStartInstall)
	}

	// The invariant: one --settings on the argv, and it names the file toolproc CREATED.
	if n := countSettingsFlags(command); n != 1 {
		t.Fatalf("argv carries %d %s occurrences, want exactly 1 (last-wins would disarm guard): %v", n, guardClaudeSettingsFlag, command)
	}
	if command[1] != guardClaudeSettingsFlag || command[2] != toolprocInstall.SettingsPath {
		t.Fatalf("the surviving %s names %q, want the file toolproc created %q: %v",
			guardClaudeSettingsFlag, command[2], toolprocInstall.SettingsPath, command)
	}
	if sessionStartInstall.SettingsPath != toolprocInstall.SettingsPath {
		t.Fatalf("SessionStart wrote %q, want it merged into toolproc's file %q (the installers must converge on one file)",
			sessionStartInstall.SettingsPath, toolprocInstall.SettingsPath)
	}

	// The consequence the count alone cannot see: the settings file the child reads must still
	// carry toolproc's three observation hooks alongside SessionStart's affordance. A second
	// file loses them to the SETTINGS_HOOKS_DROPPED fold.
	settings := readGuardHookSettings(t, command[2])
	for _, event := range []string{"PreToolUse", "PostToolUse", "SessionEnd", "SessionStart"} {
		if len(settings.Hooks[event]) == 0 {
			t.Fatalf("hook event %q missing from the child's settings file %s: the installers did not converge (%+v)",
				event, command[2], settings.Hooks)
		}
	}
}

// TestGuardSharedHookSettingsPathPrefersTheCreatedFile pins the resolver itself across the
// installer on/off combinations guard.go can present it with. The record whose installer ran
// LAST is preferred, because a merging installer records the file it merged INTO — so the
// latest non-empty SettingsPath always names the converged file.
func TestGuardSharedHookSettingsPathPrefersTheCreatedFile(t *testing.T) {
	for _, tc := range []struct {
		name      string
		toolproc  string
		stop      string
		preCompac string
		want      string
	}{
		{name: "all off", want: ""},
		{name: "precompact created it", preCompac: "/pc.json", want: "/pc.json"},
		{name: "stop created it", stop: "/stop.json", want: "/stop.json"},
		{name: "toolproc created it (#5526)", toolproc: "/toolproc.json", want: "/toolproc.json"},
		{name: "toolproc merged into precompact's", toolproc: "/pc.json", stop: "/pc.json", preCompac: "/pc.json", want: "/pc.json"},
		{name: "toolproc off after stop created it", stop: "/stop.json", preCompac: "/pc.json", want: "/stop.json"},
		{name: "blank is not a path", toolproc: "   ", stop: "/stop.json", want: "/stop.json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := guardSharedHookSettingsPath(
				guardToolprocInstall{SettingsPath: tc.toolproc},
				guardStopHookInstall{SettingsPath: tc.stop},
				guardPreCompactInstall{SettingsPath: tc.preCompac},
			)
			if got != tc.want {
				t.Fatalf("guardSharedHookSettingsPath = %q, want %q", got, tc.want)
			}
		})
	}
}
