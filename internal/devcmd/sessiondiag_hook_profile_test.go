package devcmd

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBuildCodexHookProfileExplainsPrecedenceAndCompetingHomes(t *testing.T) {
	now := time.Date(2026, 8, 17, 22, 30, 0, 0, time.UTC)
	active := filepath.Join(t.TempDir(), ".codex-profile")
	defaultHome := filepath.Join(filepath.Dir(active), ".codex")
	workspace := filepath.Join(t.TempDir(), "repo")
	workspaceHook := filepath.Join(workspace, ".codex", "hooks.json")
	pluginHook := filepath.Join(active, "plugins", "cache", "dos", "dos-kernel", "0.30.0", "hooks", "hooks.json")

	report := buildCodexHookProfile(codexHookProfileBuildInput{
		ObservedAt:        now,
		WorkingDirectory:  workspace,
		DeclaredCodexHome: active,
		ActiveCodexHome:   active,
		DefaultCodexHome:  defaultHome,
		LogDBPath:         filepath.Join(active, "logs_2.sqlite"),
		TrunkHead:         "15a8dd0423",
		Homes: []codexHomeObservation{
			{Path: active, Exists: true, Active: true, HasConfig: true, HasLogDB: true},
			{Path: defaultHome, Exists: true, Active: false, HasConfig: true, HasLogDB: true},
		},
		Hooks: []codexEffectiveHook{
			{
				Key:          workspaceHook + ":user_prompt_submit:0:0",
				EventName:    "userPromptSubmit",
				HandlerType:  "command",
				Command:      "fak sessions codex-loop-hook",
				SourcePath:   workspaceHook,
				Source:       "project",
				DisplayOrder: 0,
				Enabled:      true,
				CurrentHash:  "sha256:workspace",
				TrustStatus:  "trusted",
				Executables: []codexExecutableIdentity{{
					Name: "fak", Path: `C:\Users\USER\bin\fak.exe`, Exists: true,
					Build: "bb5089042542",
				}},
			},
			{
				Key:          "dos-kernel@dos:hooks/hooks.json:pre_tool_use:0:0",
				EventName:    "preToolUse",
				HandlerType:  "command",
				Command:      "dos-hook pretool --workspace .",
				SourcePath:   pluginHook,
				Source:       "plugin",
				PluginID:     "dos-kernel@dos",
				DisplayOrder: 1,
				Enabled:      true,
				CurrentHash:  "sha256:pre",
				TrustStatus:  "trusted",
				Executables: []codexExecutableIdentity{{
					Name: "dos-hook", Path: filepath.Join(filepath.Dir(filepath.Dir(pluginHook)), "bin", "dos-hook"), Exists: true,
				}},
			},
			{
				Key:          "dos-kernel@dos:hooks/hooks.json:post_tool_use:0:0",
				EventName:    "postToolUse",
				HandlerType:  "command",
				Command:      "dos-hook posttool --workspace .",
				SourcePath:   pluginHook,
				Source:       "plugin",
				PluginID:     "dos-kernel@dos",
				DisplayOrder: 2,
				Enabled:      true,
				CurrentHash:  "sha256:post",
				TrustStatus:  "trusted",
				Executables: []codexExecutableIdentity{{
					Name: "dos-hook", Path: filepath.Join(filepath.Dir(filepath.Dir(pluginHook)), "bin", "dos-hook"), Exists: true,
				}},
			},
		},
		RecentToolFailures: codexRecentToolFailures{
			RowWindow:      20_000,
			RouterErrors:   12,
			Interpretation: "router errors are failed tool calls, not proof that a hook refused them",
		},
	})

	if report.ActiveCodexHome != active || report.LogStore.ProfileMatch != true {
		t.Fatalf("profile=%+v log=%+v", report.Profile, report.LogStore)
	}
	if len(report.Homes) != 2 || report.Homes[1].Status != hookProfileStatusShadowed {
		t.Fatalf("homes=%+v", report.Homes)
	}
	if len(report.Hooks) != 3 ||
		report.Hooks[0].Layer != "workspace" ||
		report.Hooks[1].Layer != "plugin" ||
		report.Hooks[2].Layer != "plugin" {
		t.Fatalf("hooks=%+v", report.Hooks)
	}
	if report.Hooks[0].Status != hookProfileStatusStaleBinary ||
		report.Hooks[0].Executables[0].Build != "bb5089042542" {
		t.Fatalf("workspace hook=%+v", report.Hooks[0])
	}
	for _, idx := range []int{1, 2} {
		hook := report.Hooks[idx]
		if hook.Status != hookProfileStatusReady || hook.CurrentHash == "" || len(hook.Executables) == 0 {
			t.Fatalf("pre/post hook=%+v", hook)
		}
	}
	if report.RecentToolFailures.RouterErrors != 12 ||
		!strings.Contains(report.RecentToolFailures.Interpretation, "not proof") {
		t.Fatalf("failures=%+v", report.RecentToolFailures)
	}
	if report.Verdict != hookProfileVerdictAction {
		t.Fatalf("verdict=%s diagnoses=%+v", report.Verdict, report.Diagnoses)
	}
	for _, status := range []string{hookProfileStatusShadowed, hookProfileStatusStaleBinary} {
		found := false
		for _, diagnosis := range report.Diagnoses {
			if diagnosis.Type == status && diagnosis.Remediation != "" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing diagnosis=%s diagnoses=%+v", status, report.Diagnoses)
		}
	}
}

func TestBuildCodexHookProfileEmitsTypedHookDiagnoses(t *testing.T) {
	report := buildCodexHookProfile(codexHookProfileBuildInput{
		ObservedAt:       time.Date(2026, 8, 17, 22, 30, 0, 0, time.UTC),
		WorkingDirectory: t.TempDir(),
		ActiveCodexHome:  t.TempDir(),
		DefaultCodexHome: t.TempDir(),
		TrunkHead:        "new-build",
		Hooks: []codexEffectiveHook{
			{Key: "disabled", EventName: "preToolUse", HandlerType: "command", Enabled: false, TrustStatus: "trusted"},
			{Key: "modified", EventName: "postToolUse", HandlerType: "command", Enabled: true, TrustStatus: "modified"},
			{
				Key: "missing", EventName: "preToolUse", HandlerType: "command", Enabled: true, TrustStatus: "trusted",
				Executables: []codexExecutableIdentity{{Name: "missing-hook", Exists: false}},
			},
			{Key: "unknown", EventName: "postToolUse", HandlerType: "mcpTool", Enabled: true, TrustStatus: "trusted"},
		},
	})

	got := map[string]string{}
	for _, hook := range report.Hooks {
		got[hook.Key] = hook.Status
	}
	want := map[string]string{
		"disabled": hookProfileStatusDisabled,
		"modified": hookProfileStatusStaleHash,
		"missing":  hookProfileStatusMissingExecutable,
		"unknown":  hookProfileStatusUnknown,
	}
	for key, status := range want {
		if got[key] != status {
			t.Fatalf("%s status=%q want=%q report=%+v", key, got[key], status, report)
		}
	}
	for _, status := range []string{
		hookProfileStatusDisabled,
		hookProfileStatusStaleHash,
		hookProfileStatusMissingExecutable,
		hookProfileStatusUnknown,
	} {
		found := false
		for _, diagnosis := range report.Diagnoses {
			if diagnosis.Type == status && diagnosis.Remediation != "" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing diagnosis=%s diagnoses=%+v", status, report.Diagnoses)
		}
	}
}

func TestRunCodexHookProfileWritesCapturedJSON(t *testing.T) {
	now := time.Date(2026, 8, 17, 22, 30, 0, 0, time.UTC)
	active := filepath.Join(t.TempDir(), ".codex-profile")
	query := func(codexHookProfileQueryInput) (codexHookProfileBuildInput, error) {
		return codexHookProfileBuildInput{
			ObservedAt:        now,
			WorkingDirectory:  `C:\work\fak`,
			DeclaredCodexHome: active,
			ActiveCodexHome:   active,
			DefaultCodexHome:  filepath.Join(filepath.Dir(active), ".codex"),
			LogDBPath:         filepath.Join(active, "logs_2.sqlite"),
			TrunkHead:         "15a8dd0423",
			Homes:             []codexHomeObservation{{Path: active, Exists: true, Active: true, HasLogDB: true}},
		}, nil
	}
	var stdout, stderr bytes.Buffer
	code := runCodexHookProfileWith(&stdout, &stderr, codexHookProfileQueryInput{}, true, query)
	if code != 0 || stderr.Len() != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report codexHookProfileReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != hookProfileSchemaVersion ||
		report.ActiveCodexHome != active ||
		report.ObservedAt != now {
		t.Fatalf("report=%s", stdout.String())
	}
}

func TestBuildCodexHookProfileFlagsProfileLogStoreMismatch(t *testing.T) {
	root := t.TempDir()
	active := filepath.Join(root, ".codex-profile")
	legacy := filepath.Join(root, ".codex")
	report := buildCodexHookProfile(codexHookProfileBuildInput{
		ObservedAt:        time.Date(2026, 8, 17, 22, 30, 0, 0, time.UTC),
		WorkingDirectory:  root,
		DeclaredCodexHome: active,
		ActiveCodexHome:   active,
		DefaultCodexHome:  legacy,
		LogDBPath:         filepath.Join(legacy, "logs_2.sqlite"),
		ActiveLogDBPath:   filepath.Join(active, "logs_2.sqlite"),
	})
	if report.LogStore.ProfileMatch {
		t.Fatalf("log store unexpectedly matched: %+v", report.LogStore)
	}
	found := false
	for _, diagnosis := range report.Diagnoses {
		if diagnosis.Type == hookProfileStatusShadowed &&
			diagnosis.Subject == report.LogStore.RequestedPath &&
			diagnosis.Remediation != "" {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnoses=%+v", report.Diagnoses)
	}
}

func TestDefaultCodexLogDBUsesActiveCodexHome(t *testing.T) {
	active := t.TempDir()
	t.Setenv("CODEX_HOME", active)
	if got, want := defaultCodexLogDB(), filepath.Join(active, "logs_2.sqlite"); got != want {
		t.Fatalf("default=%q want=%q", got, want)
	}

	t.Setenv("CODEX_HOME", "")
	if got := defaultCodexLogDB(); filepath.Base(filepath.Dir(got)) != ".codex" {
		t.Fatalf("fallback=%q", got)
	}
}
