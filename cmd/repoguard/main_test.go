package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

const wsTest = "C:/Users/u/work/fak"

func TestSelftestPasses(t *testing.T) {
	var buf bytes.Buffer
	if rc := runSelftest(&buf); rc != 0 {
		t.Errorf("runSelftest returned %d; output:\n%s", rc, buf.String())
	}
}

func runHookString(t *testing.T, payload string) (int, string, string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	rc := runHook(strings.NewReader(payload), &out, &errBuf)
	return rc, out.String(), errBuf.String()
}

const ootPayload = `{"tool_name":"Bash","cwd":"` + wsTest + `","tool_input":{"command":"rm -rf ../tools"}}`

// Default posture: out-of-tree is SILENT-recorded — allow, no deny JSON on
// stdout, and nothing on stderr (the model's context stays clean).
func TestHookRecordsOutOfTreeSilentlyByDefault(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	rc, out, errOut := runHookString(t, ootPayload)
	if rc != 0 {
		t.Fatalf("hook rc = %d, want 0 (allow)", rc)
	}
	if strings.TrimSpace(out) != "" {
		t.Errorf("stdout = %q, want empty (no deny JSON by default)", out)
	}
	if strings.TrimSpace(errOut) != "" {
		t.Errorf("stderr = %q, want empty (silent record must not perturb the model)", errOut)
	}
}

// A security-minded operator dials the rung back up to deny per reason.
func TestHookDeniesOutOfTreeWhenSeverityDeny(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "OUT_OF_TREE_WRITE=deny")
	rc, out, _ := runHookString(t, ootPayload)
	if rc != 0 {
		t.Fatalf("hook rc = %d, want 0", rc)
	}
	var decision hookDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &decision); err != nil {
		t.Fatalf("hook stdout is not a decision JSON: %v (out=%q)", err, out)
	}
	if decision.HookSpecificOutput.PermissionDecision != "deny" {
		t.Errorf("permissionDecision = %q, want deny", decision.HookSpecificOutput.PermissionDecision)
	}
}

// The global master switch caps a per-reason deny back to advisory: no deny JSON.
func TestHookGlobalWarnCapsPerReasonDeny(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "warn")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "OUT_OF_TREE_WRITE=deny")
	rc, out, errOut := runHookString(t, ootPayload)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("global warn must cap the deny: rc=%d out=%q, want (0, \"\")", rc, out)
	}
	if !strings.Contains(errOut, "advisory") {
		t.Errorf("stderr = %q, want an advisory line (warn cap)", errOut)
	}
}

// Silencing a rung entirely: no deny, no stderr, allow.
func TestHookSeverityOffSilencesRung(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "OUT_OF_TREE_WRITE=off")
	rc, out, errOut := runHookString(t, ootPayload)
	if rc != 0 || strings.TrimSpace(out) != "" || strings.TrimSpace(errOut) != "" {
		t.Errorf("=off should fully silence: rc=%d out=%q err=%q", rc, out, errOut)
	}
}

// The would-hang interactive rung stays advisory by default (its fix-hint helps
// the agent): allow, no deny JSON, but a structured advisory on stderr.
func TestHookInteractiveWarnsByDefault(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	rc, out, errOut := runHookString(t,
		`{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"git rebase -i HEAD~3"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("interactive default: rc=%d out=%q, want (0, \"\")", rc, out)
	}
	if !strings.Contains(errOut, "INTERACTIVE_HANG") {
		t.Errorf("stderr = %q, want the INTERACTIVE_HANG advisory", errOut)
	}
}

func TestHookAllowsInRepo(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ./build"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("in-repo op: rc=%d out=%q, want (0, \"\")", rc, out)
	}
}

func TestHookWarnModeAllows(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "warn")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ../tools"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("warn mode: rc=%d out=%q, want (0, \"\") - no deny JSON on stdout", rc, out)
	}
}

func TestHookOffModeDisables(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "off")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ../tools"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("off mode: rc=%d out=%q, want (0, \"\")", rc, out)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("synthetic read failure") }

func TestHookMalformedPayloadHasDistinctCountableReason(t *testing.T) {
	journal := t.TempDir() + "/decisions.jsonl"
	t.Setenv("FAK_REPO_GUARD_DECISIONS", journal)

	rc, out, errOut := runHookString(t, `{not-json`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("malformed payload must fail open: rc=%d out=%q", rc, out)
	}
	if !strings.Contains(errOut, "malformed hook payload, allowing") || strings.Contains(errOut, "internal error") {
		t.Fatalf("stderr = %q, want distinct malformed-payload label", errOut)
	}

	var summary bytes.Buffer
	if rc := runSummary(journal, 10, true, &summary); rc != 0 {
		t.Fatalf("summary rc=%d output=%s", rc, summary.String())
	}
	var got struct {
		ByReason map[string]int `json:"by_reason"`
	}
	if err := json.Unmarshal(summary.Bytes(), &got); err != nil {
		t.Fatalf("decode summary: %v (%s)", err, summary.String())
	}
	if got.ByReason["MALFORMED_HOOK_PAYLOAD"] != 1 {
		t.Fatalf("by_reason=%v, want one MALFORMED_HOOK_PAYLOAD", got.ByReason)
	}
}

func TestHookInternalReadErrorKeepsGuardBugLabel(t *testing.T) {
	var out, errOut bytes.Buffer
	rc := runHook(failingReader{}, &out, &errOut)
	if rc != 0 || strings.TrimSpace(out.String()) != "" {
		t.Fatalf("internal error must fail open: rc=%d out=%q", rc, out.String())
	}
	if !strings.Contains(errOut.String(), "internal error, allowing (synthetic read failure)") || strings.Contains(errOut.String(), "malformed hook payload") {
		t.Fatalf("stderr = %q, want unchanged internal-error label", errOut.String())
	}
}

func TestHookDeniesAmbientBuildCacheCleanAcrossShellAliases(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	for _, tc := range []struct {
		tool, field, command string
	}{
		{"Bash", "command", "go clean -cache"},
		{"Bash", "command", "go clean -cache -testcache"},
		{"Bash", "command", "go clean -testcache -cache"},
		{"shell_command", "command", "env GOENV=off go clean -cache"},
		{"functions.shell_command", "command", "cd /tmp && go clean -cache"},
		{"exec_command", "cmd", "go clean --cache=true"},
	} {
		t.Run(tc.tool+"/"+tc.command, func(t *testing.T) {
			input, err := json.Marshal(map[string]any{
				"tool_name": tc.tool,
				"cwd":       wsTest,
				"tool_input": map[string]any{
					tc.field: tc.command,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, out, _ := runHookString(t, string(input))
			var decision hookDecision
			if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &decision); err != nil {
				t.Fatalf("hook stdout is not a decision: %v (out=%q)", err, out)
			}
			reason := decision.HookSpecificOutput.PermissionDecisionReason
			if decision.HookSpecificOutput.PermissionDecision != "deny" ||
				!strings.Contains(reason, "BUILD_CACHE_CLEAN_RACE") ||
				!strings.Contains(reason, "fak-dev buildcheck --vet") ||
				!strings.Contains(reason, "OS-temp directory") ||
				!strings.Contains(reason, "BUILD_CACHE_CLEAN_RACE=off") {
				t.Fatalf("decision = %+v, want typed deny with private-cache route and override", decision.HookSpecificOutput)
			}
		})
	}
}

func TestHookAllowsBuildCacheCleanNearMisses(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	for _, command := range []string{
		"go clean -testcache",
		"go test ./...",
		"go vet ./...",
		"echo 'go clean -cache'",
		"rg -n 'go clean -cache' docs",
	} {
		payload, err := json.Marshal(map[string]any{
			"tool_name": "Bash", "cwd": wsTest,
			"tool_input": map[string]any{"command": command},
		})
		if err != nil {
			t.Fatal(err)
		}
		_, out, _ := runHookString(t, string(payload))
		if strings.TrimSpace(out) != "" {
			t.Errorf("near miss %q produced deny output %q", command, out)
		}
	}
}

func TestHookBuildCacheCleanPerReasonOverride(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "BUILD_CACHE_CLEAN_RACE=off")
	_, out, errOut := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"go clean -cache"}}`)
	if strings.TrimSpace(out) != "" || strings.TrimSpace(errOut) != "" {
		t.Fatalf("per-reason override must allow quietly: stdout=%q stderr=%q", out, errOut)
	}
}
