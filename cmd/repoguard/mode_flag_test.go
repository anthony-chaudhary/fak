package main

// Tests for the --mode hook flag (exec-form hooks run with no shell, so they
// cannot prefix FAK_REPO_GUARD=...; the flag carries the posture instead) and a
// verbatim parity port of the Python original's _selftest() case table
// (tools/repo_guard.py) against the Go core.
//
// Precedence under test: env FAK_REPO_GUARD (trimmed, lower-cased) wins when
// non-empty; else the --mode flag (trimmed, lower-cased); else "enforce".

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

// modeFlagClearEnv pins every env knob the hook reads to its default so the
// ambient host environment can never leak into a mode assertion.
func modeFlagClearEnv(t *testing.T) {
	t.Helper()
	t.Setenv("FAK_REPO_GUARD", "")
	t.Setenv("FAK_REPO_GUARD_SEVERITY", "")
	t.Setenv("FAK_SLEEP_THRESHOLD_S", "")
	t.Setenv("FAK_REPO_GUARD_DECISIONS", "")
}

// modeFlagWorkspace is a throwaway repo root: a temp dir holding a .git dir, so
// the hook resolves its workspace (and therefore its decision journal) inside
// the temp dir rather than inside this checkout.
func modeFlagWorkspace(t *testing.T) string {
	t.Helper()
	ws := t.TempDir()
	if err := os.Mkdir(filepath.Join(ws, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	return ws
}

// modeFlagBashPayload renders a Bash PreToolUse payload rooted at cwd. JSON is
// built with encoding/json so a Windows temp path's backslashes are escaped.
func modeFlagBashPayload(t *testing.T, cwd, command string) string {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"tool_name":  "Bash",
		"cwd":        cwd,
		"session_id": "mode-flag-test",
		"tool_input": map[string]any{"command": command},
	})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	return string(raw)
}

func modeFlagRun(t *testing.T, payload, flagMode string) (int, string, string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	rc := runHookMode(strings.NewReader(payload), &out, &errBuf, flagMode)
	return rc, out.String(), errBuf.String()
}

// modeFlagDecision decodes stdout as a PreToolUse decision; ok is false when
// stdout is empty or not a decision JSON.
func modeFlagDecision(out string) (hookDecision, bool) {
	var d hookDecision
	s := strings.TrimSpace(out)
	if s == "" {
		return d, false
	}
	if err := json.Unmarshal([]byte(s), &d); err != nil {
		return d, false
	}
	return d, true
}

func TestResolveModePrecedence(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  string
		flag string
		want string
	}{
		{"env-empty/flag-empty->enforce", "", "", "enforce"},
		{"env-empty/flag-warn->warn", "", "warn", "warn"},
		{"env-empty/flag-padded-upper-WARN->warn", "", " WARN ", "warn"},
		{"env-empty/flag-off->off", "", "off", "off"},
		{"env-off/flag-warn->off", "off", "warn", "off"},
		{"env-enforce/flag-warn->enforce", "enforce", "warn", "enforce"},
		{"env-padded-mixed-Warn/flag-empty->warn", " Warn ", "", "warn"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_REPO_GUARD", tc.env)
			if got := resolveMode(tc.flag); got != tc.want {
				t.Errorf("resolveMode(%q) with FAK_REPO_GUARD=%q = %q, want %q", tc.flag, tc.env, got, tc.want)
			}
		})
	}

	// "unset" is distinct from "set to empty" at the OS level; both must fall
	// through to the flag. t.Setenv registers the restore, then Unsetenv removes it.
	t.Run("env-unset/flag-empty->enforce", func(t *testing.T) {
		t.Setenv("FAK_REPO_GUARD", "")
		if err := os.Unsetenv("FAK_REPO_GUARD"); err != nil {
			t.Fatalf("unsetenv: %v", err)
		}
		if got := resolveMode(""); got != "enforce" {
			t.Errorf("resolveMode(\"\") with FAK_REPO_GUARD unset = %q, want enforce", got)
		}
		if got := resolveMode("warn"); got != "warn" {
			t.Errorf("resolveMode(\"warn\") with FAK_REPO_GUARD unset = %q, want warn", got)
		}
	})
}

// BUILD_CACHE_CLEAN_RACE is the one rung that denies under the default posture;
// --mode warn must cap it to an advisory, and the env var must still beat the flag.
func TestHookModeWarnCapsDenyByDefaultReason(t *testing.T) {
	if got := repoguard.DefaultSeverity(repoguard.ReasonBuildCacheCleanRace); got != repoguard.SeverityDeny {
		t.Fatalf("precondition: default severity of %s = %v, want deny", repoguard.ReasonBuildCacheCleanRace, got)
	}
	for _, tc := range []struct {
		name     string
		env      string
		flag     string
		wantDeny bool
	}{
		{"flag-empty->deny", "", "", true},
		{"flag-warn->advisory", "", "warn", false},
		{"env-enforce-beats-flag-warn->deny", "enforce", "warn", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			modeFlagClearEnv(t)
			t.Setenv("FAK_REPO_GUARD", tc.env)
			ws := modeFlagWorkspace(t)
			rc, out, errOut := modeFlagRun(t, modeFlagBashPayload(t, ws, "go clean -cache"), tc.flag)
			if rc != 0 {
				t.Fatalf("rc = %d, want 0 (the hook always fails open; a deny rides stdout)", rc)
			}
			d, isDecision := modeFlagDecision(out)
			if tc.wantDeny {
				if !isDecision || d.HookSpecificOutput.PermissionDecision != "deny" {
					t.Fatalf("stdout = %q, want a \"permissionDecision\":\"deny\" decision JSON", out)
				}
				if !strings.Contains(out, `"permissionDecision":"deny"`) {
					t.Errorf("stdout = %q, want the literal \"permissionDecision\":\"deny\" field", out)
				}
				if !strings.Contains(d.HookSpecificOutput.PermissionDecisionReason, repoguard.ReasonBuildCacheCleanRace) {
					t.Errorf("deny reason = %q, want it to name %s", d.HookSpecificOutput.PermissionDecisionReason, repoguard.ReasonBuildCacheCleanRace)
				}
				return
			}
			if strings.TrimSpace(out) != "" || strings.Contains(out, "deny") {
				t.Fatalf("stdout = %q, want empty (warn mode must never emit a deny decision)", out)
			}
			if !strings.Contains(errOut, "advisory") || !strings.Contains(errOut, repoguard.ReasonBuildCacheCleanRace) {
				t.Errorf("stderr = %q, want an advisory naming %s", errOut, repoguard.ReasonBuildCacheCleanRace)
			}
		})
	}
}

func TestHookModeOffIsSilent(t *testing.T) {
	modeFlagClearEnv(t)
	ws := modeFlagWorkspace(t)
	rc, out, errOut := modeFlagRun(t, modeFlagBashPayload(t, ws, "sleep 300"), "off")
	if rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if out != "" {
		t.Errorf("stdout = %q, want empty under --mode off", out)
	}
	if errOut != "" {
		t.Errorf("stderr = %q, want empty under --mode off", errOut)
	}
}

func TestHookModeWarnForegroundSleepAdvisory(t *testing.T) {
	modeFlagClearEnv(t)
	ws := modeFlagWorkspace(t)
	rc, out, errOut := modeFlagRun(t, modeFlagBashPayload(t, ws, "sleep 300"), "warn")
	if rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if strings.TrimSpace(out) != "" {
		t.Errorf("stdout = %q, want empty (advisory never denies)", out)
	}
	if !strings.Contains(errOut, repoguard.ReasonForegroundSleep) {
		t.Errorf("stderr = %q, want a %s advisory", errOut, repoguard.ReasonForegroundSleep)
	}
}

func TestHookModeFailOpenOnMalformedPayload(t *testing.T) {
	modeFlagClearEnv(t)
	// A malformed payload has no cwd, so the hook falls back to the process cwd
	// for its journal; pin the journal into a temp dir so the test never writes
	// into this checkout.
	t.Setenv("FAK_REPO_GUARD_DECISIONS", filepath.Join(t.TempDir(), "decisions.jsonl"))
	rc, out, _ := modeFlagRun(t, `{this is not json`, "warn")
	if rc != 0 {
		t.Fatalf("rc = %d, want 0 (fail-open)", rc)
	}
	if strings.Contains(out, "deny") || strings.TrimSpace(out) != "" {
		t.Errorf("stdout = %q, want no deny decision on a malformed payload", out)
	}
}

// pyRow is one (tool, input) row of the Python _selftest() table.
type pyRow struct {
	tool  string
	input map[string]any
}

func pyCmd(c string) map[string]any { return map[string]any{"command": c} }
func pyFP(p string) map[string]any  { return map[string]any{"file_path": p} }

// pyRowName names a subtest after its row: tool, the command/file_path, and any
// extra input keys (e.g. run_in_background) in sorted order.
func pyRowName(r pyRow) string {
	var subject string
	var extras []string
	for k, v := range r.input {
		switch k {
		case "command", "file_path":
			subject = fmt.Sprint(v)
		default:
			extras = append(extras, fmt.Sprintf("%s=%v", k, v))
		}
	}
	sort.Strings(extras)
	name := r.tool + " " + subject
	if len(extras) > 0 {
		name += " [" + strings.Join(extras, ",") + "]"
	}
	return name
}

func countReason(vs []repoguard.Violation, reason string) int {
	n := 0
	for _, v := range vs {
		if v.Reason == reason {
			n++
		}
	}
	return n
}

// TestPythonSelftestParity ports tools/repo_guard.py _selftest() VERBATIM: the
// same WS/HOME/SAFE fixture roots and the same deny / allow / sleep_deny /
// sleep_allow rows. Python's evaluate() only classified out-of-tree writes and
// its sleep_findings() only FOREGROUND_SLEEP, while the Go Evaluate runs every
// rung — so each row is checked against its own reason class, not a raw count.
func TestPythonSelftestParity(t *testing.T) {
	const ws = "C:/Users/u/work/fak"
	const home = "C:/Users/u"
	safe := []string{"/tmp", "/var/tmp", "C:/Users/u/.cache", "C:/Users/u/Downloads"}
	safe = append(safe, repoguard.AgentStateRoots(home, []string{".claude", ".claude-gem8-netra", ".claudex", "Documents"})...)
	safe = append(safe, repoguard.PrivateCompanionRoots(ws)...)

	deny := []pyRow{ // MUST produce >=1 OUT_OF_TREE_WRITE
		{"Bash", pyCmd("go build -o ../tools/.bin/fak.exe ./cmd/fak")},
		{"Bash", pyCmd("rm -rf ../tools")},
		{"Bash", pyCmd("rm -rf /c/Users/u/work/tools")},
		{"Bash", pyCmd("echo x > ../tools/y")},
		{"Bash", pyCmd("cp a.txt ../tools/b.txt")},
		{"Bash", pyCmd("mv internal/x ../sibling/x")},
		{"Bash", pyCmd("rm -rf /")},
		{"Bash", pyCmd("cd src && rm -rf ../../other")},
		{"Write", pyFP("../tools/poison.txt")},
		{"Write", pyFP("C:/Users/u/work/tools/poison.txt")},
		// the broadened allow-list must NOT leak: a private-companion look-alike,
		// an unrelated sibling, and a .claude look-alike all still DENY.
		{"Write", pyFP("C:/Users/u/work/fak-private-evil/x.md")},
		{"Write", pyFP("C:/Users/u/work/fak-ci/x.md")},
		{"Write", pyFP("C:/Users/u/.claudex/leak.md")},
	}
	allow := []pyRow{ // MUST produce ZERO OUT_OF_TREE_WRITE
		{"Bash", pyCmd("go build -o fak.exe ./cmd/fak")},
		{"Bash", pyCmd("go build -o tools/.bin/fak.exe ./cmd/fak")},
		{"Bash", pyCmd("rm -rf ./build")},
		{"Bash", pyCmd("rm -rf internal/model/.cache")},
		{"Bash", pyCmd("echo x > /tmp/log.txt")},
		{"Bash", pyCmd("cp a.txt /var/tmp/b.txt")},
		{"Bash", pyCmd("cp a.txt ~/.cache/b.txt")},
		{"Bash", pyCmd("grep -o ../foo internal/policy/x.go")}, // read, no write verb
		{"Bash", pyCmd("cat ../README.md")},
		{"Bash", pyCmd("mv internal/a internal/b")},
		{"Write", pyFP("internal/policy/x.go")},
		{"Write", pyFP("examples/repo-guard-policy.json")},
		// the agent's own state/memory tree (per-account variant) + private companion.
		{"Write", pyFP("C:/Users/u/.claude-gem8-netra/projects/C--Users-u-work-fak/memory/note.md")},
		{"Write", pyFP("C:/Users/u/work/fak-private/MEMORY-glm52-2026-06-21.md")},
		// the null / std-stream device sinks: harmless, never a sibling repo.
		{"Bash", pyCmd("make ci > /dev/null 2>&1")},
		{"Bash", pyCmd("go test ./... > /dev/null")},
		{"Bash", pyCmd("echo done >> /dev/stderr")},
	}
	// Foreground-sleep (#2366): a long blocking sleep is flagged; short /
	// backgrounded / harness-backgrounded ones are not. Threshold is the default 120s.
	sleepDeny := []pyRow{ // MUST produce >=1 FOREGROUND_SLEEP
		{"Bash", pyCmd("sleep 300")},
		{"Bash", pyCmd("sleep 1500; echo TIMER_DONE_POLL")},
		{"Bash", pyCmd("for i in $(seq 1 16); do sleep 300; probe; done")},
		{"Bash", pyCmd("sleep 5m")},
		{"Bash", pyCmd("sudo sleep 300 && echo up")},
		{"PowerShell", pyCmd("Start-Sleep -Seconds 300")},
		{"PowerShell", pyCmd("Start-Sleep 240")},
	}
	sleepAllow := []pyRow{ // MUST produce ZERO FOREGROUND_SLEEP
		{"Bash", pyCmd("sleep 5")},
		{"Bash", pyCmd("sleep 30 && go test ./...")},
		{"Bash", pyCmd("echo sleep 300")},                                           // sleep is an ARGUMENT, not a command
		{"Bash", map[string]any{"command": "sleep 300", "run_in_background": true}}, // backgrounded at the harness level
		{"Bash", pyCmd("sleep 300 &")},                                              // shell-backgrounded, never blocks
		{"PowerShell", pyCmd("Start-Sleep -Milliseconds 5000")},
		{"PowerShell", pyCmd("Start-Sleep 5")},
	}

	// The Python table's own counts, so a dropped row is a loud failure.
	if len(deny) != 13 || len(allow) != 17 || len(sleepDeny) != 7 || len(sleepAllow) != 7 {
		t.Fatalf("ported table drifted from the Python original: deny=%d allow=%d sleep_deny=%d sleep_allow=%d, want 13/17/7/7",
			len(deny), len(allow), len(sleepDeny), len(sleepAllow))
	}

	outOfTree := repoguard.Reason // OUT_OF_TREE_WRITE
	groups := []struct {
		name    string
		rows    []pyRow
		reason  string
		flagged bool
	}{
		{"deny", deny, outOfTree, true},
		{"allow", allow, outOfTree, false},
		{"sleep_deny", sleepDeny, repoguard.ReasonForegroundSleep, true},
		{"sleep_allow", sleepAllow, repoguard.ReasonForegroundSleep, false},
	}
	for _, g := range groups {
		t.Run(g.name, func(t *testing.T) {
			for _, r := range g.rows {
				t.Run(pyRowName(r), func(t *testing.T) {
					vs := repoguard.EvaluateWithHints(r.tool, r.input, ws, safe, repoguard.Hints{})
					n := countReason(vs, g.reason)
					if g.flagged && n == 0 {
						t.Errorf("%s %v: got 0 %s findings, want >=1 (all findings: %+v)", r.tool, r.input, g.reason, vs)
					}
					if !g.flagged && n != 0 {
						t.Errorf("%s %v: got %d %s findings, want 0 (all findings: %+v)", r.tool, r.input, n, g.reason, vs)
					}
				})
			}
		})
	}
}

// FAK_SLEEP_THRESHOLD_S is read in the command layer: a positive integer
// overrides the 120 s FOREGROUND_SLEEP default; empty, non-numeric, or <= 0
// falls back to the default (a typo never fails closed). Driven through
// --mode warn so every finding lands on stderr, never as a deny.
func TestHookSleepThresholdEnv(t *testing.T) {
	for _, tc := range []struct {
		name         string
		env          string
		command      string
		wantAdvisory bool
	}{
		{"env-60/sleep-90->advisory", "60", "sleep 90", true},
		{"env-600/sleep-300->silent", "600", "sleep 300", false},
		{"env-abc/sleep-300->advisory(default-120)", "abc", "sleep 300", true},
		{"env-minus5/sleep-300->advisory(default-120)", "-5", "sleep 300", true},
		{"env-zero/sleep-300->advisory(default-120)", "0", "sleep 300", true},
		{"env-empty/sleep-300->advisory(default-120)", "", "sleep 300", true},
		{"env-abc/sleep-90->silent(default-120)", "abc", "sleep 90", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			modeFlagClearEnv(t)
			t.Setenv("FAK_SLEEP_THRESHOLD_S", tc.env)
			ws := modeFlagWorkspace(t)
			rc, out, errOut := modeFlagRun(t, modeFlagBashPayload(t, ws, tc.command), "warn")
			if rc != 0 {
				t.Fatalf("rc = %d, want 0", rc)
			}
			if strings.TrimSpace(out) != "" {
				t.Errorf("stdout = %q, want empty (warn mode never denies)", out)
			}
			got := strings.Contains(errOut, repoguard.ReasonForegroundSleep)
			if got != tc.wantAdvisory {
				t.Errorf("FAK_SLEEP_THRESHOLD_S=%q %q: stderr = %q, want %s advisory = %v",
					tc.env, tc.command, errOut, repoguard.ReasonForegroundSleep, tc.wantAdvisory)
			}
		})
	}
}

// --check reads the same knob as --hook.
func TestCheckSleepThresholdEnv(t *testing.T) {
	modeFlagClearEnv(t)
	ws := modeFlagWorkspace(t)

	t.Setenv("FAK_SLEEP_THRESHOLD_S", "60")
	var out bytes.Buffer
	if rc := runCheck("sleep 90", ws, false, &out); rc != 0 {
		t.Fatalf("runCheck(sleep 90) rc = %d, want 0 (advisory-only)", rc)
	}
	if !strings.Contains(out.String(), repoguard.ReasonForegroundSleep) {
		t.Errorf("FAK_SLEEP_THRESHOLD_S=60 --check \"sleep 90\" = %q, want a %s finding", out.String(), repoguard.ReasonForegroundSleep)
	}

	t.Setenv("FAK_SLEEP_THRESHOLD_S", "600")
	out.Reset()
	if rc := runCheck("sleep 300", ws, false, &out); rc != 0 {
		t.Fatalf("runCheck(sleep 300) rc = %d, want 0", rc)
	}
	if strings.Contains(out.String(), repoguard.ReasonForegroundSleep) {
		t.Errorf("FAK_SLEEP_THRESHOLD_S=600 --check \"sleep 300\" = %q, want no %s finding", out.String(), repoguard.ReasonForegroundSleep)
	}
}

// Benign command prefixes (with their dash-flags) stand in front of a sleep in
// command position without making it an argument; `echo sleep 300` keeps sleep
// an argument and a shell-backgrounded sleep never holds the turn.
func TestSleepPrefixParity(t *testing.T) {
	for _, tc := range []struct {
		command string
		flagged bool
	}{
		{"sudo -E sleep 300", true},
		{"nohup sleep 300", true},
		{"env FOO=1 sleep 300", true},
		{"echo sleep 300", false},
		{"sleep 300 &", false},
	} {
		t.Run(tc.command, func(t *testing.T) {
			vs := repoguard.EvaluateWithHints("Bash", pyCmd(tc.command), wsTest, nil, repoguard.Hints{})
			n := countReason(vs, repoguard.ReasonForegroundSleep)
			if tc.flagged && n == 0 {
				t.Errorf("Bash %q: got 0 %s findings, want >=1 (all findings: %+v)", tc.command, repoguard.ReasonForegroundSleep, vs)
			}
			if !tc.flagged && n != 0 {
				t.Errorf("Bash %q: got %d %s findings, want 0 (all findings: %+v)", tc.command, n, repoguard.ReasonForegroundSleep, vs)
			}
		})
	}
}

// A harness-backgrounded PowerShell call never holds the turn either; the same
// Start-Sleep in the foreground is the control and must still be flagged.
func TestPowerShellRunInBackgroundSkipsSleep(t *testing.T) {
	const command = "Start-Sleep -Seconds 300"
	bg := repoguard.EvaluateWithHints("PowerShell",
		map[string]any{"command": command, "run_in_background": true}, wsTest, nil, repoguard.Hints{})
	if n := countReason(bg, repoguard.ReasonForegroundSleep); n != 0 {
		t.Errorf("PowerShell %q [run_in_background=true]: got %d %s findings, want 0 (all findings: %+v)",
			command, n, repoguard.ReasonForegroundSleep, bg)
	}
	fg := repoguard.EvaluateWithHints("PowerShell", pyCmd(command), wsTest, nil, repoguard.Hints{})
	if countReason(fg, repoguard.ReasonForegroundSleep) == 0 {
		t.Errorf("control: PowerShell %q in the foreground: got 0 %s findings, want >=1", command, repoguard.ReasonForegroundSleep)
	}
}
