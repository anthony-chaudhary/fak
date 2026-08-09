package repoguard_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// shim_test.go — pins the .claude/settings.json PreToolUse chooser for the repo-guard
// hook (#2073).
//
// The chooser's ONLY job is to pick between the compiled repoguard binary and the
// tools/repo_guard.py source fallback. That decision is a function of what is on disk,
// yet it was paid for with a full CPython interpreter start on EVERY tool call — and
// then an os.walk over cmd/repoguard + internal/repoguard to date-check the binary.
// Measured on the Windows fleet host (interleaved A/B, n=40): the python chooser cost
// 97.3ms mean / 333.1ms p99 while the guard it selects runs in 7.8ms. The interpreter
// was ~69ms of that, the staleness walk ~5ms.
//
// So the chooser is now `sh -c` instead of `python -c`: same argv shape, same
// PATH-resolved command form, same two fallbacks, ~30ms instead of ~97ms on Windows
// (much less on POSIX, where sh is a couple of ms). Measured after: 38.7ms mean /
// 71.5ms p99 — -60% mean, -79% p99.
//
// What is deliberately NOT reproduced is the mtime staleness check. It cost a directory
// walk per tool call to guard against a case `make build` already prevents (the build
// target rebuilds BOTH tools/.bin/repoguard and the cross-built repoguard.exe on every
// green cycle), and cmd/fak/hooks_agent.go dropped it for the same reason: when it fired
// it blanked the binary and fell through to a source path it never confirmed existed.
//
// The two properties that MUST survive, and that these tests pin:
//  1. when the compiled binary is present and executable, the chooser execs it and no
//     Python interpreter enters the request path;
//  2. when it is absent, the chooser still runs the tools/repo_guard.py source fallback —
//     a missing binary must never silently mean "no guard ran".
func preToolUseChooser(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", ".claude", "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Hooks struct {
			PreToolUse []struct {
				Hooks []struct {
					Command string   `json:"command"`
					Args    []string `json:"args"`
				} `json:"hooks"`
			} `json:"PreToolUse"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Hooks.PreToolUse) == 0 || len(cfg.Hooks.PreToolUse[0].Hooks) == 0 {
		t.Fatal("settings.json declares no PreToolUse hook — this test would be vacuous")
	}
	h := cfg.Hooks.PreToolUse[0].Hooks[0]
	return append([]string{h.Command}, h.Args...)
}

// TestPreToolUseChooserIsInterpreterFree is the latency property of #2073 stated as a
// contract: no CPython interpreter may sit in front of the guard on the request path.
// It is the one assertion that cannot be satisfied by making the old shim incrementally
// cheaper, because the interpreter start IS the cost.
func TestPreToolUseChooserIsInterpreterFree(t *testing.T) {
	argv := preToolUseChooser(t)
	if strings.Contains(argv[0], "python") || strings.Contains(argv[0], "py") {
		t.Fatalf("PreToolUse chooser runs a Python interpreter per tool call: %q\n"+
			"the chooser only picks a path; it must not cost an interpreter start (#2073)", argv[0])
	}
	body := strings.Join(argv, " ")
	// The source fallback may still be NAMED (it is the absent-binary path); what must be
	// gone is the per-call directory walk that date-checked the binary.
	if strings.Contains(body, "os.walk") || strings.Contains(body, "getmtime") {
		t.Errorf("PreToolUse chooser still walks the source tree per tool call: %q", body)
	}
	if !strings.Contains(body, "repoguard") {
		t.Errorf("PreToolUse chooser no longer prefers the compiled binary: %q", body)
	}
	if !strings.Contains(body, "repo_guard.py") {
		t.Errorf("PreToolUse chooser dropped the source fallback — an absent binary would "+
			"silently mean no guard ran: %q", body)
	}
}

// TestPreToolUseChooserPrefersCompiledBinary runs the REAL chooser out of settings.json
// against a synthetic root where the "binary" is a script that stamps a marker, and
// proves the compiled path is the one that executes.
func TestPreToolUseChooserPrefersCompiledBinary(t *testing.T) {
	root, binMarker, srcMarker := chooserFixture(t, true)
	runChooser(t, root)
	if _, err := os.Stat(binMarker); err != nil {
		t.Fatalf("compiled binary was not preferred: %v", err)
	}
	if _, err := os.Stat(srcMarker); err == nil {
		t.Fatal("source fallback ran even though the compiled binary was present")
	}
}

// TestPreToolUseChooserFallsBackWhenBinaryAbsent is the safety half: dropping the
// staleness check must not drop the fallback. With no binary on disk at all the source
// guard still has to run.
func TestPreToolUseChooserFallsBackWhenBinaryAbsent(t *testing.T) {
	root, binMarker, srcMarker := chooserFixture(t, false)
	runChooser(t, root)
	if _, err := os.Stat(srcMarker); err != nil {
		t.Fatalf("source fallback did not run with the binary absent: %v", err)
	}
	if _, err := os.Stat(binMarker); err == nil {
		t.Fatal("binary marker exists but no binary was installed")
	}
}

// chooserFixture builds a throwaway repo root carrying a stub repo_guard.py and,
// optionally, a stub executable at the compiled-binary path. Each stub stamps its own
// marker file so the caller can tell which one the chooser picked.
func chooserFixture(t *testing.T, withBinary bool) (root, binMarker, srcMarker string) {
	t.Helper()
	root = t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools", ".bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	binMarker = filepath.Join(root, "binary-ran")
	srcMarker = filepath.Join(root, "source-ran")

	src := "import pathlib; pathlib.Path(" + pythonQuote(srcMarker) + ").write_text('yes')\n"
	if err := os.WriteFile(filepath.Join(root, "tools", "repo_guard.py"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if !withBinary {
		return root, binMarker, srcMarker
	}
	// The chooser tests `[ -x "$B" ]`, so the stub has to be genuinely executable. Under
	// git-bash on Windows the .exe arm is the one probed first.
	name := "repoguard"
	if runtime.GOOS == "windows" {
		name = "repoguard.exe"
	}
	stub := "#!/bin/sh\nprintf yes > " + shQuote(binMarker) + "\n"
	if err := os.WriteFile(filepath.Join(root, "tools", ".bin", name), []byte(stub), 0o755); err != nil {
		t.Fatal(err)
	}
	return root, binMarker, srcMarker
}

// runChooser executes the chooser exactly as the harness would: the argv straight out of
// settings.json, with CLAUDE_PROJECT_DIR pointing at the fixture root and the process cwd
// deliberately somewhere else — the chooser must anchor on the env, never on cwd, because
// a `cd` in one tool call is inherited by every later hook.
func runChooser(t *testing.T, root string) {
	t.Helper()
	argv := preToolUseChooser(t)
	if _, err := exec.LookPath(argv[0]); err != nil {
		t.Skipf("%s unavailable: %v", argv[0], err)
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = t.TempDir() // NOT the repo root: proves CLAUDE_PROJECT_DIR anchoring
	cmd.Env = append(os.Environ(), "CLAUDE_PROJECT_DIR="+filepath.ToSlash(root))
	cmd.Stdin = strings.NewReader(`{"hook_event_name":"PreToolUse","tool_name":"Read","tool_input":{}}`)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("chooser: %v stderr=%s", err, stderr.String())
	}
}

func pythonQuote(s string) string { return "r'" + strings.ReplaceAll(s, "'", "\\'") + "'" }

func shQuote(s string) string {
	return "'" + strings.ReplaceAll(filepath.ToSlash(s), "'", `'\''`) + "'"
}
