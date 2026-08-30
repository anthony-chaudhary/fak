package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const guardCrashRSIMarkerEnv = "FAK_GUARD_CRASH_RSI"

type guardCrashRSIRequest struct {
	Tag       string
	Source    string
	Agent     string
	Class     string
	ExitCode  int
	Workspace string
	Prompt    string
}

var guardCrashRSILaunch = launchGuardCrashRSI

// guardMaybeLaunchCrashRSI starts one bounded, independent investigation on the first
// restartable generic crash. It is deliberately fail-open: the original in-place restart
// remains the authority even when admission or launch fails.
func guardMaybeLaunchCrashRSI(stderr io.Writer, guardTraceID, agentName, class string, code, restartsSoFar int) bool {
	req, ok := guardCrashRSIAdmission(guardTraceID, agentName, class, code, restartsSoFar)
	if !ok {
		return false
	}
	if err := guardCrashRSILaunch(req); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: crash RSI launch skipped (%s): %v\n", req.Tag, err)
		}
		return false
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: spawned crash RSI session %s for original crash %s exit %d\n", req.Tag, req.Class, req.ExitCode)
	}
	return true
}

func guardCrashRSIAdmission(guardTraceID, agentName, class string, code, restartsSoFar int) (guardCrashRSIRequest, bool) {
	if restartsSoFar != 0 || strings.TrimSpace(os.Getenv(guardCrashRSIMarkerEnv)) != "" {
		return guardCrashRSIRequest{}, false
	}
	trace := strings.TrimSpace(guardTraceID)
	class = strings.TrimSpace(class)
	if trace == "" || class == "" || code == 0 {
		return guardCrashRSIRequest{}, false
	}
	agent := guardCrashRSISupportedAgent(agentName)
	if agent == "" {
		return guardCrashRSIRequest{}, false
	}
	workspace, err := os.Getwd()
	if err != nil || !filepath.IsAbs(workspace) {
		return guardCrashRSIRequest{}, false
	}
	sum := sha256.Sum256([]byte(trace))
	source := hex.EncodeToString(sum[:8])
	tag := "guard-crash-rsi/" + source
	req := guardCrashRSIRequest{
		Tag:       tag,
		Source:    source,
		Agent:     agent,
		Class:     class,
		ExitCode:  code,
		Workspace: workspace,
	}
	req.Prompt = guardCrashRSIPrompt(req)
	return req, true
}

func guardCrashRSISupportedAgent(agentName string) string {
	name := strings.ToLower(strings.TrimSpace(filepath.Base(agentName)))
	name = strings.TrimSuffix(name, ".exe")
	switch name {
	case "claude", "claude-code":
		return "claude"
	case "codex":
		return "codex"
	default:
		return ""
	}
}

func guardCrashRSIPrompt(req guardCrashRSIRequest) string {
	return fmt.Sprintf(`You are the specially tagged crash-RSI investigation session %s.
Investigate the root cause of the ORIGINAL fak guard child crash; do not restart or reproduce the crashed session merely to continue its task.
Bounded crash context:
- source_guard: %s
- harness: %s
- crash_class: %s
- exit_code: %d
- workspace: %s
Perform read-only root-cause analysis, identify the smallest durable prevention, and report evidence plus a checkable next step. Do not expose credentials or ambient environment values.`, req.Tag, req.Source, req.Agent, req.Class, req.ExitCode, req.Workspace)
}

func launchGuardCrashRSI(req guardCrashRSIRequest) error {
	var name string
	var args []string
	switch req.Agent {
	case "claude":
		name = "claude"
		args = []string{"-p", req.Prompt, "--permission-mode", "plan"}
	case "codex":
		name = "codex"
		args = []string{"exec", "--sandbox", "read-only", "--json", req.Prompt}
	default:
		return fmt.Errorf("unsupported harness %q", req.Agent)
	}
	path, err := exec.LookPath(name)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", name, err)
	}
	cmd := exec.Command(path, args...)
	cmd.Dir = req.Workspace
	cmd.Env = guardCrashRSIEnvironment(req.Tag)
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { _ = cmd.Wait() }()
	return nil
}

// The investigation receives only process-bootstrap paths and the recursion marker. In
// particular, provider keys, original argv, and the parent's full ambient environment are not
// forwarded.
func guardCrashRSIEnvironment(tag string) []string {
	allow := []string{"PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "CLAUDE_CONFIG_DIR", "CODEX_HOME"}
	env := make([]string, 0, len(allow)+1)
	for _, key := range allow {
		if value, ok := os.LookupEnv(key); ok && strings.TrimSpace(value) != "" {
			env = append(env, key+"="+value)
		}
	}
	return append(env, guardCrashRSIMarkerEnv+"="+tag)
}
