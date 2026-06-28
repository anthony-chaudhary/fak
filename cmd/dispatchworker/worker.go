// worker.go — the backend-select + launch core of the DOS dispatch worker, a Go
// port of tools/dispatch_worker.py.
//
// This is the indirection seam that lets a fleet run a MIXED worker fleet — some
// Claude workers, some opencode workers — behind one launcher. The supervisor
// (`dos loop --enact`, or the watchdog canary) spawns this; it picks the backend
// and execs the real worker (`claude -p` / `opencode run`). As a compiled binary
// it removes the Python interpreter the old `python tools/dispatch_worker.py`
// launch token spawned — and, being interpreter-free, it can't ENOENT on a
// python3-only node the way the bare `python` token did (the #22 residual).
//
// The pure functions (resolveBackend / buildCommand / childEnv / normalizeTimeout)
// mirror the Python so the ported test table is a parity witness; only launch()
// touches the OS (it execs a compiled worker, which the request-path exec ban
// does not cover — this is off-path dispatch tooling under cmd/).
package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"
)

const (
	workerSchema   = "fleet-dispatch-worker/1"
	defaultBackend = "claude"
	// defaultTimeoutS bounds an unattended worker session. A dispatch worker is a
	// full agentic `claude -p` / `opencode run` session that runs UNATTENDED, so an
	// unbounded run lets a wedged session burn tokens with nothing to stop it. 30
	// min is generous for a real lane yet bounds a runaway; opt out with 0.
	defaultTimeoutS = 1800

	// Invoke the BARE project-skill form (`/dos-dispatch-loop`), not the namespaced
	// plugin form (`/dos-kernel:dos-dispatch-loop`). The skill is git-tracked at
	// `.claude/skills/dos-dispatch-loop/SKILL.md`, so a worker launched from the
	// repo root sees it under EVERY switched account dir. The plugin form fails
	// closed ("Unknown command") whenever a per-account `.claude-<acct>` plugin
	// cache is missing/empty — which it is for freshly-enrolled worker accounts —
	// making the spawned worker exit 0 with zero work done. This mirrors
	// dispatch_worker.CLAUDE_AGENT_PROMPT, which was already fixed to the bare form.
	claudeAgentPrompt = "/dos-dispatch-loop --lane %s"
	opencodeAgent     = "dos-dispatch"
	opencodeMessage   = "dispatch lane %s"
)

var backends = []string{"claude", "opencode"}

func isBackend(b string) bool {
	for _, x := range backends {
		if x == b {
			return true
		}
	}
	return false
}

// resolveBackend picks the backend. Precedence: explicit flag > env > default.
// Mirrors dispatch_worker.resolve_backend, including the Python truthiness of the
// env map: an EMPTY map falls through to the process environment (Python's
// `env or os.environ`), a non-empty map is consulted directly.
func resolveBackend(explicit string, env map[string]string) (string, error) {
	var backend string
	switch {
	case explicit != "":
		backend = explicit
	case len(env) > 0:
		if v, ok := env["FLEET_WORKER_BACKEND"]; ok {
			backend = v
		} else {
			backend = defaultBackend
		}
	default:
		if v, ok := os.LookupEnv("FLEET_WORKER_BACKEND"); ok {
			backend = v
		} else {
			backend = defaultBackend
		}
	}
	backend = strings.ToLower(strings.TrimSpace(backend))
	if !isBackend(backend) {
		return "", fmt.Errorf("unknown backend %q; expected one of %v (via --backend or FLEET_WORKER_BACKEND)", backend, backends)
	}
	return backend, nil
}

// buildCommand is the pure logical argv for one worker launch (no path resolution).
// Mirrors dispatch_worker.build_command.
func buildCommand(lane, backend string) ([]string, error) {
	if lane == "" {
		return nil, fmt.Errorf("lane must be a non-empty string")
	}
	switch backend {
	case "claude":
		return []string{"claude", "-p", "--permission-mode", "bypassPermissions", fmt.Sprintf(claudeAgentPrompt, lane)}, nil
	case "opencode":
		return []string{"opencode", "run", "--dangerously-skip-permissions", "--agent", opencodeAgent, fmt.Sprintf(opencodeMessage, lane)}, nil
	}
	return nil, fmt.Errorf("unknown backend %q; expected one of %v", backend, backends)
}

// childEnv is the env the child worker runs under. DISPATCH_WORKSPACE/LANE/BACKEND
// are the self-describing contract a worker reads to know its assignment
// independent of prompt rendering. Mirrors dispatch_worker.child_env: base (or the
// process env) is passed through, then the three keys are stamped.
func childEnv(lane, backend, workspace string, base map[string]string) map[string]string {
	env := map[string]string{}
	if base != nil {
		for k, v := range base {
			env[k] = v
		}
	} else {
		for _, kv := range os.Environ() {
			if i := strings.IndexByte(kv, '='); i >= 0 {
				env[kv[:i]] = kv[i+1:]
			}
		}
	}
	env["DISPATCH_WORKSPACE"] = workspace
	env["DISPATCH_LANE"] = lane
	env["DISPATCH_BACKEND"] = backend
	return env
}

// normalizeTimeout maps a CLI --timeout-s value to the launch timeout. A positive
// value is the wall-clock cap; 0/negative is the explicit unbounded opt-out (the
// bool is false). Mirrors dispatch_worker.normalize_timeout.
func normalizeTimeout(value int) (time.Duration, bool) {
	if value > 0 {
		return time.Duration(value) * time.Second, true
	}
	return 0, false
}

// resolveExe resolves a backend shim to a launchable path. On Windows the npm
// shims are claude.cmd / opencode.cmd; exec.LookPath finds them via PATHEXT so we
// exec without a shell (which would mangle the prompt argument). Falls back to the
// bare name (launch then surfaces the not-found as returncode 127), matching
// dispatch_worker.resolve_exe.
func resolveExe(name string) string {
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	return name
}

type launchResult struct {
	ReturnCode int    `json:"returncode"`
	Timeout    bool   `json:"timeout,omitempty"`
	Stdout     string `json:"stdout"`
	Stderr     string `json:"stderr"`
	Error      string `json:"error,omitempty"`
}

// runnerFunc is injectable for hermetic tests (the real launcher execs).
type runnerFunc func(command []string, cwd string, env map[string]string) launchResult

// launch execs a worker command. runner is injectable for tests. The real
// launcher resolves the backend shim to a full path (so a Windows .cmd shim execs
// without a shell) and streams stdio to the parent so the supervisor sees worker
// output inline. Mirrors dispatch_worker.launch: a missing exe -> 127, a timeout
// -> 124. No-timeout (bounded=false) runs unbounded.
func launch(command []string, cwd string, env map[string]string, runner runnerFunc, timeout time.Duration, bounded bool) launchResult {
	if runner != nil {
		return runner(command, cwd, env)
	}
	resolved := append([]string(nil), command...)
	if len(resolved) > 0 {
		resolved[0] = resolveExe(resolved[0])
	}

	ctx := context.Background()
	cancel := func() {}
	if bounded {
		ctx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	cmd := exec.CommandContext(ctx, resolved[0], resolved[1:]...)
	cmd.Dir = cwd
	cmd.Env = envSlice(env)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if bounded && ctx.Err() == context.DeadlineExceeded {
		return launchResult{ReturnCode: 124, Timeout: true, Stderr: "timeout"}
	}
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			// could not start (not found / not executable)
			return launchResult{ReturnCode: 127, Error: err.Error(), Stderr: err.Error()}
		}
		return launchResult{ReturnCode: cmd.ProcessState.ExitCode()}
	}
	return launchResult{ReturnCode: 0}
}

func envSlice(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	sort.Strings(out) // deterministic order
	return out
}

type payload struct {
	Schema    string            `json:"schema"`
	OK        bool              `json:"ok"`
	Lane      string            `json:"lane"`
	Backend   string            `json:"backend"`
	Workspace string            `json:"workspace"`
	DryRun    bool              `json:"dry_run"`
	Command   []string          `json:"command"`
	Env       map[string]string `json:"env"`
	Result    *launchResult     `json:"result"`
	Error     string            `json:"error,omitempty"`
}

// buildPayload mirrors dispatch_worker.build_payload: command is the argv (empty on
// an error), ok is true iff there was no error and (no result yet or returncode 0).
func buildPayload(lane, backend, workspace string, dryRun bool, result *launchResult, errMsg string) payload {
	var command []string
	if errMsg == "" {
		command, _ = buildCommand(lane, backend)
	}
	ok := errMsg == "" && (result == nil || result.ReturnCode == 0)
	return payload{
		Schema:    workerSchema,
		OK:        ok,
		Lane:      lane,
		Backend:   backend,
		Workspace: workspace,
		DryRun:    dryRun,
		Command:   command,
		Env: map[string]string{
			"DISPATCH_WORKSPACE": workspace,
			"DISPATCH_LANE":      lane,
			"DISPATCH_BACKEND":   backend,
		},
		Result: result,
		Error:  errMsg,
	}
}

func render(p payload) string {
	cmd := "-"
	if len(p.Command) > 0 {
		cmd = strings.Join(p.Command, " ")
	}
	lines := []string{
		fmt.Sprintf("dispatch-worker: backend=%s lane=%s dry_run=%v", p.Backend, p.Lane, p.DryRun),
		"command: " + cmd,
	}
	if p.Error != "" {
		lines = append(lines, "error: "+p.Error)
	}
	if p.Result != nil {
		lines = append(lines, fmt.Sprintf("returncode: %d", p.Result.ReturnCode))
	}
	return strings.Join(lines, "\n")
}
