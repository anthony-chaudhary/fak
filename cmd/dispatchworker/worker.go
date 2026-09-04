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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

const (
	workerSchema   = "fleet-dispatch-worker/1"
	defaultBackend = "claude"
	// defaultTimeoutS bounds an unattended worker session. A dispatch worker is a
	// full agentic `claude -p` / `opencode run` session that runs UNATTENDED, so an
	// unbounded run lets a wedged session burn tokens with nothing to stop it. 30
	// min is generous for a real lane yet bounds a runaway; opt out with 0.
	defaultTimeoutS = 1800

	// launchWaitDelay is the portable backstop on top of the process-tree kill
	// (configureProcTree): after the deadline fires and the group/tree is signalled,
	// Go waits this long for a straggler to drain, then force-closes the inherited
	// pipes so launch is guaranteed to return rather than hang on a lingering
	// grandchild. Mirrors internal/nightrun's 10s WaitDelay.
	launchWaitDelay = 10 * time.Second

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
		// --print-logs surfaces opencode run-level failures in unattended logs;
		// otherwise a GLM quota wall can look like a banner-only no-op (#1275).
		return []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions", "--agent", opencodeAgent, fmt.Sprintf(opencodeMessage, lane)}, nil
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
	return launchRegistered(command, cwd, env, runner, timeout, bounded, nil)
}

type launchRegistration struct {
	Store  sessionregistry.Store
	Record sessionregistry.Record
}

func launchRegistered(command []string, cwd string, env map[string]string, runner runnerFunc, timeout time.Duration, bounded bool, registration *launchRegistration) launchResult {
	if runner != nil {
		if registration != nil {
			if err := registration.Store.Register(registration.Record); err != nil {
				return launchResult{ReturnCode: 2, Error: "child registration persist failed (worker not started): " + err.Error()}
			}
			env = registeredChildEnv(env, registration.Record)
		}
		result := runner(command, cwd, env)
		updateTerminalRegistration(registration, &result, env["FAK_WITNESS_REF"])
		return result
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
	cmd := newLaunchCmd(ctx, resolved, cwd, env)
	if registration != nil {
		if err := registration.Store.Register(registration.Record); err != nil {
			return launchResult{ReturnCode: 2, Error: "child registration persist failed (worker not started): " + err.Error()}
		}
		cmd.Env = procguard.EnvSlice(registeredChildEnv(env, registration.Record))
	}
	started := time.Now().UTC()
	err := cmd.Start()
	if err == nil && registration != nil {
		if _, regErr := registration.Store.Start(registration.Record.RegistrationID, cmd.Process.Pid, started); regErr != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			_, _ = registration.Store.Terminal(registration.Record.RegistrationID, sessionregistry.StateFailed, "start_readback_failed", "", time.Now().UTC())
			return launchResult{ReturnCode: 2, Error: "child start read-back failed; worker terminated: " + regErr.Error()}
		}
	}
	if err == nil {
		err = cmd.Wait()
	}
	result := launchResult{}
	if bounded && ctx.Err() == context.DeadlineExceeded {
		result = launchResult{ReturnCode: 124, Timeout: true, Stderr: "timeout"}
	} else if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			result = launchResult{ReturnCode: 127, Error: err.Error(), Stderr: err.Error()}
		} else {
			result = launchResult{ReturnCode: cmd.ProcessState.ExitCode()}
		}
	}
	updateTerminalRegistration(registration, &result, env["FAK_WITNESS_REF"])
	return result
}

func registeredChildEnv(base map[string]string, r sessionregistry.Record) map[string]string {
	out := map[string]string{}
	for k, v := range base {
		out[k] = v
	}
	out["FAK_REGISTRATION_ID"] = r.RegistrationID
	out["FAK_ATTEMPT_ID"] = r.AttemptID
	out["FAK_PARENT_REGISTRATION_ID"] = r.ParentRegistrationID
	out["FAK_PARENT_ATTEMPT_ID"] = r.ParentAttemptID
	out["FAK_ROOT_REGISTRATION_ID"] = r.RootRegistrationID
	out["FAK_ROOT_OUTCOME"] = r.RootOutcome
	out["FAK_ROOT_ISSUE"] = r.RootIssue
	out["FAK_TASK_ID"] = r.TaskID
	return out
}

// newLaunchCmd builds the *exec.Cmd for one real worker launch with the
// process-tree kill and WaitDelay backstop wired in. Splitting it out of launch
// keeps the OS-touching construction in one place AND lets a hermetic test witness
// that the tree-kill hook (configureProcTree) and the WaitDelay are actually set,
// without spawning a process. resolved[0] is the already-PATH-resolved exe; stdio
// is streamed to the parent so the supervisor sees worker output inline.
func newLaunchCmd(ctx context.Context, resolved []string, cwd string, env map[string]string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, resolved[0], resolved[1:]...)
	cmd.Dir = cwd
	cmd.Env = procguard.EnvSlice(env)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Assertive runaway containment: on the bounded path the deadline cancels ctx,
	// and configureProcTree's Cancel reaps the worker's ENTIRE descendant tree (not
	// just the backend shim) so a wedged session can't leave grandchildren holding
	// the box / commit lock / GPU lease and block the rest of the fleet. WaitDelay is
	// the portable backstop that force-closes the inherited pipes if a straggler
	// lingers. Both are harmless on the unbounded path (ctx never cancels).
	configureProcTree(cmd)
	cmd.WaitDelay = launchWaitDelay
	return cmd
}

type payload struct {
	Schema           string            `json:"schema"`
	OK               bool              `json:"ok"`
	Lane             string            `json:"lane"`
	Backend          string            `json:"backend"`
	Guarded          bool              `json:"guarded"`
	GuardAuditPruned int               `json:"guard_audit_pruned"`
	Workspace        string            `json:"workspace"`
	DryRun           bool              `json:"dry_run"`
	Command          []string          `json:"command"`
	Env              map[string]string `json:"env"`
	Result           *launchResult     `json:"result"`
	Error            string            `json:"error,omitempty"`
	// GuardBaselineTokens/GuardContextBudgetTokens are the OBSERVABLE for the measured
	// launch-prompt baseline and the context budget the claude guard was seeded with
	// (guard.go). Emitted only for the claude backend, so fleet drift in the launch
	// prompt is a visible number in the launch record rather than an argv int.
	GuardBaselineTokens      int `json:"guard_baseline_tokens,omitempty"`
	GuardContextBudgetTokens int `json:"guard_context_budget_tokens,omitempty"`
}

// buildPayload mirrors dispatch_worker.build_payload. command defaults to the raw
// (unguarded) worker argv when nil (backward compat); a live/dry-run launch passes
// the ACTUAL launched argv (kernel-fronted when guarded) so the record shows exactly
// what ran. ok is true iff there was no error and (no result yet or returncode 0).
func buildPayload(lane, backend, workspace string, dryRun bool, result *launchResult, errMsg string, command []string, guarded bool) payload {
	if command == nil && errMsg == "" {
		command, _ = buildCommand(lane, backend)
	}
	ok := errMsg == "" && (result == nil || result.ReturnCode == 0)
	baselineTokens, budgetTokens := 0, 0
	if backend == "claude" {
		baselineTokens, budgetTokens = claudeGuardBudgetObservable(workspace, workerModelFromCommand(command), nil)
	}
	return payload{
		Schema:    workerSchema,
		OK:        ok,
		Lane:      lane,
		Backend:   backend,
		Guarded:   guarded,
		Workspace: workspace,
		DryRun:    dryRun,
		Command:   command,
		Env: map[string]string{
			"DISPATCH_WORKSPACE":           workspace,
			"DISPATCH_LANE":                lane,
			"DISPATCH_BACKEND":             backend,
			"DISPATCH_GOAL":                envOr("DISPATCH_GOAL", lane),
			"DISPATCH_ACCOUNT":             os.Getenv("DISPATCH_ACCOUNT"),
			"DISPATCH_POOL":                os.Getenv("DISPATCH_POOL"),
			"DISPATCH_LEASE":               os.Getenv("DISPATCH_LEASE"),
			"DISPATCH_WITNESS_REQUIREMENT": os.Getenv("DISPATCH_WITNESS_REQUIREMENT"),
		},
		Result:                   result,
		Error:                    errMsg,
		GuardBaselineTokens:      baselineTokens,
		GuardContextBudgetTokens: budgetTokens,
	}
}

func render(p payload) string {
	cmd := "-"
	if len(p.Command) > 0 {
		cmd = strings.Join(p.Command, " ")
	}
	lines := []string{
		fmt.Sprintf("dispatch-worker: backend=%s lane=%s guarded=%v dry_run=%v", p.Backend, p.Lane, p.Guarded, p.DryRun),
		"command: " + cmd,
	}
	if p.Error != "" {
		lines = append(lines, "error: "+p.Error)
	}
	if p.GuardContextBudgetTokens > 0 {
		lines = append(lines, fmt.Sprintf("guard: measured_baseline=%d context_budget=%d tokens", p.GuardBaselineTokens, p.GuardContextBudgetTokens))
	}
	if p.Result != nil {
		lines = append(lines, fmt.Sprintf("returncode: %d", p.Result.ReturnCode))
	}
	return strings.Join(lines, "\n")
}

func workerModelFromCommand(command []string) string {
	for i, arg := range command {
		if (arg == "--model" || arg == "-m") && i+1 < len(command) {
			return command[i+1]
		}
		if strings.HasPrefix(arg, "--model=") {
			return strings.TrimPrefix(arg, "--model=")
		}
	}
	return ""
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func updateTerminalRegistration(reg *launchRegistration, res *launchResult, witnessRef string) {
	if reg == nil {
		return
	}
	state, reason := sessionregistry.StateCompleted, ""
	if res.ReturnCode != 0 {
		state, reason = sessionregistry.StateFailed, fmt.Sprintf("worker_exit_%d", res.ReturnCode)
	}
	if _, err := reg.Store.Terminal(reg.Record.RegistrationID, state, reason, witnessRef, time.Now().UTC()); err != nil {
		res.ReturnCode = 2
		res.Error = "terminal registration update failed: " + err.Error()
	}
}
