// Command dispatchworker launches one DOS dispatch worker on a selected backend —
// the Go port of tools/dispatch_worker.py, compiled to a single binary so the
// supervisor (`dos loop --enact`, or the watchdog canary) spawns a worker WITHOUT
// a Python interpreter (and without the bare-`python` token that ENOENTs on a
// python3-only node — the #22 residual).
//
//	dispatchworker --lane <lane>            # launch one worker on the lane
//	dispatchworker --lane <lane> --dry-run  # print the argv instead of launching
//	dispatchworker --lane <lane> --json     # machine-readable payload
//
// Backend precedence: --backend > FLEET_WORKER_BACKEND > claude.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	lane := flag.String("lane", "", "lane to dispatch on (required)")
	backendFlag := flag.String("backend", "", "worker backend (claude|opencode; default: env FLEET_WORKER_BACKEND or claude)")
	workspaceFlag := flag.String("workspace", "", "workspace root (default: repo root above cwd)")
	dryRun := flag.Bool("dry-run", false, "print the command instead of launching")
	timeoutS := flag.Int("timeout-s", defaultTimeoutS, fmt.Sprintf("child wall-clock timeout in seconds (default: %d; 0 = unbounded)", defaultTimeoutS))
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	flag.Parse()

	if *lane == "" {
		fmt.Fprintln(os.Stderr, "dispatchworker: --lane is required")
		os.Exit(2)
	}

	workspace := resolveWorkspace(*workspaceFlag)
	backend := defaultBackend
	errMsg := ""
	if b, err := resolveBackend(*backendFlag, nil); err != nil {
		errMsg = err.Error()
	} else {
		backend = b
	}

	// Resolve the argv to actually launch, fronting it with `fak guard` when dogfood
	// mode is on and a fak binary resolves (fail OPEN to an unwrapped worker otherwise).
	// Computed for BOTH paths so --dry-run reveals the kernel-fronted argv an operator
	// will actually run.
	var command []string
	guarded := false
	if errMsg == "" {
		raw, _ := buildCommand(*lane, backend)
		command, guarded = guardedLaunchCommand(raw, *lane, backend, workspace, nil)
	}

	if *dryRun || errMsg != "" {
		emit(buildPayload(*lane, backend, workspace, true, nil, errMsg, command, guarded), *asJSON)
		if errMsg != "" {
			os.Exit(2)
		}
		os.Exit(0)
	}

	env := childEnv(*lane, backend, workspace, nil)
	if guarded {
		guardEnvAugment(env)
	}
	timeout, bounded := normalizeTimeout(*timeoutS)
	result := launch(command, workspace, env, nil, timeout, bounded)
	emit(buildPayload(*lane, backend, workspace, false, &result, "", command, guarded), *asJSON)
	os.Exit(result.ReturnCode)
}

func emit(p payload, asJSON bool) {
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		_ = enc.Encode(p)
		return
	}
	fmt.Println(render(p))
}

// resolveWorkspace mirrors dispatch_worker's default: an explicit --workspace is
// made absolute; otherwise fall back to the repo root above cwd (the supervisor
// runs the worker from the workspace).
func resolveWorkspace(flagVal string) string {
	if flagVal != "" {
		if abs, err := filepath.Abs(flagVal); err == nil {
			return abs
		}
		return flagVal
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return repoRoot(cwd)
}

func repoRoot(start string) string {
	dir := start
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return start
		}
		dir = parent
	}
}
