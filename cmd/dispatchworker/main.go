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
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "inspect" {
		os.Exit(runRegistryInspect(os.Stdout, os.Stderr, os.Args[2:]))
	}
	lane := flag.String("lane", "", "lane to dispatch on (required)")
	workerModel := flag.String("worker-model", "", "worker model used for context-envelope selection")
	profileFlag := flag.String("profile", "", "worker execution profile (e.g. reflex)")
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

	resolvedModel := *workerModel
	if *profileFlag != "" {
		resolvedModel, _ = applyProfile(*profileFlag, resolvedModel, nil)
	}

	// Resolve the argv to actually launch, fronting it with `fak guard` when dogfood
	// mode is on and a fak binary resolves (fail OPEN to an unwrapped worker otherwise).
	// Computed for BOTH paths so --dry-run reveals the kernel-fronted argv an operator
	// will actually run.
	var command []string
	guarded := false
	if errMsg == "" {
		raw, _ := buildCommand(*lane, backend)
		if resolvedModel != "" {
			raw = append(raw, "--model", resolvedModel)
		}
		command, guarded = guardedLaunchCommand(raw, *lane, backend, workspace, resolvedModel, nil)
	}

	if *dryRun || errMsg != "" {
		emit(buildPayload(*lane, backend, workspace, true, nil, errMsg, command, guarded, *profileFlag), *asJSON)
		if errMsg != "" {
			os.Exit(2)
		}
		os.Exit(0)
	}

	env := childEnv(*lane, backend, workspace, nil, *profileFlag)
	if guarded {
		guardEnvAugment(env)
	}
	timeout, bounded := normalizeTimeout(*timeoutS)
	guardAuditPruned := pruneGuardAuditTick(workspace, time.Now())
	registration, regErr := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: env["FAK_REGISTRATION_ID"], ParentRegistrationID: env["FAK_PARENT_REGISTRATION_ID"], ParentAttemptID: env["FAK_PARENT_ATTEMPT_ID"], RootRegistrationID: env["FAK_ROOT_REGISTRATION_ID"], RootOutcome: env["FAK_ROOT_OUTCOME"], RootIssue: firstEnv(env, "FAK_ROOT_ISSUE", "DISPATCH_ISSUE"), TaskID: firstEnv(env, "FAK_TASK_ID", "DISPATCH_ISSUE"), GoalID: firstEnv(env, "FAK_GOAL_ID"), AttemptID: env["FAK_ATTEMPT_ID"], ResumeOfAttemptID: env["FAK_RESUME_OF_ATTEMPT_ID"], LaunchKind: "headless_worker", Scope: []string{workspace}, Lane: *lane, LeaseID: env["FAK_LEASE_ID"], Runtime: backend, HostID: env["COMPUTERNAME"],
	})
	if regErr != nil {
		emit(payload{Schema: workerSchema, OK: false, Lane: *lane, Backend: backend, Profile: *profileFlag, Workspace: workspace, Command: command, Error: "child registration refused: " + regErr.Error()}, *asJSON)
		os.Exit(2)
	}
	result := launchRegistered(command, workspace, env, nil, timeout, bounded, &launchRegistration{Store: sessionregistry.Store{Path: registryPath(env)}, Record: registration})
	p := buildPayload(*lane, backend, workspace, false, &result, "", command, guarded, *profileFlag)
	p.GuardAuditPruned = guardAuditPruned
	emit(p, *asJSON)
	os.Exit(result.ReturnCode)
}

func registryPath(env map[string]string) string {
	if p := strings.TrimSpace(env["FAK_SESSION_REGISTRY"]); p != "" {
		return p
	}
	return sessionregistry.DefaultPath()
}
func firstEnv(env map[string]string, names ...string) string {
	for _, n := range names {
		if v := strings.TrimSpace(env[n]); v != "" {
			return v
		}
	}
	return ""
}
func runRegistryInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatchworker inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("registry", sessionregistry.DefaultPath(), "registration JSONL path")
	id := fs.String("registration", "", "registration ID; emits its root/descendant chain")
	jsonOut := fs.Bool("json", false, "emit stable JSON")
	rootIssue := fs.String("root-issue", "", "filter by root issue")
	parent := fs.String("parent", "", "filter by parent registration")
	session := fs.String("session", "", "filter by session ID")
	thread := fs.String("thread", "", "filter by thread ID")
	pid := fs.Int("pid", 0, "filter by PID")
	processStart := fs.String("process-start", "", "filter by process start time (RFC3339)")
	observedPath := fs.String("observed", "", "JSON array of observed process identities to reconcile")
	lane := fs.String("lane", "", "filter by lane")
	lease := fs.String("lease", "", "filter by lease ID")
	witness := fs.String("witness", "", "filter by witness reference")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "dispatchworker inspect: unexpected arguments")
		return 2
	}
	rows, err := (sessionregistry.Store{Path: *path}).ReadAll()
	if err != nil {
		fmt.Fprintf(stderr, "dispatchworker inspect: %v\n", err)
		return 1
	}
	var startAt time.Time
	if strings.TrimSpace(*processStart) != "" {
		startAt, err = time.Parse(time.RFC3339Nano, strings.TrimSpace(*processStart))
		if err != nil {
			fmt.Fprintf(stderr, "dispatchworker inspect: invalid --process-start: %v\n", err)
			return 2
		}
	}
	if strings.TrimSpace(*id) != "" {
		rows = sessionregistry.Chain(rows, strings.TrimSpace(*id))
		if len(rows) == 0 {
			fmt.Fprintf(stderr, "dispatchworker inspect: registration %q not found\n", *id)
			return 1
		}
	}
	if strings.TrimSpace(*id) == "" {
		rows = sessionregistry.Filter(rows, sessionregistry.Query{RootIssue: strings.TrimSpace(*rootIssue), ParentRegistrationID: strings.TrimSpace(*parent), SessionID: strings.TrimSpace(*session), ThreadID: strings.TrimSpace(*thread), PID: *pid, ProcessStartedAt: startAt, Lane: strings.TrimSpace(*lane), LeaseID: strings.TrimSpace(*lease), WitnessRef: strings.TrimSpace(*witness)})
	}
	var unregistered []sessionregistry.UnregisteredObserved
	if strings.TrimSpace(*observedPath) != "" {
		b, readErr := os.ReadFile(*observedPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "dispatchworker inspect: read observed: %v\n", readErr)
			return 1
		}
		var observed []sessionregistry.ObservedProcess
		if jsonErr := json.Unmarshal(b, &observed); jsonErr != nil {
			fmt.Fprintf(stderr, "dispatchworker inspect: decode observed: %v\n", jsonErr)
			return 1
		}
		unregistered = sessionregistry.ReconcileObserved(rows, observed)
	}
	counts := sessionregistry.Summarize(rows, len(unregistered))
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(struct {
			Schema               string                                 `json:"schema"`
			GeneratedAt          time.Time                              `json:"generated_at"`
			Counts               sessionregistry.Counts                 `json:"counts"`
			Records              []sessionregistry.Record               `json:"records"`
			UnregisteredObserved []sessionregistry.UnregisteredObserved `json:"unregistered_observed,omitempty"`
		}{sessionregistry.Schema, time.Now().UTC(), counts, rows, unregistered}); err != nil {
			return 1
		}
		return 0
	}
	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintf(w, "TOTAL\t%d\tREGISTERED\t%d\tACTIVE\t%d\tTERMINAL\t%d\tUNKNOWN\t%d\n", counts.Total, counts.Registered, counts.Active, counts.Terminal, counts.Unknown)
	fmt.Fprintln(w, "REGISTRATION\tPARENT\tROOT\tISSUE\tKIND\tRUNTIME\tPID\tSTATE\tWITNESS")
	for _, r := range rows {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n", dash(r.RegistrationID), dash(r.ParentRegistrationID), dash(r.RootRegistrationID), dash(r.RootIssue), dash(r.LaunchKind), dash(r.Identity.Runtime), r.Identity.PID, r.State, dash(r.WitnessRef))
	}
	for _, u := range unregistered {
		fmt.Fprintf(w, "UNREGISTERED_OBSERVED\t-\t-\t-\tobserved_process\t%s\t%d\t%s\t-\n", dash(u.Process.Runtime), u.Process.PID, u.Reason)
	}
	_ = w.Flush()
	return 0
}
func dash(v string) string {
	if strings.TrimSpace(v) == "" {
		return "-"
	}
	return v
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
