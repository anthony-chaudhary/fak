package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// cmd/fak/hooks_agent.go — `fak hooks agent <pretool|posttool|stop>`: run ONE agent-lifecycle
// delegate from a compiled verb instead of a `python -c` wrapper, and make a delegate that COULD
// NOT RUN say so (#5607).
//
// The wiring this replaces is, verbatim from .claude/settings.json:
//
//	python -c "...; subprocess.call([sys.executable, .../dos_hook.py, 'pretool', ...]); sys.exit(0)"
//
// subprocess.call RETURNS the child's exit code; the value is bound to nothing and dropped, then
// sys.exit(0) overrides it unconditionally. So a delegate whose script was deleted, whose
// interpreter crashed, or which refused outright reports the same success as one that ran every
// check and found nothing. A hook whose absence renders identically to a hook that passed is not
// a hook (epic #5601); the audit already carries this as failclosed-audit.md FINDING 2.
//
// EXIT-CODE CONTRACT — deliberately NOT the one `fak hooks pre-commit` uses:
//
//	0 = the delegate ran and allowed
//	2 = the delegate deliberately BLOCKED — reserved by the harness protocol, forwarded verbatim
//	1 = could-not-run, or the delegate ran and failed
//
// Why could-not-run is 1 here and 2 there: for a git commit hook, nonzero means "refuse the
// commit", so 2 is free to mean could-not-run and tell the shell wrapper to fall back to Python.
// For an agent-lifecycle hook the harness reads exit 2 as the BLOCK signal — it refuses the tool
// call and feeds stderr back to the model. Reusing 2 for could-not-run would turn one missing
// script into a fleet-wide refusal of every tool call, which is precisely the failure the Python
// wrappers coerce to 0 to avoid. Exit 1 is the harness's NON-blocking error: surfaced to the
// operator, wedges nothing. That is where a could-not-run belongs — visible, but not a weapon.
//
// One invocation runs exactly ONE delegate. PreToolUse delegates answer on stdout with a JSON
// decision payload, and two payloads concatenated on one stream is not parseable by anything —
// so an event carrying more than one delegate REFUSES without --delegate rather than guessing.
// Same rule as `fak hygiene --gates` (#5604): a selection that names no single gate refuses; it
// never silently runs nothing and reports success.

// agentHookEvents is the closed set of lifecycle events this verb serves, in the order the
// refusal message lists them.
var agentHookEvents = []string{"pretool", "posttool", "stop"}

// agentHookDelegate is one child process registered against one lifecycle event — the compiled
// stand-in for a single `hooks` entry in .claude/settings.json.
type agentHookDelegate struct {
	Name  string
	Event string
	// Argv resolves the child command against a repo root. ok=false means this delegate's
	// evidence is not on this box at all (the script or binary is missing) — a could-not-run,
	// which is a REPORTED outcome here, never a silent pass.
	Argv func(root string) (argv []string, ok bool)
}

// agentHookRegistry mirrors the live .claude/settings.json wiring one delegate per hooks entry,
// so cutting an entry over to this verb is a substitution and not a redesign. The mirror is
// pinned by a test that reads settings.json, so drift in either direction fails the build.
func agentHookRegistry() []agentHookDelegate {
	dosHook := func(event string) func(string) ([]string, bool) {
		return func(root string) ([]string, bool) {
			script := filepath.Join(root, "tools", "dos_hook.py")
			if !fileExistsAt(script) {
				return nil, false
			}
			return []string{pythonExe(), script, event, "--workspace", root}, true
		}
	}
	return []agentHookDelegate{
		{Name: "repoguard", Event: "pretool", Argv: repoguardArgv},
		{Name: "dos-hook", Event: "pretool", Argv: dosHook("pretool")},
		{Name: "dos-hook", Event: "posttool", Argv: dosHook("posttool")},
		{Name: "dos-hook", Event: "stop", Argv: dosHook("stop")},
	}
}

// repoguardArgv prefers the compiled repoguard, falling back to its Python source — the same
// preference the settings.json one-liner encodes. The staleness check that wrapper performs is
// deliberately NOT reproduced: it blanked the binary and fell through to a source path it never
// confirmed existed, so a stale binary plus a missing script silently ran nothing. Here a
// resolved path that does not exist is reported as could-not-run.
func repoguardArgv(root string) ([]string, bool) {
	bin := filepath.Join(root, "tools", ".bin", "repoguard.exe")
	if !fileExistsAt(bin) {
		bin = filepath.Join(root, "tools", ".bin", "repoguard")
	}
	if fileExistsAt(bin) {
		return []string{bin, "--hook"}, true
	}
	if src := filepath.Join(root, "tools", "repo_guard.py"); fileExistsAt(src) {
		return []string{pythonExe(), src, "--hook"}, true
	}
	return nil, false
}

func fileExistsAt(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func pythonExe() string {
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
	}
	return "python"
}

// agentHookOutcome classifies one delegate run onto the harness's exit vocabulary. It is the
// whole point of the verb, kept pure so the contract is table-testable without spawning anything.
//
// ran=false is the case the issue reports: the delegate never executed. It must NOT be 0.
func agentHookOutcome(ran bool, rc int) (status string, exit int) {
	if !ran {
		return "could-not-run", 1
	}
	switch rc {
	case 0:
		return "ran", 0
	case 2:
		// The delegate's own deliberate refusal. Forward it untouched: this is the one code
		// the harness acts on, and re-deriving it would let the wrapper overrule the gate.
		return "blocked", 2
	default:
		return "failed", 1
	}
}

// agentHookPick resolves (event, name) against the registry. An unknown event, an unknown
// delegate name, or an ambiguous event named without --delegate all REFUSE with the valid set —
// the alternative is selecting nothing, running nothing, and exiting 0, which is the defect.
func agentHookPick(event, name string) (agentHookDelegate, error) {
	// Match case-insensitively but echo the operator's OWN spelling back in any refusal: they
	// have to find it in settings.json, and a normalized echo sends them looking for a string
	// that is not there.
	raw := strings.TrimSpace(event)
	event = strings.ToLower(raw)
	if event == "" {
		return agentHookDelegate{}, fmt.Errorf("event required (one of: %s)", strings.Join(agentHookEvents, ", "))
	}
	known := false
	for _, e := range agentHookEvents {
		if e == event {
			known = true
			break
		}
	}
	if !known {
		return agentHookDelegate{}, fmt.Errorf("unknown event %q (valid: %s)", raw, strings.Join(agentHookEvents, ", "))
	}

	var forEvent []agentHookDelegate
	var names []string
	for _, d := range agentHookRegistry() {
		if d.Event == event {
			forEvent = append(forEvent, d)
			names = append(names, d.Name)
		}
	}
	sort.Strings(names)
	if len(forEvent) == 0 {
		return agentHookDelegate{}, fmt.Errorf("event %q has no registered delegate", event)
	}

	name = strings.TrimSpace(name)
	if name == "" {
		if len(forEvent) > 1 {
			return agentHookDelegate{}, fmt.Errorf("event %q has %d delegates (%s) — name one with --delegate; "+
				"running both would concatenate two decision payloads on one stdout and neither would parse",
				event, len(forEvent), strings.Join(names, ", "))
		}
		return forEvent[0], nil
	}
	for _, d := range forEvent {
		if strings.EqualFold(d.Name, name) {
			return d, nil
		}
	}
	return agentHookDelegate{}, fmt.Errorf("unknown delegate %q for event %s (valid: %s)", name, event, strings.Join(names, ", "))
}

// runHooksAgent runs one lifecycle delegate and reports what actually happened.
//
// The harness envelope arrives on stdin and is forwarded verbatim to the child; the child's
// stdout is forwarded verbatim back, because for PreToolUse that stream carries the JSON
// permission decision and any rewriting here would be this wrapper overruling the gate.
func runHooksAgent(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("hooks agent", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	delegate := fs.String("delegate", "", "which registered delegate to run (required when the event has more than one)")
	asJSON := fs.Bool("json", false, "emit the run ledger as JSON")

	var event string
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		event, argv = argv[0], argv[1:]
	}
	if !parseFlags(fs, argv) {
		return 1
	}

	d, err := agentHookPick(event, *delegate)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks agent: %v\n", err)
		return 1
	}

	r := resolveRoot(*root)
	if r == "" {
		// No root means no delegate can be located. That is a could-not-run, and saying so is
		// the entire contract — it must not read as a clean pass.
		return agentHookReport(stdout, stderr, *asJSON, d, -1, "could-not-run",
			"repo root could not be resolved", 1)
	}

	child, ok := d.Argv(r)
	if !ok {
		detail := "delegate command is not present under " + r
		appendAgentHookIncident(r, d, -1, "could-not-run", detail)
		return agentHookReport(stdout, stderr, *asJSON, d, -1, "could-not-run", detail, 1)
	}

	var payload []byte
	if stdin != nil {
		payload, _ = io.ReadAll(stdin)
	}

	cmd := exec.Command(child[0], child[1:]...)
	cmd.Dir = r
	cmd.Stdin = bytes.NewReader(payload)
	cmd.Stdout = stdout // the decision payload, forwarded untouched
	cmd.Stderr = stderr
	runErr := cmd.Run()

	rc := 0
	ran := true
	detail := ""
	if runErr != nil {
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			rc = ee.ExitCode()
		} else {
			// Never started: bad interpreter, permission denied, vanished between stat and
			// spawn. The delegate did not run, so it did not pass.
			ran, rc, detail = false, -1, runErr.Error()
		}
	}
	status, exit := agentHookOutcome(ran, rc)
	if status != "ran" {
		appendAgentHookIncident(r, d, rc, status, detail)
	}
	return agentHookReport(stdout, stderr, *asJSON, d, rc, status, detail, exit)
}

const agentHookIncidentSchema = "fak.hooks-agent-incident/v1"

type agentHookIncident struct {
	Schema   string `json:"schema"`
	Ts       string `json:"ts"`
	Event    string `json:"event"`
	Delegate string `json:"delegate"`
	Status   string `json:"status"`
	ExitCode int    `json:"exit_code"`
	Detail   string `json:"detail,omitempty"`
}

// appendAgentHookIncident preserves lifecycle transport failures after transient
// hook stderr disappears. It is fail-open and records only non-clean outcomes,
// keeping the every-tool-call path quiet and small.
func appendAgentHookIncident(root string, d agentHookDelegate, rc int, status, detail string) {
	row, err := json.Marshal(agentHookIncident{
		Schema: agentHookIncidentSchema, Ts: time.Now().UTC().Format(time.RFC3339Nano),
		Event: d.Event, Delegate: d.Name, Status: status, ExitCode: rc, Detail: detail,
	})
	if err != nil {
		return
	}
	path := filepath.Join(root, ".fak", "hooks-agent-incidents.jsonl")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(row, '\n'))
}

// agentHookReport writes the run ledger. The skip keys mirror `fak hooks pre-commit --json` and
// `fak hygiene --json` (#5299, #5604) so one consumer parses all three, and they are ALWAYS
// present — a consumer must never have to read "absent" as "none".
func agentHookReport(stdout, stderr io.Writer, asJSON bool, d agentHookDelegate, rc int, status, detail string, exit int) int {
	skipped := []string{}
	if status != "ran" {
		skipped = append(skipped, d.Name)
	}
	if asJSON {
		if err := writeIndentedJSON(stdout, map[string]any{
			"event":         d.Event,
			"delegate":      d.Name,
			"status":        status,
			"exit_code":     rc,
			"detail":        detail,
			"skipped":       skipped,
			"skipped_count": len(skipped),
		}); err != nil {
			fmt.Fprintf(stderr, "fak hooks agent: %v\n", err)
			return 1
		}
		return exit
	}
	switch status {
	case "ran":
		// Deliberately silent on the happy path: a lifecycle hook prints on every tool call.
	case "blocked":
		fmt.Fprintf(stderr, "hooks agent: %s/%s BLOCKED (exit 2) — forwarding the refusal\n", d.Event, d.Name)
	default:
		msg := fmt.Sprintf("hooks agent: %s/%s %s", d.Event, d.Name, status)
		if detail != "" {
			msg += " (" + detail + ")"
		}
		fmt.Fprintf(stderr, "%s — this lifecycle event was NOT checked; exit %d is non-blocking so nothing is wedged, "+
			"but do not read this run as a pass (#5607)\n", msg, exit)
	}
	return exit
}
