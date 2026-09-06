// cron.go projects the in-kernel loop schedule DOWN to a real OS scheduler unit
// (#765, part of the `fak cron` sub-epic #749). The delegation is deliberate: the
// OS scheduler (launchd / systemd / Windows Task Scheduler) owns wall-clock
// firing; fak owns the SEMANTICS (overlap-lock via the loop ledger, missed-run
// policy). The default action every emitted unit invokes is the already-shipped
// `fak loop run --loop <id> ...`, so the emitted unit is the *delivery* mechanism,
// not a second scheduler fak supervises. `--command 'CMD...'` (#1385) instead emits
// a unit whose action is an ARBITRARY command (e.g. the stale-work garden
// watchdog's `fak garden --check`), with no `fak loop run` wrapper. The operator
// installs the printed unit.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

func cmdCron(argv []string) { os.Exit(runCron(os.Stdout, os.Stderr, argv)) }

func runCron(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		cronUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "emit":
		return runCronEmit(stdout, stderr, argv[1:])
	case "run":
		return runCronRun(stdout, stderr, argv[1:])
	case "fire":
		return runCronFire(stdout, stderr, argv[1:])
	case "audit":
		return runCronAudit(stdout, stderr, argv[1:])
	case "prompt":
		return runCronPrompt(stdout, stderr, argv[1:])
	case "chain":
		return runCronChain(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		cronUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak cron: unknown subcommand %q\n", argv[0])
		cronUsage(stderr)
		return 2
	}
}

// cronSources maps a --target to the trigger-source token the emitted unit hands
// `fak loop run --source` when it fires, so the ledger records which OS scheduler
// fired it. The keys are the accepted --target values.
var cronSources = map[string]string{
	"launchd":        "launchd",
	"systemd":        "systemd",
	"taskscheduler":  "task-scheduler",
	"tasksched":      "task-scheduler",
	"task-scheduler": "task-scheduler",
}

func runCronEmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron emit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	target := fs.String("target", "", "OS scheduler to emit for: launchd|systemd|taskscheduler")
	format := fs.String("format", "", "alias for --target (launchd|systemd|taskscheduler|tasksched)")
	loopID := fs.String("loop", "", "loop id this unit fires (may be given positionally)")
	command := fs.String("command", "", "emit a unit running this ARBITRARY command instead of `fak loop run` (e.g. 'fak garden --check')")
	interval := fs.Duration("interval", time.Hour, "firing cadence the OS scheduler enforces (e.g. 5m, 1h)")
	fakBin := fs.String("fak-bin", "fak", "path to the fak binary the unit invokes")
	label := fs.String("label", "", "unit/task name (default fak-loop-<id>)")
	ledger := fs.String("ledger", "", "loop ledger path passed through to fak loop run")
	runner := fs.Bool("runner", false, "emit a unit invoking `fak cron run` for bounded scheduled tasks")
	job := fs.String("job", "", "job id for the bounded runner (required with --runner)")
	timeout := fs.Duration("timeout", 0, "command execution timeout for the bounded runner (required with --runner)")

	// Find the trailing "--" separator for command arguments
	dashIdx := -1
	for i, arg := range argv {
		if arg == "--" {
			dashIdx = i
			break
		}
	}

	var flagArgs []string
	var dashArgs []string
	hasDash := false
	if dashIdx >= 0 {
		hasDash = true
		flagArgs = argv[:dashIdx]
		dashArgs = argv[dashIdx+1:]
	} else {
		flagArgs = argv
	}

	if !parseFlags(fs, flagArgs) {
		return 2
	}

	rest := fs.Args()
	if hasDash {
		rest = append(rest, dashArgs...)
	}

	schedTarget := strings.TrimSpace(*target)
	if schedTarget == "" {
		schedTarget = strings.TrimSpace(*format)
	}
	if schedTarget == "" {
		fmt.Fprintln(stderr, "fak cron emit: --target is required (launchd|systemd|taskscheduler)")
		return 2
	}
	switch schedTarget {
	case "tasksched", "task-scheduler":
		schedTarget = "taskscheduler"
	}
	source, ok := cronSources[schedTarget]
	if !ok {
		fmt.Fprintf(stderr, "fak cron emit: unknown --target %q (want launchd|systemd|taskscheduler)\n", schedTarget)
		return 2
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak cron emit: --interval must be positive")
		return 2
	}

	bin := strings.TrimSpace(*fakBin)
	if bin == "" {
		if exe, err := os.Executable(); err == nil && exe != "" {
			bin = exe
		} else {
			bin = "fak"
		}
	}

	// Mode 1: --runner (Ticket #11830)
	if *runner {
		if strings.TrimSpace(*command) != "" {
			fmt.Fprintln(stderr, "fak cron emit: --command is not allowed with --runner")
			return 2
		}
		if strings.TrimSpace(*loopID) != "" {
			fmt.Fprintln(stderr, "fak cron emit: --loop is not allowed with --runner")
			return 2
		}
		if strings.TrimSpace(*job) == "" {
			fmt.Fprintln(stderr, "fak cron emit: --job is required with --runner")
			return 2
		}
		if strings.TrimSpace(*ledger) == "" {
			fmt.Fprintln(stderr, "fak cron emit: --ledger is required with --runner")
			return 2
		}
		if *timeout <= 0 {
			fmt.Fprintln(stderr, "fak cron emit: --timeout must be positive with --runner")
			return 2
		}
		cmdArgs := rest
		if len(cmdArgs) == 0 {
			fmt.Fprintln(stderr, "fak cron emit: trailing command args are required with --runner")
			return 2
		}
		if strings.TrimSpace(*label) == "" {
			*label = "fak-cron-" + cronSanitizeLabel(*job)
		}

		runArgs := []string{
			bin, "cron", "run",
			"--job", *job,
			"--ledger", *ledger,
			"--interval", interval.String(),
			"--timeout", timeout.String(),
			"--",
		}
		runArgs = append(runArgs, cmdArgs...)

		joinedCmd := strings.Join(cmdArgs, " ")
		descs := cronDescs{
			service: fmt.Sprintf("fak cron run %s (%s)", *job, joinedCmd),
			timer:   "Timer for " + *label,
			task:    fmt.Sprintf("fak cron run %s (%s)", *job, joinedCmd),
		}
		cronRender(stdout, schedTarget, *label, descs, *interval, runArgs)
		return 0
	}

	// --command (or a bare trailing `-- CMD ARG...` with no loop id) emits a unit
	// whose action is exactly the arbitrary command — no `fak loop run` wrapper, no
	// loop id required. This is the general escape the stale-work garden watchdog
	// (#1386) needs (`--command 'fak garden --check'`). When neither --command nor a
	// `--`-tail is given we keep the default `fak loop run` form byte-for-byte (#1385).
	if strings.TrimSpace(*command) != "" {
		cmdVec, ok := cronShlexSplit(*command)
		if !ok {
			fmt.Fprintln(stderr, "fak cron emit: --command could not be parsed (unbalanced quote or dangling escape)")
			return 2
		}
		if len(cmdVec) == 0 {
			fmt.Fprintln(stderr, "fak cron emit: --command is empty")
			return 2
		}
		if strings.TrimSpace(*label) == "" {
			*label = "fak-cron-" + cronSanitizeLabel(cmdVec[0])
		}
		joined := strings.Join(cmdVec, " ")
		descs := cronDescs{
			service: "fak cron command " + joined + " (cron-emitted; OS fires, fak owns overlap-lock + missed-run policy)",
			timer:   "Timer for " + *label,
			task:    "fak cron command " + joined + " (cron-emitted)",
		}
		cronRender(stdout, schedTarget, *label, descs, *interval, cmdVec)
		return 0
	}

	// Default `fak loop run` form: a positional or --loop id is required.
	if strings.TrimSpace(*loopID) == "" && len(rest) > 0 {
		*loopID = rest[0]
		rest = rest[1:]
	}
	tick := rest
	if strings.TrimSpace(*loopID) == "" {
		fmt.Fprintln(stderr, "fak cron emit: a loop id is required (--loop ID or positional, or use --command)")
		return 2
	}
	if len(tick) == 0 {
		// Default the wrapped tick to `fak agent`; the operator overrides it with
		// `-- CMD ARG...`. The unit always invokes `fak loop run` either way.
		tick = []string{bin, "agent"}
	}
	if strings.TrimSpace(*label) == "" {
		*label = "fak-loop-" + cronSanitizeLabel(*loopID)
	}

	// The action every emitted unit invokes. fak loop run owns the semantics; the
	// OS scheduler only fires it on the interval.
	runArgs := []string{bin, "loop", "run", "--loop", *loopID, "--source", source}
	if strings.TrimSpace(*ledger) != "" {
		runArgs = append(runArgs, "--ledger", *ledger)
	}
	runArgs = append(runArgs, "--")
	runArgs = append(runArgs, tick...)

	descs := cronDescs{
		service: "fak loop " + *loopID + " (cron-emitted; OS fires, fak owns overlap-lock + missed-run policy)",
		timer:   "Timer for fak loop " + *loopID,
		task:    "fak loop " + *loopID + " (cron-emitted)",
	}
	cronRender(stdout, schedTarget, *label, descs, *interval, runArgs)
	return 0
}

// cronDescs carries the human-readable Description strings each renderer stamps.
// They differ by mode (default `fak loop run` vs an arbitrary --command) and by
// target (systemd uses a service + a timer line; Task Scheduler uses one), so the
// resolved values are computed once at the call site and threaded through.
type cronDescs struct {
	service string // systemd [Unit] Description for the .service
	timer   string // systemd [Unit] Description for the .timer
	task    string // Task Scheduler -Description
}

// cronRender dispatches the resolved argv to the per-target renderer.
func cronRender(stdout io.Writer, target, label string, descs cronDescs, interval time.Duration, args []string) {
	switch target {
	case "launchd":
		fmt.Fprint(stdout, cronRenderLaunchd(label, interval, args))
	case "systemd":
		fmt.Fprint(stdout, cronRenderSystemd(label, descs, interval, args))
	case "taskscheduler":
		fmt.Fprint(stdout, cronRenderTaskScheduler(label, descs.task, interval, args))
	}
}

// cronRenderLaunchd renders a launchd .plist whose ProgramArguments is the
// `fak loop run` vector and whose StartInterval is the firing cadence in seconds.
func cronRenderLaunchd(label string, interval time.Duration, args []string) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	fmt.Fprintf(&b, "<!-- Written by: fak cron emit (#765) — install: launchctl load -w %s.plist -->\n", label)
	b.WriteString(`<plist version="1.0">` + "\n")
	b.WriteString("  <dict>\n")
	fmt.Fprintf(&b, "    <key>Label</key>\n    <string>%s</string>\n", cronXMLEscape(label))
	b.WriteString("    <key>ProgramArguments</key>\n    <array>\n")
	for _, a := range args {
		fmt.Fprintf(&b, "      <string>%s</string>\n", cronXMLEscape(a))
	}
	b.WriteString("    </array>\n")
	fmt.Fprintf(&b, "    <key>StartInterval</key>\n    <integer>%d</integer>\n", int64(interval.Seconds()))
	b.WriteString("    <key>RunAtLoad</key>\n    <false/>\n")
	fmt.Fprintf(&b, "    <key>StandardOutPath</key>\n    <string>/tmp/%s.log</string>\n", cronXMLEscape(label))
	fmt.Fprintf(&b, "    <key>StandardErrorPath</key>\n    <string>/tmp/%s.err</string>\n", cronXMLEscape(label))
	b.WriteString("  </dict>\n</plist>\n")
	return b.String()
}

// cronRenderSystemd renders a systemd timer+service pair (concatenated, each under
// a `# === <name> ===` header). The service is a oneshot whose ExecStart is the
// action vector (the `fak loop run` wrapper by default, or the arbitrary --command
// vector); the timer fires it every interval.
func cronRenderSystemd(label string, descs cronDescs, interval time.Duration, args []string) string {
	sec := int64(interval.Seconds())
	var b strings.Builder
	fmt.Fprintf(&b, "# Written by: fak cron emit (#765). Install both units to ~/.config/systemd/user/,\n")
	fmt.Fprintf(&b, "# then: systemctl --user enable --now %s.timer\n", label)
	fmt.Fprintf(&b, "\n# === %s.service ===\n", label)
	b.WriteString("[Unit]\n")
	fmt.Fprintf(&b, "Description=%s\n\n", descs.service)
	b.WriteString("[Service]\n")
	b.WriteString("Type=oneshot\n")
	fmt.Fprintf(&b, "ExecStart=%s\n", cronSystemdExecLine(args))
	fmt.Fprintf(&b, "\n# === %s.timer ===\n", label)
	b.WriteString("[Unit]\n")
	fmt.Fprintf(&b, "Description=%s\n\n", descs.timer)
	b.WriteString("[Timer]\n")
	fmt.Fprintf(&b, "OnBootSec=%ds\n", sec)
	fmt.Fprintf(&b, "OnUnitActiveSec=%ds\n", sec)
	b.WriteString("Persistent=true\n\n")
	b.WriteString("[Install]\n")
	b.WriteString("WantedBy=timers.target\n")
	return b.String()
}

// cronRenderTaskScheduler renders a PowerShell `Register-ScheduledTask` snippet.
// args[0] is the executed binary (the task's -Execute); args[1:] is its -Argument
// string. desc is the task's -Description.
//
// The emitted task carries the S4U principal the fleet's reboot-survival contract
// (#3322) requires of every registered fak task. Without an explicit -Principal the task
// is registered for the INTERACTIVE user and only fires while that user is logged on: it
// survives no reboot that lands at the lock screen, which is the normal state of an
// unattended box. `-LogonType S4U` runs it without a stored password and without
// requiring a session; `-RunLevel Limited` keeps it unelevated (an emitted unit must
// never quietly ask for more privilege than the command needs); `-StartWhenAvailable`
// catches up a firing the machine slept through, so a once-a-day command still runs on a
// day the box was asleep at 03:00. That contract was previously pinned only on the
// tools/*.ps1 installers (fleet_installer_s4u_test.go) while fak's OWN emitter shipped
// units that died at every reboot — the gap this closes.
func cronRenderTaskScheduler(label, desc string, interval time.Duration, args []string) string {
	sec := int64(interval.Seconds())
	var b strings.Builder
	b.WriteString("# Written by: fak cron emit (#765). Run in PowerShell to register the task;\n")
	b.WriteString("# Task Scheduler fires on the interval, the unit's command owns the semantics.\n")
	fmt.Fprintf(&b, "$action   = New-ScheduledTaskAction -Execute '%s' -Argument '%s'\n",
		cronPSQuote(args[0]), cronWinArgString(args[1:]))
	fmt.Fprintf(&b, "$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Seconds %d)\n", sec)
	b.WriteString("$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable\n")
	b.WriteString("# Reboot survival (#3322): S4U runs with no stored password and no logged-on session.\n")
	b.WriteString("$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited\n")
	fmt.Fprintf(&b, "Register-ScheduledTask -TaskName '%s' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description '%s' -Force\n",
		cronPSQuote(label), cronPSQuote(desc))
	return b.String()
}

// cronSystemdExecLine joins an argv into a systemd ExecStart line, double-quoting
// any argument that contains whitespace, a quote, or a backslash (systemd's own quoting rules).
func cronSystemdExecLine(args []string) string {
	parts := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t\"\\") {
			parts[i] = `"` + strings.ReplaceAll(strings.ReplaceAll(a, `\`, `\\`), `"`, `\"`) + `"`
		} else {
			parts[i] = a
		}
	}
	return strings.Join(parts, " ")
}

// cronWinArgString builds the single -Argument string for Register-ScheduledTask
// from the post-binary argv: arguments with whitespace get embedded double quotes,
// then the whole string is escaped for the enclosing PowerShell single-quoted literal.
func cronWinArgString(args []string) string {
	parts := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t") {
			parts[i] = `"` + a + `"`
		} else {
			parts[i] = a
		}
	}
	return cronPSQuote(strings.Join(parts, " "))
}

// cronPSQuote escapes a string for a PowerShell single-quoted literal (a literal
// single quote is doubled).
func cronPSQuote(s string) string { return strings.ReplaceAll(s, "'", "''") }

// cronXMLEscape escapes the five XML metacharacters for plist text nodes.
func cronXMLEscape(s string) string {
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;",
	).Replace(s)
}

// cronShlexSplit splits a --command string into an argv the way a POSIX shell
// would (shlex.split, posix=True): unquoted whitespace separates tokens; single
// quotes are literal; double quotes allow a `\"` / `\\` escape; a backslash outside
// quotes escapes the next char. ok is false on an unbalanced quote or a dangling
// escape so the caller can refuse rather than emit a half-parsed unit. The split is
// only how we turn one string into a vector — each token is re-quoted faithfully by
// the per-target renderer, so embedded spaces survive into the emitted unit.
func cronShlexSplit(s string) (tokens []string, ok bool) {
	return repoguard.ShlexSplit(s)
}

// cronSanitizeLabel reduces a loop id to a safe unit/task basename (letters,
// digits, dot, underscore, dash; everything else becomes a dash).
func cronSanitizeLabel(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return b.String()
}

func cronUsage(w io.Writer) {
	fmt.Fprint(w, `fak cron - project the in-kernel loop schedule to an OS scheduler unit

  fak cron emit --target launchd|systemd|taskscheduler [--loop ID | <job>]
                [--command 'CMD ARG...'] [--interval DUR] [--fak-bin PATH]
                [--label NAME] [--ledger FILE] [-- TICK-CMD ARG...]
  fak cron emit --runner --target launchd|systemd|taskscheduler --job ID
                --ledger FILE --interval DUR --timeout DUR [--fak-bin PATH]
                [--label NAME] -- CMD ARG...

  fak cron run    --job ID --ledger FILE --interval DUR --timeout DUR [--json]
                  [--at RFC3339] [--slot KEY] -- CMD ARG...
  fak cron fire   --job ID --ledger FILE [--interval DUR] [--at RFC3339] [--slot KEY]
  fak cron audit  --ledger FILE [--job ID] [--json]
  fak cron prompt --job ID --ledger FILE [--script 'CMD'] [--context-from A,B]
                  [--base 'PROMPT'] [--interval DUR] [--at RFC3339] [--slot KEY]
  fak cron chain  --ledger FILE [--job ID] [--json]

Run is the BOUNDED TASK RUNNER (#11829): it executes an admitted scheduled task
under a timeout with deduplication and process-tree termination. Exit 0 on success,
exit 3 on dedup, child exit code on failure, and 124 on timeout. Terminal outcomes
(succeeded, failed, timeout) are witnessed in the ledger.

Fire is the FIRE WITNESS (#2886): it records each fire in the ledger under a
(job, slot) compare-and-set guarded by a dup-tick lock, so a duplicate or
overlapping tick for an already-fired slot is recorded as a `+"`deduped`"+` event
instead of double-delivering. Exit 0 = fired (run the job), 3 = deduped (skip) —
so `+"`fak cron fire ... && <run job>`"+` delivers at most once. Audit rolls the
ledger up per job into fired / missed / deduped counts (--json for machine read).

Prompt turns cron into a witnessed PIPELINE (#2888): --script runs a pre-run
command and records its stdout as a `+"`fak-cron-output/1`"+` row tagged
provenance=OBSERVED (tool output, NOT agent-authored), then injects it; and
--context-from A[,B...] chains each named job's LAST witnessed output into this
prompt, recording each A→B handoff as a `+"`fak-cron-edge/1`"+` ledger edge. A
context source with no witnessed output is refused (exit 2) — B consumes only
what A provably produced. The assembled prompt (OBSERVED blocks + --base) prints
to stdout for a dispatcher to hand the agent.

Chain reads the `+"`fak-cron-edge/1`"+` handoffs back OUT of the ledger so an operator
can traverse a chained pipeline after the fact — each row prints as
`+"`FROM@slot -> TO@slot (consumed …)`"+` (--json for machine read, --job to filter to
the edges touching one job). It is the readback half of the `+"`context-from`"+` audit
trail: prompt writes the edges, chain makes them queryable evidence.

Emit renders ONE OS scheduler unit. By default its command is `+"`fak loop run --loop <id> ...`"+`
— fak owns the semantics (overlap-lock via the ledger, missed-run policy) and the OS
scheduler (launchd / systemd / Windows Task Scheduler) only owns wall-clock firing. The
wrapped tick defaults to `+"`fak agent`"+` and is overridden with `+"`-- CMD ARG...`"+`.

--runner emits a unit whose command invokes `+"`fak cron run`"+` to execute a scheduled
task bounded by --timeout with ledger-witnessed outcomes (#11830).

--command 'CMD ARG...' instead emits a unit whose action is exactly that ARBITRARY
command (no `+"`fak loop run`"+` wrapper, no loop id) — e.g.
  fak cron emit --target taskscheduler --label FleetStaleWorkGarden --command 'fak garden --check' --interval 1h
The command is shlex-split and each token re-quoted faithfully into the unit. The
operator installs the printed unit — fak does not supervise a second scheduler.
`)
}
