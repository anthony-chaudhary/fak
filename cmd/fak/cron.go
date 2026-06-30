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
	"launchd":       "launchd",
	"systemd":       "systemd",
	"taskscheduler": "task-scheduler",
}

func runCronEmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron emit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	target := fs.String("target", "", "OS scheduler to emit for: launchd|systemd|taskscheduler")
	loopID := fs.String("loop", "", "loop id this unit fires (may be given positionally)")
	command := fs.String("command", "", "emit a unit running this ARBITRARY command instead of `fak loop run` (e.g. 'fak garden --check')")
	interval := fs.Duration("interval", time.Hour, "firing cadence the OS scheduler enforces (e.g. 5m, 1h)")
	fakBin := fs.String("fak-bin", "fak", "path to the fak binary the unit invokes")
	label := fs.String("label", "", "unit/task name (default fak-loop-<id>)")
	ledger := fs.String("ledger", "", "loop ledger path passed through to fak loop run")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// A positional job id supports the acceptance form `fak cron emit --target T <job>`.
	// Anything after it (typically after a `--`) is the wrapped tick command. In
	// --command mode the positional/`--` tail is the command vector instead.
	rest := fs.Args()

	if strings.TrimSpace(*target) == "" {
		fmt.Fprintln(stderr, "fak cron emit: --target is required (launchd|systemd|taskscheduler)")
		return 2
	}
	source, ok := cronSources[*target]
	if !ok {
		fmt.Fprintf(stderr, "fak cron emit: unknown --target %q (want launchd|systemd|taskscheduler)\n", *target)
		return 2
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak cron emit: --interval must be positive")
		return 2
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
		cronRender(stdout, *target, *label, descs, *interval, cmdVec)
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
		tick = []string{*fakBin, "agent"}
	}
	if strings.TrimSpace(*label) == "" {
		*label = "fak-loop-" + cronSanitizeLabel(*loopID)
	}

	// The action every emitted unit invokes. fak loop run owns the semantics; the
	// OS scheduler only fires it on the interval.
	runArgs := []string{*fakBin, "loop", "run", "--loop", *loopID, "--source", source}
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
	cronRender(stdout, *target, *label, descs, *interval, runArgs)
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
func cronRenderTaskScheduler(label, desc string, interval time.Duration, args []string) string {
	sec := int64(interval.Seconds())
	var b strings.Builder
	b.WriteString("# Written by: fak cron emit (#765). Run in PowerShell to register the task;\n")
	b.WriteString("# Task Scheduler fires on the interval, the unit's command owns the semantics.\n")
	fmt.Fprintf(&b, "$action   = New-ScheduledTaskAction -Execute '%s' -Argument '%s'\n",
		cronPSQuote(args[0]), cronWinArgString(args[1:]))
	fmt.Fprintf(&b, "$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Seconds %d)\n", sec)
	b.WriteString("$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable\n")
	fmt.Fprintf(&b, "Register-ScheduledTask -TaskName '%s' -Action $action -Trigger $trigger -Settings $settings -Description '%s' -Force\n",
		cronPSQuote(label), cronPSQuote(desc))
	return b.String()
}

// cronSystemdExecLine joins an argv into a systemd ExecStart line, double-quoting
// any argument that contains whitespace or a quote (systemd's own quoting rules).
func cronSystemdExecLine(args []string) string {
	parts := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t\"") {
			parts[i] = `"` + strings.ReplaceAll(a, `"`, `\"`) + `"`
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
	var cur []rune
	started := false // cur is a real token (including an empty quoted "")
	r := []rune(s)
	i, n := 0, len(r)
	for i < n {
		c := r[i]
		switch {
		case c == '\\':
			if i+1 >= n {
				return nil, false // dangling escape
			}
			cur = append(cur, r[i+1])
			started = true
			i += 2
		case c == '\'':
			j := i + 1
			for j < n && r[j] != '\'' {
				j++
			}
			if j >= n {
				return nil, false // no closing single quote
			}
			cur = append(cur, r[i+1:j]...)
			started = true
			i = j + 1
		case c == '"':
			j := i + 1
			closed := false
			for j < n {
				if r[j] == '\\' && j+1 < n && (r[j+1] == '"' || r[j+1] == '\\') {
					cur = append(cur, r[j+1])
					j += 2
					continue
				}
				if r[j] == '"' {
					closed = true
					break
				}
				cur = append(cur, r[j])
				j++
			}
			if !closed {
				return nil, false // no closing double quote
			}
			started = true
			i = j + 1
		case c == ' ' || c == '\t' || c == '\r' || c == '\n':
			if started {
				tokens = append(tokens, string(cur))
				cur = cur[:0]
				started = false
			}
			i++
		default:
			cur = append(cur, c)
			started = true
			i++
		}
	}
	if started {
		tokens = append(tokens, string(cur))
	}
	return tokens, true
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

Emit renders ONE OS scheduler unit. By default its command is `+"`fak loop run --loop <id> ...`"+`
— fak owns the semantics (overlap-lock via the ledger, missed-run policy) and the OS
scheduler (launchd / systemd / Windows Task Scheduler) only owns wall-clock firing. The
wrapped tick defaults to `+"`fak agent`"+` and is overridden with `+"`-- CMD ARG...`"+`.

--command 'CMD ARG...' instead emits a unit whose action is exactly that ARBITRARY
command (no `+"`fak loop run`"+` wrapper, no loop id) — e.g.
  fak cron emit --target taskscheduler --label FleetStaleWorkGarden --command 'fak garden --check' --interval 1h
The command is shlex-split and each token re-quoted faithfully into the unit. The
operator installs the printed unit — fak does not supervise a second scheduler.
`)
}
