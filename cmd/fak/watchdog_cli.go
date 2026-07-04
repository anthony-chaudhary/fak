package main

// watchdog_cli.go — `fak watchdog`, the operator surface over fak's DEFAULT watchdog
// monitors: the OS-scheduled fleet timers (resume, supervisor, dos-dispatch,
// stale-work-garden) that watchdog-autoheal keeps alive on every `fak serve` / `fak guard`
// boot.
//
//	fak watchdog status           # read-only: probe every default monitor + fold its heal-state
//	fak watchdog status --json    # the raw digest as JSON
//	fak watchdog status --check   # exit 1 if the layer needs a human (DOWN/UNKNOWN/GAVE_UP)
//	fak watchdog heal             # run the autoheal ONCE, now (restart dead-but-installed monitors)
//	fak watchdog heal --warn      # heal in warn-only mode (report, do not restart)
//
// The autoheal has always PERSISTED a per-monitor heal-state (attempts, last restart, last
// reason) but never read it back for a human — the only way to see whether the default
// monitors were alive was to scrape the JSON the last boot wrote to stderr / autoheal.log.
// `status` folds the live probe + that heal-state into internal/watchdoghealth's closed
// digest so the question "are my default monitors healthy right now?" has a first-class
// answer. `heal` exposes the same background routine `serve`/`guard` fire on boot as an
// on-demand verb, so an operator no longer has to bounce a long-lived process to force a
// restart. This file is the thin I/O shell; every classification decision lives in the pure
// leaf.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/watchdoghealth"
)

func cmdWatchdog(args []string) {
	if len(args) == 0 {
		watchdogUsage(os.Stderr)
		os.Exit(2)
	}
	switch args[0] {
	case "status":
		os.Exit(runWatchdogStatus(os.Stdout, os.Stderr, args[1:]))
	case "heal":
		os.Exit(runWatchdogHeal(os.Stdout, os.Stderr, args[1:]))
	case "-h", "--help", "help":
		watchdogUsage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "fak watchdog: unknown subcommand %q\n\n", args[0])
		watchdogUsage(os.Stderr)
		os.Exit(2)
	}
}

func watchdogUsage(w io.Writer) {
	fmt.Fprint(w, `fak watchdog — inspect and heal the default watchdog monitors

Usage:
  fak watchdog status [--json] [--check]   probe every default monitor + fold its heal-state
  fak watchdog heal   [--warn] [--json]    run the autoheal once, now

The default monitors are the OS-scheduled fleet timers watchdog-autoheal keeps alive on
every fak serve / fak guard boot; the set is platform-specific (Task Scheduler on Windows,
launchd on macOS, systemd --user on Linux). A monitor that is not installed on this host is
reported NOT_INSTALLED, never an error.
`)
}

// watchdogStatusMonitors probes every default monitor for this platform and joins each
// probe with its persisted heal-state into the pure leaf's Monitor input. It is the I/O the
// digest core must not do: run the scheduler query, read the state file. maxAttempts is the
// autoheal's give-up cap, needed so a dead monitor with an exhausted streak reads GAVE_UP
// rather than an eternal HEALING.
func watchdogStatusMonitors(ctx context.Context, specs []watchdogAutohealSpec, stateDir string, maxAttempts uint64) []watchdoghealth.Monitor {
	mons := make([]watchdoghealth.Monitor, 0, len(specs))
	for _, spec := range specs {
		m := watchdoghealth.Monitor{
			ID:          spec.ID,
			Manager:     spec.Manager,
			Unit:        spec.Unit,
			MaxAttempts: maxAttempts,
		}
		if spec.Probe == nil {
			m.ProbeErr = true
			m.Detail = "no probe wired for this monitor"
		} else if probe, err := spec.Probe(ctx); err != nil {
			m.ProbeErr = true
			m.Detail = err.Error()
		} else {
			m.Installed = probe.Installed
			m.Alive = probe.Alive
			m.Detail = probe.Detail
		}
		if st, err := readWatchdogHealState(stateDir, spec.ID); err == nil {
			m.Attempts = st.Attempts
			m.LastRestartUnixNano = st.LastRestartUnixNano
			m.LastFailureUnixNano = st.LastFailureUnixNano
			m.LastProbeAliveUnixNano = st.LastProbeAliveUnixNano
			m.LastReason = st.LastReason
		}
		mons = append(mons, m)
	}
	return mons
}

func runWatchdogStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("watchdog status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the raw digest as JSON instead of the human table")
	check := fs.Bool("check", false, "exit 1 (not 0) when the layer needs attention (DOWN/UNKNOWN/GAVE_UP)")
	if !parseFlags(fs, argv) {
		return 2
	}

	opts := defaultWatchdogAutohealOptions("watchdog", watchdogAutohealOn)
	specs := watchdogAutohealSpecsForGOOS(runtime.GOOS, watchdogRunCommand)
	if len(specs) == 0 {
		fmt.Fprintf(stderr, "fak watchdog status: no default watchdog monitors are defined for %s\n", runtime.GOOS)
		return 0
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	mons := watchdogStatusMonitors(ctx, specs, opts.StateDir, uint64(opts.RestartPolicy.MaxAttempts))
	digest := watchdoghealth.Fold(mons)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(digest); err != nil {
			fmt.Fprintf(stderr, "fak watchdog status: encode: %v\n", err)
			return 1
		}
	} else {
		writeWatchdogStatusTable(stdout, digest)
	}

	if *check && digest.NeedsAttention {
		return 1
	}
	return 0
}

// writeWatchdogStatusTable renders the digest as an aligned human table plus a one-line
// rollup. Timestamps are formatted here (the shell's job); the leaf carried only the raw
// nanos.
func writeWatchdogStatusTable(w io.Writer, d watchdoghealth.Digest) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MONITOR\tMANAGER\tSTATUS\tATTEMPTS\tLAST-RESTART\tDETAIL")
	for _, h := range d.Monitors {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n",
			h.ID, h.Manager, h.Status,
			watchdogAttemptsCell(h.Status, h.Attempts),
			watchdogTimeCell(h.LastRestartUnixNano),
			h.Detail)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "\nrollup: %s", d.Rollup)
	if d.NeedsAttention {
		fmt.Fprint(w, "  (needs attention)")
	}
	fmt.Fprintln(w)
}

// watchdogAttemptsCell shows the restart streak only where it is meaningful — a healthy or
// not-installed monitor has no streak to report.
func watchdogAttemptsCell(s watchdoghealth.Status, attempts uint64) string {
	switch s {
	case watchdoghealth.StatusHealthy, watchdoghealth.StatusNotInstalled:
		return "-"
	default:
		return fmt.Sprintf("%d", attempts)
	}
}

// watchdogTimeCell formats a persisted unix-nano timestamp as a local RFC3339 stamp, or "-"
// when nothing was recorded.
func watchdogTimeCell(unixNano int64) string {
	if unixNano <= 0 {
		return "-"
	}
	return time.Unix(0, unixNano).Format(time.RFC3339)
}

func runWatchdogHeal(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("watchdog heal", flag.ContinueOnError)
	fs.SetOutput(stderr)
	warn := fs.Bool("warn", false, "warn-only: report installed-but-dead monitors without restarting them")
	asJSON := fs.Bool("json", false, "emit the per-monitor heal results as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	mode := watchdogAutohealOn
	if *warn {
		mode = watchdogAutohealWarn
	}
	opts := defaultWatchdogAutohealOptions("watchdog", mode)
	if len(opts.Specs) == 0 {
		fmt.Fprintf(stderr, "fak watchdog heal: no default watchdog monitors are defined for %s\n", runtime.GOOS)
		return 0
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	t0 := time.Now()
	results := runWatchdogAutoheal(ctx, opts)
	elapsed := time.Since(t0)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(results); err != nil {
			fmt.Fprintf(stderr, "fak watchdog heal: encode: %v\n", err)
			return 1
		}
		return 0
	}
	logWatchdogAutohealResults(stdout, results)
	if b := watchdogAutohealSummaryLine("watchdog", elapsed, results); b != nil {
		fmt.Fprintf(stdout, "%s\n", b)
	}
	return 0
}
