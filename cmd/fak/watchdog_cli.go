package main

// watchdog_cli.go — `fak watchdog`, the operator surface over fak's DEFAULT watchdog
// monitors: the OS-scheduled fleet timers (resume, supervisor, dos-dispatch,
// stale-work-garden) that watchdog-autoheal keeps alive on every `fak serve` / `fak guard`
// boot.
//
//	fak watchdog status           # read-only: probe every default monitor + fold its heal-state
//	fak watchdog status --json    # the raw digest as JSON
//	fak watchdog status --check   # exit 1 if the layer needs a human (DOWN/UNKNOWN/GAVE_UP)
//	fak watchdog status --post-slack  # enqueue the health digest to Slack when it needs a human
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
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
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
	case "selfcheck":
		// The no-I/O proof of the decenter-the-human fold at the watchdog seam: a DOWN
		// monitor is the autoheal's to restart, an UNKNOWN monitor is a fresh-context
		// re-probe, and only a GAVE_UP monitor genuinely waits on a person.
		os.Exit(runReportSelfcheck(os.Stdout, os.Stderr, args[1:], "watchdog", watchdoghealth.TriageSelfcheck,
			"SELFCHECK OK -- decenter-the-human at the watchdog: a DOWN monitor is the autoheal's "+
				"restart and an UNKNOWN monitor is a fresh-context re-probe; only a GAVE_UP monitor "+
				"(automatic recovery exhausted) waits on a person."))
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
  fak watchdog status --post-slack         post the health digest to Slack when it needs a human
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
	postSlack := fs.Bool("post-slack", false, "enqueue the health digest to the Slack outbox when it needs attention (one coalesced watchdog-health card)")
	channel := fs.String("channel", "", "Slack channel for --post-slack (default: $FAK_WATCHDOG_SLACK_CHANNEL)")
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
	// Publish the folded health onto the process-global expvar surface so /debug/vars carries
	// the last-seen per-monitor status and the cross-monitor rollup (#3803).
	for _, h := range digest.Monitors {
		resumemetrics.SetMonitorStatus(h.ID, string(h.Status))
	}
	resumemetrics.SetHealthRollup(string(digest.Rollup))

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(digest); err != nil {
			fmt.Fprintf(stderr, "fak watchdog status: encode: %v\n", err)
			return 1
		}
	} else {
		writeWatchdogStatusTable(stdout, digest)
		// Decenter the human at the source: under FAK_WATCHDOG_TRIAGE_GATE=enforce, append
		// the attention split — the monitors that genuinely wait on a person (a GAVE_UP
		// monitor whose automatic recovery is exhausted) vs the ones the fleet clears itself
		// (a DOWN monitor the autoheal restarts, an UNKNOWN monitor to re-probe). Default
		// ("", "warn") leaves the readout byte-for-byte unchanged while it soaks.
		if watchdoghealth.WatchdogTriageEnforced(os.Getenv("FAK_WATCHDOG_TRIAGE_GATE")) {
			if line := watchdoghealth.AttentionTriageLine(digest); line != "" {
				fmt.Fprintln(stdout, line)
			}
		}
	}

	if *postSlack {
		if code := postWatchdogHealthDigest(ctx, stdout, stderr, digest, strings.TrimSpace(*channel)); code != 0 {
			return code
		}
	}

	if *check && watchdogCheckTrips(digest) {
		return 1
	}
	return 0
}

// postWatchdogHealthDigest folds the digest into internal/watchdoghealth's channel card and,
// when it wants attention (ShouldPost — the SAME condition `--check` exits non-zero on),
// enqueues it to the Slack outbox under a stable, host-scoped CardKey so repeat runs edit ONE
// standing card in place rather than posting a fresh message each run. An all-clear digest posts
// nothing and returns 0, so a scheduled `status --post-slack` speaks only when the fleet needs a
// person. Delivery is best-effort: a missing token or a busy drain leaves the row durably
// spooled for the next `fak slack outbox drain`.
func postWatchdogHealthDigest(ctx context.Context, stdout, stderr io.Writer, d watchdoghealth.Digest, channel string) int {
	card := watchdoghealth.SlackHealthDigest(d)
	if !card.ShouldPost {
		return 0 // healthy — the channel stays quiet
	}
	if channel == "" {
		channel = strings.TrimSpace(os.Getenv("FAK_WATCHDOG_SLACK_CHANNEL"))
	}
	if channel == "" {
		fmt.Fprintln(stderr, "fak watchdog status --post-slack: no channel: pass --channel or set FAK_WATCHDOG_SLACK_CHANNEL")
		return 2
	}

	// Durability first: the row is on disk before any send, so nothing below can lose it.
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak watchdog status --post-slack: outbox: %v\n", err)
		return 1
	}
	cardKey := watchdogHealthCardKey()
	row := slackoutbox.Row{
		Channel: channel,
		Text:    card.Title + "\n" + card.Body,
		CardKey: cardKey,
		Source:  "watchdog:health",
	}
	// Edit the standing card in place once it has posted, so a periodic health post coalesces
	// to a single message instead of a new one every run.
	if snap, err := ob.Load(); err == nil {
		if ts := lastPostedTSForCard(snap, cardKey); ts != "" {
			row.UpdateTS = ts
		}
	}
	nonce, err := ob.Enqueue(row)
	if err != nil {
		fmt.Fprintf(stderr, "fak watchdog status --post-slack: enqueue: %v\n", err)
		return 1
	}

	// Best-effort in-process drain. A missing token, a busy lock, or a transport error all leave
	// the row durably spooled for the next drain rather than failing the status command.
	wire, werr := outboxWire("", "")
	if werr != nil {
		fmt.Fprintf(stdout, "fak watchdog status: health digest enqueued to %s (nonce=%s); delivery deferred: %v — run `fak slack outbox drain`\n", channel, nonce, werr)
		return 0
	}
	if _, err := ob.Drain(ctx, wire, stdDrainOpts()); err != nil && err != slackoutbox.ErrDrainBusy {
		fmt.Fprintf(stdout, "fak watchdog status: health digest enqueued to %s (nonce=%s); delivery deferred: %v — run `fak slack outbox drain`\n", channel, nonce, err)
		return 0
	}
	fmt.Fprintf(stdout, "fak watchdog status: health digest delivered to %s (nonce=%s)\n", channel, nonce)
	return 0
}

// watchdogHealthCardKey is the stable slackoutbox CardKey the health digest posts under. It is
// host-scoped so a multi-machine fleet keeps one standing card per machine rather than fighting
// over a shared one.
func watchdogHealthCardKey() string {
	host := "local"
	if h, err := os.Hostname(); err == nil {
		if h = strings.TrimSpace(h); h != "" {
			host = h
		}
	}
	return "watchdog-health:" + host
}

// lastPostedTSForCard returns the Slack ts of the most recent posted row carrying cardKey, or ""
// if the card has never posted. Rows replay in append order, so the last posted match is the
// current card — letting a stateless CLI edit its standing card (chat.update) without holding the
// ts in memory across runs.
func lastPostedTSForCard(snap *slackoutbox.Snapshot, cardKey string) string {
	ts := ""
	for _, r := range snap.Rows {
		if r.CardKey != cardKey {
			continue
		}
		if posted := snap.PostedTS(r.Nonce); posted != "" {
			ts = posted
		}
	}
	return ts
}

// watchdogCheckTrips is the `--check` exit-1 condition. By default it is the digest's own
// NeedsAttention (DOWN / UNKNOWN / GAVE_UP), unchanged. Under FAK_WATCHDOG_TRIAGE_GATE=enforce
// it narrows to the decenter residual — only a monitor that genuinely waits on a person (a
// GAVE_UP or auth-walled monitor) — so the fleet's own DOWN/UNKNOWN recoveries stop paging.
func watchdogCheckTrips(d watchdoghealth.Digest) bool {
	if watchdoghealth.WatchdogTriageEnforced(os.Getenv("FAK_WATCHDOG_TRIAGE_GATE")) {
		return watchdoghealth.NeedsHumanAttention(d)
	}
	return d.NeedsAttention
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
