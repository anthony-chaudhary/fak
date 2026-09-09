package main

// garden_loop.go — `fak garden loop` (alias: `fak garden daemon`): the long-running,
// portable native Go cadence runner for local system gardening sweeps.
//
// It unifies the repository's housekeeping sweeps:
//   1. Ephemera GC & stale-work watchdog (`fak garden watchdog` + child `tick`)
//   2. Build-binary cleanup (`fak clean-bins`)
//   3. Tree-doctor worktree pruning & stale commit lock reap (`fak tree-doctor`)
//   4. Git loose-ref packing (`git pack-refs --all`)
//
// Designed to run portably across platforms without loose shell or PowerShell scripts:
//   - As a continuous foreground or background daemon process with graceful signal handling;
//   - Inside containerized environments (Docker, Podman, Kubernetes);
//   - Or registered into native OS schedulers via `--register` (Task Scheduler on Windows,
//     systemd user units on Linux, launchd agents on macOS).

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	gardenLoopID              = "garden-stale-work-loop"
	gardenLoopDefaultInterval = time.Hour
	gardenLoopSchema          = "fak-garden-loop/1"
	gardenLoopTaskName        = "FleetStaleWorkGarden"
)

type gardenLoopOptions struct {
	Repo            string
	Interval        time.Duration
	Live            bool
	CleanBins       bool
	TreeDoctor      bool
	PackRefs        bool
	MaxAgeDays      int
	WatchdogTimeout time.Duration
	TickBudget      time.Duration
	Iterations      int
	AsJSON          bool
	LedgerPath      string
	RegistryPath    string
}

type gardenLoopCycleReport struct {
	Schema     string                  `json:"schema"`
	Cycle      int                     `json:"cycle"`
	Timestamp  string                  `json:"timestamp"`
	DurationMS int64                   `json:"duration_ms"`
	Live       bool                    `json:"live"`
	Watchdog   *gardenWatchdogEnvelope `json:"watchdog,omitempty"`
	CleanBins  *cleanBinsResult        `json:"clean_bins,omitempty"`
	TreeDoctor []string                `json:"tree_doctor,omitempty"`
	RefsPacked bool                    `json:"refs_packed"`
	Status     string                  `json:"status"`
	Reason     string                  `json:"reason"`
}

func runGardenLoop(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "repo root to garden (default: repo root)")
	workspace := fs.String("workspace", "", "alias for --repo")
	interval := fs.Duration("interval", gardenLoopDefaultInterval, "cadence interval between gardening sweeps (e.g. 30m, 1h)")
	live := fs.Bool("live", true, "perform active mutations (delete over-age ephemera, reap stale leases, clean binaries, prune merged worktrees)")
	cleanBins := fs.Bool("clean-bins", true, "prune stray root-level build binaries each cycle")
	treeDoc := fs.Bool("tree-doctor", true, "prune merged non-live worker worktrees and reap stale commit locks each cycle")
	packRefs := fs.Bool("pack-refs", true, "pack loose git refs each cycle to bound ref-lookup pressure")
	maxAgeDays := fs.Int("max-age-days", 7, "GC gitignored ephemera older than this many days")
	watchdogTimeout := fs.Int("watchdog-timeout", gardenWatchdogTimeoutSeconds, "hard outer watchdog bound in seconds")
	tickBudget := fs.Int("tick-budget", gardenWatchdogTickBudgetSeconds, "whole-child garden tick budget in seconds")
	iterations := fs.Int("iterations", 0, "run N cycles and exit (0 = run continuously until signal)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON per sweep cycle")
	register := fs.Bool("register", false, "register this loop in the durable loop registry and native OS scheduler, then return")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	registry := fs.String("registry", "", "loop registry JSON path (default: the loop registry)")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden loop: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *repo
	if root == "" {
		root = *workspace
	}
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
	registryPath := firstNonEmpty(*registry, defaultLoopRegistry())

	if *interval <= 0 {
		fmt.Fprintf(stderr, "fak garden loop: --interval must be positive, got %v\n", *interval)
		return 2
	}

	if *register {
		return registerGardenPlatformScheduler(stdout, stderr, root, ledgerPath, registryPath, *interval, *live)
	}

	if gardenbundle.GardenOff() {
		fmt.Fprintln(stdout, "garden loop skipped: FAK_GARDEN is off")
		return 0
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	opts := gardenLoopOptions{
		Repo:            root,
		Interval:        *interval,
		Live:            *live,
		CleanBins:       *cleanBins,
		TreeDoctor:      *treeDoc,
		PackRefs:        *packRefs,
		MaxAgeDays:      *maxAgeDays,
		WatchdogTimeout: time.Duration(*watchdogTimeout) * time.Second,
		TickBudget:      time.Duration(*tickBudget) * time.Second,
		Iterations:      *iterations,
		AsJSON:          *asJSON,
		LedgerPath:      ledgerPath,
		RegistryPath:    registryPath,
	}

	return runGardenLoopProcess(ctx, stdout, stderr, opts)
}

func runGardenLoopProcess(ctx context.Context, stdout, stderr io.Writer, opts gardenLoopOptions) int {
	lock, acquired, err := acquireGardenLoopLock(opts.Repo, time.Now(), 2*opts.Interval)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden loop: acquire loop lock: %v\n", err)
		return 1
	}
	if !acquired {
		fmt.Fprintf(stderr, "fak garden loop: another garden loop process is currently active (SKIPPED_CONTENDED)\n")
		return 3
	}
	defer lock.release()

	if !opts.AsJSON {
		mode := "dry-run"
		if opts.Live {
			mode = "LIVE"
		}
		fmt.Fprintf(stdout, "fak garden loop (%s) started: interval=%v repo=%s\n", mode, opts.Interval, opts.Repo)
	}

	cycle := 0
	runCycle := func() bool {
		cycle++
		rep := executeGardenLoopCycle(ctx, opts, cycle)
		emitGardenLoopCycleReport(stdout, stderr, rep, opts.AsJSON)
		witnessGardenLoop(opts.LedgerPath, rep)

		if opts.Iterations > 0 && cycle >= opts.Iterations {
			return false
		}
		return true
	}

	// Execute initial cycle immediately on start.
	if !runCycle() {
		return 0
	}

	ticker := time.NewTicker(opts.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			if !opts.AsJSON {
				fmt.Fprintln(stdout, "\nfak garden loop: shutdown signal received; exiting cleanly.")
			}
			return 0
		case <-ticker.C:
			if !runCycle() {
				return 0
			}
		}
	}
}

func executeGardenLoopCycle(ctx context.Context, opts gardenLoopOptions, cycle int) gardenLoopCycleReport {
	start := time.Now()
	rep := gardenLoopCycleReport{
		Schema:     gardenLoopSchema,
		Cycle:      cycle,
		Timestamp:  start.UTC().Format(time.RFC3339),
		Live:       opts.Live,
		Status:     "complete",
		Reason:     "GARDEN_SWEEP_OK",
		TreeDoctor: []string{},
	}

	// Step 1: Watchdog sweep & child tick.
	var watchdogBuf, watchdogErr bytes.Buffer
	wcfg := gardenWatchdogConfig{
		Repo:           opts.Repo,
		MaxAgeDays:     opts.MaxAgeDays,
		StuckThreshold: 3,
		WIPStaleHours:  24,
		Live:           opts.Live,
		AsJSON:         true,
		Timeout:        opts.WatchdogTimeout,
		TickBudget:     opts.TickBudget,
		Now:            time.Now,
	}
	_ = runGardenWatchdogConfigured(&watchdogBuf, &watchdogErr, wcfg)
	var env gardenWatchdogEnvelope
	if json.Unmarshal(watchdogBuf.Bytes(), &env) == nil {
		rep.Watchdog = &env
	}

	// Step 2: Clean stray build binaries at module root.
	if opts.CleanBins {
		cleanRes := runCleanBins(cleanBinsOptions{
			Root:        opts.Repo,
			Apply:       opts.Live,
			IncludeLive: false,
			CmdDirs:     cmdDirSet(opts.Repo),
			IsIgnored:   func(name string) (bool, error) { return gitPathIgnored(opts.Repo, name) },
		})
		rep.CleanBins = &cleanRes
	}

	// Step 3: Tree doctor (prune merged non-live worktrees & stale commit locks).
	if opts.TreeDoctor {
		docOpts := treedoctor.Options{
			RepoRoot:     opts.Repo,
			Trunk:        "origin/main",
			WIP:          treedoctor.WIPOptions{},
			ProcessAlive: dispatchPIDAlive,
		}
		_, actions := treedoctor.Sweep(ctx, gitRunner, docOpts, opts.Live)
		rep.TreeDoctor = actions
	}

	// Step 4: Pack loose refs if requested and live.
	if opts.PackRefs && opts.Live {
		_, code, err := gitRunner(ctx, opts.Repo, "pack-refs", "--all")
		if err == nil && code == 0 {
			rep.RefsPacked = true
		}
	}

	rep.DurationMS = time.Since(start).Milliseconds()
	return rep
}

func emitGardenLoopCycleReport(stdout, stderr io.Writer, rep gardenLoopCycleReport, asJSON bool) {
	if asJSON {
		_ = encodeJSONOrFail(stdout, stderr, rep, "fak garden loop")
		return
	}

	fmt.Fprintf(stdout, "[cycle %d] %s in %dms (live=%v)\n",
		rep.Cycle, rep.Timestamp, rep.DurationMS, rep.Live)
	if rep.Watchdog != nil {
		fmt.Fprintf(stdout, "  watchdog: age-gc=%d files (%.1f MB), stuck=%d, wip=%d uncommitted (stale=%v)\n",
			rep.Watchdog.AgeGC.Files, float64(rep.Watchdog.AgeGC.Bytes)/1e6,
			len(rep.Watchdog.Stuck), rep.Watchdog.WIP.Count, rep.Watchdog.WIP.Stale)
		if rep.Watchdog.Garden.Invoked {
			fmt.Fprintf(stdout, "  garden tick: status=%s reason=%s elapsed=%dms\n",
				rep.Watchdog.Garden.Status, rep.Watchdog.Garden.Reason, rep.Watchdog.Garden.ElapsedMillis)
		}
	}
	if rep.CleanBins != nil {
		fmt.Fprintf(stdout, "  clean-bins: removed=%d files (%.1f MB freed), skipped=%d\n",
			len(rep.CleanBins.Removed), float64(rep.CleanBins.TotalBytes)/1e6, len(rep.CleanBins.Skipped))
	}
	if len(rep.TreeDoctor) > 0 {
		fmt.Fprintf(stdout, "  tree-doctor: %d action(s) applied:\n", len(rep.TreeDoctor))
		for _, a := range rep.TreeDoctor {
			fmt.Fprintf(stdout, "    - %s\n", a)
		}
	}
	if rep.RefsPacked {
		fmt.Fprintln(stdout, "  git: loose refs packed (--all)")
	}
}

func witnessGardenLoop(ledgerPath string, rep gardenLoopCycleReport) {
	if ledgerPath == "" {
		return
	}
	summary := fmt.Sprintf("garden loop cycle %d: sweep in %dms (live=%v)", rep.Cycle, rep.DurationMS, rep.Live)
	metrics := map[string]int64{
		"cycle":       int64(rep.Cycle),
		"duration_ms": rep.DurationMS,
	}
	if rep.Watchdog != nil {
		metrics["age_gc_files"] = int64(rep.Watchdog.AgeGC.Files)
		metrics["age_gc_bytes"] = rep.Watchdog.AgeGC.Bytes
		metrics["wip_count"] = int64(rep.Watchdog.WIP.Count)
	}
	if rep.CleanBins != nil {
		metrics["clean_bins_removed"] = int64(len(rep.CleanBins.Removed))
		metrics["clean_bins_freed"] = rep.CleanBins.TotalBytes
	}

	runID := fmt.Sprintf("garden-loop-%d-%d", time.Now().UnixNano(), rep.Cycle)
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenLoopID,
		RunID:   runID,
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusClaimedDone,
		Source:  "fak garden loop",
		Summary: summary,
		Metrics: metrics,
	})

	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenLoopID,
		RunID:   runID,
		Kind:    loopmgr.EventWitness,
		Status:  loopmgr.StatusWitnessedDone,
		Reason:  rep.Reason,
		Source:  "fak garden loop",
		Summary: summary,
		Metrics: metrics,
		EvidenceRefs: []loopmgr.EvidenceRef{{
			Kind:    "garden_sweep",
			Ref:     fmt.Sprintf("cycle-%d", rep.Cycle),
			Summary: summary,
		}},
	})
}

// registerGardenPlatformScheduler installs the durable loop in loopmgr and configures the native OS scheduler.
func registerGardenPlatformScheduler(stdout, stderr io.Writer, root, ledgerPath, registryPath string, interval time.Duration, live bool) int {
	// Step 1: Register in loopmgr registry.
	if err := registerGardenLoop(registryPath, gardenLoopID, int64(interval.Seconds()), 300); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: register durable loop: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "registered durable loop %q (interval %ds) in %s\n",
		gardenLoopID, int64(interval.Seconds()), registryPath)

	fakBin := "fak"
	if self, err := os.Executable(); err == nil && self != "" {
		fakBin = self
	}

	// Step 2: Configure the native OS scheduler for this platform.
	switch runtime.GOOS {
	case "windows":
		return registerWindowsTaskScheduler(stdout, stderr, fakBin, root, interval, live)
	case "linux":
		return registerLinuxSystemdTimer(stdout, stderr, fakBin, root, interval, live)
	case "darwin":
		return registerDarwinLaunchdAgent(stdout, stderr, fakBin, root, interval, live)
	default:
		fmt.Fprintf(stdout, "platform %q has no built-in OS scheduler integration; run `fak garden loop` as a service.\n", runtime.GOOS)
		return 0
	}
}

func registerWindowsTaskScheduler(stdout, stderr io.Writer, fakBin, root string, interval time.Duration, live bool) int {
	liveArg := ""
	if live {
		liveArg = " --live"
	}
	taskCmd := fmt.Sprintf("\"%s\" garden watchdog --repo \"%s\"%s", fakBin, root, liveArg)

	hours := int(interval.Hours())
	if hours < 1 {
		hours = 1
	}

	cmd := exec.Command("schtasks", "/Create",
		"/TN", gardenLoopTaskName,
		"/TR", taskCmd,
		"/SC", "HOURLY",
		"/MO", strconv.Itoa(hours),
		"/F",
	)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(stderr, "fak garden loop: register Task Scheduler task %q: %v (%s)\n",
			gardenLoopTaskName, err, strings.TrimSpace(string(out)))
		return 1
	}

	// Also ensure task is enabled.
	enableCmd := exec.Command("schtasks", "/Change", "/TN", gardenLoopTaskName, "/ENABLE")
	windowgate.ConfigureBackgroundCommand(enableCmd)
	_, _ = enableCmd.CombinedOutput()

	fmt.Fprintf(stdout, "installed Windows Scheduled Task %q (every %dh): %s\n",
		gardenLoopTaskName, hours, strings.TrimSpace(string(out)))
	return 0
}

func registerLinuxSystemdTimer(stdout, stderr io.Writer, fakBin, root string, interval time.Duration, live bool) int {
	home, _ := os.UserHomeDir()
	unitDir := filepath.Join(home, ".config", "systemd", "user")
	if err := os.MkdirAll(unitDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: create systemd dir: %v\n", err)
		return 1
	}

	liveArg := ""
	if live {
		liveArg = " --live"
	}

	serviceContent := fmt.Sprintf(`[Unit]
Description=fak garden stale-work watchdog

[Service]
Type=oneshot
WorkingDirectory=%s
ExecStart=%s garden watchdog --repo %s%s
`, root, fakBin, root, liveArg)

	secs := int64(interval.Seconds())
	timerContent := fmt.Sprintf(`[Unit]
Description=Timer for fak garden stale-work watchdog

[Timer]
OnBootSec=5m
OnUnitActiveSec=%ds
Persistent=true

[Install]
WantedBy=timers.target
`, secs)

	servicePath := filepath.Join(unitDir, "fleet-stale-work-garden.service")
	timerPath := filepath.Join(unitDir, "fleet-stale-work-garden.timer")

	if err := os.WriteFile(servicePath, []byte(serviceContent), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: write %s: %v\n", servicePath, err)
		return 1
	}
	if err := os.WriteFile(timerPath, []byte(timerContent), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: write %s: %v\n", timerPath, err)
		return 1
	}

	fmt.Fprintf(stdout, "wrote systemd user units to %s\n  to enable: systemctl --user daemon-reload && systemctl --user enable --now fleet-stale-work-garden.timer\n", unitDir)
	return 0
}

func registerDarwinLaunchdAgent(stdout, stderr io.Writer, fakBin, root string, interval time.Duration, live bool) int {
	home, _ := os.UserHomeDir()
	agentsDir := filepath.Join(home, "Library", "LaunchAgents")
	if err := os.MkdirAll(agentsDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: create LaunchAgents dir: %v\n", err)
		return 1
	}

	args := []string{fakBin, "garden", "watchdog", "--repo", root}
	if live {
		args = append(args, "--live")
	}

	argsXML := ""
	for _, a := range args {
		argsXML += fmt.Sprintf("      <string>%s</string>\n", a)
	}

	secs := int64(interval.Seconds())
	plistContent := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.fleet.stale-work-garden</string>
    <key>ProgramArguments</key>
    <array>
%s    </array>
    <key>StartInterval</key>
    <integer>%d</integer>
    <key>RunAtLoad</key>
    <true/>
  </dict>
</plist>
`, argsXML, secs)

	plistPath := filepath.Join(agentsDir, "com.fleet.stale-work-garden.plist")
	if err := os.WriteFile(plistPath, []byte(plistContent), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak garden loop: write %s: %v\n", plistPath, err)
		return 1
	}

	fmt.Fprintf(stdout, "wrote launchd agent to %s\n  to load: launchctl load -w %s\n", plistPath, plistPath)
	return 0
}

type gardenLoopLock struct{ path string }

func (l *gardenLoopLock) release() {
	if l != nil && l.path != "" {
		_ = os.Remove(l.path)
	}
}

func acquireGardenLoopLock(root string, now time.Time, staleAfter time.Duration) (*gardenLoopLock, bool, error) {
	path := filepath.Join(root, ".dos", "garden", "garden-loop.lock")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, false, err
	}
	for attempt := 0; attempt < 2; attempt++ {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err == nil {
			_, _ = fmt.Fprintf(f, `{"pid":%d,"started_unix":%d}`+"\n", os.Getpid(), now.Unix())
			if cerr := f.Close(); cerr != nil {
				_ = os.Remove(path)
				return nil, false, cerr
			}
			return &gardenLoopLock{path: path}, true, nil
		}
		if !os.IsExist(err) {
			return nil, false, err
		}
		info, statErr := os.Stat(path)
		if statErr != nil {
			if os.IsNotExist(statErr) {
				continue
			}
			return nil, false, statErr
		}
		if staleAfter > 0 && now.Sub(info.ModTime()) > staleAfter {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return nil, false, err
			}
			continue
		}
		return nil, false, nil
	}
	return nil, false, nil
}
