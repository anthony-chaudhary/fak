// procguard.go wires `fak process-guard [scan|report]` — the native port of the
// standalone operator/control-pane modes of tools/proc_resource_guard.py. It is a
// thin CLI shell over internal/procguard: collect a process snapshot, classify it
// (resource-level + the richer CPU-pin / orphan-sprawl modes), and emit either a
// human render or the machine-readable JSON contract (schema
// "fleet-proc-resource-guard/1") the control pane folds. Exit 0 == clean,
// 1 == a runaway is flagged (ACTION), 2 == usage error.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func cmdProcessGuard(args []string) {
	mode := "report"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		mode = args[0]
		args = args[1:]
	}
	switch mode {
	case "scan", "report":
	default:
		fmt.Fprintf(os.Stderr, "fak process-guard: unknown mode %q (want scan|report)\n", mode)
		os.Exit(2)
	}

	fs := flag.NewFlagSet("process-guard", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var (
		maxThreads    = fs.Int("max-threads", procguard.DefaultMaxThreads, "thread-count ceiling (0 disables)")
		maxHandles    = fs.Int("max-handles", procguard.DefaultMaxHandles, "handle-count ceiling (0 disables)")
		maxWSMB       = fs.Int("max-ws-mb", procguard.DefaultMaxWorkingSetMB, "working-set-MB ceiling (0 disables)")
		maxCPUPct     = fs.Float64("max-cpu-pct", procguard.DefaultMaxCPUPct, "sustained per-core CPU%% ceiling (0 disables; report mode only)")
		cpuWindow     = fs.Float64("cpu-window", procguard.DefaultCPUWindowSec, "seconds between consecutive CPU samples")
		cpuSamples    = fs.Int("cpu-samples", procguard.DefaultCPUSamples, "CPU snapshots to take (>=2)")
		cpuConfirm    = fs.Int("cpu-reap-confirm", procguard.DefaultCPUReapConfirm, "reap a CPU-only pin only after N consecutive flagged runs")
		reapOrphans   = fs.Bool("reap-orphans", false, "also flag orphaned ephemeral helpers whose owning session exited")
		reapIdle      = fs.Bool("reap-idle-shells", false, "also flag idle launcher shells with zero live children")
		reapDeadOwner = fs.Bool("reap-dead-owner", false, "also flag fak-owned loop/worker trees whose owning run-lease is dead/absent (report mode; keys on the loop ledger)")
		loopLedger    = fs.String("loop-ledger", "", "loop ledger path for the dead-owner lease lookup (default: the standard loop ledger)")
		idleAgeMin    = fs.Int("idle-shell-age-min", procguard.DefaultIdleShellAgeSec/60, "age floor in minutes for idle-shell flagging")
		enact         = fs.Bool("enact", false, "DESTRUCTIVE: kill flagged non-protected processes (default: report only)")
		asJSON        = fs.Bool("json", false, "emit the machine-readable JSON contract")
		logDir        = fs.String("log-dir", "", "streak-ledger dir (default: tools/_watchdog)")
	)
	var orphanPatterns multiFlag
	fs.Var(&orphanPatterns, "orphan-pattern", "extra name/cmdline substring marking an ephemeral helper (repeatable)")
	var allowNames multiFlag
	fs.Var(&allowNames, "allow", "process name to exempt from flagging (repeatable)")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}

	// scan is the lightweight resource-level snapshot: the CPU dimension and the
	// orphan modes are report-only (they need a second scan / a relation scan).
	// Hung dispatch-orphan detection also needs the CPU sample even when the
	// resource CPU threshold itself is disabled.
	cpuPinEnabled := mode == "report" && *maxCPUPct > 0
	cpuSampled := mode == "report" && (cpuPinEnabled || *reapOrphans)
	orphanEnabled := mode == "report" && (*reapOrphans || len(orphanPatterns) > 0 || *reapIdle)
	deadOwnerEnabled := mode == "report" && *reapDeadOwner

	dir := *logDir
	if dir == "" {
		dir = filepath.Join("tools", "_watchdog")
	}

	var streaksPrev map[string]int
	if cpuPinEnabled {
		streaksPrev = procguard.LoadCPUStreaks(dir)
	}

	var procs []procguard.Proc
	var collectErr string
	if cpuSampled {
		procs, collectErr = procguard.CollectProcessesCPU(*cpuWindow, *cpuSamples, nil)
	} else {
		procs, collectErr = procguard.CollectProcesses()
	}

	protectedPIDs := []int{os.Getpid(), os.Getppid()}

	var orphanRows, deadOwnerRows []procguard.Finding
	if orphanEnabled || deadOwnerEnabled {
		relations, relErr := procguard.CollectRelations()
		if relErr != "" && collectErr == "" {
			collectErr = relErr
		}
		top := procguard.NewRelationTopology(relations)
		if orphanEnabled {
			patterns := append([]string{}, procguard.DefaultOrphanPatterns...)
			patterns = append(patterns, orphanPatterns...)
			minAgeSec := max(0, *idleAgeMin) * 60
			orphanRows = procguard.ClassifyOrphans(
				relations, top.LivePIDs, top.ChildCounts,
				patterns, procguard.DefaultIdleShellNames, minAgeSec, *reapIdle,
				protectedPIDs, allowNames,
			)
			if *reapOrphans {
				if cwd, err := os.Getwd(); err == nil {
					root := findRepoRoot(cwd)
					if _, err := os.Stat(filepath.Join(root, ".git")); err == nil {
						orphanRows = append(orphanRows, classifyHungDispatchOrphans(
							relations,
							procs,
							top,
							dispatchLeasedWorkerPIDs(root),
							minAgeSec,
							protectedPIDs,
							allowNames,
						)...)
					} else {
						fmt.Fprintln(os.Stderr, "fak process-guard: hung dispatch-orphan mode skipped: no repository root, so live leases cannot be proven")
					}
				}
			}
		}
		if deadOwnerEnabled {
			// The lease lookup is fail-closed: if the loop ledger can't be read we
			// cannot prove any owner is dead, so the mode is skipped (nothing flagged)
			// rather than reaping every tagged tree. Report-first; --enact gates the kill.
			leaseAlive, note := loadLeaseLiveness(*loopLedger)
			if note != "" {
				fmt.Fprintf(os.Stderr, "fak process-guard: dead-owner mode skipped: %s\n", note)
			}
			if leaseAlive != nil {
				deadOwnerRows = procguard.ClassifyDeadOwnerOrphans(relations, top, procguard.DeadOwnerOptions{
					LeaseAlive:    leaseAlive,
					ProtectedPIDs: protectedPIDs,
					AllowNames:    allowNames,
				})
			}
		}
	}

	th := procguard.Thresholds{MaxThreads: *maxThreads, MaxHandles: *maxHandles, MaxWSMB: *maxWSMB}
	if cpuPinEnabled {
		th.MaxCPUPct = *maxCPUPct
	}

	var killer func(int) (bool, string)
	if *enact {
		killer = verifiedTreeReaper(procguard.KillPID, dispatchPIDAlive, time.Sleep)
	}

	payload := procguard.Build(procs, procguard.Options{
		Thresholds:     th,
		ProtectedPIDs:  protectedPIDs,
		AllowNames:     allowNames,
		Enact:          *enact,
		CPUReapConfirm: *cpuConfirm,
		CPUStreaksPrev: streaksPrev,
		OrphanRows:     orphanRows,
		DeadOwnerRows:  deadOwnerRows,
		Platform:       runtime.GOOS,
		CollectError:   collectErr,
		Killer:         killer,
	})

	if cpuPinEnabled {
		procguard.SaveCPUStreaks(dir, payload.CPUStreaks)
	}

	if *asJSON {
		out, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Println(string(out))
	} else {
		fmt.Println(renderProcGuard(payload))
	}
	if !payload.OK {
		os.Exit(1)
	}
}

const (
	hungDispatchOrphanKind = "hung-dispatch-orphan"
	idleWorkerCPUCutoff    = 1.0
	rootExitCheckAttempts  = 20
	rootExitCheckInterval  = 100 * time.Millisecond
)

// classifyHungDispatchOrphans closes the third procguard predicate gap: a
// dispatch-marked primary worker whose spawner is gone, whose seat lease is
// absent, and whose sampled CPU has stayed idle past the existing age canary.
// Reusing dispatchIsWorkerCmdline keeps the identity predicate byte-identical to
// dispatch preflight; missing age, CPU, owner, or repo lease evidence skips the
// row rather than risking a false reap.
func classifyHungDispatchOrphans(
	relations []procguard.Proc,
	samples []procguard.Proc,
	top procguard.RelationTopology,
	leasedPIDs map[int]bool,
	minAgeSec int,
	protectedPIDs []int,
	allowNames []string,
) []procguard.Finding {
	metrics := make(map[int]procguard.Proc, len(samples))
	for _, p := range samples {
		if p.PID > 0 {
			metrics[p.PID] = p
		}
	}
	protected := map[int]bool{}
	for _, pid := range protectedPIDs {
		protected[pid] = true
	}
	allow := map[string]bool{}
	for _, name := range allowNames {
		if stem := dispatchProcessNameStem(name); stem != "" {
			allow[stem] = true
		}
	}

	flagged := []procguard.Finding{}
	for _, p := range relations {
		if p.PID <= 0 || leasedPIDs[p.PID] || !dispatchIsWorkerCmdline(p.Cmdline) {
			continue
		}
		stem := dispatchProcessNameStem(p.Name)
		if stem == "" || allow[stem] {
			continue
		}
		if p.PPID == nil || *p.PPID <= 0 || top.LivePIDs[*p.PPID] {
			continue
		}
		if minAgeSec > 0 && (p.AgeSec == nil || *p.AgeSec < minAgeSec) {
			continue
		}
		sample, ok := metrics[p.PID]
		if !ok || sample.CPUPct == nil || *sample.CPUPct < 0 || *sample.CPUPct > idleWorkerCPUCutoff {
			continue
		}

		age := 0
		if p.AgeSec != nil {
			age = *p.AgeSec
		}
		start := sample.Start
		if start == "" {
			start = p.Start
		}
		threads := sample.Threads
		if threads == nil {
			threads = p.Threads
		}
		handles := sample.Handles
		if handles == nil {
			handles = p.Handles
		}
		ws := sample.WSMB
		if ws == nil {
			ws = p.WSMB
		}
		flagged = append(flagged, procguard.Finding{
			PID:        p.PID,
			Name:       strings.TrimSpace(p.Name),
			Threads:    threads,
			Handles:    handles,
			WSMB:       ws,
			CPUPct:     sample.CPUPct,
			PPID:       p.PPID,
			ParentName: top.ParentNames[*p.PPID],
			Start:      start,
			Reasons: []string{fmt.Sprintf(
				"hung dispatch orphan: marker present, owner pid %d not alive, no live lease, cpu %.2f%%, age %ds",
				*p.PPID,
				*sample.CPUPct,
				age,
			)},
			Protected: protected[p.PID] || procguard.ProtectedNames[stem],
			Kind:      hungDispatchOrphanKind,
		})
	}
	sort.Slice(flagged, func(i, j int) bool { return flagged[i].PID < flagged[j].PID })
	return flagged
}

// verifiedTreeReaper prevents a partial tree kill from being reported as a
// success when the root survives (the session-0/elevation failure mode). KillPID
// may return before termination becomes observable, so allow a short bounded
// settling window; a still-live root is an explicit operator-action failure.
func verifiedTreeReaper(
	kill func(int) (bool, string),
	alive func(int) bool,
	sleep func(time.Duration),
) func(int) (bool, string) {
	return func(pid int) (bool, string) {
		ok, detail := kill(pid)
		if !ok {
			return false, detail
		}
		if alive == nil {
			if detail != "" {
				detail += "; "
			}
			return false, detail + "root liveness unavailable"
		}
		for attempt := 0; attempt < rootExitCheckAttempts; attempt++ {
			if !alive(pid) {
				return true, detail
			}
			if attempt+1 < rootExitCheckAttempts && sleep != nil {
				sleep(rootExitCheckInterval)
			}
		}
		if detail != "" {
			detail += "; "
		}
		detail += fmt.Sprintf("target pid %d still alive after reap; access denied or elevation required", pid)
		return false, detail
	}
}

func renderProcGuard(p procguard.Payload) string {
	status := "ok"
	if !p.OK {
		status = "ACTION"
	}
	lines := []string{
		fmt.Sprintf("proc-resource-guard: %s (scanned %d, flagged %d)", status, p.Scanned, p.FlaggedCount),
		fmt.Sprintf("thresholds: max_threads=%d max_handles=%d max_ws_mb=%d max_cpu_pct=%.0f",
			p.Thresholds.MaxThreads, p.Thresholds.MaxHandles, p.Thresholds.MaxWSMB, p.Thresholds.MaxCPUPct),
	}
	for _, row := range p.Flagged {
		tag := row.Action
		if row.Protected {
			tag = "PROTECTED"
		}
		if tag == "" {
			tag = "report"
		}
		kind := ""
		if row.Kind != "" {
			kind = row.Kind + " "
		}
		cpuStr := ""
		if row.CPUPct != nil {
			sfx := ""
			if row.CPUStreak != nil {
				sfx = fmt.Sprintf(" streak=%d", *row.CPUStreak)
			}
			cpuStr = fmt.Sprintf("cpu=%.0f%%/core%s ", *row.CPUPct, sfx)
		}
		lines = append(lines, fmt.Sprintf("  [%s] %spid=%d %s %sthreads=%s handles=%s ws_mb=%s :: %s",
			tag, kind, row.PID, row.Name, cpuStr,
			ptrStr(row.Threads), ptrStr(row.Handles), ptrStr(row.WSMB),
			strings.Join(row.Reasons, ", ")))
	}
	lines = append(lines, "next: "+p.NextAction)
	return strings.Join(lines, "\n")
}

func ptrStr(p *int) string {
	if p == nil {
		return "None"
	}
	return fmt.Sprintf("%d", *p)
}

// loadLeaseLiveness reads the loop ledger and returns a run-id -> alive lookup
// for the dead-owner reaper (procguard.DeadOwnerOptions.LeaseAlive). A run is
// "alive" iff its recovery disposition is RUNNING (a live/recent owner); an
// orphaned, terminal, or absent run id reads as a dead owner. It reuses the same
// fold `fak loop recover` uses (loopmgr.LoadPrefix -> foldRuns -> looprecover.Plan),
// and is ledger-only (no pid probe) — the same v1 stance as loop recover.
//
// Fail-closed: on any read error it returns (nil, note) so the caller SKIPS the
// mode entirely rather than treating an unreadable registry as "every owner dead".
func loadLeaseLiveness(ledger string) (func(string) bool, string) {
	if strings.TrimSpace(ledger) == "" {
		ledger = defaultLoopLedger()
	}
	// Existence is checked explicitly BEFORE LoadPrefix: LoadPrefix treats a
	// missing ledger as an empty (no-error) run set, which would mark EVERY tagged
	// tree's owner dead — the exact false-reap the contract forbids. A missing or
	// unreadable ledger means we cannot prove any owner is dead, so skip the mode.
	if _, err := os.Stat(ledger); err != nil {
		return nil, fmt.Sprintf("loop ledger %q not readable (%v); no lease liveness, nothing flagged", ledger, err)
	}
	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		return nil, fmt.Sprintf("loop ledger %q unreadable (%v); no lease liveness, nothing flagged", ledger, err)
	}
	res := looprecover.Plan(looprecover.Input{
		Runs:    foldRuns(events),
		NowUnix: time.Now().Unix(),
	})
	alive := map[string]bool{}
	for _, r := range res.Runs {
		if r.Disposition == looprecover.DispRunning {
			alive[r.RunID] = true
		}
	}
	return func(runID string) bool { return alive[runID] }, ""
}
