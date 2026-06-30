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
	"strings"

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
		maxThreads   = fs.Int("max-threads", procguard.DefaultMaxThreads, "thread-count ceiling (0 disables)")
		maxHandles   = fs.Int("max-handles", procguard.DefaultMaxHandles, "handle-count ceiling (0 disables)")
		maxWSMB      = fs.Int("max-ws-mb", procguard.DefaultMaxWorkingSetMB, "working-set-MB ceiling (0 disables)")
		maxCPUPct    = fs.Float64("max-cpu-pct", procguard.DefaultMaxCPUPct, "sustained per-core CPU%% ceiling (0 disables; report mode only)")
		cpuWindow    = fs.Float64("cpu-window", procguard.DefaultCPUWindowSec, "seconds between consecutive CPU samples")
		cpuSamples   = fs.Int("cpu-samples", procguard.DefaultCPUSamples, "CPU snapshots to take (>=2)")
		cpuConfirm   = fs.Int("cpu-reap-confirm", procguard.DefaultCPUReapConfirm, "reap a CPU-only pin only after N consecutive flagged runs")
		reapOrphans  = fs.Bool("reap-orphans", false, "also flag orphaned ephemeral helpers whose owning session exited")
		reapIdle     = fs.Bool("reap-idle-shells", false, "also flag idle launcher shells with zero live children")
		idleAgeMin   = fs.Int("idle-shell-age-min", procguard.DefaultIdleShellAgeSec/60, "age floor in minutes for idle-shell flagging")
		enact        = fs.Bool("enact", false, "DESTRUCTIVE: kill flagged non-protected processes (default: report only)")
		asJSON       = fs.Bool("json", false, "emit the machine-readable JSON contract")
		logDir       = fs.String("log-dir", "", "streak-ledger dir (default: tools/_watchdog)")
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
	cpuEnabled := mode == "report" && *maxCPUPct > 0
	orphanEnabled := mode == "report" && (*reapOrphans || len(orphanPatterns) > 0 || *reapIdle)

	dir := *logDir
	if dir == "" {
		dir = filepath.Join("tools", "_watchdog")
	}

	var streaksPrev map[string]int
	if cpuEnabled {
		streaksPrev = procguard.LoadCPUStreaks(dir)
	}

	var procs []procguard.Proc
	var collectErr string
	if cpuEnabled {
		procs, collectErr = procguard.CollectProcessesCPU(*cpuWindow, *cpuSamples, nil)
	} else {
		procs, collectErr = procguard.CollectProcesses()
	}

	protectedPIDs := []int{os.Getpid(), os.Getppid()}

	var orphanRows []procguard.Finding
	if orphanEnabled {
		patterns := append([]string{}, procguard.DefaultOrphanPatterns...)
		patterns = append(patterns, orphanPatterns...)
		relations, relErr := procguard.CollectRelations()
		if relErr != "" && collectErr == "" {
			collectErr = relErr
		}
		livePIDs := map[int]bool{}
		for _, r := range relations {
			if r.PID > 0 {
				livePIDs[r.PID] = true
			}
		}
		orphanRows = procguard.ClassifyOrphans(
			relations, livePIDs, procguard.ChildCounts(relations),
			patterns, procguard.DefaultIdleShellNames, max(0, *idleAgeMin)*60, *reapIdle,
			protectedPIDs, allowNames,
		)
	}

	th := procguard.Thresholds{MaxThreads: *maxThreads, MaxHandles: *maxHandles, MaxWSMB: *maxWSMB}
	if cpuEnabled {
		th.MaxCPUPct = *maxCPUPct
	}

	var killer func(int) (bool, string)
	if *enact {
		killer = procguard.KillPID
	}

	payload := procguard.Build(procs, procguard.Options{
		Thresholds:     th,
		ProtectedPIDs:  protectedPIDs,
		AllowNames:     allowNames,
		Enact:          *enact,
		CPUReapConfirm: *cpuConfirm,
		CPUStreaksPrev: streaksPrev,
		OrphanRows:     orphanRows,
		Platform:       runtime.GOOS,
		CollectError:   collectErr,
		Killer:         killer,
	})

	if cpuEnabled {
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
