package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ops"
)

func runOps(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak ops <status|sweep|log|daemon> [options]")
		return 2
	}

	verb := argv[0]
	args := argv[1:]

	root := discoverRepoRoot()
	cfg := ops.DefaultConfig()

	switch verb {
	case "status":
		return runOpsStatus(stdout, stderr, root, cfg, args)
	case "sweep":
		return runOpsSweep(stdout, stderr, root, cfg, args)
	case "log":
		return runOpsLog(stdout, stderr, root, args)
	case "daemon":
		return runOpsDaemon(stdout, stderr, root, cfg, args)
	default:
		fmt.Fprintf(stderr, "fak ops: unknown command %q\n", verb)
		return 2
	}
}

type opsStatusReport struct {
	FreeDiskBytes uint64 `json:"free_disk_bytes"`
	WarningFree   bool   `json:"warning_free"`
	RefuseFree    bool   `json:"refuse_free"`
	RecentEvents  int    `json:"recent_events"`
	LastEventTime string `json:"last_event_time,omitempty"`
	Status        string `json:"status"`
}

func runOpsStatus(stdout, stderr io.Writer, root string, cfg ops.Config, args []string) int {
	fs := flag.NewFlagSet("ops status", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit JSON format")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	engine, err := ops.NewEngine(root, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "ops status: init engine: %v\n", err)
		return 1
	}

	events, _ := engine.Ledger.QueryEvents(24 * time.Hour)
	var lastTime string
	if len(events) > 0 {
		lastTime = events[len(events)-1].Timestamp.Format(time.RFC3339)
	}

	rep := opsStatusReport{
		RecentEvents:  len(events),
		LastEventTime: lastTime,
		Status:        "healthy",
	}

	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(rep)
	} else {
		fmt.Fprintf(stdout, "fak ops: status=%s recent_events_24h=%d last_event=%s\n", rep.Status, rep.RecentEvents, rep.LastEventTime)
	}
	return 0
}

func runOpsSweep(stdout, stderr io.Writer, root string, cfg ops.Config, args []string) int {
	fs := flag.NewFlagSet("ops sweep", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", false, "report what would be cleaned without executing mutations")
	asJSON := fs.Bool("json", false, "emit JSON format")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	engine, err := ops.NewEngine(root, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "ops sweep: init engine: %v\n", err)
		return 1
	}

	ctx := context.Background()
	start := time.Now()
	healRes, _ := engine.Workspace.SweepLocksAndWorktrees(ctx, *dryRun)
	reclaimRes, _ := engine.Storage.ReclaimCascade(ctx, *dryRun)
	procRes, _ := engine.Process.SweepProcessRunaways(ctx, *dryRun)
	dur := time.Since(start).Milliseconds()

	if !*dryRun {
		if len(healRes.LocksEvicted) > 0 {
			_ = engine.Ledger.Record(ops.Event{
				ActionType: ops.ActionLockEvict,
				Details:    fmt.Sprintf("evicted dead locks: %s", strings.Join(healRes.LocksEvicted, ", ")),
				DurationMS: dur,
			})
		}
		if len(healRes.WorktreesPruned) > 0 {
			_ = engine.Ledger.Record(ops.Event{
				ActionType: ops.ActionWorktreePrune,
				Details:    fmt.Sprintf("pruned cold worktrees: %s", strings.Join(healRes.WorktreesPruned, ", ")),
				DurationMS: dur,
			})
		}
		if len(procRes.PIDsReaped) > 0 {
			_ = engine.Ledger.Record(ops.Event{
				ActionType:   ops.ActionProcessReap,
				Details:      fmt.Sprintf("reaped %d processes: %s", len(procRes.PIDsReaped), strings.Join(procRes.Names, ", ")),
				PIDsAffected: procRes.PIDsReaped,
				DurationMS:   dur,
			})
		}
		if reclaimRes.TotalBytes > 0 {
			_ = engine.Ledger.Record(ops.Event{
				ActionType:     ops.ActionStorageReclaim,
				Details:        fmt.Sprintf("reclaimed %d bytes across %d files (%s)", reclaimRes.TotalBytes, reclaimRes.FilesCount, strings.Join(reclaimRes.Actions, "; ")),
				BytesReclaimed: reclaimRes.TotalBytes,
				DurationMS:     dur,
			})
		}
	}

	result := struct {
		DryRun         bool     `json:"dry_run"`
		LocksEvicted   []string `json:"locks_evicted"`
		WorktreePruned []string `json:"worktrees_pruned"`
		BytesReclaimed int64    `json:"bytes_reclaimed"`
		FilesCount     int      `json:"files_count"`
		PIDsReaped     []int    `json:"pids_reaped"`
	}{
		DryRun:         *dryRun,
		LocksEvicted:   healRes.LocksEvicted,
		WorktreePruned: healRes.WorktreesPruned,
		BytesReclaimed: reclaimRes.TotalBytes,
		FilesCount:     reclaimRes.FilesCount,
		PIDsReaped:     procRes.PIDsReaped,
	}

	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(result)
	} else {
		fmt.Fprintf(stdout, "fak ops sweep (dry-run=%v): reclaimed=%d bytes (%d files), locks_evicted=%d, worktrees_pruned=%d, pids_reaped=%d\n",
			result.DryRun, result.BytesReclaimed, result.FilesCount, len(result.LocksEvicted), len(result.WorktreePruned), len(result.PIDsReaped))
	}
	return 0
}

func runOpsLog(stdout, stderr io.Writer, root string, args []string) int {
	fs := flag.NewFlagSet("ops log", flag.ContinueOnError)
	sinceStr := fs.String("since", "24h", "duration to query (e.g. 1h, 24h, 7d)")
	asJSON := fs.Bool("json", false, "emit JSON lines")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	dur, err := time.ParseDuration(*sinceStr)
	if err != nil {
		dur = 24 * time.Hour
	}

	ledPath := ops.DefaultLedgerPath(root)
	led, err := ops.OpenLedger(ledPath)
	if err != nil {
		fmt.Fprintf(stderr, "ops log: open ledger: %v\n", err)
		return 1
	}

	events, err := led.QueryEvents(dur)
	if err != nil {
		fmt.Fprintf(stderr, "ops log: query events: %v\n", err)
		return 1
	}

	for _, ev := range events {
		if *asJSON {
			data, _ := json.Marshal(ev)
			fmt.Fprintln(stdout, string(data))
		} else {
			fmt.Fprintf(stdout, "[%s] %s: %s (reclaimed: %d bytes, pids: %v, dur: %dms)\n",
				ev.Timestamp.Format("2006-01-02 15:04:05"), ev.ActionType, ev.Details, ev.BytesReclaimed, ev.PIDsAffected, ev.DurationMS)
		}
	}
	return 0
}

func runOpsDaemon(stdout, stderr io.Writer, root string, cfg ops.Config, args []string) int {
	fs := flag.NewFlagSet("ops daemon", flag.ContinueOnError)
	interval := fs.Duration("interval", 60*time.Second, "ops loop tick interval")
	once := fs.Bool("once", false, "run one tick and exit")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	engine, err := ops.NewEngine(root, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "ops daemon: init engine: %v\n", err)
		return 1
	}

	ctx := context.Background()
	if *once {
		if err := engine.Tick(ctx, true); err != nil {
			fmt.Fprintf(stderr, "ops daemon tick: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, "fak ops: completed one full maintenance tick")
		return 0
	}

	fmt.Fprintf(stdout, "fak ops: daemon started, ticking every %v\n", *interval)
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			_ = engine.Tick(ctx, false)
		}
	}
}
