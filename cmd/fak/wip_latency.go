package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func runWipLatency(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("wip latency", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", ".", "repository root")
	cFlag := fs.String("C", "", "repository root (alias for --repo)")
	jsonOut := fs.Bool("json", false, "emit JSON report")
	budgetStr := fs.String("budget", "1h", "protection latency budget (e.g. 1h, 30m)")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	budget, err := time.ParseDuration(*budgetStr)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip latency: invalid --budget %q: %v\n", *budgetStr, err)
		return 2
	}
	if budget <= 0 {
		fmt.Fprintln(stderr, "fak wip latency: --budget must be positive")
		return 2
	}

	targetRepo := *repo
	if targetRepo == "." && *cFlag != "" {
		targetRepo = *cFlag
	}
	abs, err := filepath.Abs(targetRepo)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip latency: %v\n", err)
		return 1
	}

	rep, err := wipinventory.MeasureProtectionLatency(context.Background(), abs, wipinventory.LatencyOptions{
		Budget: budget,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak wip latency: %v\n", err)
		return 1
	}

	if *jsonOut {
		b, err := rep.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak wip latency: json marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	protectedCount := int(float64(rep.TotalSourcePaths)*rep.ProtectedWithinBudgetRatio + 0.5)
	fmt.Fprintf(stdout, "PROTECTION LATENCY — SLO verdict: %s (p50=%.1fs p95=%.1fs max=%.1fs, budget %s)\n",
		rep.SLOVerdict, rep.P50Seconds, rep.P95Seconds, rep.MaxSeconds, budget)
	fmt.Fprintf(stdout, "repo: %s\n", rep.Repo)
	fmt.Fprintf(stdout, "protected within budget: %.1f%% (%d/%d source paths)\n",
		rep.ProtectedWithinBudgetRatio*100.0, protectedCount, rep.TotalSourcePaths)
	fmt.Fprintf(stdout, "outcomes: checkpointed=%d worker_isolated=%d landed=%d unprotected=%d\n",
		rep.Outcomes["checkpointed"], rep.Outcomes["worker_isolated"], rep.Outcomes["landed"], rep.Outcomes["unprotected"])
	fmt.Fprintf(stdout, "surfaces: shared_trunk=%d checkpoint=%d detached_worker=%d\n",
		rep.Surfaces["shared_trunk"], rep.Surfaces["checkpoint"], rep.Surfaces["detached_worker"])
	fmt.Fprintf(stdout, "stale refusals: %d, unknown clocks: %d\n",
		rep.StaleRefusalCount, rep.UnknownClockCount)

	return 0
}
