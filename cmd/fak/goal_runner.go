package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalrunner"
)

func runGoalLaunchSubcommand(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("goal launch", flag.ContinueOnError)
	fs.SetOutput(stderr)

	pointer := fs.String("pointer", ".claude/goal-prompts/resolve-top-issue-witnessed.md", "pointer file relative to workspace or absolute")
	workspace := fs.String("workspace", "", "target workspace directory (default: current working directory)")
	product := fs.String("product", "claude", "worker product: claude|codex|opencode")
	workKind := fs.String("work-kind", "engineering", "work kind: engineering|gardening")
	tier := fs.String("tier", "auto", "explicit tier override: auto|t1|t2|t3")
	account := fs.String("account", "", "specific account tag to pin")
	model := fs.String("model", "", "model override for worker (e.g. opus, sonnet)")
	budgetTokens := fs.Int("budget-tokens", 2016000, "cumulative context-budget-tokens for fak guard")
	duration := fs.String("duration", "45m", "max-duration budget for fak guard")
	guarded := fs.Bool("guarded", true, "wrap worker in fak guard gateway")
	fakExe := fs.String("fak-exe", "", "explicit fak binary path")
	allowFallback := fs.Bool("allow-tier-fallback", false, "allow tier fallback if requested tier unavailable")
	planOnly := fs.Bool("plan-only", false, "resolve plan and gates but spawn nothing")
	dryRun := fs.Bool("dry-run", false, "alias for --plan-only")
	asJSON := fs.Bool("json", false, "emit plan or witness as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ws := *workspace
	if ws == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak goal launch: %v\n", err)
			return 1
		}
		ws = cwd
	}

	opts := goalrunner.LaunchOptions{
		Workspace:           ws,
		PointerFile:         *pointer,
		Product:             *product,
		WorkKind:            *workKind,
		Tier:                *tier,
		Account:             *account,
		Model:               *model,
		ContextBudgetTokens: *budgetTokens,
		MaxDuration:         *duration,
		Guarded:             *guarded,
		FakExe:              *fakExe,
		AllowTierFallback:   *allowFallback,
		PlanOnly:            *planOnly || *dryRun,
	}

	result, err := goalrunner.LaunchDetachedWorker(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak goal launch: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(result)
		return 0
	}

	if result.PlanOnly {
		fmt.Fprintf(stdout, "PLAN_ONLY: goal runner prepared (tag=%s, run_id=%s)\n", result.Tag, result.RunID)
		fmt.Fprintf(stdout, "  command: %s\n", result.Command)
		return 0
	}

	fmt.Fprintf(stdout, "LAUNCH_WITNESS pid=%d tag=%s run_id=%s\n", result.PID, result.Tag, result.RunID)
	return 0
}

func runGoalFleetSubcommand(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("goal fleet", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workspace := fs.String("workspace", "", "target workspace directory")
	contractsDir := fs.String("contracts-dir", ".claude/goal-prompts/p0-fleet", "directory containing fleet contracts")
	timeoutMin := fs.Int("timeout-min", 90, "per-worker timeout in minutes")
	dryRun := fs.Bool("dry-run", false, "validate contracts and plan without launching")
	asJSON := fs.Bool("json", false, "emit plan or witness as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ws := *workspace
	if ws == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak goal fleet: %v\n", err)
			return 1
		}
		ws = cwd
	}

	contracts, err := goalrunner.LoadFleetContracts(*contractsDir, ws)
	if err != nil {
		fmt.Fprintf(stderr, "fak goal fleet: %v\n", err)
		return 1
	}

	plan := &goalrunner.FleetPlan{
		Name:                "goal-fleet",
		Workspace:           ws,
		ContractsDir:        *contractsDir,
		Contracts:           contracts,
		PerWorkerTimeoutMin: *timeoutMin,
		PerWorkerTimeout:    time.Duration(*timeoutMin) * time.Minute,
		DryRun:              *dryRun,
	}

	witnesses, err := goalrunner.RunFleetPlan(context.Background(), plan, nil)
	if err != nil {
		fmt.Fprintf(stderr, "fak goal fleet: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(witnesses)
		return 0
	}

	fmt.Fprintf(stdout, "fak goal fleet: completed %d contracts (dry-run=%v)\n", len(witnesses), *dryRun)
	return 0
}
