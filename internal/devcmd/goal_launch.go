package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalrunner"
)

// RunGoalLaunch parses CLI arguments and launches a detached goal worker via goalrunner.LaunchDetachedWorker.
func RunGoalLaunch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("goal-launch", flag.ContinueOnError)
	fs.SetOutput(stderr)

	pointer := fs.String("pointer", "", "path to pointer file or inline goal condition")
	workspace := fs.String("workspace", "", "workspace root directory (default: current working directory)")
	product := fs.String("product", "", "target product profile name")
	workKind := fs.String("work-kind", "engineering", "work kind classifier (e.g. engineering)")
	tier := fs.String("tier", "", "account tier name")
	account := fs.String("account", "", "seat account name")
	model := fs.String("model", "", "model identifier or alias")
	guarded := fs.Bool("guarded", true, "run under fak guard capability floor")
	budgetTokens := fs.Int("budget-tokens", 2016000, "context token budget ceiling")
	duration := fs.String("duration", "45m", "maximum worker duration (e.g. 45m)")
	fakExe := fs.String("fak-exe", "", "explicit path to fak binary")
	planOnly := fs.Bool("plan-only", false, "format and write input/logs without spawning worker")
	dryRun := fs.Bool("dry-run", false, "alias for plan-only mode")
	logDir := fs.String("log-dir", "", "directory for run logs and prompts (default: <workspace>/.goal-runs)")
	tag := fs.String("tag", "", "worker tag / run identifier")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev goal-launch [flags] [pointer-file]")
		fmt.Fprintln(stderr, "flags:")
		fs.PrintDefaults()
	}

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	if *pointer == "" && fs.NArg() > 0 {
		*pointer = fs.Arg(0)
	}

	if strings.TrimSpace(*pointer) == "" {
		fs.Usage()
		return 2
	}

	opt := goalrunner.LaunchOptions{
		Workspace:           *workspace,
		Product:             *product,
		WorkKind:            *workKind,
		Tier:                *tier,
		Account:             *account,
		Model:               *model,
		Guarded:             *guarded,
		RawSpawn:            !*guarded,
		ContextBudgetTokens: *budgetTokens,
		MaxDuration:         *duration,
		FakExe:              *fakExe,
		PlanOnly:            *planOnly || *dryRun,
		LogDir:              *logDir,
		Tag:                 *tag,
	}

	rawPtr := strings.TrimSpace(*pointer)
	// Check if pointer is an existing file or inline condition
	ws := opt.Workspace
	if ws == "" {
		ws, _ = os.Getwd()
	}
	candFile := rawPtr
	if !filepath.IsAbs(candFile) && ws != "" {
		candFile = filepath.Join(ws, candFile)
	}

	if fi, err := os.Stat(candFile); err == nil && !fi.IsDir() {
		opt.PointerFile = rawPtr
	} else if fi, err := os.Stat(rawPtr); err == nil && !fi.IsDir() {
		opt.PointerFile = rawPtr
	} else if strings.Contains(rawPtr, "\n") || strings.Contains(rawPtr, " ") || strings.HasPrefix(rawPtr, "/goal") {
		opt.PointerContent = rawPtr
	} else {
		opt.PointerFile = rawPtr
	}

	res, err := goalrunner.LaunchDetachedWorker(opt)
	if err != nil {
		if *jsonOut {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "goal-launch: %v\n", err)
		}
		return 1
	}

	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(res)
		return 0
	}

	fmt.Fprintf(stdout, "launched goal worker: tag=%s run_id=%s pid=%d\n", res.Tag, res.RunID, res.PID)
	if res.PlanOnly {
		fmt.Fprintf(stdout, "plan-only mode: %s\n", res.LaunchWitness)
	}
	fmt.Fprintf(stdout, "  prompt: %s\n", res.PromptFile)
	fmt.Fprintf(stdout, "  stdout: %s\n", res.OutLog)
	fmt.Fprintf(stdout, "  stderr: %s\n", res.ErrLog)

	return 0
}

// RunGoalFleet parses CLI arguments and runs serialized goal fleet execution via goalrunner.RunFleetPlan.
func RunGoalFleet(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("goal-fleet", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workspace := fs.String("workspace", "", "workspace root path (default: current working directory)")
	contractsDir := fs.String("contracts-dir", "", "path to directory or JSON file containing goal contracts")
	timeoutMin := fs.Int("timeout-min", 90, "per-worker timeout in minutes")
	dryRun := fs.Bool("dry-run", false, "dry run without spawning workers")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev goal-fleet [flags] [contracts-dir]")
		fmt.Fprintln(stderr, "flags:")
		fs.PrintDefaults()
	}

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	if *contractsDir == "" && fs.NArg() > 0 {
		*contractsDir = fs.Arg(0)
	}

	if strings.TrimSpace(*contractsDir) == "" {
		fs.Usage()
		return 2
	}

	ws := *workspace
	if ws == "" {
		ws, _ = os.Getwd()
	}

	contracts, err := goalrunner.LoadFleetContracts(*contractsDir, ws)
	if err != nil {
		if *jsonOut {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "goal-fleet: %v\n", err)
		}
		return 1
	}

	plan := &goalrunner.FleetPlan{
		Name:                "devcmd-goal-fleet",
		Workspace:           ws,
		ContractsDir:        *contractsDir,
		Contracts:           contracts,
		PerWorkerTimeoutMin: *timeoutMin,
		PerWorkerTimeout:    time.Duration(*timeoutMin) * time.Minute,
		DryRun:              *dryRun,
	}

	results, err := goalrunner.RunFleetPlan(context.Background(), plan, nil)
	if err != nil {
		if *jsonOut {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "goal-fleet: %v\n", err)
		}
		return 1
	}

	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(results)
		return 0
	}

	fmt.Fprintf(stdout, "goal-fleet finished: %d contracts\n", len(results))
	for _, r := range results {
		fmt.Fprintf(stdout, "  #%d: %s\n", r.Issue, r.Outcome)
	}
	if plan.RollupPath != "" {
		fmt.Fprintf(stdout, "rollup report: %s\n", plan.RollupPath)
	}

	return 0
}
