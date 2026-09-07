package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issueorchestrator"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type SpawnedChatRecord struct {
	IssueNumber  int      `json:"issue_number"`
	Key          string   `json:"key"`
	Title        string   `json:"title"`
	Lane         string   `json:"lane"`
	SessionTitle string   `json:"session_title"`
	PID          int      `json:"pid"`
	Status       string   `json:"status"` // "dry_run", "spawned", "error"
	Worktree     string   `json:"worktree,omitempty"`
	LogFile      string   `json:"log_file,omitempty"`
	Command      []string `json:"command"`
	Error        string   `json:"error,omitempty"`
}

type OpencodeSpawnReceipt struct {
	Schema       string              `json:"schema"`
	Workspace    string              `json:"workspace"`
	WaveID       string              `json:"wave_id"`
	WaveIndex    int                 `json:"wave_index"`
	TotalSpawned int                 `json:"total_spawned"`
	DryRun       bool                `json:"dry_run"`
	Chats        []SpawnedChatRecord `json:"chats"`
}

const opencodeSpawnReceiptSchema = "fak.issue-orchestrator-opencode-spawn.v1"

var newWorkerSupervisorFunc = issueorchestrator.NewWorkerSupervisor

func cmdIssueOrchestrator(argv []string) {
	os.Exit(runIssueOrchestrator(os.Stdout, os.Stderr, argv))
}

func runIssueOrchestrator(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "adjust-steps" {
		return runIssueOrchestratorAdjustSteps(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("fak issue-orchestrator", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	fromIssues := fs.String("from-issues", "", "path to GitHub issue JSON or - for stdin (gh issue list --json ...)")
	fromPlan := fs.String("from-plan", "", "path to candidate plan JSON")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit plan markdown")
	waveSize := fs.Int("wave-size", 4, "maximum concurrent workers per wave")
	maxWaves := fs.Int("max-waves", 0, "maximum number of waves to plan (0 = all necessary)")
	targetIssues := fs.Int("target-issues", 0, "campaign target number of issues to resolve")
	var targetPoints int
	fs.IntVar(&targetPoints, "target-points", 0, "campaign target step budget points to retire")
	fs.IntVar(&targetPoints, "points", 0, "alias for --target-points")
	var topLimit int
	fs.IntVar(&topLimit, "top", 0, "limit evaluation to the top N candidate issues")
	fs.IntVar(&topLimit, "limit", 0, "alias for --top")
	autoExpand := fs.Bool("auto-expand", true, "dynamically expand discovery window when candidates are unplannable")
	minWindow := fs.Int("min-window", 20, "minimum candidate discovery window")
	maxWindow := fs.Int("max-window", 500, "maximum candidate discovery window")
	unplannableWarnRatio := fs.Float64("unplannable-warn-ratio", 0.65, "threshold ratio of non-dispatchable issues to trigger advisory warning")
	adaptiveConcurrency := fs.Bool("adaptive-concurrency", true, "derive wave size dynamically from host resources and SQLite contention")
	live := fs.Bool("live", false, "fetch issues directly from GitHub via live ingestion")
	view := fs.String("view", "", "view slug from .github/issue-views.json (default: ready-leaves)")
	pageSize := fs.Int("page-size", 50, "page size for dynamic live ingestion")
	rateLimitTimeout := fs.Duration("rate-limit-timeout", 30*time.Second, "timeout for rate-limit backoff before soft degradation")
	_ = fs.Bool("plan-waves", false, "plan concurrent-safe waves (default behavior; accepted for CLI compatibility)")
	excludeIssuesStr := fs.String("exclude-issues", "", "comma-separated list of issue numbers to exclude")
	excludeLanes := fs.String("exclude-lanes", "", "comma-separated list of lanes to exclude")
	noDetectHeld := fs.Bool("no-detect-held", false, "disable auto-detection of currently held leases in .dos")
	comparePath := fs.String("compare", "", "compare against a prior --json baseline payload")
	check := fs.Bool("check", false, "gate mode: exit non-zero if active dispatchable issues remain")
	subdivideOnly := fs.Bool("subdivide", false, "show only the subdivide queue (epics needing decomposition)")
	triageOnly := fs.Bool("triage", false, "show only the triage queue (issues needing scope clarification)")

	spawnOpencode := fs.Bool("spawn-opencode", false, "spawn fresh OpenCode chat sessions for planned wave issues")
	spawnWave := fs.Int("spawn-wave", 1, "1-based wave sequence number to spawn")
	opencodeCommands := fs.Bool("opencode-commands", false, "include ready-to-run opencode commands in JSON/Markdown plan")
	model := fs.String("model", "", "model for OpenCode chats (-m)")
	agent := fs.String("agent", "", "agent profile for OpenCode chats (--agent)")
	variant := fs.String("variant", "high", "reasoning effort variant for OpenCode chats (--variant, default: high)")
	interactive := fs.Bool("interactive", false, "spawn interactive chat mode (-i) instead of headless run")
	worktree := fs.Bool("worktree", false, "prepare detached worker worktrees for each spawned chat")
	dryRun := fs.Bool("dry-run", false, "preview OpenCode chat spawn commands without executing")
	logDir := fs.String("log-dir", "", "directory for OpenCode session logs (default: .dispatch-runs)")
	supervise := fs.Bool("supervise", true, "enables adaptive process supervision for spawned OpenCode chats")

	harvest := fs.Bool("harvest", false, "trigger harvest and progressive reconciliation of wave runs")
	autoLand := fs.Bool("auto-land", false, "automatically land verified cleared leaves")
	minClearRate := fs.Float64("min-clear-rate", 0.0, "minimum clear rate threshold")
	receipt := fs.String("receipt", "", "path to wave receipt JSON (defaults to latest in .dispatch-runs or workspace)")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issue-orchestrator: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	if *harvest {
		receiptPath := *receipt
		if receiptPath == "" {
			receiptPath = findLatestReceipt(root)
		}
		if receiptPath == "" {
			fmt.Fprintf(stderr, "fak issue-orchestrator: no wave receipt found in %s\n", filepath.Join(root, ".dispatch-runs"))
			return 2
		}
		if !filepath.IsAbs(receiptPath) {
			if _, err := os.Stat(receiptPath); os.IsNotExist(err) && root != "" {
				candidate := filepath.Join(root, receiptPath)
				if _, err2 := os.Stat(candidate); err2 == nil {
					receiptPath = candidate
				}
			}
		}
		opts := issueorchestrator.HarvestOptions{
			WaveReceiptPath: receiptPath,
			Workspace:       root,
			MinClearRate:    *minClearRate,
			AutoLand:        *autoLand,
		}
		harvestReceipt, err := issueorchestrator.ReconcileWave(opts)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: harvest: %v\n", err)
			return 1
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, harvestReceipt); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "=== Wave Harvest Reconciliation: %s ===\n", harvestReceipt.WaveID)
		fmt.Fprintf(stdout, "Total Leaves:   %d\n", harvestReceipt.TotalLeaves)
		fmt.Fprintf(stdout, "Cleared:        %d (%.1f%%)\n", harvestReceipt.ClearedCount, harvestReceipt.ClearRate*100)
		fmt.Fprintf(stdout, "Residual:       %d\n", harvestReceipt.ResidualCount)
		fmt.Fprintf(stdout, "Quiet:          %d\n", harvestReceipt.QuietCount)
		fmt.Fprintf(stdout, "Stalled:        %d\n", harvestReceipt.StalledCount)
		if len(harvestReceipt.LandedSHAs) > 0 {
			fmt.Fprintf(stdout, "Landed Commits: %d\n", len(harvestReceipt.LandedSHAs))
			for _, sha := range harvestReceipt.LandedSHAs {
				fmt.Fprintf(stdout, "  - %s\n", sha)
			}
		}
		if len(harvestReceipt.Leaves) > 0 {
			fmt.Fprintln(stdout, "\nLeaves:")
			for _, l := range harvestReceipt.Leaves {
				fmt.Fprintf(stdout, "  - #%d [%s]: %s (state: %s)\n", l.IssueNumber, l.Lane, l.Title, l.State)
			}
		}
		if len(harvestReceipt.ReviewQueue) > 0 {
			fmt.Fprintf(stdout, "\nReview Queue (%d leaves):\n", len(harvestReceipt.ReviewQueue))
			for _, rq := range harvestReceipt.ReviewQueue {
				fmt.Fprintf(stdout, "  - #%d: %s\n", rq.IssueNumber, rq.Title)
			}
		}
		if len(harvestReceipt.AdvisoryLogs) > 0 {
			fmt.Fprintln(stdout, "\nAdvisories:")
			for _, adv := range harvestReceipt.AdvisoryLogs {
				fmt.Fprintf(stdout, "  %s\n", adv)
			}
		}
		return 0
	}

	// 1. Load issues from input
	var issues []issueorchestrator.Issue
	var err error

	if *live {
		liveOpts := issueorchestrator.LiveIngestOptions{
			Workspace:        root,
			View:             *view,
			PageSize:         *pageSize,
			RateLimitTimeout: *rateLimitTimeout,
			TargetIssues:     *targetIssues,
			LogWarning: func(w string) {
				fmt.Fprintln(stderr, w)
			},
		}
		issues, err = issueorchestrator.FetchLiveIssues(context.Background(), liveOpts)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: live ingest: %v\n", err)
			return 2
		}
	} else {
		inputPath := *fromIssues
		if inputPath == "" {
			inputPath = *fromPlan
		}

		issues, err = issueorchestrator.LoadIssues(inputPath, root)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: %v\n", err)
			return 2
		}
	}
	if !*autoExpand && topLimit > 0 && len(issues) > topLimit {
		issues = issues[:topLimit]
	}

	// 2. Parse exclusion flags
	var excludedIssues []int
	if *excludeIssuesStr != "" {
		for _, part := range strings.Split(*excludeIssuesStr, ",") {
			trimmed := strings.TrimSpace(part)
			trimmed = strings.TrimPrefix(trimmed, "#")
			if num, err := strconv.Atoi(trimmed); err == nil && num > 0 {
				excludedIssues = append(excludedIssues, num)
			}
		}
	}

	var excludedLanesList []string
	if *excludeLanes != "" {
		for _, part := range strings.Split(*excludeLanes, ",") {
			trimmed := strings.TrimSpace(part)
			if trimmed != "" {
				excludedLanesList = append(excludedLanesList, trimmed)
			}
		}
	}

	// 3. Generate wave plan
	effectiveWaveSize := *waveSize
	if *adaptiveConcurrency {
		effectiveWaveSize = issueorchestrator.AdaptiveWaveSize(issueorchestrator.WavePlanOptions{WaveSize: *waveSize})
	}

	waveOpts := issueorchestrator.WavePlanOptions{
		WaveSize:             effectiveWaveSize,
		MaxWaves:             *maxWaves,
		TargetIssues:         *targetIssues,
		TargetPoints:         targetPoints,
		Limit:                topLimit,
		ExcludedIssues:       excludedIssues,
		ExcludedLanes:        excludedLanesList,
		AutoDetectHeld:       !*noDetectHeld,
		WorkspaceRoot:        root,
		AutoExpand:           *autoExpand,
		MinWindow:            *minWindow,
		MaxWindow:            *maxWindow,
		UnplannableWarnRatio: *unplannableWarnRatio,
	}
	if *spawnOpencode || *opencodeCommands {
		waveOpts.IncludeOpencodeCommands = true
		waveOpts.OpencodeOptions = issueorchestrator.OpencodeChatOptions{
			Model:       *model,
			Agent:       *agent,
			Variant:     *variant,
			Interactive: *interactive,
			AutoApprove: true,
			PrintLogs:   true,
		}
	}
	plan := issueorchestrator.PlanWaves(issues, waveOpts)
	if plan.Diagnostics != nil && len(plan.Diagnostics.AdvisoryWarnings) > 0 {
		for _, warn := range plan.Diagnostics.AdvisoryWarnings {
			fmt.Fprintln(stderr, warn)
		}
	}

	// 4. Handle baseline comparison if requested
	if *comparePath != "" {
		baseBytes, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: read compare baseline: %v\n", err)
			return 2
		}
		var base issueorchestrator.Plan
		if err := json.Unmarshal(baseBytes, &base); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: decode compare baseline JSON: %v\n", err)
			return 2
		}

		if *asJSON {
			cmpRes := issueorchestrator.Compare(plan, base)
			if err := writeIndentedJSON(stdout, cmpRes); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}

		fmt.Fprint(stdout, issueorchestrator.CompareReport(plan, base))
		if *check && plan.PlannedIssues > 0 {
			return 1
		}
		return 0
	}

	// 5. Handle queue filters
	if *subdivideOnly {
		if *asJSON {
			if err := writeIndentedJSON(stdout, plan.Subdivide); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		if len(plan.Subdivide) == 0 {
			fmt.Fprintln(stdout, "No issues in subdivide queue.")
			return 0
		}
		fmt.Fprintf(stdout, "Subdivide Queue (%d epics requiring decomposition before dispatch):\n", len(plan.Subdivide))
		for _, s := range plan.Subdivide {
			fmt.Fprintf(stdout, "  - #%d: %s (steps: %d, child budget: %d)\n", s.IssueNumber, s.Title, s.ExpectedSteps, s.ChildIssueBudget)
		}
		return 0
	}

	if *triageOnly {
		if *asJSON {
			if err := writeIndentedJSON(stdout, plan.Triage); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		if len(plan.Triage) == 0 {
			fmt.Fprintln(stdout, "No issues in triage queue.")
			return 0
		}
		fmt.Fprintf(stdout, "Triage Queue (%d issues requiring scope/acceptance repair):\n", len(plan.Triage))
		for _, t := range plan.Triage {
			fmt.Fprintf(stdout, "  - #%d: %s [%s]\n", t.IssueNumber, t.Title, t.Dispatchability)
		}
		return 0
	}

	// 6. Handle OpenCode chat spawning if requested
	if *spawnOpencode {
		if len(plan.Waves) == 0 {
			receipt := OpencodeSpawnReceipt{
				Schema:       opencodeSpawnReceiptSchema,
				Workspace:    root,
				WaveIndex:    *spawnWave,
				TotalSpawned: 0,
				DryRun:       *dryRun,
				Chats:        []SpawnedChatRecord{},
			}
			if *asJSON {
				if err := writeIndentedJSON(stdout, receipt); err != nil {
					fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
					return 1
				}
			} else {
				fmt.Fprintln(stdout, "No planned waves available to spawn OpenCode chats.")
			}
			return 0
		}

		targetWaveIdx := *spawnWave - 1
		if targetWaveIdx < 0 || targetWaveIdx >= len(plan.Waves) {
			fmt.Fprintf(stderr, "fak issue-orchestrator: spawn wave %d out of range (1..%d)\n", *spawnWave, len(plan.Waves))
			return 2
		}

		selectedWave := plan.Waves[targetWaveIdx]
		receipt := OpencodeSpawnReceipt{
			Schema:    opencodeSpawnReceiptSchema,
			Workspace: root,
			WaveID:    selectedWave.ID,
			WaveIndex: *spawnWave,
			DryRun:    *dryRun,
			Chats:     make([]SpawnedChatRecord, 0, len(selectedWave.Issues)),
		}

		hasError := false
		for issueIdx, issue := range selectedWave.Issues {
			if issueIdx > 0 && !*dryRun {
				res := issueorchestrator.CurrentSystemResources("")
				delay := issueorchestrator.CalculateSpawnDelay(res, func(w string) {
					fmt.Fprintln(stderr, w)
				})
				time.Sleep(delay)
			}
			var wtDir string
			if *worktree {
				res := workerworktree.Prepare(root, issue.Lane, strconv.Itoa(issue.Number), "", "", nil)
				if res.OK {
					wtDir = res.Path
				}
			}

			chatOpts := issueorchestrator.OpencodeChatOptions{
				Model:       *model,
				Agent:       *agent,
				Variant:     *variant,
				Interactive: *interactive,
				WorktreeDir: wtDir,
				AutoApprove: true,
				PrintLogs:   true,
			}
			chat := issueorchestrator.BuildOpencodeChat(issue, chatOpts)

			record := SpawnedChatRecord{
				IssueNumber:  issue.Number,
				Key:          issue.Key,
				Title:        issue.Title,
				Lane:         issue.Lane,
				SessionTitle: chat.SessionTitle,
				Worktree:     chat.Worktree,
				Command:      chat.Command,
			}

			if *dryRun {
				record.Status = "dry_run"
				receipt.Chats = append(receipt.Chats, record)
				receipt.TotalSpawned++
				continue
			}

			targetLogDir := *logDir
			if targetLogDir == "" {
				targetLogDir = filepath.Join(root, ".dispatch-runs")
			}
			if err := os.MkdirAll(targetLogDir, 0o755); err != nil {
				record.Status = "error"
				record.Error = fmt.Sprintf("create log dir: %v", err)
				receipt.Chats = append(receipt.Chats, record)
				hasError = true
				continue
			}

			stamp := time.Now().UTC().Format("20060102-150405")
			logFile := filepath.Join(targetLogDir, fmt.Sprintf("resolve-%d-%s.log", issue.Number, stamp))
			fh, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				record.Status = "error"
				record.Error = fmt.Sprintf("open log file: %v", err)
				receipt.Chats = append(receipt.Chats, record)
				hasError = true
				continue
			}

			fmt.Fprintf(fh, "# fak-spawn %s issue=%d lane=%s backend=opencode\n", stamp, issue.Number, issue.Lane)

			exe := resolveDispatchWorkerExecutable("opencode", "opencode")
			cmdArgs := []string{}
			if len(chat.Command) > 1 {
				cmdArgs = chat.Command[1:]
			}
			cmd := exec.Command(exe, cmdArgs...)
			if chat.Worktree != "" {
				cmd.Dir = chat.Worktree
				cmd.Env = envSliceFromMap(workerworktree.WorktreeEnv(envMap(os.Environ()), chat.Worktree))
			} else {
				cmd.Dir = root
			}

			cmd.Stdout = fh
			cmd.Stderr = fh

			if !*interactive {
				devNull, err := os.Open(os.DevNull)
				if err == nil {
					defer devNull.Close()
					cmd.Stdin = devNull
				}
			}

			configureDispatchSpawn(cmd)
			configureDispatchWorkerConsole(cmd, "opencode")

			if err := cmd.Start(); err != nil {
				_ = fh.Close()
				record.Status = "error"
				record.Error = err.Error()
				record.LogFile = logFile
				receipt.Chats = append(receipt.Chats, record)
				hasError = true
				continue
			}

			_ = fh.Close()
			pid := cmd.Process.Pid
			pidStem := strings.TrimSuffix(logFile, ".log")
			_ = os.WriteFile(pidStem+".pid", []byte(strconv.Itoa(pid)), 0o644)
			if chat.Worktree != "" {
				_ = workerworktree.HandoffOwner(chat.Worktree, pid)
			}
			_ = cmd.Process.Release()

			if *supervise {
				supCfg := issueorchestrator.WorkerSupervisorConfig{
					PID:         pid,
					WorktreeDir: chat.Worktree,
					LogFile:     logFile,
				}
				_ = newWorkerSupervisorFunc(supCfg)
			}

			record.PID = pid
			record.Status = "spawned"
			record.LogFile = logFile
			receipt.Chats = append(receipt.Chats, record)
			receipt.TotalSpawned++
		}

		if *asJSON {
			if err := writeIndentedJSON(stdout, receipt); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
		} else {
			statusStr := "LIVE"
			if receipt.DryRun {
				statusStr = "DRY RUN"
			}
			fmt.Fprintf(stdout, "=== OpenCode Chat Spawner: %s (%s) ===\n", receipt.WaveID, statusStr)
			fmt.Fprintf(stdout, "Total Chats: %d\n\n", len(receipt.Chats))
			for _, c := range receipt.Chats {
				fmt.Fprintf(stdout, "- Issue #%d [%s]: %s\n", c.IssueNumber, c.Lane, c.Title)
				fmt.Fprintf(stdout, "  Status:   %s\n", c.Status)
				if c.PID > 0 {
					fmt.Fprintf(stdout, "  PID:      %d\n", c.PID)
				}
				if c.LogFile != "" {
					fmt.Fprintf(stdout, "  Log:      %s\n", c.LogFile)
				}
				if c.Worktree != "" {
					fmt.Fprintf(stdout, "  Worktree: %s\n", c.Worktree)
				}
				if c.Error != "" {
					fmt.Fprintf(stdout, "  Error:    %s\n", c.Error)
				}
				if len(c.Command) > 0 {
					fmt.Fprintf(stdout, "  Command:  %s\n", strings.Join(c.Command, " "))
				}
				fmt.Fprintln(stdout)
			}
		}

		if hasError {
			return 1
		}
		return 0
	}

	// 7. Normal output
	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, issueorchestrator.MarkdownWaves(plan))
	default:
		fmt.Fprint(stdout, issueorchestrator.RenderWaves(plan))
	}

	if *check && plan.PlannedIssues > 0 {
		return 1
	}

	return 0
}

func runIssueOrchestratorAdjustSteps(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak issue-orchestrator adjust-steps", flag.ContinueOnError)
	fs.SetOutput(stderr)

	issueNum := fs.Int("issue", 0, "issue number to adjust (required)")
	steps := fs.Int("steps", -1, "new expected steps budget (required)")
	reason := fs.String("reason", "", "reason for adjusting steps")
	planPath := fs.String("plan", "", "path to plan JSON to modify in-place")
	fromPlan := fs.String("from-plan", "", "alias for --plan")
	workspace := fs.String("workspace", "", "workspace root")
	asJSON := fs.Bool("json", false, "emit updated plan JSON")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	if *issueNum <= 0 {
		fmt.Fprintln(stderr, "fak issue-orchestrator adjust-steps: --issue is required and must be positive")
		return 2
	}
	if *steps < 0 {
		fmt.Fprintln(stderr, "fak issue-orchestrator adjust-steps: --steps is required and must be non-negative")
		return 2
	}

	targetPlanPath := *planPath
	if targetPlanPath == "" {
		targetPlanPath = *fromPlan
	}

	if targetPlanPath != "" && !filepath.IsAbs(targetPlanPath) && *workspace != "" {
		if _, err := os.Stat(targetPlanPath); os.IsNotExist(err) {
			candidate := filepath.Join(*workspace, targetPlanPath)
			if _, err2 := os.Stat(candidate); err2 == nil {
				targetPlanPath = candidate
			}
		}
	}

	var plan issueorchestrator.Plan
	if targetPlanPath != "" {
		data, err := os.ReadFile(targetPlanPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: read plan: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(data, &plan); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: decode plan JSON: %v\n", err)
			return 2
		}
	} else {
		plan = issueorchestrator.Plan{
			Schema:        issueorchestrator.WavePlanSchema,
			TotalIssues:   1,
			PlannedIssues: 1,
			TotalWaves:    1,
			Waves: []issueorchestrator.Wave{
				{
					ID:       "wave-1",
					WaveSize: 1,
					Issues: []issueorchestrator.Issue{
						{
							Number:          *issueNum,
							ExpectedSteps:   0,
							Dispatchability: "dispatchable",
						},
					},
				},
			},
		}
	}

	if err := issueorchestrator.AdjustIssueSteps(&plan, *issueNum, *steps, *reason); err != nil {
		fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: %v\n", err)
		return 1
	}

	if targetPlanPath != "" {
		b, err := json.MarshalIndent(plan, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: marshal plan: %v\n", err)
			return 1
		}
		b = append(b, '\n')
		if err := os.WriteFile(targetPlanPath, b, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: write plan: %v\n", err)
			return 1
		}
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator adjust-steps: encode json: %v\n", err)
			return 1
		}
	} else {
		if targetPlanPath != "" {
			fmt.Fprintf(stdout, "Adjusted issue #%d steps to %d in plan\n", *issueNum, *steps)
		} else {
			fmt.Fprintf(stdout, "Adjusted issue #%d steps to %d\n", *issueNum, *steps)
		}
		if *reason != "" {
			fmt.Fprintf(stdout, "Reason: %s\n", *reason)
		}
	}

	return 0
}

func findLatestReceipt(root string) string {
	dirs := []string{
		filepath.Join(root, ".dispatch-runs"),
		root,
	}
	var latestPath string
	var latestMod time.Time

	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if !strings.HasSuffix(strings.ToLower(name), ".json") {
				continue
			}
			fullPath := filepath.Join(dir, name)
			info, err := entry.Info()
			if err != nil {
				continue
			}
			data, err := os.ReadFile(fullPath)
			if err != nil {
				continue
			}
			var probe struct {
				Schema string `json:"schema"`
				WaveID string `json:"wave_id"`
				Leaves []any  `json:"leaves"`
				Chats  []any  `json:"chats"`
			}
			if err := json.Unmarshal(data, &probe); err == nil {
				if probe.WaveID != "" || len(probe.Leaves) > 0 || len(probe.Chats) > 0 ||
					strings.Contains(probe.Schema, "harvest") || strings.Contains(probe.Schema, "spawn") {
					if info.ModTime().After(latestMod) {
						latestMod = info.ModTime()
						latestPath = fullPath
					}
				}
			}
		}
	}
	return latestPath
}
