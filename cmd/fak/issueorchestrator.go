package main

import (
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

func cmdIssueOrchestrator(argv []string) {
	os.Exit(runIssueOrchestrator(os.Stdout, os.Stderr, argv))
}

func runIssueOrchestrator(stdout, stderr io.Writer, argv []string) int {
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
	interactive := fs.Bool("interactive", false, "spawn interactive chat mode (-i) instead of headless run")
	worktree := fs.Bool("worktree", false, "prepare detached worker worktrees for each spawned chat")
	dryRun := fs.Bool("dry-run", false, "preview OpenCode chat spawn commands without executing")
	logDir := fs.String("log-dir", "", "directory for OpenCode session logs (default: .dispatch-runs)")

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

	// 1. Load issues from input
	var issues []issueorchestrator.Issue
	var err error

	inputPath := *fromIssues
	if inputPath == "" {
		inputPath = *fromPlan
	}

	issues, err = issueorchestrator.LoadIssues(inputPath, root)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue-orchestrator: %v\n", err)
		return 2
	}
	if topLimit > 0 && len(issues) > topLimit {
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
	waveOpts := issueorchestrator.WavePlanOptions{
		WaveSize:       *waveSize,
		MaxWaves:       *maxWaves,
		TargetIssues:   *targetIssues,
		TargetPoints:   targetPoints,
		Limit:          topLimit,
		ExcludedIssues: excludedIssues,
		ExcludedLanes:  excludedLanesList,
		AutoDetectHeld: !*noDetectHeld,
		WorkspaceRoot:  root,
	}
	if *spawnOpencode || *opencodeCommands {
		waveOpts.IncludeOpencodeCommands = true
		waveOpts.OpencodeOptions = issueorchestrator.OpencodeChatOptions{
			Model:       *model,
			Agent:       *agent,
			Interactive: *interactive,
			AutoApprove: true,
			PrintLogs:   true,
		}
	}
	plan := issueorchestrator.PlanWaves(issues, waveOpts)

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
		for _, issue := range selectedWave.Issues {
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
				cmd.Env = envSliceFromMap(workerworktree.WorktreeEnv(nil, chat.Worktree))
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
