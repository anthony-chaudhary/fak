package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// OpsScheduleReport provides structured JSON output for one-touch ops scheduling.
type OpsScheduleReport struct {
	Schema    string                `json:"schema"` // "fak-ops-schedule/1"
	Platform  string                `json:"platform"`
	RepoRoot  string                `json:"repo_root"`
	FakBinary string                `json:"fak_binary"`
	Action    string                `json:"action"` // "enroll", "status", "unregister", "dry_run"
	Tasks     []OpsScheduleTaskItem `json:"tasks"`
}

// OpsScheduleTaskItem describes one planned or enrolled ops schedule item.
type OpsScheduleTaskItem struct {
	Workload   string `json:"workload"`
	TaskName   string `json:"task_name"`
	Interval   string `json:"interval"`
	Timeout    string `json:"timeout"`
	RunHours   int    `json:"run_hours"`
	Ledger     string `json:"ledger"`
	Registered bool   `json:"registered"`
	Status     string `json:"status"`
	Detail     string `json:"detail,omitempty"`
	Definition string `json:"definition,omitempty"`
}

func runOpsSchedule(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("ops schedule", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, `Usage: fak ops schedule [flags]

One-touch cross-platform scheduler enrollment for developer workstations.
Provisions the 10-minute OpenCode issue-orchestrator runner, the 15-minute
OpenCode debt-orchestrator burndown runner, and the 1-hour landing & git-sync
merge runner across Windows (Task Scheduler), Linux (systemd user timers),
and macOS (launchd agents).

Flags:
`)
		fs.PrintDefaults()
	}

	repoRootFlag := fs.String("repo-root", "", "repository root path (default: auto-detected)")
	companionRootFlag := fs.String("companion-root", "", "companion repository root path (default: auto-detected)")
	intervalIssue := fs.Duration("interval-issue", 10*time.Minute, "cadence for OpenCode issue-orchestrator")
	intervalDebt := fs.Duration("interval-debt", 15*time.Minute, "cadence for OpenCode debt-orchestrator burndown")
	intervalSync := fs.Duration("interval-sync", 1*time.Hour, "cadence for landing & git-sync merge")
	timeoutIssue := fs.Duration("timeout-issue", 25*time.Minute, "timeout for issue-orchestrator runs")
	timeoutDebt := fs.Duration("timeout-debt", 25*time.Minute, "timeout for debt-orchestrator runs")
	timeoutSync := fs.Duration("timeout-sync", 15*time.Minute, "timeout for git-sync runs")
	runHours := fs.Int("run-hours", 24, "repetition duration limit in hours (default: 24)")
	taskPrefix := fs.String("task-prefix", "FakOps", "task identifier prefix")
	fakBinFlag := fs.String("fak-bin", "", "explicit path to fak binary")
	targetFlag := fs.String("target", "auto", "scheduler target: auto, taskscheduler, systemd, launchd")
	status := fs.Bool("status", false, "inspect host scheduler status for all workloads")
	unregister := fs.Bool("unregister", false, "remove scheduled tasks from the host OS")
	dryRun := fs.Bool("dry-run", false, "output generated configuration without registering")
	asJSON := fs.Bool("json", false, "emit structured JSON output")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if *intervalIssue <= 0 || *intervalDebt <= 0 || *intervalSync <= 0 {
		fmt.Fprintln(stderr, "ops schedule: --interval-issue, --interval-debt, and --interval-sync must be positive durations")
		return 2
	}
	if *timeoutIssue <= 0 || *timeoutDebt <= 0 || *timeoutSync <= 0 {
		fmt.Fprintln(stderr, "ops schedule: --timeout-issue, --timeout-debt, and --timeout-sync must be positive durations")
		return 2
	}
	if *runHours <= 0 || *runHours > 168 {
		fmt.Fprintln(stderr, "ops schedule: --run-hours must be between 1 and 168 (up to 7 days)")
		return 2
	}

	target := strings.ToLower(strings.TrimSpace(*targetFlag))
	if target == "" || target == "auto" {
		switch runtime.GOOS {
		case "windows":
			target = "taskscheduler"
		case "linux":
			target = "systemd"
		case "darwin":
			target = "launchd"
		default:
			fmt.Fprintf(stderr, "ops schedule: unsupported operating system %q\n", runtime.GOOS)
			return 2
		}
	}
	if target != "taskscheduler" && target != "systemd" && target != "launchd" {
		fmt.Fprintf(stderr, "ops schedule: unknown --target %q (want auto|taskscheduler|systemd|launchd)\n", target)
		return 2
	}

	root := *repoRootFlag
	if root == "" {
		root = discoverOpsRepoRoot()
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "ops schedule: resolve repo root: %v\n", err)
		return 2
	}

	companionRoot := *companionRootFlag
	if companionRoot == "" {
		candidates := []string{
			filepath.Join(absRoot, "..", "fak-private"),
			filepath.Join(absRoot, "fak-private"),
		}
		for _, cand := range candidates {
			if info, err := os.Stat(cand); err == nil && info.IsDir() {
				companionRoot = cand
				break
			}
		}
	}
	if companionRoot != "" {
		if absComp, err := filepath.Abs(companionRoot); err == nil {
			companionRoot = absComp
		}
	}

	resolvedFak := resolveFakExecutable(*fakBinFlag, absRoot)

	taskNameIssue := formatOpsTaskName(*taskPrefix, "IssueOrchestrator", *intervalIssue)
	taskNameDebt := formatOpsTaskName(*taskPrefix, "DebtOrchestrator", *intervalDebt)
	taskNameSync := formatOpsTaskName(*taskPrefix, "GitSync", *intervalSync)

	issueArgs := buildOpsCronRunArgs(resolvedFak, absRoot, companionRoot, "issue-orchestrator", "opencode-issue-orchestrator", *intervalIssue, *timeoutIssue)
	debtArgs := buildOpsCronRunArgs(resolvedFak, absRoot, companionRoot, "debt-orchestrator", "opencode-debt-orchestrator", *intervalDebt, *timeoutDebt)
	syncArgs := buildOpsCronRunArgs(resolvedFak, absRoot, companionRoot, "git-sync", "opencode-git-sync", *intervalSync, *timeoutSync)

	actionName := "enroll"
	if *dryRun {
		actionName = "dry_run"
	} else if *status {
		actionName = "status"
	} else if *unregister {
		actionName = "unregister"
	}

	report := OpsScheduleReport{
		Schema:    "fak-ops-schedule/1",
		Platform:  target,
		RepoRoot:  absRoot,
		FakBinary: resolvedFak,
		Action:    actionName,
		Tasks:     make([]OpsScheduleTaskItem, 0, 3),
	}

	ctx := context.Background()

	switch {
	case *status:
		itemIssue := queryOpsTaskStatus(ctx, target, taskNameIssue, "issue-orchestrator", *intervalIssue, *timeoutIssue, *runHours, absRoot, companionRoot)
		itemDebt := queryOpsTaskStatus(ctx, target, taskNameDebt, "debt-orchestrator", *intervalDebt, *timeoutDebt, *runHours, absRoot, companionRoot)
		itemSync := queryOpsTaskStatus(ctx, target, taskNameSync, "git-sync", *intervalSync, *timeoutSync, *runHours, absRoot, companionRoot)
		report.Tasks = append(report.Tasks, itemIssue, itemDebt, itemSync)

		if *asJSON {
			return writeOpsReportJSON(stdout, report)
		}
		renderOpsScheduleStatus(stdout, report)
		return 0

	case *unregister:
		errIssue := unregisterOpsTask(ctx, target, taskNameIssue)
		errDebt := unregisterOpsTask(ctx, target, taskNameDebt)
		errSync := unregisterOpsTask(ctx, target, taskNameSync)

		itemIssue := OpsScheduleTaskItem{
			Workload:   "issue-orchestrator",
			TaskName:   taskNameIssue,
			Interval:   formatOpsDuration(*intervalIssue),
			Timeout:    formatOpsDuration(*timeoutIssue),
			RunHours:   *runHours,
			Ledger:     resolveOpsLedgerPath(absRoot, companionRoot, "opencode-issue-orchestrator"),
			Registered: false,
			Status:     "Unregistered",
		}
		if errIssue != nil {
			itemIssue.Detail = errIssue.Error()
		}
		itemDebt := OpsScheduleTaskItem{
			Workload:   "debt-orchestrator",
			TaskName:   taskNameDebt,
			Interval:   formatOpsDuration(*intervalDebt),
			Timeout:    formatOpsDuration(*timeoutDebt),
			RunHours:   *runHours,
			Ledger:     resolveOpsLedgerPath(absRoot, companionRoot, "opencode-debt-orchestrator"),
			Registered: false,
			Status:     "Unregistered",
		}
		if errDebt != nil {
			itemDebt.Detail = errDebt.Error()
		}
		itemSync := OpsScheduleTaskItem{
			Workload:   "git-sync",
			TaskName:   taskNameSync,
			Interval:   formatOpsDuration(*intervalSync),
			Timeout:    formatOpsDuration(*timeoutSync),
			RunHours:   *runHours,
			Ledger:     resolveOpsLedgerPath(absRoot, companionRoot, "opencode-git-sync"),
			Registered: false,
			Status:     "Unregistered",
		}
		if errSync != nil {
			itemSync.Detail = errSync.Error()
		}
		report.Tasks = append(report.Tasks, itemIssue, itemDebt, itemSync)

		if *asJSON {
			return writeOpsReportJSON(stdout, report)
		}
		fmt.Fprintf(stdout, "Unregistered scheduled tasks: %s, %s, %s\n", taskNameIssue, taskNameDebt, taskNameSync)
		return 0

	case *dryRun:
		itemIssue := generateOpsDefinition(target, taskNameIssue, "issue-orchestrator", *intervalIssue, *timeoutIssue, *runHours, issueArgs, absRoot, companionRoot)
		itemDebt := generateOpsDefinition(target, taskNameDebt, "debt-orchestrator", *intervalDebt, *timeoutDebt, *runHours, debtArgs, absRoot, companionRoot)
		itemSync := generateOpsDefinition(target, taskNameSync, "git-sync", *intervalSync, *timeoutSync, *runHours, syncArgs, absRoot, companionRoot)
		report.Tasks = append(report.Tasks, itemIssue, itemDebt, itemSync)

		if *asJSON {
			return writeOpsReportJSON(stdout, report)
		}
		renderOpsScheduleDryRun(stdout, report)
		return 0

	default:
		ledgerDir := filepath.Dir(resolveOpsLedgerPath(absRoot, companionRoot, "opencode-issue-orchestrator"))
		_ = os.MkdirAll(ledgerDir, 0755)

		descIssue := fmt.Sprintf("Fak automated OpenCode issue-orchestrator (%s, every %s)", taskNameIssue, formatOpsDuration(*intervalIssue))
		descDebt := fmt.Sprintf("Fak automated OpenCode debt-orchestrator burndown (%s, every %s)", taskNameDebt, formatOpsDuration(*intervalDebt))
		descSync := fmt.Sprintf("Fak automated OpenCode git-sync (%s, every %s)", taskNameSync, formatOpsDuration(*intervalSync))

		if err := enrollOpsTask(ctx, target, taskNameIssue, descIssue, *intervalIssue, *runHours, issueArgs, absRoot); err != nil {
			fmt.Fprintf(stderr, "ops schedule: failed to enroll %s: %v\n", taskNameIssue, err)
			return 1
		}
		if err := enrollOpsTask(ctx, target, taskNameDebt, descDebt, *intervalDebt, *runHours, debtArgs, absRoot); err != nil {
			fmt.Fprintf(stderr, "ops schedule: failed to enroll %s: %v\n", taskNameDebt, err)
			return 1
		}
		if err := enrollOpsTask(ctx, target, taskNameSync, descSync, *intervalSync, *runHours, syncArgs, absRoot); err != nil {
			fmt.Fprintf(stderr, "ops schedule: failed to enroll %s: %v\n", taskNameSync, err)
			return 1
		}

		itemIssue := queryOpsTaskStatus(ctx, target, taskNameIssue, "issue-orchestrator", *intervalIssue, *timeoutIssue, *runHours, absRoot, companionRoot)
		itemDebt := queryOpsTaskStatus(ctx, target, taskNameDebt, "debt-orchestrator", *intervalDebt, *timeoutDebt, *runHours, absRoot, companionRoot)
		itemSync := queryOpsTaskStatus(ctx, target, taskNameSync, "git-sync", *intervalSync, *timeoutSync, *runHours, absRoot, companionRoot)
		report.Tasks = append(report.Tasks, itemIssue, itemDebt, itemSync)

		if *asJSON {
			return writeOpsReportJSON(stdout, report)
		}
		renderOpsScheduleEnroll(stdout, report)
		return 0
	}
}

func formatOpsTaskName(prefix, workload string, interval time.Duration) string {
	return fmt.Sprintf("%s-%s-%s", prefix, formatOpsDuration(interval), workload)
}

func formatOpsDuration(d time.Duration) string {
	if d%time.Hour == 0 {
		return fmt.Sprintf("%dh", d/time.Hour)
	}
	if d%time.Minute == 0 {
		return fmt.Sprintf("%dm", d/time.Minute)
	}
	return d.String()
}

func resolveOpsLedgerPath(repoRoot, companionRoot, jobName string) string {
	if companionRoot != "" {
		return filepath.Join(companionRoot, ".fak", "ledgers", jobName+".jsonl")
	}
	return filepath.Join(repoRoot, ".fak", "ledgers", jobName+".jsonl")
}

func buildOpsCronRunArgs(fakBin, repoRoot, companionRoot, workload, jobName string, interval, timeout time.Duration) []string {
	ledgerPath := resolveOpsLedgerPath(repoRoot, companionRoot, jobName)
	intervalStr := formatOpsDuration(interval)
	timeoutStr := formatOpsDuration(timeout)

	var childCmd []string
	if runtime.GOOS == "windows" {
		scriptRoot := companionRoot
		if scriptRoot == "" {
			scriptRoot = repoRoot
		}
		scriptPath := filepath.Join(scriptRoot, "ops", fmt.Sprintf("run-opencode-%s.ps1", workload))
		if _, err := os.Stat(scriptPath); err == nil {
			childCmd = []string{
				"powershell.exe",
				"-NoProfile",
				"-ExecutionPolicy", "Bypass",
				"-File", scriptPath,
			}
		} else {
			opsScript := filepath.Join(scriptRoot, "ops", "run-opencode-ops.ps1")
			if _, err := os.Stat(opsScript); err == nil {
				childCmd = []string{
					"powershell.exe",
					"-NoProfile",
					"-ExecutionPolicy", "Bypass",
					"-File", opsScript,
					"-Workload", workload,
				}
			} else {
				commandPrompt := "/issue-orchestrator"
				if workload == "git-sync" {
					commandPrompt = "/git-subagent-sync"
				} else if workload == "debt-orchestrator" {
					commandPrompt = "/debt-orchestrator"
				}
				childCmd = []string{
					resolveOpencodeExecutable(), "run",
					"--dir", repoRoot,
					"--format", "json",
					commandPrompt,
				}
			}
		}
	} else {
		commandPrompt := "/issue-orchestrator"
		if workload == "git-sync" {
			commandPrompt = "/git-subagent-sync"
		} else if workload == "debt-orchestrator" {
			commandPrompt = "/debt-orchestrator"
		}
		childCmd = []string{
			resolveOpencodeExecutable(), "run",
			"--dir", repoRoot,
			"--format", "json",
			commandPrompt,
		}
	}

	runArgs := []string{
		fakBin, "cron", "run",
		"--job", jobName,
		"--ledger", ledgerPath,
		"--interval", intervalStr,
		"--timeout", timeoutStr,
		"--interrupt-ceiling", timeoutStr,
		"--workdir", repoRoot,
		"--",
	}
	return append(runArgs, childCmd...)
}

func discoverOpsRepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := cwd
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return cwd
}

func resolveFakExecutable(explicit, repoRoot string) string {
	if explicit != "" {
		if abs, err := filepath.Abs(explicit); err == nil {
			if _, err := os.Stat(abs); err == nil {
				return abs
			}
		}
		return explicit
	}

	exeName := "fak"
	if runtime.GOOS == "windows" {
		exeName = "fak.exe"
	}

	if current, err := os.Executable(); err == nil && strings.EqualFold(filepath.Base(current), exeName) {
		return current
	}

	if p, err := exec.LookPath(exeName); err == nil {
		if abs, err := filepath.Abs(p); err == nil {
			return abs
		}
		return p
	}

	if gobin := os.Getenv("GOBIN"); gobin != "" {
		cand := filepath.Join(gobin, exeName)
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	if gopath := os.Getenv("GOPATH"); gopath != "" {
		cand := filepath.Join(gopath, "bin", exeName)
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		cand := filepath.Join(home, "go", "bin", exeName)
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}

	candidates := []string{
		filepath.Join(repoRoot, exeName),
		filepath.Join(repoRoot, "cmd", "fak", exeName),
		filepath.Join(repoRoot, "..", "fak", exeName),
	}
	for _, cand := range candidates {
		if abs, err := filepath.Abs(cand); err == nil {
			if _, err := os.Stat(abs); err == nil {
				return abs
			}
		}
	}

	return exeName
}

func resolveOpencodeExecutable() string {
	if explicit := strings.TrimSpace(os.Getenv("OPENCODE_BIN")); explicit != "" {
		if _, err := os.Stat(explicit); err == nil {
			return explicit
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		userBin := filepath.Join(home, ".opencode", "bin", "opencode")
		if runtime.GOOS == "windows" {
			userBin += ".exe"
		}
		if _, err := os.Stat(userBin); err == nil {
			return userBin
		}
	}
	for _, cand := range []string{
		"/opt/homebrew/bin/opencode",
		"/usr/local/bin/opencode",
	} {
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	if found, err := exec.LookPath("opencode"); err == nil {
		return found
	}
	return "opencode"
}

func enrollOpsTask(ctx context.Context, target, taskName, desc string, interval time.Duration, runHours int, args []string, repoRoot string) error {
	switch target {
	case "taskscheduler":
		sec := int64(interval.Seconds())
		execPath := args[0]
		argString := opsWinArgString(args[1:])

		psScript := fmt.Sprintf(`
$action    = New-ScheduledTaskAction -Execute '%s' -Argument '%s'
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Seconds %d) -RepetitionDuration (New-TimeSpan -Hours %d)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '%s' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description '%s' -Force | Out-Null
`, opsPSQuote(execPath), opsPSQuote(argString), sec, runHours, opsPSQuote(taskName), opsPSQuote(desc))

		cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", psScript)
		configureDispatchHelperCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("powershell Register-ScheduledTask: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return nil

	case "systemd":
		unit := strings.ToLower(taskName)
		serviceContent := fmt.Sprintf("[Unit]\nDescription=%s\n\n[Service]\nType=oneshot\nExecStart=%s\n",
			desc, opsSystemdExecLine(args))
		timerContent := fmt.Sprintf("[Unit]\nDescription=Timer for %s\n\n[Timer]\nOnBootSec=2m\nOnUnitActiveSec=%s\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n",
			desc, formatOpsDuration(interval))

		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		userDir := filepath.Join(home, ".config", "systemd", "user")
		if err := os.MkdirAll(userDir, 0755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(userDir, unit+".service"), []byte(serviceContent), 0644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(userDir, unit+".timer"), []byte(timerContent), 0644); err != nil {
			return err
		}
		_ = exec.CommandContext(ctx, "systemctl", "--user", "daemon-reload").Run()
		return exec.CommandContext(ctx, "systemctl", "--user", "enable", "--now", unit+".timer").Run()

	case "launchd":
		plistContent := opsRenderLaunchd(taskName, interval, args, repoRoot)
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		agentDir := filepath.Join(home, "Library", "LaunchAgents")
		if err := os.MkdirAll(agentDir, 0755); err != nil {
			return err
		}
		plistPath := filepath.Join(agentDir, taskName+".plist")
		if err := os.WriteFile(plistPath, []byte(plistContent), 0644); err != nil {
			return err
		}
		_ = exec.CommandContext(ctx, "launchctl", "unload", plistPath).Run()
		return exec.CommandContext(ctx, "launchctl", "load", "-w", plistPath).Run()

	default:
		return fmt.Errorf("unsupported scheduler target: %s", target)
	}
}

func unregisterOpsTask(ctx context.Context, target, taskName string) error {
	switch target {
	case "taskscheduler":
		cmd := exec.CommandContext(ctx, "schtasks", "/Delete", "/TN", taskName, "/F")
		configureDispatchHelperCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err != nil && !strings.Contains(string(out), "cannot find") {
			return fmt.Errorf("schtasks /Delete: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return nil

	case "systemd":
		unit := strings.ToLower(taskName)
		_ = exec.CommandContext(ctx, "systemctl", "--user", "disable", "--now", unit+".timer").Run()
		if home, err := os.UserHomeDir(); err == nil {
			userDir := filepath.Join(home, ".config", "systemd", "user")
			_ = os.Remove(filepath.Join(userDir, unit+".service"))
			_ = os.Remove(filepath.Join(userDir, unit+".timer"))
		}
		_ = exec.CommandContext(ctx, "systemctl", "--user", "daemon-reload").Run()
		return nil

	case "launchd":
		if home, err := os.UserHomeDir(); err == nil {
			plistPath := filepath.Join(home, "Library", "LaunchAgents", taskName+".plist")
			_ = exec.CommandContext(ctx, "launchctl", "unload", plistPath).Run()
			_ = os.Remove(plistPath)
		}
		return nil

	default:
		return fmt.Errorf("unsupported scheduler target: %s", target)
	}
}

func queryOpsTaskStatus(ctx context.Context, target, taskName, workload string, interval, timeout time.Duration, runHours int, repoRoot, companionRoot string) OpsScheduleTaskItem {
	item := OpsScheduleTaskItem{
		Workload:   workload,
		TaskName:   taskName,
		Interval:   formatOpsDuration(interval),
		Timeout:    formatOpsDuration(timeout),
		RunHours:   runHours,
		Ledger:     resolveOpsLedgerPath(repoRoot, companionRoot, fmt.Sprintf("opencode-%s", workload)),
		Registered: false,
		Status:     "NotRegistered",
	}

	switch target {
	case "taskscheduler":
		cmd := exec.CommandContext(ctx, "schtasks", "/Query", "/TN", taskName, "/FO", "CSV")
		configureDispatchHelperCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err == nil {
			item.Registered = true
			item.Status = "Ready"
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(lines) > 1 {
				item.Detail = strings.Trim(lines[1], "\r\n\"")
			}
		} else {
			item.Detail = "Task not registered in Windows Task Scheduler"
		}

	case "systemd":
		unit := strings.ToLower(taskName)
		out, err := exec.CommandContext(ctx, "systemctl", "--user", "is-active", unit+".timer").CombinedOutput()
		trimmed := strings.TrimSpace(string(out))
		if err == nil && trimmed == "active" {
			item.Registered = true
			item.Status = "Ready"
			item.Detail = "active"
		} else {
			item.Detail = trimmed
		}

	case "launchd":
		out, err := exec.CommandContext(ctx, "launchctl", "list", taskName).CombinedOutput()
		if err == nil {
			item.Registered = true
			item.Status = "Ready"
			item.Detail = "loaded"
			outStr := string(out)
			if strings.Contains(outStr, "\"LastExitStatus\" = ") {
				idx := strings.Index(outStr, "\"LastExitStatus\" = ")
				sub := outStr[idx+len("\"LastExitStatus\" = "):]
				if semi := strings.Index(sub, ";"); semi > 0 {
					codeStr := strings.TrimSpace(sub[:semi])
					if code, parseErr := strconv.Atoi(codeStr); parseErr == nil && code != 0 {
						exitCode := code
						if exitCode > 255 {
							exitCode = exitCode >> 8
						}
						item.Status = "Failed"
						item.Detail = fmt.Sprintf("last exit status %d", exitCode)
					}
				}
			}
			if item.Status == "Ready" && item.Ledger != "" {
				if lastOutcome, ok := readLastOpsLedgerOutcome(item.Ledger); ok {
					if lastOutcome == "failed" || lastOutcome == "timeout" {
						item.Status = "Failed"
						item.Detail = fmt.Sprintf("last ledger outcome: %s", lastOutcome)
					}
				}
			}
		} else {
			item.Detail = strings.TrimSpace(string(out))
		}
	}

	return item
}

func readLastOpsLedgerOutcome(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err == nil {
			if outcome, ok := m["outcome"].(string); ok && outcome != "" {
				return outcome, true
			}
		}
	}
	return "", false
}

func generateOpsDefinition(target, taskName, workload string, interval, timeout time.Duration, runHours int, args []string, repoRoot, companionRoot string) OpsScheduleTaskItem {
	item := OpsScheduleTaskItem{
		Workload:   workload,
		TaskName:   taskName,
		Interval:   formatOpsDuration(interval),
		Timeout:    formatOpsDuration(timeout),
		RunHours:   runHours,
		Ledger:     resolveOpsLedgerPath(repoRoot, companionRoot, fmt.Sprintf("opencode-%s", workload)),
		Registered: false,
		Status:     "DryRun",
	}

	desc := fmt.Sprintf("Fak automated OpenCode %s (%s, every %s)", workload, taskName, formatOpsDuration(interval))

	var buf strings.Builder
	switch target {
	case "taskscheduler":
		fmt.Fprint(&buf, opsRenderTaskScheduler(taskName, desc, interval, runHours, args))
	case "systemd":
		fmt.Fprint(&buf, opsRenderSystemd(strings.ToLower(taskName), desc, interval, args))
	case "launchd":
		fmt.Fprint(&buf, opsRenderLaunchd(taskName, interval, args, repoRoot))
	}

	item.Definition = buf.String()
	return item
}

func opsRenderTaskScheduler(label, desc string, interval time.Duration, runHours int, args []string) string {
	sec := int64(interval.Seconds())
	var b strings.Builder
	b.WriteString("# Written by: fak ops schedule. Run in PowerShell to register the task;\n")
	b.WriteString("# Task Scheduler fires on the interval, fak cron run bounds the execution.\n")
	fmt.Fprintf(&b, "$action    = New-ScheduledTaskAction -Execute '%s' -Argument '%s'\n",
		opsPSQuote(args[0]), opsWinArgString(args[1:]))
	fmt.Fprintf(&b, "$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Seconds %d) -RepetitionDuration (New-TimeSpan -Hours %d)\n", sec, runHours)
	b.WriteString("$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew\n")
	b.WriteString("$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited\n")
	fmt.Fprintf(&b, "Register-ScheduledTask -TaskName '%s' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description '%s' -Force\n",
		opsPSQuote(label), opsPSQuote(desc))
	return b.String()
}

func opsRenderSystemd(label, desc string, interval time.Duration, args []string) string {
	sec := int64(interval.Seconds())
	var b strings.Builder
	fmt.Fprintf(&b, "# Written by: fak ops schedule. Install both units to ~/.config/systemd/user/,\n")
	fmt.Fprintf(&b, "# then: systemctl --user enable --now %s.timer\n", label)
	fmt.Fprintf(&b, "\n# === %s.service ===\n", label)
	b.WriteString("[Unit]\n")
	fmt.Fprintf(&b, "Description=%s\n\n", desc)
	b.WriteString("[Service]\n")
	b.WriteString("Type=oneshot\n")
	fmt.Fprintf(&b, "ExecStart=%s\n", opsSystemdExecLine(args))
	fmt.Fprintf(&b, "\n# === %s.timer ===\n", label)
	b.WriteString("[Unit]\n")
	fmt.Fprintf(&b, "Description=Timer for %s\n\n", desc)
	b.WriteString("[Timer]\n")
	fmt.Fprintf(&b, "OnBootSec=2m\n")
	fmt.Fprintf(&b, "OnUnitActiveSec=%ds\n", sec)
	b.WriteString("Persistent=true\n\n")
	b.WriteString("[Install]\n")
	b.WriteString("WantedBy=timers.target\n")
	return b.String()
}

func opsRenderLaunchd(label string, interval time.Duration, args []string, repoRoot string) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	fmt.Fprintf(&b, "<!-- Written by: fak ops schedule — install: launchctl load -w %s.plist -->\n", label)
	b.WriteString(`<plist version="1.0">` + "\n")
	b.WriteString("  <dict>\n")
	fmt.Fprintf(&b, "    <key>Label</key>\n    <string>%s</string>\n", opsXMLEscape(label))
	b.WriteString("    <key>ProgramArguments</key>\n    <array>\n")
	for _, a := range args {
		fmt.Fprintf(&b, "      <string>%s</string>\n", opsXMLEscape(a))
	}
	b.WriteString("    </array>\n")
	fmt.Fprintf(&b, "    <key>StartInterval</key>\n    <integer>%d</integer>\n", int64(interval.Seconds()))
	b.WriteString("    <key>RunAtLoad</key>\n    <false/>\n")
	if strings.TrimSpace(repoRoot) != "" {
		fmt.Fprintf(&b, "    <key>WorkingDirectory</key>\n    <string>%s</string>\n", opsXMLEscape(strings.TrimSpace(repoRoot)))
	}
	homeDir, _ := os.UserHomeDir()
	b.WriteString("    <key>EnvironmentVariables</key>\n    <dict>\n")
	if homeDir != "" {
		fmt.Fprintf(&b, "      <key>HOME</key>\n      <string>%s</string>\n", opsXMLEscape(homeDir))
	}
	activePath := os.Getenv("PATH")
	if activePath == "" {
		if homeDir != "" {
			activePath = fmt.Sprintf("/opt/homebrew/bin:%s/.local/bin:%s/.opencode/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", homeDir, homeDir)
		} else {
			activePath = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
		}
	}
	fmt.Fprintf(&b, "      <key>PATH</key>\n      <string>%s</string>\n", opsXMLEscape(activePath))
	b.WriteString("    </dict>\n")
	logDir := "/tmp"
	if homeDir != "" {
		userLogDir := filepath.Join(homeDir, "Library", "Logs", "fak")
		_ = os.MkdirAll(userLogDir, 0755)
		logDir = userLogDir
	}
	fmt.Fprintf(&b, "    <key>StandardOutPath</key>\n    <string>%s/%s.log</string>\n", opsXMLEscape(logDir), opsXMLEscape(label))
	fmt.Fprintf(&b, "    <key>StandardErrorPath</key>\n    <string>%s/%s.err</string>\n", opsXMLEscape(logDir), opsXMLEscape(label))
	b.WriteString("  </dict>\n</plist>\n")
	return b.String()
}

func opsPSQuote(s string) string { return strings.ReplaceAll(s, "'", "''") }

func opsWinArgString(args []string) string {
	parts := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t") {
			parts[i] = `"` + a + `"`
		} else {
			parts[i] = a
		}
	}
	return opsPSQuote(strings.Join(parts, " "))
}

func opsSystemdExecLine(args []string) string {
	parts := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t\"\\") {
			parts[i] = `"` + strings.ReplaceAll(strings.ReplaceAll(a, `\`, `\\`), `"`, `\"`) + `"`
		} else {
			parts[i] = a
		}
	}
	return strings.Join(parts, " ")
}

func opsXMLEscape(s string) string {
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;",
	).Replace(s)
}

func renderOpsScheduleEnroll(w io.Writer, r OpsScheduleReport) {
	fmt.Fprintf(w, "=== One-Touch Developer Workstation Ops Enrollment ===\n")
	fmt.Fprintf(w, "Platform:    %s\n", r.Platform)
	fmt.Fprintf(w, "Repository:  %s\n", r.RepoRoot)
	fmt.Fprintf(w, "Fak Binary:  %s\n\n", r.FakBinary)

	for i, t := range r.Tasks {
		fmt.Fprintf(w, "Workload %d: %s\n", i+1, t.Workload)
		fmt.Fprintf(w, "  Task:      %s\n", t.TaskName)
		fmt.Fprintf(w, "  Interval:  every %s (timeout %s, limit %dh)\n", t.Interval, t.Timeout, t.RunHours)
		fmt.Fprintf(w, "  Ledger:    %s\n", t.Ledger)
		fmt.Fprintf(w, "  Status:    %s\n\n", t.Status)
	}

	fmt.Fprintf(w, "Enrollment complete: all scheduled tasks are active and bounded.\n")
	fmt.Fprintf(w, "Inspect status any time:  fak ops schedule --status\n")
	fmt.Fprintf(w, "Preview configuration:    fak ops schedule --dry-run\n")
	fmt.Fprintf(w, "Unregister schedulers:    fak ops schedule --unregister\n")
}

func renderOpsScheduleStatus(w io.Writer, r OpsScheduleReport) {
	fmt.Fprintf(w, "=== Developer Workstation Ops Status ===\n")
	fmt.Fprintf(w, "Platform:    %s\n", r.Platform)
	fmt.Fprintf(w, "Repository:  %s\n\n", r.RepoRoot)

	for _, t := range r.Tasks {
		registeredStr := "NO"
		if t.Registered {
			registeredStr = "YES"
		}
		fmt.Fprintf(w, "%-20s Registered: %-4s Status: %-10s Interval: %-5s Task: %s\n", t.Workload, registeredStr, t.Status, t.Interval, t.TaskName)
		if t.Ledger != "" {
			fmt.Fprintf(w, "  Ledger: %s\n", t.Ledger)
		}
	}
}

func renderOpsScheduleDryRun(w io.Writer, r OpsScheduleReport) {
	fmt.Fprintf(w, "=== Dry Run: Planned Workstation Ops Schedulers (%s) ===\n", r.Platform)
	for _, t := range r.Tasks {
		fmt.Fprintf(w, "\n--- %s (%s, every %s) ---\n", t.TaskName, t.Workload, t.Interval)
		fmt.Fprintln(w, t.Definition)
	}
}

func writeOpsReportJSON(w io.Writer, r OpsScheduleReport) int {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return 1
	}
	return 0
}
