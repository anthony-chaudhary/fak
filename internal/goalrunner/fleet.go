package goalrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// RankHostTractability ranks host execution ability:
// 0: full (host completable)
// 1: partial (host partial / external GPU/bench)
// 2: blocked (host blocked, e.g. CUDA node)
// 3: other/unspecified
func RankHostTractability(host string) int {
	h := strings.ToLower(strings.TrimSpace(host))
	switch {
	case h == "full" || h == "":
		return 0
	case strings.HasPrefix(h, "partial"):
		return 1
	case strings.HasPrefix(h, "blocked"):
		return 2
	default:
		return 3
	}
}

// OrderContracts sorts contracts in priced serial order:
// host-completable first, then partial, then blocked.
func OrderContracts(contracts []GoalContract) []GoalContract {
	ordered := make([]GoalContract, len(contracts))
	copy(ordered, contracts)

	sort.SliceStable(ordered, func(i, j int) bool {
		rankI := RankHostTractability(ordered[i].Host)
		rankJ := RankHostTractability(ordered[j].Host)
		if rankI != rankJ {
			return rankI < rankJ
		}
		if ordered[i].Priority != 0 && ordered[j].Priority != 0 && ordered[i].Priority != ordered[j].Priority {
			return ordered[i].Priority < ordered[j].Priority
		}
		return ordered[i].N < ordered[j].N
	})

	return ordered
}

// KillProcess terminates a process by PID.
func KillProcess(pid int) error {
	if pid <= 0 {
		return nil
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return proc.Kill()
}

// MonitorProcess polls for process completion until timeout or context cancellation.
// If timeout occurs, it attempts to kill the process.
func MonitorProcess(ctx context.Context, pid int, timeout time.Duration, pollInterval time.Duration) (bool, error) {
	if pollInterval <= 0 {
		pollInterval = 500 * time.Millisecond
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			_ = KillProcess(pid)
			return false, ctx.Err()
		case <-timer.C:
			_ = KillProcess(pid)
			return false, nil
		case <-ticker.C:
			if !IsProcessLive(pid) {
				return true, nil
			}
		}
	}
}

// WitnessIssue inspects git commits and workspace state to independently verify goal completion.
func WitnessIssue(workspace string, issue int, preSha string, workerLog string, hostFact string, preDirty []string) (*WitnessResult, error) {
	res := &WitnessResult{
		Issue:      issue,
		WorkerLog:  workerLog,
		IssueState: "unknown",
	}

	// 1. Check candidate commits on git if workspace has git
	if preSha != "" && workspace != "" {
		gitRange := fmt.Sprintf("%s..HEAD", preSha)
		grepPattern := fmt.Sprintf("(#%d)", issue)
		cmd := exec.Command("git", "-C", workspace, "log", gitRange, "-E", "--grep", grepPattern, "--format=%H\x1f%s")
		out, err := cmd.Output()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			for _, line := range lines {
				if line == "" {
					continue
				}
				parts := strings.Split(line, "\x1f")
				if len(parts) >= 2 {
					sha := parts[0]
					subj := parts[1]
					issueTag := fmt.Sprintf("#%d", issue)
					if strings.Contains(subj, issueTag) {
						res.ShippedSHA = sha
						res.ShippedVerdict = "OK"
						res.ShippedWitness = "diff-witnessed"
						res.BestSeenSHA = sha
						res.BestSeenVerdict = "OK"
						break
					}
				}
			}
		}
	}

	// 2. Swept-WIP guard: did shipped commit touch pre-existing dirty files?
	if res.ShippedSHA != "" && len(preDirty) > 0 && workspace != "" {
		showCmd := exec.Command("git", "-C", workspace, "show", "--name-only", "--format=", res.ShippedSHA)
		showOut, err := showCmd.Output()
		if err == nil {
			touchedFiles := strings.Split(strings.TrimSpace(string(showOut)), "\n")
			var sweepHits []string
			preDirtyMap := make(map[string]bool, len(preDirty))
			for _, d := range preDirty {
				preDirtyMap[strings.TrimSpace(d)] = true
			}
			for _, f := range touchedFiles {
				f = strings.TrimSpace(f)
				if f != "" && preDirtyMap[f] {
					sweepHits = append(sweepHits, f)
				}
			}
			if len(sweepHits) > 0 {
				limit := 5
				if len(sweepHits) < limit {
					limit = len(sweepHits)
				}
				res.SweepReview = fmt.Sprintf("REVIEW: commit touches %d pre-existing-dirty path(s): %s",
					len(sweepHits), strings.Join(sweepHits[:limit], ", "))
			}
		}
	}

	// 3. Inspect issue state via gh if available
	if issue > 0 {
		ghCmd := exec.Command("gh", "issue", "view", strconv.Itoa(issue), "--json", "state")
		ghOut, err := ghCmd.Output()
		if err == nil {
			var iv struct {
				State string `json:"state"`
			}
			if jsonErr := json.Unmarshal(ghOut, &iv); jsonErr == nil && iv.State != "" {
				res.IssueState = iv.State
			}
		}
	}

	// 4. Determine final outcome
	hostLower := strings.ToLower(hostFact)
	if res.ShippedSHA != "" && res.SweepReview == "" {
		res.Outcome = "met (witnessed ship)"
	} else if res.ShippedSHA != "" {
		res.Outcome = "shipped-needs-review (" + res.SweepReview + ")"
	} else if strings.HasPrefix(hostLower, "blocked") || strings.HasPrefix(hostLower, "partial") {
		res.Outcome = "host-precluded-block (host=" + hostFact + ")"
	} else {
		res.Outcome = "no-witnessed-effect"
	}

	return res, nil
}

// SerializePlanJSON serializes a FleetPlan to JSON.
func SerializePlanJSON(plan *FleetPlan) (string, error) {
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// RenderRollup generates a Markdown report for a fleet plan execution.
func RenderRollup(plan *FleetPlan, results []*WitnessResult) string {
	var buf bytes.Buffer
	buf.WriteString("# P0 goal-fleet serial run\n\n")
	buf.WriteString(fmt.Sprintf("- run root: %s\n", plan.RunRoot))
	buf.WriteString(fmt.Sprintf("- workspace: %s\n", plan.Workspace))
	buf.WriteString("- seat pool: SERVING=1 (single Claude engineering seat) -> serial\n")

	var orderTags []string
	for _, c := range plan.Contracts {
		orderTags = append(orderTags, fmt.Sprintf("#%d", c.N))
	}
	buf.WriteString(fmt.Sprintf("- order (host-tractability): %s\n\n", strings.Join(orderTags, " -> ")))

	for _, res := range results {
		buf.WriteString(fmt.Sprintf("## #%d\n", res.Issue))
		buf.WriteString(fmt.Sprintf("- outcome: **%s**\n", res.Outcome))
		if res.PID > 0 {
			buf.WriteString(fmt.Sprintf("- pid: %d | timed_out: %t\n", res.PID, res.TimedOut))
		}
		if res.ShippedSHA != "" {
			buf.WriteString(fmt.Sprintf("- witnessed commit: %s (verdict=%s, witness=%s)\n",
				res.ShippedSHA, res.ShippedVerdict, res.ShippedWitness))
		}
		if res.BestSeenSHA != "" && res.BestSeenSHA != res.ShippedSHA {
			buf.WriteString(fmt.Sprintf("- best-seen audit: %s (verdict=%s)\n",
				res.BestSeenSHA, res.BestSeenVerdict))
		}
		if res.IssueState != "" {
			buf.WriteString(fmt.Sprintf("- issue state: %s\n", res.IssueState))
		}
		if res.SweepReview != "" {
			buf.WriteString(fmt.Sprintf("- sweep check: %s\n", res.SweepReview))
		}
		if res.WorkerLog != "" {
			buf.WriteString(fmt.Sprintf("- log: %s\n", filepath.Base(res.WorkerLog)))
		}
		buf.WriteString("\n")
	}

	return buf.String()
}

// Launcher abstracts worker launching for modularity and testability.
type Launcher interface {
	Launch(opt LaunchOptions) (*LaunchResult, error)
}

// LauncherFunc adapts a function to the Launcher interface.
type LauncherFunc func(opt LaunchOptions) (*LaunchResult, error)

// Launch executes the adapted launch function.
func (f LauncherFunc) Launch(opt LaunchOptions) (*LaunchResult, error) {
	return f(opt)
}

// DefaultLauncher is the default launcher using LaunchDetachedWorker.
var DefaultLauncher = LauncherFunc(LaunchDetachedWorker)

// RunFleetPlan runs the contracts in a fleet plan serially and records witness evidence.
func RunFleetPlan(ctx context.Context, plan *FleetPlan, launcher Launcher) ([]*WitnessResult, error) {
	if plan == nil {
		return nil, fmt.Errorf("plan is nil")
	}
	if launcher == nil {
		launcher = DefaultLauncher
	}

	if plan.Workspace == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		plan.Workspace = wd
	}

	if plan.RunRoot == "" {
		stamp := time.Now().UTC().Format("20060102-150405Z")
		plan.RunRoot = filepath.Join(plan.Workspace, fmt.Sprintf(".goal-runs/p0-fleet-%s", stamp))
	}
	if err := os.MkdirAll(plan.RunRoot, 0755); err != nil {
		return nil, fmt.Errorf("create run root: %w", err)
	}

	if plan.RollupPath == "" {
		plan.RollupPath = filepath.Join(plan.RunRoot, "rollup.md")
	}
	if plan.StatusPath == "" {
		plan.StatusPath = filepath.Join(plan.RunRoot, "STATUS.txt")
	}

	timeout := plan.PerWorkerTimeout
	if timeout <= 0 {
		if plan.PerWorkerTimeoutMin > 0 {
			timeout = time.Duration(plan.PerWorkerTimeoutMin) * time.Minute
		} else {
			timeout = 90 * time.Minute
		}
	}

	// Order contracts according to host tractability
	plan.Contracts = OrderContracts(plan.Contracts)

	var results []*WitnessResult

	for _, g := range plan.Contracts {
		select {
		case <-ctx.Done():
			return results, ctx.Err()
		default:
		}

		relPointer := g.Pointer
		if plan.ContractsDir != "" && !filepath.IsAbs(relPointer) {
			relPointer = filepath.Join(plan.ContractsDir, g.Pointer)
		}

		var preSha string
		if shaOut, err := exec.Command("git", "-C", plan.Workspace, "rev-parse", "HEAD").Output(); err == nil {
			preSha = strings.TrimSpace(string(shaOut))
		}

		var preDirty []string
		if statusOut, err := exec.Command("git", "-C", plan.Workspace, "status", "--porcelain").Output(); err == nil {
			for _, line := range strings.Split(strings.TrimSpace(string(statusOut)), "\n") {
				if len(line) > 3 {
					preDirty = append(preDirty, strings.TrimSpace(line[3:]))
				}
			}
		}

		if plan.DryRun {
			w := &WitnessResult{
				Issue:   g.N,
				Outcome: "dry-run (skipped launch)",
			}
			results = append(results, w)
			continue
		}

		// Launch detached worker
		launchOpt := LaunchOptions{
			Workspace:   plan.Workspace,
			PointerFile: relPointer,
			LogDir:      plan.RunRoot,
			WorkKind:    "engineering",
		}

		res, err := launcher.Launch(launchOpt)
		if err != nil {
			results = append(results, &WitnessResult{
				Issue:   g.N,
				Outcome: fmt.Sprintf("launch-failed: %v", err),
			})
			continue
		}

		// Monitor process with timeout
		timedOut := false
		if res.PID > 0 {
			exited, _ := MonitorProcess(ctx, res.PID, timeout, 500*time.Millisecond)
			if !exited {
				timedOut = true
			}
		}

		w, _ := WitnessIssue(plan.Workspace, g.N, preSha, res.OutLog, g.Host, preDirty)
		w.PID = res.PID
		w.TimedOut = timedOut
		results = append(results, w)
	}

	rollupContent := RenderRollup(plan, results)
	_ = os.WriteFile(plan.RollupPath, []byte(rollupContent), 0644)
	if plan.StatusPath != "" {
		statusLine := fmt.Sprintf("%s  driver done -- completed %d contracts (dry-run=%v)\n",
			time.Now().UTC().Format(time.RFC3339), len(results), plan.DryRun)
		_ = os.WriteFile(plan.StatusPath, []byte(statusLine), 0644)
	}

	return results, nil
}

var issueNumberRegex = regexp.MustCompile(`(?:issue-?|#)?(\d+)`)

// LoadFleetContracts loads goal contracts from a directory, JSON file, or individual contract markdown files.
func LoadFleetContracts(contractsDir, workspace string) ([]GoalContract, error) {
	if contractsDir == "" {
		return nil, fmt.Errorf("contracts-dir is required")
	}

	target := contractsDir
	if !filepath.IsAbs(target) && workspace != "" {
		target = filepath.Join(workspace, target)
	}

	st, err := os.Stat(target)
	if err != nil {
		return nil, fmt.Errorf("stat contracts path: %w", err)
	}

	if !st.IsDir() {
		// Single JSON file (could be FleetPlan or []GoalContract or GoalContract)
		data, err := os.ReadFile(target)
		if err != nil {
			return nil, fmt.Errorf("read contract file: %w", err)
		}

		var plan FleetPlan
		if err := json.Unmarshal(data, &plan); err == nil && len(plan.Contracts) > 0 {
			return plan.Contracts, nil
		}

		var slice []GoalContract
		if err := json.Unmarshal(data, &slice); err == nil && len(slice) > 0 {
			return slice, nil
		}

		var single GoalContract
		if err := json.Unmarshal(data, &single); err == nil && (single.N > 0 || single.Pointer != "") {
			return []GoalContract{single}, nil
		}

		return nil, fmt.Errorf("unable to parse contracts from %s", target)
	}

	// Target is a directory
	planJSON := filepath.Join(target, "plan.json")
	if data, err := os.ReadFile(planJSON); err == nil {
		var plan FleetPlan
		if err := json.Unmarshal(data, &plan); err == nil && len(plan.Contracts) > 0 {
			return plan.Contracts, nil
		}
	}

	contractsJSON := filepath.Join(target, "contracts.json")
	if data, err := os.ReadFile(contractsJSON); err == nil {
		var slice []GoalContract
		if err := json.Unmarshal(data, &slice); err == nil && len(slice) > 0 {
			return slice, nil
		}
	}

	entries, err := os.ReadDir(target)
	if err != nil {
		return nil, fmt.Errorf("read contracts dir: %w", err)
	}

	var contracts []GoalContract
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		path := filepath.Join(target, e.Name())
		ext := strings.ToLower(filepath.Ext(e.Name()))

		if ext == ".json" {
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			var c GoalContract
			if err := json.Unmarshal(data, &c); err == nil && (c.N > 0 || c.Pointer != "") {
				if c.Pointer == "" {
					c.Pointer = e.Name()
				}
				contracts = append(contracts, c)
				continue
			}
			var slice []GoalContract
			if err := json.Unmarshal(data, &slice); err == nil && len(slice) > 0 {
				contracts = append(contracts, slice...)
				continue
			}
		}

		if ext == ".md" || ext == ".txt" {
			n := 0
			if m := issueNumberRegex.FindStringSubmatch(e.Name()); len(m) > 1 {
				n, _ = strconv.Atoi(m[1])
			}
			contracts = append(contracts, GoalContract{
				N:       n,
				Pointer: e.Name(),
				Host:    "full",
			})
		}
	}

	if len(contracts) == 0 {
		return nil, fmt.Errorf("no valid goal contracts found in %s", target)
	}

	sort.SliceStable(contracts, func(i, j int) bool {
		return contracts[i].N < contracts[j].N
	})

	return contracts, nil
}
