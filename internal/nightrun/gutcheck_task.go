package nightrun

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	// GutcheckBenchmarkName is the catalog name of the paired-arm gut-check benchmark.
	GutcheckBenchmarkName = "tb4-gutcheck"
	// GutcheckTaskID is the canonical nightrun collection task ID for tb4-gutcheck.
	GutcheckTaskID = "bench-tb4-gutcheck"
	// GutcheckSuiteVersion identifies the pinned 12-task gut-check suite schema version.
	GutcheckSuiteVersion = "tb4-gutcheck-v1"
	// GutcheckDefaultDatasetRel is the repository-relative path to the gut-check suite manifest.
	GutcheckDefaultDatasetRel = "testdata/tb4bench/gutcheck_suite.json"

	// GutcheckDefaultMaxCostUSD bounds per-pass spend for scheduled gut-check sweeps.
	GutcheckDefaultMaxCostUSD = 2.00
	// GutcheckDefaultMaxTokens bounds total prompt+completion tokens across a 12-task pass.
	GutcheckDefaultMaxTokens int64 = 100000
	// GutcheckDefaultMaxDurationSec bounds total wall-clock execution time per pass.
	GutcheckDefaultMaxDurationSec = 600

	// GutcheckParityDeltaFloor is the minimum candidate-minus-reference solve-rate delta for PARITY.
	GutcheckParityDeltaFloor = -0.02
	// GutcheckNearDeltaFloor is the minimum candidate-minus-reference solve-rate delta for NEAR.
	GutcheckNearDeltaFloor = -0.10

	// Closed paired-arm gut-check verdict vocabulary.
	GutcheckVerdictParity   = "PARITY"
	GutcheckVerdictNear     = "NEAR"
	GutcheckVerdictDegraded = "DEGRADED"
	GutcheckVerdictRefused  = "REFUSED"
)

// GutcheckBudget defines hard cost, token, and wall-clock guards for a single
// scheduled gut-check pass. Zero values fall back to package defaults; negative
// values are rejected.
type GutcheckBudget struct {
	MaxCostUSD     float64 `json:"max_cost_usd"`
	MaxTotalTokens int64   `json:"max_total_tokens"`
	MaxDurationSec int     `json:"max_duration_sec"`
}

// DefaultGutcheckBudget returns the standard bounded cost/token/time guard for nightrun.
func DefaultGutcheckBudget() GutcheckBudget {
	return GutcheckBudget{
		MaxCostUSD:     GutcheckDefaultMaxCostUSD,
		MaxTotalTokens: GutcheckDefaultMaxTokens,
		MaxDurationSec: GutcheckDefaultMaxDurationSec,
	}
}

// Normalize fills unset (zero) fields with defaults and rejects negative bounds.
func (b GutcheckBudget) Normalize() (GutcheckBudget, error) {
	if b.MaxCostUSD < 0 {
		return b, fmt.Errorf("nightrun: gutcheck max_cost_usd must be non-negative, got %.4f", b.MaxCostUSD)
	}
	if b.MaxTotalTokens < 0 {
		return b, fmt.Errorf("nightrun: gutcheck max_total_tokens must be non-negative, got %d", b.MaxTotalTokens)
	}
	if b.MaxDurationSec < 0 {
		return b, fmt.Errorf("nightrun: gutcheck max_duration_sec must be non-negative, got %d", b.MaxDurationSec)
	}
	if b.MaxCostUSD == 0 {
		b.MaxCostUSD = GutcheckDefaultMaxCostUSD
	}
	if b.MaxTotalTokens == 0 {
		b.MaxTotalTokens = GutcheckDefaultMaxTokens
	}
	if b.MaxDurationSec == 0 {
		b.MaxDurationSec = GutcheckDefaultMaxDurationSec
	}
	return b, nil
}

// Check verifies observed pass consumption against the hard budget limits.
func (b GutcheckBudget) Check(costUSD float64, totalTokens int64, durationSec int) error {
	var violations []string
	if costUSD > b.MaxCostUSD {
		violations = append(violations, fmt.Sprintf("cost $%.4f > budget $%.2f", costUSD, b.MaxCostUSD))
	}
	if totalTokens > b.MaxTotalTokens {
		violations = append(violations, fmt.Sprintf("tokens %d > budget %d", totalTokens, b.MaxTotalTokens))
	}
	if durationSec > b.MaxDurationSec {
		violations = append(violations, fmt.Sprintf("duration %ds > budget %ds", durationSec, b.MaxDurationSec))
	}
	if len(violations) > 0 {
		return fmt.Errorf("cost budget exceeded: %s", strings.Join(violations, "; "))
	}
	return nil
}

// GutcheckResult captures the paired-arm evaluation, verdict, and budget telemetry
// of a scheduled tb4-gutcheck nightrun pass.
type GutcheckResult struct {
	SuiteVersion     string             `json:"suite_version"`
	DatasetPath      string             `json:"dataset_path"`
	OutDir           string             `json:"out_dir"`
	Host             string             `json:"host"`
	Timestamp        string             `json:"timestamp"`
	TasksEvaluated   int                `json:"tasks_evaluated"`
	ArmASolveRate    float64            `json:"arm_a_solve_rate"`
	ArmBSolveRate    float64            `json:"arm_b_solve_rate"`
	SolveRateDelta   float64            `json:"solve_rate_delta"`
	ArmAPromptTokens int64              `json:"arm_a_prompt_tokens"`
	ArmBPromptTokens int64              `json:"arm_b_prompt_tokens"`
	TotalTokens      int64              `json:"total_tokens"`
	EstimatedCostUSD float64            `json:"estimated_cost_usd"`
	DurationSec      int                `json:"duration_sec"`
	Budget           GutcheckBudget     `json:"budget"`
	BudgetExceeded   bool               `json:"budget_exceeded"`
	Verdict          string             `json:"verdict"`
	TelemetryDeltas  map[string]float64 `json:"telemetry_deltas"`
	ThresholdsMet    bool               `json:"thresholds_met"`
	Outcome          string             `json:"outcome"`
	LogPath          string             `json:"log_path"`
	MarkdownReport   string             `json:"markdown_report,omitempty"`
	Error            string             `json:"error,omitempty"`
}

// GutcheckRunnerFunc executes a paired-arm gut-check pass on the provided dataset under budget.
type GutcheckRunnerFunc func(ctx context.Context, datasetPath string, outDir string, budget GutcheckBudget) (*GutcheckResult, error)

var (
	gutcheckRunnerMu sync.RWMutex
	gutcheckRunner   GutcheckRunnerFunc
)

// RegisterGutcheckRunner installs the in-process gut-check execution runner.
// This inverted registration seam ensures tier-2 nightrun does not upward-import tier-4 tb4bench.
func RegisterGutcheckRunner(fn GutcheckRunnerFunc) {
	gutcheckRunnerMu.Lock()
	defer gutcheckRunnerMu.Unlock()
	gutcheckRunner = fn
}

// IsGutcheckTask reports whether a collection task represents the tb4-gutcheck benchmark.
func IsGutcheckTask(t Task) bool {
	if t.ID == GutcheckTaskID {
		return true
	}
	if strings.Contains(t.Run, GutcheckBenchmarkName) || strings.Contains(t.Run, "gutcheck_suite.json") {
		return true
	}
	return false
}

// FindGutcheckTask locates the tb4-gutcheck benchmark collection task in a task slice.
func FindGutcheckTask(tasks []Task) (Task, bool) {
	for _, t := range tasks {
		if IsGutcheckTask(t) {
			return t, true
		}
	}
	return Task{}, false
}

// ClassifyGutcheckVerdict maps a paired-arm solve-rate delta (candidate - reference)
// to a closed verdict label.
func ClassifyGutcheckVerdict(solveRateDelta float64) string {
	switch {
	case solveRateDelta >= GutcheckParityDeltaFloor:
		return GutcheckVerdictParity
	case solveRateDelta >= GutcheckNearDeltaFloor:
		return GutcheckVerdictNear
	default:
		return GutcheckVerdictDegraded
	}
}

// RunGutcheckNightrun executes a bounded paired-arm gut-check pass for the nightly collector:
//  1. Normalizes and validates the cost/token/time budget.
//  2. Resolves the gut-check suite manifest (default: testdata/tb4bench/gutcheck_suite.json).
//  3. Runs paired arms via the registered in-process runner or subprocess fallback.
//  4. Enforces the hard cost/token/duration budget guard (REFUSED on overrun).
//  5. Classifies paired-arm verdict (PARITY / NEAR / DEGRADED / REFUSED).
//  6. Emits a structured collector log to experiments/nightrun/<host>/<timestamp>-bench-tb4-gutcheck.log.
func RunGutcheckNightrun(ctx context.Context, datasetPath string, outDir string, budget GutcheckBudget) (*GutcheckResult, error) {
	normBudget, err := budget.Normalize()
	if err != nil {
		return nil, err
	}

	root := findRepoRoot()
	if strings.TrimSpace(datasetPath) == "" {
		datasetPath = filepath.Join("testdata", "tb4bench", "gutcheck_suite.json")
	}
	datasetPath = resolveDatasetPath(datasetPath)

	if outDir == "" {
		outDir, err = os.MkdirTemp("", "tb4-gutcheck-nightrun-*")
		if err != nil {
			return nil, fmt.Errorf("nightrun: failed to create temporary gutcheck run dir: %w", err)
		}
	} else {
		if err := os.MkdirAll(outDir, 0755); err != nil {
			return nil, fmt.Errorf("nightrun: failed to create output directory %s: %w", outDir, err)
		}
	}

	stamp := time.Now().UTC().Format("20060102T150405Z")
	host, _ := os.Hostname()
	if host == "" {
		host = "localhost"
	}
	host = sanitize(host)

	runCtx, cancel := context.WithTimeout(ctx, time.Duration(normBudget.MaxDurationSec)*time.Second)
	defer cancel()

	gutcheckRunnerMu.RLock()
	runner := gutcheckRunner
	gutcheckRunnerMu.RUnlock()

	start := time.Now()
	var result *GutcheckResult
	var runErr error
	if runner != nil {
		result, runErr = runner(runCtx, datasetPath, outDir, normBudget)
	} else {
		result, runErr = runSubprocessGutcheck(runCtx, root, datasetPath, outDir)
	}
	elapsedSec := int(time.Since(start).Seconds())

	if runErr != nil && result == nil {
		return nil, runErr
	}
	if result == nil {
		result = &GutcheckResult{}
	}

	if result.SuiteVersion == "" {
		result.SuiteVersion = GutcheckSuiteVersion
	}
	result.DatasetPath = datasetPath
	result.OutDir = outDir
	result.Host = host
	result.Timestamp = stamp
	result.Budget = normBudget
	if result.DurationSec == 0 {
		result.DurationSec = elapsedSec
	}
	if result.TotalTokens == 0 {
		result.TotalTokens = result.ArmAPromptTokens + result.ArmBPromptTokens
	}
	if result.SolveRateDelta == 0 && (result.ArmASolveRate != 0 || result.ArmBSolveRate != 0) {
		result.SolveRateDelta = result.ArmASolveRate - result.ArmBSolveRate
	}

	if result.TelemetryDeltas == nil {
		result.TelemetryDeltas = make(map[string]float64)
	}
	result.TelemetryDeltas["solve_rate_delta"] = result.SolveRateDelta
	result.TelemetryDeltas["total_tokens"] = float64(result.TotalTokens)
	result.TelemetryDeltas["estimated_cost_usd"] = result.EstimatedCostUSD
	result.TelemetryDeltas["duration_sec"] = float64(result.DurationSec)

	var gateErrors []string
	if budgetErr := normBudget.Check(result.EstimatedCostUSD, result.TotalTokens, result.DurationSec); budgetErr != nil {
		result.BudgetExceeded = true
		result.Verdict = GutcheckVerdictRefused
		gateErrors = append(gateErrors, budgetErr.Error())
	} else if result.Verdict == "" {
		result.Verdict = ClassifyGutcheckVerdict(result.SolveRateDelta)
	}

	if result.Verdict == GutcheckVerdictDegraded {
		gateErrors = append(gateErrors, fmt.Sprintf("paired-arm verdict DEGRADED (solve_rate_delta=%.4f < %.2f)", result.SolveRateDelta, GutcheckNearDeltaFloor))
	} else if result.Verdict == GutcheckVerdictRefused && !result.BudgetExceeded {
		gateErrors = append(gateErrors, "paired-arm verdict REFUSED")
	}

	result.ThresholdsMet = len(gateErrors) == 0 && runErr == nil
	if result.ThresholdsMet {
		result.Outcome = "collected"
	} else {
		result.Outcome = "failed"
		if len(gateErrors) > 0 {
			result.Error = strings.Join(gateErrors, "; ")
		} else if runErr != nil {
			result.Error = runErr.Error()
		}
	}

	if _, err := emitGutcheckLog(root, result); err != nil {
		return result, fmt.Errorf("nightrun: failed to emit gutcheck log: %w", err)
	}

	if len(gateErrors) > 0 {
		return result, fmt.Errorf("nightrun: gutcheck gate failure: %s", result.Error)
	}
	if runErr != nil {
		return result, runErr
	}
	return result, nil
}

func emitGutcheckLog(root string, res *GutcheckResult) (string, error) {
	stamp := res.Timestamp
	if stamp == "" {
		stamp = time.Now().UTC().Format("20060102T150405Z")
		res.Timestamp = stamp
	}
	host := res.Host
	if host == "" {
		h, err := os.Hostname()
		if err != nil || h == "" {
			h = "localhost"
		}
		host = sanitize(h)
		res.Host = host
	}

	logDir := filepath.Join(root, "experiments", "nightrun", host)
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create nightrun log directory: %w", err)
	}

	logName := fmt.Sprintf("%s-bench-tb4-gutcheck.log", stamp)
	logPath := filepath.Join(logDir, logName)

	var b strings.Builder
	b.WriteString("================================================================================\n")
	b.WriteString("TB4 Paired-Arm Gut-Check Nightrun Collector\n")
	b.WriteString("================================================================================\n")
	fmt.Fprintf(&b, "timestamp: %s\n", stamp)
	fmt.Fprintf(&b, "host: %s\n", host)
	fmt.Fprintf(&b, "suite_version: %s\n", res.SuiteVersion)
	fmt.Fprintf(&b, "dataset: %s\n", res.DatasetPath)
	fmt.Fprintf(&b, "out_dir: %s\n", res.OutDir)
	fmt.Fprintf(&b, "task_id: %s\n", GutcheckTaskID)
	fmt.Fprintf(&b, "outcome=%q\n", res.Outcome)
	fmt.Fprintf(&b, "verdict: %s\n", res.Verdict)
	fmt.Fprintf(&b, "tasks_evaluated: %d\n", res.TasksEvaluated)
	fmt.Fprintf(&b, "arm_a_solve_rate: %.4f\n", res.ArmASolveRate)
	fmt.Fprintf(&b, "arm_b_solve_rate: %.4f\n", res.ArmBSolveRate)
	fmt.Fprintf(&b, "solve_rate_delta: %+.4f\n", res.SolveRateDelta)
	fmt.Fprintf(&b, "total_tokens: %d (budget: <= %d)\n", res.TotalTokens, res.Budget.MaxTotalTokens)
	fmt.Fprintf(&b, "estimated_cost_usd: %.4f (budget: <= %.2f)\n", res.EstimatedCostUSD, res.Budget.MaxCostUSD)
	fmt.Fprintf(&b, "duration_sec: %d (budget: <= %d)\n", res.DurationSec, res.Budget.MaxDurationSec)
	fmt.Fprintf(&b, "budget_exceeded: %t\n", res.BudgetExceeded)
	fmt.Fprintf(&b, "thresholds_met: %t\n", res.ThresholdsMet)
	if res.Error != "" {
		fmt.Fprintf(&b, "error: %s\n", res.Error)
	}
	if res.MarkdownReport != "" {
		b.WriteString("\n--- Comparative Report ---\n")
		b.WriteString(res.MarkdownReport)
		b.WriteString("\n")
	}
	b.WriteString("================================================================================\n")

	if err := os.WriteFile(logPath, []byte(b.String()), 0644); err != nil {
		return "", fmt.Errorf("failed to write gutcheck nightrun log: %w", err)
	}
	res.LogPath = logPath
	return logPath, nil
}

func runSubprocessGutcheck(ctx context.Context, root, datasetPath, outDir string) (*GutcheckResult, error) {
	fakBin := "fak"
	if runtime.GOOS == "windows" {
		fakBin = "fak.exe"
	}
	binPath := filepath.Join(root, fakBin)
	hasBin := false
	if _, err := os.Stat(binPath); err == nil {
		hasBin = true
	} else if p, err := exec.LookPath("fak"); err == nil {
		binPath = p
		hasBin = true
	}

	execCommand := func(args ...string) error {
		var cmd *exec.Cmd
		if hasBin {
			cmd = exec.CommandContext(ctx, binPath, args...)
		} else {
			fullArgs := append([]string{"run", "./cmd/fak"}, args...)
			cmd = exec.CommandContext(ctx, "go", fullArgs...)
		}
		cmd.Dir = root
		configureDispatchHelperCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("command %v failed (%w): %s", args, err, string(out))
		}
		return nil
	}

	if err := execCommand("bench", "tb4", "run", "--arm", "both", "--mock", "--dataset", datasetPath, "--out", outDir); err != nil {
		return nil, fmt.Errorf("gutcheck run failed: %w", err)
	}
	if err := execCommand("bench", "tb4", "eval", "--run-dir", outDir, "--dataset", datasetPath); err != nil {
		return nil, fmt.Errorf("gutcheck eval failed: %w", err)
	}

	compareJSON := filepath.Join(outDir, "compare.json")
	compareMD := filepath.Join(outDir, "compare.md")
	fakDir := filepath.Join(outDir, "fak")
	opencodeDir := filepath.Join(outDir, "opencode")
	if err := execCommand("bench", "tb4", "compare", "--fak-dir", fakDir, "--opencode-dir", opencodeDir, "--dataset", datasetPath, "--out-json", compareJSON, "--out-md", compareMD); err != nil {
		return nil, fmt.Errorf("gutcheck compare failed: %w", err)
	}

	data, err := os.ReadFile(compareJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", compareJSON, err)
	}
	var summary compareJSONSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return nil, fmt.Errorf("failed to unmarshal compare.json: %w", err)
	}
	mdData, _ := os.ReadFile(compareMD)

	return &GutcheckResult{
		SuiteVersion:     GutcheckSuiteVersion,
		ArmASolveRate:    summary.ArmAMetrics.Official.SolveRate,
		ArmBSolveRate:    summary.ArmBMetrics.Official.SolveRate,
		SolveRateDelta:   summary.SolveRateDelta,
		ArmAPromptTokens: summary.ArmAMetrics.Telemetry.TotalPromptTokens,
		ArmBPromptTokens: summary.ArmBMetrics.Telemetry.TotalPromptTokens,
		TotalTokens:      summary.ArmAMetrics.Telemetry.TotalPromptTokens + summary.ArmBMetrics.Telemetry.TotalPromptTokens,
		MarkdownReport:   string(mdData),
	}, nil
}
