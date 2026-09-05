package nightrun

import (
	"context"
	"encoding/json"
	"errors"
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
	// TB4BenchmarkName is the catalog name of the Terminal-Bench 4 benchmark.
	TB4BenchmarkName = "tb4bench"
	// TB4TaskID is the canonical nightrun collection task ID for TB4.
	TB4TaskID = "bench-tb4bench"
	// TB4TaskIDAlias is the short task ID alias for TB4.
	TB4TaskIDAlias = "bench-tb4"

	// TB4MinSolveRateArmA is the required solve rate threshold for fak in-kernel (100% on synthetic suite).
	TB4MinSolveRateArmA = 1.00
	// TB4MinTokenReduction is the minimum required prompt token reduction via vDSO / KV caching (80%).
	TB4MinTokenReduction = 0.80
)

// TB4RegressionResult captures the comparative evaluation and telemetry
// of a TB4 nightrun regression execution.
type TB4RegressionResult struct {
	DatasetPath      string             `json:"dataset_path"`
	OutDir           string             `json:"out_dir"`
	Host             string             `json:"host"`
	Timestamp        string             `json:"timestamp"`
	ArmASolveRate    float64            `json:"arm_a_solve_rate"`
	ArmBSolveRate    float64            `json:"arm_b_solve_rate"`
	SolveRateDelta   float64            `json:"solve_rate_delta"`
	ArmAPromptTokens int64              `json:"arm_a_prompt_tokens"`
	ArmBPromptTokens int64              `json:"arm_b_prompt_tokens"`
	TokenReduction   float64            `json:"token_reduction"`
	VDSOHits         int64              `json:"vdso_hits"`
	TelemetryDeltas  map[string]float64 `json:"telemetry_deltas"`
	ThresholdsMet    bool               `json:"thresholds_met"`
	Outcome          string             `json:"outcome"`
	LogPath          string             `json:"log_path"`
	MarkdownReport   string             `json:"markdown_report,omitempty"`
	Error            string             `json:"error,omitempty"`
}

// TB4RunnerFunc executes a TB4 regression campaign across both arms on the provided dataset.
type TB4RunnerFunc func(ctx context.Context, datasetPath string, outDir string) (*TB4RegressionResult, error)

var (
	tb4RunnerMu sync.RWMutex
	tb4Runner   TB4RunnerFunc
)

// RegisterTB4Runner installs the in-process TB4 campaign execution runner.
// This inverted registration seam ensures tier-2 nightrun does not upward-import tier-4 tb4bench.
func RegisterTB4Runner(fn TB4RunnerFunc) {
	tb4RunnerMu.Lock()
	defer tb4RunnerMu.Unlock()
	tb4Runner = fn
}

// IsTB4Task reports whether a collection task represents the Terminal-Bench 4 benchmark.
func IsTB4Task(t Task) bool {
	if t.ID == TB4TaskID || t.ID == TB4TaskIDAlias {
		return true
	}
	if strings.Contains(t.Run, "bench tb4") || strings.Contains(t.Run, "tb4bench") {
		return true
	}
	return false
}

// FindTB4Task locates the TB4 benchmark collection task in a task slice.
func FindTB4Task(tasks []Task) (Task, bool) {
	for _, t := range tasks {
		if IsTB4Task(t) {
			return t, true
		}
	}
	return Task{}, false
}

// RunTB4NightrunRegression executes the TB4 all-in-one vs OpenCode regression smoke test
// for the nightly collector:
// 1. Runs TB4 campaign in mock mode across both arms on the dataset (default: testdata/tb4bench/synthetic_suite.json).
// 2. Evaluates task workspaces with verification oracles.
// 3. Generates comparative report.
// 4. Validates regression thresholds (Arm A solve rate >= 1.0, token reduction >= 0.80).
// 5. Appends telemetry deltas and vDSO hit counts.
// 6. Emits structured log to experiments/nightrun/<host>/<timestamp>-bench-tb4.log with outcome="collected".
func RunTB4NightrunRegression(ctx context.Context, datasetPath string, outDir string) (*TB4RegressionResult, error) {
	root := findRepoRoot()
	datasetPath = resolveDatasetPath(datasetPath)

	if outDir == "" {
		var err error
		outDir, err = os.MkdirTemp("", "tb4-nightrun-regression-*")
		if err != nil {
			return nil, fmt.Errorf("nightrun: failed to create temporary run dir: %w", err)
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

	tb4RunnerMu.RLock()
	runner := tb4Runner
	tb4RunnerMu.RUnlock()

	var result *TB4RegressionResult
	var runErr error

	if runner != nil {
		result, runErr = runner(ctx, datasetPath, outDir)
	} else {
		result, runErr = runSubprocessTB4(ctx, root, datasetPath, outDir)
	}

	if runErr != nil && result == nil {
		return nil, runErr
	}
	if result == nil {
		result = &TB4RegressionResult{}
	}

	result.DatasetPath = datasetPath
	result.OutDir = outDir
	result.Host = host
	result.Timestamp = stamp

	// Compute token reduction if prompt token counts are present
	if result.ArmBPromptTokens > 0 {
		result.TokenReduction = float64(result.ArmBPromptTokens-result.ArmAPromptTokens) / float64(result.ArmBPromptTokens)
	}

	// Compute telemetry deltas
	if result.TelemetryDeltas == nil {
		result.TelemetryDeltas = make(map[string]float64)
	}
	result.TelemetryDeltas["prompt_tokens_saved"] = float64(result.ArmBPromptTokens - result.ArmAPromptTokens)
	result.TelemetryDeltas["token_reduction_pct"] = result.TokenReduction * 100.0
	result.TelemetryDeltas["vdso_hits"] = float64(result.VDSOHits)
	result.TelemetryDeltas["solve_rate_delta"] = result.SolveRateDelta

	// Validate regression thresholds
	thresholdsMet := true
	var regErrors []string
	if result.ArmASolveRate < TB4MinSolveRateArmA {
		thresholdsMet = false
		regErrors = append(regErrors, fmt.Sprintf("arm A solve rate %.2f < threshold %.2f", result.ArmASolveRate, TB4MinSolveRateArmA))
	}
	if result.TokenReduction < TB4MinTokenReduction {
		thresholdsMet = false
		regErrors = append(regErrors, fmt.Sprintf("token reduction %.2f < threshold %.2f", result.TokenReduction, TB4MinTokenReduction))
	}
	result.ThresholdsMet = thresholdsMet

	if thresholdsMet {
		result.Outcome = "collected"
	} else {
		result.Outcome = "failed"
		result.Error = strings.Join(regErrors, "; ")
	}

	// Emit collector log
	if _, err := emitTB4Log(root, result); err != nil {
		return result, fmt.Errorf("nightrun: failed to emit log: %w", err)
	}

	if !thresholdsMet {
		return result, fmt.Errorf("nightrun: tb4 regression gate failure: %s", result.Error)
	}
	if runErr != nil {
		return result, runErr
	}

	return result, nil
}

func emitTB4Log(root string, res *TB4RegressionResult) (string, error) {
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

	logName := fmt.Sprintf("%s-bench-tb4.log", stamp)
	logPath := filepath.Join(logDir, logName)

	var b strings.Builder
	b.WriteString("================================================================================\n")
	b.WriteString("Terminal-Bench 4 Nightrun Regression Smoke Collector\n")
	b.WriteString("================================================================================\n")
	fmt.Fprintf(&b, "timestamp: %s\n", stamp)
	fmt.Fprintf(&b, "host: %s\n", host)
	fmt.Fprintf(&b, "dataset: %s\n", res.DatasetPath)
	fmt.Fprintf(&b, "out_dir: %s\n", res.OutDir)
	fmt.Fprintf(&b, "task_id: %s\n", TB4TaskID)
	fmt.Fprintf(&b, "outcome=\"%s\"\n", res.Outcome)
	fmt.Fprintf(&b, "arm_a_solve_rate: %.4f (threshold: >= %.2f)\n", res.ArmASolveRate, TB4MinSolveRateArmA)
	fmt.Fprintf(&b, "arm_b_solve_rate: %.4f\n", res.ArmBSolveRate)
	fmt.Fprintf(&b, "solve_rate_delta: %+.4f\n", res.SolveRateDelta)
	fmt.Fprintf(&b, "arm_a_prompt_tokens: %d\n", res.ArmAPromptTokens)
	fmt.Fprintf(&b, "arm_b_prompt_tokens: %d\n", res.ArmBPromptTokens)
	fmt.Fprintf(&b, "token_reduction: %.4f (threshold: >= %.2f)\n", res.TokenReduction, TB4MinTokenReduction)
	fmt.Fprintf(&b, "vdso_hits: %d\n", res.VDSOHits)
	fmt.Fprintf(&b, "thresholds_met: %t\n", res.ThresholdsMet)
	if res.Error != "" {
		fmt.Fprintf(&b, "error: %s\n", res.Error)
	}
	b.WriteString("\n--- Telemetry Deltas ---\n")
	for k, v := range res.TelemetryDeltas {
		fmt.Fprintf(&b, "%s: %.2f\n", k, v)
	}
	if res.MarkdownReport != "" {
		b.WriteString("\n--- Comparative Report ---\n")
		b.WriteString(res.MarkdownReport)
		b.WriteString("\n")
	}
	b.WriteString("================================================================================\n")

	if err := os.WriteFile(logPath, []byte(b.String()), 0644); err != nil {
		return "", fmt.Errorf("failed to write nightrun log: %w", err)
	}
	res.LogPath = logPath
	return logPath, nil
}

type compareJSONSummary struct {
	Benchmark      string  `json:"benchmark"`
	SolveRateDelta float64 `json:"solve_rate_delta"`
	ArmAMetrics    struct {
		Official struct {
			SolveRate float64 `json:"solve_rate"`
		} `json:"official"`
		Telemetry struct {
			TotalPromptTokens int64 `json:"total_prompt_tokens"`
			VDSOHits          int64 `json:"vdso_hits"`
		} `json:"telemetry"`
	} `json:"arm_a_metrics"`
	ArmBMetrics struct {
		Official struct {
			SolveRate float64 `json:"solve_rate"`
		} `json:"official"`
		Telemetry struct {
			TotalPromptTokens int64 `json:"total_prompt_tokens"`
		} `json:"telemetry"`
	} `json:"arm_b_metrics"`
}

func runSubprocessTB4(ctx context.Context, root, datasetPath, outDir string) (*TB4RegressionResult, error) {
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

	// 1. Run
	if err := execCommand("bench", "tb4", "run", "--arm", "both", "--mock", "--dataset", datasetPath, "--out", outDir); err != nil {
		return nil, fmt.Errorf("tb4 run failed: %w", err)
	}

	// 2. Eval
	if err := execCommand("bench", "tb4", "eval", "--run-dir", outDir, "--dataset", datasetPath); err != nil {
		return nil, fmt.Errorf("tb4 eval failed: %w", err)
	}

	// 3. Compare
	compareJSON := filepath.Join(outDir, "compare.json")
	compareMD := filepath.Join(outDir, "compare.md")
	fakDir := filepath.Join(outDir, "fak")
	opencodeDir := filepath.Join(outDir, "opencode")
	if err := execCommand("bench", "tb4", "compare", "--fak-dir", fakDir, "--opencode-dir", opencodeDir, "--dataset", datasetPath, "--out-json", compareJSON, "--out-md", compareMD); err != nil {
		return nil, fmt.Errorf("tb4 compare failed: %w", err)
	}

	// Read comparative report
	data, err := os.ReadFile(compareJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", compareJSON, err)
	}

	var summary compareJSONSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return nil, fmt.Errorf("failed to unmarshal compare.json: %w", err)
	}

	mdData, _ := os.ReadFile(compareMD)

	res := &TB4RegressionResult{
		ArmASolveRate:    summary.ArmAMetrics.Official.SolveRate,
		ArmBSolveRate:    summary.ArmBMetrics.Official.SolveRate,
		SolveRateDelta:   summary.SolveRateDelta,
		ArmAPromptTokens: summary.ArmAMetrics.Telemetry.TotalPromptTokens,
		ArmBPromptTokens: summary.ArmBMetrics.Telemetry.TotalPromptTokens,
		VDSOHits:         summary.ArmAMetrics.Telemetry.VDSOHits,
		MarkdownReport:   string(mdData),
	}

	return res, nil
}

func findRepoRoot() string {
	if ws := os.Getenv("FAK_WORKSPACE_ROOT"); ws != "" {
		if _, err := os.Stat(filepath.Join(ws, "go.mod")); err == nil {
			return filepath.Clean(ws)
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Clean(dir)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

func resolveDatasetPath(datasetPath string) string {
	if datasetPath == "" {
		datasetPath = filepath.Join("testdata", "tb4bench", "synthetic_suite.json")
	}
	if _, err := os.Stat(datasetPath); err == nil {
		return datasetPath
	}
	root := findRepoRoot()
	relRoot := filepath.Join(root, datasetPath)
	if _, err := os.Stat(relRoot); err == nil {
		return relRoot
	}
	relParent := filepath.Join("..", "..", datasetPath)
	if _, err := os.Stat(relParent); err == nil {
		return relParent
	}
	return datasetPath
}

// Ensure error interface is satisfied if unused.
var _ = errors.New
