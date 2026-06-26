package swebench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/secretload"
)

// RunnerType identifies the agent runner being used for a solve run.
type RunnerType string

const (
	// RunnerFleet uses the fak gateway agent (the session-value stack).
	RunnerFleet RunnerType = "fleet"
	// RunnerDeepSWE uses the DeepSWE/R2E-Gym baseline (reference implementation).
	RunnerDeepSWE RunnerType = "deepswe"
	// RunnerMock generates dummy patches for testing the harness.
	RunnerMock RunnerType = "mock"
)

// RunConfig drives a solve run — N instances, with resource limits and output
// capture. A run produces a preds.json (for grading) and a run meta JSON that
// records runner type, timing, and per-instance status.
type RunConfig struct {
	Runner      RunnerType    // fleet | deepswe | mock
	Filter      string        // "smoke" (~5), "l3" (~50), "full" (all 500)
	Limit       int           // cap to first N instances (0 = all from filter)
	MaxSteps    int           // max agent steps per instance (default 50)
	Timeout     time.Duration // per-instance timeout (0 = no limit)
	OutputDir   string        // where preds.json and meta.json land
	DatasetPath string        // optional full dataset path (for real problem statements)
	Difficulty  string        // optional difficulty map path
	// Fleet-specific
	GatewayAddr string      // fak gateway address (default: localhost:8080) — used by the integrator to build Planner
	AllowExec   bool        // allow the fleet agent's `run` (shell) tool — use ONLY in a sandboxed/containerized run
	LintWrites  bool        // lint each agent file write with the kernel's language-server packs, feeding parse/compile errors back to the model (off => benchmark behavior unchanged)
	Planner     CodePlanner `json:"-"` // injected by the integrator (cmd/fak) for the fleet runner; nil => fleet errors
	// DeepSWE-specific
	DeepSWERepo string // path to R2E-Gym/DeepSWE repo (for local baseline)
	Model       string // model endpoint or path (for DeepSWE)
}

// DefaultRunConfig returns a production-ready config for a quick smoke test.
func DefaultRunConfig() RunConfig {
	return RunConfig{
		Runner:    RunnerFleet,
		Filter:    "smoke",
		MaxSteps:  50,
		Timeout:   10 * time.Minute,
		OutputDir: "",
	}
}

// RunResult is the outcome of a solve run — the predictions file plus metadata.
type RunResult struct {
	PredictionsPath string        // path to preds.json
	MetaPath        string        // path to meta.json
	Predictions     []Prediction  // the predictions (folded in for convenience)
	Meta            RunMeta       // run metadata
	Elapsed         time.Duration // wall-clock time for the run
}

// RunMeta captures run-level metadata for reproducibility and comparison.
type RunMeta struct {
	Runner         RunnerType     `json:"runner"`
	Filter         string         `json:"filter"`
	StartedAt      string         `json:"started_at"`
	CompletedAt    string         `json:"completed_at"`
	ElapsedSec     float64        `json:"elapsed_sec"`
	TotalInstances int            `json:"total_instances"`
	DoneInstances  int            `json:"done_instances"`
	Skipped        int            `json:"skipped"`
	Failed         int            `json:"failed"`
	InstanceMeta   []InstanceMeta `json:"instance_meta,omitempty"`
	Config         RunConfig      `json:"config"`
}

// InstanceMeta records per-instance metadata from a run.
type InstanceMeta struct {
	InstanceID string  `json:"instance_id"`
	Status     string  `json:"status"` // "done", "failed", "skipped", "timeout"
	Steps      int     `json:"steps"`
	ElapsedSec float64 `json:"elapsed_sec"`
	Error      string  `json:"error,omitempty"`
	PatchSize  int     `json:"patch_size,omitempty"`
}

// Run executes a solve run per cfg and returns the result. It loads the
// instance set, applies the filter, runs the selected runner over each instance,
// writes preds.json + meta.json, and returns the full artifact.
func Run(ctx context.Context, cfg RunConfig) (*RunResult, error) {
	start := time.Now()
	if cfg.Runner == "" {
		cfg.Runner = RunnerFleet
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 50
	}
	if cfg.OutputDir == "" {
		cfg.OutputDir = fmt.Sprintf("swebench-run-%s-%s", cfg.Runner, time.Now().Format("20060102T150405Z"))
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return nil, fmt.Errorf("mkdir output: %w", err)
	}

	// Load instances.
	d, _, err := loadSwebenchInstances(cfg.Difficulty, cfg.DatasetPath)
	if err != nil {
		return nil, fmt.Errorf("load instances: %w", err)
	}

	// Apply filter.
	d = applyFilter(d, cfg.Filter)
	if cfg.Limit > 0 && cfg.Limit < d.Len() {
		d = d.Limit(cfg.Limit)
	}

	// Select runner strategy.
	strat, err := newRunnerStrategy(cfg.Runner, cfg)
	if err != nil {
		return nil, fmt.Errorf("runner strategy: %w", err)
	}

	// Run each instance.
	preds := make([]Prediction, 0, d.Len())
	meta := make([]InstanceMeta, 0, d.Len())
	done, skipped, failed := 0, 0, 0

	for _, in := range d.Instances {
		instStart := time.Now()
		instMeta := InstanceMeta{InstanceID: in.InstanceID}

		// Check context cancellation.
		if ctx.Err() != nil {
			instMeta.Status = "skipped"
			instMeta.Error = "run canceled"
			skipped++
			meta = append(meta, instMeta)
			break
		}

		// Run the instance. A zero timeout means "no per-instance limit": a
		// context.WithTimeout(ctx, 0) would carry an already-expired deadline and
		// fail every instance, contradicting the documented "0 = no limit".
		instCtx, cancel := ctx, func() {}
		if cfg.Timeout > 0 {
			instCtx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		}
		pred, err := strat.RunInstance(instCtx, in)
		cancel()

		elapsed := time.Since(instStart)
		instMeta.ElapsedSec = elapsed.Seconds()

		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) || errors.Is(instCtx.Err(), context.DeadlineExceeded) {
				instMeta.Status = "timeout"
			} else {
				instMeta.Status = "failed"
			}
			instMeta.Error = err.Error()
			failed++
			// Still append a dummy prediction so the eval harness has a full set.
			preds = append(preds, Prediction{
				InstanceID:      in.InstanceID,
				ModelNameOrPath: string(cfg.Runner),
				ModelPatch:      "",
			})
		} else {
			instMeta.Status = "done"
			instMeta.PatchSize = len(pred.ModelPatch)
			done++
			preds = append(preds, pred)
		}
		meta = append(meta, instMeta)
	}

	// Write predictions.
	predsPath := filepath.Join(cfg.OutputDir, "predictions.json")
	if err := WritePredictions(predsPath, preds); err != nil {
		return nil, fmt.Errorf("write predictions: %w", err)
	}

	// Write metadata.
	elapsed := time.Since(start)
	runMeta := RunMeta{
		Runner:         cfg.Runner,
		Filter:         cfg.Filter,
		StartedAt:      start.Format(time.RFC3339),
		CompletedAt:    time.Now().Format(time.RFC3339),
		ElapsedSec:     elapsed.Seconds(),
		TotalInstances: d.Len(),
		DoneInstances:  done,
		Skipped:        skipped,
		Failed:         failed,
		InstanceMeta:   meta,
		Config:         cfg,
	}
	metaPath := filepath.Join(cfg.OutputDir, "meta.json")
	metaData, _ := json.MarshalIndent(runMeta, "", "  ")
	if err := os.WriteFile(metaPath, metaData, 0o644); err != nil {
		return nil, fmt.Errorf("write meta: %w", err)
	}

	return &RunResult{
		PredictionsPath: predsPath,
		MetaPath:        metaPath,
		Predictions:     preds,
		Meta:            runMeta,
		Elapsed:         elapsed,
	}, nil
}

// loadSwebenchInstances loads instances from difficulty or dataset paths.
func loadSwebenchInstances(difficulty, dataset string) (*Dataset, string, error) {
	if dataset != "" {
		d, err := LoadDataset(dataset)
		if err != nil {
			return nil, "", err
		}
		if difficulty != "" {
			if dd, _, err := LoadDifficulty(difficulty); err == nil {
				d.MergeDifficulty(dd)
			}
		}
		return d, fmt.Sprintf("dataset %s", dataset), nil
	}
	if difficulty != "" {
		d, _, err := LoadDifficulty(difficulty)
		if err != nil {
			return nil, "", err
		}
		return d, fmt.Sprintf("difficulty %s", difficulty), nil
	}
	return nil, "", fmt.Errorf("pass --difficulty or --dataset")
}

// applyFilter filters the dataset by bucket name (smoke/l3/full).
func applyFilter(d *Dataset, filter string) *Dataset {
	// The real filter would load bench's swebench_verified_l3_smoke.json or
	// swebench_verified_l3.json and intersect. For now, we use simple limits.
	switch filter {
	case "smoke":
		return d.Limit(5)
	case "l3":
		return d.Limit(50)
	case "full":
		return d
	default:
		// Treat unknown filters as full.
		return d
	}
}

// runnerStrategy is the per-runner execution interface.
type runnerStrategy interface {
	RunInstance(ctx context.Context, in Instance) (Prediction, error)
}

// newRunnerStrategy constructs a runnerStrategy for the given type.
func newRunnerStrategy(rt RunnerType, cfg RunConfig) (runnerStrategy, error) {
	switch rt {
	case RunnerMock:
		return &mockRunner{}, nil
	case RunnerFleet:
		return &fleetRunner{cfg: cfg}, nil
	case RunnerDeepSWE:
		return &deepSWERunner{cfg: cfg}, nil
	default:
		return nil, fmt.Errorf("unknown runner: %s", rt)
	}
}

// mockRunner generates dummy predictions for harness testing.
type mockRunner struct{}

// RunInstance returns a fixed dummy patch for the instance, for exercising the harness
// without invoking a real agent.
func (m *mockRunner) RunInstance(ctx context.Context, in Instance) (Prediction, error) {
	// Generate a plausible-looking dummy patch.
	patch := fmt.Sprintf(`# Mock patch for %s (fleet test harness)
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def test():
-    pass
+    return 42
`, in.InstanceID)
	return Prediction{
		InstanceID:      in.InstanceID,
		ModelNameOrPath: "mock",
		ModelPatch:      patch,
	}, nil
}

// fleetRunner (the fak gateway coding agent) lives in fleet.go.

// deepSWERunner executes instances via the DeepSWE baseline (R2E-Gym).
type deepSWERunner struct {
	cfg RunConfig
}

type deepSWERequest struct {
	Schema    string   `json:"schema"`
	Instance  Instance `json:"instance"`
	Model     string   `json:"model,omitempty"`
	MaxSteps  int      `json:"max_steps,omitempty"`
	Repo      string   `json:"repo,omitempty"`
	Runner    string   `json:"runner"`
	StartedAt string   `json:"started_at"`
}

type deepSWEResponse struct {
	InstanceID      string `json:"instance_id"`
	ModelNameOrPath string `json:"model_name_or_path"`
	ModelPatch      string `json:"model_patch"`
	Patch           string `json:"patch,omitempty"`
	Error           string `json:"error,omitempty"`
}

// RunInstance executes a configured DeepSWE/R2E-Gym adapter. The adapter receives
// a JSON request on stdin and must return either a SWE-bench prediction JSON object
// or a unified diff on stdout. Missing adapter configuration is a failed instance,
// not a synthetic prediction.
func (d *deepSWERunner) RunInstance(ctx context.Context, in Instance) (Prediction, error) {
	exe, args, workdir, err := d.adapterCommand()
	if err != nil {
		return Prediction{}, err
	}
	req := deepSWERequest{
		Schema:    "fak.swebench.deepswe-request.v1",
		Instance:  in,
		Model:     d.cfg.Model,
		MaxSteps:  d.cfg.MaxSteps,
		Repo:      d.cfg.DeepSWERepo,
		Runner:    string(RunnerDeepSWE),
		StartedAt: time.Now().UTC().Format(time.RFC3339),
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return Prediction{}, fmt.Errorf("deepswe request encode: %w", err)
	}

	cmd := exec.CommandContext(ctx, exe, args...)
	if workdir != "" {
		cmd.Dir = workdir
	}
	cmd.Stdin = bytes.NewReader(payload)
	cmd.Env = secretload.SandboxEnv(os.Environ(),
		"FAK_SWEBENCH_INSTANCE_ID="+in.InstanceID,
		"FAK_SWEBENCH_REPO="+in.RepoFull(),
		"FAK_SWEBENCH_BASE_COMMIT="+in.BaseCommit,
		"FAK_DEEPSWE_MODEL="+d.cfg.Model,
		"FAK_DEEPSWE_MAX_STEPS="+fmt.Sprint(d.cfg.MaxSteps),
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg != "" {
			return Prediction{}, fmt.Errorf("deepswe adapter failed: %w: %s", err, msg)
		}
		return Prediction{}, fmt.Errorf("deepswe adapter failed: %w", err)
	}
	return parseDeepSWEPrediction(in, d.cfg.Model, stdout.Bytes())
}

func (d *deepSWERunner) adapterCommand() (exe string, args []string, workdir string, err error) {
	if exe = strings.TrimSpace(os.Getenv("FAK_DEEPSWE_RUNNER")); exe != "" {
		return exe, splitCommandArgs(os.Getenv("FAK_DEEPSWE_RUNNER_ARGS")), strings.TrimSpace(d.cfg.DeepSWERepo), nil
	}
	if d.cfg.DeepSWERepo == "" {
		return "", nil, "", fmt.Errorf("deepswe adapter not configured: set FAK_DEEPSWE_RUNNER or pass DeepSWERepo with a fak-deepswe-runner executable")
	}
	for _, name := range deepSWEAdapterNames() {
		candidate := filepath.Join(d.cfg.DeepSWERepo, name)
		if info, statErr := os.Stat(candidate); statErr == nil && !info.IsDir() {
			return candidate, nil, d.cfg.DeepSWERepo, nil
		}
	}
	return "", nil, "", fmt.Errorf("deepswe adapter not found in %s: expected one of %s", d.cfg.DeepSWERepo, strings.Join(deepSWEAdapterNames(), ", "))
}

func deepSWEAdapterNames() []string {
	if runtime.GOOS == "windows" {
		return []string{"fak-deepswe-runner.exe", "fak-deepswe-runner.cmd", "fak-deepswe-runner.bat", "fak-deepswe-runner"}
	}
	return []string{"fak-deepswe-runner"}
}

func splitCommandArgs(s string) []string {
	return strings.Fields(strings.TrimSpace(s))
}

func parseDeepSWEPrediction(in Instance, model string, raw []byte) (Prediction, error) {
	text := strings.TrimSpace(string(raw))
	if text == "" {
		return Prediction{}, fmt.Errorf("deepswe adapter returned empty stdout")
	}
	var resp deepSWEResponse
	if err := json.Unmarshal([]byte(text), &resp); err == nil {
		if resp.Error != "" {
			return Prediction{}, fmt.Errorf("deepswe adapter returned error: %s", resp.Error)
		}
		patch := resp.ModelPatch
		if patch == "" {
			patch = resp.Patch
		}
		return normalizeDeepSWEPrediction(in, model, Prediction{
			InstanceID:      resp.InstanceID,
			ModelNameOrPath: resp.ModelNameOrPath,
			ModelPatch:      patch,
		})
	}
	return normalizeDeepSWEPrediction(in, model, Prediction{
		InstanceID:      in.InstanceID,
		ModelNameOrPath: model,
		ModelPatch:      text,
	})
}

func normalizeDeepSWEPrediction(in Instance, model string, pred Prediction) (Prediction, error) {
	if pred.InstanceID == "" {
		pred.InstanceID = in.InstanceID
	}
	if pred.InstanceID != in.InstanceID {
		return Prediction{}, fmt.Errorf("deepswe adapter returned instance_id %q for %q", pred.InstanceID, in.InstanceID)
	}
	if pred.ModelNameOrPath == "" {
		if model != "" {
			pred.ModelNameOrPath = model
		} else {
			pred.ModelNameOrPath = string(RunnerDeepSWE)
		}
	}
	if strings.TrimSpace(pred.ModelPatch) == "" {
		return Prediction{}, fmt.Errorf("deepswe adapter returned empty patch for %s", in.InstanceID)
	}
	return pred, nil
}
