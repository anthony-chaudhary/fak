package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
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
	GatewayAddr string // fak gateway address (default: localhost:8080)
	AllowExec   bool   // allow the fleet agent's `run` (shell) tool — use ONLY in a sandboxed/containerized run
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
			instMeta.Status = "failed"
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

func (d *deepSWERunner) RunInstance(ctx context.Context, in Instance) (Prediction, error) {
	// The DeepSWE/R2E-Gym baseline is not wired yet. Return an error (the Run loop
	// records it as a failed instance with an empty patch) rather than a placeholder
	// patch that would masquerade as a real prediction in preds.json.
	return Prediction{}, fmt.Errorf("deepswe runner not wired (model=%q): point --model at an R2E-Gym/DeepSWE endpoint (baseline integration pending)", d.cfg.Model)
}
