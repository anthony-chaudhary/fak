package swebench

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Prediction is one model's proposed fix for an instance, in the exact shape the
// official SWE-bench evaluation harness consumes. bench grades a run by shelling
// to `python -m swebench.harness.run_evaluation --predictions_path <dir>/preds.json`
// (see the Benchmark repo's commands/_swebench_grade.py), so emitting this shape
// lets a fak `solve` run be graded by the identical harness — the apples-to-apples
// resolve-rate path on a box with Docker + the dataset (the DGX).
type Prediction struct {
	InstanceID      string `json:"instance_id"`
	ModelNameOrPath string `json:"model_name_or_path"`
	ModelPatch      string `json:"model_patch"` // a unified diff applied to the repo at base_commit
}

// WritePredictions writes preds in the harness's canonical preds.json form: a
// JSON array of {instance_id, model_name_or_path, model_patch}. The harness also
// accepts a dict keyed by instance_id and JSONL; the array form is the most
// widely accepted and is what bench's grader expects at preds.json. Predictions
// are sorted by instance_id for a reproducible file.
func WritePredictions(path string, preds []Prediction) error {
	cp := append([]Prediction(nil), preds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i].InstanceID < cp[j].InstanceID })
	b, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("write predictions %s: %w", path, err)
	}
	return nil
}

// ReadPredictions reads a preds.json (array form OR a dict keyed by instance_id,
// the two shapes the harness accepts) back into a slice. This lets `eval` consume
// predictions produced by an earlier `solve` run — or by any other harness — for
// grading and the comparison report.
func ReadPredictions(path string) ([]Prediction, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Try the array form first.
	var arr []Prediction
	if err := json.Unmarshal(b, &arr); err == nil {
		return arr, nil
	}
	// Fall back to the dict form: {instance_id: {model_name_or_path, model_patch}}.
	var m map[string]Prediction
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("%s: not a predictions array or dict: %w", path, err)
	}
	out := make([]Prediction, 0, len(m))
	for id, p := range m {
		if p.InstanceID == "" {
			p.InstanceID = id
		}
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].InstanceID < out[j].InstanceID })
	return out, nil
}
