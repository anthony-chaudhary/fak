package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"
)

// ComparisonRun holds the results of two runner types for side-by-side analysis.
type ComparisonRun struct {
	Schema      string                `json:"schema"`
	GeneratedAt string                `json:"generated_at"`
	Dataset     DatasetRef            `json:"dataset"`
	Runs        map[string]RunResult  `json:"runs"`           // keyed by runner name
	Eval        map[string]EvalResult `json:"eval,omitempty"` // keyed by runner name
	Summary     ComparisonSummary     `json:"summary"`
}

// ComparisonSummary aggregates across runs for the headline comparison.
type ComparisonSummary struct {
	TotalInstances int                `json:"total_instances"`
	Runners        []string           `json:"runners"`
	ResolveRates   map[string]float64 `json:"resolve_rates,omitempty"` // keyed by runner
	Headline       string             `json:"headline"`
}

// CompareConfig drives a comparison run.
type CompareConfig struct {
	Runners     []RunnerType  // which runners to execute (e.g., [RunnerFleet, RunnerDeepSWE])
	Filter      string        // smoke | l3 | full
	Limit       int           // instance cap
	MaxSteps    int           // per-instance step limit
	Timeout     time.Duration // per-instance timeout
	OutputDir   string        // where comparison results land
	DatasetPath string        // optional full dataset
	Difficulty  string        // optional difficulty map
	// Runner-specific configs.
	FleetGateway string      // fleet gateway address
	FleetPlanner CodePlanner // injected by the integrator (cmd/fak) for the fleet runner
	DeepSWEModel string      // deepswe model endpoint
	DeepSWERepo  string      // deepswe repo path
}

// RunComparison executes both runners (or more) and builds the comparison.
func RunComparison(ctx context.Context, cfg CompareConfig) (*ComparisonRun, error) {
	if len(cfg.Runners) == 0 {
		cfg.Runners = []RunnerType{RunnerFleet, RunnerDeepSWE}
	}
	var err error
	cfg.OutputDir, err = ensureOutputDir(cfg.OutputDir,
		fmt.Sprintf("swebench-compare-%s", time.Now().Format("20060102T150405Z")))
	if err != nil {
		return nil, err
	}

	// Load the instance set once (both runners use the same subset).
	d, err := loadFilterLimit(cfg.Difficulty, cfg.DatasetPath, cfg.Filter, cfg.Limit)
	if err != nil {
		return nil, err
	}

	// Build a ComparisonRun skeleton.
	cr := &ComparisonRun{
		Schema:      "swebench-compare/1",
		GeneratedAt: time.Now().Format(time.RFC3339),
		Dataset: DatasetRef{
			Instances:      d.Len(),
			DifficultyDist: difficultyDist(d),
			GeometrySource: geometrySources(d),
			TotalTurns:     totalTurns(d, DefaultGeometryModel()),
		},
		Runs: make(map[string]RunResult),
		Eval: make(map[string]EvalResult),
		Summary: ComparisonSummary{
			TotalInstances: d.Len(),
			Runners:        runnerNames(cfg.Runners),
			ResolveRates:   make(map[string]float64),
		},
	}

	// Run each runner sequentially (could parallelize in the future).
	for _, rt := range cfg.Runners {
		runCfg := buildRunConfig(cfg, rt, cfg.OutputDir)
		res, err := Run(ctx, runCfg)
		if err != nil {
			return nil, fmt.Errorf("runner %s: %w", rt, err)
		}
		cr.Runs[string(rt)] = *res

		// Run evaluation if available.
		evalRes, err := RunEval(EvalConfig{
			PredictionsPath: res.PredictionsPath,
			RunID:           fmt.Sprintf("compare-%s", rt),
		})
		if err == nil {
			cr.Eval[string(rt)] = evalRes
			if evalRes.Available {
				cr.Summary.ResolveRates[string(rt)] = evalRes.ResolveRatePct
			}
		}
	}

	// Compute headline.
	cr.Summary.Headline = computeHeadline(cr.Summary)

	// Write the comparison JSON.
	cmpPath := fmt.Sprintf("%s/comparison.json", cfg.OutputDir)
	cmpData, _ := json.MarshalIndent(cr, "", "  ")
	if err := os.WriteFile(cmpPath, cmpData, 0o644); err != nil {
		return cr, fmt.Errorf("write comparison: %w", err)
	}

	// Write markdown report.
	mdPath := fmt.Sprintf("%s/comparison.md", cfg.OutputDir)
	if err := os.WriteFile(mdPath, []byte(RenderComparisonMarkdown(cr)), 0o644); err != nil {
		return cr, fmt.Errorf("write markdown: %w", err)
	}

	return cr, nil
}

// loadFilterLimit loads the SWE-bench instance set for the given difficulty map
// and optional dataset path, applies the named filter, and caps it to limit
// (limit <= 0 means no cap). It is the shared front of every run/compare entry.
func loadFilterLimit(difficulty, datasetPath, filter string, limit int) (*Dataset, error) {
	d, _, err := loadSwebenchInstances(difficulty, datasetPath)
	if err != nil {
		return nil, fmt.Errorf("load instances: %w", err)
	}
	d = applyFilter(d, filter)
	if limit > 0 && limit < d.Len() {
		d = d.Limit(limit)
	}
	return d, nil
}

// buildRunConfig constructs a RunConfig for a single runner within a comparison.
func buildRunConfig(base CompareConfig, rt RunnerType, outDir string) RunConfig {
	return RunConfig{
		Runner:      rt,
		Filter:      base.Filter,
		Limit:       base.Limit,
		MaxSteps:    base.MaxSteps,
		Timeout:     base.Timeout,
		OutputDir:   fmt.Sprintf("%s/run-%s", outDir, rt),
		DatasetPath: base.DatasetPath,
		Difficulty:  base.Difficulty,
		GatewayAddr: base.FleetGateway,
		Planner:     base.FleetPlanner,
		Model:       base.DeepSWEModel,
		DeepSWERepo: base.DeepSWERepo,
	}
}

// runnerNames converts a RunnerType slice to strings.
func runnerNames(rts []RunnerType) []string {
	out := make([]string, len(rts))
	for i, rt := range rts {
		out[i] = string(rt)
	}
	sort.Strings(out)
	return out
}

// difficultyDist extracts the difficulty distribution from a dataset.
func difficultyDist(d *Dataset) map[string]int {
	m := make(map[string]int)
	for _, in := range d.Instances {
		if in.Difficulty != "" {
			m[in.Difficulty]++
		}
	}
	return m
}

// geometrySources returns a dummy "geometry_sources" map for the comparison.
func geometrySources(d *Dataset) map[string]int {
	return map[string]int{"difficulty_derived": d.Len()}
}

// totalTurns computes total expected turns using a geometry model.
func totalTurns(d *Dataset, gm GeometryModel) int64 {
	var sum int64
	for _, in := range d.Instances {
		g := gm.Derive(in)
		sum += int64(g.Turns)
	}
	return sum
}

// computeHeadline generates a human-readable headline for the comparison.
func computeHeadline(s ComparisonSummary) string {
	if len(s.ResolveRates) == 0 {
		return "No resolve rates available (harness gated on this box)"
	}
	// Find the winner.
	bestRunner := ""
	bestRate := 0.0
	for rt, rate := range s.ResolveRates {
		if rate > bestRate {
			bestRate = rate
			bestRunner = rt
		}
	}
	if bestRunner == "" {
		return "No successful resolve rates"
	}
	return fmt.Sprintf("%s leads with %.1f%% resolve rate", bestRunner, bestRate)
}

// RenderComparisonMarkdown generates a markdown report for the comparison.
func RenderComparisonMarkdown(cr *ComparisonRun) string {
	md := fmt.Sprintf("# Fleet vs DeepSWE — SWE-bench Comparison\n\n")
	md += fmt.Sprintf("_Generated %s_\n\n", cr.GeneratedAt)
	md += fmt.Sprintf("**Dataset:** %d instances\n\n", cr.Dataset.Instances)

	if len(cr.Dataset.DifficultyDist) > 0 {
		md += "### Difficulty Distribution\n\n"
		md += "| Difficulty | Count |\n"
		md += "|-----------|-------|\n"
		for _, d := range sortedDifficultyKeys(cr.Dataset.DifficultyDist) {
			md += fmt.Sprintf("| %s | %d |\n", d, cr.Dataset.DifficultyDist[d])
		}
		md += "\n"
	}

	md += "## Runner Results\n\n"
	for _, rt := range cr.Summary.Runners {
		runRes, ok := cr.Runs[rt]
		if !ok {
			continue
		}
		md += fmt.Sprintf("### %s\n\n", rt)
		md += fmt.Sprintf("- **Status:** %d done, %d failed, %d skipped\n", runRes.Meta.DoneInstances, runRes.Meta.Failed, runRes.Meta.Skipped)
		md += fmt.Sprintf("- **Elapsed:** %.1fs\n", runRes.Elapsed.Seconds())
		md += fmt.Sprintf("- **Predictions:** `%s`\n", runRes.PredictionsPath)

		evalRes, ok := cr.Eval[rt]
		if ok && evalRes.Available {
			md += fmt.Sprintf("- **Resolve Rate:** %.1f%% (%d/%d)\n", evalRes.ResolveRatePct, evalRes.Resolved, evalRes.Total)
		} else if ok {
			md += fmt.Sprintf("- **Resolve Rate:** GATED (%s)\n", evalRes.Reason)
		}
		md += "\n"
	}

	md += "## Summary\n\n"
	md += fmt.Sprintf("**Headline:** %s\n\n", cr.Summary.Headline)

	if len(cr.Summary.ResolveRates) > 0 {
		md += "| Runner | Resolve Rate |\n"
		md += "|--------|-------------|\n"
		for _, rt := range cr.Summary.Runners {
			if rate, ok := cr.Summary.ResolveRates[rt]; ok {
				md += fmt.Sprintf("| %s | %.1f%% |\n", rt, rate)
			} else {
				md += fmt.Sprintf("| %s | — |\n", rt)
			}
		}
		md += "\n"
	}

	md += "---\n\n"
	md += "*Generated by fak swebench*\n"
	return md
}

// sortedDifficultyKeys returns sorted difficulty bucket keys.
func sortedDifficultyKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
