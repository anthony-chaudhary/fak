package webbench

import (
	"fmt"
	"strings"
)

// Comparison is the full fak<->benchmark comparison, mirroring swebench's structure.
type Comparison struct {
	GeneratedAt string         `json:"generated_at"`
	Dataset     *Dataset       `json:"-"`
	Summary     Summary        `json:"summary"`
	Families    []MetricFamily `json:"families"`
	Bench       *BenchSide     `json:"bench,omitempty"` // sidecar if comparing to external benchmark
}

// MetricFamily is one group of metrics reported in the comparison.
type MetricFamily struct {
	Name       string `json:"name"`       // e.g., "Prefill work-elimination"
	Kind       string `json:"kind"`       // "fak-native" | "comparable"
	Provenance string `json:"provenance"` // "computed" | "measured" | "gated"
	Rows       []Row  `json:"rows"`
}

// Row is one data row within a family (e.g., one worker count).
type Row struct {
	Label  string                 `json:"label"`
	Values map[string]interface{} `json:"values"`
}

// BenchSide holds data from an external benchmark run when doing side-by-side.
type BenchSide struct {
	ProfileName      string  `json:"profile_name"`
	SchemaVersion    int     `json:"schema_version"`
	TokenHitRatioPct float64 `json:"token_hit_ratio_pct"`
	Present          bool    `json:"present"`
}

// CompareInputs describes what goes into a comparison.
type CompareInputs struct {
	Dataset     *Dataset
	Geometry    GeometryModel
	Workers     []int
	BenchResult string // optional path to external benchmark results
	// PredictionsPath, when set, is a predictions JSON whose task success rate is folded
	// into the "Task success rate + safety" family via the official harness (RunEval). It
	// is OBSERVED, not computed: the row is "measured" when the harness ran and "gated"
	// (with the specific reason) when this box lacks the harness/runtime — never silently
	// dropped, so the comparison can't claim a success rate it did not actually grade.
	PredictionsPath string
	GeneratedAt     string
}

// BuildComparison constructs the comparison from inputs.
func BuildComparison(in CompareInputs) *Comparison {
	summary := Describe(in.Dataset, in.Geometry, in.Workers)

	c := &Comparison{
		GeneratedAt: in.GeneratedAt,
		Dataset:     in.Dataset,
		Summary:     summary,
	}

	// Build the four metric families (parallel to swebench).
	c.Families = []MetricFamily{
		{
			Name:       "Prefill / KV-reuse work-elimination",
			Kind:       "fak-native",
			Provenance: "computed",
			Rows:       buildPrefillRows(summary.Prefill),
		},
		{
			Name:       "Navigation turns + tokens",
			Kind:       "comparable",
			Provenance: "computed",
			Rows:       buildTurnRows(summary),
		},
		{
			Name:       "In-process adjudication cost",
			Kind:       "fak-native",
			Provenance: "gated", // Would need measurement
			Rows:       []Row{{Label: "Gated on this box - requires trace data"}},
		},
		buildSuccessRateFamily(in.PredictionsPath),
	}

	return c
}

// buildSuccessRateFamily folds a predictions JSON's task success rate into the comparison
// via the official harness (RunEval). With no --predictions it stays the honest gated
// placeholder. With predictions it runs the eval: a "measured" row carrying the real
// passed/total/success-rate when the harness ran, or a "gated" row naming the SPECIFIC
// reason (no harness, no browser runtime) when this box can't grade — the OBSERVED success
// rate is never fabricated and never silently dropped.
func buildSuccessRateFamily(predictionsPath string) MetricFamily {
	fam := MetricFamily{
		Name:       "Task success rate + safety",
		Kind:       "comparable",
		Provenance: "gated", // Requires official harness
		Rows:       []Row{{Label: "Gated on this box - requires harness + model"}},
	}
	if predictionsPath == "" {
		return fam
	}
	res, err := RunEval(EvalConfig{PredictionsPath: predictionsPath, RunID: "webbench-compare"})
	if err != nil {
		fam.Rows = []Row{{
			Label:  "Eval errored",
			Values: map[string]interface{}{"predictions": predictionsPath, "error": err.Error()},
		}}
		return fam
	}
	if !res.Available {
		fam.Rows = []Row{{
			Label:  "Gated",
			Values: map[string]interface{}{"predictions": predictionsPath, "reason": res.Reason, "command": res.Command},
		}}
		return fam
	}
	fam.Provenance = "measured"
	fam.Rows = []Row{{
		Label: "Official harness",
		Values: map[string]interface{}{
			"predictions":      predictionsPath,
			"passed":           res.Passed,
			"total":            res.Total,
			"success_rate_pct": fmt.Sprintf("%.1f%%", res.SuccessRatePct),
		},
	}}
	return fam
}

func buildPrefillRows(prefill []PrefillRow) []Row {
	rows := make([]Row, len(prefill))
	for i, p := range prefill {
		rows[i] = Row{
			Label: fmt.Sprintf("%d workers", p.Workers),
			Values: map[string]interface{}{
				"A naive":            p.ANaive,
				"B per-agent KV":     p.BAgent,
				"C fak fused":        p.CFak,
				"A/C (net)":          fmt.Sprintf("%.1fx", p.AOverC),
				"B/C (cross-worker)": fmt.Sprintf("%.2fx", p.BOverC),
				"A/B (turn-tax)":     fmt.Sprintf("%.1fx", p.AOverB),
			},
		}
	}
	return rows
}

func buildTurnRows(s Summary) []Row {
	return []Row{
		{
			Label: "Aggregate",
			Values: map[string]interface{}{
				"Total turns":  s.TotalTurns,
				"Min turns":    s.TurnsMin,
				"Median turns": s.TurnsMedian,
				"Max turns":    s.TurnsMax,
			},
		},
	}
}

// RenderMarkdown generates a house-style markdown report.
func RenderMarkdown(c *Comparison) string {
	var b strings.Builder

	b.WriteString("# Frontier WebBench Comparison Report\n\n")
	b.WriteString(fmt.Sprintf("**Generated:** %s\n\n", c.GeneratedAt))
	b.WriteString(fmt.Sprintf("**Instances:** %d\n\n", c.Summary.Instances))

	// Turn statistics.
	b.WriteString("## Navigation Turn Statistics\n\n")
	b.WriteString("| Metric | Value |\n")
	b.WriteString("|-------|-------|\n")
	b.WriteString(fmt.Sprintf("| Total turns | %d |\n", c.Summary.TotalTurns))
	b.WriteString(fmt.Sprintf("| Min turns | %d |\n", c.Summary.TurnsMin))
	b.WriteString(fmt.Sprintf("| Median turns | %d |\n", c.Summary.TurnsMedian))
	b.WriteString(fmt.Sprintf("| Max turns | %d |\n", c.Summary.TurnsMax))
	b.WriteString("\n")

	// Difficulty distribution.
	if len(c.Summary.DifficultyDist) > 0 {
		b.WriteString("### Difficulty Distribution\n\n")
		b.WriteString("| Difficulty | Count |\n")
		b.WriteString("|------------|-------|\n")
		for _, diff := range sortedKeys(c.Summary.DifficultyDist) {
			b.WriteString(fmt.Sprintf("| %s | %d |\n", diff, c.Summary.DifficultyDist[diff]))
		}
		b.WriteString("\n")
	}

	// Category distribution.
	if len(c.Summary.CategoryDist) > 0 {
		b.WriteString("### Category Distribution\n\n")
		b.WriteString("| Category | Count |\n")
		b.WriteString("|----------|-------|\n")
		for _, cat := range sortedKeys(c.Summary.CategoryDist) {
			b.WriteString(fmt.Sprintf("| %s | %d |\n", cat, c.Summary.CategoryDist[cat]))
		}
		b.WriteString("\n")
	}

	// Prefill work-elimination table (the core value-stack floor).
	b.WriteString("## Prefill-Token Work-Elimination (deterministic floor)\n\n")
	b.WriteString("**The cost comparison arms:**\n\n")
	b.WriteString("- **A (naive):** Re-prefill full context every turn, every worker\n")
	b.WriteString("- **B (per-agent KV):** Each agent reuses its own prefix, no cross-worker sharing\n")
	b.WriteString("- **C (fak fused):** Shared prefix across all agents, cross-worker reuse\n\n")

	b.WriteString("| Workers | A naive | B per-agent KV | C fak fused | A/C (net) | B/C (cross-worker) | A/B (turn-tax) |\n")
	b.WriteString("|---------|---------|----------------|-------------|-----------|---------------------|----------------|\n")
	for _, p := range c.Summary.Prefill {
		b.WriteString(fmt.Sprintf("| %d | %s | %s | %s | %.1fx | %.2fx | %.1fx |\n",
			p.Workers,
			formatTokens(p.ANaive),
			formatTokens(p.BAgent),
			formatTokens(p.CFak),
			p.AOverC,
			p.BOverC,
			p.AOverB,
		))
	}
	b.WriteString("\n")

	b.WriteString("**Interpretation:**\n\n")
	b.WriteString("- **A/C** = Net prefill work-elimination vs the naive re-prefill-every-turn harness\n")
	b.WriteString("- **B/C** = Cross-worker prefix reuse (the value stack; bites at workers>1)\n")
	b.WriteString("- **A/B** = The turn-tax (re-prefill vs KV persistence), worker-independent\n\n")

	// Metric families.
	b.WriteString("## Metric Families\n\n")
	for _, fam := range c.Families {
		b.WriteString(fmt.Sprintf("### %s\n\n", fam.Name))
		b.WriteString(fmt.Sprintf("**Kind:** %s  |  **Provenance:** %s\n\n", fam.Kind, fam.Provenance))
		for _, row := range fam.Rows {
			b.WriteString(fmt.Sprintf("- %s\n", row.Label))
		}
		b.WriteString("\n")
	}

	// Bench sidecar if present.
	if c.Bench != nil && c.Bench.Present {
		b.WriteString("## External Benchmark Sidecar\n\n")
		b.WriteString(fmt.Sprintf("**Profile:** %s (schema v%d)\n\n", c.Bench.ProfileName, c.Bench.SchemaVersion))
		b.WriteString(fmt.Sprintf("**Token hit ratio:** %.1f%%\n\n", c.Bench.TokenHitRatioPct))
	}

	b.WriteString("---\n\n")
	b.WriteString("*Generated by fak webbench*\n")

	return b.String()
}

func formatTokens(n int64) string {
	if n < 1_000_000 {
		return fmt.Sprintf("%d", n)
	}
	m := float64(n) / 1_000_000
	if m < 1000 {
		return fmt.Sprintf("%.1f M", m)
	}
	g := m / 1000
	return fmt.Sprintf("%.2f G", g)
}

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Simple sort.
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if keys[i] > keys[j] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	return keys
}
