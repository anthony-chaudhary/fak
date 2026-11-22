// Adapted from ai-dynamo/aiperf, src/aiperf/orchestrator/aggregation/sweep.py:100
// (identify_pareto_optimal, SweepAnalyzer._resolve_latency_key/_compute_pareto) at
// ab610add4e8876fca223197596c728e17365f381. Apache-2.0; re-expressed as a
// `fak bench-runs pareto` view over catalog records, reusing
// perfledger.ParetoFrontier for the strict-dominance check upstream also uses.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// benchRunsLatencyCandidates is the preference order for the minimized axis:
// TTFT p99 for streaming runs, total request latency p99 when TTFT was never
// emitted (non-streaming endpoints).
var benchRunsLatencyCandidates = []string{"ttft_p99_ms", "request_latency_p99_ms"}

// benchRunsParetoPoint is one frontier member as rendered by the command.
type benchRunsParetoPoint struct {
	RunID      string  `json:"run_id"`
	Machine    string  `json:"machine_id,omitempty"`
	Model      string  `json:"model,omitempty"`
	Throughput float64 `json:"throughput"`
	LatencyMS  float64 `json:"latency_ms"`
}

// benchRunsParetoResult is the frontier plus the axes it was computed on.
type benchRunsParetoResult struct {
	ThroughputKey string                 `json:"throughput_key"`
	LatencyKey    string                 `json:"latency_key"`
	Considered    int                    `json:"considered"`
	Skipped       []string               `json:"skipped,omitempty"`
	Frontier      []benchRunsParetoPoint `json:"frontier"`
}

// resolveBenchRunsLatencyKey picks the first candidate present on every run,
// so the frontier never compares TTFT on one run against request latency on
// another. It returns "" when no candidate is common to all runs.
func resolveBenchRunsLatencyKey(runs []benchruns.Run) string {
	for _, key := range benchRunsLatencyCandidates {
		all := len(runs) > 0
		for _, r := range runs {
			if _, ok := benchRunsFloat(r, key); !ok {
				all = false
				break
			}
		}
		if all {
			return key
		}
	}
	return ""
}

// benchRunsPareto computes the throughput-MAX x latency-MIN frontier. A run
// lacking either metric is skipped (and reported) rather than voiding the
// whole frontier, since a catalog mixes runs from unrelated sweeps.
func benchRunsPareto(runs []benchruns.Run, throughputKey, latencyKey string) benchRunsParetoResult {
	res := benchRunsParetoResult{ThroughputKey: throughputKey, LatencyKey: latencyKey, Frontier: []benchRunsParetoPoint{}}
	byID := make(map[string]benchruns.Run, len(runs))
	points := make([]perfledger.ConfigPoint, 0, len(runs))
	for i, r := range runs {
		id, _ := r["run_id"].(string)
		if id == "" {
			id = fmt.Sprintf("#%d", i)
		}
		tp, okT := benchRunsFloat(r, throughputKey)
		lat, okL := benchRunsFloat(r, latencyKey)
		p := perfledger.ConfigPoint{Config: id, LatencyMS: lat, Throughput: tp}
		if !okT || !okL || lat <= 0 || tp < 0 {
			res.Skipped = append(res.Skipped, id)
			continue
		}
		byID[id] = r
		points = append(points, p)
	}
	res.Considered = len(points)
	for _, p := range perfledger.ParetoFrontier(points) {
		r := byID[p.Config]
		machine, _ := r["machine_id"].(string)
		model, _ := r["model"].(string)
		res.Frontier = append(res.Frontier, benchRunsParetoPoint{
			RunID: p.Config, Machine: machine, Model: model,
			Throughput: p.Throughput, LatencyMS: p.LatencyMS,
		})
	}
	return res
}

func benchRunsFloat(r benchruns.Run, key string) (float64, bool) {
	switch x := r[key].(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func benchRunsParetoCmd(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs pareto", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	model := fs.String("model", "", "filter by model substring")
	machine := fs.String("machine", "", "filter by machine ID")
	precision := fs.String("precision", "", "filter by precision")
	throughput := fs.String("throughput", "peak_tok_per_sec", "catalog field to maximize")
	latency := fs.String("latency", "", "catalog field to minimize (default: first of "+strings.Join(benchRunsLatencyCandidates, ", ")+" present on every run)")
	format := fs.String("format", "table", "output format: table|json")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *format != "table" && *format != "json" {
		fmt.Fprintf(stderr, "fak bench-runs pareto: unknown --format %q\n", *format)
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	runs := benchruns.FilterRuns(cat.Runs, benchruns.Filter{Machine: *machine, Model: *model, Precision: *precision})
	latencyKey := *latency
	if latencyKey == "" {
		latencyKey = resolveBenchRunsLatencyKey(runs)
	}
	if latencyKey == "" {
		fmt.Fprintf(stderr, "fak bench-runs pareto: no latency field (%s) present on every selected run; pass --latency\n", strings.Join(benchRunsLatencyCandidates, ", "))
		return 1
	}
	res := benchRunsPareto(runs, *throughput, latencyKey)
	if *format == "json" {
		_ = writeIndentedJSON(stdout, res)
		return 0
	}
	fmt.Fprintf(stdout, "Pareto frontier: %s (max) x %s (min), %d of %d run(s) considered\n",
		res.ThroughputKey, res.LatencyKey, res.Considered, len(runs))
	for _, p := range res.Frontier {
		fmt.Fprintf(stdout, "  %-24s %-16s %12.2f %12.2f\n", p.RunID, p.Machine, p.Throughput, p.LatencyMS)
	}
	if len(res.Skipped) > 0 {
		fmt.Fprintf(stdout, "skipped (missing %s or %s): %s\n", res.ThroughputKey, res.LatencyKey, strings.Join(res.Skipped, ", "))
	}
	return 0
}
