package benchruns

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const CatalogRel = "experiments/benchmark/catalog.json"

type Run map[string]any

type Catalog struct {
	Runs []Run `json:"runs"`
}

type Detail struct {
	Entry    Run            `json:"entry"`
	Manifest map[string]any `json:"manifest,omitempty"`
	Results  map[string]any `json:"results"`
}

type Filter struct {
	Machine   string
	Model     string
	Precision string
	Since     string
	Until     string
}

func LoadCatalog(root string) (Catalog, error) {
	path := filepath.Join(root, filepath.FromSlash(CatalogRel))
	raw, err := os.ReadFile(path)
	if err != nil {
		return Catalog{}, fmt.Errorf("catalog not found at %s: %w", path, err)
	}
	var cat Catalog
	if err := json.Unmarshal(raw, &cat); err != nil {
		return Catalog{}, fmt.Errorf("failed to load catalog: %w", err)
	}
	return cat, nil
}

func FilterRuns(runs []Run, f Filter) []Run {
	var out []Run
	modelQ := strings.ToLower(f.Model)
	for _, r := range runs {
		if f.Machine != "" && stringField(r, "machine_id") != f.Machine {
			continue
		}
		if modelQ != "" && !strings.Contains(strings.ToLower(stringField(r, "model")), modelQ) {
			continue
		}
		if f.Precision != "" && stringField(r, "precision") != f.Precision {
			continue
		}
		ts := stringField(r, "timestamp")
		if f.Since != "" && ts < f.Since {
			continue
		}
		if f.Until != "" && ts > f.Until {
			continue
		}
		out = append(out, r)
	}
	return out
}

func LoadRun(root string, cat Catalog, runID string) (Detail, error) {
	var entry Run
	for _, r := range cat.Runs {
		if stringField(r, "run_id") == runID {
			entry = r
			break
		}
	}
	if entry == nil {
		return Detail{}, fmt.Errorf("run %q not found in catalog", runID)
	}
	runDir := filepath.Join(root, filepath.FromSlash(stringField(entry, "path")))
	d := Detail{Entry: entry, Results: map[string]any{}}
	if m, ok := readJSONMap(filepath.Join(runDir, "manifest.json")); ok {
		d.Manifest = m
	}
	for _, name := range []string{"kernel", "batch", "modelbench", "fleetbench"} {
		if data, ok := readJSONMap(filepath.Join(runDir, name+".json")); ok {
			d.Results[name] = data
		}
	}
	return d, nil
}

func RenderList(runs []Run) string {
	var b bytes.Buffer
	if len(runs) == 0 {
		return "No runs found matching filters\n"
	}
	fmt.Fprintf(&b, "%-50s %-20s %-20s %-8s %12s\n", "RUN ID", "MACHINE", "MODEL", "PREC", "PEAK T/S")
	fmt.Fprintln(&b, strings.Repeat("-", 110))
	sorted := append([]Run(nil), runs...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return stringField(sorted[i], "timestamp") > stringField(sorted[j], "timestamp")
	})
	for _, r := range sorted {
		peak := "-"
		if v, ok := floatField(r, "peak_tok_per_sec"); ok {
			peak = fmt.Sprintf("%.1f", v)
		}
		fmt.Fprintf(&b, "%-50s %-20s %-20s %-8s %12s\n",
			stringField(r, "run_id"), stringField(r, "machine_id"),
			fieldOrDash(r, "model"), fieldOrDash(r, "precision"), peak)
	}
	fmt.Fprintf(&b, "\n%d run(s)\n", len(runs))
	return b.String()
}

func RenderShow(d Detail) string {
	var b bytes.Buffer
	entry := d.Entry
	fmt.Fprintf(&b, "Run: %s\n", stringField(entry, "run_id"))
	fmt.Fprintf(&b, "Machine: %s\n", stringField(entry, "machine_id"))
	fmt.Fprintf(&b, "Timestamp: %s\n\n", fieldOr(entry, "timestamp", "?"))
	if d.Manifest != nil {
		fmt.Fprintln(&b, "Config:")
		cfg, _ := d.Manifest["config"].(map[string]any)
		fmt.Fprintf(&b, "  Batch sizes: %v\n", cfg["batch_sizes"])
		fmt.Fprintf(&b, "  Workers: %s\n", anyOr(cfg["workers"], "?"))
		fmt.Fprintf(&b, "  Decode steps: %s\n\n", anyOr(cfg["decode_steps"], "?"))
	}
	fmt.Fprintln(&b, "Results:")
	if batch, ok := d.Results["batch"].(map[string]any); ok {
		baseline, _ := batch["baseline"].(map[string]any)
		peak, _ := batch["peak"].(map[string]any)
		fmt.Fprintf(&b, "  Baseline (B=1): %.1f tok/s\n", floatAny(baseline["tok_per_sec"]))
		fmt.Fprintf(&b, "  Peak (B=%s): %.1f tok/s\n", anyOr(peak["batch"], "?"), floatAny(peak["agg_tok_per_sec"]))
		fmt.Fprintf(&b, "  Speedup: %.2fx\n", floatAny(peak["speedup_vs_baseline"]))
	} else {
		fmt.Fprintln(&b, "  No batch results")
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Path: %s\n", stringField(entry, "path"))
	return b.String()
}

func RenderCompare(a, b Detail) string {
	var out bytes.Buffer
	fmt.Fprintln(&out, "=== Benchmark Run Comparison ===")
	fmt.Fprintln(&out)
	fmt.Fprintf(&out, "Run 1: %s\n", stringField(a.Entry, "run_id"))
	fmt.Fprintf(&out, "  Machine: %s\n", stringField(a.Entry, "machine_id"))
	fmt.Fprintf(&out, "  Timestamp: %s\n\n", fieldOr(a.Entry, "timestamp", "?"))
	fmt.Fprintf(&out, "Run 2: %s\n", stringField(b.Entry, "run_id"))
	fmt.Fprintf(&out, "  Machine: %s\n", stringField(b.Entry, "machine_id"))
	fmt.Fprintf(&out, "  Timestamp: %s\n\n", fieldOr(b.Entry, "timestamp", "?"))
	b1, ok1 := a.Results["batch"].(map[string]any)
	b2, ok2 := b.Results["batch"].(map[string]any)
	if ok1 && ok2 {
		base1, _ := b1["baseline"].(map[string]any)
		base2, _ := b2["baseline"].(map[string]any)
		peak1, _ := b1["peak"].(map[string]any)
		peak2, _ := b2["peak"].(map[string]any)
		fmt.Fprintln(&out, "Batch Decode Comparison:")
		fmt.Fprintf(&out, "  %-25s %15s %15s %10s\n", "Metric", "Run 1", "Run 2", "Ratio")
		fmt.Fprintln(&out, strings.Repeat("-", 65))
		printRatio(&out, "Baseline (B=1) tok/s", floatAny(base1["tok_per_sec"]), floatAny(base2["tok_per_sec"]))
		printRatio(&out, "Peak tok/s", floatAny(peak1["agg_tok_per_sec"]), floatAny(peak2["agg_tok_per_sec"]))
		fmt.Fprintf(&out, "  %-25s %15s %15s -\n", "Peak batch size", anyOr(peak1["batch"], "0"), anyOr(peak2["batch"], "0"))
		fmt.Fprintf(&out, "  %-25s %14.2fx %14.2fx -\n", "Speedup vs baseline", floatAny(peak1["speedup_vs_baseline"]), floatAny(peak2["speedup_vs_baseline"]))
	}
	return out.String()
}

func Best(runs []Run, model, metric string) (Run, error) {
	filtered := FilterRuns(runs, Filter{Model: model})
	if len(filtered) == 0 {
		return nil, errors.New("no runs found")
	}
	if metric == "" {
		metric = "peak_tok_per_sec"
	}
	if metric != "peak_tok_per_sec" && metric != "speedup" {
		return nil, fmt.Errorf("unknown metric: %s", metric)
	}
	best := filtered[0]
	bestVal, _ := floatField(best, metric)
	for _, r := range filtered[1:] {
		v, _ := floatField(r, metric)
		if v > bestVal {
			best, bestVal = r, v
		}
	}
	return best, nil
}

func RenderBest(r Run, metric string) string {
	if metric == "" {
		metric = "peak_tok_per_sec"
	}
	return fmt.Sprintf("Best run by %s:\n  Run ID: %s\n  Machine: %s\n  Model: %s\n  Value: %s\n  Timestamp: %s\n",
		metric, stringField(r, "run_id"), stringField(r, "machine_id"),
		fieldOr(r, "model", "?"), anyOr(r[metric], "?"), fieldOr(r, "timestamp", "?"))
}

func RenderMarkdownTable(runs []Run) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "| %-20s | %12s | %14s | %8s |\n", "Machine", "Peak T/S", "Baseline T/S", "Speedup")
	fmt.Fprintf(&b, "|%s|%s|%s|%s|\n", strings.Repeat("-", 21), strings.Repeat("-", 14), strings.Repeat("-", 16), strings.Repeat("-", 10))
	for _, r := range runs {
		fmt.Fprintf(&b, "| %-20s | %12.1f | %14.1f | %8.2fx |\n",
			stringField(r, "machine_id"), floatAny(r["peak_tok_per_sec"]),
			floatAny(r["baseline_tok_per_sec"]), floatAny(r["speedup"]))
	}
	return b.String()
}

func RenderSummary(runs []Run, groupBy string) string {
	var b bytes.Buffer
	switch groupBy {
	case "", "machine":
		fmt.Fprintln(&b, "=== Summary by Machine ===")
		fmt.Fprintln(&b)
		groups := groupRuns(runs, "machine_id")
		keys := sortedKeys(groups)
		for _, key := range keys {
			rs := groups[key]
			best, _ := Best(rs, "", "peak_tok_per_sec")
			total := 0.0
			for _, r := range rs {
				v, _ := floatField(r, "peak_tok_per_sec")
				total += v
			}
			fmt.Fprintf(&b, "%s:\n", key)
			fmt.Fprintf(&b, "  Runs: %d\n", len(rs))
			fmt.Fprintf(&b, "  Best peak: %.1f tok/s\n", floatAny(best["peak_tok_per_sec"]))
			fmt.Fprintf(&b, "  Avg peak: %.1f tok/s\n\n", total/float64(len(rs)))
		}
	case "model":
		fmt.Fprintln(&b, "=== Summary by Model ===")
		fmt.Fprintln(&b)
		groups := groupRuns(runs, "model")
		for _, key := range sortedKeys(groups) {
			fmt.Fprintf(&b, "%s: %d run(s)\n", key, len(groups[key]))
		}
	}
	return b.String()
}

func readJSONMap(path string) (map[string]any, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var data map[string]any
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, false
	}
	return data, true
}

func stringField(r Run, key string) string {
	if v, ok := r[key].(string); ok {
		return v
	}
	return ""
}

func fieldOrDash(r Run, key string) string { return fieldOr(r, key, "-") }

func fieldOr(r Run, key, def string) string {
	if s := stringField(r, key); s != "" {
		return s
	}
	return def
}

func floatField(r Run, key string) (float64, bool) {
	v, ok := r[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
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

func floatAny(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case json.Number:
		f, _ := x.Float64()
		return f
	default:
		return 0
	}
}

func anyOr(v any, def string) string {
	if v == nil {
		return def
	}
	switch x := v.(type) {
	case string:
		if x == "" {
			return def
		}
		return x
	case float64:
		if x == float64(int64(x)) {
			return fmt.Sprintf("%.0f", x)
		}
		return fmt.Sprintf("%g", x)
	default:
		return fmt.Sprint(x)
	}
}

func groupRuns(runs []Run, field string) map[string][]Run {
	groups := map[string][]Run{}
	for _, r := range runs {
		key := fieldOr(r, field, "unknown")
		groups[key] = append(groups[key], r)
	}
	return groups
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func printRatio(b *bytes.Buffer, label string, a, c float64) {
	ratio := 0.0
	if a != 0 {
		ratio = c / a
	}
	fmt.Fprintf(b, "  %-25s %15.1f %15.1f %9.2fx\n", label, a, c, ratio)
}
