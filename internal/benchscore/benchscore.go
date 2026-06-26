package benchscore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const Schema = "fak.benchscore-report.v1"
const scoreSchema = "fak.arm64-qkernel-score.v1"

type Report struct {
	Schema   string         `json:"schema"`
	Root     string         `json:"root"`
	Rows     []Row          `json:"rows"`
	Models   []ModelSummary `json:"models"`
	Issues   []Issue        `json:"issues,omitempty"`
	Warnings []Issue        `json:"warnings,omitempty"`
}

type Row struct {
	Path         string  `json:"path"`
	Machine      string  `json:"machine,omitempty"`
	CapturedAt   string  `json:"captured_at,omitempty"`
	Model        string  `json:"model,omitempty"`
	SourceKind   string  `json:"source_kind,omitempty"`
	Status       string  `json:"status,omitempty"`
	Workload     string  `json:"workload"`
	Metric       string  `json:"metric"`
	Value        float64 `json:"value"`
	ValueUnit    string  `json:"value_unit,omitempty"`
	Baseline     float64 `json:"baseline,omitempty"`
	Speedup      float64 `json:"speedup,omitempty"`
	Verification string  `json:"verification,omitempty"`
}

type ModelSummary struct {
	Model       string `json:"model"`
	Rows        int    `json:"rows"`
	Accepted    int    `json:"accepted"`
	Negative    int    `json:"negative"`
	Exploratory int    `json:"exploratory"`
}

type Issue struct {
	Path    string `json:"path"`
	Field   string `json:"field,omitempty"`
	Message string `json:"message"`
}

type doc map[string]any

type meta struct {
	path         string
	machine      string
	capturedAt   string
	model        string
	sourceKind   string
	status       string
	verification string
}

type collector struct {
	meta     meta
	rows     []Row
	issues   []Issue
	warnings []Issue
}

func Scan(root string) (Report, error) {
	var report Report
	report.Schema = Schema
	report.Root = filepath.ToSlash(root)

	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || entry.Name() != "score.json" {
			return nil
		}
		rows, issues, warnings, err := scanFile(path)
		if err != nil {
			report.Issues = append(report.Issues, Issue{
				Path:    filepath.ToSlash(path),
				Message: err.Error(),
			})
			return nil
		}
		report.Rows = append(report.Rows, rows...)
		report.Issues = append(report.Issues, issues...)
		report.Warnings = append(report.Warnings, warnings...)
		return nil
	})
	if err != nil {
		return report, err
	}
	sort.Slice(report.Rows, func(i, j int) bool {
		a, b := report.Rows[i], report.Rows[j]
		return rowKey(a) < rowKey(b)
	})
	sort.Slice(report.Issues, func(i, j int) bool {
		return issueKey(report.Issues[i]) < issueKey(report.Issues[j])
	})
	sort.Slice(report.Warnings, func(i, j int) bool {
		return issueKey(report.Warnings[i]) < issueKey(report.Warnings[j])
	})
	report.Models = summarizeModels(report.Rows)
	return report, nil
}

func scanFile(path string) ([]Row, []Issue, []Issue, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, err
	}
	var root doc
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return nil, nil, nil, err
	}
	if got := stringAt(root, "schema"); got != scoreSchema {
		return nil, nil, nil, nil
	}
	c := collector{meta: extractMeta(path, root)}
	c.checkAcceptedVerification()
	c.checkKernelRatios(root)
	c.checkFullModelDecode(root)
	c.checkPrefill(root)
	c.checkEndToEnd(root)
	c.checkDecodeProbes(root)
	c.checkExploratoryDecode(root)
	if len(c.rows) == 0 {
		c.warn("", "score schema found but no recognized benchmark rows were extracted")
	}
	return c.rows, c.issues, c.warnings, nil
}

func extractMeta(path string, root doc) meta {
	m := meta{
		path:         filepath.ToSlash(path),
		machine:      firstNonEmpty(stringAt(root, "machine"), stringAt(root, "host")),
		capturedAt:   firstNonEmpty(stringAt(root, "captured_at"), stringAt(root, "generated_at")),
		model:        firstNonEmpty(stringAt(root, "model", "name"), stringAt(root, "full_model", "model")),
		sourceKind:   stringAt(root, "model", "source_kind"),
		status:       stringAt(root, "interpretation", "status"),
		verification: stringAt(root, "verification", "status"),
	}
	return m
}

func (c *collector) checkAcceptedVerification() {
	if strings.HasPrefix(c.meta.status, "accepted_") && c.meta.verification != "pass" {
		c.issue("verification.status", fmt.Sprintf("accepted status %q requires verification.status=pass, got %q", c.meta.status, c.meta.verification))
	}
}

func (c *collector) checkKernelRatios(root doc) {
	kernelBench := arrayAt(root, "kernel_bench")
	ratios := arrayAt(root, "kernel_ratios")
	if len(kernelBench) == 0 || len(ratios) == 0 {
		return
	}
	byShapeKernel := map[string]float64{}
	for _, raw := range kernelBench {
		row, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		shape := stringAt(row, "shape")
		kernel := stringAt(row, "kernel")
		mac, ok := floatAt(row, "mac_per_ns")
		if shape != "" && kernel != "" && ok {
			byShapeKernel[shape+"\x00"+kernel] = mac
		}
	}
	for _, raw := range ratios {
		row, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		shape := stringAt(row, "shape")
		amort := byShapeKernel[shape+"\x00amort"]
		if unroll := byShapeKernel[shape+"\x00unroll4"]; amort > 0 && unroll > 0 {
			if got, ok := floatAt(row, "amort_over_unroll4"); ok {
				c.checkNear("kernel_ratios."+shape+".amort_over_unroll4", got, amort/unroll)
			}
		}
		if asm := byShapeKernel[shape+"\x00asm"]; amort > 0 && asm > 0 {
			if got, ok := floatAt(row, "amort_over_asm"); ok {
				c.checkNear("kernel_ratios."+shape+".amort_over_asm", got, amort/asm)
			}
		}
	}
}

func (c *collector) checkFullModelDecode(root doc) {
	baseline, ok := floatAt(root, "baseline", "decode_tok_per_sec")
	if !ok || baseline <= 0 {
		return
	}
	defTok, defOK := floatAt(root, "full_model", "default_decode_tok_per_sec")
	amortTok, amortOK := floatAt(root, "full_model", "amort_decode_tok_per_sec")
	if defOK {
		if got, ok := floatAt(root, "improvement", "default_over_baseline"); ok {
			c.checkNear("improvement.default_over_baseline", got, defTok/baseline)
			c.addRow("decode", "default_over_baseline", defTok, "tok/s", baseline, got)
		}
	}
	if amortOK {
		if got, ok := floatAt(root, "improvement", "amort_over_baseline"); ok {
			c.checkNear("improvement.amort_over_baseline", got, amortTok/baseline)
			c.addRow("decode", "amort_over_baseline", amortTok, "tok/s", baseline, got)
		}
	}
	if defOK && amortOK {
		if got, ok := floatAt(root, "full_model", "amort_over_default"); ok {
			c.checkNear("full_model.amort_over_default", got, amortTok/defTok)
		}
	}
}

func (c *collector) checkPrefill(root doc) {
	baseline, ok := floatAt(root, "baseline", "fak_cpu_q8_prefill_p256_tok_per_sec")
	if !ok || baseline <= 0 {
		return
	}
	tok, ok := prefillTokPerSec(objectAt(root, "results"), 256)
	if !ok {
		return
	}
	speedup, ok := floatAt(root, "improvement", "prefill_over_fak_cpu_q8")
	if !ok {
		return
	}
	c.checkNear("improvement.prefill_over_fak_cpu_q8", speedup, tok/baseline)
	c.addRow("prefill", "P256_over_fak_cpu_q8", tok, "tok/s", baseline, speedup)

	if llamaBaseline, ok := floatAt(root, "baseline", "llamacpp_cpu_prefill_p256_tok_per_sec"); ok && llamaBaseline > 0 {
		if got, ok := floatAt(root, "improvement", "prefill_over_llamacpp_cpu"); ok {
			c.checkNear("improvement.prefill_over_llamacpp_cpu", got, tok/llamaBaseline)
		}
	}
	if decodeTok, ok := floatAt(root, "results", "decode", "tok_per_sec"); ok {
		if decodeBase, ok := floatAt(root, "baseline", "fak_cpu_q8_decode_tok_per_sec"); ok && decodeBase > 0 {
			if got, ok := floatAt(root, "improvement", "decode_over_fak_cpu_q8"); ok {
				c.checkNear("improvement.decode_over_fak_cpu_q8", got, decodeTok/decodeBase)
			}
		}
	}
}

func (c *collector) checkEndToEnd(root doc) {
	rows := arrayAt(root, "end_to_end")
	if len(rows) == 0 {
		return
	}
	decodeSteps, _ := floatAt(root, "workload", "decode_steps")
	cpuDecodeMS, cpuOK := floatAt(root, "arms", "cpu_q8", "decode", "per_token_median_ms")
	metalDecodeMS, metalOK := floatAt(root, "arms", "metal", "decode", "per_token_median_ms")
	if !cpuOK || !metalOK {
		c.issue("arms.*.decode.per_token_median_ms", "end_to_end rows require CPU and Metal decode per-token timings")
		return
	}
	for i, raw := range rows {
		row, ok := raw.(map[string]any)
		if !ok {
			c.issue(fmt.Sprintf("end_to_end[%d]", i), "row is not an object")
			continue
		}
		promptTokens, ok := floatAt(row, "prefill_tokens")
		if !ok {
			c.issue(fmt.Sprintf("end_to_end[%d].prefill_tokens", i), "missing prompt token count")
			continue
		}
		steps := decodeSteps
		if rowSteps, ok := floatAt(row, "decode_steps"); ok {
			steps = rowSteps
		}
		cpuPrefillMS, cpuPrefillOK := prefillMedianMS(objectAt(root, "arms", "cpu_q8"), int(promptTokens))
		metalPrefillMS, metalPrefillOK := prefillMedianMS(objectAt(root, "arms", "metal"), int(promptTokens))
		if !cpuPrefillOK || !metalPrefillOK {
			c.issue(fmt.Sprintf("end_to_end[%d]", i), fmt.Sprintf("missing prefill median for P%d", int(promptTokens)))
			continue
		}
		wantCPU := cpuPrefillMS + steps*cpuDecodeMS
		wantMetal := metalPrefillMS + steps*metalDecodeMS
		if got, ok := floatAt(row, "cpu_q8_ms"); ok {
			c.checkNear(fmt.Sprintf("end_to_end[%d].cpu_q8_ms", i), got, wantCPU)
		}
		if got, ok := floatAt(row, "metal_ms"); ok {
			c.checkNear(fmt.Sprintf("end_to_end[%d].metal_ms", i), got, wantMetal)
		}
		if got, ok := floatAt(row, "speedup"); ok {
			c.checkNear(fmt.Sprintf("end_to_end[%d].speedup", i), got, wantCPU/wantMetal)
			metric := fmt.Sprintf("P%d+D%d_over_cpu_q8", int(promptTokens), int(steps))
			c.addRow("prompt-heavy-e2e", metric, wantMetal, "ms", wantCPU, got)
		}
	}
}

func (c *collector) checkDecodeProbes(root doc) {
	probes := objectAt(root, "probes")
	if probes == nil {
		return
	}
	baseline, _ := floatAt(root, "baseline", "decode_tok_per_sec")
	names := make([]string, 0, len(probes))
	for name := range probes {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		probe, ok := probes[name].(map[string]any)
		if !ok {
			c.issue("probes."+name, "probe is not an object")
			continue
		}
		if name == "q8_gguf_lean" {
			tok, tokOK := floatAt(probe, "decode_tok_per_sec")
			speedup, speedOK := floatAt(probe, "speedup_vs_canonical_q8")
			if tokOK && speedOK && baseline > 0 {
				c.checkNear("probes."+name+".speedup_vs_canonical_q8", speedup, tok/baseline)
				c.addRow("decode", name+"_vs_canonical_q8", tok, "tok/s", baseline, speedup)
			}
			continue
		}
		peak := objectAt(probe, "peak")
		if peak == nil {
			continue
		}
		agg, aggOK := floatAt(peak, "agg_tok_per_sec")
		if !aggOK {
			continue
		}
		if points := arrayAt(probe, "points"); len(points) > 0 {
			maxAgg := math.Inf(-1)
			for _, raw := range points {
				if point, ok := raw.(map[string]any); ok {
					if pAgg, ok := floatAt(point, "agg_tok_per_sec"); ok && pAgg > maxAgg {
						maxAgg = pAgg
					}
				}
			}
			if !math.IsInf(maxAgg, -1) {
				c.checkNear("probes."+name+".peak.agg_tok_per_sec", agg, maxAgg)
			}
		}
		if baseB1, ok := floatAt(probe, "baseline_b1_tok_per_sec"); ok && baseB1 > 0 {
			if got, ok := floatAt(peak, "speedup_vs_in_run_b1"); ok {
				c.checkNear("probes."+name+".peak.speedup_vs_in_run_b1", got, agg/baseB1)
			}
		}
		if baseline > 0 {
			if got, ok := floatAt(peak, "speedup_vs_canonical_q8"); ok {
				c.checkNear("probes."+name+".peak.speedup_vs_canonical_q8", got, agg/baseline)
				c.addRow("decode", name+"_vs_canonical_q8", agg, "tok/s", baseline, got)
			}
		}
	}
}

func (c *collector) checkExploratoryDecode(root doc) {
	if len(c.rows) != 0 {
		return
	}
	tok, ok := floatAt(root, "results", "decode", "tok_per_sec")
	if !ok {
		return
	}
	metric := "exploratory_tok_per_sec"
	if q4k, ok := valueAt(root, "model", "q4k"); ok {
		if b, _ := q4k.(bool); b {
			metric = "q4k_exploratory_tok_per_sec"
		}
	}
	c.addRow("decode", metric, tok, "tok/s", 0, 0)
}

func (c *collector) addRow(workload, metric string, value float64, unit string, baseline, speedup float64) {
	c.rows = append(c.rows, Row{
		Path:         c.meta.path,
		Machine:      c.meta.machine,
		CapturedAt:   c.meta.capturedAt,
		Model:        c.meta.model,
		SourceKind:   c.meta.sourceKind,
		Status:       c.meta.status,
		Workload:     workload,
		Metric:       metric,
		Value:        value,
		ValueUnit:    unit,
		Baseline:     baseline,
		Speedup:      speedup,
		Verification: c.meta.verification,
	})
}

func (c *collector) checkNear(field string, got, want float64) {
	if near(got, want) {
		return
	}
	c.issue(field, fmt.Sprintf("got %.9g, want %.9g", got, want))
}

func near(got, want float64) bool {
	if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		return false
	}
	tol := math.Max(1e-4, math.Abs(want)*5e-4)
	return math.Abs(got-want) <= tol
}

func (c *collector) issue(field, message string) {
	c.issues = append(c.issues, Issue{Path: c.meta.path, Field: field, Message: message})
}

func (c *collector) warn(field, message string) {
	c.warnings = append(c.warnings, Issue{Path: c.meta.path, Field: field, Message: message})
}

func prefillTokPerSec(root map[string]any, tokens int) (float64, bool) {
	for _, raw := range arrayAt(root, "prefill") {
		row, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		tok, ok := floatAt(row, "tokens")
		if !ok || int(tok) != tokens {
			continue
		}
		return floatAt(row, "tok_per_sec")
	}
	return 0, false
}

func prefillMedianMS(root map[string]any, tokens int) (float64, bool) {
	for _, raw := range arrayAt(root, "prefill") {
		row, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		tok, ok := floatAt(row, "tokens")
		if !ok || int(tok) != tokens {
			continue
		}
		return floatAt(row, "median_ms")
	}
	return 0, false
}

func objectAt(root map[string]any, path ...string) map[string]any {
	v, ok := valueAt(root, path...)
	if !ok {
		return nil
	}
	obj, _ := v.(map[string]any)
	return obj
}

func arrayAt(root map[string]any, path ...string) []any {
	v, ok := valueAt(root, path...)
	if !ok {
		return nil
	}
	arr, _ := v.([]any)
	return arr
}

func stringAt(root map[string]any, path ...string) string {
	v, ok := valueAt(root, path...)
	if !ok {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	default:
		return ""
	}
}

func floatAt(root map[string]any, path ...string) (float64, bool) {
	v, ok := valueAt(root, path...)
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return f, err == nil
	case float64:
		return x, true
	case int:
		return float64(x), true
	default:
		return 0, false
	}
}

func valueAt(root map[string]any, path ...string) (any, bool) {
	var cur any = root
	for _, key := range path {
		obj, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		cur, ok = obj[key]
		if !ok {
			return nil, false
		}
	}
	return cur, true
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func summarizeModels(rows []Row) []ModelSummary {
	byModel := map[string]*ModelSummary{}
	for _, row := range rows {
		model := row.Model
		if model == "" {
			model = "(unknown)"
		}
		s := byModel[model]
		if s == nil {
			s = &ModelSummary{Model: model}
			byModel[model] = s
		}
		s.Rows++
		switch {
		case strings.HasPrefix(row.Status, "accepted_"):
			s.Accepted++
		case strings.Contains(row.Status, "negative"):
			s.Negative++
		case strings.Contains(row.Status, "exploratory"):
			s.Exploratory++
		}
	}
	out := make([]ModelSummary, 0, len(byModel))
	for _, s := range byModel {
		out = append(out, *s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}

func RenderMarkdown(report Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Benchmark score matrix\n\n")
	fmt.Fprintf(&b, "- root: `%s`\n", report.Root)
	fmt.Fprintf(&b, "- rows: %d\n", len(report.Rows))
	fmt.Fprintf(&b, "- models: %d\n", len(report.Models))
	fmt.Fprintf(&b, "- issues: %d\n\n", len(report.Issues))

	if len(report.Models) > 0 {
		fmt.Fprintf(&b, "## Models\n\n")
		fmt.Fprintf(&b, "| model | rows | accepted | negative | exploratory |\n")
		fmt.Fprintf(&b, "|---|---:|---:|---:|---:|\n")
		for _, m := range report.Models {
			fmt.Fprintf(&b, "| %s | %d | %d | %d | %d |\n", m.Model, m.Rows, m.Accepted, m.Negative, m.Exploratory)
		}
		fmt.Fprintf(&b, "\n")
	}

	if len(report.Rows) > 0 {
		fmt.Fprintf(&b, "## Rows\n\n")
		fmt.Fprintf(&b, "| model | workload | metric | value | baseline | speedup | status | verify |\n")
		fmt.Fprintf(&b, "|---|---|---|---:|---:|---:|---|---|\n")
		for _, row := range report.Rows {
			fmt.Fprintf(&b, "| %s | %s | %s | %.3f %s | %.3f | %.3fx | %s | %s |\n",
				row.Model, row.Workload, row.Metric, row.Value, row.ValueUnit,
				row.Baseline, row.Speedup, row.Status, row.Verification)
		}
		fmt.Fprintf(&b, "\n")
	}

	if len(report.Issues) > 0 {
		fmt.Fprintf(&b, "## Issues\n\n")
		for _, issue := range report.Issues {
			field := issue.Field
			if field == "" {
				field = "(file)"
			}
			fmt.Fprintf(&b, "- `%s` %s: %s\n", issue.Path, field, issue.Message)
		}
		fmt.Fprintf(&b, "\n")
	}
	return b.String()
}

func rowKey(row Row) string {
	return strings.Join([]string{row.Path, row.Model, row.Workload, row.Metric}, "\x00")
}

func issueKey(issue Issue) string {
	return strings.Join([]string{issue.Path, issue.Field, issue.Message}, "\x00")
}
