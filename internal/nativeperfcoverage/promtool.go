package nativeperfcoverage

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// PromtoolChecker parses PromQL by placing it in a temporary recording rule.
// Path defaults to promtool from PATH. The temporary directory is always
// outside the repository and is removed before Check returns.
type PromtoolChecker struct {
	Path string
}

// Check validates the supplied PromQL expression using promtool rules checking in an isolated directory.
func (p PromtoolChecker) Check(ctx context.Context, expr string) error {
	path := p.Path
	if path == "" {
		var err error
		path, err = exec.LookPath("promtool")
		if err != nil {
			return fmt.Errorf("promtool is required: %w", err)
		}
	}
	dir, err := os.MkdirTemp("", "fak-nativeperf-promql-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	rule := "groups:\n- name: nativeperfcoverage\n  rules:\n  - record: fak_nativeperfcoverage_query\n    expr: |-\n      " + strings.ReplaceAll(expr, "\n", "\n      ") + "\n"
	rulePath := filepath.Join(dir, "query.yml")
	if err := os.WriteFile(rulePath, []byte(rule), 0o600); err != nil {
		return err
	}
	output, err := exec.CommandContext(ctx, path, "check", "rules", rulePath).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

// EvaluateControlledPromQL executes representative rate and histogram queries
// from the kernel-performance dashboard over ten minutes of multi-point native
// counter data. This catches evaluator/type/range behavior that syntax parsing
// alone cannot prove.
func EvaluateControlledPromQL(ctx context.Context, promtoolPath string) error {
	path := promtoolPath
	if path == "" {
		var err error
		path, err = exec.LookPath("promtool")
		if err != nil {
			return fmt.Errorf("promtool is required: %w", err)
		}
	}
	dir, err := os.MkdirTemp("", "fak-nativeperf-eval-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	rules := `groups:
- name: nativeperfcoverage
  rules:
  - record: nativeperfcoverage:req_rate
    expr: sum(rate(fak_gateway_inference_requests_total{engine=~"fak-native",model=~"Qwen3.8-4B"}[5m])) or vector(-1)
  - record: nativeperfcoverage:ttft_p95
    expr: histogram_quantile(0.95, sum by (le) (rate(fak_native_ttft_seconds_bucket{engine=~"fak-native",model=~"Qwen3.8-4B"}[5m]))) or vector(-1)
`
	tests := `rule_files:
- rules.yml
evaluation_interval: 1m
tests:
- interval: 1m
  input_series:
  - series: 'fak_gateway_inference_requests_total{engine="fak-native",model="Qwen3.8-4B"}'
    values: '0+60x10'
  - series: 'fak_native_ttft_seconds_bucket{engine="fak-native",model="Qwen3.8-4B",le="0.25"}'
    values: '0+70x10'
  - series: 'fak_native_ttft_seconds_bucket{engine="fak-native",model="Qwen3.8-4B",le="0.5"}'
    values: '0+170x10'
  - series: 'fak_native_ttft_seconds_bucket{engine="fak-native",model="Qwen3.8-4B",le="1"}'
    values: '0+230x10'
  - series: 'fak_native_ttft_seconds_bucket{engine="fak-native",model="Qwen3.8-4B",le="+Inf"}'
    values: '0+240x10'
  promql_expr_test:
  - expr: nativeperfcoverage:req_rate
    eval_time: 10m
    exp_samples:
    - labels: '{__name__="nativeperfcoverage:req_rate"}'
      value: 1
  - expr: nativeperfcoverage:ttft_p95
    eval_time: 10m
    exp_samples:
    - labels: '{__name__="nativeperfcoverage:ttft_p95"}'
      value: 0.9833333333333332
`
	if err := os.WriteFile(filepath.Join(dir, "rules.yml"), []byte(rules), 0o600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "test.yml"), []byte(tests), 0o600); err != nil {
		return err
	}
	command := exec.CommandContext(ctx, path, "test", "rules", "test.yml")
	command.Dir = dir
	output, err := command.CombinedOutput()
	if err != nil {
		return fmt.Errorf("controlled PromQL evaluation: %v: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

// EvaluateMatrixControlledPromQL evaluates every extracted panel, annotation,
// and variable query over multi-point controlled fixture data. Each query is
// recorded and then asserted non-empty at the evaluation horizon. Queries whose
// honest populated state is conditional (backend unavailable reason, active SLO
// violation, and annotations) receive a bounded scenario override instead of
// being excused from evaluation.
func EvaluateMatrixControlledPromQL(ctx context.Context, root, promtoolPath string, matrix Matrix) error {
	path := promtoolPath
	if path == "" {
		var err error
		path, err = exec.LookPath("promtool")
		if err != nil {
			return fmt.Errorf("promtool is required: %w", err)
		}
	}
	dir, err := os.MkdirTemp("", "fak-nativeperf-matrix-eval-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	var rules strings.Builder
	rules.WriteString("groups:\n- name: nativeperfcoverage-matrix\n  interval: 1m\n  rules:\n")
	type evaluatedQuery struct {
		record string
		expr   string
	}
	var queries []evaluatedQuery
	queryIndex := 0
	add := func(uid string, panelID int, kind QueryKind, query QueryCoverage) error {
		expr := query.Expr
		switch {
		case uid == "fak-native-backends" && panelID == 3:
			expr = strings.ReplaceAll(expr, "$backend", "cuda")
		case uid == "fak-native-slo" && panelID == 14:
			expr = strings.ReplaceAll(expr, "$scenario", "regression")
		case uid == "fak-native-artifacts" && kind == Annotation:
			expr = strings.ReplaceAll(expr, "$correlation_key", "npc1_11111111111111111111111111111111")
		}
		normalized, err := NormalizePromQL(expr)
		if err != nil {
			return err
		}
		// Populated evaluation must come from controlled fixture data. The
		// dashboard's honest -1 sentinel is validated separately, but it cannot
		// satisfy the populated assertion by itself.
		normalized = strings.TrimSpace(strings.TrimSuffix(normalized, " or vector(-1)"))
		queryIndex++
		record := fmt.Sprintf("nativeperfcoverage:q%03d", queryIndex)
		fmt.Fprintf(&rules, "  - record: %s\n    expr: |-\n      %s\n", record, strings.ReplaceAll(normalized, "\n", "\n      "))
		queries = append(queries, evaluatedQuery{record: record, expr: normalized})
		return nil
	}
	for _, dashboard := range matrix.Dashboards {
		for _, panel := range dashboard.Panels {
			for _, query := range panel.Queries {
				if err := add(dashboard.UID, panel.ID, query.Kind, query); err != nil {
					return fmt.Errorf("%s panel %d: %w", dashboard.UID, panel.ID, err)
				}
			}
		}
		for _, query := range dashboard.Queries {
			if err := add(dashboard.UID, 0, query.Kind, query); err != nil {
				return fmt.Errorf("%s %s %q: %w", dashboard.UID, query.Kind, query.Name, err)
			}
		}
	}

	var tests strings.Builder
	tests.WriteString("rule_files:\n- rules.yml\nevaluation_interval: 1m\ntests:\n- interval: 30s\n  input_series:\n")
	for _, spec := range Specs() {
		raw, err := os.ReadFile(filepath.Join(root, spec.Fixture))
		if err != nil {
			return err
		}
		series, err := ParseFixture(raw)
		if err != nil {
			return err
		}
		for _, sample := range series {
			fmt.Fprintf(&tests, "  - series: %s\n    values: %q\n", promtoolSeries(sample), promtoolValues(sample))
		}
	}
	// The populated backend fixture intentionally has reason=none. This one
	// bounded series proves the explicit-unavailable-reason panel separately
	// while measurement panels continue to gate on reason=none.
	tests.WriteString("  - series: 'fak_native_backend_available{backend=\"cuda\",reason=\"driver_unavailable\"}'\n")
	tests.WriteString("    values: '0x19 1'\n")
	for _, sample := range currentProducerSamples() {
		fmt.Fprintf(&tests, "  - series: %s\n    values: %q\n", promtoolSeries(sample), promtoolValues(sample))
	}
	tests.WriteString("  promql_expr_test:\n")
	for _, query := range queries {
		fmt.Fprintf(&tests, "  - expr: count(%s) > bool 0\n    eval_time: 10m\n    exp_samples:\n    - labels: '{}'\n      value: 1\n", query.record)
	}

	if err := os.WriteFile(filepath.Join(dir, "rules.yml"), []byte(rules.String()), 0o600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "test.yml"), []byte(tests.String()), 0o600); err != nil {
		return err
	}
	command := exec.CommandContext(ctx, path, "test", "rules", "test.yml")
	command.Dir = dir
	output, err := command.CombinedOutput()
	if err != nil {
		return fmt.Errorf("full controlled PromQL matrix evaluation: %v: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func currentProducerSamples() []Series {
	labels := map[string]string{"engine": "inkernel", "backend": "cuda", "forward_path": "qwen_cuda"}
	samples := []Series{
		{Metric: "fak_native_receipt_requests_total", Labels: labels, Value: 400},
		{Metric: "fak_native_receipt_unsupported_total", Labels: map[string]string{"reason": "none"}, Value: 20},
		{Metric: "fak_native_receipt_latest_age_seconds", Value: 10},
		{Metric: "fak_native_receipt_latest_stale", Value: 0},

		{Metric: "fak_gateway_inference_output_tokens_per_second", Labels: map[string]string{"engine": "fak-native", "backend": "cuda"}, Value: 48},
		{Metric: "fak_gateway_inference_cached_prompt_hit_ratio", Labels: map[string]string{"engine": "fak-native", "backend": "cuda"}, Value: 0.75},
		{Metric: "fak_native_slo_state", Labels: map[string]string{"engine": "fak-native", "backend": "cuda", "state": "regression"}, Value: 1},
	}
	for _, signal := range []string{"ttft", "tpot", "queue", "prefill", "decode", "kernel", "kv_reuse", "transfer"} {
		samples = append(samples, Series{Metric: "fak_native_receipt_signal_supported", Labels: map[string]string{"signal": signal}, Value: 1})
	}
	for _, phase := range []string{"queue", "prefill", "decode", "kernel"} {
		samples = append(samples, Series{Metric: "fak_native_receipt_phase_seconds_total", Labels: map[string]string{"engine": "inkernel", "backend": "cuda", "forward_path": "qwen_cuda", "phase": phase}, Value: 40})
	}
	for _, kind := range []string{"kv", "transfer"} {
		samples = append(samples, Series{Metric: "fak_native_receipt_bytes_total", Labels: map[string]string{"engine": "inkernel", "backend": "cuda", "forward_path": "qwen_cuda", "kind": kind}, Value: 4096})
	}
	for _, metric := range []string{"fak_gateway_inference_ttft_seconds_bucket", "fak_gateway_inference_tpot_seconds_bucket"} {
		for i, le := range []string{"0.1", "0.5", "1", "+Inf"} {
			samples = append(samples, Series{Metric: metric, Labels: map[string]string{"engine": "fak-native", "backend": "cuda", "le": le}, Value: float64((i + 1) * 100)})
		}
	}
	return samples
}

func promtoolSeries(sample Series) string {
	var labels []string
	for name, value := range sample.Labels {
		labels = append(labels, name+"="+strconv.Quote(value))
	}
	sort.Strings(labels)
	if len(labels) == 0 {
		return strconv.Quote(sample.Metric)
	}
	return strconv.Quote(sample.Metric + "{" + strings.Join(labels, ",") + "}")
}

func promtoolValues(sample Series) string {
	// changes() must observe a transition inside its final one-minute range.
	if sample.Metric == "fak_native_artifact_info" || sample.Metric == "fak_native_lifecycle_event_info" || (sample.Metric == "fak_native_slo_state" && sample.Labels["state"] == "regression") {
		return "0x19 1"
	}
	if strings.HasSuffix(sample.Metric, "_total") || strings.HasSuffix(sample.Metric, "_bucket") {
		delta := math.Abs(sample.Value) / 20
		if delta < 1 {
			delta = 1
		}
		return "0+" + strconv.FormatFloat(delta, 'g', -1, 64) + "x20"
	}
	value := strconv.FormatFloat(sample.Value, 'g', -1, 64)
	return value + "x20"
}
