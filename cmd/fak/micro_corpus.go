package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

type microCorpusTask struct {
	ID         string `json:"id"`
	Complexity string `json:"complexity"`
	Task       string `json:"task"`
	Expected   string `json:"expected"`
}

type microCorpusCase struct {
	Task             microCorpusTask `json:"task"`
	ExecutionVerdict string          `json:"execution_verdict"`
	Micro            pairedArm       `json:"microagent"`
	Baseline         pairedArm       `json:"managed_baseline"`
}

type microCorpusTotals struct {
	Tasks              int      `json:"tasks"`
	MicroCorrect       int      `json:"micro_correct"`
	BaselineCorrect    int      `json:"baseline_correct"`
	MicroInputTokens   int      `json:"micro_input_tokens"`
	MicroOutputTokens  int      `json:"micro_output_tokens"`
	BaselineInput      int      `json:"baseline_input_tokens"`
	BaselineOutput     int      `json:"baseline_output_tokens"`
	MicroWallMS        int64    `json:"micro_wall_ms"`
	BaselineWallMS     int64    `json:"baseline_wall_ms"`
	MicroCostUSD       *float64 `json:"micro_cost_usd"`
	BaselineCostUSD    *float64 `json:"baseline_cost_usd"`
	MicroCostStatus    string   `json:"micro_cost_status"`
	BaselineCostStatus string   `json:"baseline_cost_status"`
}

type microCorpusAblation struct {
	Layer  string `json:"layer"`
	Status string `json:"status"`
	Reason string `json:"reason"`
}

type microCorpusReport struct {
	Schema           string                `json:"schema"`
	Corpus           string                `json:"corpus"`
	ExecutionVerdict string                `json:"execution_verdict"`
	ValueVerdict     string                `json:"value_verdict"`
	Reason           string                `json:"reason"`
	Cases            []microCorpusCase     `json:"cases"`
	Totals           microCorpusTotals     `json:"totals"`
	Ablations        []microCorpusAblation `json:"ablations"`
	RetryAblation    microRetryAblation    `json:"retry_ablation"`
}

// pinnedMicroCorpus is deliberately small: one exact instruction, one structured
// extraction, and one policy classification. It proves the paired measurement
// path over common agent subtasks without pretending to be SWE-bench.
var pinnedMicroCorpus = []microCorpusTask{
	{ID: "instruction", Complexity: "one-step", Task: "Reply with exactly READY", Expected: "READY"},
	{ID: "extract", Complexity: "structured", Task: "From this Go declaration, reply with only the function name: func AdmitRequest(ctx context.Context) error", Expected: "AdmitRequest"},
	{ID: "policy", Complexity: "reasoning", Task: `A policy has allowed_tools ["search_kb"] and denied_tools ["refund_payment"]. The requested tool is refund_payment. Reply with exactly DENY.`, Expected: "DENY"},
}

func cmdMicroCorpus(args []string) {
	fs := flag.NewFlagSet("micro corpus", flag.ExitOnError)
	gateway := fs.String("gateway", "", "running fak serve address for the microagent arm")
	model := fs.String("model", "", "model requested through the fak gateway")
	cliModel := fs.String("cli-model", "sonnet", "Claude model used by the managed baseline")
	markdown := fs.String("markdown", "", "optional Markdown report output path")
	jsonOut := fs.Bool("json", false, "emit the corpus receipt as JSON")
	fs.Parse(args)
	if strings.TrimSpace(*gateway) == "" || strings.TrimSpace(*model) == "" {
		fmt.Fprintln(os.Stderr, "fak micro corpus: --gateway and --model are required")
		os.Exit(2)
	}
	report := runMicroCorpus(context.Background(), *gateway, *model, *cliModel)
	retry, err := runMicroRetryAblation(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak micro corpus: retry ablation: %v\n", err)
		os.Exit(1)
	}
	report.RetryAblation = retry
	applyRetryAblation(&report)
	if *markdown != "" {
		if err := os.WriteFile(*markdown, []byte(formatMicroCorpusReport(report)), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro corpus: write Markdown: %v\n", err)
			os.Exit(2)
		}
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		fmt.Printf("%s execution=%s value=%s tasks=%d micro=%d baseline=%d — %s\n", report.Schema, report.ExecutionVerdict, report.ValueVerdict, report.Totals.Tasks, report.Totals.MicroCorrect, report.Totals.BaselineCorrect, report.Reason)
	}
	if report.ExecutionVerdict != "PASS" {
		os.Exit(1)
	}
}

func runMicroCorpus(ctx context.Context, gateway, model, cliModel string) microCorpusReport {
	cases := make([]microCorpusCase, 0, len(pinnedMicroCorpus))
	for _, task := range pinnedMicroCorpus {
		paired := runMicroPaired(ctx, gateway, model, task.Task, task.Expected, cliModel, task.Complexity)
		cases = append(cases, microCorpusCase{Task: task, ExecutionVerdict: paired.ExecutionVerdict, Micro: paired.Micro, Baseline: paired.CLI})
	}
	return foldMicroCorpus(cases)
}

func foldMicroCorpus(cases []microCorpusCase) microCorpusReport {
	r := microCorpusReport{
		Schema: "fak-micro-corpus/1", Corpus: "popular-agent-subtasks-v1", ExecutionVerdict: "PASS", ValueVerdict: "NOT_YET", Cases: cases,
		Reason: "paired corpus execution is measured, but gateway dollars and retry/context/verify/mode ablations are not yet available; no quality/$ winner is claimed",
		Ablations: []microCorpusAblation{
			{Layer: "retry", Status: "NOT_YET", Reason: "pinned tasks do not inject a witnessed transient failure"},
			{Layer: "context", Status: "NOT_YET", Reason: "pinned tasks do not cross the context compaction threshold"},
			{Layer: "verify", Status: "NOT_YET", Reason: "exact-answer scoring is external and does not exercise the microagent Verifier hook"},
			{Layer: "mode", Status: "NOT_YET", Reason: "the real gateway microagent currently exposes completion mode only; #2026 owns bash/tool mode parity"},
		},
	}
	baselineCost := 0.0
	baselineCostComplete := len(cases) > 0
	for _, row := range cases {
		r.Totals.Tasks++
		if row.Micro.Correct {
			r.Totals.MicroCorrect++
		}
		if row.Baseline.Correct {
			r.Totals.BaselineCorrect++
		}
		r.Totals.MicroInputTokens += row.Micro.InputTokens
		r.Totals.MicroOutputTokens += row.Micro.OutputTokens
		r.Totals.BaselineInput += row.Baseline.InputTokens
		r.Totals.BaselineOutput += row.Baseline.OutputTokens
		r.Totals.MicroWallMS += row.Micro.WallMS
		r.Totals.BaselineWallMS += row.Baseline.WallMS
		if row.Baseline.CostUSD == nil {
			baselineCostComplete = false
		} else {
			baselineCost += *row.Baseline.CostUSD
		}
		if row.ExecutionVerdict != "PASS" {
			r.ExecutionVerdict = "FAIL"
		}
	}
	if len(cases) == 0 {
		r.ExecutionVerdict = "FAIL"
	}
	r.Totals.MicroCostStatus = "provider-unsupported"
	r.Totals.BaselineCostStatus = "provider-unreported"
	if baselineCostComplete {
		r.Totals.BaselineCostUSD = &baselineCost
		r.Totals.BaselineCostStatus = "provider-reported"
	}
	return r
}

func applyRetryAblation(report *microCorpusReport) {
	if report == nil {
		return
	}
	for i := range report.Ablations {
		if report.Ablations[i].Layer == "retry" && retryAblationPassed(report.RetryAblation) {
			report.Ablations[i] = microCorpusAblation{
				Layer:  "retry",
				Status: "PASS",
				Reason: "retry-off failed after one attempt; retry-on completed after exact transient evidence was fed back, bounded at two attempts",
			}
		}
	}
	if retryAblationPassed(report.RetryAblation) {
		report.Reason = "paired corpus execution and grounded retry contribution are measured, but gateway dollars and context/verify/mode ablations are not yet available; no quality/$ winner is claimed"
	}
}

func formatMicroCorpusReport(r microCorpusReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# fak microagent paired corpus\n\n- Schema: `%s`\n- Corpus: `%s`\n- Execution: **%s**\n- Value: **%s**\n- Reason: %s\n\n", r.Schema, r.Corpus, r.ExecutionVerdict, r.ValueVerdict, r.Reason)
	b.WriteString("| Task | Complexity | Micro | Managed baseline | Micro tokens | Baseline tokens | Micro ms | Baseline ms |\n|---|---|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range r.Cases {
		fmt.Fprintf(&b, "| %s | %s | %t | %t | %d | %d | %d | %d |\n", c.Task.ID, c.Task.Complexity, c.Micro.Correct, c.Baseline.Correct, c.Micro.InputTokens+c.Micro.OutputTokens, c.Baseline.InputTokens+c.Baseline.OutputTokens, c.Micro.WallMS, c.Baseline.WallMS)
	}
	b.WriteString("\n## Ablation readiness\n\n| Layer | Status | Reason |\n|---|---|---|\n")
	for _, a := range r.Ablations {
		fmt.Fprintf(&b, "| %s | %s | %s |\n", a.Layer, a.Status, a.Reason)
	}
	if retryAblationPassed(r.RetryAblation) {
		b.WriteString("\n### Retry witness\n\n")
		fmt.Fprintf(&b, "- retry off: completed=%t, attempts=%d\n", r.RetryAblation.WithoutRetryCompleted, r.RetryAblation.WithoutRetryAttempts)
		fmt.Fprintf(&b, "- retry on: completed=%t, attempts=%d\n", r.RetryAblation.WithRetryCompleted, r.RetryAblation.WithRetryAttempts)
		fmt.Fprintf(&b, "- evidence re-fed verbatim: `%s`\n", r.RetryAblation.Evidence[0])
	}
	return b.String()
}
