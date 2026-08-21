package agenticbench

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

const AccountingReportSchema = "fak.agentic-benchmark-accounting-report.v1"

type AccountingClaimState string

const (
	AccountingClaimAllowed AccountingClaimState = "ALLOWED"
	AccountingClaimRefused AccountingClaimState = "REFUSED"
)

// AccountingArm is the narrow ArmBench-to-AgenticBench adapter. It carries the
// shared receipt unchanged so the benchmark layer cannot reinterpret missing
// usage or silently choose a different authority.
type AccountingArm struct {
	Arm     string                     `json:"arm"`
	Receipt armbench.AccountingReceipt `json:"receipt"`
}

func AccountingArmFromSummary(summary armbench.ArmSummary) AccountingArm {
	return AccountingArm{Arm: summary.ArmID, Receipt: summary.Accounting}
}

type AccountingComparisonRequest struct {
	LeftArm  string                    `json:"left_arm"`
	RightArm string                    `json:"right_arm"`
	Metric   armbench.AccountingMetric `json:"metric"`
}

type AccountingComparison struct {
	LeftArm  string                    `json:"left_arm"`
	RightArm string                    `json:"right_arm"`
	Metric   armbench.AccountingMetric `json:"metric"`
	State    AccountingClaimState      `json:"state"`
	Left     armbench.AccountingField  `json:"left"`
	Right    armbench.AccountingField  `json:"right"`
	Delta    *float64                  `json:"delta"`
	Detail   string                    `json:"detail"`
}

type AccountingReport struct {
	Schema      string                 `json:"schema"`
	Arms        []AccountingArm        `json:"arms"`
	Comparisons []AccountingComparison `json:"comparisons"`
}

// BuildAccountingReport applies the shared comparison gate to every requested
// token, cache, or cost claim. A refusal remains a report row rather than an
// error because it is evidence about the benchmark, not a malformed request.
func BuildAccountingReport(arms []AccountingArm, requests []AccountingComparisonRequest) (AccountingReport, error) {
	if len(arms) == 0 {
		return AccountingReport{}, fmt.Errorf("accounting report requires at least one arm")
	}
	report := AccountingReport{Schema: AccountingReportSchema}
	byArm := make(map[string]armbench.AccountingReceipt, len(arms))
	for _, arm := range arms {
		if strings.TrimSpace(arm.Arm) == "" || strings.TrimSpace(arm.Arm) != arm.Arm {
			return AccountingReport{}, fmt.Errorf("accounting arm %q must be non-empty with no surrounding whitespace", arm.Arm)
		}
		if _, exists := byArm[arm.Arm]; exists {
			return AccountingReport{}, fmt.Errorf("duplicate accounting arm %q", arm.Arm)
		}
		if arm.Receipt.Schema != armbench.AccountingReceiptSchema {
			return AccountingReport{}, fmt.Errorf("arm %q accounting schema %q, want %q", arm.Arm, arm.Receipt.Schema, armbench.AccountingReceiptSchema)
		}
		byArm[arm.Arm] = arm.Receipt
		report.Arms = append(report.Arms, arm)
	}
	sort.Slice(report.Arms, func(i, j int) bool { return report.Arms[i].Arm < report.Arms[j].Arm })
	for _, request := range requests {
		leftReceipt, leftOK := byArm[request.LeftArm]
		rightReceipt, rightOK := byArm[request.RightArm]
		if !leftOK || !rightOK {
			return AccountingReport{}, fmt.Errorf("comparison arms %q/%q must name accounting arms", request.LeftArm, request.RightArm)
		}
		left, leftMetricOK := leftReceipt.Field(request.Metric)
		right, rightMetricOK := rightReceipt.Field(request.Metric)
		if !leftMetricOK || !rightMetricOK {
			return AccountingReport{}, fmt.Errorf("unknown accounting metric %q", request.Metric)
		}
		gate := armbench.CompareAccountingFields(request.Metric, left, right)
		comparison := AccountingComparison{
			LeftArm: request.LeftArm, RightArm: request.RightArm, Metric: request.Metric,
			State: AccountingClaimRefused, Left: left, Right: right, Detail: gate.Detail,
		}
		if gate.Comparable {
			comparison.State = AccountingClaimAllowed
			comparison.Delta = gate.Delta
		}
		report.Comparisons = append(report.Comparisons, comparison)
	}
	return report, nil
}

func FormatAccountingMarkdown(report AccountingReport) string {
	var b strings.Builder
	fmt.Fprintln(&b, "# Agentic Benchmark Accounting Authority Receipt")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Schema: `%s`\n", report.Schema)
	fmt.Fprintf(&b, "- Arms: `%d`\n\n", len(report.Arms))
	fmt.Fprintln(&b, "| Arms | Metric | State | Left authority / coverage | Right authority / coverage | Delta | Detail |")
	fmt.Fprintln(&b, "|---|---|---|---|---|---:|---|")
	for _, comparison := range report.Comparisons {
		delta := "n/a"
		if comparison.Delta != nil {
			delta = fmt.Sprintf("%g", *comparison.Delta)
		}
		fmt.Fprintf(&b, "| `%s` vs `%s` | `%s` | `%s` | `%s` %d/%d `%s` | `%s` %d/%d | %s | %s |\n",
			comparison.LeftArm, comparison.RightArm, comparison.Metric, comparison.State,
			comparison.Left.Authority, comparison.Left.Coverage.Observed, comparison.Left.Coverage.Expected, comparison.Left.Coverage.Scope,
			comparison.Right.Authority, comparison.Right.Coverage.Observed, comparison.Right.Coverage.Expected,
			delta, mdCell(comparison.Detail))
	}
	return b.String()
}
