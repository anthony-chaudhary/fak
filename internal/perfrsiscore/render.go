package perfrsiscore

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func summarizeHealth(dimensions []Result) (HealthSummary, DebtSummary) {
	debt := DebtSummary{
		DimensionsTotal: len(dimensions),
		Evidence:        make([]DebtEvidence, 0),
	}
	var healthRatio float64
	for _, d := range dimensions {
		if d.NormalizedRatio != nil {
			debt.DimensionsMeasured++
			// Cap credit at the target. Over-performing one dimension must not
			// erase debt in another, while UNKNOWN contributes an honest zero.
			healthRatio += math.Min(*d.NormalizedRatio, 1)
		}
		switch d.Status {
		case "BEHIND":
			debt.Behind++
		case "UNKNOWN":
			debt.Unknown++
		default:
			continue
		}
		debt.Evidence = append(debt.Evidence, DebtEvidence{
			Dimension:       d.ID,
			Status:          d.Status,
			NormalizedRatio: d.NormalizedRatio,
			Source:          d.Source,
			EvidenceKind:    d.EvidenceKind,
			Engine:          d.Engine,
			NextAction:      d.NextAction,
		})
	}
	debt.PerformanceRSIDebt = debt.Behind + debt.Unknown
	debt.Total = debt.PerformanceRSIDebt

	score := 0.0
	if debt.DimensionsTotal > 0 {
		score = scorecard.Round1(100 * healthRatio / float64(debt.DimensionsTotal))
	}
	return HealthSummary{
		Score:          score,
		Grade:          scorecard.GradeStd(score),
		Clean:          debt.PerformanceRSIDebt == 0,
		Interpretation: "grade describes performance-RSI loop health; it does not prove the explicit target multiplier was achieved",
	}, debt
}

func reportHealth(r Report) (HealthSummary, DebtSummary) {
	if r.LoopHealth != nil && r.DebtSummary != nil {
		return *r.LoopHealth, *r.DebtSummary
	}
	return summarizeHealth(r.Dimensions)
}

func Compare(current *Report, prior Report) error {
	if prior.Schema != ReportSchema {
		return fmt.Errorf("prior schema %q, want %q; fix: supply a prior scorecard report conforming to %s", prior.Schema, ReportSchema, ReportSchema)
	}
	pm := map[string]Result{}
	for _, d := range prior.Dimensions {
		pm[d.ID] = d
	}
	c := &Comparison{PriorSnapshot: prior.Snapshot}
	for _, d := range current.Dimensions {
		p, ok := pm[d.ID]
		if !ok {
			return fmt.Errorf("prior snapshot missing dimension %q; fix: ensure the prior scorecard contains all canonical dimensions", d.ID)
		}
		c.Deltas = append(c.Deltas, Delta{ID: d.ID, PriorStatus: p.Status, CurrentStatus: d.Status, PriorRatio: p.NormalizedRatio, CurrentRatio: d.NormalizedRatio})
	}
	current.Comparison = c
	return nil
}

func DecodeReport(r io.Reader) (Report, error) {
	var p Report
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return p, fmt.Errorf("decode report: %w; fix: provide valid JSON conforming to %s", err, ReportSchema)
	}
	return p, nil
}

func RenderHuman(r Report) string {
	var b strings.Builder
	health, debt := reportHealth(r)
	fmt.Fprintf(&b, "performance RSI: %s | target %.0fx | health %s %.1f/100 | performance RSI debt %d\n", r.Snapshot, r.TargetMultiplier, health.Grade, health.Score, debt.PerformanceRSIDebt)
	fmt.Fprintf(&b, "loop health: clean=%t | measured %d/%d | BEHIND %d | UNKNOWN %d\n", health.Clean, debt.DimensionsMeasured, debt.DimensionsTotal, debt.Behind, debt.Unknown)
	fmt.Fprintf(&b, "invocation outcomes: success=%d refusal=%d error=%d\n", r.InvocationOutcomes.Success, r.InvocationOutcomes.Refusal, r.InvocationOutcomes.Error)
	fmt.Fprintf(&b, "grade scope: %s\n", health.Interpretation)
	fmt.Fprintf(&b, "dominant bottleneck: %s\n", r.DominantBottleneck)
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "%-25s %-7s current=%s target=%s ratio=%s source=%s next=%s\n", d.ID, d.Status, number(d.Current), number(d.Target), number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "compared with: %s\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func RenderMarkdown(r Report) string {
	var b strings.Builder
	health, debt := reportHealth(r)
	fmt.Fprintf(&b, "# Performance RSI — %s\n\n- Explicit target: **%.0fx** (unsaturated)\n- Loop-health grade: **%s** (%.1f/100; clean: **%t**)\n- Performance RSI debt: **%d** (%d BEHIND, %d UNKNOWN; %d/%d measured)\n- invocation outcomes: success=%d refusal=%d error=%d\n- Grade scope: %s\n- Dominant bottleneck: `%s`\n\n", r.Snapshot, r.TargetMultiplier, health.Grade, health.Score, health.Clean, debt.PerformanceRSIDebt, debt.Behind, debt.Unknown, debt.DimensionsMeasured, debt.DimensionsTotal, r.InvocationOutcomes.Success, r.InvocationOutcomes.Refusal, r.InvocationOutcomes.Error, health.Interpretation, r.DominantBottleneck)
	b.WriteString("| Dimension | Status | Current | Target | Normalized ratio | Source | Next action |\n|---|---:|---:|---:|---:|---|---|\n")
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "| %s | %s | %s %s | %s %s | %s | %s | %s |\n", d.ID, d.Status, number(d.Current), d.Unit, number(d.Target), d.Unit, number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "\nCompared with `%s`.\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func number(v *float64) string {
	if v == nil {
		return "UNKNOWN"
	}
	if math.IsInf(*v, 1) {
		return "+Inf"
	}
	return fmt.Sprintf("%.6g", *v)
}

func MarshalJSON(r Report) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

func SortResultsForTest(rs []Result) {
	sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
}
