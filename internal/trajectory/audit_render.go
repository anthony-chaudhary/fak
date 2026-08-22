package trajectory

import (
	"fmt"
	"io"
	"strings"
)

// WriteAuditMarkdown renders the same rows as a compact operator report.
func WriteAuditMarkdown(w io.Writer, result AuditResult) error {
	var out strings.Builder
	summary := result.Summary
	out.WriteString("# Cross-harness trajectory audit\n\n")
	fmt.Fprintf(&out, "Schema: `%s`\n\n", AuditSchema)
	out.WriteString("## Exact totals\n\n")
	fmt.Fprintf(&out, "- Sources: %d; sessions: %d; files scanned: %d/%d; records: %d.\n", summary.Sources, summary.Transcripts, summary.FilesScanned, summary.FilesDiscovered, summary.Records)
	fmt.Fprintf(&out, "- Input: %d; output: %d; cache create: %d; cache read: %d exact tokens.\n", summary.Tokens.InputTokens, summary.Tokens.OutputTokens, summary.Tokens.CacheCreateTokens, summary.Tokens.CacheReadTokens)
	fmt.Fprintf(&out, "- Input:output ratio: %s; cache-create burden: %s.\n", auditFloat(summary.InputOutputRatio, 3), auditPercent(summary.PromptWriteFraction))
	fmt.Fprintf(&out, "- Repeated failures: %d; mutation churn: %d; hook p95: %s ms.\n", summary.RepeatedFailures, summary.MutationChurn, auditInt(summary.HookP95MS))

	out.WriteString("\n## Source denominator\n\n")
	out.WriteString("| Source | Root | Present | Files scanned/discovered | Records | Exact usage | Applied usage | Duplicates | Refused |\n")
	out.WriteString("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, denominator := range result.Denominators {
		fmt.Fprintf(&out, "| %s | `%s` | %t | %d/%d | %d | %d/%d | %d | %d | %d |\n",
			denominator.Source, denominator.Root, denominator.RootPresent, denominator.FilesScanned, denominator.FilesDiscovered,
			denominator.Records, denominator.UsageRecordsExact, denominator.UsageRecordsSeen, denominator.UsageRecordsApplied,
			denominator.DuplicateUsageRecords, denominator.RefusedRecords)
		fmt.Fprintf(&out, "\n%s token semantics: %s.\n", denominator.Source, denominator.TokenSemantics)
	}

	out.WriteString("\n## Token-weighted bottlenecks\n\n")
	if len(result.Bottlenecks) == 0 {
		out.WriteString("No sessions were in the selected window.\n")
	} else {
		out.WriteString("| Rank | Source/session | Accounted tokens | Dominant bucket |\n")
		out.WriteString("|---:|---|---:|---|\n")
		for _, row := range result.Bottlenecks {
			fmt.Fprintf(&out, "| %d | %s/`%s` | %d | %s (%d) |\n", row.Rank, row.Source, row.TranscriptID, row.AccountedTokens, row.DominantBucket, row.DominantTokens)
		}
		first := result.Bottlenecks[0]
		fmt.Fprintf(&out, "\nHighest-cost bottleneck: %s/`%s` with %d accounted tokens; deterministic ties sort by source, transcript, then relative path.\n", first.Source, first.TranscriptID, first.AccountedTokens)
	}

	if len(result.Baseline) > 0 {
		out.WriteString("\n## Baseline deltas\n\n")
		out.WriteString("| Metric | Current | Baseline | Delta | Comparable | Regression |\n")
		out.WriteString("|---|---:|---:|---:|---:|---:|\n")
		for _, row := range result.Baseline {
			fmt.Fprintf(&out, "| %s | %s | %s | %s | %t | %t |\n", row.Metric, auditFloat(row.Current, 4), auditFloat(row.Baseline, 4), auditFloat(row.Delta, 4), row.Comparable, row.Regression)
		}
	}

	out.WriteString("\n## Refused transcript shapes\n\n")
	if len(result.Refusals) == 0 {
		out.WriteString("None. Every usage candidate in the selected window matched a versioned exact shape.\n")
	} else {
		out.WriteString("The audit refused to estimate these records and must exit non-zero.\n\n")
		out.WriteString("| Source | Relative path | Line | Code | Detail |\n")
		out.WriteString("|---|---|---:|---|---|\n")
		for _, row := range result.Refusals {
			fmt.Fprintf(&out, "| %s | `%s` | %d | `%s` | %s |\n", row.Source, row.SourcePath, row.Line, row.Code, escapeAuditMarkdown(row.Detail))
		}
	}

	if _, err := io.WriteString(w, out.String()); err != nil {
		return fmt.Errorf("trajectory audit: write markdown: %w", err)
	}
	return nil
}

func auditFloat(value *float64, digits int) string {
	if value == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.*f", digits, *value)
}

func auditPercent(value *float64) string {
	if value == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.2f%%", *value*100)
}

func auditInt(value *int64) string {
	if value == nil {
		return "n/a"
	}
	return fmt.Sprintf("%d", *value)
}

func escapeAuditMarkdown(value string) string {
	return strings.ReplaceAll(value, "|", "\\|")
}
