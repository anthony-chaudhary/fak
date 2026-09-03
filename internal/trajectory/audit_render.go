package trajectory

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// WriteAuditMarkdown renders the same rows as a compact operator report.
func WriteAuditMarkdown(w io.Writer, result AuditResult) error {
	var out strings.Builder
	summary := result.Summary
	out.WriteString("# Cross-harness trajectory audit\n\n")
	status := "blocked"
	if result.ConclusionStatus.BroadEfficiencySupported {
		status = "supported"
	}
	fmt.Fprintf(&out, "Broad efficiency conclusions: **%s** (refusals: %d).\n\n", status, result.ConclusionStatus.RefusalCount)
	fmt.Fprintf(&out, "Transcript schema drift: %d change(s), %d breaking.\n\n", result.ConclusionStatus.SchemaDriftCount, result.ConclusionStatus.BreakingSchemaDrift)
	fmt.Fprintf(&out, "Schema: `%s`\n\n", AuditSchema)
	out.WriteString("## Exact totals\n\n")
	fmt.Fprintf(&out, "- Sources: %d; sessions: %d; files scanned: %d/%d; records: %d.\n", summary.Sources, summary.Transcripts, summary.FilesScanned, summary.FilesDiscovered, summary.Records)
	if summary.FilesMatched > 0 {
		fmt.Fprintf(&out, "- Topical cohort: %d user-prompt-matched transcript files.\n", summary.FilesMatched)
	}
	fmt.Fprintf(&out, "- Input: %d; output: %d; cache create: %d; cache read: %d exact tokens.\n", summary.Tokens.InputTokens, summary.Tokens.OutputTokens, summary.Tokens.CacheCreateTokens, summary.Tokens.CacheReadTokens)
	fmt.Fprintf(&out, "- Input:output ratio: %s; cache-create burden: %s.\n", auditFloat(summary.InputOutputRatio, 3), auditPercent(summary.PromptWriteFraction))
	fmt.Fprintf(&out, "- Repeated failures: %d/%d sessions (%s per session); expected bounded wait_agent timeouts: %d; mutation churn: %d; hook p95: %s ms.\n",
		summary.RepeatedFailures, summary.Transcripts, auditFloat(summary.RepeatedFailuresPerSession, 4), summary.ExpectedWaitTimeouts, summary.MutationChurn, auditInt(summary.HookP95MS))
	fmt.Fprintf(&out, "- Repeated-failure semantics: `%s`; normalization: `%s`.\n", summary.RepeatedFailureSemantics, summary.RepeatedFailureNormalization)
	fmt.Fprintf(&out, "- Distinct transcripts: %d; duplicate fragments: %d; empty-usage files: %d.\n", summary.DistinctTranscripts, summary.DuplicateFragments, summary.EmptyUsageFiles)
	fmt.Fprintf(&out, "- Tool errors: %d/%d (%s); top-10 token concentration: %s.\n", summary.ToolErrors, summary.ToolCalls, auditPercent(summary.ToolErrorFraction), auditPercent(summary.TopTenTokenFraction))
	fmt.Fprintf(&out, "- Payload distribution unit: `%s` — %s\n", summary.DistributionUnit, summary.DistributionProvenance)
	writeAuditCodexCacheObservations(&out, result.Transcripts)
	out.WriteString("\n## Transcript schema drift\n\n")
	if len(result.SchemaDrift) == 0 {
		out.WriteString("No event-type or field-shape drift from the checked-in baseline.\n")
	} else {
		out.WriteString("| Source/event | Baseline builds | Current builds | Change | Compatibility | Fields | Proposed update |\n")
		out.WriteString("|---|---|---|---|---|---|---|\n")
		for _, row := range result.SchemaDrift {
			fields := append([]string(nil), row.AdditiveFields...)
			fields = append(fields, row.BreakingChanges...)
			fmt.Fprintf(&out, "| %s/`%s` | %s | %s | `%s` | **%s** | %s | %s |\n",
				row.Source, escapeAuditMarkdown(row.EventType), auditBuildsMarkdown(row.BaselineBuilds), auditBuildsMarkdown(row.CurrentBuilds),
				row.Change, row.Compatibility, escapeAuditMarkdown(strings.Join(fields, "; ")), escapeAuditMarkdown(row.ProposedAction))
		}
	}
	if len(summary.StorageDistribution) > 0 {
		out.WriteString("\n## Transcript storage and telemetry overhead\n\n| Source | Subtype | Records | Serialized UTF-8 bytes | Source share | Unknown exemplar IDs |\n|---|---|---:|---:|---:|---|\n")
		for _, r := range summary.StorageDistribution {
			fmt.Fprintf(&out, "| `%s` | `%s` | %d | %d | %.1f%% | %s |\n", r.Source, r.Subtype, r.Records, r.Bytes, r.Share*100, auditExemplarIDLinks(r.ExemplarIDs))
		}
	}
	if len(summary.Distribution) > 0 {
		out.WriteString("\n## Token destination distribution\n\n| Category | UTF-8 bytes | Share | Unknown exemplar IDs |\n|---|---:|---:|---|\n")
		for _, r := range summary.Distribution {
			fmt.Fprintf(&out, "| `%s` | %d | %.1f%% | %s |\n", r.Name, r.Bytes, r.Share*100, auditExemplarIDLinks(r.ExemplarIDs))
		}
	}
	if len(summary.ToolDistribution) > 0 {
		out.WriteString("\n### Per tool\n\n| Tool | Calls | UTF-8 bytes | Share |\n|---|---:|---:|---:|\n")
		for _, r := range summary.ToolDistribution {
			fmt.Fprintf(&out, "| `%s` | %d | %d | %.1f%% |\n", r.Name, r.Calls, r.Bytes, r.Share*100)
		}
		fmt.Fprintf(&out, "\n`%s`\n", CompactAuditDistributionLine(summary.Distribution, summary.ToolDistribution, 100))
	}
	if len(summary.ToolResults) > 0 {
		out.WriteString("\n### Tool result outcomes\n\n| Tool | Subtype | Results | Success | Errors | Timeouts | Truncated | Unknown | Unmatched | Exit 0/nonzero/unknown | Duration known/total ms | Output channel stdout/stderr/combined/unknown | UTF-8 bytes |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
		for _, r := range summary.ToolResults {
			fmt.Fprintf(&out, "| `%s` | `%s` | %d | %d | %d | %d | %d | %d | %d | %d/%d/%d | %d/%d | %d/%d/%d/%d | %d |\n",
				escapeAuditMarkdown(r.Name), escapeAuditMarkdown(r.Subtype), r.Results, r.Success, r.Errors, r.Timeouts, r.Truncated, r.Unknown, r.Unmatched,
				r.ExitZero, r.ExitNonzero, r.Results-r.ExitKnown, r.DurationKnown, r.DurationMS,
				r.Stdout, r.Stderr, r.CombinedOutput, r.ChannelUnknown, r.Bytes)
		}
	}
	if summary.UnknownExemplars.Retained > 0 || summary.UnknownExemplars.DroppedObservations > 0 {
		reservoir := summary.UnknownExemplars
		out.WriteString("\n## Unknown event structural exemplars\n\n")
		fmt.Fprintf(&out, "Retained %d content-free shapes (%d serialized bytes) within fixed limits of %d shapes / %d bytes; dropped observations: %d. Structures contain scrubbed field names and JSON types only; scalar payload values are discarded before hashing and persistence.\n\n",
			reservoir.Retained, reservoir.StoredBytes, reservoir.CardinalityLimit, reservoir.ByteLimit, reservoir.DroppedObservations)
		out.WriteString("| ID | Source/subtype | Visibility → aggregate | Observations | Observed bytes | Shape hash | Structure |\n")
		out.WriteString("|---|---|---|---:|---:|---|---|\n")
		for _, exemplar := range reservoir.Exemplars {
			fmt.Fprintf(&out, "| `%s` | `%s` / `%s` | `%s` → `%s` | %d | %d | `%s` | `%s` |\n",
				exemplar.ID, exemplar.Source, exemplar.Subtype, exemplar.Visibility, exemplar.Aggregate,
				exemplar.Observations, exemplar.ObservedBytes, exemplar.ShapeHash, escapeAuditMarkdown(exemplar.Structure))
		}
	}
	if summary.QwenTopContributorTokenFraction == nil {
		out.WriteString("- Qwen top-contributor token concentration: unknown (no Qwen token usage).\n")
	} else {
		fmt.Fprintf(&out, "- Qwen top contributor: `%s` with %d/%d tokens (%s); concentrated: %t (threshold: %.0f%%).\n", *summary.QwenTopContributor, *summary.QwenTopContributorTokens, *summary.QwenTotalTokens, auditPercent(summary.QwenTopContributorTokenFraction), *summary.QwenTopContributorTokenConcentrated, summary.QwenTokenConcentrationThreshold*100)
	}

	out.WriteString("\n## Source denominator\n\n")
	out.WriteString("| Source | Root | Present | Files scanned/matched/non-session/discovered | Records | Exact usage | Applied usage | Duplicates | Refused |\n")
	out.WriteString("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, denominator := range result.Denominators {
		fmt.Fprintf(&out, "| %s | `%s` | %t | %d/%d/%d/%d | %d | %d/%d | %d | %d | %d |\n",
			denominator.Source, denominator.Root, denominator.RootPresent, denominator.FilesScanned, denominator.FilesMatched, denominator.FixtureFilesExcluded, denominator.FilesDiscovered,
			denominator.Records, denominator.UsageRecordsExact, denominator.UsageRecordsSeen, denominator.UsageRecordsApplied,
			denominator.DuplicateUsageRecords, denominator.RefusedRecords)
		fmt.Fprintf(&out, "\n%s token semantics: %s.\n", denominator.Source, denominator.TokenSemantics)
	}

	out.WriteString("\n## Mutation churn interventions\n\n")
	out.WriteString("| Transcript | Target | Writes | Accounted tokens | Intervention |\n")
	out.WriteString("|---|---|---:|---:|---|\n")
	churnRows := 0
	for _, transcript := range result.Transcripts {
		for _, churn := range transcript.MutationChurnEvents {
			fmt.Fprintf(&out, "| %s | %s | %d | %d | %s |\n", escapeAuditMarkdown(churn.TranscriptID), escapeAuditMarkdown(churn.Target), churn.Count, churn.AccountedTokens, escapeAuditMarkdown(string(churn.Intervention)))
			churnRows++
		}
	}
	if churnRows == 0 {
		out.WriteString("| none | none | 0 | 0 | observe-only |\n")
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

	if len(result.ToolErrorFamilies) > 0 {
		out.WriteString("\n## Tool error families\n\n")
		out.WriteString("| Family | Calls | Accounted tokens | Repeated failures | Mutation churn | First event | Last event |\n")
		out.WriteString("|---|---:|---:|---:|---:|---:|---:|\n")
		for _, family := range result.ToolErrorFamilies {
			fmt.Fprintf(&out, "| %s | %d | %d | %d | %d | %d | %d |\n", family.Family, family.Count, family.Tokens, family.RepeatedFailures, family.MutationChurn, family.FirstIndex, family.LastIndex)
		}
	}

	if len(result.Baseline) > 0 {
		out.WriteString("\n## Baseline deltas\n\n")
		out.WriteString("| Metric | Raw current / baseline | Compared current / baseline | Normalization (current / baseline exposure) | Delta | Comparability | Regression |\n")
		out.WriteString("|---|---:|---:|---|---:|---|---:|\n")
		for _, row := range result.Baseline {
			fmt.Fprintf(&out, "| %s | %s | %s / %s | %s | %s | %s | %t |\n",
				row.Metric, auditRawCounts(row), auditFloat(auditDeltaComparedCurrent(row), 4), auditFloat(auditDeltaComparedBaseline(row), 4),
				auditDeltaNormalization(row), auditFloat(auditDeltaComparedDelta(row), 4), auditDeltaComparability(row), auditDeltaComparedRegression(row))
		}
	}

	writeAuditTerminalFailures(&out, result.Summary)

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

func writeAuditCodexCacheObservations(out *strings.Builder, transcripts []AuditTranscriptRow) {
	rows := make([]AuditTranscriptRow, 0)
	for _, transcript := range transcripts {
		if transcript.CodexCache != nil {
			rows = append(rows, transcript)
		}
	}
	if len(rows) == 0 {
		return
	}
	out.WriteString("\n## Codex per-request cache observations\n\n")
	out.WriteString("| Transcript | Transcript producer | Configured provider path | Samples | Observed min | Observed max |\n")
	out.WriteString("|---|---|---|---:|---:|---:|\n")
	for _, transcript := range rows {
		observation := transcript.CodexCache
		provider := "not recorded"
		if observation.ModelProvider != "" {
			provider = "`" + escapeAuditMarkdown(observation.ModelProvider) + "`"
		}
		fmt.Fprintf(out, "| `%s` | `%s` | %s | %d | %s | %s |\n",
			escapeAuditMarkdown(transcript.TranscriptID), escapeAuditMarkdown(observation.TranscriptProducer), provider,
			observation.LastTokenUsageCachedInputSamples,
			auditInt(observation.LastTokenUsageCachedInputMin), auditInt(observation.LastTokenUsageCachedInputMax))
	}
	out.WriteString("\n`cached_input_tokens` is emitted by the transcript producer for the configured provider path; it does not prove physical provider cache residency or process-local ownership. ")
	out.WriteString("fak-owned caches are not observed by Codex `token_count` rows and require fak telemetry for attribution.\n")
}

func writeAuditTerminalFailures(out *strings.Builder, summary AuditSummaryRow) {
	if summary.TerminalFailures == 0 {
		return
	}
	out.WriteString("\n## Terminal failures and stalls\n\n")
	fmt.Fprintf(out, "- Total terminal failures: %d\n\n", summary.TerminalFailures)
	if len(summary.TerminalFailureClasses) > 0 {
		out.WriteString("| Failure class | Occurrences |\n")
		out.WriteString("|---|---:|\n")
		keys := make([]string, 0, len(summary.TerminalFailureClasses))
		for k := range summary.TerminalFailureClasses {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(out, "| `%s` | %d |\n", escapeAuditMarkdown(k), summary.TerminalFailureClasses[k])
		}
	}
	if len(summary.TerminalStallDurationBuckets) > 0 {
		out.WriteString("\n| Stall duration bucket | Failures |\n")
		out.WriteString("|---|---:|\n")
		keys := make([]string, 0, len(summary.TerminalStallDurationBuckets))
		for k := range summary.TerminalStallDurationBuckets {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(out, "| `%s` | %d |\n", escapeAuditMarkdown(k), summary.TerminalStallDurationBuckets[k])
		}
	}
}

func auditBuildsMarkdown(builds []AuditBuildIdentity) string {
	if len(builds) == 0 {
		return "n/a"
	}
	values := make([]string, 0, len(builds))
	for _, build := range builds {
		harness := build.Harness
		if build.HarnessBuild != "" {
			harness += "@" + build.HarnessBuild
		}
		provider := build.Provider
		if build.ProviderBuild != "" {
			provider += "@" + build.ProviderBuild
		}
		if provider != "" {
			values = append(values, "`"+provider+" / "+harness+"`")
		} else {
			values = append(values, "`"+harness+"`")
		}
	}
	return strings.Join(values, "<br>")
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

func auditRawCounts(row AuditDeltaRow) string {
	if row.RawCurrent == nil || row.RawBaseline == nil {
		return "n/a"
	}
	return fmt.Sprintf("%d / %d", *row.RawCurrent, *row.RawBaseline)
}

func auditDeltaNormalization(row AuditDeltaRow) string {
	if row.Normalization == "" {
		return "as_reported"
	}
	if row.CurrentExposure == nil || row.BaselineExposure == nil {
		return row.Normalization
	}
	return fmt.Sprintf("%s (%d / %d)", row.Normalization, *row.CurrentExposure, *row.BaselineExposure)
}

func auditDeltaComparability(row AuditDeltaRow) string {
	status := row.ComparabilityStatus
	if status == "" {
		status = "not_comparable"
	}
	if row.RawComparable != nil {
		status += fmt.Sprintf(" (raw comparable: %t", *row.RawComparable)
		if row.NormalizedComparable != nil {
			status += fmt.Sprintf(", normalized comparable: %t", *row.NormalizedComparable)
		}
		status += ")"
	}
	if row.ComparabilityReason != "" {
		status += ": " + row.ComparabilityReason
	}
	return status
}

func auditDeltaComparedCurrent(row AuditDeltaRow) *float64 {
	if row.NormalizedCurrent != nil {
		return row.NormalizedCurrent
	}
	return row.Current
}

func auditDeltaComparedBaseline(row AuditDeltaRow) *float64 {
	if row.NormalizedBaseline != nil {
		return row.NormalizedBaseline
	}
	return row.Baseline
}

func auditDeltaComparedDelta(row AuditDeltaRow) *float64 {
	if row.NormalizedDelta != nil {
		return row.NormalizedDelta
	}
	return row.Delta
}

func auditDeltaComparedRegression(row AuditDeltaRow) bool {
	if row.NormalizedRegression != nil {
		return *row.NormalizedRegression
	}
	return row.Regression
}

func escapeAuditMarkdown(value string) string {
	return strings.ReplaceAll(value, "|", "\\|")
}

func auditExemplarIDLinks(ids []string) string {
	if len(ids) == 0 {
		return "—"
	}
	quoted := make([]string, len(ids))
	for i, id := range ids {
		quoted[i] = "`" + id + "`"
	}
	return strings.Join(quoted, ", ")
}
