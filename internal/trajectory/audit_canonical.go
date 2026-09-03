package trajectory

import (
	"sort"
)

// canonicalAuditTranscripts merges exact source transcript identities for aggregate
// calculations. The raw AuditResult.Transcripts rows remain untouched for provenance.
func canonicalAuditTranscripts(raw []AuditTranscriptRow) ([]AuditTranscriptRow, []AuditRefusalRow) {
	byID := make(map[string]*AuditTranscriptRow)
	var refusals []AuditRefusalRow
	for _, row := range raw {
		key := row.Source + "\x00" + row.TranscriptID
		rollup := byID[key]
		if rollup == nil {
			copy := row
			copy.SourcePaths = []string{row.SourcePath}
			copy.ToolResults = append([]AuditToolResultRow(nil), row.ToolResults...)
			copy.usageByID = cloneAuditUsage(row.usageByID)
			copy.failureCounts = cloneAuditFailureCounts(row.failureCounts)
			copy.fragmentDigests = map[string]struct{}{}
			if row.fragmentDigest != "" {
				copy.fragmentDigests[row.fragmentDigest] = struct{}{}
			}
			byID[key] = &copy
			continue
		}
		rollup.SourcePaths = append(rollup.SourcePaths, row.SourcePath)
		rollup.Models = appendUniqueStrings(rollup.Models, row.Models...)
		_, duplicateFragment := rollup.fragmentDigests[row.fragmentDigest]
		duplicateFragment = duplicateFragment && row.fragmentDigest != ""
		if !duplicateFragment {
			rollup.Distribution = mergeAuditDistributionRows(rollup.Distribution, row.Distribution)
			rollup.ToolDistribution = mergeAuditDistributionRows(rollup.ToolDistribution, row.ToolDistribution)
			rollup.StorageDistribution = mergeAuditStorageRows(rollup.StorageDistribution, row.StorageDistribution)
			rollup.UnknownExemplars = mergeAuditUnknownExemplarReservoirs(rollup.UnknownExemplars, row.UnknownExemplars)
			linkAuditUnknownExemplars(rollup.Distribution, rollup.StorageDistribution, rollup.UnknownExemplars.Exemplars)
			if row.fragmentDigest != "" {
				rollup.fragmentDigests[row.fragmentDigest] = struct{}{}
			}
		}
		if row.Source == AuditSourceClaude {
			for id, usage := range row.usageByID {
				if previous, exists := rollup.usageByID[id]; exists {
					if previous != usage {
						refusals = append(refusals, newAuditRefusal(row.Source, row.SourcePath, 0, "claude_duplicate_usage_mismatch", "duplicate message id carries different usage across transcript fragments"))
					}
					continue
				}
				rollup.usageByID[id] = usage
				rollup.Tokens.add(usage)
				rollup.UsageRecords++
			}
		} else if rollup.Tokens == row.Tokens {
			// Codex files carry cumulative transcript usage. Equal totals are split
			// provenance, not additive usage.
		} else {
			refusals = append(refusals, newAuditRefusal(row.Source, row.SourcePath, 0, "codex_fragment_usage_mismatch", "duplicate session id carries different cumulative usage"))
		}
		if duplicateFragment {
			continue
		}
		rollup.ToolCalls += row.ToolCalls
		rollup.ToolErrors += row.ToolErrors
		rollup.ToolResults = mergeAuditToolResultRows(rollup.ToolResults, row.ToolResults)
		if rollup.failureCounts != nil && row.failureCounts != nil {
			for signature, count := range row.failureCounts {
				rollup.failureCounts[signature] += count
			}
		} else {
			rollup.failureCounts = nil
			rollup.RepeatedFailures += row.RepeatedFailures
		}
		rollup.ExpectedWaitTimeouts += row.ExpectedWaitTimeouts
		rollup.MutationChurn += row.MutationChurn
		rollup.TerminalFailures += row.TerminalFailures
		if row.TerminalStallSeconds > rollup.TerminalStallSeconds {
			rollup.TerminalStallSeconds = row.TerminalStallSeconds
		}
		for k, v := range row.TerminalFailureClasses {
			if rollup.TerminalFailureClasses == nil {
				rollup.TerminalFailureClasses = map[string]int{}
			}
			rollup.TerminalFailureClasses[k] += v
		}
		for k, v := range row.TerminalStallDurationBuckets {
			if rollup.TerminalStallDurationBuckets == nil {
				rollup.TerminalStallDurationBuckets = map[string]int{}
			}
			rollup.TerminalStallDurationBuckets[k] += v
		}
	}
	out := make([]AuditTranscriptRow, 0, len(byID))
	for _, row := range byID {
		sort.Strings(row.SourcePaths)
		row.SourcePath = row.SourcePaths[0]
		if row.failureCounts != nil {
			row.RepeatedFailures = auditRepeatedFailureCount(row.failureCounts)
		}
		out = append(out, *row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Source != out[j].Source {
			return out[i].Source < out[j].Source
		}
		return out[i].TranscriptID < out[j].TranscriptID
	})
	return out, refusals
}

func cloneAuditUsage(src map[string]AuditTokens) map[string]AuditTokens {
	dst := make(map[string]AuditTokens, len(src))
	for id, usage := range src {
		dst[id] = usage
	}
	return dst
}

func cloneAuditFailureCounts(src map[string]int) map[string]int {
	if src == nil {
		return nil
	}
	dst := make(map[string]int, len(src))
	for signature, count := range src {
		dst[signature] = count
	}
	return dst
}

func appendUniqueStrings(dst []string, values ...string) []string {
	seen := make(map[string]bool, len(dst)+len(values))
	for _, value := range dst {
		seen[value] = true
	}
	for _, value := range values {
		if !seen[value] {
			dst = append(dst, value)
			seen[value] = true
		}
	}
	sort.Strings(dst)
	return dst
}
