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
			copy.usageByID = cloneAuditUsage(row.usageByID)
			byID[key] = &copy
			continue
		}
		rollup.SourcePaths = append(rollup.SourcePaths, row.SourcePath)
		rollup.Models = appendUniqueStrings(rollup.Models, row.Models...)
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
		rollup.ToolCalls += row.ToolCalls
		rollup.ToolErrors += row.ToolErrors
		rollup.RepeatedFailures += row.RepeatedFailures
		rollup.MutationChurn += row.MutationChurn
	}
	out := make([]AuditTranscriptRow, 0, len(byID))
	for _, row := range byID {
		sort.Strings(row.SourcePaths)
		row.SourcePath = row.SourcePaths[0]
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
