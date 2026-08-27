package trajectory

import "testing"

func TestCanonicalAuditTranscriptsRollsUpFragmentsAndPreservesRawRows(t *testing.T) {
	raw := []AuditTranscriptRow{
		{Source: AuditSourceClaude, TranscriptID: "same", SourcePath: "b.jsonl", Tokens: AuditTokens{InputTokens: 7}, usageByID: map[string]AuditTokens{"m1": {InputTokens: 7}}, failureCounts: map[string]int{"same failure": 1}, ToolCalls: 2, ToolErrors: 1, ExpectedWaitTimeouts: 1},
		{Source: AuditSourceClaude, TranscriptID: "same", SourcePath: "a.jsonl", Tokens: AuditTokens{OutputTokens: 3}, usageByID: map[string]AuditTokens{"m2": {OutputTokens: 3}}, failureCounts: map[string]int{"same failure": 3}, ToolCalls: 3, RepeatedFailures: 2, ExpectedWaitTimeouts: 2},
		{Source: AuditSourceClaude, TranscriptID: "other", SourcePath: "other.jsonl", Tokens: AuditTokens{InputTokens: 5}, usageByID: map[string]AuditTokens{"m3": {InputTokens: 5}}},
	}
	canonical, refusals := canonicalAuditTranscripts(raw)
	if len(refusals) != 0 {
		t.Fatalf("refusals = %#v", refusals)
	}
	if len(raw) != 3 || len(canonical) != 2 {
		t.Fatalf("counts = raw %d canonical %d", len(raw), len(canonical))
	}
	var same AuditTranscriptRow
	for _, row := range canonical {
		if row.TranscriptID == "same" {
			same = row
		}
	}
	if same.Tokens.InputTokens != 7 || same.Tokens.OutputTokens != 3 || same.ToolCalls != 5 || same.ToolErrors != 1 || same.RepeatedFailures != 3 || same.ExpectedWaitTimeouts != 3 {
		t.Fatalf("rollup = %+v", same)
	}
	if len(same.SourcePaths) != 2 || same.SourcePaths[0] != "a.jsonl" || same.SourcePaths[1] != "b.jsonl" {
		t.Fatalf("provenance = %#v", same.SourcePaths)
	}
	if raw[0].SourcePaths != nil || raw[1].SourcePaths != nil {
		t.Fatalf("raw rows mutated: %#v", raw)
	}
	bottlenecks := rankAuditBottlenecks(canonical)
	if len(bottlenecks) != 2 || bottlenecks[0].TranscriptID != "same" || bottlenecks[0].AccountedTokens != 10 {
		t.Fatalf("bottlenecks = %#v", bottlenecks)
	}
}
func TestCanonicalAuditTranscriptsDeduplicatesClaudeMessageIDs(t *testing.T) {
	usage := AuditTokens{InputTokens: 7, OutputTokens: 3}
	raw := []AuditTranscriptRow{
		{Source: AuditSourceClaude, TranscriptID: "same", SourcePath: "part-1.jsonl", Tokens: usage, UsageRecords: 1, usageByID: map[string]AuditTokens{"m1": usage}},
		{Source: AuditSourceClaude, TranscriptID: "same", SourcePath: "part-2.jsonl", Tokens: usage, UsageRecords: 1, usageByID: map[string]AuditTokens{"m1": usage}},
	}
	canonical, refusals := canonicalAuditTranscripts(raw)
	if len(refusals) != 0 || len(canonical) != 1 {
		t.Fatalf("canonical = %#v, refusals = %#v", canonical, refusals)
	}
	if canonical[0].Tokens != usage || canonical[0].UsageRecords != 1 {
		t.Fatalf("duplicate usage counted twice: %+v", canonical[0])
	}
}

func TestCanonicalAuditTranscriptsRefusesAmbiguousCodexFragments(t *testing.T) {
	raw := []AuditTranscriptRow{
		{Source: AuditSourceCodex, TranscriptID: "same", SourcePath: "part-1.jsonl", Tokens: AuditTokens{InputTokens: 7}},
		{Source: AuditSourceCodex, TranscriptID: "same", SourcePath: "part-2.jsonl", Tokens: AuditTokens{InputTokens: 8}},
	}
	_, refusals := canonicalAuditTranscripts(raw)
	if len(refusals) != 1 || refusals[0].Code != "codex_fragment_usage_mismatch" {
		t.Fatalf("refusals = %#v", refusals)
	}
}
