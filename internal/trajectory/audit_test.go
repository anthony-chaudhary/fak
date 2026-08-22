package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunAuditPinnedCrossHarnessParity(t *testing.T) {
	result := runPinnedAudit(t, nil)
	if result.Summary.Transcripts != 2 {
		t.Fatalf("sessions = %d, want 2", result.Summary.Transcripts)
	}
	wantTotals := AuditTokens{InputTokens: 220, OutputTokens: 47, CacheCreateTokens: 75, CacheReadTokens: 330}
	if result.Summary.Tokens != wantTotals {
		t.Fatalf("totals = %+v, want %+v", result.Summary.Tokens, wantTotals)
	}
	if result.Summary.InputOutputRatio == nil || math.Abs(*result.Summary.InputOutputRatio-(625.0/47.0)) > 1e-9 {
		t.Fatalf("input:output = %v, want %v", result.Summary.InputOutputRatio, 625.0/47.0)
	}
	if result.Summary.PromptWriteFraction == nil || math.Abs(*result.Summary.PromptWriteFraction-0.12) > 1e-9 {
		t.Fatalf("cache-create burden = %v, want 0.12", result.Summary.PromptWriteFraction)
	}
	if result.Summary.RepeatedFailures != 1 || result.Summary.MutationChurn != 2 {
		t.Fatalf("behavior = repeats:%d churn:%d, want 1/2", result.Summary.RepeatedFailures, result.Summary.MutationChurn)
	}
	if result.Summary.HookP95MS == nil || *result.Summary.HookP95MS != 120 {
		t.Fatalf("hook p95 = %v, want 120", result.Summary.HookP95MS)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("refusals = %+v, want none", result.Refusals)
	}
	if len(result.Bottlenecks) != 2 {
		t.Fatalf("bottlenecks = %d, want 2", len(result.Bottlenecks))
	}
	top := result.Bottlenecks[0]
	if top.Rank != 1 || top.Source != AuditSourceClaude || top.TranscriptID != "claude-session" || top.AccountedTokens != 462 || top.DominantBucket != "cache_read" {
		t.Fatalf("top bottleneck = %+v", top)
	}

	claude := auditSession(t, result, AuditSourceClaude)
	oracleBytes, err := os.ReadFile(filepath.Join("testdata", "audit", "claude-parity-oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Input       int64 `json:"input_tokens"`
		Output      int64 `json:"output_tokens"`
		CacheRead   int64 `json:"cache_read_tokens"`
		CacheCreate int64 `json:"cache_create_tokens"`
		ToolCalls   int   `json:"tool_calls"`
		ToolErrors  int   `json:"tool_errors"`
	}
	if err := json.Unmarshal(oracleBytes, &oracle); err != nil {
		t.Fatal(err)
	}
	wantClaude := AuditTokens{InputTokens: oracle.Input, OutputTokens: oracle.Output, CacheReadTokens: oracle.CacheRead, CacheCreateTokens: oracle.CacheCreate}
	if claude.Tokens != wantClaude || claude.ToolCalls != oracle.ToolCalls || claude.ToolErrors != oracle.ToolErrors {
		t.Fatalf("Claude parity = tokens:%+v calls:%d errors:%d, oracle tokens:%+v calls:%d errors:%d", claude.Tokens, claude.ToolCalls, claude.ToolErrors, wantClaude, oracle.ToolCalls, oracle.ToolErrors)
	}

	codex := auditSession(t, result, AuditSourceCodex)
	wantCodex := AuditTokens{InputTokens: 90, OutputTokens: 30, CacheReadTokens: 80, CacheCreateTokens: 10}
	if codex.Tokens != wantCodex {
		t.Fatalf("Codex normalized totals = %+v, want %+v", codex.Tokens, wantCodex)
	}
	if codex.UsageRecords != 1 {
		t.Fatalf("Codex applied usage records = %d, want final cumulative row only", codex.UsageRecords)
	}

	denominators := map[string]AuditDenominatorRow{}
	for _, row := range result.Denominators {
		denominators[row.Source] = row
	}
	if got := denominators[AuditSourceClaude]; got.UsageRecordsSeen != 4 || got.UsageRecordsExact != 4 || got.UsageRecordsApplied != 3 || got.DuplicateUsageRecords != 1 {
		t.Fatalf("Claude denominator = %+v", got)
	}
	if got := denominators[AuditSourceCodex]; got.UsageRecordsSeen != 2 || got.UsageRecordsExact != 2 || got.UsageRecordsApplied != 1 {
		t.Fatalf("Codex denominator = %+v", got)
	}
}

func TestAuditVersionedJSONLMarkdownAndBaseline(t *testing.T) {
	ioRatio := 10.0
	cacheBurden := 0.05
	hook := int64(100)
	baseline := &AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", InputOutputRatio: &ioRatio,
		PromptWriteFraction: &cacheBurden, RepeatedFailures: 0, HookP95MS: &hook,
	}
	result := runPinnedAudit(t, baseline)
	if len(result.Baseline) != 4 {
		t.Fatalf("baseline rows = %d, want 4", len(result.Baseline))
	}
	for _, row := range result.Baseline {
		if !row.Comparable || !row.Regression || row.Delta == nil || *row.Delta <= 0 {
			t.Fatalf("baseline delta = %+v, want positive regression", row)
		}
	}

	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(jsonl.Bytes()))
	kinds := map[string]int{}
	for scanner.Scan() {
		var row struct {
			Schema string `json:"schema"`
			Kind   string `json:"kind"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		if row.Schema != AuditSchema {
			t.Fatalf("row schema = %q", row.Schema)
		}
		kinds[row.Kind]++
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"summary", "source_denominator", "session", "bottleneck", "baseline_delta"} {
		if kinds[kind] == 0 {
			t.Fatalf("missing JSONL kind %q in %v", kind, kinds)
		}
	}

	var markdown bytes.Buffer
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Source denominator", "Highest-cost bottleneck: claude/`claude-session`", "Baseline deltas", "Refused transcript shapes"} {
		if !strings.Contains(markdown.String(), want) {
			t.Fatalf("markdown missing %q:\n%s", want, markdown.String())
		}
	}

	baselinePath := filepath.Join(t.TempDir(), "baseline.jsonl")
	file, err := os.Create(baselinePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteAuditJSONL(file, AuditResult{Summary: *baseline}); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadAuditBaseline(baselinePath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.InputOutputRatio == nil || *loaded.InputOutputRatio != ioRatio {
		t.Fatalf("loaded baseline = %+v", loaded)
	}
}

func TestAuditUnsupportedUsageShapeIsExplicit(t *testing.T) {
	tests := []struct {
		name       string
		source     string
		root       string
		rootLabel  string
		code       string
		detailPart string
	}{
		{"claude wrong token type", AuditSourceClaude, filepath.Join("testdata", "audit", "unsupported", "claude", "projects"), "claude/projects", "claude_input_tokens", "non-negative integer"},
		{"codex last-only usage", AuditSourceCodex, filepath.Join("testdata", "audit", "unsupported", "codex", "sessions"), "codex/sessions", "codex_total_usage_missing", "not summed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: test.source, Root: test.root, RootLabel: test.rootLabel}}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Refusals) != 1 {
				t.Fatalf("refusals = %+v, want one", result.Refusals)
			}
			row := result.Refusals[0]
			if row.Code != test.code || !strings.Contains(row.Detail, test.detailPart) {
				t.Fatalf("unsupported row = %+v", row)
			}
			if result.Summary.RefusedRecords != 1 {
				t.Fatalf("summary refused = %d, want 1", result.Summary.RefusedRecords)
			}
		})
	}
}

func runPinnedAudit(t *testing.T, baseline *AuditSummaryRow) AuditResult {
	t.Helper()
	result, err := RunAudit(AuditOptions{
		Sources: []AuditSource{
			{Name: AuditSourceClaude, Root: filepath.Join("testdata", "audit", "claude", "projects"), RootLabel: "claude/projects"},
			{Name: AuditSourceCodex, Root: filepath.Join("testdata", "audit", "codex", "sessions"), RootLabel: "codex/sessions"},
		},
		Baseline: baseline,
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func auditSession(t *testing.T, result AuditResult, source string) AuditTranscriptRow {
	t.Helper()
	for _, row := range result.Transcripts {
		if row.Source == source {
			return row
		}
	}
	t.Fatalf("missing %s session", source)
	return AuditTranscriptRow{}
}
