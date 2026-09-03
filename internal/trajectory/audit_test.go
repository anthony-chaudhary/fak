package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAuditIntegratesToolErrorFamilies(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "session.jsonl")
	lines := []string{
		`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"one","name":"exec_command","input":{}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"one","is_error":true,"content":"permission denied"}]}}`,
		`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"two","name":"exec_command","input":{}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"two","is_error":true,"content":"deadline exceeded"}]}}`,
		`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"three","name":"exec_command","input":{}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"three","is_error":true,"content":"permission denied"}]}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "fixture"}}})
	if err != nil {
		t.Fatal(err)
	}
	want := []QwenToolErrorFamily{
		{Family: "permission", Count: 2, FirstIndex: 2, LastIndex: 6, RepeatedFailures: 1},
		{Family: "timeout", Count: 1, FirstIndex: 4, LastIndex: 4},
	}
	if !reflect.DeepEqual(result.ToolErrorFamilies, want) {
		t.Fatalf("tool error families = %#v, want %#v", result.ToolErrorFamilies, want)
	}

	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{`"tool_error_families"`, `"family":"permission"`, `"first_index":2`, `"tokens":0`} {
		if !bytes.Contains(encoded, []byte(fragment)) {
			t.Fatalf("JSON missing %s: %s", fragment, encoded)
		}
	}
	var markdown bytes.Buffer
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	wantTable := "| Family | Calls | Accounted tokens | Repeated failures | Mutation churn | First event | Last event |\n|---|---:|---:|---:|---:|---:|---:|\n| permission | 2 | 0 | 1 | 0 | 2 | 6 |\n| timeout | 1 | 0 | 0 | 0 | 4 | 4 |"
	if !strings.Contains(markdown.String(), wantTable) {
		t.Fatalf("Markdown missing ranked tool-error table:\n%s", markdown.String())
	}
}

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
	claudeChurn := auditSession(t, result, AuditSourceClaude).MutationChurnEvents
	if len(claudeChurn) != 1 || claudeChurn[0].Target != "fixture.go" || claudeChurn[0].Count != 2 || claudeChurn[0].Intervention != QwenMutationObserveReproFirst {
		t.Fatalf("Claude churn events = %+v, want one attributed observe-only fixture.go run", claudeChurn)
	}
	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(jsonl.String(), "\"kind\":\"mutation_churn\"") || !strings.Contains(jsonl.String(), "\"target\":\"fixture.go\"") {
		t.Fatalf("audit JSONL missing queryable attributed churn row: %s", jsonl.String())
	}
	var rendered bytes.Buffer
	if err := WriteAuditMarkdown(&rendered, result); err != nil {
		t.Fatal(err)
	}
	wantChurnRow := "| claude-session | fixture.go | 2 | "
	if !strings.Contains(rendered.String(), wantChurnRow) {
		t.Fatalf("audit markdown missing captured churn intervention row %q", wantChurnRow)
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
	wantClaude := AuditTokens{InputTokens: 130, OutputTokens: 17, CacheReadTokens: 250, CacheCreateTokens: 65}
	if claude.Tokens != wantClaude || claude.ToolCalls != 4 || claude.ToolErrors != 2 {
		t.Fatalf("Claude exact fixture = tokens:%+v calls:%d errors:%d, want tokens:%+v calls:4 errors:2", claude.Tokens, claude.ToolCalls, claude.ToolErrors, wantClaude)
	}

	codex := auditSession(t, result, AuditSourceCodex)
	wantCodex := AuditTokens{InputTokens: 90, OutputTokens: 30, CacheReadTokens: 80, CacheCreateTokens: 10}
	if codex.Tokens != wantCodex {
		t.Fatalf("Codex normalized totals = %+v, want %+v", codex.Tokens, wantCodex)
	}
	if codex.UsageRecords != 1 {
		t.Fatalf("Codex applied usage records = %d, want final cumulative row only", codex.UsageRecords)
	}
	if codex.CodexCache == nil {
		t.Fatal("Codex cache observation is nil")
	}
	if codex.CodexCache.LastTokenUsageCachedInputMin == nil || codex.CodexCache.LastTokenUsageCachedInputMax == nil {
		t.Fatalf("Codex cache bounds = min:%v max:%v", codex.CodexCache.LastTokenUsageCachedInputMin, codex.CodexCache.LastTokenUsageCachedInputMax)
	}
	if got := *codex.CodexCache; got.TranscriptProducer != AuditSourceCodex ||
		got.ModelProvider != "openai" ||
		got.ModelProviderSource != "session_meta.model_provider" ||
		got.LastTokenUsageCachedInputSamples != 2 ||
		*got.LastTokenUsageCachedInputMin != 40 ||
		*got.LastTokenUsageCachedInputMax != 40 ||
		got.PhysicalProviderCacheResidency != "not_inferable_from_cached_input_tokens" ||
		got.FakOwnedCacheCoverage != "not_observed_by_codex_token_count" {
		t.Fatalf("Codex cache observation = %+v", got)
	}

	var codexJSON bytes.Buffer
	if err := WriteAuditJSONL(&codexJSON, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"codex_cache":{"transcript_producer":"codex","model_provider":"openai","model_provider_source":"session_meta.model_provider"`,
		`"last_token_usage_cached_input_samples":2`,
		`"last_token_usage_cached_input_min":40`,
		`"last_token_usage_cached_input_max":40`,
		`"physical_provider_cache_residency":"not_inferable_from_cached_input_tokens"`,
		`"fak_owned_cache_coverage":"not_observed_by_codex_token_count"`,
	} {
		if !strings.Contains(codexJSON.String(), want) {
			t.Fatalf("audit JSONL missing %q:\n%s", want, codexJSON.String())
		}
	}
	for _, want := range []string{
		"## Codex per-request cache observations",
		"| `codex-session` | `codex` | `openai` | 2 | 40 | 40 |",
		"`cached_input_tokens` is emitted by the transcript producer for the configured provider path; it does not prove physical provider cache residency or process-local ownership.",
		"fak-owned caches are not observed by Codex `token_count` rows",
	} {
		if !strings.Contains(rendered.String(), want) {
			t.Fatalf("audit markdown missing %q:\n%s", want, rendered.String())
		}
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

func TestAuditCodexCacheBoundsPreserveCumulativeAccounting(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "bounded-cache.jsonl")
	lines := []string{
		`{"type":"session_meta","payload":{"id":"bounded-cache","model_provider":"fak"}}`,
		`{"type":"session_meta","payload":{"id":"copied-parent","model_provider":"openai"}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"cached_input_tokens":0},"total_token_usage":{"input_tokens":100,"cached_input_tokens":10,"cache_write_input_tokens":5,"output_tokens":10}}}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"cached_input_tokens":80},"total_token_usage":{"input_tokens":200,"cached_input_tokens":30,"cache_write_input_tokens":10,"output_tokens":20}}}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"cached_input_tokens":40},"total_token_usage":{"input_tokens":300,"cached_input_tokens":60,"cache_write_input_tokens":20,"output_tokens":30}}}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("refusals = %+v", result.Refusals)
	}
	if len(result.Transcripts) != 1 {
		t.Fatalf("transcripts = %d, want 1", len(result.Transcripts))
	}
	row := result.Transcripts[0]
	wantTokens := AuditTokens{InputTokens: 220, OutputTokens: 30, CacheReadTokens: 60, CacheCreateTokens: 20}
	if row.Tokens != wantTokens || row.UsageRecords != 1 {
		t.Fatalf("cumulative accounting = tokens:%+v usage_records:%d, want %+v/1", row.Tokens, row.UsageRecords, wantTokens)
	}
	if row.CodexCache == nil || row.CodexCache.LastTokenUsageCachedInputMin == nil || row.CodexCache.LastTokenUsageCachedInputMax == nil {
		t.Fatalf("cache observation = %+v", row.CodexCache)
	}
	if got := row.CodexCache; got.ModelProvider != "fak" ||
		got.LastTokenUsageCachedInputSamples != 3 ||
		*got.LastTokenUsageCachedInputMin != 0 ||
		*got.LastTokenUsageCachedInputMax != 80 {
		t.Fatalf("cache observation = %+v", got)
	}
	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"model_provider":"fak"`,
		`"last_token_usage_cached_input_samples":3`,
		`"last_token_usage_cached_input_min":0`,
		`"last_token_usage_cached_input_max":80`,
	} {
		if !strings.Contains(jsonl.String(), want) {
			t.Fatalf("JSONL missing %q:\n%s", want, jsonl.String())
		}
	}
}

func TestAuditIssue9418PreservesPrimaryCodexFragmentIdentity(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9418", "codex-shared-session", "sessions")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("refusals = %+v, want none for distinct primary rollout ids", result.Refusals)
	}
	if result.Summary.Transcripts != 2 || result.Summary.RawFragments != 2 || result.Summary.CanonicalTranscripts != 2 || result.Summary.DuplicateFragments != 0 {
		t.Fatalf("summary fragment counts = %+v, want two distinct exact rollouts", result.Summary)
	}
	wantTokens := AuditTokens{InputTokens: 62, OutputTokens: 18, CacheCreateTokens: 8, CacheReadTokens: 100}
	if result.Summary.Tokens != wantTokens {
		t.Fatalf("tokens = %+v, want each rollout counted once as %+v", result.Summary.Tokens, wantTokens)
	}
	ids := map[string]bool{}
	for _, row := range result.Transcripts {
		ids[row.TranscriptID] = true
	}
	if !ids["root-rollout"] || !ids["child-rollout"] {
		t.Fatalf("transcript ids = %#v, want file-local root and child ids", ids)
	}
	denominator := result.Denominators[0]
	if denominator.UsageRecordsSeen != 2 || denominator.UsageRecordsExact != 2 || denominator.UsageRecordsApplied != 2 || denominator.RefusedRecords != 0 {
		t.Fatalf("denominator = %+v, want two exact applied rollouts and no refusal", denominator)
	}
}

func TestAuditIssue9418ClaudeDuplicateMismatchRemainsTyped(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9418", "claude-duplicate", "projects")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 1 || result.Refusals[0].Code != "claude_duplicate_usage_mismatch" {
		t.Fatalf("refusals = %+v, want one typed Claude duplicate mismatch", result.Refusals)
	}
	if result.Summary.RefusedRecords != 1 || result.Denominators[0].RefusedRecords != 1 {
		t.Fatalf("refusal totals = summary:%d denominator:%d, want 1/1", result.Summary.RefusedRecords, result.Denominators[0].RefusedRecords)
	}
	if result.ConclusionStatus.BroadEfficiencySupported || result.ConclusionStatus.RefusalCount != 1 {
		t.Fatalf("conclusion = %+v, want blocked by one refusal", result.ConclusionStatus)
	}
}

func TestAuditIssue9418CanonicalRefusalTotalsAgreeEverywhere(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9418", "codex-ambiguous", "sessions")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 1 || result.Refusals[0].Code != "codex_fragment_usage_mismatch" {
		t.Fatalf("refusals = %+v, want one canonical fragment refusal", result.Refusals)
	}
	if result.Summary.RefusedRecords != 1 || result.Denominators[0].RefusedRecords != 1 {
		t.Fatalf("refusal totals = summary:%d denominator:%d, want 1/1", result.Summary.RefusedRecords, result.Denominators[0].RefusedRecords)
	}
	if result.ConclusionStatus.BroadEfficiencySupported || result.ConclusionStatus.RefusalCount != 1 {
		t.Fatalf("conclusion = %+v, want blocked by one refusal", result.ConclusionStatus)
	}

	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	var summaryRefused, denominatorRefused, refusalRows int
	scanner := bufio.NewScanner(bytes.NewReader(jsonl.Bytes()))
	for scanner.Scan() {
		var row struct {
			Kind           string `json:"kind"`
			RefusedRecords int    `json:"refused_records"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		switch row.Kind {
		case "summary":
			summaryRefused = row.RefusedRecords
		case "source_denominator":
			denominatorRefused += row.RefusedRecords
		case "refusal":
			refusalRows++
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if summaryRefused != 1 || denominatorRefused != 1 || refusalRows != 1 {
		t.Fatalf("JSONL refusal totals = summary:%d denominators:%d rows:%d, want 1/1/1", summaryRefused, denominatorRefused, refusalRows)
	}

	var markdown bytes.Buffer
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Broad efficiency conclusions: **blocked** (refusals: 1).", "`codex_fragment_usage_mismatch`"} {
		if !strings.Contains(markdown.String(), want) {
			t.Fatalf("markdown missing %q:\n%s", want, markdown.String())
		}
	}
}

func TestAuditExcludesPinnedClaudePytestFixture(t *testing.T) {
	baseline := runPinnedAudit(t, nil)
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{
		{Name: AuditSourceClaude, Root: filepath.Join("testdata", "audit", "claude", "projects"), RootLabel: "claude/projects"},
		{Name: AuditSourceCodex, Root: filepath.Join("testdata", "audit", "codex", "sessions"), RootLabel: "codex/sessions"},
		{Name: AuditSourceClaude, Root: filepath.Join("testdata", "a", "i8508", "c", "p"), RootLabel: "claude/projects/pytest-fixture"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("pytest fixture refusals = %+v, want none", result.Refusals)
	}
	if result.Summary.Tokens != baseline.Summary.Tokens {
		t.Fatalf("tokens with fixture = %+v, want supported totals %+v", result.Summary.Tokens, baseline.Summary.Tokens)
	}
	if result.Summary.Transcripts != baseline.Summary.Transcripts {
		t.Fatalf("sessions with fixture = %d, want supported sessions %d", result.Summary.Transcripts, baseline.Summary.Transcripts)
	}
	var fixtureDenominator AuditDenominatorRow
	for _, row := range result.Denominators {
		if row.Root == "claude/projects/pytest-fixture" {
			fixtureDenominator = row
		}
	}
	if fixtureDenominator.FilesDiscovered != 1 || fixtureDenominator.FilesScanned != 0 || fixtureDenominator.FixtureFilesExcluded != 1 {
		t.Fatalf("pytest fixture denominator = %+v, want discovered=1 scanned=0 non-session=1", fixtureDenominator)
	}
}

func TestAuditClaudePytestLookalikeKeepsExactUsage(t *testing.T) {
	pinned := filepath.Join("testdata", "a", "i8508", "c", "p",
		"C--Users-U-AppData-Local-Temp-pytest-of-U-pytest-1-test_main_json_runs_and_emits_0-ws",
		"00000000-aaaa-bbbb-cccc-000000000000.jsonl")
	data, err := os.ReadFile(pinned)
	if err != nil {
		t.Fatal(err)
	}
	data = bytes.ReplaceAll(data, []byte(`"usage": {}`), []byte(`"usage": {"input_tokens": 7, "output_tokens": 2}`))
	root := filepath.Join(t.TempDir(), "projects")
	destination := filepath.Join(root,
		"C--Users-U-AppData-Local-Temp-pytest-of-U-pytest-1-test_main_json_runs_and_emits_0-ws",
		"00000000-aaaa-bbbb-cccc-000000000000.jsonl")
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, data, 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}}})
	if err != nil {
		t.Fatal(err)
	}
	want := AuditTokens{InputTokens: 28, OutputTokens: 8}
	if len(result.Refusals) != 0 || result.Summary.Tokens != want || result.Summary.UsageRecordsExact != 4 || result.Summary.Transcripts != 1 {
		t.Fatalf("supported lookalike = summary:%+v refusals:%+v, want exact tokens %+v", result.Summary, result.Refusals, want)
	}
	if result.Summary.FixtureFilesExcluded != 0 {
		t.Fatalf("supported lookalike classified non-session: %+v", result.Summary)
	}
}

func TestAuditClaudePytestFixtureVariants(t *testing.T) {
	pinned := filepath.Join("testdata", "a", "i8508", "c", "p",
		"C--Users-U-AppData-Local-Temp-pytest-of-U-pytest-1-test_main_json_runs_and_emits_0-ws",
		"00000000-aaaa-bbbb-cccc-000000000000.jsonl")
	data, err := os.ReadFile(pinned)
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(t.TempDir(), "projects")
	for _, name := range []string{
		"test_main_json_runs_and_emits_0",
		"test_route_findings_off_writes0",
		"test_route_findings_on_appends0",
	} {
		variant := bytes.ReplaceAll(data, []byte("test_main_json_runs_and_emits_0"), []byte(name))
		destination := filepath.Join(root,
			"C--Users-U-AppData-Local-Temp-pytest-of-U-pytest-1-"+name+"-ws",
			"00000000-aaaa-bbbb-cccc-000000000000.jsonl")
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(destination, variant, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 || result.Summary.Transcripts != 0 || result.Summary.FixtureFilesExcluded != 3 || result.Summary.FilesDiscovered != 3 {
		t.Fatalf("pytest fixture variants = summary:%+v refusals:%+v", result.Summary, result.Refusals)
	}
}

func TestAuditVersionedJSONLMarkdownAndBaseline(t *testing.T) {
	ioRatio := 10.0
	cacheBurden := 0.05
	hook := int64(100)
	baseline := &AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", InputOutputRatio: &ioRatio,
		PromptWriteFraction: &cacheBurden, RepeatedFailures: 0, Transcripts: 1, ToolCalls: 5, HookP95MS: &hook,
		RepeatedFailureSemantics: auditRepeatedFailureSemantics, RepeatedFailureNormalization: auditRepeatedFailureNormalization,
	}
	result := runPinnedAudit(t, baseline)
	if len(result.Baseline) != 4 {
		t.Fatalf("baseline rows = %d, want 4", len(result.Baseline))
	}
	for _, row := range result.Baseline {
		if row.Metric == "repeated_failures" {
			if row.Comparable || row.Regression || row.RawComparable == nil || *row.RawComparable ||
				row.NormalizedComparable == nil || !*row.NormalizedComparable || row.NormalizedRegression == nil || !*row.NormalizedRegression ||
				row.NormalizedDelta == nil || *row.NormalizedDelta <= 0 {
				t.Fatalf("normalized repeated-failure delta = %+v, want raw-incomparable normalized regression", row)
			}
			continue
		}
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

func TestAuditSeparatesExpectedWaitTimeoutsAndNormalizesRegression(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9417")
	baseline, err := RunAudit(AuditOptions{Sources: []AuditSource{{
		Name: AuditSourceClaude, Root: filepath.Join(root, "baseline", "claude", "projects"), RootLabel: "claude/projects",
	}}})
	if err != nil {
		t.Fatal(err)
	}
	current, err := RunAudit(AuditOptions{
		Sources: []AuditSource{
			{Name: AuditSourceClaude, Root: filepath.Join(root, "current", "claude", "projects"), RootLabel: "claude/projects"},
			{Name: AuditSourceCodex, Root: filepath.Join(root, "current", "codex", "sessions"), RootLabel: "codex/sessions"},
		},
		Baseline: &baseline.Summary,
	})
	if err != nil {
		t.Fatal(err)
	}
	if current.Summary.Transcripts != 4 || current.Summary.ToolCalls != 11 {
		t.Fatalf("current exposure = sessions:%d calls:%d, want 4/11", current.Summary.Transcripts, current.Summary.ToolCalls)
	}
	if current.Summary.ToolErrors != 4 || current.Summary.RepeatedFailures != 3 {
		t.Fatalf("current actionable failures = errors:%d repeats:%d, want 4/3", current.Summary.ToolErrors, current.Summary.RepeatedFailures)
	}
	if current.Summary.ExpectedWaitTimeouts != 6 || current.Summary.RepeatedFailuresPerSession == nil || *current.Summary.RepeatedFailuresPerSession != 0.75 {
		t.Fatalf("current wait/rate classification = waits:%d rate:%v, want 6/0.75", current.Summary.ExpectedWaitTimeouts, current.Summary.RepeatedFailuresPerSession)
	}
	transcripts := make(map[string]AuditTranscriptRow, len(current.Transcripts))
	for _, row := range current.Transcripts {
		transcripts[row.TranscriptID] = row
	}
	waits := transcripts["issue-9417-current-waits"]
	if waits.ToolCalls != 4 || waits.ExpectedWaitTimeouts != 4 || waits.ToolErrors != 0 || waits.RepeatedFailures != 0 {
		t.Fatalf("bounded namespaced wait_agent polling = %+v, want calls/waits/errors/repeats 4/4/0/0", waits)
	}
	codexWaits := transcripts["issue-9417-codex-waits"]
	if codexWaits.ToolCalls != 2 || codexWaits.ExpectedWaitTimeouts != 2 || codexWaits.ToolErrors != 0 || codexWaits.RepeatedFailures != 0 {
		t.Fatalf("Codex bounded wait_agent polling = %+v, want calls/waits/errors/repeats 2/2/0/0", codexWaits)
	}
	failures := transcripts["issue-9417-current-failures"]
	if failures.ToolCalls != 4 || failures.ExpectedWaitTimeouts != 0 || failures.ToolErrors != 4 || failures.RepeatedFailures != 3 {
		t.Fatalf("identical non-wait timeout failures = %+v, want calls/waits/errors/repeats 4/0/4/3", failures)
	}
	var repeatedDelta AuditDeltaRow
	for _, row := range current.Baseline {
		if row.Metric == "repeated_failures" {
			repeatedDelta = row
		}
	}
	if repeatedDelta.RawCurrent == nil || *repeatedDelta.RawCurrent != 3 || repeatedDelta.RawBaseline == nil || *repeatedDelta.RawBaseline != 1 ||
		repeatedDelta.CurrentExposure == nil || *repeatedDelta.CurrentExposure != 4 || repeatedDelta.BaselineExposure == nil || *repeatedDelta.BaselineExposure != 1 {
		t.Fatalf("repeated-failure raw counts/exposure = %+v, want 3/1 over 4/1", repeatedDelta)
	}
	if repeatedDelta.RawComparable == nil || *repeatedDelta.RawComparable || repeatedDelta.Comparable || repeatedDelta.Regression ||
		repeatedDelta.NormalizedComparable == nil || !*repeatedDelta.NormalizedComparable ||
		repeatedDelta.NormalizedRegression == nil || *repeatedDelta.NormalizedRegression ||
		repeatedDelta.ComparabilityStatus != "normalized_only" {
		t.Fatalf("unequal exposure comparability = %+v, want raw-incomparable normalized non-regression", repeatedDelta)
	}
	if repeatedDelta.Current == nil || *repeatedDelta.Current != 3 || repeatedDelta.Baseline == nil || *repeatedDelta.Baseline != 1 || repeatedDelta.Delta != nil ||
		repeatedDelta.NormalizedCurrent == nil || *repeatedDelta.NormalizedCurrent != 0.75 ||
		repeatedDelta.NormalizedBaseline == nil || *repeatedDelta.NormalizedBaseline != 1 ||
		repeatedDelta.NormalizedDelta == nil || *repeatedDelta.NormalizedDelta != -0.25 {
		t.Fatalf("normalized repeated-failure delta = %+v, want 0.75 - 1 = -0.25", repeatedDelta)
	}
	waitSummary := summarizeAudit(nil, []AuditTranscriptRow{waits, codexWaits}, nil)
	waitDelta := auditBaselineDeltas(waitSummary, baseline.Summary)[2]
	if waitDelta.NormalizedComparable == nil || !*waitDelta.NormalizedComparable || waitDelta.NormalizedRegression == nil || *waitDelta.NormalizedRegression ||
		waitDelta.NormalizedCurrent == nil || *waitDelta.NormalizedCurrent != 0 {
		t.Fatalf("expected waits alone created a regression verdict: %+v", waitDelta)
	}
	cleanSummary := summarizeAudit(nil, []AuditTranscriptRow{{TranscriptID: "clean"}}, nil)
	failureSummary := summarizeAudit(nil, []AuditTranscriptRow{failures}, nil)
	failureDelta := auditBaselineDeltas(failureSummary, cleanSummary)[2]
	if !failureDelta.Comparable || !failureDelta.Regression ||
		failureDelta.NormalizedComparable == nil || !*failureDelta.NormalizedComparable ||
		failureDelta.NormalizedRegression == nil || !*failureDelta.NormalizedRegression {
		t.Fatalf("identical failed action lost its regression verdict: %+v", failureDelta)
	}

	var jsonl bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, current); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"expected_wait_timeouts":6`, `"repeated_failures_per_session":0.75`,
		`"repeated_failure_semantics":"actionable-identical-failed-action/2"`,
		`"current":3`, `"baseline":1`, `"delta":null`, `"raw_current":3`, `"raw_baseline":1`,
		`"current_exposure":4`, `"baseline_exposure":1`, `"normalization":"per_session"`,
		`"raw_comparable":false`, `"normalized_current":0.75`, `"normalized_baseline":1`,
		`"normalized_delta":-0.25`, `"normalized_comparable":true`, `"normalized_regression":false`,
		`"comparable":false`, `"comparability_status":"normalized_only"`, `"regression":false`,
	} {
		if !strings.Contains(jsonl.String(), want) {
			t.Fatalf("JSONL missing %q:\n%s", want, jsonl.String())
		}
	}

	var markdown bytes.Buffer
	if err := WriteAuditMarkdown(&markdown, current); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"Repeated failures: 3/4 sessions (0.7500 per session); expected bounded wait_agent timeouts: 6",
		"| repeated_failures | 3 / 1 | 0.7500 / 1.0000 | per_session (4 / 1) | -0.2500 | normalized_only (raw comparable: false, normalized comparable: true): raw counts have unequal session exposure; only the per-session rate is comparable | false |",
	} {
		if !strings.Contains(markdown.String(), want) {
			t.Fatalf("markdown missing %q:\n%s", want, markdown.String())
		}
	}
}

func TestAuditRepeatedFailureDeltaRefusesLegacyBaselineSemantics(t *testing.T) {
	currentRatio, baselineRatio := 2.0, 1.0
	current := AuditSummaryRow{
		InputOutputRatio: &currentRatio, RepeatedFailures: 3, Transcripts: 4, ToolCalls: 8,
		RepeatedFailureSemantics: auditRepeatedFailureSemantics, RepeatedFailureNormalization: auditRepeatedFailureNormalization,
	}
	legacy := AuditSummaryRow{InputOutputRatio: &baselineRatio, RepeatedFailures: 1, Transcripts: 2, ToolCalls: 2}
	deltas := auditBaselineDeltas(current, legacy)
	if !deltas[0].Comparable || !deltas[0].Regression || deltas[0].ComparabilityStatus != "comparable" {
		t.Fatalf("unrelated ratio delta must remain comparable: %+v", deltas[0])
	}
	repeated := deltas[2]
	if repeated.Metric != "repeated_failures" || repeated.Comparable || repeated.Regression || repeated.ComparabilityStatus != "semantics_mismatch" {
		t.Fatalf("legacy repeated-failure baseline did not fail closed: %+v", repeated)
	}
	if repeated.RawCurrent == nil || *repeated.RawCurrent != 3 || repeated.RawBaseline == nil || *repeated.RawBaseline != 1 ||
		repeated.Current == nil || *repeated.Current != 3 || repeated.Baseline == nil || *repeated.Baseline != 1 || repeated.Delta != nil ||
		repeated.NormalizedCurrent == nil || *repeated.NormalizedCurrent != 0.75 ||
		repeated.NormalizedBaseline == nil || *repeated.NormalizedBaseline != 0.5 || repeated.NormalizedDelta != nil ||
		repeated.NormalizedComparable == nil || *repeated.NormalizedComparable ||
		repeated.NormalizedRegression == nil || *repeated.NormalizedRegression {
		t.Fatalf("legacy baseline raw/current-normalized rendering = %+v", repeated)
	}
}

func TestAuditPrivacySafeLiveReplayFailsClosedAgainstLegacyBaseline(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-9417", "live-replay")
	current, err := ReadAuditBaseline(filepath.Join(root, "current-summary.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := ReadAuditBaseline(filepath.Join(root, "baseline-summary.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if current.Transcripts != 3679 || current.ToolCalls != 277518 || current.ExpectedWaitTimeouts != 523 || current.RepeatedFailures != 0 {
		t.Fatalf("privacy-safe current replay = %+v", current)
	}
	repeated := auditBaselineDeltas(*current, *baseline)[2]
	if repeated.RawCurrent == nil || *repeated.RawCurrent != 0 || repeated.RawBaseline == nil || *repeated.RawBaseline != 12 ||
		repeated.CurrentExposure == nil || *repeated.CurrentExposure != 3679 ||
		repeated.BaselineExposure == nil || *repeated.BaselineExposure != 889 ||
		repeated.Comparable || repeated.Regression || repeated.ComparabilityStatus != "semantics_mismatch" ||
		repeated.NormalizedComparable == nil || *repeated.NormalizedComparable ||
		repeated.NormalizedRegression == nil || *repeated.NormalizedRegression {
		t.Fatalf("privacy-safe legacy comparison did not fail closed: %+v", repeated)
	}
}

func TestAuditExpectedWaitTimeoutClassifierIsToolSpecific(t *testing.T) {
	timeout := map[string]any{"status": map[string]any{}, "timed_out": true}
	for _, name := range []string{"wait_agent", "functions.wait_agent", "tools/wait_agent", "mcp:wait_agent", "mcp__dos__wait_agent"} {
		if !auditExpectedWaitTimeout(auditToolCall{name: name}, timeout) {
			t.Errorf("namespaced bounded wait %q was not classified as expected", name)
		}
	}
	for _, test := range []struct {
		name   string
		output any
	}{
		{name: "exec_command", output: timeout},
		{name: "not_wait_agent", output: timeout},
		{name: "wait_agent", output: map[string]any{"status": "failed", "error": "unknown worker"}},
		{name: "wait_agent", output: map[string]any{"timed_out": true, "status": "failed", "error": "worker crashed"}},
		{name: "wait_agent", output: "timeout waiting for the worker service"},
	} {
		if auditExpectedWaitTimeout(auditToolCall{name: test.name}, test.output) {
			t.Errorf("non-expected outcome for %q was classified as bounded polling: %#v", test.name, test.output)
		}
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

func TestAuditUserContainsSelectsTopicalCohort(t *testing.T) {
	root := t.TempDir()
	qwen := `{"type":"session_meta","payload":{"id":"qwen-session"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Measure QWEN decode performance"}]}}`
	other := `{"type":"session_meta","payload":{"id":"other-session"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Unrelated task"}]}}`
	if err := os.WriteFile(filepath.Join(root, "qwen.jsonl"), []byte(qwen), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "other.jsonl"), []byte(other), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := RunAudit(AuditOptions{
		Sources:      []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}},
		UserContains: "qwen",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Transcripts) != 1 || result.Transcripts[0].TranscriptID != "qwen-session" {
		t.Fatalf("transcripts = %#v, want only qwen-session", result.Transcripts)
	}
	denominator := result.Denominators[0]
	if denominator.FilesDiscovered != 2 || denominator.FilesMatched != 1 || denominator.FilesScanned != 1 {
		t.Fatalf("denominator = %#v, want discovered=2 matched=1 scanned=1", denominator)
	}
	if result.Summary.FilesMatched != 1 || result.Summary.Transcripts != 1 {
		t.Fatalf("summary = %#v, want matched=1 sessions=1", result.Summary)
	}
	if result.Summary.DistinctTranscripts != 1 || result.Summary.DuplicateFragments != 0 || result.Summary.EmptyUsageFiles != 1 {
		t.Fatalf("summary process signals = %#v", result.Summary)
	}
}

func TestAuditUserContainsRejectsNonUserEchoesAcrossSources(t *testing.T) {
	root := t.TempDir()
	rows := map[string]string{
		"codex-user.jsonl": `{"type":"session_meta","payload":{"id":"codex-user"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"qwen task"}]}}`,
		"codex-tool.jsonl": `{"type":"session_meta","payload":{"id":"codex-tool"}}
{"type":"response_item","payload":{"type":"function_call_output","output":"qwen echoed by tool"}}`,
		"codex-system.jsonl": `{"type":"session_meta","payload":{"id":"codex-system"}}
{"type":"response_item","payload":{"type":"message","role":"system","content":[{"type":"input_text","text":"qwen injected context"}]}}`,
	}
	for name, body := range rows {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}}, UserContains: "QWEN"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Transcripts) != 1 || result.Transcripts[0].TranscriptID != "codex-user" {
		t.Fatalf("transcripts=%+v", result.Transcripts)
	}
	if got := result.Denominators[0]; got.FilesDiscovered != 3 || got.FilesMatched != 1 || got.FilesScanned != 1 {
		t.Fatalf("denominator=%+v", got)
	}
}

func TestAuditUserContainsRejectsCodexRepositoryInstructions(t *testing.T) {
	root := t.TempDir()
	instructions := `{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"# AGENTS.md instructions for /workspace/project\n\n<INSTRUCTIONS>\nNative inference work prefers Qwen.\n</INSTRUCTIONS>"},{"type":"input_text","text":"<environment_context>\n  <cwd>/workspace/project</cwd>\n</environment_context>"}]}}`
	rows := map[string]string{
		"injected-only.jsonl": `{"type":"session_meta","payload":{"id":"injected-only"}}
` + instructions + `
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Audit the documentation index"}]}}`,
		"operator-task.jsonl": `{"type":"session_meta","payload":{"id":"operator-task"}}
` + instructions + `
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Measure Qwen decode performance"}]}}`,
	}
	for name, body := range rows {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	result, err := RunAudit(AuditOptions{
		Sources:      []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}},
		UserContains: "qwen",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Transcripts) != 1 || result.Transcripts[0].TranscriptID != "operator-task" {
		t.Fatalf("transcripts = %#v, want only operator-task", result.Transcripts)
	}
	if got := result.Denominators[0]; got.FilesDiscovered != 2 || got.FilesMatched != 1 || got.FilesScanned != 1 {
		t.Fatalf("denominator = %#v, want discovered=2 matched=1 scanned=1", got)
	}
}

func TestAuditUserTextRejectsClaudeToolAndInjectedRows(t *testing.T) {
	user := map[string]any{"type": "user", "message": map[string]any{"role": "user", "content": "qwen task"}}
	tool := map[string]any{"type": "assistant", "message": map[string]any{"role": "assistant", "content": "qwen tool echo"}}
	injected := map[string]any{"type": "user", "message": map[string]any{"role": "system", "content": "qwen injected"}}
	if auditUserText(user) != "qwen task" || auditUserText(tool) != "" || auditUserText(injected) != "" {
		t.Fatal("only user-authored Claude text may select the cohort")
	}
}
func TestAuditEdgeMatrix(t *testing.T) {
	fixtureRoot := filepath.Join("testdata", "audit", "issue-8493", "edge", "empty")
	for _, test := range []struct {
		name   string
		source AuditSource
	}{
		{"empty Claude file", AuditSource{Name: AuditSourceClaude, Root: filepath.Join(fixtureRoot, "claude", "projects"), RootLabel: "claude/projects"}},
		{"empty Codex file", AuditSource{Name: AuditSourceCodex, Root: filepath.Join(fixtureRoot, "codex", "sessions"), RootLabel: "codex/sessions"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{test.source}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Denominators) != 1 || !result.Denominators[0].RootPresent || result.Denominators[0].FilesScanned != 1 || result.Denominators[0].Records != 0 {
				t.Fatalf("empty denominator = %+v", result.Denominators)
			}
			if len(result.Transcripts) != 1 || result.Transcripts[0].Tokens != (AuditTokens{}) || len(result.Refusals) != 0 {
				t.Fatalf("empty result = transcripts:%+v refusals:%+v", result.Transcripts, result.Refusals)
			}
		})
	}

	for _, source := range []string{AuditSourceClaude, AuditSourceCodex} {
		t.Run("missing "+source+" root", func(t *testing.T) {
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: source, Root: filepath.Join(t.TempDir(), "missing"), RootLabel: source + "/missing"}}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Denominators) != 1 || result.Denominators[0].RootPresent || result.Denominators[0].FilesDiscovered != 0 || len(result.Transcripts) != 0 || len(result.Refusals) != 0 {
				t.Fatalf("missing-root result = %+v", result)
			}
		})
	}
}

func TestAuditCodexCumulativeUsageResetAtTaskBoundary(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-8509", "codex-reset", "sessions")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "codex/sessions"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("refusals = %+v, want none", result.Refusals)
	}
	if len(result.Transcripts) != 1 {
		t.Fatalf("sessions = %d, want 1", len(result.Transcripts))
	}
	wantTokens := AuditTokens{InputTokens: 18, OutputTokens: 7, CacheReadTokens: 8, CacheCreateTokens: 2}
	if got := result.Transcripts[0]; got.Tokens != wantTokens || got.UsageRecords != 2 {
		t.Fatalf("Codex reset accounting = tokens:%+v applied:%d, want tokens:%+v applied:2", got.Tokens, got.UsageRecords, wantTokens)
	}
	if got := result.Denominators[0]; got.UsageRecordsSeen != 4 || got.UsageRecordsExact != 4 || got.UsageRecordsApplied != 2 || got.TokenSemantics != "final cumulative input per segment; only a versioned task_started boundary may begin a segment after a decrease; cached/cache-write subsets remain exact subtraction" {
		t.Fatalf("Codex reset denominator = %+v", got)
	}
}

func TestAuditCodexResetBoundaryFailsClosed(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-8509")
	for _, test := range []struct {
		name string
		dir  string
	}{
		{name: "unversioned task boundary", dir: "codex-unversioned-reset"},
		{name: "task boundary without turn id", dir: "codex-untyped-reset"},
		{name: "consumed task boundary", dir: "codex-consumed-boundary"},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: filepath.Join(root, test.dir, "sessions"), RootLabel: "codex/sessions"}}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Refusals) != 1 || result.Refusals[0].Code != "codex_total_usage_decreased" || !strings.Contains(result.Refusals[0].Detail, "without a versioned task_started boundary") {
				t.Fatalf("refusals = %+v, want one unexplained-decrease refusal", result.Refusals)
			}
		})
	}
}

func TestAuditRefusalMatrix(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-8493", "refusals")
	tests := []struct {
		name       string
		source     AuditSource
		code       string
		seen       int
		exact      int
		applied    int
		wantTokens AuditTokens
		oversized  bool
	}{
		{
			name: "Claude duplicate usage mismatch", source: AuditSource{Name: AuditSourceClaude, Root: filepath.Join(root, "claude-duplicate", "projects"), RootLabel: "claude/projects"},
			code: "claude_duplicate_usage_mismatch", seen: 2, exact: 1, applied: 1,
			wantTokens: AuditTokens{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 3, CacheCreateTokens: 4},
		},
		{
			name: "Codex cumulative total decreased", source: AuditSource{Name: AuditSourceCodex, Root: filepath.Join(root, "codex-decreasing", "sessions"), RootLabel: "codex/sessions"},
			code: "codex_total_usage_decreased", seen: 2, exact: 1, applied: 1,
			wantTokens: AuditTokens{InputTokens: 6, OutputTokens: 2, CacheReadTokens: 3, CacheCreateTokens: 1},
		},
		{
			name: "malformed JSON", source: AuditSource{Name: AuditSourceClaude, Root: filepath.Join(root, "malformed", "claude", "projects"), RootLabel: "claude/projects"},
			code: "malformed_json",
		},
		{
			name: "oversized line", source: AuditSource{Name: AuditSourceClaude, Root: filepath.Join(root, "oversized", "claude", "projects"), RootLabel: "claude/projects"},
			code: "line_too_large", oversized: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var result AuditResult
			if !test.oversized {
				var err error
				result, err = RunAudit(AuditOptions{Sources: []AuditSource{test.source}})
				if err != nil {
					t.Fatal(err)
				}
			} else {
				matches, err := filepath.Glob(filepath.Join(test.source.Root, "fak", "*.jsonl"))
				if err != nil || len(matches) != 1 {
					t.Fatalf("oversized fixture matches=%v err=%v", matches, err)
				}
				seed, err := os.ReadFile(matches[0])
				if err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(t.TempDir(), "oversized.jsonl")
				file, err := os.Create(path)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := file.Write(bytes.TrimSpace(seed)); err != nil {
					file.Close()
					t.Fatal(err)
				}
				chunk := bytes.Repeat([]byte("x"), 1024*1024)
				for written := len(seed); written <= 32*1024*1024; written += len(chunk) {
					if _, err := file.Write(chunk); err != nil {
						file.Close()
						t.Fatal(err)
					}
				}
				if err := file.Close(); err != nil {
					t.Fatal(err)
				}
				denominator := AuditDenominatorRow{Schema: AuditSchema, Kind: "source_denominator", Source: test.source.Name, Root: test.source.RootLabel, RootPresent: true, FilesDiscovered: 1, RecordTypes: map[string]int{}}
				row, refusals, hooks, _, err := parseAuditFile(test.source.Name, path, "fak/oversized.jsonl", &denominator)
				if err != nil {
					t.Fatal(err)
				}
				denominator.FilesScanned = 1
				result = AuditResult{Denominators: []AuditDenominatorRow{denominator}, Transcripts: []AuditTranscriptRow{row}, Refusals: refusals}
				result.Summary = summarizeAudit(result.Denominators, result.Transcripts, hooks)
			}

			if len(result.Refusals) != 1 || result.Refusals[0].Schema != AuditSchema || result.Refusals[0].Code != test.code {
				t.Fatalf("refusals = %+v, want one %s row", result.Refusals, test.code)
			}
			denominator := result.Denominators[0]
			if denominator.RefusedRecords != 1 || result.Summary.RefusedRecords != 1 {
				t.Fatalf("refused denominator=%d summary=%d, want 1/1", denominator.RefusedRecords, result.Summary.RefusedRecords)
			}
			if denominator.UsageRecordsSeen != test.seen || denominator.UsageRecordsExact != test.exact || denominator.UsageRecordsApplied != test.applied {
				t.Fatalf("usage denominator = %+v", denominator)
			}
			if len(result.Transcripts) != 1 || result.Transcripts[0].Tokens != test.wantTokens {
				t.Fatalf("tokens = %+v, want %+v", result.Transcripts, test.wantTokens)
			}
		})
	}
}

func TestAuditDeterministicByteOrder(t *testing.T) {
	root := filepath.Join("testdata", "audit", "issue-8493", "deterministic")
	a := AuditSource{Name: AuditSourceClaude, Root: filepath.Join(root, "claude-a", "projects"), RootLabel: "claude/projects/a"}
	b := AuditSource{Name: AuditSourceClaude, Root: filepath.Join(root, "claude-b", "projects"), RootLabel: "claude/projects/b"}
	run := func(sources []AuditSource) []byte {
		t.Helper()
		result, err := RunAudit(AuditOptions{Sources: sources})
		if err != nil {
			t.Fatal(err)
		}
		var output bytes.Buffer
		if err := WriteAuditJSONL(&output, result); err != nil {
			t.Fatal(err)
		}
		return output.Bytes()
	}
	forward := run([]AuditSource{a, b})
	reverse := run([]AuditSource{b, a})
	if !bytes.Equal(forward, reverse) {
		t.Fatalf("JSONL byte order depends on source input order\nforward:\n%s\nreverse:\n%s", forward, reverse)
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

func TestQwenTopContributorTokenConcentration(t *testing.T) {
	row := func(id string, tokens int64) AuditTranscriptRow {
		return AuditTranscriptRow{Source: AuditSourceCodex, TranscriptID: id, Models: []string{"Qwen3"}, Tokens: AuditTokens{InputTokens: tokens}}
	}
	balanced := 0.5
	concentrated := 0.8
	for _, tc := range []struct {
		name         string
		rows         []AuditTranscriptRow
		want         *float64
		concentrated bool
	}{
		{name: "balanced", rows: []AuditTranscriptRow{row("a", 50), row("b", 50)}, want: &balanced},
		{name: "concentrated", rows: []AuditTranscriptRow{row("a", 80), row("b", 20)}, want: &concentrated, concentrated: true},
		{name: "missing"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			summary := summarizeAudit(nil, tc.rows, nil)
			if summary.QwenTokenConcentrationThreshold != 0.5 {
				t.Fatalf("threshold = %v", summary.QwenTokenConcentrationThreshold)
			}
			if tc.want == nil {
				if summary.QwenTopContributor != nil || summary.QwenTopContributorTokenFraction != nil || summary.QwenTopContributorTokenConcentrated != nil {
					t.Fatalf("missing Qwen usage must remain unknown: %+v", summary)
				}
				return
			}
			if summary.QwenTopContributor == nil || *summary.QwenTopContributor != "codex:a" {
				t.Fatalf("top contributor = %v", summary.QwenTopContributor)
			}
			if summary.QwenTopContributorTokenFraction == nil || *summary.QwenTopContributorTokenFraction != *tc.want {
				t.Fatalf("fraction = %v, want %v", summary.QwenTopContributorTokenFraction, *tc.want)
			}
			if summary.QwenTopContributorTokenConcentrated == nil || *summary.QwenTopContributorTokenConcentrated != tc.concentrated {
				t.Fatalf("concentrated = %v, want %t", summary.QwenTopContributorTokenConcentrated, tc.concentrated)
			}
		})
	}
}

func TestQwenTopContributorConcentrationJSONAndRendering(t *testing.T) {
	balanced := summarizeAudit(nil, []AuditTranscriptRow{
		{Source: AuditSourceCodex, TranscriptID: "b", Models: []string{"qwen3"}, Tokens: AuditTokens{InputTokens: 50}},
		{Source: AuditSourceCodex, TranscriptID: "a", Models: []string{"qwen3"}, Tokens: AuditTokens{InputTokens: 50}},
	}, nil)
	encoded, err := json.Marshal(balanced)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"qwen_top_contributor":"codex:a"`, `"qwen_top_contributor_token_fraction":0.5`, `"qwen_token_concentration_threshold":0.5`, `"qwen_top_contributor_token_concentrated":false`} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Fatalf("JSON %s missing %s", encoded, want)
		}
	}
	var rendered bytes.Buffer
	if err := WriteAuditMarkdown(&rendered, AuditResult{Summary: balanced}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rendered.String(), "Qwen top contributor: `codex:a` with 50/100 tokens (50.00%); concentrated: false (threshold: 50%).") {
		t.Fatalf("rendered output:\n%s", rendered.String())
	}

	missing := summarizeAudit(nil, nil, nil)
	encoded, err = json.Marshal(missing)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"qwen_top_contributor":null`, `"qwen_top_contributor_token_fraction":null`, `"qwen_top_contributor_token_concentrated":null`} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Fatalf("missing JSON %s lacks %s", encoded, want)
		}
	}
	rendered.Reset()
	if err := WriteAuditMarkdown(&rendered, AuditResult{Summary: missing}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rendered.String(), "Qwen top-contributor token concentration: unknown") {
		t.Fatalf("missing render:\n%s", rendered.String())
	}
}
