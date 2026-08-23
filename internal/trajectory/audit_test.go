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
				row, refusals, hooks, err := parseAuditFile(test.source.Name, path, "fak/oversized.jsonl", &denominator)
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
