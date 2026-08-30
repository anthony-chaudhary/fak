package trajectory

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAttributionNightlyPopulatedReceiptAndTrend(t *testing.T) {
	budget := attributionTestBudget()
	now := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	sources, root := attributionNightlyFixtureSources(t, now.Add(-time.Minute))
	receipt := RunAttributionNightly(AttributionNightlyOptions{Sources: sources, Budget: budget, Now: now, Corpus: "fleet"})
	if receipt.Status != AttributionStatusPass {
		t.Fatalf("status=%s breaches=%+v error=%s", receipt.Status, receipt.Breaches, receipt.CollectionError)
	}
	if len(receipt.Coverage) != 2 || receipt.Coverage[0].FilesScanned+receipt.Coverage[1].FilesScanned != 2 {
		t.Fatalf("coverage=%+v", receipt.Coverage)
	}
	if receipt.Metrics.DuplicateEvents != 1 || receipt.Metrics.MalformedRows != 0 || receipt.Metrics.UnmatchedToolEvents != 0 || receipt.Metrics.SchemaDriftSignals != 0 {
		t.Fatalf("metrics=%+v", receipt.Metrics)
	}
	var encoded bytes.Buffer
	if err := WriteAttributionReceipt(&encoded, receipt); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"audit the fixture", "exit code 1", filepath.ToSlash(root), filepath.ToSlash(mustAbs(t, root))} {
		if strings.Contains(encoded.String(), forbidden) {
			t.Fatalf("receipt leaked transcript content or root %q:\n%s", forbidden, encoded.String())
		}
	}
	history := filepath.Join(t.TempDir(), "history.jsonl")
	if err := AppendAttributionReceipt(history, &receipt); err != nil {
		t.Fatal(err)
	}
	second := RunAttributionNightly(AttributionNightlyOptions{Sources: sources, Budget: budget, Now: now.Add(time.Minute), Corpus: "fleet"})
	if err := AppendAttributionReceipt(history, &second); err != nil {
		t.Fatal(err)
	}
	if !second.Trend.Comparable || second.Trend.Deltas["duplicate_events"] != 0 || second.Trend.PreviousObservedAt != receipt.ObservedAt {
		t.Fatalf("trend=%+v", second.Trend)
	}
	rows, err := os.ReadFile(history)
	if err != nil {
		t.Fatal(err)
	}
	if lines := bytes.Count(bytes.TrimSpace(rows), []byte{'\n'}) + 1; lines != 2 {
		t.Fatalf("history rows=%d, want 2:\n%s", lines, rows)
	}
}

func attributionNightlyFixtureSources(t *testing.T, modTime time.Time) ([]AuditSource, string) {
	t.Helper()
	sourceRoot := filepath.Join("testdata", "audit")
	root := t.TempDir()
	fixtures := []struct {
		source string
		target string
	}{
		{source: filepath.Join(sourceRoot, "claude", "projects", "fak", "claude-session.jsonl"), target: filepath.Join(root, "claude", "projects", "fak", "claude-session.jsonl")},
		{source: filepath.Join(sourceRoot, "codex", "sessions", "2026", "08", "21", "codex-session.jsonl"), target: filepath.Join(root, "codex", "sessions", "2026", "08", "21", "codex-session.jsonl")},
	}
	for _, fixture := range fixtures {
		contents, err := os.ReadFile(fixture.source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(fixture.target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fixture.target, contents, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(fixture.target, modTime, modTime); err != nil {
			t.Fatal(err)
		}
	}
	return []AuditSource{
		{Name: AuditSourceClaude, Root: filepath.Join(root, "claude", "projects"), RootLabel: "claude/projects"},
		{Name: AuditSourceCodex, Root: filepath.Join(root, "codex", "sessions"), RootLabel: "codex/sessions"},
	}, root
}

func TestAttributionNightlyNoDataAndCollectionFailureAreDistinct(t *testing.T) {
	budget := attributionTestBudget()
	missing := filepath.Join(t.TempDir(), "missing")
	noData := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: missing, RootLabel: "claude/projects"}},
		Budget:  budget, Now: time.Now(), Corpus: "local",
	})
	if noData.Status != AttributionStatusNoData || len(noData.Coverage) != 1 || noData.Coverage[0].RootPresent {
		t.Fatalf("no-data receipt=%+v", noData)
	}
	file := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	failed := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: file, RootLabel: "claude/projects"}},
		Budget:  budget, Now: time.Now(), Corpus: "local",
	})
	if failed.Status != AttributionStatusCollectionFailed || failed.CollectionError != "source_collection_failed" {
		t.Fatalf("collection-failure receipt=%+v", failed)
	}
}

func TestAttributionNightlyBoundedNewestFilesAndCoordinates(t *testing.T) {
	root := t.TempDir()
	sensitive := filepath.Join(root, "Customer-Acme", "anthony-laptop", "secret-project")
	if err := os.MkdirAll(sensitive, 0o700); err != nil {
		t.Fatal(err)
	}
	older := filepath.Join(sensitive, "older.jsonl")
	newer := filepath.Join(root, "newer.jsonl")
	row := []byte(`{"type":"user","sessionId":"s","message":{"content":"private"}}` + "\n")
	for _, path := range []string{older, newer} {
		if err := os.WriteFile(path, row, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	if err := os.Chtimes(older, now.Add(-time.Hour), now.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(newer, now, now); err != nil {
		t.Fatal(err)
	}
	budget := attributionTestBudget()
	budget.MaxFilesPerSource = 1
	receipt := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}},
		Budget:  budget, Now: now.Add(time.Second), Corpus: "local",
	})
	if receipt.Status != AttributionStatusBudgetFailed || receipt.Metrics.BoundedFilesOmitted != 1 || len(receipt.Coverage) != 1 || receipt.Coverage[0].FilesScanned != 1 {
		t.Fatalf("bounded receipt=%+v", receipt)
	}
	if len(receipt.Samples) == 0 || !strings.HasPrefix(receipt.Samples[0].SourcePath, "sha256:") {
		t.Fatalf("bounded samples=%+v", receipt.Samples)
	}
	for _, private := range []string{"Customer-Acme", "anthony-laptop", "secret-project", "older.jsonl", "/", `\`} {
		if strings.Contains(receipt.Samples[0].SourcePath, private) {
			t.Fatalf("sample path leaked %q: %+v", private, receipt.Samples[0])
		}
	}
}

func TestAttributionNightlyTrendRejectsMismatchedEnvelopes(t *testing.T) {
	newPair := func() (AttributionReceipt, AttributionReceipt) {
		coverage := []AttributionSourceCoverage{{
			Source: AuditSourceClaude, Root: "claude/projects", RootPresent: true,
			FilesDiscovered: 2, FilesScanned: 2, Records: 10,
		}}
		current := AttributionReceipt{
			Schema: AttributionReceiptSchema, BudgetSchema: AttributionBudgetSchema,
			BudgetVersion: "test/1", Window: "24h", Status: AttributionStatusPass,
			Coverage: append([]AttributionSourceCoverage(nil), coverage...),
		}
		previous := current
		previous.Coverage = append([]AttributionSourceCoverage(nil), coverage...)
		return current, previous
	}
	tests := []struct {
		name   string
		reason string
		mutate func(*AttributionReceipt)
	}{
		{"budget schema", "budget_schema_changed", func(r *AttributionReceipt) { r.BudgetSchema = "budget/2" }},
		{"budget version", "budget_version_changed", func(r *AttributionReceipt) { r.BudgetVersion = "test/2" }},
		{"window", "window_changed", func(r *AttributionReceipt) { r.Window = "48h" }},
		{"source coverage", "source_coverage_changed", func(r *AttributionReceipt) { r.Coverage[0].Root = "other/root" }},
		{"source root", "source_coverage_changed", func(r *AttributionReceipt) { r.Coverage[0].RootFingerprint = "sha256:other" }},
		{"file exposure", "exposure_changed", func(r *AttributionReceipt) { r.Coverage[0].FilesScanned++ }},
		{"record exposure", "exposure_changed", func(r *AttributionReceipt) { r.Coverage[0].Records++ }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			current, previous := newPair()
			test.mutate(&previous)
			trend := compareAttributionReceipts(&current, previous, true)
			if trend.Comparable || trend.Reason != test.reason {
				t.Fatalf("trend=%+v, want non-comparable %q", trend, test.reason)
			}
		})
	}
}

func TestAttributionNightlyBoundsPhysicalIOAndDetectsGrowth(t *testing.T) {
	const readLimit = 7
	var observed bytes.Buffer
	contents, err := readAttributionBounded(io.TeeReader(strings.NewReader(strings.Repeat("x", 100)), &observed), readLimit)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != readLimit || observed.Len() != len(contents) {
		t.Fatalf("read=%d observed=%d, want exact %d-byte ceiling", len(contents), observed.Len(), readLimit)
	}
	path := filepath.Join(t.TempDir(), "growing.jsonl")
	if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	candidate := attributionAuditFile{path: path, rel: "growing.jsonl", size: info.Size(), modTime: info.ModTime()}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("{}\n"); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := readAttributionCandidate(candidate, 1<<20); !errors.Is(err, errAttributionSourceChanged) {
		t.Fatalf("growth error=%v, want %v", err, errAttributionSourceChanged)
	}
}

func TestAttributionNightlyStopsAtWalkCeiling(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a.jsonl", "b.jsonl", "c.jsonl"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("{}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	budget := attributionTestBudget()
	budget.MaxWalkEntriesPerSource = 2
	receipt := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}},
		Budget:  budget, Now: time.Now(), Corpus: "local",
	})
	if receipt.Status != AttributionStatusCollectionFailed || receipt.CollectionError != "source_discovery_limit" {
		t.Fatalf("receipt=%+v", receipt)
	}
}

func TestAttributionNightlyFixtureRecognitionStaysInsideReadBudget(t *testing.T) {
	root := t.TempDir()
	project := "C--Users-x-AppData-Local-Temp-pytest-of-x-pytest-1-test_main_json_runs_and_emits_0-ws"
	dir := filepath.Join(root, project)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "00000000-aaaa-bbbb-cccc-000000000000.jsonl")
	if err := os.WriteFile(path, []byte(strings.Repeat("x", 128)), 0o600); err != nil {
		t.Fatal(err)
	}
	budget := attributionTestBudget()
	budget.MaxBytesPerFile = 16
	budget.MaxBytesPerSource = 32
	receipt := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "ignored"}},
		Budget:  budget, Now: time.Now(), Corpus: "local",
	})
	if receipt.Metrics.BoundedFilesOmitted != 1 || receipt.Coverage[0].FixtureFilesExcluded != 0 || receipt.Coverage[0].FilesScanned != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
}

func TestAttributionNightlyBoundsAndDedupesSubtypeCoverage(t *testing.T) {
	root := t.TempDir()
	var rows strings.Builder
	for _, subtype := range []string{"private_a", "private_b", "private_c", "private_d", "private_a"} {
		fmt.Fprintf(&rows, "{\"type\":%q,\"sessionId\":\"s\",\"message\":{\"content\":\"private\"}}\n", subtype)
	}
	if err := os.WriteFile(filepath.Join(root, "session.jsonl"), []byte(rows.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	budget := attributionTestBudget()
	budget.MaxSubtypesPerSource = 2
	receipt := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "ignored"}},
		Budget:  budget, Now: time.Now(), Corpus: "local",
	})
	coverage := receipt.Coverage[0]
	if coverage.SubtypesObserved != 4 || coverage.SubtypesOmitted != 2 || len(coverage.RecordTypes) != 2 || receipt.Metrics.BoundedSubtypesOmitted != 2 {
		t.Fatalf("coverage=%+v metrics=%+v", coverage, receipt.Metrics)
	}
	foundBreach := false
	for _, breach := range receipt.Breaches {
		foundBreach = foundBreach || breach.Metric == "bounded_subtypes_omitted"
	}
	if !foundBreach {
		t.Fatalf("breaches=%+v", receipt.Breaches)
	}
	var encoded bytes.Buffer
	if err := WriteAttributionReceipt(&encoded, receipt); err != nil {
		t.Fatal(err)
	}
	if encoded.Len() >= 4*1024*1024 {
		t.Fatalf("bounded receipt exceeds history reader: %d bytes", encoded.Len())
	}
}

func TestAttributionNightlyHashesUntrustedSafeSubtype(t *testing.T) {
	root := t.TempDir()
	rawSubtype := "customer_secret_project"
	row := []byte(`{"type":"` + rawSubtype + `","sessionId":"s","message":{"content":"private"}}` + "\n")
	if err := os.WriteFile(filepath.Join(root, "session.jsonl"), row, 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := RunAttributionNightly(AttributionNightlyOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}},
		Budget:  attributionTestBudget(), Now: time.Now(), Corpus: "local",
	})
	var encoded bytes.Buffer
	if err := WriteAttributionReceipt(&encoded, receipt); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(encoded.String(), rawSubtype) {
		t.Fatalf("receipt leaked untrusted safe subtype: %s", encoded.String())
	}
	if len(receipt.Coverage) != 1 || len(receipt.Coverage[0].RecordTypes) != 1 || !strings.HasPrefix(receipt.Coverage[0].RecordTypes[0].Subtype, "sha256:") {
		t.Fatalf("coverage=%+v", receipt.Coverage)
	}
}

func attributionTestBudget() AttributionBudget {
	return AttributionBudget{
		Schema: AttributionBudgetSchema, Version: "test/1", Window: "24h",
		MaxFilesPerSource: 20, MaxWalkEntriesPerSource: 1000, MaxSubtypesPerSource: 100,
		MaxBytesPerFile: 1 << 20, MaxBytesPerSource: 10 << 20,
		MaxSampleCoordinates: 10, MaxUnknownShare: 1, MaxDuplicateEvents: 10,
		MaxMalformedRows: 0, MaxUnmatchedToolEvents: 0, MaxSchemaDriftSignals: 0,
		MaxBoundedFilesOmitted: 0, MaxBoundedSubtypesOmitted: 0,
		KnownSourceRecordTypes: map[string][]string{
			AuditSourceClaude: {"assistant", "attachment", "user"},
			AuditSourceCodex: {
				"event_msg:token_count", "response_item:custom_tool_call", "response_item:custom_tool_call_output",
				"response_item:function_call", "response_item:function_call_output", "session_meta", "turn_context",
			},
		},
	}
}

func mustAbs(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}
