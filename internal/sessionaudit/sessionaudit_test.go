package sessionaudit

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func assistantRecord(id string, out, cread, ccreate int64, opts ...func(map[string]any)) map[string]any {
	msg := map[string]any{
		"id":    id,
		"model": "claude-opus-4-8",
		"usage": map[string]any{
			"input_tokens":                int64(0),
			"output_tokens":               out,
			"cache_read_input_tokens":     cread,
			"cache_creation_input_tokens": ccreate,
		},
		"content": []any{},
	}
	rec := map[string]any{
		"type":      "assistant",
		"timestamp": "2026-06-20T00:00:00.000Z",
		"uuid":      "uuid-" + id,
		"message":   msg,
	}
	for _, opt := range opts {
		opt(rec)
	}
	return rec
}

func withTool(name string) func(map[string]any) {
	return func(rec map[string]any) {
		msg := rec["message"].(map[string]any)
		msg["content"] = []any{map[string]any{"type": "tool_use", "name": name, "input": map[string]any{}}}
	}
}

func withModel(model string) func(map[string]any) {
	return func(rec map[string]any) {
		rec["message"].(map[string]any)["model"] = model
	}
}

func withServerWeb(searches, fetches int64) func(map[string]any) {
	return func(rec map[string]any) {
		usage := rec["message"].(map[string]any)["usage"].(map[string]any)
		usage["server_tool_use"] = map[string]any{"web_search_requests": searches, "web_fetch_requests": fetches}
	}
}

func withoutID() func(map[string]any) {
	return func(rec map[string]any) {
		delete(rec["message"].(map[string]any), "id")
	}
}

func writeTranscript(t *testing.T, records []map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "session-a.jsonl")
	writeJSONL(t, path, records)
	return path
}

func writeTranscriptIn(t *testing.T, root, ns, rel string, records []map[string]any) string {
	t.Helper()
	path := filepath.Join(root, ns, filepath.FromSlash(rel))
	writeJSONL(t, path, records)
	return path
}

func writeJSONL(t *testing.T, path string, records []map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	_ = os.Chtimes(path, now, now)
}

func TestDuplicateBilledTurnCountedOnce(t *testing.T) {
	recs := []map[string]any{}
	for i := 0; i < 4; i++ {
		recs = append(recs, assistantRecord("msg-A", 400, 50_000, 6_000))
	}
	for i := 0; i < 2; i++ {
		recs = append(recs, assistantRecord("msg-B", 500, 60_000, 7_000, withTool("Bash")))
	}
	s := Analyze(writeTranscript(t, recs))
	if s.Error != "" {
		t.Fatal(s.Error)
	}
	if s.AssistantTurns != 2 {
		t.Fatalf("assistant turns = %d, want 2", s.AssistantTurns)
	}
	if s.DupAssistantLines != 4 {
		t.Fatalf("duplicate assistant lines = %d, want 4", s.DupAssistantLines)
	}
	if s.Tokens.Output != 900 || s.Tokens.CacheRead != 110_000 || s.Tokens.CacheCreate != 13_000 {
		t.Fatalf("wrong tokens: %+v", s.Tokens)
	}
	if s.NToolUse != 1 || s.Tools["Bash"] != 1 {
		t.Fatalf("duplicated tool_use was counted: n=%d tools=%v", s.NToolUse, s.Tools)
	}
}

func TestIDlessLinesEachCount(t *testing.T) {
	recs := []map[string]any{
		assistantRecord("x", 50, 5_000, 500, withoutID()),
		assistantRecord("x", 50, 5_000, 500, withoutID()),
	}
	s := Analyze(writeTranscript(t, recs))
	if s.AssistantTurns != 2 {
		t.Fatalf("assistant turns = %d, want 2", s.AssistantTurns)
	}
	if s.Tokens.Output != 100 {
		t.Fatalf("output = %d, want 100", s.Tokens.Output)
	}
}

func TestCostIsPerDedupedTurn(t *testing.T) {
	recs := []map[string]any{
		assistantRecord("msg-only", 1_000, 0, 0),
		assistantRecord("msg-only", 1_000, 0, 0),
		assistantRecord("msg-only", 1_000, 0, 0),
	}
	s := Analyze(writeTranscript(t, recs))
	want := 1_000 * 75.0 / 1e6
	if math.Abs(s.CostUSD-want) > 1e-9 {
		t.Fatalf("cost = %.12f, want %.12f", s.CostUSD, want)
	}
}

func TestWebActivityRendering(t *testing.T) {
	client := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("msg-1", 100, 1_000, 100, withTool("WebFetch")),
	}))
	if client.Tools["WebFetch"] != 1 {
		t.Fatalf("WebFetch not counted: %v", client.Tools)
	}
	if client.Tokens.WebFetch != 0 {
		t.Fatalf("server fetch = %d, want 0", client.Tokens.WebFetch)
	}
	if client.ReadOnlyFrac == nil || *client.ReadOnlyFrac != 1.0 {
		t.Fatalf("read-only frac = %v, want 1.0", client.ReadOnlyFrac)
	}
	md := ReportMarkdown([]Session{client}, AggregateSessions([]Session{client}), "", nil, false, 0, 1, nil, time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC))
	if !strings.Contains(md, "WebFetch 1") {
		t.Fatalf("client WebFetch hidden from report:\n%s", md)
	}

	server := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("msg-2", 100, 1_000, 100, withServerWeb(3, 2)),
	}))
	md = ReportMarkdown([]Session{server}, AggregateSessions([]Session{server}), "", nil, false, 0, 1, nil, time.Now())
	if !strings.Contains(md, "search 3 / fetch 2") {
		t.Fatalf("server web counts hidden:\n%s", md)
	}
}

func TestReportMarkdownHighlightsOpusHeavySessions(t *testing.T) {
	root := t.TempDir()
	opusHeavy := Analyze(writeTranscriptIn(t, root, "C--work-fak", "opus-heavy.jsonl", []map[string]any{
		assistantRecord("opus-1", 900, 10_000, 1_000),
		assistantRecord("fable-1", 100, 10_000, 1_000, withModel("claude-fable-5")),
	}))
	mixed := Analyze(writeTranscriptIn(t, root, "C--work-fak", "mixed.jsonl", []map[string]any{
		assistantRecord("opus-2", 300, 10_000, 1_000),
		assistantRecord("fable-2", 700, 10_000, 1_000, withModel("claude-fable-5")),
	}))
	fableOnly := Analyze(writeTranscriptIn(t, root, "C--work-fak", "fable-only.jsonl", []map[string]any{
		assistantRecord("fable-3", 1200, 10_000, 1_000, withModel("claude-fable-5")),
	}))

	md := ReportMarkdown([]Session{mixed, fableOnly, opusHeavy}, AggregateSessions([]Session{mixed, fableOnly, opusHeavy}), "C--work-fak", nil, false, 0, 3, nil, time.Now())
	if !strings.Contains(md, "## Opus-heavy sessions") {
		t.Fatalf("report missed Opus-heavy section:\n%s", md)
	}
	section := md[strings.Index(md, "## Opus-heavy sessions"):]
	if end := strings.Index(section, "\n## Long-context sessions"); end >= 0 {
		section = section[:end]
	}
	if !strings.Contains(section, "| opus-hea | C--work-fak | 900 | 90.0% | $0.10 | 1,000 | $0.11 | claude-opus-4-8 |") {
		t.Fatalf("report missed sorted Opus-heavy row:\n%s", md)
	}
	if !strings.Contains(section, "| mixed | C--work-fak | 300 | 30.0% | $0.06 | 1,000 | $0.07 | claude-fable-5 |") {
		t.Fatalf("report missed mixed Opus row:\n%s", md)
	}
	if strings.Contains(section, "fable-on | C--work-fak |") {
		t.Fatalf("fable-only session should not appear in Opus-heavy section:\n%s", md)
	}
}

func TestReportMarkdownHighlightsLongContextSessions(t *testing.T) {
	root := t.TempDir()
	heavy := Analyze(writeTranscriptIn(t, root, "C--work-fak", "heavyctx.jsonl", []map[string]any{
		assistantRecord("heavy-1", 100, 900_000, 50_000),
		assistantRecord("heavy-2", 100, 100_000, 50_000),
	}))
	light := Analyze(writeTranscriptIn(t, root, "C--work-fak", "lightctx.jsonl", []map[string]any{
		assistantRecord("light-1", 100, 1_000, 100, withModel("claude-fable-5")),
	}))

	md := ReportMarkdown([]Session{light, heavy}, AggregateSessions([]Session{light, heavy}), "C--work-fak", nil, false, 0, 2, nil, time.Now())
	if !strings.Contains(md, "## Long-context sessions") {
		t.Fatalf("report missed long-context section:\n%s", md)
	}
	section := md[strings.Index(md, "## Long-context sessions"):]
	if end := strings.Index(section, "\n## Distributions"); end >= 0 {
		section = section[:end]
	}
	if !strings.Contains(section, "| heavyctx | C--work-fak | 1,100,000 | 0 | 1,000,000 | 90.9% | 200 | 5500.0 | claude-opus-4-8 |") {
		t.Fatalf("report missed long-context heavy row:\n%s", md)
	}
	if !strings.Contains(section, "| lightctx | C--work-fak | 1,100 | 0 | 1,000 | 90.9% | 100 | 11.0 | claude-fable-5 |") {
		t.Fatalf("report missed long-context light row:\n%s", md)
	}
	if strings.Index(section, "1,100,000") > strings.Index(section, "1,100 |") {
		t.Fatalf("long-context section is not sorted by total context descending:\n%s", section)
	}
}

func TestReadOnlyClassification(t *testing.T) {
	for _, name := range []string{"Monitor", "TaskGet", "TaskList", "TaskOutput", "ReadMcpResourceTool"} {
		if !ReadOnlyTools[name] {
			t.Fatalf("%s should be read-only", name)
		}
	}
	for _, name := range []string{"TaskCreate", "TaskUpdate", "TaskStop"} {
		if ReadOnlyTools[name] {
			t.Fatalf("%s should not be read-only", name)
		}
	}
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("m1", 10, 100, 10, withTool("Monitor")),
		assistantRecord("m2", 10, 100, 10, withTool("Bash")),
	}))
	if s.ReadOnlyFrac == nil || *s.ReadOnlyFrac != 0.5 {
		t.Fatalf("read-only frac = %v, want 0.5", s.ReadOnlyFrac)
	}
}

func TestDefaultDiscoversAllNonExcludedNamespaces(t *testing.T) {
	if NamespaceIncludePrefix != "" {
		t.Fatalf("default namespace filter = %q, want empty", NamespaceIncludePrefix)
	}
	root := t.TempDir()
	for _, ns := range []string{"-Users-USER-Documents-GitHub-fleet", "C--work-fak", "AppData-Local-Temp-fixture"} {
		writeTranscriptIn(t, root, ns, ns+".jsonl", []map[string]any{{}})
	}
	found, err := Discover(DiscoverOptions{Roots: []string{root}})
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, r := range found {
		names[r.NS] = true
	}
	if !names["-Users-USER-Documents-GitHub-fleet"] || !names["C--work-fak"] {
		t.Fatalf("expected namespaces not discovered: %v", names)
	}
	if names["AppData-Local-Temp-fixture"] {
		t.Fatalf("excluded namespace discovered: %v", names)
	}
	narrowed, err := Discover(DiscoverOptions{Roots: []string{root}, NamespacePrefix: "C--work"})
	if err != nil {
		t.Fatal(err)
	}
	if len(narrowed) != 1 || narrowed[0].NS != "C--work-fak" {
		t.Fatalf("narrowed = %+v", narrowed)
	}
}

func TestProjectNamespaceMatchesClaudeProjectsKey(t *testing.T) {
	for _, tc := range []struct {
		path string
		want string
	}{
		{`C:\work\fak`, "C--work-fak"},
		{`C:\work\fak repo`, "C--work-fak-repo"},
		{`/home/u/p`, "-home-u-p"},
	} {
		if got := ProjectNamespace(tc.path); got != tc.want {
			t.Fatalf("ProjectNamespace(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestScopeHeaderSubagentWarningAndModelMix(t *testing.T) {
	root := t.TempDir()
	topPath := writeTranscriptIn(t, root, "C--work-fak", "session-a.jsonl", []map[string]any{
		assistantRecord("opus", 850, 1_000, 100, withModel("claude-opus-4-8")),
		assistantRecord("haiku", 150, 0, 0, withModel("claude-haiku-4-5")),
	})
	subPath := writeTranscriptIn(t, root, "C--work-fak", "session-a/subagents/worker.jsonl", []map[string]any{
		assistantRecord("sub", 2_000, 3_000, 400),
	})
	top := Analyze(topPath)
	sub := Analyze(subPath)
	sum := SummarizeAnalyses([]Session{sub})
	agg := AggregateSessions([]Session{top})
	md := ReportMarkdown([]Session{top}, agg, "C--work-fak", nil, false, 0, 1, &sum, time.Now())
	for _, want := range []string{
		"# Session-Transcript Audit - active scope",
		"1 namespaces folded (C--work-fak)",
		"namespace filter: C--work-fak",
		"time window: all-time",
		"## Scope totals (EXACT token counts)",
		"scope I:O ratio",
		"NOTE: +1 subagent transcripts uncounted",
		"re-run with `--include-subagents`",
		"+2,000 output tok",
		"## Model-mix KPI (tier shares)",
		"| opus | 850 | 85.0% |",
		"| haiku | 150 | 15.0% |",
		"Opus output share",
		"| C--work-fak | 1 | 1,000 | 85.0% |",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("report missing %q:\n%s", want, md)
		}
	}
	if strings.Contains(md, "recent sessions, this machine") || strings.Contains(md, "Machine-wide totals") || strings.Contains(md, "machine-wide I:O") {
		t.Fatalf("report contains stale scope language:\n%s", md)
	}
}

func TestProviderBucketAndCostBehavior(t *testing.T) {
	if _, ok := PriceFor("gemini-2.5-pro"); ok {
		t.Fatal("Gemini should not get a Claude rate card")
	}
	if _, ok := PriceFor("gpt-5"); ok {
		t.Fatal("OpenAI should not get a Claude rate card")
	}
	if _, ok := PriceFor("qwen2.5:14b"); ok {
		t.Fatal("local model should not get a Claude rate card")
	}
	if _, ok := PriceFor("<synthetic>"); ok {
		t.Fatal("synthetic should be non-billed")
	}
	if _, ok := PriceFor("claude-opus-4-8"); !ok {
		t.Fatal("opus should resolve")
	}
	if CostUSD("gemini-2.5-pro", 0, 0, 0, 1_000_000) != 0 {
		t.Fatal("unpriced Gemini should cost 0")
	}
	if got := CostUSD("claude-opus-4-8", 0, 0, 0, 1_000_000); got != 75.0 {
		t.Fatalf("opus cost = %.2f, want 75", got)
	}
	wantBuckets := map[string]string{
		"claude-opus-4-8":   "Anthropic (Claude)",
		"gemini-2.5-pro":    "Google (Gemini)",
		"gpt-5":             "OpenAI",
		"qwen2.5:14b":       "local / self-hosted",
		"<synthetic>":       "non-billed (harness)",
		"some-future-model": "UNKNOWN (unpriced bucket)",
	}
	for model, want := range wantBuckets {
		if got := ProviderBucket(model); got != want {
			t.Fatalf("ProviderBucket(%q) = %q, want %q", model, got, want)
		}
	}
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("c1", 1_000, 0, 0, withModel("claude-opus-4-8")),
		assistantRecord("g1", 2_000, 0, 0, withModel("gemini-2.5-pro")),
		assistantRecord("syn", 0, 0, 0, withModel("<synthetic>")),
	}))
	agg := AggregateSessions([]Session{s})
	want := 1_000 * 75.0 / 1e6
	if math.Abs(agg.TotalCostUSD-want) > 1e-9 {
		t.Fatalf("total cost = %.12f, want %.12f", agg.TotalCostUSD, want)
	}
	if agg.PerBucket["Google (Gemini)"].Output != 2_000 {
		t.Fatalf("Gemini bucket = %+v", agg.PerBucket["Google (Gemini)"])
	}
	if _, ok := s.PerModel["<synthetic>"]; !ok {
		t.Fatal("synthetic turn missing from per-model")
	}
	if ModelCost("<synthetic>", agg.PerModel["<synthetic>"]) != 0 {
		t.Fatal("synthetic should cost 0")
	}
	md := ReportMarkdown([]Session{s}, agg, "", nil, false, 0, 1, nil, time.Now())
	for _, want := range []string{"Cost by billing bucket", "Google (Gemini)", "- (no card)", "Other billing buckets present"} {
		if !strings.Contains(md, want) {
			t.Fatalf("report missing %q:\n%s", want, md)
		}
	}
}
