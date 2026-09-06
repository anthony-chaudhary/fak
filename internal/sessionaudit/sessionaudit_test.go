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

func TestBuildCompactReportSummarizesModelMixAndLongContext(t *testing.T) {
	root := t.TempDir()
	heavy := Analyze(writeTranscriptIn(t, root, "C--work-fak", "heavyctx.jsonl", []map[string]any{
		assistantRecord("heavy-1", 100, 900_000, 50_000),
		assistantRecord("heavy-2", 100, 100_000, 50_000),
	}))
	fable := Analyze(writeTranscriptIn(t, root, "C--work-fak", "fablectx.jsonl", []map[string]any{
		assistantRecord("fable-1", 300, 20_000, 1_000, withModel("claude-fable-5")),
	}))
	agg := AggregateSessions([]Session{heavy, fable})
	rep := BuildCompactReport([]Session{heavy, fable}, agg, "C--work-fak", nil, false, 2, 3, nil, time.Date(2026, 7, 2, 0, 0, 0, 0, time.UTC))
	if rep.Schema != "fak.session_audit.summary.v1" || !rep.Scope.Scoped || !rep.Scope.Clipped {
		t.Fatalf("compact scope = %+v schema=%q", rep.Scope, rep.Schema)
	}
	if rep.Totals.TotalContextTokens != 1_121_000 || rep.Totals.OutputTokens != 500 {
		t.Fatalf("compact totals = %+v", rep.Totals)
	}
	tiers := map[string]CompactTier{}
	for _, tier := range rep.Tiers {
		tiers[tier.Tier] = tier
	}
	if tiers["fable"].OutputTokens != 300 || tiers["opus"].OutputTokens != 200 {
		t.Fatalf("compact tiers = %+v", rep.Tiers)
	}
	if len(rep.TopLongContext) == 0 || rep.TopLongContext[0].Session != "heavyctx" ||
		rep.TopLongContext[0].TotalContextTokens != 1_100_000 {
		t.Fatalf("compact long-context rows = %+v", rep.TopLongContext)
	}
}

func TestBuildCompactReportRecommendsForOpusCostAndLongContextPressure(t *testing.T) {
	root := t.TempDir()
	opus := Analyze(writeTranscriptIn(t, root, "C--work-fak", "opusctx.jsonl", []map[string]any{
		assistantRecord("opus-1", 1_000, 30_000_000, 0, withModel("claude-opus-4-8")),
	}))
	fable := Analyze(writeTranscriptIn(t, root, "C--work-fak", "fablectx.jsonl", []map[string]any{
		assistantRecord("fable-1", 5_000, 1_000, 0, withModel("claude-fable-5")),
	}))
	rep := BuildCompactReport([]Session{opus, fable}, AggregateSessions([]Session{opus, fable}), "C--work-fak", nil, false, 0, 2, nil, time.Now())
	byKind := map[string]CompactRecommendation{}
	for _, rec := range rep.Recommendations {
		byKind[rec.Kind] = rec
	}
	if byKind["opus_cost_pressure"].Severity != "high" ||
		!strings.Contains(byKind["opus_cost_pressure"].Evidence, "opus_cost_share=") ||
		!strings.Contains(byKind["opus_cost_pressure"].Action, "Fable") {
		t.Fatalf("opus cost recommendation = %+v", byKind["opus_cost_pressure"])
	}
	if byKind["long_context_pressure"].Severity != "high" ||
		!strings.Contains(byKind["long_context_pressure"].Evidence, "opusctx") ||
		!strings.Contains(byKind["long_context_pressure"].Action, "ctxvalue/vcache") {
		t.Fatalf("long-context recommendation = %+v", byKind["long_context_pressure"])
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
	explicitExcluded, err := Discover(DiscoverOptions{Roots: []string{root}, NamespacePrefix: "AppData-Local-Temp-fixture"})
	if err != nil {
		t.Fatal(err)
	}
	if len(explicitExcluded) != 1 || explicitExcluded[0].NS != "AppData-Local-Temp-fixture" {
		t.Fatalf("explicit excluded namespace = %+v, want scoped temp namespace", explicitExcluded)
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

func TestProjectNamespaceResolvesSymlinks(t *testing.T) {
	target := filepath.Join(t.TempDir(), "target")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skip("symlink not supported in this environment")
	}
	targetNS := ProjectNamespace(target)
	linkNS := ProjectNamespace(link)
	if targetNS != linkNS {
		t.Fatalf("ProjectNamespace(link) = %q, want %q (target)", linkNS, targetNS)
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

func behaviorSession(name, ns string, b Behavior) Session {
	return Session{
		Session:  name,
		Path:     filepath.Join("root", ns, name+".jsonl"),
		Behavior: b,
	}
}

func TestAggregateCompactBehaviorJoinsRecurringFailuresAcrossSessions(t *testing.T) {
	sig := "go: cannot find module providing package"
	sessions := []Session{
		behaviorSession("s1", "C--work-fak", Behavior{
			FailureMass:  []RepeatFailureRow{{Tool: "Bash", Sig: sig, Count: 3}},
			TimeoutKills: 4,
			EditChurn:    map[string]int64{"a.go": 2},
		}),
		behaviorSession("s2", "C--work-fak", Behavior{
			FailureMass: []RepeatFailureRow{{Tool: "Bash", Sig: sig, Count: 5}},
		}),
		behaviorSession("s3", "-home-u-other", Behavior{
			FailureMass: []RepeatFailureRow{{Tool: "Bash", Sig: sig, Count: 3}},
			SleepPolls:  2,
		}),
		behaviorSession("s4", "C--work-fak", Behavior{}), // clean session must not join
	}
	cb := aggregateCompactBehavior(sessions)
	if cb == nil {
		t.Fatal("expected non-nil behavior aggregate")
	}
	if len(cb.RecurringFailures) != 1 {
		t.Fatalf("recurring = %+v, want 1 joined class", cb.RecurringFailures)
	}
	row := cb.RecurringFailures[0]
	if row.Tool != "Bash" || row.Sessions != 3 || row.Occurrences != 11 || len(row.Namespaces) != 2 {
		t.Fatalf("joined row = %+v, want Bash/3 sessions/11 occ/2 namespaces", row)
	}
	if row.ExampleSession != "s1" {
		t.Fatalf("example session = %q, want first-seen s1", row.ExampleSession)
	}
	if cb.TimeoutKills != 4 || cb.SleepPolls != 2 || cb.StuckSessions != 3 || cb.WastedMutationCalls != 2 {
		t.Fatalf("window aggregates = %+v", cb)
	}
}

func TestAggregateCompactBehaviorIgnoresSingleSessionClass(t *testing.T) {
	sessions := []Session{
		behaviorSession("s1", "C--work-fak", Behavior{FailureMass: []RepeatFailureRow{{Tool: "Bash", Sig: "x", Count: 9}}}),
		behaviorSession("s2", "C--work-fak", Behavior{}),
	}
	cb := aggregateCompactBehavior(sessions)
	if cb == nil {
		t.Fatal("expected non-nil aggregate (one stuck session present)")
	}
	if len(cb.RecurringFailures) != 0 {
		t.Fatalf("single-session class must not join across sessions: %+v", cb.RecurringFailures)
	}
	if _, ok := compactProcessIssuePressure(cb); ok {
		t.Fatal("a single-session loop must not raise a cross-session process recommendation")
	}
}

func TestAggregateCompactBehaviorNilWhenBenign(t *testing.T) {
	sessions := []Session{
		behaviorSession("s1", "ns", Behavior{}),
		behaviorSession("s2", "ns", Behavior{}),
	}
	if cb := aggregateCompactBehavior(sessions); cb != nil {
		t.Fatalf("benign window should yield nil behavior: %+v", cb)
	}
}

func TestBuildCompactReportRecommendsProcessIssue(t *testing.T) {
	sig := "exit status 143: command timed out"
	mk := func(name, ns string, count int64) Session {
		return behaviorSession(name, ns, Behavior{FailureMass: []RepeatFailureRow{{Tool: "Bash", Sig: sig, Count: count}}})
	}
	sessions := []Session{
		mk("s1", "C--work-fak", 3),
		mk("s2", "C--work-fak", 3),
		mk("s3", "C--work-fak", 4),
	}
	rep := BuildCompactReport(sessions, AggregateSessions(sessions), "C--work-fak", nil, false, 0, 3, nil, time.Now())
	var rec *CompactRecommendation
	for i := range rep.Recommendations {
		if rep.Recommendations[i].Kind == "process_issue_pressure" {
			rec = &rep.Recommendations[i]
		}
	}
	if rec == nil {
		t.Fatalf("expected process_issue_pressure recommendation, got %+v", rep.Recommendations)
	}
	if rec.Severity != "high" { // recurs across 3 sessions -> systemic -> high
		t.Fatalf("severity = %q, want high", rec.Severity)
	}
	if !strings.Contains(rec.Evidence, "sessions=3") || !strings.Contains(rec.Evidence, "tool=Bash") {
		t.Fatalf("evidence = %q", rec.Evidence)
	}
	if rep.Behavior == nil || len(rep.Behavior.RecurringFailures) != 1 {
		t.Fatalf("report behavior = %+v", rep.Behavior)
	}
}

func TestCompactReportCarriesShellChoice(t *testing.T) {
	// Corpus reproducing the #3227 window:
	// Bash: 194 calls / 5 errors (2.6%)
	// PowerShell: 33 calls / 6 errors (18.2%)
	sessions3227 := []Session{
		{
			Path:    "ns/s1.jsonl",
			Session: "s1",
			Tools:   map[string]int64{"Bash": 100, "PowerShell": 10, "Read": 500},
			Behavior: Behavior{
				ToolErrors: map[string]int64{"Bash": 1, "PowerShell": 2},
			},
		},
		{
			Path:    "ns/s2.jsonl",
			Session: "s2",
			Tools:   map[string]int64{"Bash": 94, "PowerShell": 23, "Edit": 20},
			Behavior: Behavior{
				ToolErrors: map[string]int64{"Bash": 4, "PowerShell": 4},
			},
		},
	}
	agg3227 := AggregateSessions(sessions3227)
	rep3227 := BuildCompactReport(sessions3227, agg3227, "ns", nil, false, 0, len(sessions3227), nil, time.Now())

	// 1. The compact record carries the fold
	sc := rep3227.ShellChoice
	if sc.Calls != 227 || sc.Errors != 11 {
		t.Fatalf("compact shell choice calls/errors = %d/%d, want 227/11", sc.Calls, sc.Errors)
	}
	if sc.Preferred != "Bash" {
		t.Errorf("compact preferred = %q, want Bash", sc.Preferred)
	}
	approx(t, "all-shell error rate", sc.ErrorRate, 11.0/227)

	bashRow := shellRow(t, sc.ShellChoice, "Bash")
	if bashRow.Calls != 194 || bashRow.Errors != 5 {
		t.Errorf("bash calls/errors = %d/%d, want 194/5", bashRow.Calls, bashRow.Errors)
	}
	approx(t, "bash error rate", bashRow.ErrorRate, 5.0/194)
	approx(t, "bash call share", bashRow.CallShare, 194.0/227)

	pwshRow := shellRow(t, sc.ShellChoice, "PowerShell")
	if pwshRow.Calls != 33 || pwshRow.Errors != 6 {
		t.Errorf("powershell calls/errors = %d/%d, want 33/6", pwshRow.Calls, pwshRow.Errors)
	}
	approx(t, "pwsh error rate", pwshRow.ErrorRate, 6.0/33)
	approx(t, "pwsh call share", pwshRow.CallShare, 33.0/227)

	// Per-session distribution is carried
	if sc.ShellErrorRate.Median == nil {
		t.Fatal("expected per-session shell error rate median in compact record")
	}
	approx(t, "shell-error-rate median", sc.ShellErrorRate.Median, (3.0/110+8.0/117)/2)

	// JSON round-trip preserves the fold
	data, err := json.Marshal(rep3227)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var decoded CompactReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if decoded.ShellChoice.Calls != 227 || decoded.ShellChoice.Errors != 11 {
		t.Fatalf("decoded calls/errors = %d/%d, want 227/11", decoded.ShellChoice.Calls, decoded.ShellChoice.Errors)
	}
	if decoded.ShellChoice.Preferred != "Bash" {
		t.Fatalf("decoded preferred = %q, want Bash", decoded.ShellChoice.Preferred)
	}

	// 2. The recommendation fires on a corpus reproducing the #3227 window
	var shellRec *CompactRecommendation
	for i := range rep3227.Recommendations {
		if strings.Contains(rep3227.Recommendations[i].Kind, "shell_friction") {
			shellRec = &rep3227.Recommendations[i]
			break
		}
	}
	if shellRec == nil {
		t.Fatalf("expected shell friction recommendation, got recommendations: %+v", rep3227.Recommendations)
	}
	if shellRec.Severity != "high" {
		t.Errorf("recommendation severity = %q, want high", shellRec.Severity)
	}
	if !strings.Contains(shellRec.Reason, "PowerShell") || !strings.Contains(shellRec.Evidence, "tool=PowerShell") {
		t.Errorf("recommendation should name PowerShell: %+v", shellRec)
	}

	// 3. A 3-call shell does not trip it
	sessionsNoise := []Session{
		{
			Path:    "ns/noise.jsonl",
			Session: "noise",
			Tools:   map[string]int64{"Bash": 100, "PowerShell": 3},
			Behavior: Behavior{
				ToolErrors: map[string]int64{"Bash": 1, "PowerShell": 1},
			},
		},
	}
	aggNoise := AggregateSessions(sessionsNoise)
	repNoise := BuildCompactReport(sessionsNoise, aggNoise, "ns", nil, false, 0, len(sessionsNoise), nil, time.Now())
	for _, rec := range repNoise.Recommendations {
		if strings.Contains(rec.Kind, "shell_friction") {
			t.Fatalf("3-call shell should not trip recommendation: %+v", rec)
		}
	}

	// 4. Empty window reports UNKNOWN / no error rate
	sessionsEmpty := []Session{
		{
			Path:    "ns/empty.jsonl",
			Session: "empty",
			Tools:   map[string]int64{"Read": 10},
		},
	}
	aggEmpty := AggregateSessions(sessionsEmpty)
	repEmpty := BuildCompactReport(sessionsEmpty, aggEmpty, "ns", nil, false, 0, len(sessionsEmpty), nil, time.Now())
	scEmpty := repEmpty.ShellChoice
	if scEmpty.Calls != 0 || scEmpty.Errors != 0 {
		t.Errorf("empty window calls/errors = %d/%d, want 0/0", scEmpty.Calls, scEmpty.Errors)
	}
	if scEmpty.Preferred != "UNKNOWN" {
		t.Errorf("empty window preferred = %q, want UNKNOWN", scEmpty.Preferred)
	}
	if scEmpty.ErrorRate != nil {
		t.Errorf("empty window error rate = %v, want nil (no rate, never 0%%)", *scEmpty.ErrorRate)
	}
	for _, s := range scEmpty.Shells {
		if s.ErrorRate != nil {
			t.Errorf("empty window shell %s error rate = %v, want nil", s.Tool, *s.ErrorRate)
		}
	}
	if scEmpty.ShellErrorRate.Median != nil {
		t.Errorf("empty window shell error rate median = %v, want nil", *scEmpty.ShellErrorRate.Median)
	}
	for _, rec := range repEmpty.Recommendations {
		if strings.Contains(rec.Kind, "shell_friction") {
			t.Fatalf("empty window should not trip recommendation: %+v", rec)
		}
	}

	// Empty window JSON output check
	emptyData, err := json.Marshal(repEmpty)
	if err != nil {
		t.Fatalf("json.Marshal empty failed: %v", err)
	}
	var decodedEmpty CompactReport
	if err := json.Unmarshal(emptyData, &decodedEmpty); err != nil {
		t.Fatalf("json.Unmarshal empty failed: %v", err)
	}
	if decodedEmpty.ShellChoice.Preferred != "UNKNOWN" {
		t.Errorf("decoded empty preferred = %q, want UNKNOWN", decodedEmpty.ShellChoice.Preferred)
	}
	if decodedEmpty.ShellChoice.ErrorRate != nil {
		t.Errorf("decoded empty error rate = %v, want nil", *decodedEmpty.ShellChoice.ErrorRate)
	}

	// Zero sessions slice also reports UNKNOWN with no error rate
	repZero := BuildCompactReport(nil, Aggregate{}, "ns", nil, false, 0, 0, nil, time.Now())
	if repZero.ShellChoice.Preferred != "UNKNOWN" || repZero.ShellChoice.ErrorRate != nil {
		t.Errorf("zero sessions preferred = %q (want UNKNOWN), error rate = %v (want nil)",
			repZero.ShellChoice.Preferred, repZero.ShellChoice.ErrorRate)
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
		"qwen2.5:14b":       BucketOpenWeights,
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

// TestNonClaudeProviderCostRates is the failure-class proof for #4823: before the
// price rows landed, PriceFor returned ok=false for deepseek/glm/kimi so the
// provider-cost KPI folded them UNMEASURED; after, each resolves to its published
// per-MTok rate. The still-unpriced providers (gpt/gemini) must STAY unpriced so
// the #4490 honesty rule (no fabricated $0.00) still fires for them.
func TestNonClaudeProviderCostRates(t *testing.T) {
	priced := []struct {
		model    string
		tier     string
		axis     int64 // 1 MTok exercised on one axis
		wantUSD  float64
		onOutput bool
	}{
		{"deepseek-v4-pro", "deepseek", 1_000_000, 0.435, false},
		{"glm-5.2", "glm", 1_000_000, 4.4, true},
		{"kimi-k2.6", "kimi", 1_000_000, 0.95, false},
	}
	for _, p := range priced {
		r, ok := PriceFor(p.model)
		if !ok {
			t.Fatalf("PriceFor(%q) = !ok, want a published rate card (#4823)", p.model)
		}
		if r.Input <= 0 || r.Output <= 0 {
			t.Fatalf("%s rate card has non-positive rates: %+v", p.model, r)
		}
		if got := ModelTier(p.model); got != p.tier {
			t.Fatalf("ModelTier(%q) = %q, want %q", p.model, got, p.tier)
		}
		var got float64
		if p.onOutput {
			got = CostUSD(p.model, 0, 0, 0, p.axis)
		} else {
			got = CostUSD(p.model, p.axis, 0, 0, 0)
		}
		if math.Abs(got-p.wantUSD) > 1e-9 {
			t.Fatalf("CostUSD(%q, 1 MTok) = %.6f, want %.6f", p.model, got, p.wantUSD)
		}
	}
	// Providers with no clean published card stay UNPRICED (honest UNMEASURED, not
	// a fabricated $0.00) — this guards the boundary the KPI relies on.
	//
	// gpt/gemini are still here after #5115 tried to move them into the priced set.
	// Three findings from that attempt, so the next implementer does not re-derive them:
	//
	//  1. A bare family key over-matches. PriceFor is substring/first-hit, so "gpt"
	//     also captures gpt-oss, which fak serves itself — see
	//     TestOpenWeightsFamiliesAreNeverPricedAsVendor for the executable form.
	//  2. One Rates row cannot carry either vendor's tier structure. Rates has four
	//     flat axes and no tier or context axis, while the ids already in this tree
	//     (gpt-5.6-sol/gpt-5.5/gpt-5-codex/gpt-4o-mini; gemini-3-pro/gemini-3.5-flash)
	//     span a >20x published spread. Claude is keyed PER TIER for exactly this
	//     reason; deepseek/glm/kimi got single keys only because each cited one
	//     flagship rate. Pricing a family therefore emits a number wrong by up to that
	//     spread with NO UnpricedModels hold — the #4490 failure this boundary prevents.
	//  3. The rates themselves were not citable from this host: fetching the OpenAI and
	//     Google pricing pages is refused (TRUST_VIOLATION, terminal), and search
	//     returned only third-party trackers that disagree with each other. The
	//     deepseek/glm/kimi rows above carry a primary source and a retrieval date;
	//     matching that discipline means a real page, not a tracker.
	//
	// Moving these into the priced set needs per-tier keys matched longest-first (an
	// unrecognized tier staying UNPRICED rather than falling back to a family rate)
	// AND a cited rate table. Until both hold, unpriced is the honest answer.
	for _, m := range []string{"gpt-5", "gemini-2.5-pro", "qwen2.5:14b"} {
		if _, ok := PriceFor(m); ok {
			t.Fatalf("PriceFor(%q) = ok, but this provider has no rate card and must stay UNMEASURED", m)
		}
		if got := ModelTier(m); got != "unpriced" {
			t.Fatalf("ModelTier(%q) = %q, want \"unpriced\"", m, got)
		}
	}
}

func TestReportMarkdownRendersTrajectoryOnlyWhenPresent(t *testing.T) {
	plain := Session{Session: "plain"}
	if md := ReportMarkdown([]Session{plain}, AggregateSessions([]Session{plain}), "", nil, false, 0, 1, nil, time.Now()); strings.Contains(md, "## Trajectory") {
		t.Fatal("non-instrumented session rendered trajectory panel")
	}
	inst := Session{Session: "sess-a", Trajectory: &TrajectoryPanel{Objectives: []TrajectoryObjective{{ObjectiveID: "o", Title: "ship", Score: .6, Signal: "STALL", Nudges: 1, NudgesDelivered: 1}}}}
	md := ReportMarkdown([]Session{inst}, AggregateSessions([]Session{inst}), "", nil, false, 0, 1, nil, time.Now())
	for _, want := range []string{"## Trajectory", "sess-a", "ship", "0.600", "STALL"} {
		if !strings.Contains(md, want) {
			t.Fatalf("missing %q", want)
		}
	}
}
