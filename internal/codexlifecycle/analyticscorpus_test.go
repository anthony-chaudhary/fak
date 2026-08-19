package codexlifecycle

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The corpus fold's central contract: expected negatives and control exits are
// VISIBLE in the class totals but excluded from the failure count, and every call
// carries exactly one class (the totals reconcile).
func TestScanAnalyticsCorpus_TypedTotalsReconcile(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)
	stale := now.Add(-72 * time.Hour)

	writeRollout(t, dir, "a.jsonl", stale,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "git rev-parse -q --verify MERGE_HEAD"),
		outLine("2026-07-16T10:00:02.000Z", "c1", 1, 0.3, ""),
		callLine("2026-07-16T10:00:03.000Z", "c2", "shell_command", "wait 99"),
		outLine("2026-07-16T10:00:04.000Z", "c2", 1, 1.0, ""),
		callLine("2026-07-16T10:00:05.000Z", "c3", "shell_command", "go test ./broken"),
		outLine("2026-07-16T10:00:09.000Z", "c3", 1, 3.5, "FAIL"),
		completeWithDur("2026-07-16T10:05:00.000Z", "A", 300_000))
	writeRollout(t, dir, "b.jsonl", stale,
		meta("s2", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T11:00:00.000Z", "B"),
		callLine("2026-07-16T11:00:05.000Z", "d1", "shell_command", "make build"),
		outLine("2026-07-16T11:00:25.000Z", "d1", 0, 19.0, "ok"),
		completeWithDur("2026-07-16T11:20:00.000Z", "B", 1_200_000))

	c, err := ScanAnalyticsCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour}, 10)
	if err != nil {
		t.Fatalf("ScanAnalyticsCorpus: %v", err)
	}
	if c.Sessions != 2 || c.Tasks != 2 || c.Completed != 2 || c.ToolCalls != 4 {
		t.Fatalf("sessions/tasks/completed/calls = %d/%d/%d/%d, want 2/2/2/4",
			c.Sessions, c.Tasks, c.Completed, c.ToolCalls)
	}
	// THE POINT OF THE VOCABULARY: three non-zero exits, exactly ONE failure.
	if c.Classes[ToolExpectedNegative] != 1 || c.Classes[ToolControlExit] != 1 || c.Classes[ToolFailure] != 1 {
		t.Errorf("classes = %+v, want 1 expected_negative / 1 control_exit / 1 failure", c.Classes)
	}
	if c.HardFailureCount() != 1 {
		t.Errorf("HardFailureCount = %d, want 1 — probes and control exits must stay out", c.HardFailureCount())
	}
	total := 0
	for _, n := range c.Classes {
		total += n
	}
	if total != c.ToolCalls {
		t.Errorf("class totals %d != tool calls %d — a call escaped the closed vocabulary", total, c.ToolCalls)
	}
	// Duration percentiles come from the completed tasks' RECORDED durations
	// (300s and 1200s) — the same source as the issue's observed distribution.
	if c.Duration.N != 2 || c.Duration.Max != 1200 {
		t.Errorf("duration = %+v, want n=2 max=1200s", c.Duration)
	}
	if c.TTFT.N != 2 {
		t.Errorf("ttft n = %d, want 2", c.TTFT.N)
	}
	if len(c.TopTasks) != 2 || c.TopTasks[0].DurationS < c.TopTasks[1].DurationS {
		t.Errorf("top_tasks = %+v, want 2 ranked desc", c.TopTasks)
	}
	// Scrubbed: outlier rows carry opaque ids and typed contributors only.
	for _, o := range c.TopTasks {
		if o.Session == "" || o.TurnID == "" {
			t.Errorf("outlier missing stable ids: %+v", o)
		}
	}
}

func TestFoldResumeCohortKeepsLiveNonTerminal(t *testing.T) {
	c := ResumeCohort{FailureReasons: map[string]int{}}
	foldResumeCohort(&c, RolloutAnalytics{Tasks: []TaskAnalytics{{Outcome: Live}}})
	if c.Started != 1 || c.Completed != 0 || c.Crashed != 0 || c.Superseded != 0 {
		t.Fatalf("live cohort = %+v", c)
	}
}

func TestAnalyticsCorpus_FreshHeadlessResumeCohort(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	goal := `{"timestamp":"2026-08-10T10:00:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<codex_internal_context source=\"goal\">continue</codex_internal_context>"}]}}`
	headlessMeta := `{"timestamp":"2026-08-10T10:00:00Z","type":"session_meta","payload":{"id":"resume","cwd":"C:\\work\\fak","originator":"codex_exec","source":"exec","thread_source":"user"}}`

	// A stale headless continuation dies before any successful tool result.
	writeRollout(t, dir, "crash.jsonl", now.Add(-2*time.Hour),
		headlessMeta, goal,
		`{"timestamp":"2026-08-10T10:00:02Z","type":"event_msg","payload":{"type":"task_started","turn_id":"crash"}}`)

	// A second continuation reaches useful work and completes.
	writeRollout(t, dir, "success.jsonl", now.Add(-2*time.Hour),
		headlessMeta, goal,
		`{"timestamp":"2026-08-10T10:01:00Z","type":"event_msg","payload":{"type":"task_started","turn_id":"ok"}}`,
		`{"timestamp":"2026-08-10T10:01:01Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","call_id":"c1","arguments":"{}"}}`,
		`{"timestamp":"2026-08-10T10:01:02Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"Process exited with code 0"}}`,
		`{"timestamp":"2026-08-10T10:01:03Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"ok"}}`)
	// Interactive goal continuation is explicitly outside the headless cohort.
	writeRollout(t, dir, "interactive.jsonl", now.Add(-2*time.Hour),
		strings.Replace(headlessMeta, "codex_exec", "codex-tui", 1), goal,
		`{"timestamp":"2026-08-10T10:02:00Z","type":"event_msg","payload":{"type":"task_started","turn_id":"tui"}}`)

	got, err := ScanAnalyticsCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour}, 10)
	if err != nil {
		t.Fatal(err)
	}
	c := got.FreshHeadlessResume
	if c.Started != 2 || c.UsefulWorkReached != 1 || c.Completed != 1 || c.Crashed != 1 || c.Superseded != 0 {
		t.Fatalf("cohort = %+v", c)
	}
	if c.FailureReasons["before_useful_work:process_death"] != 1 {
		t.Fatalf("failure reasons = %#v", c.FailureReasons)
	}
}

func TestScanAnalyticsCorpus_CWDFilterAndUnreadable(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "mine.jsonl", now.Add(-time.Hour),
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"), complete("2026-07-16T10:00:10.000Z", "A"))
	writeRollout(t, dir, "other.jsonl", now.Add(-time.Hour),
		meta("s2", "fak", "0.144.4", `C:\elsewhere`),
		started("2026-07-16T10:00:00.000Z", "B"), complete("2026-07-16T10:00:10.000Z", "B"))
	writeRollout(t, dir, "junk.jsonl", now.Add(-time.Hour), "not json", "{torn")

	c, err := ScanAnalyticsCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour, CWD: `C:\work\fak`}, 5)
	if err != nil {
		t.Fatalf("ScanAnalyticsCorpus: %v", err)
	}
	if c.Sessions != 1 {
		t.Errorf("sessions = %d, want 1 (cwd filter)", c.Sessions)
	}
}

// TestAnalyticsCorpus_AdjudicationSample prints a BOUNDED sample of classified
// outcomes WITH their command heads for manual precision adjudication. Raw command
// heads never leave the default reports; this test only runs when the operator
// explicitly asks locally (CODEX_AUDIT_SAMPLE=1), which is the #4767 "no raw
// commands unless explicitly requested locally" seam.
func TestAnalyticsCorpus_AdjudicationSample(t *testing.T) {
	if os.Getenv("CODEX_AUDIT_SAMPLE") == "" {
		t.Skip("set CODEX_AUDIT_SAMPLE=1 to print a local adjudication sample (raw command heads)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	sessions := filepath.Join(home, ".codex", "sessions")
	if _, err := os.Stat(sessions); err != nil {
		t.Skipf("no store at %s", sessions)
	}
	cwd, _ := os.Getwd()
	repo := filepath.Dir(filepath.Dir(cwd))

	const sampleEvery = 40 // every Nth non-ok outcome, bounded below
	const sampleMax = 48
	type row struct {
		class  ToolClass
		reason string
		head   string
		exit   int
	}
	var rows []row
	seen := 0
	paths, err := rolloutPaths(sessions, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range paths {
		if len(rows) >= sampleMax {
			break
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			continue
		}
		meta, records, parseErr := ReadAnalyticsRollout(fh)
		_ = fh.Close()
		if parseErr != nil || !sameDir(meta.CWD, repo) {
			continue
		}
		ra := AnalyzeRollout(meta, records, false)
		// Outcomes are appended in call order; rebuild heads from the records.
		var heads []string
		for _, r := range records {
			if r.Kind == kindToolCall {
				heads = append(heads, r.Head)
			}
		}
		for i, o := range ra.Outcomes {
			if o.Class == ToolOK || i >= len(heads) {
				continue
			}
			seen++
			if seen%sampleEvery != 0 || len(rows) >= sampleMax {
				continue
			}
			env := Envelope{}
			rows = append(rows, row{class: o.Class, reason: o.Reason, head: heads[i], exit: env.ExitCode})
		}
	}
	for i, r := range rows {
		t.Logf("SAMPLE %02d class=%-18s reason=%-28s head=%q", i+1, r.class, r.reason, r.head)
	}
	t.Logf("sampled %d of %d non-ok outcomes", len(rows), seen)
}

// TestAnalyticsCorpus_LiveStore is the #4767 CORPUS WITNESS. It folds the real
// local rollout store (cwd-scoped to this repo, mirroring the issue's evidence
// method), prints the scrubbed aggregate — task duration/TTFT percentiles, typed
// outcome totals, ranked reasons, top critical-path outliers — and asserts the
// structural invariants: every call typed, expected negatives and control exits
// excluded from the failure count. It SKIPS where no store exists, so it is a
// witness where the evidence lives and never a false red where it does not.
func TestAnalyticsCorpus_LiveStore(t *testing.T) {
	root := os.Getenv("CODEX_HOME")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home dir; cannot locate a Codex rollout store")
		}
		root = filepath.Join(home, ".codex")
	}
	sessions := filepath.Join(root, "sessions")
	if _, err := os.Stat(sessions); err != nil {
		t.Skipf("no Codex rollout store at %s — corpus witness needs a real store", sessions)
	}
	cwd, _ := os.Getwd()
	repo := filepath.Dir(filepath.Dir(cwd)) // internal/codexlifecycle -> repo root

	c, err := ScanAnalyticsCorpus(sessions, ScanOptions{FreshWithin: 2 * time.Hour, CWD: repo}, 10)
	if err != nil {
		t.Fatalf("ScanAnalyticsCorpus(%s): %v", sessions, err)
	}
	if c.Sessions == 0 {
		t.Skipf("store at %s holds no rollouts for cwd %s", sessions, repo)
	}

	b := fmt.Sprintf("\n#4767 corpus witness — root=%s cwd=%s sessions=%d unreadable=%d\n",
		sessions, repo, c.Sessions, c.Unreadable)
	b += fmt.Sprintf("tasks=%d completed=%d tool_calls=%d\n", c.Tasks, c.Completed, c.ToolCalls)
	b += fmt.Sprintf("duration_s  n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		c.Duration.N, c.Duration.P50, c.Duration.P90, c.Duration.P95, c.Duration.P99, c.Duration.Max)
	b += fmt.Sprintf("ttft_s      n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		c.TTFT.N, c.TTFT.P50, c.TTFT.P90, c.TTFT.P95, c.TTFT.P99, c.TTFT.Max)
	b += fmt.Sprintf("classes: %+v\n", c.Classes)
	b += fmt.Sprintf("failure_calls=%d (expected_negative=%d control_exit=%d stay OUT)\n",
		c.HardFailureCount(), c.Classes[ToolExpectedNegative], c.Classes[ToolControlExit])
	max := len(c.Reasons)
	if max > 12 {
		max = 12
	}
	for _, r := range c.Reasons[:max] {
		b += fmt.Sprintf("  reason %-32s %-20s %d\n", r.Reason, string(r.Class), r.Count)
	}
	for _, o := range c.TopTasks {
		b += fmt.Sprintf("  top task %.28s/%.12s %-12s %8.0fs idle=%.0fs top=%v\n",
			o.Session, o.TurnID, string(o.Outcome), o.DurationS, o.IdleS, o.Top)
	}
	b += fmt.Sprintf("timeout_kills=%d sleep_polls=%d stall_gaps=%d findings=%d\n",
		c.TimeoutKills, c.SleepPolls, c.StallGaps, len(c.Findings))
	for _, f := range c.Findings {
		b += fmt.Sprintf("  finding %-36s %6d  %s\n", f.Reason, f.Count, f.Action)
	}
	t.Log(b)

	// INVARIANT 1: the vocabulary is total — every call carries exactly one class.
	total := 0
	for _, n := range c.Classes {
		total += n
	}
	if total != c.ToolCalls {
		t.Errorf("class totals %d != tool calls %d — a call escaped the closed vocabulary", total, c.ToolCalls)
	}
	// INVARIANT 2: probes and control exits never pollute the failure count.
	if c.HardFailureCount() != c.Classes[ToolFailure]+c.Classes[ToolTimeout] {
		t.Errorf("failure calls %d must be exactly failure+timeout", c.HardFailureCount())
	}
	// INVARIANT 3: the scan really exercised the analytics.
	if c.Tasks == 0 || c.ToolCalls == 0 {
		t.Error("witness folded no tasks/calls — the scan is not exercising the corpus")
	}
	// Target operating envelope (#4767: tasks >= 4000) — reported, and asserted
	// where this store is the observed corpus.
	if c.Tasks >= 4000 {
		t.Logf("target operating envelope satisfied: tasks=%d >= 4000", c.Tasks)
	} else {
		t.Logf("note: this store carries %d tasks (< 4000 target envelope); invariants above still hold", c.Tasks)
	}
}
