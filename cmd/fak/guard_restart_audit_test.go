package main

// Tests for the restart-chain witness (#3057). The suite name RestartChain is
// the gate: `go test ./cmd/fak -run 'RestartChain'` (plus the journal-side
// twin) is the witness command for the issue.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestGuardRestartAuditSurfacesCrashGiveUpReason(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	// Close before the test returns: the journal holds an open handle on audit.jsonl, and on
	// Windows an open handle makes t.TempDir()'s RemoveAll cleanup fail the test.
	defer func() { _ = j.Close() }()
	guardRecordCrashRestartGiveUp(j, "claude", "trace-crash")

	var out bytes.Buffer
	if code := runGuardRestartAudit(&out, &out, []string{"--journal-dir", dir, "--scan-temp=false", "--json"}); code != 0 {
		t.Fatalf("runGuardRestartAudit code=%d output=%s", code, out.String())
	}
	if !strings.Contains(out.String(), guardCrashRestartExhaustedReason) {
		t.Fatalf("restart audit output %q does not surface typed give-up reason", out.String())
	}
}

func TestGuardRestartAuditRotatedSegmentsMatchUnrotated(t *testing.T) {
	writeJournal := func(t *testing.T, rotate bool) string {
		t.Helper()
		dir := t.TempDir()
		path := filepath.Join(dir, "audit.jsonl")
		j, err := journal.Open(path)
		if err != nil {
			t.Fatal(err)
		}

		j.AppendRestartHop("claude", "guard-before", journal.RestartHop{
			Hop: 1, FromTrace: "trace-1", ToTrace: "trace-2", Child: "trace-2",
			Handback: guardRestartHandbackContinue, Status: journal.RestartHopOK,
		})
		guardRecordCrashRestartGiveUp(j, "claude", "guard-before")
		if rotate {
			if _, err := j.Cut(); err != nil {
				_ = j.Close()
				t.Fatalf("cut restart-audit fixture: %v", err)
			}
		}
		j.AppendRestartHop("claude", "guard-after", journal.RestartHop{
			Hop: 2, FromTrace: "trace-2", ToTrace: "trace-3", Child: "trace-3",
			Handback: guardRestartHandbackContinue, Status: journal.RestartHopOK,
		})
		guardRecordCrashRestartGiveUp(j, "claude", "guard-after")
		if err := j.Close(); err != nil {
			t.Fatalf("close restart-audit fixture: %v", err)
		}
		return dir
	}

	type totals struct {
		journals int
		hops     int
		giveUps  int
		counts   map[string]int
	}
	scanTotals := func(dir string) totals {
		rep := guardRestartAuditScan(dir, nil, "")
		return totals{journals: rep.Journals, hops: len(rep.Hops), giveUps: len(rep.GiveUps), counts: rep.Counts}
	}

	unrotated := scanTotals(writeJournal(t, false))
	rotated := scanTotals(writeJournal(t, true))
	want := totals{journals: 1, hops: 2, giveUps: 2, counts: map[string]int{journal.RestartHopOK: 2}}
	if !reflect.DeepEqual(unrotated, want) {
		t.Fatalf("unrotated totals = %+v, want %+v", unrotated, want)
	}
	if !reflect.DeepEqual(rotated, unrotated) {
		t.Fatalf("rotated totals = %+v, want unrotated totals %+v", rotated, unrotated)
	}
}

func TestRestartChainHopFromEvent(t *testing.T) {
	full := guardBudgetRestartEvent{
		Schema:      "fak.guard.budget_restart.v1",
		FromTraceID: "gw-1",
		ToTraceID:   "gw-2",
		SeedFile:    `C:\tmp\fak-guard-reset-1\reset-gw-1-to-gw-2.json`,
		SeedText:    "carry the task forward", // 22 bytes -> 6 approx tokens
	}
	cases := []struct {
		name     string
		ev       guardBudgetRestartEvent
		agent    string
		handback string
		status   string
	}{
		{"recognized agent engages continue", full, "claude", guardRestartHandbackContinue, journal.RestartHopOK},
		{"windows launcher path still recognized", full, `C:\tools\claude.exe`, guardRestartHandbackContinue, journal.RestartHopOK},
		{"unrecognized agent is an honest orphan", full, "mysteryagent", guardRestartHandbackOrphaned, journal.RestartHopInert},
		{"no continuation trace is a break", func() guardBudgetRestartEvent { e := full; e.ToTraceID = ""; return e }(), "claude", guardRestartHandbackContinue, journal.RestartHopBreak},
		{"no surviving seed is a break", func() guardBudgetRestartEvent { e := full; e.SeedFile, e.SeedText = "", ""; return e }(), "claude", guardRestartHandbackContinue, journal.RestartHopBreak},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hop := guardRestartHopFromEvent(tc.ev, 3, tc.agent)
			if hop.Handback != tc.handback || hop.Status != tc.status {
				t.Fatalf("(handback, status) = (%q, %q), want (%q, %q)", hop.Handback, hop.Status, tc.handback, tc.status)
			}
			if hop.Schema != journal.RestartChainSchema || hop.Hop != 3 {
				t.Fatalf("record identity wrong: %+v", hop)
			}
			if hop.FromTrace != tc.ev.FromTraceID || hop.ToTrace != tc.ev.ToTraceID || hop.Child != tc.ev.ToTraceID {
				t.Fatalf("trace lineage wrong: %+v", hop)
			}
			if want := guardApproxTokens(tc.ev.SeedText); hop.SeedTokens != want {
				t.Fatalf("SeedTokens = %d, want %d", hop.SeedTokens, want)
			}
		})
	}
	if got := guardApproxTokens(""); got != 0 {
		t.Fatalf("guardApproxTokens(\"\") = %d, want 0", got)
	}
	if got := guardApproxTokens("ab"); got != 1 {
		t.Fatalf("guardApproxTokens(\"ab\") = %d, want ceiling 1", got)
	}
}

func TestRestartChainOneLiner(t *testing.T) {
	hop := journal.RestartHop{
		Hop: 1, FromTrace: "gw-1", ToTrace: "gw-2", SeedTokens: 214,
		Handback: guardRestartHandbackContinue, Child: "gw-2", Status: journal.RestartHopOK,
		SeedFile: `C:\tmp\fak-guard-reset-1\reset-gw-1-to-gw-2.json`,
	}
	line := guardRestartHopOneLiner(hop)
	for _, want := range []string{
		"fak guard: restart #1 ", "from=gw-1", "to=gw-2", "seed=214tok",
		"handback=continue", "child=gw-2", "status=ok",
		"seed_file=C:/tmp/fak-guard-reset-1/reset-gw-1-to-gw-2.json",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("one-liner missing %q:\n%s", want, line)
		}
	}
	if strings.Contains(line, `\`) {
		t.Fatalf("one-liner must be forward-slash normalized on every OS:\n%s", line)
	}
	if noSeed := guardRestartHopOneLiner(journal.RestartHop{Hop: 2, Status: journal.RestartHopBreak}); strings.Contains(noSeed, "seed_file=") {
		t.Fatalf("seed_file must be omitted when the seed write failed:\n%s", noSeed)
	}
}

func TestRestartChainShrinkSignal(t *testing.T) {
	// Fix #2 no-shrink signal: only a seed-prompt handback boots fresh on the distilled seed (and
	// strips --continue), so only it SHRINKS the exhausted window. A --continue reattach or an
	// ORPHANED cold relaunch re-inflates (or leaves the seed unread) → shrink=no, the hop an operator
	// should watch for a re-exhaustion loop.
	seed := journal.RestartHop{Hop: 1, Handback: guardRestartHandbackSeedPrompt, Status: journal.RestartHopOK}
	if guardRestartHopShrink(seed) != "yes" || !strings.Contains(guardRestartHopOneLiner(seed), "shrink=yes") {
		t.Fatalf("seed-prompt hop must report shrink=yes, got %q", guardRestartHopOneLiner(seed))
	}
	for _, hb := range []string{guardRestartHandbackContinue, guardRestartHandbackOrphaned} {
		h := journal.RestartHop{Hop: 2, Handback: hb, Status: journal.RestartHopOK}
		if guardRestartHopShrink(h) != "no" || !strings.Contains(guardRestartHopOneLiner(h), "shrink=no") {
			t.Fatalf("%s hop must report shrink=no (re-inflates), got %q", hb, guardRestartHopOneLiner(h))
		}
	}
}

func TestRestartChainEmitWritesRowAndLine(t *testing.T) {
	j := journal.OpenMemory()
	var stderr bytes.Buffer
	hop := guardRestartHopFromEvent(guardBudgetRestartEvent{
		FromTraceID: "gw-1", ToTraceID: "gw-2", SeedFile: "/tmp/s.json", SeedText: "resume",
	}, 1, "claude")
	guardEmitRestartHop(j, &stderr, "claude", "guard-abc", hop)

	rows := j.Recent(0)
	if len(rows) != 1 || rows[0].Kind != journal.KindRestartHop {
		t.Fatalf("journal rows = %+v, want one RESTART_HOP", rows)
	}
	if rows[0].Tool != "claude" || rows[0].TraceID != "guard-abc" || rows[0].Restart == nil {
		t.Fatalf("row identity/payload wrong: %+v", rows[0])
	}
	if got := stderr.String(); !strings.Contains(got, "restart #1 ") || !strings.Contains(got, "handback=continue") {
		t.Fatalf("stderr one-liner wrong: %q", got)
	}
	if n := strings.Count(stderr.String(), "\n"); n != 1 {
		t.Fatalf("want exactly ONE correlated stderr line, got %d:\n%s", n, stderr.String())
	}
	// Both halves are nil-safe: no journal (--no-audit) and no stderr (quiet).
	guardEmitRestartHop(nil, nil, "claude", "guard-abc", hop)
}

// writeRestartAuditFixture lays out one journal (a DECIDE row the scanner must
// skip + two RESTART_HOP rows) and two seed files (one recorded, one orphan)
// with FIXED timestamps, so scans over it are byte-deterministic. The rows are
// written as raw JSONL — ReadRows does not verify hashes (that is `fak audit
// verify`'s job), which is exactly the robustness the scanner leans on.
func writeRestartAuditFixture(t *testing.T) (journalDir, seedDir string) {
	t.Helper()
	journalDir, seedDir = filepath.Join(t.TempDir(), "ja"), filepath.Join(t.TempDir(), "seeds")
	for _, d := range []string{journalDir, seedDir} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	rows := strings.Join([]string{
		`{"seq":1,"ts_unix_nano":1699999999000000000,"kind":"DECIDE","tool":"bash","trace_id":"gw-1","verdict":"allow","by":"policy","prev_hash":"","hash":"h1"}`,
		`{"seq":2,"ts_unix_nano":1700000000000000000,"kind":"RESTART_HOP","tool":"claude","trace_id":"guard-abc","reason":"ok","by":"guard-supervisor","prev_hash":"h1","hash":"h2","restart":{"schema":"fak.guard.restart_chain.v1","hop":1,"from_trace_id":"gw-1","to_trace_id":"gw-2","seed_file":"/tmp/fak-guard-reset-1/reset-gw-1-to-gw-2.json","seed_tokens":12,"handback":"continue","child_session_id":"gw-2","status":"ok"}}`,
		`{"seq":3,"ts_unix_nano":1700000001000000000,"kind":"RESTART_HOP","tool":"mysteryagent","trace_id":"guard-abc","reason":"inert","by":"guard-supervisor","prev_hash":"h2","hash":"h3","restart":{"schema":"fak.guard.restart_chain.v1","hop":2,"from_trace_id":"gw-2","to_trace_id":"gw-3","seed_file":"/tmp/fak-guard-reset-2/reset-gw-2-to-gw-3.json","seed_tokens":6,"handback":"ORPHANED","child_session_id":"gw-3","status":"inert"}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(journalDir, "interactive-1.jsonl"), []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	seeds := map[string]string{
		// Recorded by the seq=3 hop row above: counted, not double-reported.
		"reset-gw-2-to-gw-3.json": `{"schema":"fak.guard.budget_restart.v1","from_trace_id":"gw-2","to_trace_id":"gw-3","seed_text":"resume the triage task","note":"n"}`,
		// No recorded hop anywhere: the pre-#3057 orphan, backfilled as loss.
		"reset-gw-7-to-gw-8.json": `{"schema":"fak.guard.budget_restart.v1","from_trace_id":"gw-7","to_trace_id":"gw-8","seed_text":"carry the task forward","note":"n"}`,
	}
	mtime := time.Unix(1700000100, 0)
	for name, body := range seeds {
		p := filepath.Join(seedDir, name)
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, mtime, mtime); err != nil {
			t.Fatal(err)
		}
	}
	return journalDir, seedDir
}

func TestRestartChainAuditScanJoinsAndBackfills(t *testing.T) {
	journalDir, seedDir := writeRestartAuditFixture(t)
	// A malformed seed must degrade to a NOTE, never a silent narrowing.
	if err := os.WriteFile(filepath.Join(seedDir, "reset-broken.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	rep := guardRestartAuditScan(journalDir, []string{seedDir}, "")
	if rep.Journals != 1 || rep.Seeds != 2 {
		t.Fatalf("(journals, seeds) = (%d, %d), want (1, 2)", rep.Journals, rep.Seeds)
	}
	if len(rep.Hops) != 3 {
		t.Fatalf("hops = %d, want 3 (2 recorded + 1 backfilled): %+v", len(rep.Hops), rep.Hops)
	}
	if rep.Counts[journal.RestartHopOK] != 1 || rep.Counts[journal.RestartHopInert] != 1 || rep.Counts[journal.RestartHopLoss] != 1 {
		t.Fatalf("counts = %v, want ok=1 inert=1 loss=1", rep.Counts)
	}
	orphan := rep.Hops[2] // fixed timestamps order the backfill last
	if orphan.Source != "backfill" || orphan.Handback != guardRestartHandbackOrphaned || orphan.Status != journal.RestartHopLoss {
		t.Fatalf("orphan not backfilled honestly: %+v", orphan)
	}
	if orphan.Child != "" || orphan.Hop != 0 {
		t.Fatalf("backfill must not fabricate a child session or ordinal: %+v", orphan)
	}
	if orphan.SeedTokens != guardApproxTokens("carry the task forward") {
		t.Fatalf("orphan SeedTokens = %d", orphan.SeedTokens)
	}
	if len(rep.Notes) != 1 || !strings.Contains(rep.Notes[0], "reset-broken.json") {
		t.Fatalf("malformed seed must surface as a note: %v", rep.Notes)
	}
}

func TestRestartChainAuditTraceFilter(t *testing.T) {
	journalDir, seedDir := writeRestartAuditFixture(t)
	rep := guardRestartAuditScan(journalDir, []string{seedDir}, "gw-8")
	if len(rep.Hops) != 1 || rep.Hops[0].ToTrace != "gw-8" {
		t.Fatalf("--trace gw-8 must isolate the orphan hop: %+v", rep.Hops)
	}
	if rep.Counts[journal.RestartHopLoss] != 1 || len(rep.Counts) != 1 {
		t.Fatalf("filtered counts = %v, want only loss=1", rep.Counts)
	}
}

func TestRestartChainAuditColor(t *testing.T) {
	journalDir, seedDir := writeRestartAuditFixture(t)
	var out, errOut bytes.Buffer
	if rc := runGuardRestartAudit(&out, &errOut, []string{
		"--journal-dir", journalDir, "--seed-dir", seedDir, "--scan-temp=false", "--color=always",
	}); rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, errOut.String())
	}
	if !strings.Contains(out.String(), "\x1b[31m") {
		t.Fatalf("--color=always must paint the loss hop red:\n%s", out.String())
	}
	out.Reset()
	if rc := runGuardRestartAudit(&out, &errOut, []string{
		"--journal-dir", journalDir, "--seed-dir", seedDir, "--scan-temp=false", "--color=never",
	}); rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, errOut.String())
	}
	if strings.Contains(out.String(), "\x1b[") {
		t.Fatalf("--color=never must emit no ANSI:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "restart-chain audit: 3 hop(s) (ok=1 inert=1 break=0 loss=1)") {
		t.Fatalf("human summary line wrong:\n%s", out.String())
	}
}

// restartAuditGoldenPath is the committed fixture pinning the read-only scan
// report shape (fak.guard.restart_audit.v1). Regenerate deliberately with
// FAK_UPDATE_GOLDEN=1 after an intentional schema change.
const restartAuditGoldenPath = "../../experiments/restart-chain/restart-audit-golden.json"

func TestRestartChainAuditGolden(t *testing.T) {
	journalDir, seedDir := writeRestartAuditFixture(t)
	var out, errOut bytes.Buffer
	if rc := runGuardRestartAudit(&out, &errOut, []string{
		"--journal-dir", journalDir, "--seed-dir", seedDir, "--scan-temp=false", "--json",
	}); rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, errOut.String())
	}
	var rep guardRestartAuditReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("report is not valid JSON: %v\n%s", err, out.String())
	}
	// The absolute temp paths are the ONLY non-deterministic bytes in the
	// report; pin everything else by reducing them to their basenames.
	for i := range rep.Hops {
		if rep.Hops[i].SeedFile != "" {
			rep.Hops[i].SeedFile = filepath.Base(rep.Hops[i].SeedFile)
		}
		if rep.Hops[i].Journal != "" {
			rep.Hops[i].Journal = filepath.Base(rep.Hops[i].Journal)
		}
	}
	got, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')

	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(restartAuditGoldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(restartAuditGoldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Log("golden fixture updated; commit it and re-run without FAK_UPDATE_GOLDEN")
		return
	}
	want, err := os.ReadFile(restartAuditGoldenPath)
	if err != nil {
		t.Fatalf("golden fixture missing (run once with FAK_UPDATE_GOLDEN=1): %v", err)
	}
	// Normalize line endings: the fixture may be checked out with CRLF on
	// Windows depending on .gitattributes; the report itself is LF.
	if g, w := strings.ReplaceAll(string(got), "\r\n", "\n"), strings.ReplaceAll(string(want), "\r\n", "\n"); g != w {
		t.Fatalf("report drifted from golden fixture %s\n--- got ---\n%s\n--- want ---\n%s", restartAuditGoldenPath, g, w)
	}
}

func TestRestartChainStatusAddendumRendersHops(t *testing.T) {
	journalDir, seedDir := writeRestartAuditFixture(t)
	rep := guardRestartAuditScan(journalDir, []string{seedDir}, "gw-2")
	var buf bytes.Buffer
	for _, h := range rep.Hops {
		fmt.Fprintf(&buf, "  %s\n", guardRestartAuditHopLine(h, false))
	}
	got := buf.String()
	if !strings.Contains(got, "hop#1 from=gw-1 to=gw-2") || !strings.Contains(got, "[journal interactive-1.jsonl]") {
		t.Fatalf("addendum hop line wrong:\n%s", got)
	}
	if !strings.Contains(got, "2026-") && !strings.Contains(got, "2023-") {
		t.Fatalf("addendum must carry a human timestamp:\n%s", got)
	}
}
