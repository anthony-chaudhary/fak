package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGuardStopDispositionKind pins the disposition -> coarse-kind rollup: every known
// disposition maps to its documented kind, and an unknown value rolls up conservatively
// to failopen (never silently to clean).
func TestGuardStopDispositionKind(t *testing.T) {
	cases := map[guardStopDisposition]guardStopKind{
		stopDispCleanCompletion:          stopKindClean,
		stopDispCleanWrapup:              stopKindClean,
		stopDispToolFeedbackContinue:     stopKindContinue,
		stopDispDenyAllContinue:          stopKindContinue,
		stopDispSameIssueContinue:        stopKindContinue,
		stopDispHandoffBlock:             stopKindContinue,
		stopDispBlindGiveUp:              stopKindStandDown,
		stopDispSameIssueGiveUp:          stopKindStandDown,
		stopDispModeOff:                  stopKindOff,
		stopDispShadow:                   stopKindShadow,
		stopDispFailOpenBadArgs:          stopKindFailOpen,
		stopDispFailOpenBadMode:          stopKindFailOpen,
		stopDispFailOpenNoMetricsURL:     stopKindFailOpen,
		stopDispFailOpenGaugeUnavailable: stopKindFailOpen,
	}
	for disp, want := range cases {
		if got := guardStopDispositionKind(disp); got != want {
			t.Errorf("kind(%q) = %q, want %q", disp, got, want)
		}
	}
	if got := guardStopDispositionKind(guardStopDisposition("totally_unknown")); got != stopKindFailOpen {
		t.Errorf("kind(unknown) = %q, want failopen (conservative)", got)
	}
}

// TestTranscriptNotesNoAllowedPath covers the sanctioned wrap-up matcher: it is
// case-insensitive and substring-anchored so the agent's one-line "no allowed path:"
// note is recognized wherever it lands in the final turn's text.
func TestTranscriptNotesNoAllowedPath(t *testing.T) {
	yes := []string{
		"no allowed path: the last step is a protected boundary",
		"Done. No Allowed Path: SECRET_EXFIL is terminal, wrapping up.",
		"...prose... no allowed path ...more prose",
	}
	no := []string{"", "task complete", "found a path forward", "allowed the write"}
	for _, s := range yes {
		if !transcriptNotesNoAllowedPath(s) {
			t.Errorf("expected match for %q", s)
		}
	}
	for _, s := range no {
		if transcriptNotesNoAllowedPath(s) {
			t.Errorf("unexpected match for %q", s)
		}
	}
}

// TestParseHookTranscriptPath extracts transcript_path from a Stop-hook payload and is
// fail-open (empty on missing key or malformed JSON).
func TestParseHookTranscriptPath(t *testing.T) {
	if got := parseHookTranscriptPath([]byte(`{"transcript_path":" /tmp/t.jsonl "}`)); got != "/tmp/t.jsonl" {
		t.Errorf("path = %q, want trimmed /tmp/t.jsonl", got)
	}
	if got := parseHookTranscriptPath([]byte(`{"session_id":"abc"}`)); got != "" {
		t.Errorf("missing key: got %q, want empty", got)
	}
	if got := parseHookTranscriptPath([]byte("not json")); got != "" {
		t.Errorf("malformed: got %q, want empty", got)
	}
	if got := parseHookTranscriptPath(nil); got != "" {
		t.Errorf("nil: got %q, want empty", got)
	}
}

// TestReadGuardStopTranscript covers the fail-open contract: an empty path yields nil
// (no transcript context at all), while a given-but-unreadable path yields a record with
// Read=false so a reader can tell "no path" from "path present but nothing parsed".
func TestReadGuardStopTranscript(t *testing.T) {
	if got := readGuardStopTranscript(""); got != nil {
		t.Errorf("empty path: got %+v, want nil", got)
	}
	if got := readGuardStopTranscript("   "); got != nil {
		t.Errorf("blank path: got %+v, want nil", got)
	}
	missing := filepath.Join(t.TempDir(), "does-not-exist.jsonl")
	got := readGuardStopTranscript(missing)
	if got == nil || got.Read {
		t.Errorf("missing file: got %+v, want non-nil with Read=false", got)
	}
}

// TestAppendAndSummarizeGuardStops round-trips rows through the ledger and folds them: the
// tally groups by kind and disposition, isolates the guard-ended (stand-down + fail-open)
// rows, and tracks the ts bounds. Foreign/blank lines are ignored.
func TestAppendAndSummarizeGuardStops(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "nested", "guard-stops.jsonl")
	rows := []guardStopRecord{
		{Ts: "2026-07-01T00:00:00Z", Disposition: string(stopDispCleanCompletion), Kind: string(stopKindClean)},
		{Ts: "2026-07-01T00:01:00Z", Disposition: string(stopDispDenyAllContinue), Kind: string(stopKindContinue), Blocked: true, Exit: 2},
		{Ts: "2026-07-01T00:02:00Z", Disposition: string(stopDispBlindGiveUp), Kind: string(stopKindStandDown), Depth: 25, Bound: 24},
		{Ts: "2026-07-01T00:03:00Z", Disposition: string(stopDispFailOpenGaugeUnavailable), Kind: string(stopKindFailOpen), Note: "dial tcp: refused"},
	}
	for _, r := range rows {
		r := r
		if err := appendGuardStopRecord(ledger, r); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	// a blank line and a foreign JSONL row must be skipped by the fold
	f, err := os.OpenFile(ledger, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("\n{\"schema\":\"other.v1\",\"disposition\":\"clean_completion\"}\n"); err != nil {
		t.Fatal(err)
	}
	f.Close()

	content, err := readGuardStopsLedger(ledger)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	sum := summarizeGuardStops(content, 10)
	if sum.Total != 4 {
		t.Errorf("Total = %d, want 4 (foreign/blank skipped)", sum.Total)
	}
	if sum.ByKind[stopKindClean] != 1 || sum.ByKind[stopKindContinue] != 1 || sum.ByKind[stopKindStandDown] != 1 || sum.ByKind[stopKindFailOpen] != 1 {
		t.Errorf("ByKind = %+v, want one of each of clean/continue/standdown/failopen", sum.ByKind)
	}
	if sum.StandDown != 1 || sum.FailOpen != 1 {
		t.Errorf("StandDown=%d FailOpen=%d, want 1 and 1", sum.StandDown, sum.FailOpen)
	}
	if sum.FirstTs != "2026-07-01T00:00:00Z" || sum.LastTs != "2026-07-01T00:03:00Z" {
		t.Errorf("ts bounds = %q..%q", sum.FirstTs, sum.LastTs)
	}
	// Recent holds only the guard-ended rows (stand-down + fail-open), in file order.
	if len(sum.Recent) != 2 {
		t.Fatalf("Recent len = %d, want 2", len(sum.Recent))
	}
	if sum.Recent[0].Disposition != string(stopDispBlindGiveUp) || sum.Recent[1].Disposition != string(stopDispFailOpenGaugeUnavailable) {
		t.Errorf("Recent order = [%s, %s]", sum.Recent[0].Disposition, sum.Recent[1].Disposition)
	}
}

// TestSummarizeGuardStopsRecentCap verifies the recent list keeps the LAST N ended rows.
func TestSummarizeGuardStopsRecentCap(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 5; i++ {
		rec := guardStopRecord{Schema: guardStopRecordSchema, Disposition: string(stopDispBlindGiveUp), Note: string(rune('a' + i))}
		js, _ := json.Marshal(rec)
		b.Write(js)
		b.WriteByte('\n')
	}
	sum := summarizeGuardStops(b.String(), 2)
	if len(sum.Recent) != 2 {
		t.Fatalf("Recent len = %d, want 2 (capped)", len(sum.Recent))
	}
	if sum.Recent[0].Note != "d" || sum.Recent[1].Note != "e" {
		t.Errorf("Recent = [%s, %s], want last two [d, e]", sum.Recent[0].Note, sum.Recent[1].Note)
	}
}

// TestSummarizeMissingKindRecomputes ensures an older row without a Kind field still groups
// by recomputing the kind from its disposition.
func TestSummarizeMissingKindRecomputes(t *testing.T) {
	line := `{"schema":"` + guardStopRecordSchema + `","disposition":"fail_open_bad_args"}` + "\n"
	sum := summarizeGuardStops(line, 5)
	if sum.FailOpen != 1 || sum.ByKind[stopKindFailOpen] != 1 {
		t.Errorf("recompute failed: FailOpen=%d ByKind=%+v", sum.FailOpen, sum.ByKind)
	}
}

// TestRunGuardStopsJSONAndHuman drives the command end-to-end over a written ledger, in
// both JSON and human render modes, and confirms a missing ledger is a valid empty view.
func TestRunGuardStopsJSONAndHuman(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "guard-stops.jsonl")
	for _, r := range []guardStopRecord{
		{Ts: "2026-07-01T00:00:00Z", Disposition: string(stopDispCleanCompletion)},
		{Ts: "2026-07-01T00:01:00Z", Disposition: string(stopDispFailOpenGaugeUnavailable), Note: "refused"},
	} {
		if err := appendGuardStopRecord(ledger, r); err != nil {
			t.Fatal(err)
		}
	}

	// JSON mode
	var out, errb bytes.Buffer
	if code := runGuardStops(&out, &errb, []string{"--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("json run exit=%d stderr=%s", code, errb.String())
	}
	var got guardStopsSummary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("json decode: %v\n%s", err, out.String())
	}
	if got.Total != 2 || got.FailOpen != 1 || got.Ledger != ledger {
		t.Errorf("json summary = %+v", got)
	}

	// Human mode: must name the fail-open caveat and the guard-ended headline.
	out.Reset()
	errb.Reset()
	if code := runGuardStops(&out, &errb, []string{"--ledger", ledger}); code != 0 {
		t.Fatalf("human run exit=%d stderr=%s", code, errb.String())
	}
	human := out.String()
	for _, want := range []string{"2 decision(s) recorded", "guard-ended sessions: 1", "fail-open"} {
		if !strings.Contains(human, want) {
			t.Errorf("human output missing %q:\n%s", want, human)
		}
	}

	// Missing ledger is empty, not an error.
	out.Reset()
	errb.Reset()
	if code := runGuardStops(&out, &errb, []string{"--ledger", filepath.Join(dir, "absent.jsonl")}); code != 0 {
		t.Fatalf("missing-ledger exit=%d, want 0", code)
	}
	if !strings.Contains(out.String(), "no stop decisions recorded yet") {
		t.Errorf("missing-ledger output = %q", out.String())
	}
}

// TestRunGuardStopHookRecordsFailOpenAndOff proves the Stop hook actuator emits exactly one
// typed row per invocation via its defer, classifying the non-gateway terminal outcomes
// without any network: mode=off -> mode_off, a bad --mode -> fail_open_bad_mode, and
// enforce-without-a-metrics-URL -> fail_open_no_metrics_url. Recording is gated on the
// wired ledger env, so it is hermetic.
func TestRunGuardStopHookRecordsFailOpenAndOff(t *testing.T) {
	cases := []struct {
		name     string
		argv     []string
		wantExit int
		wantDisp guardStopDisposition
	}{
		{"mode-off", []string{"--mode", "off"}, 0, stopDispModeOff},
		{"bad-mode", []string{"--mode", "bogus"}, 0, stopDispFailOpenBadMode},
		{"no-metrics-url", []string{"--mode", "enforce"}, 0, stopDispFailOpenNoMetricsURL},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ledger := filepath.Join(t.TempDir(), "stops.jsonl")
			t.Setenv(guardStopsLedgerEnv, ledger)
			t.Setenv(guardStopsModeEnv, "")
			// Neutralize env that could hand the hook a metrics URL, so no-metrics-url is reachable.
			t.Setenv(guardStopHookEnvMetricsURL, "")
			t.Setenv("ANTHROPIC_BASE_URL", "")

			var errb bytes.Buffer
			exit := runGuardStopHook(&errb, strings.NewReader(""), tc.argv)
			if exit != tc.wantExit {
				t.Fatalf("exit=%d want %d (stderr=%s)", exit, tc.wantExit, errb.String())
			}
			recs := readGuardStopRows(t, ledger)
			if len(recs) != 1 {
				t.Fatalf("recorded %d rows, want exactly 1", len(recs))
			}
			r := recs[0]
			if r.Disposition != string(tc.wantDisp) {
				t.Errorf("disposition = %q, want %q", r.Disposition, tc.wantDisp)
			}
			if r.Kind != string(guardStopDispositionKind(tc.wantDisp)) {
				t.Errorf("kind = %q, want %q", r.Kind, guardStopDispositionKind(tc.wantDisp))
			}
			if r.Exit != tc.wantExit {
				t.Errorf("recorded exit = %d, want %d", r.Exit, tc.wantExit)
			}
			if r.Schema != guardStopRecordSchema {
				t.Errorf("schema = %q", r.Schema)
			}
		})
	}
}

// TestRunGuardStopHookRecordingDisabled confirms recording is a no-op when no ledger is
// wired and when the mode switch is off — the hook still returns its decision.
func TestRunGuardStopHookRecordingDisabled(t *testing.T) {
	// no ledger env at all
	t.Setenv(guardStopsLedgerEnv, "")
	var errb bytes.Buffer
	if exit := runGuardStopHook(&errb, strings.NewReader(""), []string{"--mode", "off"}); exit != 0 {
		t.Fatalf("exit=%d, want 0", exit)
	}

	// ledger wired but mode=off suppresses the write
	ledger := filepath.Join(t.TempDir(), "stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)
	t.Setenv(guardStopsModeEnv, "off")
	errb.Reset()
	if exit := runGuardStopHook(&errb, strings.NewReader(""), []string{"--mode", "off"}); exit != 0 {
		t.Fatalf("exit=%d, want 0", exit)
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Errorf("ledger written despite mode=off (err=%v)", err)
	}
}

// readGuardStopRows reads a ledger file back into typed rows for assertions.
func readGuardStopRows(t *testing.T, path string) []guardStopRecord {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	var out []guardStopRecord
	for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		var r guardStopRecord
		if err := json.Unmarshal([]byte(ln), &r); err != nil {
			t.Fatalf("decode row %q: %v", ln, err)
		}
		out = append(out, r)
	}
	return out
}

func TestGuardStopsDefaultLedgerStaysInIgnoredRuntimeState(t *testing.T) {
	if got, want := filepath.ToSlash(guardStopsLedgerDefaultRel), ".fak/guard-stops.jsonl"; got != want {
		t.Fatalf("default guard-stop ledger = %q, want %q", got, want)
	}
}
