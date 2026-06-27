package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runResumeAt drives the testable core and returns stdout, stderr, and the exit code.
func runResumeAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runResume(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// TestResumePlanColdHeadline is the CLI half of the goal: the 250k / idle-2h case prints a
// COLD posture and recommends CUT.
func TestResumePlanColdHeadline(t *testing.T) {
	out, errb, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "7200")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "posture=COLD") {
		t.Errorf("output missing COLD posture:\n%s", out)
	}
	if !strings.Contains(out, "recommended: CUT") {
		t.Errorf("output missing CUT recommendation:\n%s", out)
	}
}

// TestResumePlanJSON: --json emits a parseable Report whose recommendation is cut on the
// cold headline case.
func TestResumePlanJSON(t *testing.T) {
	out, errb, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "7200", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rep struct {
		Posture     string `json:"posture"`
		Recommended string `json:"recommended"`
		Reason      string `json:"reason"`
		Strategies  []struct {
			Strategy string `json:"strategy"`
		} `json:"strategies"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if rep.Posture != "cold" || rep.Recommended != "cut" || rep.Reason != "cold_prefill_shed" {
		t.Errorf("got posture=%q recommended=%q reason=%q, want cold/cut/cold_prefill_shed", rep.Posture, rep.Recommended, rep.Reason)
	}
	if len(rep.Strategies) != 3 {
		t.Errorf("want 3 strategies, got %d", len(rep.Strategies))
	}
}

// TestResumePlanWarmKeepsFull: idle within the TTL with a short horizon recommends keeping
// the whole transcript.
func TestResumePlanWarmKeepsFull(t *testing.T) {
	out, _, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "60", "--horizon", "3")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out, "posture=WARM") || !strings.Contains(out, "recommended: RESUME_FULL") {
		t.Errorf("warm short-horizon case should keep full:\n%s", out)
	}
}

// TestResumeUsageErrors covers the exit-2 paths: no subcommand, unknown subcommand, bad TTL,
// and a missing resident size.
func TestResumeUsageErrors(t *testing.T) {
	cases := [][]string{
		{},             // no subcommand
		{"frobnicate"}, // unknown subcommand
		{"plan", "--resident-tokens", "100", "--ttl", "7d"}, // bad TTL
		{"plan", "--idle-seconds", "10"},                    // no resident size
	}
	for _, argv := range cases {
		if _, _, code := runResumeAt(argv...); code != 2 {
			t.Errorf("argv %v: exit = %d, want 2", argv, code)
		}
	}
}

// claudeTranscriptFixture is a real-shaped Claude Code session: the last assistant turn
// reports a 250k-token prompt (4000 uncached + 230000 cache-read + 16000 cache-creation),
// which is the resident context a resume would re-prefill.
const claudeTranscriptFixture = `{"type":"user","timestamp":"2026-06-26T10:00:00Z","message":{"role":"user","content":"start"}}
{"type":"assistant","timestamp":"2026-06-26T10:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":1200,"cache_read_input_tokens":0,"cache_creation_input_tokens":1200,"output_tokens":300}}}
{"type":"assistant","timestamp":"2026-06-26T10:05:09Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":4000,"cache_read_input_tokens":230000,"cache_creation_input_tokens":16000,"output_tokens":520}}}
`

// TestResumePlanFromTranscript: --transcript grounds the plan on a real Claude Code session,
// deriving the 250k resident size from the last assistant turn's prompt.
func TestResumePlanFromTranscript(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	if err := os.WriteFile(path, []byte(claudeTranscriptFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	out, errb, code := runResumeAt("plan", "--transcript", path, "--idle-seconds", "7200")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "resident=250000 tok") {
		t.Errorf("did not derive 250000 resident tokens from the last assistant turn:\n%s", out)
	}
	if !strings.Contains(out, "posture=COLD") || !strings.Contains(out, "recommended: CUT") {
		t.Errorf("expected COLD/CUT for a 2h-idle 250k transcript:\n%s", out)
	}
}

// TestResumeTranscriptNoUsage: a transcript with no assistant usage cannot derive a resident
// size and exits 1 (pass --resident-tokens), rather than silently planning a zero session.
func TestResumeTranscriptNoUsage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.jsonl")
	if err := os.WriteFile(path, []byte(`{"type":"user","message":{"role":"user","content":"hi"}}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, code := runResumeAt("plan", "--transcript", path); code != 1 {
		t.Errorf("transcript with no usage: exit = %d, want 1", code)
	}
}

// TestResumeHelp: the help subcommand exits 0 and prints the example.
func TestResumeHelp(t *testing.T) {
	out, _, code := runResumeAt("help")
	if code != 0 {
		t.Fatalf("help exit = %d, want 0", code)
	}
	if !strings.Contains(out, "fak resume plan") {
		t.Errorf("help missing usage:\n%s", out)
	}
	if !strings.Contains(out, "fak resume validate") {
		t.Errorf("help missing the validate verb:\n%s", out)
	}
}

// backtestCorpusFixture is a real-shaped single session with two scorable boundaries: a 5s
// warm pair (the prefix is fully re-served, the projection calls it warm — agree) and a 2h
// cold pair (nothing re-served and the prompt is re-written, the projection calls it cold —
// agree, a confirmed-cold boundary whose write ratio validates the cold-cost premise).
const backtestCorpusFixture = `{"type":"assistant","timestamp":"2026-06-26T10:00:05Z","message":{"role":"assistant","usage":{"input_tokens":1200,"cache_read_input_tokens":0,"cache_creation_input_tokens":1200,"output_tokens":300}}}
{"type":"assistant","timestamp":"2026-06-26T10:00:10Z","message":{"role":"assistant","usage":{"input_tokens":200,"cache_read_input_tokens":2300,"cache_creation_input_tokens":300,"output_tokens":120}}}
{"type":"assistant","timestamp":"2026-06-26T12:00:10Z","message":{"role":"assistant","usage":{"input_tokens":100,"cache_read_input_tokens":0,"cache_creation_input_tokens":2500,"output_tokens":140}}}
`

// TestResumeValidateBacktest: validate scans a corpus of real-shaped transcripts and scores
// the projection — here both boundaries agree (100% accuracy) and the cold pair is confirmed
// cold with a near-total re-write ratio.
func TestResumeValidateBacktest(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "session.jsonl"), []byte(backtestCorpusFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	out, errb, code := runResumeAt("validate", "--corpus", dir)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "accuracy: 100.0%") {
		t.Errorf("want 100%% accuracy on the agreeing fixture:\n%s", out)
	}
	if !strings.Contains(out, "1 confirmed-cold") {
		t.Errorf("want one confirmed-cold boundary:\n%s", out)
	}
}

// TestResumeValidateJSON: --json emits a parseable BacktestReport with the scored counts.
func TestResumeValidateJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.jsonl"), []byte(backtestCorpusFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	out, errb, code := runResumeAt("validate", "--corpus", dir, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rep struct {
		Pairs         int     `json:"pairs"`
		Scored        int     `json:"scored"`
		Agree         int     `json:"agree"`
		Accuracy      float64 `json:"accuracy"`
		ConfirmedCold int     `json:"confirmed_cold"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if rep.Pairs != 2 || rep.Scored != 2 || rep.Agree != 2 || rep.Accuracy != 1.0 || rep.ConfirmedCold != 1 {
		t.Errorf("got %+v, want pairs/scored/agree=2, accuracy=1.0, confirmedCold=1", rep)
	}
}

// largeColdResumeFixture is a session whose FIRST assistant turn is a large cold re-prefill (a
// 30k prompt with zero cache_read), the cross-file resume case within-file gaps under-sample.
const largeColdResumeFixture = `{"type":"assistant","timestamp":"2026-06-26T12:00:00Z","message":{"role":"assistant","usage":{"input_tokens":10000,"cache_read_input_tokens":0,"cache_creation_input_tokens":20000,"output_tokens":200}}}
{"type":"assistant","timestamp":"2026-06-26T12:00:05Z","message":{"role":"assistant","usage":{"input_tokens":50,"cache_read_input_tokens":29900,"cache_creation_input_tokens":100,"output_tokens":150}}}
`

// TestResumeValidateCrossFileInstrument: validate's cross-file instrument flags a large cold
// first turn as a genuine resume re-prefill and reports its write-premium share.
func TestResumeValidateCrossFileInstrument(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "resumed.jsonl"), []byte(largeColdResumeFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	out, errb, code := runResumeAt("validate", "--corpus", dir, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rep struct {
		FirstTurnResumes int `json:"first_turn_resumes"`
		FirstTurnCold    int `json:"first_turn_cold"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, out)
	}
	if rep.FirstTurnResumes != 1 || rep.FirstTurnCold != 1 {
		t.Errorf("got resumes=%d cold=%d, want 1/1", rep.FirstTurnResumes, rep.FirstTurnCold)
	}
	// the human table surfaces the cross-file section too.
	tout, _, _ := runResumeAt("validate", "--corpus", dir)
	if !strings.Contains(tout, "cross-file resume re-prefills") {
		t.Errorf("table missing the cross-file section:\n%s", tout)
	}
}

// TestResumeValidateNeedsCorpus: validate with no --corpus is a usage error (exit 2); an empty
// corpus directory is a runtime error (exit 1).
func TestResumeValidateNeedsCorpus(t *testing.T) {
	if _, _, code := runResumeAt("validate"); code != 2 {
		t.Errorf("validate with no --corpus: exit = %d, want 2", code)
	}
	if _, _, code := runResumeAt("validate", "--corpus", t.TempDir()); code != 1 {
		t.Errorf("validate on an empty corpus: exit = %d, want 1", code)
	}
}
