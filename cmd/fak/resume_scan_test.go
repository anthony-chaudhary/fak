package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// runResumeScanAt drives the scan core directly (the `scan` subcommand dispatch is wired
// separately in resume.go) and returns stdout, stderr, and the exit code.
func runResumeScanAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runResumeScan(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// crashedFixture is a real-shaped Claude Code session that died on a rate limit: a genuine
// 250k-token model turn, then the SYNTHETIC rate-limit refusal (model "<synthetic>",
// isApiErrorMessage, all-zero usage), then bookkeeping. The resident size must come from the
// real turn, never the zero-usage refusal — the bug this verb fixes.
const crashedFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":230000,"cache_creation_input_tokens":16000,"output_tokens":500}}}
{"type":"assistant","timestamp":"2020-01-01T00:00:06Z","isApiErrorMessage":true,"message":{"role":"assistant","model":"<synthetic>","content":[{"type":"text","text":"You've hit your session limit · resets 8pm (America/Los_Angeles)"}],"usage":{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}
{"type":"system","timestamp":"2020-01-01T00:00:06Z","content":"bookkeeping"}
`

// cleanFixture ends on a real model turn — a clean enough end, never flagged for restart.
const cleanFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"hi"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":56000,"cache_creation_input_tokens":0,"output_tokens":200}}}
`

// otherFixture ends on a NON-rate api-error — an unclean end, but not a rate-limit one.
const otherFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"hi"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":66000,"cache_creation_input_tokens":0,"output_tokens":200}}}
{"type":"assistant","timestamp":"2020-01-01T00:00:06Z","isApiErrorMessage":true,"message":{"role":"assistant","model":"<synthetic>","content":[{"type":"text","text":"API Error: socket hang up"}],"usage":{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}
`

// writeStore lays out a fixture store and returns its directory.
func writeStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range map[string]string{
		"crashed0.jsonl": crashedFixture,
		"cleanab1.jsonl": cleanFixture,
		"othererr.jsonl": otherFixture,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestResumeScanFlagsRateLimitCrash is the goal in one test: scan a store, and the one session
// that crashed on a rate limit and never resumed is flagged with a managed CUT restart, sized
// from its real model turn (250k) — NOT the zero-usage synthetic refusal.
func TestResumeScanFlagsRateLimitCrash(t *testing.T) {
	dir := writeStore(t)
	out, errb, code := runResumeScanAt("--store", dir)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "1 crashed on a rate limit") {
		t.Errorf("want exactly 1 rate-limit crash counted:\n%s", out)
	}
	for _, want := range []string{"crashed0", "session_limit", "cut", "250000"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	// The clean session is not a restart candidate, so it is not in the default table.
	if strings.Contains(out, "cleanab1") {
		t.Errorf("clean session should not be flagged for restart:\n%s", out)
	}
	// The operator-facing reset hint is surfaced from the refusal text.
	if !strings.Contains(out, "8pm") {
		t.Errorf("output missing the reset hint:\n%s", out)
	}
}

// TestResumeScanJSON: --json emits per-session diagnoses; the crashed one needs a restart, is
// sized from the real turn, and carries the closed session-limit reason.
func TestResumeScanJSON(t *testing.T) {
	dir := writeStore(t)
	out, errb, code := runResumeScanAt("--store", dir, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rows []scanRow
	if err := json.Unmarshal([]byte(out), &rows); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if len(rows) != 3 {
		t.Fatalf("want 3 diagnosed sessions, got %d", len(rows))
	}
	var crashed *scanRow
	for i := range rows {
		if rows[i].SessionID == "crashed0" {
			crashed = &rows[i]
		}
	}
	if crashed == nil {
		t.Fatalf("crashed session missing from scan rows:\n%s", out)
	}
	d := crashed.Diagnosis
	if !d.NeedsRestart || d.Crash != resume.CrashRateLimit || d.LimitReason != resume.LimitSession {
		t.Errorf("crashed diagnosis = needsRestart=%v crash=%q reason=%q, want true/rate_limit/session_limit", d.NeedsRestart, d.Crash, d.LimitReason)
	}
	if d.ResidentTokens != 250000 {
		t.Errorf("resident = %d, want 250000 (from the real turn, not the synthetic refusal)", d.ResidentTokens)
	}
	if d.Plan.Recommended != resume.StrategyCut {
		t.Errorf("recommended = %q, want cut", d.Plan.Recommended)
	}
}

// TestResumeScanAllShowsOtherError: --all also surfaces the non-rate unclean end, classified as
// other_error (and never as a rate-limit restart).
func TestResumeScanAllShowsOtherError(t *testing.T) {
	dir := writeStore(t)
	out, _, code := runResumeScanAt("--store", dir, "--all")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out, "othererr") || !strings.Contains(out, "other_error") {
		t.Errorf("--all should surface the non-rate unclean end:\n%s", out)
	}
}

// TestResumeScanUsageErrors covers the exit-2 / exit-1 paths: no --store, bad TTL, and a store
// directory that does not exist.
func TestResumeScanUsageErrors(t *testing.T) {
	if _, _, code := runResumeScanAt(); code != 2 {
		t.Errorf("no --store: exit = %d, want 2", code)
	}
	if _, _, code := runResumeScanAt("--store", t.TempDir(), "--ttl", "7d"); code != 2 {
		t.Errorf("bad --ttl: exit = %d, want 2", code)
	}
	if _, _, code := runResumeScanAt("--store", filepath.Join(t.TempDir(), "nope")); code != 1 {
		t.Errorf("missing store dir: exit = %d, want 1", code)
	}
}

// TestResumeScanEmptyStore: a store with no .jsonl transcripts is a runtime error (exit 1).
func TestResumeScanEmptyStore(t *testing.T) {
	if _, _, code := runResumeScanAt("--store", t.TempDir()); code != 1 {
		t.Errorf("empty store: exit = %d, want 1", code)
	}
}

// TestResumeScanRealStore is the operational run, gated on FAK_RESUME_SCAN_STORE so it never
// fires in CI. Point it at a real Claude Code project transcript directory to print the live
// rate-limited crashes and their managed-cache restart plans.
func TestResumeScanRealStore(t *testing.T) {
	store := os.Getenv("FAK_RESUME_SCAN_STORE")
	if store == "" {
		t.Skip("set FAK_RESUME_SCAN_STORE=<claude project transcript dir> to run the live scan")
	}
	out, errb, code := runResumeScanAt("--store", store)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	t.Logf("\n%s", out)
}
