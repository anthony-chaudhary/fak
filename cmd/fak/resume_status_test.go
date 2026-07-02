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

// runResumeStatusAt drives the status core directly (the `status` subcommand dispatch is
// wired in resume.go) and returns stdout, stderr, and the exit code.
func runResumeStatusAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runResumeStatus(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// statusCrashedFixture crashed on a session limit and never resumed: a real 250k model
// turn, then the SYNTHETIC rate-limit refusal (the terminal turn). With no ledger it reads
// PENDING — crashed, no resume fired yet.
const statusCrashedFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":230000,"cache_creation_input_tokens":16000,"output_tokens":500}}}
{"type":"assistant","timestamp":"2020-01-01T00:00:06Z","isApiErrorMessage":true,"message":{"role":"assistant","model":"<synthetic>","content":[{"type":"text","text":"You've hit your session limit · resets 8pm (America/Los_Angeles)"}],"usage":{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}
`

// statusTookFixture is a session whose last turn is a clean real model turn AFTER a resume
// fired (a ledger launch precedes the final turn's timestamp) — it TOOK.
const statusTookFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","isApiErrorMessage":true,"message":{"role":"assistant","model":"<synthetic>","content":[{"type":"text","text":"You've hit your session limit · resets 8pm"}],"usage":{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}
{"type":"assistant","timestamp":"2020-01-01T01:00:00Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"back and progressing"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":90000,"cache_creation_input_tokens":0,"output_tokens":300}}}
{"type":"user","timestamp":"2020-01-01T01:00:10Z","message":{"role":"user","content":"keep going"}}
`

// writeStatusStore lays out the fixture store and a ledger recording that the "took"
// session was launched (before its final clean turn). Returns the store dir and ledger path.
func writeStatusStore(t *testing.T) (store, ledger string) {
	t.Helper()
	store = t.TempDir()
	for name, body := range map[string]string{
		"crashed0.jsonl": statusCrashedFixture,
		"tookabc1.jsonl": statusTookFixture,
	} {
		if err := os.WriteFile(filepath.Join(store, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ledger = filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	// A launch for tookabc1 at 00:30, BEFORE its 01:00 clean turn → that turn is new-after-launch.
	row := `{"ts":"2020-01-01T00:30:00Z","session":"tookabc1","phase":"launched","action":"auto_resume"}` + "\n"
	if err := os.WriteFile(ledger, []byte(row), 0o644); err != nil {
		t.Fatal(err)
	}
	return store, ledger
}

// TestResumeStatusWired is the wiring proof: `fak resume status` dispatches through
// runResume (not just the direct core) and folds each session's resume state. The crashed
// session with no ledger reads pending; the launched-then-clean session reads took.
func TestResumeStatusWired(t *testing.T) {
	store, ledger := writeStatusStore(t)

	// Drive through the top-level runResume switch to prove the subcommand is reachable.
	var out, errb bytes.Buffer
	if code := runResume(&out, &errb, []string{"status", "--store", store, "--ledger", ledger}); code != 0 {
		t.Fatalf("runResume status exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "pending") || !strings.Contains(got, "took") {
		t.Errorf("status table missing pending/took states:\n%s", got)
	}
	if !strings.Contains(got, "crashed0") || !strings.Contains(got, "tookabc1") {
		t.Errorf("status table missing the fixture sessions:\n%s", got)
	}
	// The crashed pending session is idle years past any reset window, so the runbook says
	// run and emits the exact resolve+resume command (self-healing -wait form: resolve
	// sleeps out an imminent owner reset on its own).
	if !strings.Contains(got, "run now") || !strings.Contains(got, "fak resume resolve -wait crashed0") {
		t.Errorf("runbook should list the fire command for the crashed session:\n%s", got)
	}
}

// TestResumeStatusUnknownSubcommandListsStatus proves the dispatch error names status, so an
// agent that fat-fingers the verb is told it exists.
func TestResumeStatusUnknownSubcommandListsStatus(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runResume(&out, &errb, []string{"bogus"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2", code)
	}
	if !strings.Contains(errb.String(), "status") {
		t.Errorf("unknown-subcommand error should list status:\n%s", errb.String())
	}
}

// TestResumeStatusJSON: --json emits the per-session fold, with the closed schema tag and
// the folded resume_state each session reached.
func TestResumeStatusJSON(t *testing.T) {
	store, ledger := writeStatusStore(t)
	out, errb, code := runResumeStatusAt("--store", store, "--ledger", ledger, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var doc struct {
		Schema       string `json:"schema"`
		HostAdmitted bool   `json:"host_admitted"`
		Sessions     []struct {
			SessionID  string             `json:"session_id"`
			State      resume.ResumeState `json:"resume_state"`
			NewTurns   int                `json:"new_turns_since_resume"`
			NextAction resume.NextAction  `json:"next_action"`
			Fire       bool               `json:"fire"`
			Command    string             `json:"command"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if doc.Schema != "fak.resume-status.v1" {
		t.Errorf("schema = %q, want fak.resume-status.v1", doc.Schema)
	}
	if !doc.HostAdmitted {
		t.Errorf("host_admitted = false, want true (no source policy configured in the test)")
	}
	var sawCrashed, sawTook bool
	for _, s := range doc.Sessions {
		switch s.SessionID {
		case "crashed0":
			sawCrashed = true
			if s.State != resume.ResumePending {
				t.Errorf("crashed0 state = %q, want pending", s.State)
			}
			// Idle years past any reset window + gate open + host admits ⇒ the runbook fires it.
			if s.NextAction != resume.ActRun || !s.Fire {
				t.Errorf("crashed0 next = %q fire=%v, want run/true", s.NextAction, s.Fire)
			}
			if !strings.Contains(s.Command, "fak resume resolve -wait crashed0") {
				t.Errorf("crashed0 command = %q, want the resolve+resume recipe", s.Command)
			}
		case "tookabc1":
			sawTook = true
			if s.State != resume.ResumeTook {
				t.Errorf("tookabc1 state = %q, want took", s.State)
			}
			// A resume that took is blocked from re-firing: no action, no command.
			if s.NextAction != resume.ActDone || s.Fire {
				t.Errorf("tookabc1 next = %q fire=%v, want done/false", s.NextAction, s.Fire)
			}
			if s.NewTurns < 1 {
				t.Errorf("tookabc1 new-turns-since-resume = %d, want >= 1", s.NewTurns)
			}
		}
	}
	if !sawCrashed || !sawTook {
		t.Fatalf("missing fixture sessions in JSON (crashed=%v took=%v):\n%s", sawCrashed, sawTook, out)
	}
}

// TestResumeStatusUsageErrors covers exit-2 (no --store) and exit-1 (missing store dir).
func TestResumeStatusUsageErrors(t *testing.T) {
	if _, _, code := runResumeStatusAt(); code != 2 {
		t.Errorf("no --store: exit = %d, want 2", code)
	}
	if _, _, code := runResumeStatusAt("--store", filepath.Join(t.TempDir(), "nope")); code != 1 {
		t.Errorf("missing store dir: exit = %d, want 1", code)
	}
}

// statusInterruptedFixture is the invisible-death shape (the 392f-class crash): a real
// model turn, then a user prompt that never got a reply — no refusal record, no error,
// the process just died. Years idle, so the interrupted floor has long passed.
const statusInterruptedFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":230000,"cache_creation_input_tokens":16000,"output_tokens":500}}}
{"type":"user","timestamp":"2020-01-01T00:01:00Z","message":{"role":"user","content":"go on"}}
{"type":"last-prompt","lastPrompt":"go on"}
`

// statusMetaTailFixture ends on an isMeta user line AFTER a clean model turn — meta lines
// get no reply, so this is a clean end, not an interruption.
const statusMetaTailFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":90000,"cache_creation_input_tokens":0,"output_tokens":300}}}
{"type":"user","timestamp":"2020-01-01T00:00:10Z","isMeta":true,"message":{"role":"user","content":"Caveat: local commands"}}
`

// statusSidechainTailFixture ends on a SIDECHAIN user record after a clean main-chain
// turn — a subagent's record must not mark the main chain as owed a reply.
const statusSidechainTailFixture = `{"type":"user","timestamp":"2020-01-01T00:00:00Z","message":{"role":"user","content":"go"}}
{"type":"assistant","timestamp":"2020-01-01T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":4000,"cache_read_input_tokens":90000,"cache_creation_input_tokens":0,"output_tokens":300}}}
{"type":"user","timestamp":"2020-01-01T00:00:10Z","isSidechain":true,"message":{"role":"user","content":[{"type":"tool_result","content":"ok"}]}}
`

// TestResumeStatusSurfacesInterrupted is the visibility regression for the mid-turn death:
// a session killed between a user turn and the reply used to read "ended cleanly" and
// silently VANISH from the default status report — the operator saw a random error and no
// tool would even name the session. It must now appear, classified interrupted; the meta-
// and sidechain-tailed clean sessions must stay out of the default report.
func TestResumeStatusSurfacesInterrupted(t *testing.T) {
	store := t.TempDir()
	for name, body := range map[string]string{
		"deadmid1.jsonl": statusInterruptedFixture,
		"metatail.jsonl": statusMetaTailFixture,
		"sidetail.jsonl": statusSidechainTailFixture,
	} {
		if err := os.WriteFile(filepath.Join(store, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ledger := filepath.Join(t.TempDir(), "resume_ledger.jsonl")

	out, errb, code := runResumeStatusAt("--store", store, "--ledger", ledger)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "deadmid1") || !strings.Contains(out, "interrupted") {
		t.Errorf("mid-turn death missing from the default report:\n%s", out)
	}
	if strings.Contains(out, "metatail") || strings.Contains(out, "sidetail") {
		t.Errorf("meta/sidechain-tailed clean sessions must not be reported as crashed:\n%s", out)
	}
}

// TestResumeStatusInterruptedRowJSON pins the interrupted row's fold: crashed with no
// limit reason (nothing to wait out), unresumed, and — with the host admitting — the
// runbook verdict is run with the self-healing command attached.
func TestResumeStatusInterruptedRowJSON(t *testing.T) {
	store := t.TempDir()
	if err := os.WriteFile(filepath.Join(store, "deadmid1.jsonl"), []byte(statusInterruptedFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	ledger := filepath.Join(t.TempDir(), "resume_ledger.jsonl")

	out, errb, code := runResumeStatusAt("--store", store, "--ledger", ledger, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rep struct {
		HostAdmitted bool        `json:"host_admitted"`
		Sessions     []statusRow `json:"sessions"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("bad JSON: %v\n%s", err, out)
	}
	if len(rep.Sessions) != 1 {
		t.Fatalf("sessions = %d, want 1", len(rep.Sessions))
	}
	row := rep.Sessions[0]
	if row.Crash != resume.CrashInterrupted || row.LimitReason != "" {
		t.Errorf("crash = (%q,%q), want (interrupted, no limit reason)", row.Crash, row.LimitReason)
	}
	if rep.HostAdmitted {
		if row.NextAction != resume.ActRun || !strings.Contains(row.Command, "resolve -wait deadmid1") {
			t.Errorf("admitted host: next = %q cmd = %q, want run with the -wait resolve command", row.NextAction, row.Command)
		}
	} else if row.NextAction != resume.ActHoldAdmission {
		t.Errorf("refused host: next = %q, want hold_admission", row.NextAction)
	}
}
