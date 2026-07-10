package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// selfObsJSON is the observation slice of the `fak resume self --json` envelope.
type selfObsJSON struct {
	Observation struct {
		Session      string `json:"session"`
		HasHistory   bool   `json:"has_history"`
		Attempts     int    `json:"attempts"`
		NewTurns     int    `json:"new_turns_since_resume"`
		Outcome      string `json:"outcome"`
		State        string `json:"resume_state"`
		RetryBlocked bool   `json:"retry_blocked"`
		RetryReason  string `json:"retry_reason"`
		EarnedBudget int    `json:"earned_budget"`
		NextHint     string `json:"next_hint"`
	} `json:"observation"`
}

// TestResumeSelfTook drives `fak resume self` through the top-level dispatch on the launched-
// then-clean session: it must read its OWN history as a resume that TOOK and report the gate
// burned — the same verdict the operator `resume status` table gives, from the worker's seat.
func TestResumeSelfTook(t *testing.T) {
	store, ledger := writeStatusStore(t)

	// Human readout via the runResume switch (proves the subcommand is wired).
	var out, errb bytes.Buffer
	if code := runResume(&out, &errb, []string{"self", "--session", "tookabc1", "--store", store, "--ledger", ledger}); code != 0 {
		t.Fatalf("runResume self exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "state=took") {
		t.Errorf("self readout should show state=took:\n%s", got)
	}
	if !strings.Contains(got, "BLOCKED") {
		t.Errorf("a resume that took burns the gate — readout should show BLOCKED:\n%s", got)
	}

	// JSON path: field-level proof of the fold.
	var jb, je bytes.Buffer
	if code := runResumeSelf(&jb, &je, []string{"--session", "tookabc1", "--store", store, "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("self --json exit = %d, want 0 (stderr: %s)", code, je.String())
	}
	var doc selfObsJSON
	if err := json.Unmarshal(jb.Bytes(), &doc); err != nil {
		t.Fatalf("decode self json: %v\n%s", err, jb.String())
	}
	o := doc.Observation
	if o.Session != "tookabc1" || !o.HasHistory || o.State != "took" || !o.RetryBlocked || o.Attempts != 1 {
		t.Errorf("self observation wrong: %+v", o)
	}
	if o.NewTurns < 1 {
		t.Errorf("a resume that took must witness >=1 post-launch turn, got %d", o.NewTurns)
	}
}

// TestResumeSelfNoHistoryFailClosed proves the fail-closed floor: a crashed session with NO
// ledger row reads pending with the "nothing to recover" hint — never a fabricated took.
func TestResumeSelfNoHistoryFailClosed(t *testing.T) {
	store, ledger := writeStatusStore(t)

	var jb, je bytes.Buffer
	if code := runResumeSelf(&jb, &je, []string{"--session", "crashed0", "--store", store, "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("self --json exit = %d, want 0 (stderr: %s)", code, je.String())
	}
	var doc selfObsJSON
	if err := json.Unmarshal(jb.Bytes(), &doc); err != nil {
		t.Fatalf("decode self json: %v\n%s", err, jb.String())
	}
	o := doc.Observation
	if o.HasHistory || o.State != "pending" || o.RetryBlocked || o.Attempts != 0 {
		t.Errorf("no-history session should be the fail-closed floor, got %+v", o)
	}
	if !strings.Contains(o.NextHint, "nothing to recover") {
		t.Errorf("floor hint should say nothing to recover, got %q", o.NextHint)
	}
}

// TestResumeSelfDefaultsToNewestTranscript proves --session is optional: with only --store the
// worker gets "the session I am in" (the newest transcript). writeStatusStore writes tookabc1
// last-ish; either fixture is a valid pick, so we only assert a session was resolved and folded.
func TestResumeSelfDefaultsToNewestTranscript(t *testing.T) {
	store, ledger := writeStatusStore(t)
	var jb, je bytes.Buffer
	if code := runResumeSelf(&jb, &je, []string{"--store", store, "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("self --json (no --session) exit = %d, want 0 (stderr: %s)", code, je.String())
	}
	var doc selfObsJSON
	if err := json.Unmarshal(jb.Bytes(), &doc); err != nil {
		t.Fatalf("decode self json: %v\n%s", err, jb.String())
	}
	if doc.Observation.Session == "" {
		t.Errorf("expected a resolved session from --store, got empty")
	}
}

// TestResumeSelfNeedsATarget proves the usage guard: no --session and no --store is a usage
// error (exit 2) that names what is missing.
func TestResumeSelfNeedsATarget(t *testing.T) {
	// Clear the ambient session id so the guard, not a stray env session, decides.
	t.Setenv("CLAUDE_SESSION_ID", "")
	var out, errb bytes.Buffer
	code := runResumeSelf(&out, &errb, []string{"--ledger", "does-not-exist.jsonl"})
	if code != 2 {
		t.Fatalf("no target exit = %d, want 2 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "need --session") {
		t.Errorf("usage error should name the missing target:\n%s", errb.String())
	}
}
