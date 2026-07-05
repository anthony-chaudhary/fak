package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These pin the operator WRITER half of the drive-state alignment: `fak resume hold` records a
// durable, UUID-keyed hold the watchdog reads, `--list` shows it, and `fak resume release`
// reverses a reversible hold. The reader (rwLoadDriveStates) and the watchdog guard share this
// same store, so a hold written here is honored by TestResumeWatchdogOperatorHoldDoesNotSpawn.

func TestResumeHoldWritesListsAndValidates(t *testing.T) {
	regDir := t.TempDir()
	sid := "sid-verb-1234567890"

	var out, errb bytes.Buffer
	if rc := runResumeHold(&out, &errb, []string{"--reg-dir", regDir, "--state", "stopped", "--reason", "superseded", sid}); rc != 0 {
		t.Fatalf("hold rc=%d stderr=%s", rc, errb.String())
	}
	body, err := os.ReadFile(filepath.Join(regDir, "resume_drivestate.jsonl"))
	if err != nil {
		t.Fatalf("read drive-state store: %v", err)
	}
	for _, want := range []string{`"session":"` + sid + `"`, `"state":"stopped"`, `"via":"fak resume hold"`, `"reason":"superseded"`} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("drive-state row missing %s:\n%s", want, body)
		}
	}
	// The watchdog reader folds the same store back to a hold.
	if !rwLoadDriveStates(regDir)[sid].HeldByOperator() {
		t.Fatalf("held session not read back as a hold: %v", rwLoadDriveStates(regDir))
	}

	// --list surfaces the held session.
	out.Reset()
	errb.Reset()
	if rc := runResumeHold(&out, &errb, []string{"--reg-dir", regDir, "--list"}); rc != 0 {
		t.Fatalf("hold --list rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "stopped") {
		t.Fatalf("--list missing the held session:\n%s", out.String())
	}

	// A bad --state is a usage error, not a silent no-op.
	out.Reset()
	errb.Reset()
	if rc := runResumeHold(&out, &errb, []string{"--reg-dir", regDir, "--state", "banana", sid}); rc != 2 {
		t.Fatalf("bad --state rc=%d, want 2", rc)
	}

	// release needs a session id.
	out.Reset()
	errb.Reset()
	if rc := runResumeRelease(&out, &errb, []string{"--reg-dir", regDir}); rc != 2 {
		t.Fatalf("release without a sid rc=%d, want 2", rc)
	}
}

func TestResumeReleaseLiftsAReversibleHold(t *testing.T) {
	regDir := t.TempDir()
	sid := "sid-rev-1234567890"

	var out, errb bytes.Buffer
	if rc := runResumeHold(&out, &errb, []string{"--reg-dir", regDir, "--state", "paused", sid}); rc != 0 {
		t.Fatalf("hold rc=%d stderr=%s", rc, errb.String())
	}
	if !rwLoadDriveStates(regDir)[sid].HeldByOperator() {
		t.Fatal("precondition: a paused session should be held")
	}

	out.Reset()
	errb.Reset()
	if rc := runResumeRelease(&out, &errb, []string{"--reg-dir", regDir, sid}); rc != 0 {
		t.Fatalf("release rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if rwLoadDriveStates(regDir)[sid].HeldByOperator() {
		t.Fatalf("released session is still held: %v", rwLoadDriveStates(regDir))
	}
	if !strings.Contains(out.String(), "released") {
		t.Fatalf("release output missing confirmation:\n%s", out.String())
	}
}
