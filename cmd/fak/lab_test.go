package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

// TestLabEmbeddedRosterValid is the load-bearing guard on the shipped default: the
// embedded generic roster must always parse and validate, so `fak lab` has a working
// fleet with zero setup. It also asserts the roster stays GENERIC — no box id, class,
// or group may look like a real lab host/channel (a dotted hostname or a Slack-style
// channel id), which would breach the public/private boundary.
func TestLabEmbeddedRosterValid(t *testing.T) {
	ro, err := fleet.LoadRoster(bytes.NewReader(labDefaultRosterJSON))
	if err != nil {
		t.Fatalf("embedded roster does not parse: %v", err)
	}
	if probs := ro.Validate(); len(probs) != 0 {
		t.Fatalf("embedded roster does not validate: %v", probs)
	}
	if len(ro.Boxes) == 0 {
		t.Fatal("embedded roster has no boxes")
	}
	for _, b := range ro.Boxes {
		for field, v := range map[string]string{"id": b.ID, "class": b.Class, "group": b.Group, "endpoint": b.Endpoint} {
			if strings.Contains(v, ".") {
				t.Fatalf("box %q %s %q looks like a hostname — the roster must stay generic", b.ID, field, v)
			}
			// A Slack channel id is an uppercase C-prefixed token; a generic roster never carries one.
			if len(v) >= 9 && v[0] == 'C' && v == strings.ToUpper(v) {
				t.Fatalf("box %q %s %q looks like a channel id — the roster must stay generic", b.ID, field, v)
			}
		}
	}
}

// TestLabReportThenStatusClosesLoop is the end-to-end witness: `fak lab report`
// writes a self-report the next `fak lab status --json` reads back as reachable, with
// no private bridge. This is the public producer half closing the loop for one box.
func TestLabReportThenStatusClosesLoop(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)

	// Self-report one of the embedded boxes.
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "da-cpu", "--state", "live", "--version", "0.31.0"}); rc != 0 {
		t.Fatalf("lab report exited %d, want 0", rc)
	}
	if _, err := os.Stat(filepath.Join(dir, "da-cpu.json")); err != nil {
		t.Fatalf("report file not written: %v", err)
	}

	// status --json must now fold that box as reachable.
	var out bytes.Buffer
	if rc := runLab(&out, io.Discard, []string{"status", "--json"}); rc != 0 {
		t.Fatalf("lab status exited %d, want 0", rc)
	}
	var snap fleet.Snapshot
	if err := json.Unmarshal(out.Bytes(), &snap); err != nil {
		t.Fatalf("status --json did not emit a snapshot: %v\n%s", err, out.String())
	}
	if snap.Reachable != 1 {
		t.Fatalf("reachable = %d, want 1 (the one self-reported box)", snap.Reachable)
	}
	var found bool
	for _, r := range snap.Rows {
		if r.ID == "da-cpu" {
			found = true
			if r.State != fleet.StateLive || r.Version != "0.31.0" {
				t.Fatalf("da-cpu row = %+v, want live 0.31.0", r)
			}
		}
	}
	if !found {
		t.Fatal("da-cpu not present in the folded snapshot rows")
	}
}

// TestLabStatusHonestDegrade: with an empty/missing reports dir, status exits 0 (NOT a
// failure), every box reads unknown, and the output tells the operator how to populate
// liveness — it must never read as a confirmed fleet-wide outage.
func TestLabStatusHonestDegrade(t *testing.T) {
	dir := t.TempDir() // exists but empty -> no live reports
	t.Setenv("FAK_FLEET_REPORTS", dir)

	var out bytes.Buffer
	if rc := runLab(&out, io.Discard, []string{"status"}); rc != 0 {
		t.Fatalf("status with no reports should exit 0 (honest degrade), got %d", rc)
	}
	s := out.String()
	if !strings.Contains(s, "no live reports") {
		t.Fatalf("missing the honest-degrade hint:\n%s", s)
	}
	if !strings.Contains(s, "fak lab report") {
		t.Fatalf("the hint should point at `fak lab report`:\n%s", s)
	}
}

// TestLabReportRejectsBadInput: an unknown state and an escaping id are refused at the
// CLI boundary (the producer fails closed).
func TestLabReportRejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "x", "--state", "bogus"}); rc == 0 {
		t.Fatal("an unknown --state must be refused")
	}
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "../evil", "--state", "live"}); rc == 0 {
		t.Fatal("an escaping --id must be refused")
	}
}

func TestLabReadinessMissingFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.json")
	t.Setenv("FAK_LAB_READINESS", path)

	var out bytes.Buffer
	if rc := runLab(&out, io.Discard, []string{"readiness", "--json"}); rc != 1 {
		t.Fatalf("missing readiness should fail closed with exit 1, got %d", rc)
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness --json did not emit a record: %v\n%s", err, out.String())
	}
	if rec.Status != fleet.LabIndeterminate || rec.AdmitLabDispatch {
		t.Fatalf("missing readiness = %+v, want INDETERMINATE and not admitting", rec)
	}
	if rec.NextAction != "publish-lab-readiness" || rec.Evidence != "no-readiness-record" {
		t.Fatalf("missing readiness action/evidence = %+v", rec)
	}
	if rec.Commands == nil || !strings.Contains(rec.Commands.MarkReady, "--write-default") {
		t.Fatalf("missing readiness should include public mark-ready command hints, got %+v", rec.Commands)
	}
}

func TestLabReadinessWriteThenRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{
		"readiness",
		"--status", fleet.LabReadyForDevWork,
		"--write", path,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("writing READY_FOR_DEV_WORK should exit 0, got %d: %s", rc, out.String())
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("readiness record not written: %v", err)
	}

	out.Reset()
	rc = runLab(&out, io.Discard, []string{"readiness", "--file", path, "--json"})
	if rc != 0 {
		t.Fatalf("reading READY_FOR_DEV_WORK should exit 0, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness read did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Status != fleet.LabReadyForDevWork || !rec.AdmitLabDispatch {
		t.Fatalf("readiness read = %+v, want READY_FOR_DEV_WORK admitting", rec)
	}
}

func TestLabReadinessWriteDefaultUsesConfiguredPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	t.Setenv("FAK_LAB_READINESS", path)

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{
		"readiness",
		"--status", fleet.LabWaitPrivateRecover,
		"--write-default",
		"--json",
	})
	if rc != 1 {
		t.Fatalf("WAIT_PRIVATE_RECOVERY should write but fail closed with exit 1, got %d", rc)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("default readiness record not written: %v", err)
	}

	out.Reset()
	rc = runLab(&out, io.Discard, []string{"readiness", "--json"})
	if rc != 1 {
		t.Fatalf("reading WAIT_PRIVATE_RECOVERY should fail closed with exit 1, got %d", rc)
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness read did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Status != fleet.LabWaitPrivateRecover || rec.AdmitLabDispatch {
		t.Fatalf("readiness read = %+v, want WAIT_PRIVATE_RECOVERY and not admitting", rec)
	}
}

func TestLabReadinessWriteDefaultRejectsConflictingWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{
		"readiness",
		"--status", fleet.LabReadyForDevWork,
		"--write", path,
		"--write-default",
	})
	if rc != 2 {
		t.Fatalf("conflicting write flags should exit 2, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "only one") {
		t.Fatalf("conflict error should explain the flag problem, got:\n%s", stderr.String())
	}
}
