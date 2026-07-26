package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scmbridge"
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

const bridgeSpecDoc = `{
  "schema": "fak.service.v1",
  "identity": {"node": "lab-1", "service": "fak-guard", "workload": "FakGuardControl"},
  "kind": "service",
  "desired": "desired-running",
  "command": ["C:\\opt\\fak\\fak.exe", "service", "windows-run"]
}`

func writeBridgeSpec(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "spec.json")
	if err := os.WriteFile(p, []byte(bridgeSpecDoc), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestServiceBridgeProjectsMachineRole(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runService(&out, &errb, []string{"bridge", "--spec", writeBridgeSpec(t), "--role", "machine", "--sha256", "abcd", "--json"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	var got bridgeOutput
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	p := got.Projection
	if p.Manager != scmbridge.ManagerSCM || p.Principal != scmbridge.MachinePrincipal || p.UnitName != "FakGuardControl" {
		t.Fatalf("projection = %+v", p)
	}
	if len(p.Recovery) != 3 || p.Recovery[0].DelayMS != 1000 {
		t.Fatalf("recovery = %+v", p.Recovery)
	}
}

func TestServiceBridgeReconcileFlagsDivergenceWithExit4(t *testing.T) {
	spec := writeBridgeSpec(t)
	obs := scmbridge.Observed{
		Present:   true,
		Manager:   scmbridge.ManagerSCM,
		UnitName:  "FakGuardControl",
		Principal: "LocalSystem", // diverges: contract demands LocalService
		Command:   []string{`C:\opt\fak\fak.exe service windows-run`},
		Status:    "running",
		PID:       77,
	}
	// The single-string native command line equals the joined spec command.
	obsPath := filepath.Join(t.TempDir(), "observed.json")
	b, _ := json.Marshal(obs)
	if err := os.WriteFile(obsPath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := runService(&out, &errb, []string{"bridge", "--spec", spec, "--role", "machine", "--observed", obsPath})
	if rc != 4 {
		t.Fatalf("rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "diverged principal") {
		t.Fatalf("output = %s", out.String())
	}
}

func TestServiceBridgeInSyncRecordsPhaseInLedger(t *testing.T) {
	spec := writeBridgeSpec(t)
	dir := t.TempDir()
	obs := scmbridge.Observed{
		Present:     true,
		Manager:     scmbridge.ManagerSCM,
		UnitName:    "FakGuardControl",
		Principal:   "NT AUTHORITY\\LocalService",
		Command:     []string{`C:\opt\fak\fak.exe service windows-run`},
		StartOnBoot: true,
		Status:      "running",
		PID:         77,
	}
	obsPath := filepath.Join(dir, "observed.json")
	b, _ := json.Marshal(obs)
	if err := os.WriteFile(obsPath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	ledgerDir := filepath.Join(dir, "ledger")
	old := bridgeNowMS
	bridgeNowMS = func() int64 { return 123456 }
	t.Cleanup(func() { bridgeNowMS = old })
	var out, errb bytes.Buffer
	rc := runService(&out, &errb, []string{"bridge", "--spec", spec, "--role", "machine", "--observed", obsPath, "--ledger-dir", ledgerDir, "--json"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	var got bridgeOutput
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Report.InSync || got.Phase != servicespec.PhaseReady {
		t.Fatalf("output = %+v", got)
	}
	led, err := serviceledger.Open(ledgerDir)
	if err != nil {
		t.Fatal(err)
	}
	evs := led.Events()
	if len(evs) != 1 || evs[0].Type != serviceledger.EventReadiness || evs[0].Phase != servicespec.PhaseReady || evs[0].Correlation.PID != 77 {
		t.Fatalf("ledger rows = %+v", evs)
	}
}

func TestServiceBridgeJudgeRefereesTheCapturedLedger(t *testing.T) {
	dir := t.TempDir()
	led, err := serviceledger.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	id := servicespec.Identity{Node: "lab-1", Service: "fak-guard"}
	rows := []serviceledger.Event{
		{Type: serviceledger.EventProcessExit, AtUnixMS: 100, Source: serviceledger.SourceWindowsEventLog,
			SourceUID: "System/1", Identity: id,
			Exit: &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: 100}},
		{Type: serviceledger.EventManagerRestart, AtUnixMS: 200, Source: serviceledger.SourceWindowsEventLog,
			SourceUID: "System/2", Identity: id},
	}
	if _, _, err := led.AppendAll(rows); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := runService(&out, &errb, []string{"bridge", "--judge", "scm-process-kill", "--ledger-dir", dir, "--json"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	var v scmbridge.ProbeVerdict
	if err := json.Unmarshal(out.Bytes(), &v); err != nil {
		t.Fatal(err)
	}
	if !v.Corroborated || !v.Resumed {
		t.Fatalf("verdict = %+v", v)
	}
	// The termservice probe is NOT satisfied by this ledger: no resume row
	// carrying the resumed session identity.
	out.Reset()
	rc = runService(&out, &errb, []string{"bridge", "--judge", "termservice-reset", "--ledger-dir", dir})
	if rc != 4 || !strings.Contains(out.String(), "missing=") {
		t.Fatalf("rc=%d stdout=%s", rc, out.String())
	}
	// Unknown probe names are refused, not guessed.
	if rc := runService(&out, &errb, []string{"bridge", "--judge", "chaos-monkey", "--ledger-dir", dir}); rc != 2 {
		t.Fatalf("unknown probe rc=%d", rc)
	}
}

func TestServiceBridgeUsageAndPlacementRefusals(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runService(&out, &errb, []string{"bridge"}); rc != 2 {
		t.Fatalf("bare bridge rc=%d", rc)
	}
	// Broker without a principal: the placement contract refuses.
	rc := runService(&out, &errb, []string{"bridge", "--spec", writeBridgeSpec(t), "--role", "broker"})
	if rc != 1 || !strings.Contains(errb.String(), "principal") {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
}
