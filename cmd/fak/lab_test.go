package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/linkstate"
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

func TestLabReportInferenceThenStatusClosesLoop(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)

	if rc := runLab(io.Discard, io.Discard, []string{
		"report",
		"--id", "box-a",
		"--state", "live",
		"--inference", "ready",
		"--engine", "fak",
		"--model", "qwen",
		"--output-tps", "1.75",
		"--probe-latency-ms", "1250",
	}); rc != 0 {
		t.Fatalf("lab report with inference exited %d, want 0", rc)
	}

	var out bytes.Buffer
	if rc := runLab(&out, io.Discard, []string{"status", "--json"}); rc != 0 {
		t.Fatalf("lab status exited %d, want 0", rc)
	}
	var snap fleet.Snapshot
	if err := json.Unmarshal(out.Bytes(), &snap); err != nil {
		t.Fatalf("status --json did not emit a snapshot: %v\n%s", err, out.String())
	}
	if snap.Inference == nil || snap.Inference.Useful != 1 || snap.Inference.Ready != 1 {
		t.Fatalf("inference summary = %+v, want one useful ready box", snap.Inference)
	}
	var found bool
	for _, r := range snap.Rows {
		if r.ID != "box-a" {
			continue
		}
		found = true
		if r.Inference == nil || r.Inference.Status != fleet.InferenceReady || r.Inference.Engine != "fak" || r.Inference.Model != "qwen" || r.Inference.OutputTPS != 1.75 || r.Inference.ProbeLatencyMS != 1250 {
			t.Fatalf("box-a inference row = %+v", r.Inference)
		}
	}
	if !found {
		t.Fatal("box-a not present in the folded snapshot rows")
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
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "x", "--state", "live", "--inference", "private-ok"}); rc == 0 {
		t.Fatal("an unknown --inference status must be refused")
	}
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "x", "--state", "live", "--inference", "ready", "--output-tps", "-1"}); rc == 0 {
		t.Fatal("a negative --output-tps must be refused")
	}
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "x", "--state", "live", "--inference", "ready", "--probe-latency-ms", "-1"}); rc == 0 {
		t.Fatal("a negative --probe-latency-ms must be refused")
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
	if rec.Phase != linkstate.Waiting || rec.Detail != linkstate.DetailIndeterminate || rec.AdmitDispatch {
		t.Fatalf("missing readiness = %+v, want WAITING/indeterminate and not admitting", rec)
	}
	if rec.NextAction != "publish-lab-readiness" || rec.Evidence != "no-readiness-record" {
		t.Fatalf("missing readiness action/evidence = %+v", rec)
	}
	if rec.Commands == nil || !strings.Contains(rec.Commands.MarkClear, "--write-default") {
		t.Fatalf("missing readiness should include public mark-clear command hints, got %+v", rec.Commands)
	}
}

func TestLabReadinessWriteThenRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{
		"readiness",
		"--phase", "CLEAR",
		"--write", path,
		"--json",
	})
	if rc != 0 {
		t.Fatalf("writing CLEAR should exit 0, got %d: %s", rc, out.String())
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("readiness record not written: %v", err)
	}

	out.Reset()
	rc = runLab(&out, io.Discard, []string{"readiness", "--file", path, "--json"})
	if rc != 0 {
		t.Fatalf("reading CLEAR should exit 0, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness read did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Clear || !rec.AdmitDispatch {
		t.Fatalf("readiness read = %+v, want CLEAR admitting", rec)
	}
}

func TestLabReadinessWriteDefaultUsesConfiguredPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	t.Setenv("FAK_LAB_READINESS", path)

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{
		"readiness",
		"--phase", "WAITING",
		"--write-default",
		"--json",
	})
	if rc != 1 {
		t.Fatalf("WAITING should write but fail closed with exit 1, got %d", rc)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("default readiness record not written: %v", err)
	}

	out.Reset()
	rc = runLab(&out, io.Discard, []string{"readiness", "--json"})
	if rc != 1 {
		t.Fatalf("reading WAITING should fail closed with exit 1, got %d", rc)
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness read did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Waiting || rec.AdmitDispatch {
		t.Fatalf("readiness read = %+v, want WAITING and not admitting", rec)
	}
}

func TestLabReadinessWriteDefaultRejectsConflictingWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lab-readiness.json")
	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{
		"readiness",
		"--phase", "CLEAR",
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

func TestLabReadinessFromReportsAdmitsUsefulInference(t *testing.T) {
	reportsDir := t.TempDir()
	readinessPath := filepath.Join(t.TempDir(), "lab-readiness.json")
	t.Setenv("FAK_FLEET_REPORTS", reportsDir)
	t.Setenv("FAK_LAB_READINESS", readinessPath)

	if err := fleet.WriteReport(reportsDir, "box-a", fleet.Report{
		State: fleet.StateLive,
		Inference: &fleet.InferenceStats{
			Status: fleet.InferenceReady,
			Engine: "fak",
			Model:  "glm-5.2",
			Reason: "v1-models",
		},
	}); err != nil {
		t.Fatalf("write report: %v", err)
	}

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"readiness", "--from-reports", "--write-default", "--json"})
	if rc != 0 {
		t.Fatalf("readiness --from-reports should admit useful inference, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness --from-reports did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Clear || !rec.AdmitDispatch {
		t.Fatalf("readiness from useful inference = %+v, want CLEAR admitting", rec)
	}
	if rec.Evidence != "scrubbed-fleet-report" {
		t.Fatalf("evidence = %q, want scrubbed-fleet-report", rec.Evidence)
	}
	if _, err := os.Stat(readinessPath); err != nil {
		t.Fatalf("--write-default did not persist the derived readiness record: %v", err)
	}
	rawPersisted, err := os.ReadFile(readinessPath)
	if err != nil {
		t.Fatalf("read persisted readiness: %v", err)
	}
	var persisted fleet.LabReadiness
	if err := json.Unmarshal(rawPersisted, &persisted); err != nil {
		t.Fatalf("persisted readiness is not JSON: %v\n%s", err, rawPersisted)
	}
	if persisted.Phase != rec.Phase || persisted.Evidence != "scrubbed-fleet-report" || !persisted.AdmitDispatch {
		t.Fatalf("persisted readiness = %+v, want emitted CLEAR from scrubbed report %+v", persisted, rec)
	}
}

func TestLabReadinessFromReportsHoldsWarmingInference(t *testing.T) {
	reportsDir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", reportsDir)

	if err := fleet.WriteReport(reportsDir, "box-a", fleet.Report{
		State:     fleet.StateLive,
		Inference: &fleet.InferenceStats{Status: fleet.InferenceWarming, Engine: "fak", Reason: "loading"},
	}); err != nil {
		t.Fatalf("write report: %v", err)
	}

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"readiness", "--from-reports", "--json"})
	if rc != 1 {
		t.Fatalf("warming inference should fail closed, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness --from-reports did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Waiting || rec.AdmitDispatch {
		t.Fatalf("readiness from warming inference = %+v, want WAITING hold", rec)
	}
	if rec.NextAction != "wait-lab-inference-ready" {
		t.Fatalf("next_action = %q, want wait-lab-inference-ready", rec.NextAction)
	}
}

func TestLabReadinessFromReportsHoldsSlowProbeLatency(t *testing.T) {
	reportsDir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", reportsDir)

	if err := fleet.WriteReport(reportsDir, "box-a", fleet.Report{
		State: fleet.StateLive,
		Inference: &fleet.InferenceStats{
			Status:         fleet.InferenceReady,
			Engine:         "route-proxy",
			Model:          "glm-5.2",
			ProbeLatencyMS: float64(labTargetLatencyBudgetMS) + 24,
			Reason:         "route-chat-timeout",
		},
	}); err != nil {
		t.Fatalf("write report: %v", err)
	}

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"readiness", "--from-reports", "--json"})
	if rc != 1 {
		t.Fatalf("slow ready inference should fail closed, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness --from-reports did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Waiting || rec.AdmitDispatch {
		t.Fatalf("readiness from slow ready inference = %+v, want WAITING hold", rec)
	}
	if rec.NextAction != "route-latency-exceeds-dev-budget-refresh-report-or-use-fallback" {
		t.Fatalf("next_action = %q, want route latency hold", rec.NextAction)
	}
	if rec.Evidence != "scrubbed-fleet-report" {
		t.Fatalf("evidence = %q, want scrubbed-fleet-report", rec.Evidence)
	}
}

func TestLabReadinessFromReportsRejectsManualPhase(t *testing.T) {
	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{"readiness", "--from-reports", "--phase", "CLEAR"})
	if rc != 2 {
		t.Fatalf("--from-reports + --phase should exit 2, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "use only one of --phase or --from-reports") {
		t.Fatalf("missing conflict diagnostic:\n%s", stderr.String())
	}
}

// TestLabReadinessStatusAliasCoarsens pins the rollover safety net: the deprecated
// --status flag still works and coarsens a legacy status onto the new phase+detail,
// so a caller that has not yet moved to --phase keeps producing valid records.
func TestLabReadinessStatusAliasCoarsens(t *testing.T) {
	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"readiness", "--status", "WAIT_PRIVATE_RECOVERY", "--json"})
	if rc != 1 {
		t.Fatalf("deprecated --status WAIT_PRIVATE_RECOVERY should fail closed, got %d: %s", rc, out.String())
	}
	var rec fleet.LabReadiness
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("readiness --status did not emit JSON: %v\n%s", err, out.String())
	}
	if rec.Phase != linkstate.Waiting || rec.Detail != linkstate.DetailPrivateRecovery || rec.AdmitDispatch {
		t.Fatalf("deprecated --status should coarsen to WAITING/private-recovery, got %+v", rec)
	}
}

func TestLabReadinessFromSnapshotIgnoresStaleUsefulInference(t *testing.T) {
	rec := labReadinessFromSnapshot("gpu-server", fleet.Snapshot{
		Reachable: 1,
		Rows: []fleet.BoxRow{{
			ID:        "box-a",
			State:     fleet.StateLive,
			AgeSec:    fleet.DefaultStaleSec + 1,
			Inference: &fleet.InferenceStats{Status: fleet.InferenceReady, Model: "glm-5.2"},
		}},
	}, time.Now())
	if rec.AdmitDispatch || rec.Phase == linkstate.Clear {
		t.Fatalf("stale ready inference must fail closed, got %+v", rec)
	}
	if rec.Evidence != "no-useful-lab-report" {
		t.Fatalf("evidence = %q, want no-useful-lab-report", rec.Evidence)
	}
}
