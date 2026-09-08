package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDispatchTrajectoryAuditCountsWitnessesAndFriction(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("resolve-101-20260819-010000.log", "# fak-spawn\nLOCK_BUSY\nPRESTAGED_PATH_OVERLAP\n")
	write("resolve-101-20260819-010000.witness", `{"claim":"CLAIM_WITNESSED","issue":101,"sha":"abc","verdict":"OK"}`)
	write("resolve-102-20260819-010100.log", "No prompt provided via stdin.\nCRASH_RESTART_EXHAUSTED\n")
	write("resolve-102-20260819-010100.witness", `{"claim":"CLAIM_NO_COMMIT","issue":102,"reason":"died_before_epilogue"}`)

	rep, err := auditDispatchTrajectories(dir, time.Date(2026, 8, 19, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Sessions != 2 || rep.WitnessedShips != 1 || rep.NoCommit != 1 || rep.ShipYield != 0.5 {
		t.Fatalf("unexpected rollup: %+v", rep)
	}
	got := map[string]dispatchTrajectoryFriction{}
	for _, row := range rep.Friction {
		got[row.ID] = row
	}
	for _, id := range []string{"commit_lock_contention", "prestaged_path_overlap", "missing_prompt", "crash_restart_exhausted"} {
		if got[id].Sessions != 1 || got[id].Events != 1 {
			t.Fatalf("friction %s = %+v", id, got[id])
		}
	}
	if len(rep.Recommendations) == 0 {
		t.Fatal("expected repeatability recommendations")
	}
}

func TestDispatchTrajectoryAuditSinceAndJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "resolve-200-20260819-020000.log"), []byte("make ci\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "resolve-200-20260819-020000.witness"), []byte(`{"claim":"CLAIM_NO_COMMIT","issue":200}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	rc := runDispatchTrajectoryAudit(&stdout, &stderr, []string{"--runs-dir", dir, "--since", "2026-08-19T01:00:00Z", "--json"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var rep dispatchTrajectoryAuditReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Schema != dispatchTrajectoryAuditSchema || rep.Sessions != 1 || !strings.Contains(rep.Provenance[0], "structured") {
		t.Fatalf("unexpected report: %+v", rep)
	}
}

func TestDispatchTrajectoryAuditFullTreeGateRetryRecommendsValidateOwnedPaths(t *testing.T) {
	// #12083: full_tree_gate_retry produces the validate_owned_paths recommendation.
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("resolve-301-20260819-030000.log", "# fak-spawn\n== go build ==\nmake ci\nscripts\\ci.ps1\n")
	write("resolve-301-20260819-030000.witness", `{"claim":"CLAIM_NO_COMMIT","issue":301,"reason":"full_tree_gate_retry"}`)

	rep, err := auditDispatchTrajectories(dir, time.Date(2026, 8, 19, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	var foundRec bool
	for _, rec := range rep.Recommendations {
		if rec.ID == "validate_owned_paths" {
			foundRec = true
			if !strings.Contains(rec.Evidence, "full-tree gate retries") {
				t.Fatalf("expected full-tree gate retries in evidence, got: %s", rec.Evidence)
			}
			if !strings.Contains(rec.Action, "fak validate --mine") {
				t.Fatalf("expected action to cite fak validate --mine, got: %s", rec.Action)
			}
		}
	}
	if !foundRec {
		t.Fatalf("expected validate_owned_paths recommendation, got: %+v", rep.Recommendations)
	}
}

func TestDispatchTrajectoryAuditIdleSleepWasteRecommendsEpilogueQueue(t *testing.T) {
	// #12084: idle_sleep_waste produces the queue_commit_epilogues recommendation.
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("resolve-401-20260819-040000.log", "# fak-spawn\nStart-Sleep -Seconds 10\nStart-Sleep 5\n")
	write("resolve-401-20260819-040000.witness", `{"claim":"CLAIM_NO_COMMIT","issue":401,"reason":"idle_sleep_waste"}`)

	rep, err := auditDispatchTrajectories(dir, time.Date(2026, 8, 19, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	var foundFriction bool
	for _, f := range rep.Friction {
		if f.ID == "idle_sleep_waste" {
			foundFriction = true
			if f.Sessions != 1 || f.Events != 2 {
				t.Fatalf("expected 1 session and 2 events for idle_sleep_waste, got %+v", f)
			}
		}
	}
	if !foundFriction {
		t.Fatalf("expected idle_sleep_waste friction in report, got: %+v", rep.Friction)
	}

	var foundRec bool
	for _, rec := range rep.Recommendations {
		if rec.ID == "queue_commit_epilogues" {
			foundRec = true
			if !strings.Contains(rec.Evidence, "Start-Sleep") {
				t.Fatalf("expected Start-Sleep in evidence, got: %s", rec.Evidence)
			}
			if !strings.Contains(rec.Action, "fak commit --queue-on-busy") || !strings.Contains(rec.Action, "fak dispatch epilogue") {
				t.Fatalf("expected action to cite fak commit --queue-on-busy and fak dispatch epilogue, got: %s", rec.Action)
			}
		}
	}
	if !foundRec {
		t.Fatalf("expected queue_commit_epilogues recommendation, got: %+v", rep.Recommendations)
	}
}
