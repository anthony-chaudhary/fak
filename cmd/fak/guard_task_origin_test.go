package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestTaskGuardOriginAppearsWithEvidenceAndWitness(t *testing.T) {
	t.Setenv("FAK_TASKMGR", "1")
	oldOnce, oldMgr := procTaskMgrOnce, procTaskMgr
	procTaskMgrOnce = &sync.Once{}
	procTaskMgr = nil
	t.Cleanup(func() { procTaskMgrOnce, procTaskMgr = oldOnce, oldMgr })

	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "policy.json"),
		filepath.Join(dir, "transcript.jsonl"),
		filepath.Join(dir, "budget.json"),
		filepath.Join(dir, "stops.jsonl"),
	}
	for i, path := range paths {
		if i == 1 {
			continue // transcript/identity evidence location is created at origin before its first row
		}
		if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	registerGuardChildOriginTask("trace-origin", "claude", paths[0], paths[1], paths[2], paths[3])

	snapAny, enabled := serveTasksSnapshot()
	if !enabled {
		t.Fatal("task endpoint provider disabled")
	}
	b, err := json.Marshal(snapAny)
	if err != nil {
		t.Fatal(err)
	}
	var snap taskmgr.Snapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		t.Fatal(err)
	}
	var guard *taskmgr.TaskSnapshot
	for i := range snap.Tasks {
		if snap.Tasks[i].TaskID == "guard_trace-origin" {
			guard = &snap.Tasks[i]
			break
		}
	}
	if guard == nil {
		t.Fatalf("guard task absent from GET snapshot: %s", b)
	}
	if len(guard.EvidenceRefs) != 4 {
		t.Fatalf("evidence refs=%d want 4: %+v", len(guard.EvidenceRefs), guard.EvidenceRefs)
	}
	for _, label := range []string{"policy", "transcript", "budget_envelope", "stop_hook_ledger"} {
		found := false
		for _, ref := range guard.EvidenceRefs {
			if ref.Note == "guard-origin:"+label {
				found = true
			}
		}
		if !found {
			t.Errorf("missing %s evidence: %+v", label, guard.EvidenceRefs)
		}
	}
	if _, err := os.Stat(paths[1]); err != nil {
		t.Fatalf("transcript evidence location not materialized: %v", err)
	}
	if guard.Witness == nil || guard.Witness.VerifiedState != taskmgr.VerifiedDone || !strings.Contains(guard.Witness.Detail, "verified") {
		t.Fatalf("initial origin witness not confirmed: %+v", guard.Witness)
	}
}

func TestTaskGuardOriginNoopWhenManagerDisabled(t *testing.T) {
	t.Setenv("FAK_TASKMGR", "")
	registerGuardChildOriginTask("trace-disabled", "claude", "policy", "transcript", "budget", "stops")
	if _, enabled := serveTasksSnapshot(); enabled {
		t.Fatal("disabled task manager unexpectedly enabled")
	}
}
