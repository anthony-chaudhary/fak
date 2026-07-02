package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// deadDispatchPID is a pid no live process plausibly holds on either OS, so the
// dead-pid gate sees a provably-finished worker without faking the probe.
const deadDispatchPID = 99999999

func withWitnessStubs(t *testing.T, sha func(root string, issue int, base string) string, verdict, witness string) {
	t.Helper()
	oldSHA := dispatchWitnessResolvingSHA
	oldAudit := dispatchWitnessCommitAudit
	dispatchWitnessResolvingSHA = sha
	dispatchWitnessCommitAudit = func(root, gotSHA string) (string, string) { return verdict, witness }
	t.Cleanup(func() {
		dispatchWitnessResolvingSHA = oldSHA
		dispatchWitnessCommitAudit = oldAudit
	})
}

func writeWitnessWorker(t *testing.T, runsDir, stem, body string, pid int) {
	t.Helper()
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+".log"), []byte(body), 0o644); err != nil {
		t.Fatalf("write worker log: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+".pid"), []byte(fmt.Sprint(pid)), 0o644); err != nil {
		t.Fatalf("write worker pid: %v", err)
	}
}

// TestWitnessExitedWorkersGradesFinishedSlots is the #1324-proposal-#2 witness for the
// Go tick: each DEAD worker slot is graded exactly once into CLAIM_WITNESSED /
// CLAIM_NO_COMMIT, the verdict lands in a .witness sidecar on a live sweep, and a
// still-running / already-audited / pid-less slot is left alone.
func TestWitnessExitedWorkersGradesFinishedSlots(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	withWitnessStubs(t, func(_ string, issue int, base string) string {
		if issue == 2001 {
			if base != "base2001" {
				t.Errorf("resolving-sha base = %q, want base2001 from the .basesha sidecar", base)
			}
			return "abc123"
		}
		return ""
	}, "OK", dispatchtick.WitnessOK)

	// A finished worker that landed a diff-witnessed commit.
	writeWitnessWorker(t, runsDir, "resolve-2001-20260702-010101",
		"# fak-spawn issue=2001 lane=docs\nreal work\n", deadDispatchPID)
	if err := os.WriteFile(filepath.Join(runsDir, "resolve-2001-20260702-010101"+dispatchtick.BaseSHASidecarSuffix), []byte("base2001\n"), 0o644); err != nil {
		t.Fatalf("write basesha sidecar: %v", err)
	}
	// A finished worker structurally blocked by the guard: no commit, SELF_MODIFY tail.
	writeWitnessWorker(t, runsDir, "resolve-2002-20260702-020202",
		"# fak-spawn issue=2002 lane=cmd\nguard refused: reason=SELF_MODIFY\n", deadDispatchPID)
	// A still-running worker must not be graded (it may not have committed yet).
	writeWitnessWorker(t, runsDir, "resolve-2003-20260702-030303",
		"# fak-spawn issue=2003 lane=docs\nstreaming\n", os.Getpid())
	// An already-audited worker must stay audited-once.
	writeWitnessWorker(t, runsDir, "resolve-2004-20260702-040404",
		"# fak-spawn issue=2004 lane=docs\ndone\n", deadDispatchPID)
	if err := os.WriteFile(filepath.Join(runsDir, "resolve-2004-20260702-040404"+dispatchtick.WitnessSidecarSuffix), []byte(`{"claim":"CLAIM_WITNESSED"}`), 0o644); err != nil {
		t.Fatalf("write existing witness sidecar: %v", err)
	}
	// A pid-less log cannot prove the worker finished.
	if err := os.WriteFile(filepath.Join(runsDir, "resolve-2005-20260702-050505.log"), []byte("no pid\n"), 0o644); err != nil {
		t.Fatalf("write pid-less log: %v", err)
	}

	payload, records := witnessExitedWorkers(root, runsDir, true)
	if len(records) != 2 {
		t.Fatalf("records = %+v, want exactly the two dead unaudited slots", records)
	}
	byIssue := map[int]dispatchtick.WitnessRecord{}
	for _, rec := range records {
		byIssue[rec.Issue] = rec
	}
	if rec := byIssue[2001]; rec.Claim != dispatchtick.ClaimWitnessed || rec.SHA != "abc123" || rec.Verdict != "OK" {
		t.Fatalf("slot 2001 = %+v, want CLAIM_WITNESSED abc123", rec)
	}
	if rec := byIssue[2002]; rec.Claim != dispatchtick.ClaimNoCommit || rec.Reason != dispatchtick.NoCommitSelfModify {
		t.Fatalf("slot 2002 = %+v, want CLAIM_NO_COMMIT/self_modify", rec)
	}
	if got := len(payload["witnessed"].([]any)); got != 1 {
		t.Fatalf("witnessed bucket = %v, want 1 row", payload["witnessed"])
	}
	if got := len(payload["no_commit"].([]any)); got != 1 {
		t.Fatalf("no_commit bucket = %v, want 1 row", payload["no_commit"])
	}
	for _, stem := range []string{"resolve-2001-20260702-010101", "resolve-2002-20260702-020202"} {
		side := filepath.Join(runsDir, stem+dispatchtick.WitnessSidecarSuffix)
		b, err := os.ReadFile(side)
		if err != nil {
			t.Fatalf("live sweep must write %s: %v", side, err)
		}
		var doc map[string]any
		if err := json.Unmarshal(b, &doc); err != nil {
			t.Fatalf("sidecar %s is not JSON: %v (%s)", side, err, b)
		}
	}
	for _, stem := range []string{"resolve-2003-20260702-030303", "resolve-2005-20260702-050505"} {
		if _, err := os.Stat(filepath.Join(runsDir, stem+dispatchtick.WitnessSidecarSuffix)); err == nil {
			t.Fatalf("sweep graded a slot it must skip: %s", stem)
		}
	}

	// The sidecar gates re-audit: a second sweep finds nothing new.
	if _, again := witnessExitedWorkers(root, runsDir, true); len(again) != 0 {
		t.Fatalf("second sweep re-audited: %+v, want audited-once", again)
	}
}

// TestWitnessExitedWorkersDryRunWritesNoSidecars pins the read-only half of the
// mirror: only a LIVE sweep may leave the .witness side effect behind.
func TestWitnessExitedWorkersDryRunWritesNoSidecars(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
	writeWitnessWorker(t, runsDir, "resolve-2010-20260702-010101",
		"# fak-spawn issue=2010 lane=docs\nPOLICY_BLOCK\n", deadDispatchPID)

	payload, records := witnessExitedWorkers(root, runsDir, false)
	if len(records) != 1 || records[0].Reason != dispatchtick.NoCommitPolicyBlock {
		t.Fatalf("records = %+v, want one policy_block no-commit", records)
	}
	if payload["live"] != false {
		t.Fatalf("payload live = %v, want false", payload["live"])
	}
	if _, err := os.Stat(filepath.Join(runsDir, "resolve-2010-20260702-010101"+dispatchtick.WitnessSidecarSuffix)); err == nil {
		t.Fatalf("dry sweep wrote a .witness sidecar")
	}
}

// TestDispatchTickLiveHoldsStructurallyBlockedIssue is the #1396 pick-held-invariant
// witness for the Go verb: a LIVE `fak dispatch tick` whose lane's only open issue
// just exited SELF_MODIFY-blocked must HOLD it (NO_ISSUE + held_no_commit evidence)
// instead of re-storming the same guard, while a dry run of the identical state (the
// sweep is live-only, mirroring the Python dispatcher) still reports WOULD_SPAWN.
func TestDispatchTickLiveHoldsStructurallyBlockedIssue(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	// The docs lane's only routed issue (#12 in the stub router) just finished
	// guard-blocked: dead pid, no resolving commit, SELF_MODIFY in the log tail.
	writeWitnessWorker(t, runsDir, "resolve-12-20260702-060606",
		"# fak-spawn 20260702-060606 issue=12 lane=docs backend=claude argv0=claude\nguard summary: refused reason=SELF_MODIFY\n", deadDispatchPID)

	// Dry run first: the witness sweep is live-only, so the tick still plans a spawn.
	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--cooldown-min", "0", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("dry exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	var dry map[string]any
	if err := json.Unmarshal([]byte(out), &dry); err != nil {
		t.Fatalf("bad dry json: %v\n%s", err, out)
	}
	if dry["verdict"] != "WOULD_SPAWN" || dry["target_issue"] != float64(12) {
		t.Fatalf("dry tick = verdict %v target %v, want WOULD_SPAWN/12", dry["verdict"], dry["target_issue"])
	}
	if _, ok := dry["held_no_commit"]; ok {
		t.Fatalf("dry tick surfaced held_no_commit: %v", dry["held_no_commit"])
	}
	if _, ok := dry["witnessed_slots"]; ok {
		t.Fatalf("dry tick surfaced witnessed_slots: %v", dry["witnessed_slots"])
	}

	// Live tick: the sweep grades the dead slot, records it, and the pick HOLDS #12.
	out, errb, code = runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--cooldown-min", "0", "--live", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("live exit = %d, want 1 (held issue leaves nothing to dispatch) (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out)
	}
	if got["verdict"] != "NO_ISSUE" || got["action"] != "no_issue" {
		t.Fatalf("live tick = verdict %v action %v, want NO_ISSUE/no_issue (the hold, not a spawn)", got["verdict"], got["action"])
	}
	held := got["held_no_commit"].([]any)
	if len(held) != 1 || held[0] != float64(12) {
		t.Fatalf("held_no_commit = %v, want [12]", got["held_no_commit"])
	}
	slots := mapAt(got, "witnessed_slots")
	noCommit, _ := slots["no_commit"].([]any)
	if len(noCommit) != 1 {
		t.Fatalf("witnessed_slots.no_commit = %v, want the graded slot", slots["no_commit"])
	}
	row := noCommit[0].(map[string]any)
	if dispatchMapInt(row, "issue") != 12 || dispatchMapString(row, "reason") != dispatchtick.NoCommitSelfModify {
		t.Fatalf("no_commit row = %#v, want issue 12 reason self_modify", row)
	}
	if !strings.Contains(dispatchMapString(got, "reason"), "structural guard refusal") {
		t.Fatalf("NO_ISSUE reason %q should name the structural hold", got["reason"])
	}
	assertFileContains(t, filepath.Join(runsDir, "resolve-12-20260702-060606"+dispatchtick.WitnessSidecarSuffix), dispatchtick.NoCommitSelfModify)
}
