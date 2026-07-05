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

// TestWitnessExitedWorkersScrapesModelSidecar is the Layer-5b witness: a finished slot
// that carried a .model sidecar (a pinned, un-blanked worker) is graded with that model
// scraped into WitnessRecord.Model and emitted in the .witness sidecar, while a
// seat-default slot (no .model sidecar) grades with an empty Model and no model key.
func TestWitnessExitedWorkersScrapesModelSidecar(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")

	// A pinned worker: the resolver un-blanked its model, so the spawn wrote a .model sidecar.
	writeWitnessWorker(t, runsDir, "resolve-3001-20260704-010101",
		"# fak-spawn issue=3001 lane=gateway\nusage limit reached\n", deadDispatchPID)
	writeDispatchModelSidecar(filepath.Join(runsDir, "resolve-3001-20260704-010101.log"), "claude-opus-4-8")
	// A seat-default worker: no .model sidecar (byte-identical to the pre-seam floor).
	writeWitnessWorker(t, runsDir, "resolve-3002-20260704-020202",
		"# fak-spawn issue=3002 lane=docs\nusage limit reached\n", deadDispatchPID)

	_, records := witnessExitedWorkers(root, runsDir, true)
	byIssue := map[int]dispatchtick.WitnessRecord{}
	for _, rec := range records {
		byIssue[rec.Issue] = rec
	}
	if got := byIssue[3001].Model; got != "claude-opus-4-8" {
		t.Fatalf("pinned slot Model = %q, want claude-opus-4-8 (scraped from .model)", got)
	}
	if got := byIssue[3002].Model; got != "" {
		t.Fatalf("seat-default slot Model = %q, want empty (no .model sidecar)", got)
	}

	// The .witness sidecar carries the model key ONLY for the pinned slot.
	pinned, err := os.ReadFile(filepath.Join(runsDir, "resolve-3001-20260704-010101"+dispatchtick.WitnessSidecarSuffix))
	if err != nil {
		t.Fatalf("read pinned witness: %v", err)
	}
	var pinnedDoc map[string]any
	if err := json.Unmarshal(pinned, &pinnedDoc); err != nil {
		t.Fatalf("pinned witness not JSON: %v", err)
	}
	if pinnedDoc["model"] != "claude-opus-4-8" {
		t.Fatalf("pinned witness model = %v, want claude-opus-4-8", pinnedDoc["model"])
	}
	floor, err := os.ReadFile(filepath.Join(runsDir, "resolve-3002-20260704-020202"+dispatchtick.WitnessSidecarSuffix))
	if err != nil {
		t.Fatalf("read floor witness: %v", err)
	}
	var floorDoc map[string]any
	if err := json.Unmarshal(floor, &floorDoc); err != nil {
		t.Fatalf("floor witness not JSON: %v", err)
	}
	if _, ok := floorDoc["model"]; ok {
		t.Fatalf("seat-default witness must omit the model key, got %v", floorDoc["model"])
	}
}

// layer2DowngradeTick seeds #12's last slot as a model-switchable usage-cap wall, then runs
// a live docs tick, stubbing the broker (allow) and spawner (capture). It returns the parsed
// tick payload. modelDowngrade toggles the Layer-2 flag.
func layer2DowngradeTick(t *testing.T, modelDowngrade bool) (map[string]any, string) {
	t.Helper()
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	// #12's last slot walled on a usage cap — model-switchable, and seat-default (no .model
	// sidecar), so the next downgrade rung is the chain head claude-opus-4-8.
	writeWitnessWorker(t, runsDir, "resolve-12-20260704-050505",
		"# fak-spawn 20260704-050505 issue=12 lane=docs backend=claude argv0=claude\nusage limit reached\n", deadDispatchPID)

	oldBroker := launchSpawnBroker
	oldSpawner := dispatchIssueWorkerSpawner
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, rd string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		logPath := filepath.Join(rd, "resolve-12-20260704-060606.log")
		if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
			t.Fatalf("mkdir runs dir: %v", err)
		}
		if err := os.WriteFile(logPath, []byte("# fak-spawn\nworking\n"), 0o644); err != nil {
			t.Fatalf("write spawn log: %v", err)
		}
		return dispatchSpawnResult{PID: 4243, Log: logPath, Issue: issue, Lane: lane, Backend: backend}, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; dispatchIssueWorkerSpawner = oldSpawner })

	args := []string{"tick", "--workspace", root, "--lane", "docs", "--cooldown-min", "0", "--no-refresh", "--no-loop-ledger", "--live", "--json"}
	if modelDowngrade {
		args = append(args, "--model-downgrade")
	}
	out, errb, code := runDispatchAt(args...)
	if code != 0 {
		t.Fatalf("live exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	return got, runsDir
}

// TestDispatchTickLayer2DowngradeReDispatch is the Layer-2 witness: with --model-downgrade, a
// target whose last slot walled on a model-switchable reason is RE-DISPATCHED on the next
// downgrade-chain model (un-blanked), surfaced in the payload and written as the new slot's
// .model sidecar.
func TestDispatchTickLayer2DowngradeReDispatch(t *testing.T) {
	got, runsDir := layer2DowngradeTick(t, true)
	if got["action"] != "spawned" || got["verdict"] != "SPAWNED" {
		t.Fatalf("tick = action %v verdict %v, want a spawn (the switchable wall is re-dispatched, not held)", got["action"], got["verdict"])
	}
	dg := mapAt(got, "model_downgrade")
	if dispatchMapInt(dg, "issue") != 12 || dispatchMapString(dg, "model") != "claude-opus-4-8" {
		t.Fatalf("model_downgrade = %#v, want issue 12 -> claude-opus-4-8", dg)
	}
	wm := mapAt(got, "worker_model")
	if dispatchMapString(wm, "source") != modelSourceDowngrade || dispatchMapString(wm, "model") != "claude-opus-4-8" {
		t.Fatalf("worker_model = %#v, want model-downgrade source claude-opus-4-8", wm)
	}
	// The re-dispatched slot records the downgraded model as its .model sidecar.
	assertFileContains(t, filepath.Join(runsDir, "resolve-12-20260704-060606"+dispatchtick.ModelSidecarSuffix), "claude-opus-4-8")
}

// TestDispatchTickLayer2DefaultOff proves the seam is inert without the flag: the identical
// switchable-wall state spawns on the seat default, with no downgrade override or .model pin.
func TestDispatchTickLayer2DefaultOff(t *testing.T) {
	got, runsDir := layer2DowngradeTick(t, false)
	if got["action"] != "spawned" {
		t.Fatalf("tick = action %v, want a plain spawn", got["action"])
	}
	if _, ok := got["model_downgrade"]; ok {
		t.Fatalf("default-off tick surfaced model_downgrade: %v", got["model_downgrade"])
	}
	if _, ok := got["worker_model"]; ok {
		t.Fatalf("default-off tick pinned a worker_model: %v", got["worker_model"])
	}
	if _, err := os.Stat(filepath.Join(runsDir, "resolve-12-20260704-060606"+dispatchtick.ModelSidecarSuffix)); err == nil {
		t.Fatalf("default-off tick wrote a .model sidecar (seat-default worker must not)")
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
	if code != 0 {
		t.Fatalf("live exit = %d, want 0 (held issue is reported in the payload) (stderr: %s)\n%s", code, errb, out)
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
