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
	oldTestRun := dispatchWitnessTestRun
	dispatchWitnessResolvingSHA = sha
	dispatchWitnessCommitAudit = func(root, gotSHA string) (string, string) { return verdict, witness }
	// Default the #3838 test-run seam to a deterministic UNRUN so existing witness tests
	// neither shell out to `go test` nor depend on the ambient FAK_WITNESS_TEST_RUN env.
	// A test that exercises GREEN/RED overrides dispatchWitnessTestRun after this call.
	dispatchWitnessTestRun = func(root, gotSHA string) (bool, bool) { return false, false }
	t.Cleanup(func() {
		dispatchWitnessResolvingSHA = oldSHA
		dispatchWitnessCommitAudit = oldAudit
		dispatchWitnessTestRun = oldTestRun
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

// TestWitnessBindsTestRunToDoneClaim is the #3838 witness: at witness time the resolving
// commit's changed-package tests are run through the injectable seam and the GREEN / RED /
// UNRUN binding is recorded ALONGSIDE (never replacing) the diff-shape verdict, both in the
// WitnessRecord and in the .witness sidecar. A stubbed runner drives all three states: a
// diff-witnessed commit whose tests fail still grades CLAIM_TEST_RED, proving the rung is
// the promised layer between "the diff looks like the claim" and "the claim holds".
func TestWitnessBindsTestRunToDoneClaim(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	// Every slot resolves to a diff-witnessed commit; only the test-run rung differs.
	withWitnessStubs(t, func(_ string, issue int, _ string) string {
		return fmt.Sprintf("sha%d", issue)
	}, "OK", dispatchtick.WitnessOK)
	// Stubbed runner: green passed, red ran-but-failed, unrun never fired.
	dispatchWitnessTestRun = func(_ string, sha string) (bool, bool) {
		switch sha {
		case "sha5001":
			return true, true // GREEN
		case "sha5002":
			return true, false // RED
		default:
			return false, false // UNRUN
		}
	}

	writeWitnessWorker(t, runsDir, "resolve-5001-20260710-010101", "# fak-spawn issue=5001 lane=cmd\ndone\n", deadDispatchPID)
	writeWitnessWorker(t, runsDir, "resolve-5002-20260710-020202", "# fak-spawn issue=5002 lane=cmd\ndone\n", deadDispatchPID)
	writeWitnessWorker(t, runsDir, "resolve-5003-20260710-030303", "# fak-spawn issue=5003 lane=cmd\ndone\n", deadDispatchPID)

	_, records := witnessExitedWorkers(root, runsDir, true)
	byIssue := map[int]dispatchtick.WitnessRecord{}
	for _, rec := range records {
		byIssue[rec.Issue] = rec
	}
	want := map[int]string{
		5001: dispatchtick.ClaimTestGreen,
		5002: dispatchtick.ClaimTestRed,
		5003: dispatchtick.ClaimTestUnrun,
	}
	for issue, wantClaim := range want {
		rec := byIssue[issue]
		// The diff-shape verdict is never replaced — the test rung is strictly additive.
		if rec.Claim != dispatchtick.ClaimWitnessed {
			t.Fatalf("issue %d diff-shape claim = %q, want CLAIM_WITNESSED (test rung must be additive)", issue, rec.Claim)
		}
		if rec.TestClaim != wantClaim {
			t.Fatalf("issue %d TestClaim = %q, want %q", issue, rec.TestClaim, wantClaim)
		}
		// The binding must survive into the persisted .witness sidecar.
		side := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-", issue))
		matches, _ := filepath.Glob(side + "*" + dispatchtick.WitnessSidecarSuffix)
		if len(matches) != 1 {
			t.Fatalf("issue %d: want one .witness sidecar, got %v", issue, matches)
		}
		b, err := os.ReadFile(matches[0])
		if err != nil {
			t.Fatalf("read sidecar %s: %v", matches[0], err)
		}
		var doc map[string]any
		if err := json.Unmarshal(b, &doc); err != nil {
			t.Fatalf("sidecar %s not JSON: %v", matches[0], err)
		}
		if doc["test_claim"] != wantClaim {
			t.Fatalf("issue %d sidecar test_claim = %v, want %q", issue, doc["test_claim"], wantClaim)
		}
		if doc["claim"] != dispatchtick.ClaimWitnessed {
			t.Fatalf("issue %d sidecar must keep the diff-shape claim, got %v", issue, doc["claim"])
		}
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

// withWitnessLandReapStub swaps the #3168 land+reap seam for a recorder and returns
// a pointer to the ordered log of land+reap calls (each "<wtPath>|<base>").
func withWitnessLandReapStub(t *testing.T, fail bool) *[]string {
	t.Helper()
	old := dispatchWitnessLandReap
	calls := &[]string{}
	dispatchWitnessLandReap = func(root, wtPath, base string, tree []string) {
		*calls = append(*calls, wtPath+"|"+base)
		// fail=true models a land/reap that errored: the seam swallows it (returns
		// normally), so the sweep must still proceed to audit.
	}
	t.Cleanup(func() { dispatchWitnessLandReap = old })
	return calls
}

// TestWitnessLandsAndReapsWorkerWorktreeBeforeAudit is the #3168 witness: a dead-pid
// worker WITH a .worktree sidecar has its worktree landed+reaped BEFORE the
// resolving-SHA scan (so the just-landed commit is what gets witnessed), the sidecar
// is consumed (a second sweep never re-lands), and a land/reap fault is swallowed —
// the slot is still graded exactly as today.
func TestWitnessLandsAndReapsWorkerWorktreeBeforeAudit(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	stem := "resolve-3168-20260708-010101"

	var landRanBeforeScan bool
	landReap := withWitnessLandReapStub(t, true)
	// The resolving-SHA scan asserts land already fired: on the FIRST worker the
	// land+reap recorder must already hold an entry when the scan runs.
	withWitnessStubs(t, func(_ string, _ int, base string) string {
		if len(*landReap) > 0 {
			landRanBeforeScan = true
		}
		if base != "base3168" {
			t.Errorf("resolving-sha base = %q, want base3168 from the .basesha sidecar", base)
		}
		return "sha3168"
	}, "OK", "diff-witnessed")

	writeWitnessWorker(t, runsDir, stem, "# fak-spawn\nworked\n", deadDispatchPID)
	wtPath := filepath.FromSlash("/wt/fak-worker-wt-tools-3168abc")
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchWorktreeSidecarSuffix), []byte(wtPath), 0o644); err != nil {
		t.Fatalf("write worktree sidecar: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchtick.BaseSHASidecarSuffix), []byte("base3168\n"), 0o644); err != nil {
		t.Fatalf("write basesha sidecar: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchLeaseTreeSidecarSuffix), []byte(`["tools/**"]`), 0o644); err != nil {
		t.Fatalf("write lease-tree sidecar: %v", err)
	}

	_, records := witnessExitedWorkers(root, runsDir, true)

	if len(*landReap) != 1 {
		t.Fatalf("want exactly one land+reap call, got %v", *landReap)
	}
	if (*landReap)[0] != wtPath+"|base3168" {
		t.Fatalf("land+reap got %q, want worktree+base %q", (*landReap)[0], wtPath+"|base3168")
	}
	if !landRanBeforeScan {
		t.Fatal("land+reap must run BEFORE the resolving-SHA scan, so the landed commit is witnessed")
	}
	if len(records) != 1 || records[0].Claim != dispatchtick.ClaimWitnessed {
		t.Fatalf("slot should still grade CLAIM_WITNESSED after land, got %+v", records)
	}
	// The .worktree sidecar is consumed so a second sweep never re-lands.
	if _, err := os.Stat(filepath.Join(runsDir, stem+dispatchWorktreeSidecarSuffix)); !os.IsNotExist(err) {
		t.Fatalf("worktree sidecar should be removed after landing, stat err=%v", err)
	}
}

// TestWitnessLeavesSharedTrunkWorkerUntouched is the default-off regression guard:
// a dead-pid worker with NO .worktree sidecar (isolation off / prepare failed) is
// graded exactly as today, with the land+reap seam never invoked.
func TestWitnessLeavesSharedTrunkWorkerUntouched(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	landReap := withWitnessLandReapStub(t, false)
	withWitnessStubs(t, func(string, int, string) string { return "shaXYZ" }, "OK", "diff-witnessed")

	writeWitnessWorker(t, runsDir, "resolve-4242-20260708-020202", "# fak-spawn\nworked\n", deadDispatchPID)

	_, records := witnessExitedWorkers(root, runsDir, true)

	if len(*landReap) != 0 {
		t.Fatalf("no .worktree sidecar -> land+reap must never fire, got %v", *landReap)
	}
	if len(records) != 1 || records[0].Claim != dispatchtick.ClaimWitnessed {
		t.Fatalf("shared-trunk worker should grade unchanged, got %+v", records)
	}
}

// TestWitnessDryRunNeverLands proves a dry-run sweep (live=false) never mutates the
// trunk even when a .worktree sidecar is present.
func TestWitnessDryRunNeverLands(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	landReap := withWitnessLandReapStub(t, false)
	withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")

	stem := "resolve-4343-20260708-030303"
	writeWitnessWorker(t, runsDir, stem, "# fak-spawn\nworked\n", deadDispatchPID)
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchWorktreeSidecarSuffix),
		[]byte(filepath.FromSlash("/wt/fak-worker-wt-tools-4343")), 0o644); err != nil {
		t.Fatalf("write worktree sidecar: %v", err)
	}

	_, _ = witnessExitedWorkers(root, runsDir, false)

	if len(*landReap) != 0 {
		t.Fatalf("dry-run sweep must never land+reap, got %v", *landReap)
	}
	if _, err := os.Stat(filepath.Join(runsDir, stem+dispatchWorktreeSidecarSuffix)); err != nil {
		t.Fatalf("dry-run must leave the worktree sidecar in place, stat err=%v", err)
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

// TestDispatchLandVerify is the #3178 witness: the live land site wires a real
// (non-nil) verify, so a worktree whose edit REDS the injected build refuses the land
// (applied:false, committed:false) while a GREEN one commits. The fake git drives the
// land end-to-end without a real toolchain; dispatchLandVerify is pinned red/green.
func TestDispatchLandVerify(t *testing.T) {
	// The live land site must wire a real verify — not the old verify=nil the spine
	// (#3168) shipped, which let a red edit land on main.
	if dispatchLandVerify == nil {
		t.Fatal("live land site must wire a non-nil verify hook (#3178 regressed to verify=nil)")
	}
	orig := dispatchLandVerify
	t.Cleanup(func() { dispatchLandVerify = orig })

	// A fake git that yields a non-empty worktree diff and rubber-stamps the rest of
	// the land (message read, apply, commit) so the ONLY gate is the verify hook.
	fakeGit := func(_ string, args []string) (int, string) {
		if len(args) == 0 {
			return 0, ""
		}
		switch args[0] {
		case "diff":
			return 0, "diff --git a/x.go b/x.go\n--- a/x.go\n+++ b/x.go\n@@ -0,0 +1 @@\n+var x = 1\n"
		case "log":
			return 0, "feat(cmd): worker edit (#3178) (fak cmd)\n"
		case "apply":
			return 0, ""
		case "commit":
			return 0, "[main abc1234] worker edit\n"
		default:
			return 0, ""
		}
	}

	// Refuse-on-red: a worktree whose build reds lands NOTHING.
	dispatchLandVerify = func(string) (bool, string) { return false, "go build ./... failed: boom" }
	red := landWorkerWorktreeVerified(t.TempDir(), "wt", "base", nil, fakeGit)
	if red.OK || red.Applied || red.Committed {
		t.Fatalf("red verify must refuse the land (nothing applied/committed), got %+v", red)
	}
	if !strings.Contains(red.Reason, "refusing to land") {
		t.Fatalf("refusal must name why the land was refused (operator log), got %q", red.Reason)
	}

	// Pass-on-green: a worktree that builds lands normally (applied + committed).
	dispatchLandVerify = func(string) (bool, string) { return true, "" }
	green := landWorkerWorktreeVerified(t.TempDir(), "wt", "base", nil, fakeGit)
	if !green.OK || !green.Applied || !green.Committed {
		t.Fatalf("green verify must land the diff (applied+committed), got %+v", green)
	}
}

func TestWitnessExitedWorkersGradesCommittedFootprintAgainstLeaseTree(t *testing.T) {
	oldResolve, oldAudit, oldTest, oldPaths := dispatchWitnessResolvingSHA, dispatchWitnessCommitAudit, dispatchWitnessTestRun, dispatchWitnessCommitPaths
	defer func() {
		dispatchWitnessResolvingSHA, dispatchWitnessCommitAudit, dispatchWitnessTestRun, dispatchWitnessCommitPaths = oldResolve, oldAudit, oldTest, oldPaths
	}()
	dispatchWitnessResolvingSHA = func(string, int, string) string { return "abc123" }
	dispatchWitnessCommitAudit = func(string, string) (string, string) { return "OK", dispatchtick.WitnessOK }
	dispatchWitnessTestRun = func(string, string) (bool, bool) { return false, false }

	for _, tc := range []struct {
		name        string
		tree        []string
		paths       []string
		ok          bool
		wantClaim   string
		wantOutside int
	}{
		{"inside", []string{"internal/tools"}, []string{"internal/tools/a.go"}, true, "CLAIM_SCOPE_CLEAN", 0},
		{"outside", []string{"internal/tools"}, []string{"internal/tools/a.go", "docs/escaped.md"}, true, "CLAIM_OUT_OF_LANE", 1},
		{"empty-tree", nil, []string{"docs/escaped.md"}, true, "", 0},
		{"git-unknown", []string{"internal/tools"}, nil, false, "", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			stem := filepath.Join(dir, "resolve-4599-20260714-120000")
			if err := os.WriteFile(stem+".log", []byte("done\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(stem+".pid", []byte("99999999"), 0o644); err != nil {
				t.Fatal(err)
			}
			if tc.tree != nil {
				b, _ := json.Marshal(tc.tree)
				if err := os.WriteFile(stem+dispatchLeaseTreeSidecarSuffix, b, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			dispatchWitnessCommitPaths = func(string, string) ([]string, bool) { return tc.paths, tc.ok }
			_, records := witnessExitedWorkers(".", dir, true)
			if len(records) != 1 {
				t.Fatalf("records = %d", len(records))
			}
			got := records[0]
			if got.FootprintClaim != tc.wantClaim || got.OutOfLanePathCount != tc.wantOutside {
				t.Fatalf("footprint = (%q,%d), want (%q,%d)", got.FootprintClaim, got.OutOfLanePathCount, tc.wantClaim, tc.wantOutside)
			}
			row := got.Map()
			_, emitted := row["footprint_claim"]
			if emitted != (tc.wantClaim != "") {
				t.Fatalf("footprint_claim emitted=%v row=%v", emitted, row)
			}
			if got.Claim != dispatchtick.ClaimWitnessed || got.TestClaim != dispatchtick.ClaimTestUnrun {
				t.Fatalf("existing rungs changed: %+v", got)
			}
		})
	}
}
