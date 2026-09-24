package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestDispatchAgentQueueManagedWitnessRequiresExactLandedSHA(t *testing.T) {
	const issue = 7821
	const workerSHA = "worker-sha-7821"

	t.Run("same-issue peer SHA remains held", func(t *testing.T) {
		store, stem, worktreePath, rec := newManagedAgentQueueWitnessFixture(t, issue, "peer-sha-7821")
		writeManagedAgentQueueWitnessRow(t, stem, rec, &workerworktree.Result{
			OK: true, Committed: true, CommitSHA: workerSHA, Path: worktreePath,
		})
		if err := resolveDispatchAgentQueueWitness(stem, rec, rec.Map()); err != nil {
			t.Fatalf("resolve peer witness: %v", err)
		}
		assertDispatchWitnessAttemptHeld(t, store)
		if _, err := os.Stat(stem + dispatchAgentQueueBindingSuffix + ".done"); !os.IsNotExist(err) {
			t.Fatalf("peer SHA wrote completion marker: %v", err)
		}
	})

	t.Run("exact landed SHA completes", func(t *testing.T) {
		store, stem, worktreePath, rec := newManagedAgentQueueWitnessFixture(t, issue, workerSHA)
		writeManagedAgentQueueWitnessRow(t, stem, rec, &workerworktree.Result{
			OK: true, Committed: true, CommitSHA: workerSHA, Path: worktreePath,
		})
		if err := resolveDispatchAgentQueueWitness(stem, rec, rec.Map()); err != nil {
			t.Fatalf("resolve exact witness: %v", err)
		}
		snapshot, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.Attempts[0].State != agentqueue.AttemptSucceeded || snapshot.Attempts[0].WitnessDigest == "" ||
			snapshot.Intents[0].State != agentqueue.IntentCompleted {
			t.Fatalf("exact landed SHA did not complete: attempt=%+v intent=%+v", snapshot.Attempts[0], snapshot.Intents[0])
		}
		if _, err := os.Stat(stem + dispatchAgentQueueBindingSuffix + ".done"); err != nil {
			t.Fatalf("exact SHA completion marker missing: %v", err)
		}
	})
}

func TestDispatchAgentQueueManagedWitnessMissingOrFailedLandRemainsHeld(t *testing.T) {
	for _, tc := range []struct {
		name string
		land func(string) *workerworktree.Result
	}{
		{name: "missing land"},
		{name: "failed land", land: func(path string) *workerworktree.Result {
			return &workerworktree.Result{OK: false, Committed: false, CommitSHA: "worker-sha-7822", Path: path}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store, stem, worktreePath, rec := newManagedAgentQueueWitnessFixture(t, 7822, "worker-sha-7822")
			var land *workerworktree.Result
			if tc.land != nil {
				land = tc.land(worktreePath)
			}
			writeManagedAgentQueueWitnessRow(t, stem, rec, land)
			if err := resolveDispatchAgentQueueWitness(stem, rec, rec.Map()); err != nil {
				t.Fatalf("resolve incomplete land: %v", err)
			}
			assertDispatchWitnessAttemptHeld(t, store)
		})
	}
}

func TestDispatchAgentQueueManagedWitnessSweepUsesRegisteredPIDWithoutSidecar(t *testing.T) {
	const issue = 7823
	const workerSHA = "worker-sha-7823"
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stem := filepath.Join(runsDir, "resolve-7823-20260923-050505")
	if err := os.WriteFile(stem+".log", []byte("# fak-spawn issue=7823 lane=cmd\ndone\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, statePath, attemptID, nonce, _ := seedDispatchWitnessHeldAttempt(t, issue)
	worktreePath := filepath.Join(root, workerworktree.WorktreeMarker+"-pid-fallback")
	binding := dispatchAgentQueueBinding{
		Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
		AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: "cmd", Stem: filepath.Clean(stem),
		WorktreePath: filepath.Clean(worktreePath),
	}
	raw, err := json.Marshal(binding)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchAgentQueueBindingSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchWorktreeSidecarSuffix, []byte(worktreePath), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stem + ".pid"); !os.IsNotExist(err) {
		t.Fatalf("fixture unexpectedly has PID sidecar: %v", err)
	}

	withWitnessStubs(t, func(_ string, gotIssue int, _ string) string {
		if gotIssue != issue {
			t.Fatalf("resolving issue=%d, want %d", gotIssue, issue)
		}
		return "peer-sha-7823"
	}, "OK", dispatchtick.WitnessOK)
	stubDispatchExactIssueCitation(t, issue, workerSHA, true)
	dispatchWitnessTestRun = func(_ string, sha string) (bool, bool) {
		return sha == workerSHA, sha == workerSHA
	}
	oldLand := dispatchWitnessLandReap
	dispatchWitnessLandReap = func(_ string, path, _ string, _ []string) workerworktree.Result {
		return workerworktree.Result{
			OK: true, Committed: true, CommitSHA: workerSHA, Path: path,
			Code: workerworktree.LandResultSuccess, Applied: true, Removed: true,
		}
	}
	t.Cleanup(func() { dispatchWitnessLandReap = oldLand })

	_, records := witnessExitedWorkers(root, runsDir, true)
	if len(records) != 1 || records[0].SHA != workerSHA || records[0].Claim != dispatchtick.ClaimWitnessed || records[0].TestClaim != dispatchtick.ClaimTestGreen {
		t.Fatalf("sweep records = %+v, want exact green worker SHA", records)
	}
	snapshot, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Attempts[0].State != agentqueue.AttemptSucceeded || snapshot.Attempts[0].WitnessDigest == "" ||
		snapshot.Intents[0].State != agentqueue.IntentCompleted {
		t.Fatalf("registered PID fallback did not resolve queue: attempt=%+v intent=%+v", snapshot.Attempts[0], snapshot.Intents[0])
	}
}

func TestDispatchAgentQueueManagedWitnessSweepReplaysDurableLandReceipt(t *testing.T) {
	const issue = 7824
	const workerSHA = "worker-sha-7824"
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stem := filepath.Join(runsDir, "resolve-7824-20260923-060606")
	if err := os.WriteFile(stem+".log", []byte("# fak-spawn issue=7824 lane=cmd\ndone\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, statePath, attemptID, nonce, _ := seedDispatchWitnessHeldAttempt(t, issue)
	worktreePath := filepath.Join(root, workerworktree.WorktreeMarker+"-land-replay")
	binding := dispatchAgentQueueBinding{
		Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
		AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: "cmd", Stem: filepath.Clean(stem),
		WorktreePath: filepath.Clean(worktreePath),
	}
	raw, err := json.Marshal(binding)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchAgentQueueBindingSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	land := workerworktree.Result{
		OK: true, Committed: true, CommitSHA: workerSHA, Path: worktreePath,
		Code: workerworktree.LandResultSuccess, Applied: true, Removed: true,
	}
	raw, err = json.Marshal(land)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchAgentQueueLandSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stem + dispatchWorktreeSidecarSuffix); !os.IsNotExist(err) {
		t.Fatalf("fixture unexpectedly retained worktree sidecar: %v", err)
	}

	withWitnessStubs(t, func(_ string, gotIssue int, _ string) string {
		if gotIssue != issue {
			t.Fatalf("resolving issue=%d, want %d", gotIssue, issue)
		}
		return workerSHA
	}, "OK", dispatchtick.WitnessOK)
	stubDispatchExactIssueCitation(t, issue, workerSHA, true)
	dispatchWitnessTestRun = func(_ string, sha string) (bool, bool) {
		return sha == workerSHA, sha == workerSHA
	}
	oldLand := dispatchWitnessLandReap
	dispatchWitnessLandReap = func(string, string, string, []string) workerworktree.Result {
		t.Fatal("later sweep attempted to re-land after durable receipt")
		return workerworktree.Result{}
	}
	t.Cleanup(func() { dispatchWitnessLandReap = oldLand })

	_, records := witnessExitedWorkers(root, runsDir, true)
	if len(records) != 1 || records[0].SHA != workerSHA || records[0].TestClaim != dispatchtick.ClaimTestGreen {
		t.Fatalf("replay sweep records = %+v, want exact green worker SHA", records)
	}
	snapshot, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Attempts[0].State != agentqueue.AttemptSucceeded || snapshot.Attempts[0].WitnessDigest == "" ||
		snapshot.Intents[0].State != agentqueue.IntentCompleted {
		t.Fatalf("durable land replay did not resolve queue: attempt=%+v intent=%+v", snapshot.Attempts[0], snapshot.Intents[0])
	}
	var durable struct {
		WorktreeLand *workerworktree.Result `json:"worktree_land"`
	}
	witnessRaw, err := os.ReadFile(stem + dispatchtick.WitnessSidecarSuffix)
	if err != nil || json.Unmarshal(witnessRaw, &durable) != nil || durable.WorktreeLand == nil || durable.WorktreeLand.CommitSHA != workerSHA {
		t.Fatalf("replay witness did not retain durable land receipt: err=%v body=%s", err, witnessRaw)
	}
}

func TestDispatchAgentQueueManagedWitnessSweepRejectsUncitedLandedSHA(t *testing.T) {
	const issue = 7825
	const workerSHA = "worker-sha-7825"
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stem := filepath.Join(runsDir, "resolve-7825-20260923-070707")
	if err := os.WriteFile(stem+".log", []byte("# fak-spawn issue=7825 lane=cmd\ndone\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, statePath, attemptID, nonce, _ := seedDispatchWitnessHeldAttempt(t, issue)
	worktreePath := filepath.Join(root, workerworktree.WorktreeMarker+"-uncited")
	binding := dispatchAgentQueueBinding{
		Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
		AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: "cmd", Stem: filepath.Clean(stem),
		WorktreePath: filepath.Clean(worktreePath),
	}
	raw, err := json.Marshal(binding)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchAgentQueueBindingSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchWorktreeSidecarSuffix, []byte(worktreePath), 0o600); err != nil {
		t.Fatal(err)
	}

	withWitnessStubs(t, func(string, int, string) string { return "peer-sha-7825" }, "OK", dispatchtick.WitnessOK)
	stubDispatchExactIssueCitation(t, issue, workerSHA, false)
	dispatchWitnessTestRun = func(_ string, sha string) (bool, bool) {
		return sha == workerSHA, sha == workerSHA
	}
	oldLand := dispatchWitnessLandReap
	dispatchWitnessLandReap = func(_ string, path, _ string, _ []string) workerworktree.Result {
		return workerworktree.Result{
			OK: true, Committed: true, CommitSHA: workerSHA, Path: path,
			Code: workerworktree.LandResultSuccess, Applied: true, Removed: true,
		}
	}
	t.Cleanup(func() { dispatchWitnessLandReap = oldLand })

	_, records := witnessExitedWorkers(root, runsDir, true)
	if len(records) != 1 || records[0].SHA != workerSHA || records[0].Claim != dispatchtick.ClaimUnwitnessed {
		t.Fatalf("uncited exact SHA records = %+v, want worker SHA unwitnessed", records)
	}
	assertDispatchWitnessAttemptHeld(t, store)
}

func TestDispatchAgentQueueManagedWitnessSweepRetriesTransientProofFailure(t *testing.T) {
	for i, tc := range []struct {
		name             string
		citationError    bool
		auditUnavailable bool
	}{
		{name: "citation probe error", citationError: true},
		{name: "commit audit unavailable", auditUnavailable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			issue := 7826 + i
			workerSHA := fmt.Sprintf("worker-sha-%d", issue)
			root := t.TempDir()
			runsDir := filepath.Join(root, dispatchtick.RunsDirName)
			if err := os.MkdirAll(runsDir, 0o755); err != nil {
				t.Fatal(err)
			}
			stem := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-20260923-080808", issue))
			if err := os.WriteFile(stem+".log", []byte(fmt.Sprintf("# fak-spawn issue=%d lane=cmd\ndone\n", issue)), 0o600); err != nil {
				t.Fatal(err)
			}
			store, statePath, attemptID, nonce, _ := seedDispatchWitnessHeldAttempt(t, issue)
			worktreePath := filepath.Join(root, workerworktree.WorktreeMarker+"-transient-proof")
			binding := dispatchAgentQueueBinding{
				Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
				AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: "cmd", Stem: filepath.Clean(stem),
				WorktreePath: filepath.Clean(worktreePath),
			}
			raw, err := json.Marshal(binding)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(stem+dispatchAgentQueueBindingSuffix, raw, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(stem+dispatchWorktreeSidecarSuffix, []byte(worktreePath), 0o600); err != nil {
				t.Fatal(err)
			}

			withWitnessStubs(t, func(string, int, string) string { return "peer-sha" }, "OK", dispatchtick.WitnessOK)
			dispatchWitnessTestRun = func(_ string, sha string) (bool, bool) {
				return sha == workerSHA, sha == workerSHA
			}
			oldCitation := dispatchWitnessExactIssueCites
			citationCalls := 0
			dispatchWitnessExactIssueCites = func(_ string, gotSHA string, gotIssue int) (bool, error) {
				citationCalls++
				if gotSHA != workerSHA || gotIssue != issue {
					t.Fatalf("exact issue citation called with sha=%q issue=%d, want %q/%d", gotSHA, gotIssue, workerSHA, issue)
				}
				if tc.citationError && citationCalls == 1 {
					return false, fmt.Errorf("transient git show failure")
				}
				return true, nil
			}
			t.Cleanup(func() { dispatchWitnessExactIssueCites = oldCitation })
			if tc.auditUnavailable {
				oldAudit := dispatchWitnessCommitAudit
				auditCalls := 0
				dispatchWitnessCommitAudit = func(_ string, gotSHA string) (string, string) {
					auditCalls++
					if gotSHA != workerSHA {
						t.Fatalf("commit audit SHA=%q, want %q", gotSHA, workerSHA)
					}
					if auditCalls == 1 {
						return "", ""
					}
					return "OK", dispatchtick.WitnessOK
				}
				t.Cleanup(func() { dispatchWitnessCommitAudit = oldAudit })
			}
			oldLand := dispatchWitnessLandReap
			landCalls := 0
			dispatchWitnessLandReap = func(_ string, path, _ string, _ []string) workerworktree.Result {
				landCalls++
				return workerworktree.Result{
					OK: true, Committed: true, CommitSHA: workerSHA, Path: path,
					Code: workerworktree.LandResultSuccess, Applied: true, Removed: true,
				}
			}
			t.Cleanup(func() { dispatchWitnessLandReap = oldLand })

			_, records := witnessExitedWorkers(root, runsDir, true)
			if len(records) != 0 {
				t.Fatalf("transient proof failure persisted records: %+v", records)
			}
			if _, err := os.Stat(stem + dispatchtick.WitnessSidecarSuffix); !os.IsNotExist(err) {
				t.Fatalf("transient proof failure persisted witness: %v", err)
			}
			assertDispatchWitnessAttemptHeld(t, store)

			_, records = witnessExitedWorkers(root, runsDir, true)
			if len(records) != 1 || records[0].SHA != workerSHA || records[0].Claim != dispatchtick.ClaimWitnessed || records[0].TestClaim != dispatchtick.ClaimTestGreen {
				t.Fatalf("successful retry records=%+v, want exact green worker SHA", records)
			}
			if landCalls != 1 {
				t.Fatalf("land calls=%d, want durable receipt replay without re-land", landCalls)
			}
			snapshot, err := store.Load()
			if err != nil {
				t.Fatal(err)
			}
			if snapshot.Attempts[0].State != agentqueue.AttemptSucceeded || snapshot.Intents[0].State != agentqueue.IntentCompleted {
				t.Fatalf("successful proof retry did not resolve queue: attempt=%+v intent=%+v", snapshot.Attempts[0], snapshot.Intents[0])
			}
		})
	}
}

func newManagedAgentQueueWitnessFixture(t *testing.T, issue int, sha string) (agentqueue.Store, string, string, dispatchtick.WitnessRecord) {
	t.Helper()
	store, statePath, attemptID, nonce, pid := seedDispatchWitnessHeldAttempt(t, issue)
	dir := t.TempDir()
	stem := filepath.Join(dir, fmt.Sprintf("resolve-%d-20260923-040404", issue))
	worktreePath := filepath.Join(dir, workerworktree.WorktreeMarker+"-witness")
	binding := dispatchAgentQueueBinding{
		Schema: dispatchAgentQueueBindingSchema, StatePath: filepath.Clean(statePath),
		AttemptID: attemptID, Nonce: nonce, Issue: issue, Lane: "cmd", Stem: filepath.Clean(stem),
		WorktreePath: filepath.Clean(worktreePath),
	}
	raw, err := json.Marshal(binding)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchAgentQueueBindingSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(pid)), 0o600); err != nil {
		t.Fatal(err)
	}
	rec := dispatchGreenWitnessRecord(issue, stem)
	rec.SHA = sha
	return store, stem, worktreePath, rec
}

func writeManagedAgentQueueWitnessRow(t *testing.T, stem string, rec dispatchtick.WitnessRecord, land *workerworktree.Result) {
	t.Helper()
	row := rec.Map()
	if land != nil {
		row["worktree_land"] = *land
	}
	raw, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+dispatchtick.WitnessSidecarSuffix, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func stubDispatchExactIssueCitation(t *testing.T, issue int, sha string, result bool) {
	t.Helper()
	original := dispatchWitnessExactIssueCites
	dispatchWitnessExactIssueCites = func(_ string, gotSHA string, gotIssue int) (bool, error) {
		if gotSHA != sha || gotIssue != issue {
			t.Fatalf("exact issue citation called with sha=%q issue=%d, want %q/%d", gotSHA, gotIssue, sha, issue)
		}
		return result, nil
	}
	t.Cleanup(func() { dispatchWitnessExactIssueCites = original })
}
