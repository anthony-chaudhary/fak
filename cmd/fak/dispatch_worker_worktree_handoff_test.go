package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestDispatchWorktreeOwnerHandoffProtectsSpawnedWorker(t *testing.T) {
	parent := t.TempDir()
	wt := filepath.Join(parent, workerworktree.WorktreeMarker+"-spawn")
	if err := os.Mkdir(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	stamp := workerworktree.OwnerStamp{Schema: "fak-worker-worktree-owner/1", PID: 101, LeaseID: "lease", CreatedAt: time.Now().UTC()}
	raw, err := json.Marshal(stamp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(workerworktree.OwnerStampPath(wt)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(workerworktree.OwnerStampPath(wt), raw, 0o600); err != nil {
		t.Fatal(err)
	}

	payload := map[string]any{"worker_worktree": wt}
	if err := handoffDispatchWorktreeOwner(payload, 202); err != nil {
		t.Fatal(err)
	}
	updated, err := os.ReadFile(workerworktree.OwnerStampPath(wt))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(updated, &stamp); err != nil {
		t.Fatal(err)
	}
	if stamp.PID != 202 || payload["worker_owner_pid"] != 202 {
		t.Fatalf("stamp=%+v payload=%v, want spawned pid ownership", stamp, payload)
	}
}

func TestDispatchWorktreeOwnerHandoffFailureKeepsSpawnedWorkerLease(t *testing.T) {
	root := initRegionTestRepo(t)
	workerDir := filepath.Join(root, "prepared-worker-missing-owner-record")
	if err := os.MkdirAll(workerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	const leaseOwner = "live-worker-before-owner-handoff"
	t.Setenv("FAK_LEASE_OWNER", leaseOwner)
	t.Setenv("FLEET_DOGFOOD_GUARD", "1")
	t.Setenv("FLEET_WORKER_WORKTREE", "1")

	oldPrepare, oldBroker, oldSpawner := prepareManagedWorkerWorktreeFunc, launchSpawnBroker, dispatchIssueWorkerSpawner
	prepareManagedWorkerWorktreeFunc = func(string, string, string, string, string, workerworktree.GitRunner) workerworktree.Result {
		return workerworktree.Result{OK: true, Path: workerDir, BaseSHA: "base-sha"}
	}
	launchSpawnBroker = func(attempt launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(attempt, "unit-test-allow")
	}
	spawnCalls := 0
	spawnedPID := os.Getpid() // a process known live for the full assertion window
	dispatchIssueWorkerSpawner = func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
		spawnCalls++
		return dispatchSpawnResult{PID: spawnedPID, Log: filepath.Join(root, "spawned.log")}, nil
	}
	t.Cleanup(func() {
		prepareManagedWorkerWorktreeFunc = oldPrepare
		launchSpawnBroker = oldBroker
		dispatchIssueWorkerSpawner = oldSpawner
	})

	got, err := dispatchTickLiveSpawn(
		root, filepath.Join(root, dispatchtick.RunsDirName),
		dispatchTickOptions{Backend: "codex", Live: true, WorkerTimeoutS: 1200},
		dispatchLanePick{Lane: "gateway", Tree: []string{"internal/gateway/**"}},
		"resolve-gateway", dispatchtick.Account{Tag: "seat-a"}, dispatchtick.WorkerLaunch{},
		dispatchWorkerPreflightRequest{}, nil, 42,
		map[string]any{"prompt": "resolve #42"}, map[string]any{},
		func(payload map[string]any) map[string]any { return payload },
	)
	if err != nil {
		t.Fatalf("dispatchTickLiveSpawn: %v", err)
	}
	if spawnCalls != 1 || got["action"] != "worktree_owner_handoff_failed" || got["verdict"] != "WORKTREE_OWNER_HANDOFF_FAILED" || got["pid"] != spawnedPID {
		t.Fatalf("post-spawn handoff receipt = calls %d action %v verdict %v pid %v", spawnCalls, got["action"], got["verdict"], got["pid"])
	}
	if strings.TrimSpace(dispatchMapString(got, "reason")) == "" || got["worker_owner_pid"] != nil {
		t.Fatalf("failed handoff lacks honest evidence or claims ownership: reason=%v owner_pid=%v", got["reason"], got["worker_owner_pid"])
	}
	record, held := dispatchLiveLease(t, root)
	if !held {
		t.Fatal("spawned worker's lane lease was released after owner handoff failure")
	}
	if record.ID != "resolve-gateway" || record.Holder != leaseOwner || record.TTLSeconds <= 0 {
		t.Fatalf("retained lease = %+v, want live resolve-gateway lease owned by %q", record, leaseOwner)
	}

	t.Setenv("FAK_LEASE_OWNER", "peer-worker")
	peer := acquireDispatchLaneLease(root, "resolve-gateway-peer", "gateway", []string{"internal/gateway/**"}, 1800, "")
	if refused, _ := peer["refused"].(bool); !refused || peer["acquired"] == true {
		t.Fatalf("peer was not fenced from the live spawned worker's tree: %+v", peer)
	}
}
