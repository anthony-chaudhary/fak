package main

// #4324 release-on-exit for the dispatch lane lease. The acceptance criterion is a
// LATENCY one — "a finished worker's lane is re-acquirable by a peer within seconds,
// not the TTL window" — so every case here acquires with a full-length TTL (1800s, the
// production worker-timeout order) and never advances the clock. Nothing in these tests
// can pass by waiting: the only mechanism that can hand the lane back inside the test's
// runtime is the explicit fenced release.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// dispatchLeaseReleaseFixture stands up a repo with a lane lease held by a finished
// worker: it acquires "resolve-gateway" over the gateway tree through the REAL dispatch
// admission path, then lays down the dead-pid worker slot and the two sidecars the spawn
// writes (the lease id and the #4324 fencing token). It returns the repo root, the runs
// dir, the slot stem and the acquire result.
func dispatchLeaseReleaseFixture(t *testing.T, stem string) (root, runsDir string, lease map[string]any) {
	t.Helper()
	root = initRegionTestRepo(t)
	runsDir = filepath.Join(root, dispatchtick.RunsDirName)
	t.Setenv("FAK_LEASE_OWNER", "worker-that-finished")
	lease = acquireDispatchLaneLease(root, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 1800, "")
	if acquired, _ := lease["acquired"].(bool); !acquired {
		t.Fatalf("fixture could not acquire the lane lease: %+v", lease)
	}
	writeWitnessWorker(t, runsDir, stem, "# fak-spawn issue=4324 lane=gateway\nwork\n", deadDispatchPID)
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchLeaseIDSidecarSuffix), []byte("resolve-gateway"), 0o644); err != nil {
		t.Fatalf("write lease-id sidecar: %v", err)
	}
	if p := writeDispatchLeaseFenceSidecar(filepath.Join(runsDir, stem+".log"), lease); p == "" {
		t.Fatalf("the spawn must persist a fencing token for an acquired lease: %+v", lease)
	}
	return root, runsDir, lease
}

// dispatchWitnessRow returns the single graded row a sweep produced, so an assertion can
// read the release evidence the sweep stamped onto it.
func dispatchWitnessRow(t *testing.T, payload map[string]any) map[string]any {
	t.Helper()
	rows, _ := payload["audited"].([]any)
	if len(rows) != 1 {
		t.Fatalf("audited rows = %v, want exactly one graded slot", payload["audited"])
	}
	row, ok := rows[0].(map[string]any)
	if !ok {
		t.Fatalf("graded row is not a map: %T", rows[0])
	}
	return row
}

// dispatchLiveLease reads the live lane lease straight off refs/fak/locks/, the only
// state that decides whether a peer is admitted.
func dispatchLiveLease(t *testing.T, root string) (leaseref.Record, bool) {
	t.Helper()
	rec, ok, err := leaseref.NewInDir(root).Get(context.Background(), "resolve-gateway")
	if err != nil {
		t.Fatalf("read live lease: %v", err)
	}
	return rec, ok
}

func TestDispatchWitnessReleasesLaneLeaseOnNormalExit(t *testing.T) {
	// The headline acceptance: a worker that finished normally hands its lane back at
	// witness time, so a peer that would have been refused LANE_LEASE_HELD for the whole
	// 1800s TTL is admitted immediately.
	t.Run("finished_worker_lane_is_reacquirable_immediately", func(t *testing.T) {
		stem := "resolve-4324-20260802-010101"
		root, runsDir, _ := dispatchLeaseReleaseFixture(t, stem)

		// Pre-condition: while the lease is held, a peer on the same tree IS refused.
		// Without this the acceptance below could pass against a lane nobody ever held.
		t.Setenv("FAK_LEASE_OWNER", "peer-waiting")
		blocked := acquireDispatchLaneLease(root, "resolve-gateway-peer", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if refused, _ := blocked["refused"].(bool); !refused {
			t.Fatalf("a live lane lease must refuse a peer, got %+v", blocked)
		}

		withWitnessStubs(t, func(string, int, string) string { return "sha4324" }, "OK", dispatchtick.WitnessOK)
		payload, records := witnessExitedWorkers(root, runsDir, true)
		if len(records) != 1 || records[0].Claim != dispatchtick.ClaimWitnessed {
			t.Fatalf("records = %+v, want one CLAIM_WITNESSED slot", records)
		}
		if got := dispatchWitnessRow(t, payload)["lease_released"]; got != "resolve-gateway" {
			t.Errorf("graded row lease_released = %v, want the freed lease id", got)
		}
		if _, ok := dispatchLiveLease(t, root); ok {
			t.Errorf("the finished worker's lease is still on refs/fak/locks/ after the sweep")
		}

		// ACCEPTANCE (#4324): the peer gets in NOW. The lease was acquired seconds ago
		// with a 1800s TTL and the clock was never advanced, so nothing but the explicit
		// release can admit this.
		t.Setenv("FAK_LEASE_OWNER", "peer-after-exit")
		back := acquireDispatchLaneLease(root, "resolve-gateway-next", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if acquired, _ := back["acquired"].(bool); !acquired {
			t.Fatalf("a finished worker's lane must be re-acquirable within seconds, got %+v", back)
		}
	})

	// The dangerous half. A janitor reclaimed this lane and re-issued it while the worker
	// was finishing; the fencing token the worker recorded is now stale. Releasing on the
	// holder string alone would free a lane a DIFFERENT live worker owns — two writers in
	// one lane, strictly worse than the stranding this ticket cures. The reclaimer here
	// deliberately carries the SAME holder string (one box, one FAK_LEASE_OWNER, two
	// ticks), so the ONLY thing standing between the sweep and a peer's lane is the
	// generation check.
	t.Run("fence_refuses_release_of_a_reclaimed_lane", func(t *testing.T) {
		stem := "resolve-4325-20260802-020202"
		root, runsDir, lease := dispatchLeaseReleaseFixture(t, stem)
		holder := dispatchMapString(lease, "holder")

		// The reclaim: past the TTL the lease transitions to a new instance and the
		// generation bumps, exactly as AcquireFenced's TRANSITION rung does in production.
		reissued, v, err := leaseref.NewInDir(root).AcquireFenced(context.Background(), leaseref.Record{
			ID: "resolve-gateway", Holder: holder, TreeGlobs: []string{"internal/gateway/**"}, TTLSeconds: 1800,
		}, time.Now().Add(2*time.Hour))
		if err != nil || !v.OK {
			t.Fatalf("reclaim setup: %+v %v", v, err)
		}
		if reissued.Generation <= dispatchLeaseGeneration(lease["generation"]) {
			t.Fatalf("reclaim must advance the generation, got %d after %v", reissued.Generation, lease["generation"])
		}

		withWitnessStubs(t, func(string, int, string) string { return "sha4325" }, "OK", dispatchtick.WitnessOK)
		payload, _ := witnessExitedWorkers(root, runsDir, true)
		row := dispatchWitnessRow(t, payload)
		if got, ok := row["lease_released"]; ok {
			t.Fatalf("the sweep freed a lane it no longer owns (lease_released=%v) — two writers in one lane", got)
		}
		if got := row["lease_release_refused"]; got != "stale_lease" {
			t.Fatalf("graded row lease_release_refused = %v, want stale_lease", got)
		}
		live, ok := dispatchLiveLease(t, root)
		if !ok {
			t.Fatalf("the RECLAIMED lease was deleted by a stale release — the peer's lane is now unfenced")
		}
		if live.Generation != reissued.Generation {
			t.Fatalf("live lease generation = %d, want the reclaimer's %d", live.Generation, reissued.Generation)
		}
	})

	// An unclassifiable exit is the crash / panic / SIGKILL bucket: the classifier found
	// no terminating signature, so the lane may be mid-write. It deliberately keeps its
	// lease and waits out the TTL — the pre-#4324 behaviour, which is the correct path
	// for a worker that did not stop on purpose.
	t.Run("crashed_worker_keeps_its_lease", func(t *testing.T) {
		stem := "resolve-4326-20260802-030303"
		root, runsDir, _ := dispatchLeaseReleaseFixture(t, stem)

		withWitnessStubs(t, func(string, int, string) string { return "" }, "", "")
		payload, records := witnessExitedWorkers(root, runsDir, true)
		if len(records) != 1 || records[0].Reason != dispatchtick.NoCommitUnknown {
			t.Fatalf("records = %+v, want one CLAIM_NO_COMMIT/unknown slot", records)
		}
		row := dispatchWitnessRow(t, payload)
		if got, ok := row["lease_released"]; ok {
			t.Fatalf("a crashed worker's lane was freed (lease_released=%v); it may be mid-write", got)
		}
		if _, ok := dispatchLiveLease(t, root); !ok {
			t.Fatalf("a crashed worker's lease must survive the sweep and expire by TTL")
		}
	})
}

// TestDispatchInProcessLaneLeaseRelease covers the second acquire site, the host-enroll
// path: it holds the SAME lane lease but never detaches, so no witness sweep will ever
// grade it and its lease stranded for the whole TTL every single time. The release rides
// the same fenced CAS, reading its token straight off the acquire result.
func TestDispatchInProcessLaneLeaseRelease(t *testing.T) {
	t.Run("in_process_lane_is_reacquirable_immediately", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "enroll-that-finished")
		lease := acquireDispatchLaneLease(root, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if acquired, _ := lease["acquired"].(bool); !acquired {
			t.Fatalf("fixture could not acquire: %+v", lease)
		}
		if got := releaseInProcessLaneLease(root, lease); got != "released" {
			t.Fatalf("in-process release = %q, want released", got)
		}
		t.Setenv("FAK_LEASE_OWNER", "peer-after-enroll")
		back := acquireDispatchLaneLease(root, "resolve-gateway-next", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if acquired, _ := back["acquired"].(bool); !acquired {
			t.Fatalf("a retired enrollment's lane must be re-acquirable within seconds, got %+v", back)
		}
	})

	t.Run("in_process_fence_refuses_a_reclaimed_lane", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "enroll-that-finished")
		lease := acquireDispatchLaneLease(root, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if acquired, _ := lease["acquired"].(bool); !acquired {
			t.Fatalf("fixture could not acquire: %+v", lease)
		}
		reissued, v, err := leaseref.NewInDir(root).AcquireFenced(context.Background(), leaseref.Record{
			ID: "resolve-gateway", Holder: dispatchMapString(lease, "holder"),
			TreeGlobs: []string{"internal/gateway/**"}, TTLSeconds: 1800,
		}, time.Now().Add(2*time.Hour))
		if err != nil || !v.OK {
			t.Fatalf("reclaim setup: %+v %v", v, err)
		}
		if got := releaseInProcessLaneLease(root, lease); got != "stale_lease" {
			t.Fatalf("in-process release = %q, want stale_lease — it must not free a reclaimed lane", got)
		}
		live, ok := dispatchLiveLease(t, root)
		if !ok || live.Generation != reissued.Generation {
			t.Fatalf("live lease = %+v ok=%v, want the reclaimer's generation %d intact", live, ok, reissued.Generation)
		}
	})

	// A fail-open acquire (no git repo -> acquired:false) carries no token at all, so the
	// release is a no-op instead of a blind delete against whatever the id names.
	t.Run("failopen_acquire_releases_nothing", func(t *testing.T) {
		root := t.TempDir()
		if got := releaseInProcessLaneLease(root, map[string]any{"acquired": false, "fail_open": true, "id": "resolve-gateway"}); got != "no_lease_id" {
			t.Fatalf("fail-open acquire release = %q, want no_lease_id", got)
		}
		if got := releaseInProcessLaneLease(root, map[string]any{"acquired": true, "id": "resolve-gateway", "holder": "owner-a"}); got != "no_fence_token" {
			t.Fatalf("zero-generation release = %q, want no_fence_token", got)
		}
	})
}

// dispatchSpawnerFunc is dispatchIssueWorkerSpawner's signature, aliased so the
// spawn-failure cases below can hand one in without restating fourteen parameters.
type dispatchSpawnerFunc = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error)

// dispatchSpawnFailureTick drives the REAL dispatchTickLiveSpawn — the function that
// acquires the lane lease — with the broker and the spawner stubbed to reproduce one
// failure edge. backend is a parameter so the argv-build refusal can be reached too. The
// lease is taken with a 1200+600 = 1800s TTL and the clock is never advanced, so nothing
// below can pass by waiting: only an explicit hand-back can free the lane.
func dispatchSpawnFailureTick(t *testing.T, root, backend string, broker func(launchBrokerAttempt) launchBrokerGrant, spawner dispatchSpawnerFunc) (map[string]any, error) {
	t.Helper()
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")   // no `fak guard` wrapper: the argv is not what is under test
	t.Setenv("FLEET_WORKER_WORKTREE", "0") // #3168 isolation off: no worktree to prepare or reap
	oldBroker, oldSpawner := launchSpawnBroker, dispatchIssueWorkerSpawner
	launchSpawnBroker = broker
	dispatchIssueWorkerSpawner = spawner
	t.Cleanup(func() { launchSpawnBroker = oldBroker; dispatchIssueWorkerSpawner = oldSpawner })

	opts := dispatchTickOptions{Backend: backend, Live: true, WorkerTimeoutS: 1200}
	pick := dispatchLanePick{Lane: "gateway", Tree: []string{"internal/gateway/**"}}
	return dispatchTickLiveSpawn(root, filepath.Join(root, dispatchtick.RunsDirName), opts, pick,
		"resolve-gateway", dispatchtick.Account{Tag: "seat-a"}, dispatchtick.WorkerLaunch{}, 5565,
		map[string]any{"prompt": "resolve #5565"}, map[string]any{},
		func(p map[string]any) map[string]any { return p })
}

// dispatchDenyingBroker refuses every launch, reproducing the SPAWN_BROKER_DENIED edge.
func dispatchDenyingBroker(a launchBrokerAttempt) launchBrokerGrant {
	return denyLaunchBrokerGrant(a, "unit-test-deny")
}

// dispatchAllowingBroker admits every launch, so the edge under test is the spawner's.
func dispatchAllowingBroker(a launchBrokerAttempt) launchBrokerGrant {
	return allowLaunchBrokerGrant(a, "unit-test-allow")
}

// dispatchLaneIsFree asserts the gateway lane is re-acquirable by a peer RIGHT NOW — the
// acceptance shape of #5565, and the only thing that distinguishes a real hand-back from a
// payload key that merely says one happened.
func dispatchLaneIsFree(t *testing.T, root string) {
	t.Helper()
	if _, ok := dispatchLiveLease(t, root); ok {
		t.Errorf("the abandoned lane lease is still on refs/fak/locks/ after the failed spawn")
	}
	t.Setenv("FAK_LEASE_OWNER", "peer-after-spawn-failure")
	back := acquireDispatchLaneLease(root, "resolve-gateway-next", "gateway", []string{"internal/gateway/**"}, 1800, "")
	if acquired, _ := back["acquired"].(bool); !acquired {
		t.Fatalf("a lane no worker ever ran in must be re-acquirable within seconds, got %+v", back)
	}
}

// TestDispatchSpawnFailureReleasesLaneLease is #5565: dispatchTickLiveSpawn acquires the
// lane lease as its first statement and then has several ways out before any worker is
// live. Each one used to return with the lane still leased for the full ~40-min TTL, and
// the ones that never produce a log stem are unreachable by the #4324 witness sweep — the
// only other releaser — so nothing could ever free them.
func TestDispatchSpawnFailureReleasesLaneLease(t *testing.T) {
	// The broker refused the launch: no process, no log stem, no fence sidecar.
	t.Run("broker_denied_hands_the_lane_back", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "tick-that-was-denied")
		got, err := dispatchSpawnFailureTick(t, root, "claude", dispatchDenyingBroker, nil)
		if err != nil {
			t.Fatalf("a denied launch is a payload, not an error: %v", err)
		}
		if got["verdict"] != "SPAWN_BROKER_DENIED" {
			t.Fatalf("verdict = %v, want SPAWN_BROKER_DENIED", got["verdict"])
		}
		if got["lease_release"] != "released" {
			t.Fatalf("lease_release = %v, want released", got["lease_release"])
		}
		dispatchLaneIsFree(t, root)
	})

	// The spawner itself errored: same shape, the other sidecar-less path.
	t.Run("spawner_error_hands_the_lane_back", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "tick-whose-spawner-failed")
		got, err := dispatchSpawnFailureTick(t, root, "claude", dispatchAllowingBroker,
			func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
				return dispatchSpawnResult{}, errors.New("exec: worker binary not found")
			})
		if err != nil {
			t.Fatalf("a spawn fault is a SPAWN_FAILED payload, not an error: %v", err)
		}
		if got["verdict"] != "SPAWN_FAILED" || got["lease_release"] != "released" {
			t.Fatalf("verdict/lease_release = %v/%v, want SPAWN_FAILED/released", got["verdict"], got["lease_release"])
		}
		dispatchLaneIsFree(t, root)
	})

	// The spawned process exited immediately. This slot DOES get a fence sidecar, so a
	// witness sweep could in principle reach it — but only through
	// dispatchWorkerExitReleasesLease, and the SILENT early exit's log tail carries no
	// terminal signature, so it grades NoCommitUnknown (the crash bucket) and the sweep
	// deliberately keeps the lease. The tick must therefore free it here.
	t.Run("early_exit_hands_the_lane_back", func(t *testing.T) {
		root := initRegionTestRepo(t)
		runsDir := filepath.Join(root, dispatchtick.RunsDirName)
		stem := filepath.Join(runsDir, "resolve-5565-20260802-050505")
		t.Setenv("FAK_LEASE_OWNER", "tick-whose-worker-died-at-once")
		got, err := dispatchSpawnFailureTick(t, root, "claude", dispatchAllowingBroker,
			func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
				if err := os.MkdirAll(runsDir, 0o755); err != nil {
					return dispatchSpawnResult{}, err
				}
				if err := os.WriteFile(stem+".log", []byte("# fak-spawn issue=5565 lane=gateway\n"), 0o644); err != nil {
					return dispatchSpawnResult{}, err
				}
				return dispatchSpawnResult{PID: deadDispatchPID, Log: stem + ".log", Issue: 5565, Lane: "gateway", Backend: "claude",
					EarlyExit: map[string]any{"alive": false, "silent": true, "wait_s": 0.4}}, nil
			})
		if err != nil {
			t.Fatalf("an early exit is a SPAWN_FAILED payload, not an error: %v", err)
		}
		if got["verdict"] != "SPAWN_FAILED" || got["lease_release"] != "released" {
			t.Fatalf("verdict/lease_release = %v/%v, want SPAWN_FAILED/released", got["verdict"], got["lease_release"])
		}
		// The premise of this case: the sweep's own token WAS written for this slot, and
		// the sweep still would not have freed it (an empty tail grades unknown).
		if _, ok := readDispatchLeaseFence(stem); !ok {
			t.Errorf("the early-exit slot must still carry its #4324 fence sidecar")
		}
		if dispatchWorkerExitReleasesLease(dispatchtick.WitnessRecord{
			Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.ClassifyNoCommitReason("# fak-spawn issue=5565 lane=gateway\n", 36)}, 0) {
			t.Errorf("a silent early exit must still grade as the crash bucket the sweep keeps")
		}
		dispatchLaneIsFree(t, root)
	})

	// THE DANGEROUS HALF, the same one #4324 guards: a janitor reclaimed and re-issued
	// this lane between the acquire and the failure. The broker stub is the injection
	// point because it runs after the acquire and before the release. The reclaimer
	// deliberately carries the SAME holder string (one box, one FAK_LEASE_OWNER), so the
	// generation is the ONLY thing standing between this release and freeing a lane a
	// different live worker owns.
	t.Run("fence_refuses_a_reclaimed_lane", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "tick-that-was-denied")
		var reissued leaseref.Record
		got, err := dispatchSpawnFailureTick(t, root, "claude", func(a launchBrokerAttempt) launchBrokerGrant {
			rec, v, aerr := leaseref.NewInDir(root).AcquireFenced(context.Background(), leaseref.Record{
				ID: "resolve-gateway", Holder: dispatchLeaseHolder(),
				TreeGlobs: []string{"internal/gateway/**"}, TTLSeconds: 1800,
			}, time.Now().Add(2*time.Hour))
			if aerr != nil || !v.OK {
				t.Fatalf("reclaim setup: %+v %v", v, aerr)
			}
			reissued = rec
			return denyLaunchBrokerGrant(a, "unit-test-deny")
		}, nil)
		if err != nil {
			t.Fatalf("live spawn: %v", err)
		}
		if got["lease_release"] != "stale_lease" {
			t.Fatalf("lease_release = %v, want stale_lease — the tick must not free a reclaimed lane", got["lease_release"])
		}
		live, ok := dispatchLiveLease(t, root)
		if !ok {
			t.Fatalf("the RECLAIMED lease was deleted by a stale release — the peer's lane is now unfenced")
		}
		if live.Generation != reissued.Generation {
			t.Fatalf("live lease generation = %d, want the reclaimer's %d", live.Generation, reissued.Generation)
		}
	})

	// The argv-build refusal returns (nil, err) rather than a payload, and it too sits
	// after the acquire. The error must still propagate AND the lane must still be freed.
	t.Run("argv_build_refusal_hands_the_lane_back", func(t *testing.T) {
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "tick-with-an-unknown-backend")
		if _, err := dispatchSpawnFailureTick(t, root, "no-such-backend", dispatchAllowingBroker, nil); err == nil {
			t.Fatalf("an unknown backend must still fail the tick")
		}
		dispatchLaneIsFree(t, root)
	})

	// A failed hand-back never fails the tick: with no git repo the acquire fails open
	// (nothing was leased), so the release is a recorded no-op, not a blind delete.
	t.Run("failopen_acquire_records_a_noop_and_never_fails_the_tick", func(t *testing.T) {
		root := t.TempDir()
		t.Chdir(root)
		got, err := dispatchSpawnFailureTick(t, root, "claude", dispatchDenyingBroker, nil)
		if err != nil {
			t.Fatalf("a fail-open acquire must not fail the tick: %v", err)
		}
		if got["lease_release"] != "no_lease_id" {
			t.Fatalf("lease_release = %v, want no_lease_id", got["lease_release"])
		}
	})
}

// TestDispatchLeaseZeroGenerationIsLoadBearing is the anti-vacuity witness for the one
// invariant the release path may not lose. leaseref.ReleaseFenced states it plainly:
// "presenting a non-zero generation ADDITIONALLY requires it to match the live lease's" —
// it skips the comparison entirely when either side is 0, leaving only the holder-string
// check, and holders DO collide (a reused daemon presents the same host:pid, and one box
// under one FAK_LEASE_OWNER presents one string for every tick). This case runs the
// mutant directly: the SAME reclaimed lane that the guarded release refuses is FREED when
// a zero-generation token reaches the fenced delete. Nothing here is a call shape the
// product makes — dispatchLeaseFenceReleasable stands between every release site and this
// outcome — it exists so a future edit that drops the generation cannot pass silently.
func TestDispatchLeaseZeroGenerationIsLoadBearing(t *testing.T) {
	reclaimed := func(t *testing.T) (string, string) {
		t.Helper()
		root := initRegionTestRepo(t)
		t.Setenv("FAK_LEASE_OWNER", "one-box-one-owner-string")
		lease := acquireDispatchLaneLease(root, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 1800, "")
		if acquired, _ := lease["acquired"].(bool); !acquired {
			t.Fatalf("fixture could not acquire: %+v", lease)
		}
		// The janitor reclaim: a DIFFERENT live worker now owns the lane, at a bumped
		// generation but — deliberately — under the identical holder string.
		if _, v, err := leaseref.NewInDir(root).AcquireFenced(context.Background(), leaseref.Record{
			ID: "resolve-gateway", Holder: dispatchMapString(lease, "holder"),
			TreeGlobs: []string{"internal/gateway/**"}, TTLSeconds: 1800,
		}, time.Now().Add(2*time.Hour)); err != nil || !v.OK {
			t.Fatalf("reclaim setup: %+v %v", v, err)
		}
		return root, dispatchMapString(lease, "holder")
	}

	// The guard in place: refused, the peer keeps its lane.
	root, holder := reclaimed(t)
	if got := releaseInProcessLaneLease(root, map[string]any{
		"acquired": true, "id": "resolve-gateway", "holder": holder, "generation": int64(1),
	}); got != "stale_lease" {
		t.Fatalf("guarded release = %q, want stale_lease", got)
	}
	if _, ok := dispatchLiveLease(t, root); !ok {
		t.Fatalf("the guarded release freed the reclaimer's lane")
	}

	// The mutant: the same release with the generation dropped to zero frees the lane the
	// peer now owns — two writers in one lane. This MUST keep failing to be a guard.
	mutantRoot, mutantHolder := reclaimed(t)
	if _, outcome := releaseLaneLeaseFenced(mutantRoot, "resolve-gateway", dispatchLeaseFence{Holder: mutantHolder}); outcome != "released" {
		t.Fatalf("zero-generation release = %q; if this no longer frees a reclaimed lane the fence has a second line of defence and this test should be re-derived, not deleted", outcome)
	}
	if _, ok := dispatchLiveLease(t, mutantRoot); ok {
		t.Fatalf("zero-generation release left the lease standing; see above")
	}
}

// TestDispatchWorkerExitReleasesLease pins the normal-exit vocabulary directly: which
// terminal states hand the lane back, and which deliberately do not.
func TestDispatchWorkerExitReleasesLease(t *testing.T) {
	release := []dispatchtick.WitnessRecord{
		{Claim: dispatchtick.ClaimWitnessed},
		{Claim: dispatchtick.ClaimUnwitnessed},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitSelfModify},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitPolicyBlock},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitAuthWall},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitUsageCap},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitModelUnknown},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitRateLimit},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitOffTrunk},
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitBannerNoop},
	}
	for _, rec := range release {
		if !dispatchWorkerExitReleasesLease(rec, 0) {
			t.Errorf("%+v must hand its lane back: it stopped on purpose", rec)
		}
		// Stranded lane-scoped edits are mid-write evidence that overrides every
		// otherwise-normal exit.
		if dispatchWorkerExitReleasesLease(rec, 1) {
			t.Errorf("%+v left stranded edits and must keep its lease", rec)
		}
	}
	keep := []dispatchtick.WitnessRecord{
		{Claim: dispatchtick.ClaimNoCommit, Reason: dispatchtick.NoCommitUnknown},
		{Claim: dispatchtick.ClaimNoCommit},
		{Claim: ""},
		{Claim: "CLAIM_SOMETHING_NEW"},
	}
	for _, rec := range keep {
		if dispatchWorkerExitReleasesLease(rec, 0) {
			t.Errorf("%+v is not a proven normal exit and must keep its lease (fail closed)", rec)
		}
	}
}

// TestDispatchLeaseFenceTokenStrength pins the token floor. ReleaseFenced skips its
// generation comparison when either side is 0, so a zero-generation token degrades to a
// holder-string match — and two ticks on one box share a holder string. Such a token
// must never be written and never authorize a release.
func TestDispatchLeaseFenceTokenStrength(t *testing.T) {
	if !dispatchLeaseFenceReleasable(dispatchLeaseFence{Holder: "owner-a", Generation: 1}) {
		t.Fatalf("a holder + non-zero generation is the releasable token")
	}
	for _, weak := range []dispatchLeaseFence{
		{Holder: "owner-a", Generation: 0},
		{Holder: "", Generation: 3},
		{Holder: "   ", Generation: 3},
		{},
	} {
		if dispatchLeaseFenceReleasable(weak) {
			t.Errorf("token %+v cannot prove ownership and must never authorize a release", weak)
		}
	}
	// A refused acquire carries no token at all, so nothing is persisted for it.
	root := t.TempDir()
	log := filepath.Join(root, "resolve-1-20260802-040404.log")
	for _, lease := range []map[string]any{
		nil,
		{"acquired": false, "refused": true, "holder": "owner-a"},
		{"acquired": true, "holder": "owner-a"},
		{"acquired": true, "holder": "", "generation": int64(2)},
	} {
		if p := writeDispatchLeaseFenceSidecar(log, lease); p != "" {
			t.Errorf("lease %+v must persist no fencing token, wrote %s", lease, p)
		}
	}
	if p := writeDispatchLeaseFenceSidecar(log, map[string]any{"acquired": true, "holder": "owner-a", "generation": int64(2)}); p == "" {
		t.Fatalf("an acquired lease with a real generation must persist its token")
	}
	fence, ok := readDispatchLeaseFence(filepath.Join(root, "resolve-1-20260802-040404"))
	if !ok || fence.Holder != "owner-a" || fence.Generation != 2 {
		t.Fatalf("round-tripped token = %+v ok=%v, want owner-a/2", fence, ok)
	}
}
