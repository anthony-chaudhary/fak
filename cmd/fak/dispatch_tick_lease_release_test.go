package main

// #4324 release-on-exit for the dispatch lane lease. The acceptance criterion is a
// LATENCY one — "a finished worker's lane is re-acquirable by a peer within seconds,
// not the TTL window" — so every case here acquires with a full-length TTL (1800s, the
// production worker-timeout order) and never advances the clock. Nothing in these tests
// can pass by waiting: the only mechanism that can hand the lane back inside the test's
// runtime is the explicit fenced release.

import (
	"context"
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
