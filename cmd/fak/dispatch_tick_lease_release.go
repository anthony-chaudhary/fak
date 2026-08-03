package main

// Release-on-exit for the dispatch tick's lane lease (#4324). Until now a lane lease
// acquired by `acquireDispatchLaneLease` had exactly ONE way to end: TTL expiry. The
// tick that acquires it spawns a DETACHED worker and returns immediately, so there is
// no in-process defer that could ever hand the lane back — the lease strands for the
// full worker-timeout + margin window (~40 min) after the worker has already finished,
// and every peer that would not actually collide is refused LANE_LEASE_HELD against a
// holder that no longer exists. That phantom share is exactly what the counter landed
// in 025950c23 (blocking_stranded_count) was added to measure.
//
// THE SEAM. The one place in the Go dispatch stack that OBSERVES a worker finishing is
// the witness sweep (dispatch_tick_witness.go): it walks the runs dir, gates on a
// provably-DEAD pid, and grades each finished slot exactly once. That is the finalizer
// this lease never had — it already lands+reaps the worker's worktree and reverts its
// stranded edits under the same authority. So the release rides there, at the END of
// the per-slot body: the lease stays held for the whole time the sweep is still
// mutating the worker's lane on its behalf, and is handed back only once nothing more
// will be written under it.
//
// WHY NOT THE ANNOUNCE PLANE. internal/leaseref/announce.go carries an AnnounceRelease
// action and FoldAnnouncements drops a lease id on it, which looks like the release
// seam but is not: that file states its own boundary — "A COMMENT IS EVIDENCE, NEVER A
// LOCK ... FoldAnnouncements' output is ADVISORY visibility context ... never an
// admission input on its own. Admission stays the refs/fak/locks compare-and-swap".
// Folding a release announce would make a lease look free to a human reading the
// coordination issue while refs/fak/locks/<lane> still refused every peer. The lane is
// re-acquirable only when PLANE 0 lets go, so the release must be the fenced ref
// delete; the announce plane stays a separate visibility concern.
//
// THE HAZARD (why this is fenced, not a delete). A release is a strictly more dangerous
// primitive than an acquire: freeing a lease that a janitor already reclaimed and
// reassigned puts TWO writers in one lane, which is worse than the stranding it cures.
// So the release presents the fencing token the acquire was WRITTEN under and goes
// through leaseref.ReleaseFenced, whose CAS discipline is the acquire's mirror: it
// re-reads the live lease, admits only when the caller is still the live holder AND the
// presented generation still matches, and deletes under an `update-ref -d <ref> <old>`
// old-value compare-and-swap. A reclaimed-and-reacquired lane has advanced its
// generation (AcquireFenced's TRANSITION bump), so the stale release reads STALE_LEASE
// and the peer's lane is left alone.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// dispatchLeaseFenceSidecarSuffix records the fencing token a slot's lane lease was
// acquired under, beside the .lease-id that names the lease itself. The witness sweep
// runs in a LATER tick process that never saw the acquire, so without this durable
// token it could not prove the lease is still the exited worker's — and an unprovable
// release is the two-writers hazard, not a cure for stranding. Absent for every slot
// spawned before this seam, and that absence is honest: those leases keep the old
// TTL-only lifetime.
const dispatchLeaseFenceSidecarSuffix = ".lease-fence.json"

// dispatchLeaseFence is the durable half of the fencing token: WHO held the lane lease
// and at WHICH generation. Deliberately the minimum needed to satisfy
// leaseref.ReleaseFenced's holder+generation check — nothing about the worker, the
// account, or the box rides along.
type dispatchLeaseFence struct {
	Holder     string `json:"holder"`
	Generation int64  `json:"generation"`
}

// writeDispatchLeaseFenceSidecar persists the fencing token from a successful
// acquireDispatchLaneLease result beside the worker's log, returning the path written
// ("" when nothing was). Best-effort in the same shape as the .model / .zone sidecars:
// a slot whose token cannot be persisted simply keeps the pre-#4324 TTL-only lifetime.
// A refused/fail-open acquire and a zero generation write NOTHING — see
// dispatchLeaseFenceReleasable for why a zero token may never authorize a release.
func writeDispatchLeaseFenceSidecar(log string, lease map[string]any) string {
	path := dispatchtick.SidecarPath(log, dispatchLeaseFenceSidecarSuffix)
	if path == "" || len(lease) == 0 {
		return ""
	}
	if acquired, _ := lease["acquired"].(bool); !acquired {
		return ""
	}
	fence := dispatchLeaseFence{
		Holder:     strings.TrimSpace(dispatchMapString(lease, "holder")),
		Generation: dispatchLeaseGeneration(lease["generation"]),
	}
	if !dispatchLeaseFenceReleasable(fence) {
		return ""
	}
	b, err := json.Marshal(fence)
	if err != nil {
		return ""
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return ""
	}
	return path
}

// dispatchLeaseGeneration reads a generation off the lease map through both dialects it
// can arrive in: the in-process int64 the fenced acquire returns, and the float64 a
// JSON round-trip (a startup bundle, a recorded payload) yields. Anything else is 0 —
// which is refused, never guessed.
func dispatchLeaseGeneration(v any) int64 {
	switch n := v.(type) {
	case int64:
		return n
	case int:
		return int64(n)
	case float64:
		return int64(n)
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0
		}
		return i
	}
	return 0
}

// dispatchLeaseFenceReleasable reports whether a recorded token is strong enough to
// authorize a release. BOTH halves are required, and the generation floor is the
// load-bearing one: ReleaseFenced skips its generation comparison when either side is
// 0, so a zero-generation token degrades to a holder-string match alone — and two ticks
// on one box share a holder string (FAK_LEASE_OWNER, or host:pid of a reused daemon).
// A reclaim-and-reacquire between those two ticks would then pass the holder check and
// free a lane a DIFFERENT live worker owns. Refusing the zero token costs only the
// pre-#4324 behaviour (TTL expiry) and removes the sole path to a wrong release.
func dispatchLeaseFenceReleasable(fence dispatchLeaseFence) bool {
	return strings.TrimSpace(fence.Holder) != "" && fence.Generation > 0
}

// readDispatchLeaseFence loads a slot's fencing token. ok is false for an absent,
// unreadable, malformed, or under-strength token — every one of which means "cannot
// prove this lease is ours", so the caller must NOT release.
func readDispatchLeaseFence(stem string) (dispatchLeaseFence, bool) {
	b, err := fsReadFile(stem + dispatchLeaseFenceSidecarSuffix)
	if err != nil {
		return dispatchLeaseFence{}, false
	}
	var fence dispatchLeaseFence
	if err := json.Unmarshal(b, &fence); err != nil {
		return dispatchLeaseFence{}, false
	}
	if !dispatchLeaseFenceReleasable(fence) {
		return dispatchLeaseFence{}, false
	}
	return fence, true
}

// dispatchWorkerExitReleasesLease is the NORMAL-EXIT gate: only a worker that reached a
// terminal state it chose hands its lane back early. A crashing / panicking / killed
// worker deliberately does NOT — its lane may be mid-write, and TTL expiry plus the
// existing dead-holder reclaim (internal/leaseref/cascade.go) is the correct, slower
// path there. Fails CLOSED: an unrecognized claim keeps the lease.
//
// Releases:
//   - CLAIM_WITNESSED / CLAIM_UNWITNESSED — the worker LANDED a commit, so its write
//     reached the trunk and finished. (The unwitnessed grade is about claim quality,
//     not about whether the write completed.)
//   - CLAIM_NO_COMMIT with a NAMED stop: self_modify / policy_block (a guard refused it
//     up front), auth_wall / usage_cap / model_unknown / rate_limit (it hit a wall and
//     stopped), off_trunk (refused at the commit), banner_noop (it never started). Each
//     is a worker that stopped on purpose with a recognizable terminal signature.
//
// Keeps the lease:
//   - CLAIM_NO_COMMIT with reason "unknown" — the classifier found NO terminating
//     signature in the log tail. That is exactly the crash / panic / SIGKILL bucket:
//     the worker vanished mid-turn and may have left the lane half-written.
//   - stranded > 0 — the sweep found (and stashed) uncommitted lane-scoped edits this
//     worker left behind. Whatever the log tail says, stranded edits ARE the mid-write
//     evidence, so the lane keeps its fence until the TTL.
func dispatchWorkerExitReleasesLease(rec dispatchtick.WitnessRecord, stranded int) bool {
	if stranded > 0 {
		return false
	}
	switch rec.Claim {
	case dispatchtick.ClaimWitnessed, dispatchtick.ClaimUnwitnessed:
		return true
	case dispatchtick.ClaimNoCommit:
		switch rec.Reason {
		case dispatchtick.NoCommitSelfModify, dispatchtick.NoCommitPolicyBlock,
			dispatchtick.NoCommitAuthWall, dispatchtick.NoCommitUsageCap,
			dispatchtick.NoCommitModelUnknown, dispatchtick.NoCommitRateLimit,
			dispatchtick.NoCommitOffTrunk, dispatchtick.NoCommitBannerNoop:
			return true
		}
	}
	return false
}

// dispatchLeaseReleaser is the seam the witness sweep fires the release through,
// injectable so a test can pin the call site without a real ref store.
var dispatchLeaseReleaser = releaseDispatchLaneLeaseFenced

// releaseDispatchLaneLeaseFenced hands one finished worker's lane lease back through
// leaseref.ReleaseFenced, presenting the fencing token recorded at spawn. It reports
// the lease id it freed (or "") and a short machine-readable outcome for the graded row.
//
// WHAT THE FENCE REFUSES. ReleaseFenced re-reads the live lease and admits the CAS
// delete only when the caller is still the live holder AND the presented generation
// still matches; a lane that was reclaimed and re-acquired has advanced its generation,
// so the release reads STALE_LEASE and the ref is left untouched — a peer that now owns
// the lane keeps it. A ref that moved between the read and the delete loses the
// old-value CAS and reads LEASE_CONTENDED. Both are refusals, not errors.
//
// A FAILED RELEASE NEVER FAILS THE SWEEP. Every outcome except a clean OK — no token,
// no lease id, a git fault, a fence refusal — degrades to exactly today's behaviour: the
// lease keeps its TTL and expires as it always has. The reason is returned (and surfaced
// on the graded row) and nothing is propagated to the caller.
func releaseDispatchLaneLeaseFenced(root, stem string) (string, string) {
	id := readResolveLeaseID(stem, "")
	if id == "" {
		return "", "no_lease_id"
	}
	fence, ok := readDispatchLeaseFence(stem)
	if !ok {
		// No provable token: releasing anyway is the two-writers hazard. Keep the TTL.
		return "", "no_fence_token"
	}
	return releaseLaneLeaseFenced(root, id, fence)
}

// releaseInProcessLaneLease is the SAME fenced release for the acquire site that never
// detaches: the host-enroll path (dispatch_tick_hostenroll.go) takes the identical lane
// lease, runs its microagent to completion in THIS process, and returns — so no witness
// sweep will ever grade it and the lease strands for the whole TTL every single time.
// Here the fencing token needs no sidecar at all: the acquire result is still in hand,
// so the token is read straight off it. Returns the outcome word for the payload.
func releaseInProcessLaneLease(root string, lease map[string]any) string {
	id := strings.TrimSpace(dispatchMapString(lease, "id"))
	if acquired, _ := lease["acquired"].(bool); !acquired || id == "" {
		return "no_lease_id"
	}
	fence := dispatchLeaseFence{
		Holder:     strings.TrimSpace(dispatchMapString(lease, "holder")),
		Generation: dispatchLeaseGeneration(lease["generation"]),
	}
	if !dispatchLeaseFenceReleasable(fence) {
		return "no_fence_token"
	}
	_, outcome := releaseLaneLeaseFenced(root, id, fence)
	return outcome
}

// releaseLaneLeaseFenced is the shared fenced delete both release sites end in: present
// the recorded holder AND generation to leaseref.ReleaseFenced and let its CAS decide.
func releaseLaneLeaseFenced(root, id string, fence dispatchLeaseFence) (string, string) {
	v, err := leaseref.NewInDir(root).ReleaseFenced(context.Background(), id, fence.Holder, fence.Generation, time.Now())
	if err != nil {
		return "", "release_error"
	}
	if !v.OK {
		// The holder/generation check refused: the lane is no longer provably ours.
		// The reason is a closed leaseref vocabulary word (STALE_LEASE /
		// LEASE_CONTENDED); the live holder's identity is deliberately NOT logged.
		fmt.Fprintf(os.Stderr, "fak dispatch: lane lease %s was not handed back at exit (%s); it keeps its TTL (#4324)\n", id, v.Reason)
		return "", strings.ToLower(v.Reason)
	}
	return id, "released"
}
