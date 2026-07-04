// lease.go — rung H2 (issue #1895): lease continuity across legs. The Baton type
// (rung C1, #1870) carries ProgressCursor.HeldRegion — the lane/path region (globs)
// the prior leg held — precisely so the successor re-acquires the SAME lease and
// does not collide with peers on the shared tree (CONCEPT-PERPETUAL-SESSIONS,
// "held_region to re-acquire (disjointness)"). This file is the consumer of that
// field: the leg-boundary re-acquire step that asks the DOS lane arbiter —
// internal/laneadmit, the in-binary twin of `dos arbitrate`, the same kernel
// session.Rewind (#2427) and the dispatch loop consult — whether leg N+1 may take
// the region now.
//
// Continuity, not collision. The relay re-acquires under a lease id that is STABLE
// across every leg of the relay (derived from RelayID), so leg N+1's re-acquire is
// the prior leg's own lease renewed, not a second lease racing it. laneadmit treats
// a live lease whose id equals the request's LeaseID as the caller's own and never
// conflicts with it — that is the continuity property. A peer worker holding an
// OVERLAPPING tree is the one collision that must refuse, with the arbiter's closed
// lane-conflict reason (COLLISION_RISK); on that refusal the leg defers rather than
// writes. No new lease system is minted (out of scope per #1895): this reuses
// dos arbitrate, exactly the way Rewind reuses it for workspace restore.
//
// This is a pure, no-I/O fold: the caller supplies the live leases (projected from
// refs/fak/locks/* via leaseref) and the dos.toml lane taxonomy. The rung J7
// witness (#1909) exercises it as a dos-arbitrate-level simulation — no real
// multi-process fleet.
package relay

import "github.com/anthony-chaudhary/fak/internal/laneadmit"

// SurfaceRelay is the execution-surface name a relay leg identifies as when it
// asks the lane arbiter to re-acquire its region — alongside laneadmit's
// SurfaceDispatch / SurfaceLoop / SurfaceManual and session's SurfaceRestore, so
// every surface that can mutate the shared tree asks the same kernel.
const SurfaceRelay = "relay"

// LeaseID mints the stable lease id a relay re-acquires under: the SAME id for
// every leg of a relay (derived from RelayID), so leg N+1's re-acquire is the
// prior leg's own lease renewed, not a second lease racing it. laneadmit treats a
// live lease whose id equals the request's LeaseID as the caller's own and never
// conflicts with it — the continuity property. A peer's lease carries a different
// id and conflicts normally on an overlapping tree.
func LeaseID(relayID string) string {
	return laneadmit.LeaseID(SurfaceRelay, "", relayID)
}

// Reacquire is the leg-boundary re-acquire step. The successor reads the prior
// leg's baton, takes its ProgressCursor.HeldRegion, and asks the DOS lane arbiter
// whether it may re-acquire that region now, under the relay's stable lease id.
//
//   - A free region (no overlapping peer) admits: leg N+1 takes the same lease
//     and may write.
//   - A live peer over an overlapping tree refuses with the arbiter's closed
//     lane-conflict reason (COLLISION_RISK), naming the holder; the leg defers
//     instead of colliding.
//   - The relay's own prior-leg lease (same LeaseID) never conflicts — the
//     continuity property — so a renew across the leg boundary is always clean.
//
// It is pure: no clock, no I/O. The caller supplies the live leases and the
// dos.toml taxonomy. The returned Verdict is the arbiter's decision verbatim;
// Reacquire adds no force path (a held region is deferred, not overridden).
func Reacquire(b Baton, live []laneadmit.Lease, tax laneadmit.Taxonomy) laneadmit.Verdict {
	return laneadmit.Decide(laneadmit.Request{
		Surface: SurfaceRelay,
		Tree:    b.ProgressCursor.HeldRegion,
		Holder:  b.RelayID,
		LeaseID: LeaseID(b.RelayID),
	}, live, tax)
}
