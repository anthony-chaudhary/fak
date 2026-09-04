// Package wiplease projects the ATTRIBUTED DIRTY TREE into the lease geometry the
// shared-tree admission decision already understands, so a session's real footprint is
// visible to peers from its first dirty byte instead of only during cleanup.
//
// THE GAP THIS CLOSES, measured on this fleet. The repo runs two halves of one fact and
// no bridge between them:
//
//   - The LEASE plane (refs/fak/locks/*, internal/leaseref) knows OWNERS WITHOUT PATHS.
//     Of 24 live leases read on 2026-08-13, 21 were `session-*` heartbeat records
//     carrying only {id, host, pcb_state, updated_at, ttl_seconds} — liveness with no
//     `tree_globs` at all. Exactly 2 declared a footprint. So the geometry
//     internal/laneadmit decides on is EMPTY for ~88% of live sessions, and laneadmit's
//     own conservative rule ("an ABSENT tree is UNKNOWN blast radius") is the only thing
//     standing in for a fact the tree could have supplied.
//   - The WIP plane (internal/wipattr) knows PATHS WITHOUT OWNERS, and knows them only
//     RETROACTIVELY: attribution reads checkpoints, a checkpoint snapshots a delta, so a
//     session's first edit is ORPHAN by construction and stays that way until its next
//     checkpoint boundary.
//
// Neither half can answer the question a peer actually asks before starting work — "who
// is editing this path right now?" — and the ~226-path dirty tree those 24 sessions share
// is the standing evidence that nobody can.
//
// WHY A PROJECTION AND NOT A NEW GATE. The overlap decision already exists and is good:
// laneadmit.Decide takes requested tree globs against live lease globs and refuses with
// the closed-vocabulary COLLISION_RISK, carrying the conflicting leases as evidence. A
// second admission gate over the same geometry would be a duplicate decision that can
// disagree with the first. What is missing is not a decision — it is INPUT. So this leaf
// emits laneadmit.Lease values and stops. It decides nothing.
//
// ACTIVE IS NOT THE SAME FACT AS OCCUPIED, and the split is the point. A dirty path whose
// owner is LIVE is another session's in-flight work: real blast radius, project it as a
// lease so the existing decision serializes against it. A dirty path whose owner is DEAD,
// or that no checkpoint claims at all, is the opposite — it is the abandoned WIP a peer
// should be able to ADOPT, and projecting it as a blocking lease would wall off exactly
// the work that most needs picking up, permanently, since no live holder remains to
// release it. Reclaimable carries those with the reason they are reclaimable, so
// "somebody is working here" and "somebody left this here" stop being one undifferentiated
// dirty-tree fact.
//
// Pure: no git, no clock, no I/O. The caller reads attributions (wipattr.Attribute) and
// the live-session set (leaseref) and hands both in.
package wiplease

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// DefaultIDPrefix namespaces projected lease ids so an operator reading a refusal can
// tell a lease INFERRED from the dirty tree from one a session deliberately acquired.
// The two carry different authority: an acquired lease is a promise, a projected one is
// an observation, and a refusal that conflates them would misreport why it refused.
const DefaultIDPrefix = "wip:"

// Reclaim reasons — the closed vocabulary for why a dirty path is NOT active work.
const (
	// ReclaimOwnerDead: a checkpoint attributes the path, but that session is gone.
	// The delta is recoverable and its provenance is known.
	ReclaimOwnerDead = "OWNER_DEAD"
	// ReclaimUnattributed: no checkpoint records the edit. Either nobody ever
	// checkpointed it, or it is a live session's first edit before its first
	// checkpoint — the two are indistinguishable from attribution alone, which is
	// precisely why this is reported rather than treated as free.
	ReclaimUnattributed = "UNATTRIBUTED"
)

// Reclaimable is one dirty path that no live session is provably working on.
type Reclaimable struct {
	Path   string   `json:"path"`
	Owner  string   `json:"owner,omitempty"`  // the dead sole owner, when OWNED
	Owners []string `json:"owners,omitempty"` // all dead claimants, when SHARED
	Reason string   `json:"reason"`
}

// Occupancy is the projection: who is provably working where, and what is left over.
type Occupancy struct {
	// Active is one lease per LIVE owner, its Tree being that owner's real dirty
	// footprint. Feed straight into laneadmit.Decide alongside the declared leases.
	Active []laneadmit.Lease `json:"active"`
	// Reclaimable is dirty work with no live owner — adoptable, not blocking.
	Reclaimable []Reclaimable `json:"reclaimable"`
	// Undeclared counts live owners whose footprint came ONLY from the dirty tree.
	// It is the adoption gap in one number: how many sessions the lease plane could
	// not have told you anything about.
	Undeclared int `json:"undeclared"`
}

// Options tunes the projection. The zero value is the documented default.
type Options struct {
	IDPrefix string // lease-id namespace; "" => DefaultIDPrefix
	// Declared is the set of session ids that ALREADY hold a lease carrying tree
	// globs. Those sessions are still projected (the dirty tree may be wider than
	// what they declared) but they do not count toward Undeclared.
	Declared map[string]bool
}

// Project folds attributions plus the live-session set into an Occupancy.
//
// Contract: WIP lease projection is fail-closed, pure, and deterministic.
// Invariant: WIP lease projection is fail-closed and deterministic.
// Guard: Absent or empty session IDs fail toward reclaimable rather than active blocking leases.
//
// live maps session id -> alive. A session absent from the map is treated as NOT live:
// the projection fails toward "reclaimable", never toward "blocked". That direction is
// deliberate. Over-reporting a lease would serialize a peer behind a session that no
// longer exists, and the tree carries no mechanism to release a lease whose holder is
// already gone — the deadlock would be permanent and manual. Under-reporting one costs
// at worst a collision the existing retrospective surfaces (wipattr.SweepGuard, the
// reconcile family) already detect and can repair.
func Project(attrs []wipattr.Attribution, live map[string]bool, opts Options) Occupancy {
	prefix := opts.IDPrefix
	if prefix == "" {
		prefix = DefaultIDPrefix
	}
	isLive := func(id string) bool { return id != "" && live[id] }

	trees := map[string]map[string]bool{} // owner -> set of paths
	addPath := func(owner, path string) {
		if trees[owner] == nil {
			trees[owner] = map[string]bool{}
		}
		trees[owner][path] = true
	}

	out := Occupancy{}
	for _, a := range attrs {
		if a.File == "" {
			continue
		}
		switch a.State {
		case wipattr.AttrOwned:
			if isLive(a.Owner) {
				addPath(a.Owner, a.File)
				continue
			}
			out.Reclaimable = append(out.Reclaimable, Reclaimable{
				Path: a.File, Owner: a.Owner, Reason: ReclaimOwnerDead,
			})
		case wipattr.AttrShared:
			// Ambiguous ownership resolves CONSERVATIVELY for the active case: every
			// live claimant gets the path in its footprint. A shared path cannot be
			// assigned to one session without guessing, and guessing wrong here means
			// telling a peer a path is free while a live session edits it.
			liveOwners := 0
			for _, o := range a.Owners {
				if isLive(o) {
					addPath(o, a.File)
					liveOwners++
				}
			}
			if liveOwners == 0 {
				dead := append([]string(nil), a.Owners...)
				sort.Strings(dead)
				out.Reclaimable = append(out.Reclaimable, Reclaimable{
					Path: a.File, Owners: dead, Reason: ReclaimOwnerDead,
				})
			}
		case wipattr.AttrOrphan:
			out.Reclaimable = append(out.Reclaimable, Reclaimable{
				Path: a.File, Reason: ReclaimUnattributed,
			})
		}
	}

	owners := make([]string, 0, len(trees))
	for o := range trees {
		owners = append(owners, o)
	}
	sort.Strings(owners)
	for _, o := range owners {
		paths := make([]string, 0, len(trees[o]))
		for p := range trees[o] {
			paths = append(paths, p)
		}
		sort.Strings(paths)
		out.Active = append(out.Active, laneadmit.Lease{
			ID:     prefix + o,
			Holder: o,
			Tree:   paths,
		})
		if !opts.Declared[o] {
			out.Undeclared++
		}
	}
	sort.Slice(out.Reclaimable, func(i, j int) bool {
		if out.Reclaimable[i].Path != out.Reclaimable[j].Path {
			return out.Reclaimable[i].Path < out.Reclaimable[j].Path
		}
		return out.Reclaimable[i].Reason < out.Reclaimable[j].Reason
	})
	return out
}
