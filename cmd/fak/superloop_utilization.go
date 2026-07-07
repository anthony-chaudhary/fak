package main

// superloop_utilization.go — the impure shell readers for the KindUtilization members
// of the `run-the-night` super loop (the overnight productivity meta-loop). Unlike a
// scorecard member (a committed baseline value) or a loop member (a ledger liveness
// fold), a utilization member's status is read LIVE at walk time: its debt is UNUSED
// CAPACITY — a resource the night is paying for and not spending.
//
// Keeping the live probing HERE preserves the package split the rest of superloop keeps:
// internal/superloop stays pure (it folds MemberStatus it is handed and reads no clock,
// no disk, no fleet), and the shell does the live reads and hands in the measured debt.
// Two pools are measured:
//
//	account-limits  — offerable-but-idle account SEATS. rotationHeadroom() bands each
//	                  account bucket into offerable (>1), unknown (0), or walled (<0).
//	                  An OFFERABLE bucket is a limit with room to give; if it is not
//	                  being spent it is wasted headroom. Debt = offerable buckets (room
//	                  available to put to work). A walled bucket is NOT debt here — its
//	                  limit is already fully used (the good, expected overnight state).
//	node-resources  — up-but-idle lab BOXES (Mac, A100s, dgx). The same fleet snapshot
//	                  `fak lab status` folds: a box that is reachable and idle/draining is
//	                  silicon that is on but not serving. Debt = idle + draining boxes,
//	                  plus one unit per whole-fleet GPU-idle when a GPU-util stat is
//	                  reported and the fleet is under-busy.

import (
	"bytes"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// utilization reads one KindUtilization member's LIVE unused-capacity debt. An unknown
// Ref is refused as UNMEASURED (a registry bug must red the walk, never vanish), the
// same posture descend() takes for an unknown sub-super-loop.
func (c *superloopCollector) utilization(m superloop.Member) superloop.MemberStatus {
	switch m.Ref {
	case "account-limits":
		return accountLimitsUtilization(m)
	case "node-resources":
		return nodeResourcesUtilization(m)
	default:
		return superloop.MemberStatus{
			Member:   m,
			Measured: false,
			Detail:   fmt.Sprintf("unknown utilization pool %q (registry drift; try `fak superloop list`)", m.Ref),
		}
	}
}

// accountLimitsUtilization folds the live rotation-headroom signal into an unused-seat
// debt. rotationHeadroom bands each account bucket; an OFFERABLE bucket (score > 1) is a
// seat with room to give — capacity the night should be spending. Debt is the count of
// offerable buckets: seats available to put more work on. When no runtime roster exists
// (a hermetic/empty home), the signal is nil — reported UNMEASURED, never a false clean,
// so an unread account layer cannot make the night read "fully utilized".
func accountLimitsUtilization(m superloop.Member) superloop.MemberStatus {
	st := superloop.MemberStatus{Member: m}
	hr := rotationHeadroom("")
	if hr == nil {
		st.Measured = false
		st.Detail = "no runtime account roster on this host (fak fleet-accounts has no rows) — cannot read seat utilization"
		return st
	}
	offerable, walled, unknown := 0, 0, 0
	for _, score := range hr {
		switch {
		case score > 1:
			offerable++
		case score < 0:
			walled++
		default:
			unknown++
		}
	}
	st.Measured = true
	// Debt = offerable-but-idle seats: limits with room the night should be spending.
	// Walled buckets are the limit already fully used — the expected good overnight
	// state — so they are NOT debt. Unknown buckets carry one unit each: an unread
	// bucket is not proven busy, so it cannot read as fully utilized.
	st.Debt = offerable + unknown
	st.Detail = fmt.Sprintf("%d offerable seat(s) with room to spend, %d walled (limit already in use), %d unknown across %d account bucket(s)",
		offerable, walled, unknown, len(hr))
	return st
}

// nodeResourcesUtilization folds the live lab fleet snapshot into an up-but-idle box
// debt — the same snapshot `fak lab status` prints. A reachable box in idle/draining
// state is on but not serving: wasted silicon. Debt = idle + draining boxes; a reported
// GPU-util stat that shows the fleet under-busy adds one unit (whole-fleet compute is
// on but not working). An empty/absent reports dir means no box has phoned home yet —
// reported UNMEASURED (the honest-degrade `fak lab status` itself takes), never a false
// clean that would say "every node is busy" when we simply have not heard from any.
func nodeResourcesUtilization(m superloop.Member) superloop.MemberStatus {
	st := superloop.MemberStatus{Member: m}
	ro, err := fleet.LoadRoster(bytes.NewReader(labDefaultRosterJSON))
	if err != nil {
		st.Measured = false
		st.Detail = "embedded lab roster is corrupt: " + err.Error()
		return st
	}
	dir, err := labReportsDir("")
	if err != nil {
		st.Measured = false
		st.Detail = "cannot resolve lab reports dir: " + err.Error()
		return st
	}
	if !labReportsPopulated(dir) {
		st.Measured = false
		st.Detail = fmt.Sprintf("no live box reports under %s yet (run the fleet probe) — cannot read node utilization", dir)
		return st
	}
	reps := fleet.ReadReports(dir, ro)
	snap := fleet.Fold(ro, reps, fleet.FoldOpts{})

	idle := snap.ByState[fleet.StateIdle]
	draining := snap.ByState[fleet.StateDraining]
	live := snap.ByState[fleet.StateLive]
	st.Measured = true
	st.Debt = idle + draining

	gpuNote := ""
	if g := snap.GPUUtil; g != nil && g.Total > 0 {
		// A GPU-util producer that reports the fleet's silicon mostly parked (< half the
		// GPUs busy) adds one unit: whole-fleet compute is on but not working, distinct
		// from box-level idle. Kept coarse (one unit) so a busy fleet with a few parked
		// GPUs is not slandered as underused.
		if g.Busy*2 < g.Total {
			st.Debt++
			gpuNote = fmt.Sprintf(", GPUs %d/%d busy (fleet compute mostly parked)", g.Busy, g.Total)
		} else {
			gpuNote = fmt.Sprintf(", GPUs %d/%d busy", g.Busy, g.Total)
		}
	}
	st.Detail = fmt.Sprintf("%d live, %d idle, %d draining across %d reachable box(es)%s",
		live, idle, draining, snap.Reachable, gpuNote)
	return st
}
