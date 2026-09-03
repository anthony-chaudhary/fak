package superstream

import (
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// GenerateLeaseID builds a deterministic lease identity for an item within a stream.
func GenerateLeaseID(streamID, itemID, holder string) string {
	cleanStream := strings.ReplaceAll(streamID, ":", "-")
	cleanItem := strings.ReplaceAll(itemID, ":", "-")
	cleanHolder := strings.ReplaceAll(holder, ":", "-")
	return fmt.Sprintf("stream:%s:%s:%s", cleanStream, cleanItem, cleanHolder)
}

// BuildAdmissionRequest constructs a laneadmit.Request for a specific work item.
func BuildAdmissionRequest(streamID string, item WorkItem, holder string) laneadmit.Request {
	leaseID := GenerateLeaseID(streamID, item.ID, holder)
	return laneadmit.Request{
		Surface: "superstream",
		Lane:    item.Lane,
		Tree:    item.Tree,
		Holder:  holder,
		LeaseID: leaseID,
	}
}

// EvaluateLeaseAdmission evaluates whether a work item can acquire its required lane lease
// given the current live lease topology and taxonomy.
func EvaluateLeaseAdmission(streamID string, item WorkItem, holder string, liveLeases []laneadmit.Lease, tax laneadmit.Taxonomy) (laneadmit.Verdict, laneadmit.Request) {
	req := BuildAdmissionRequest(streamID, item, holder)
	verdict := laneadmit.Decide(req, liveLeases, tax)
	return verdict, req
}

// FindNextDisjointItem searches forward from startIndex for the next pending WorkItem
// whose required lane and tree are completely disjoint from live leases.
// This allows a stream to progress non-colliding items rather than stalling completely
// when the head item is temporarily contended by a peer session.
func FindNextDisjointItem(streamID string, queue []WorkItem, startIndex int, holder string, liveLeases []laneadmit.Lease, tax laneadmit.Taxonomy) (int, bool) {
	for i := startIndex; i < len(queue); i++ {
		if queue[i].Status != ItemPending {
			continue
		}
		verdict, _ := EvaluateLeaseAdmission(streamID, queue[i], holder, liveLeases, tax)
		if verdict.Admit {
			return i, true
		}
	}
	return -1, false
}

// CanCoexistWithHeld checks if nextItem can safely run alongside an already held lease.
// It synthesizes a live lease from held and tests nextItem against it.
func CanCoexistWithHeld(streamID string, held *HeldLease, nextItem WorkItem, holder string, tax laneadmit.Taxonomy) bool {
	if held == nil {
		return true
	}
	synthLease := laneadmit.Lease{
		ID:     held.LeaseID,
		Lane:   held.Lane,
		Tree:   held.Tree,
		Holder: held.Holder,
	}
	req := BuildAdmissionRequest(streamID, nextItem, holder)
	verdict := laneadmit.Decide(req, []laneadmit.Lease{synthLease}, tax)
	return verdict.Admit
}

// MakeHeldLease constructs a HeldLease record upon successful admission.
func MakeHeldLease(streamID string, item WorkItem, holder string, now time.Time, fencingToken string) HeldLease {
	return HeldLease{
		LeaseID:      GenerateLeaseID(streamID, item.ID, holder),
		Lane:         item.Lane,
		Tree:         item.Tree,
		Holder:       holder,
		AcquiredAt:   now,
		FencingToken: fencingToken,
	}
}
