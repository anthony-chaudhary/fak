package model

import (
	"fmt"
	"strings"
)

// ep_fanout_topology.go — the expert-parallel (EP) fanout coverage preflight (#4264):
// the EP analogue of TPPlan.Validate, borrowed from zml's Sharding.covers (the
// "is this split well-formed over this topology?" check that runs BEFORE executors
// are committed). A sharded EP serve of world size N runs N processes; rank 0 is the
// front rank and ranks 1..N-1 are followers. The front rank mirrors each inbound chat
// request to a set of follower endpoints (FAK_EP_FANOUT_ADDRS, one URL per follower
// rank) so every process reaches the routed-expert AllReduce concurrently. If that
// address set does not bijectively cover the N-1 follower ranks — wrong count, a rank
// addressed twice, an empty binding — the collective can only make partial progress,
// and today that is discovered ONLY after followers are already spun up.
//
// ValidateEPFanoutCoverage is the cheap, total, up-front refusal: given the world size
// and the resolved follower address set, it returns a typed *EPFanoutTopologyError the
// front rank can log and refuse on (no followers launched) instead of committing a
// miscovering fanout. It is pure (no I/O, no rank state) so it is trivially table-tested,
// exactly like TPPlan.Validate — the front-rank wiring (gateway lane) calls it before
// startEPFanoutFollowers commits any follower request.
//
// The mapping from zml's coverage axes onto fak's flat address→rank fanout:
//
//   - zml "undeclared axis / no binding"           → EPCoverEmptyAddress (a follower
//     rank whose endpoint is blank — a rank with no real binding).
//   - zml "axis used twice"                         → EPCoverDuplicateRank (the same
//     endpoint listed twice — two ranks cannot share one follower process).
//   - zml "size that doesn't divide / inconsistent  → EPCoverMiscount (the follower
//     cardinality"                                     count is not exactly N-1; in a
//     1-address-per-rank bijection "short by one", "doesn't divide the world", and
//     "front rank included in its own fanout set" all collapse to this cardinality miss).
//
// plus a degenerate-world guard (EPCoverDegenerateWorld): a fanout set only means
// something inside a real world of >1 ranks, mirroring resolveEPRankConfig's own
// "a rank index needs a world of >1" fail-closed stance.

// EPFanoutCoverReason is the closed, typed vocabulary of ways a fanout address set can
// fail to cover the follower ranks. A caller switches on it to emit a specific refusal.
type EPFanoutCoverReason string

const (
	// EPCoverDegenerateWorld: worldSize < 2, so there are no follower ranks to cover —
	// a fanout set is meaningless without a real world.
	EPCoverDegenerateWorld EPFanoutCoverReason = "degenerate-world"
	// EPCoverEmptyAddress: a follower entry is blank, i.e. a rank with no real endpoint
	// (zml's "undeclared axis / no binding").
	EPCoverEmptyAddress EPFanoutCoverReason = "empty-address"
	// EPCoverDuplicateRank: the same endpoint appears twice, i.e. one follower process
	// claimed by two ranks (zml's "axis used twice") — not a bijection.
	EPCoverDuplicateRank EPFanoutCoverReason = "duplicate-rank"
	// EPCoverMiscount: the follower count is not exactly worldSize-1 (zml's "size that
	// doesn't divide / inconsistent cardinality").
	EPCoverMiscount EPFanoutCoverReason = "miscount"
)

// EPFanoutTopologyError is the typed refusal ValidateEPFanoutCoverage returns when a
// fanout address set does not bijectively cover the follower ranks. It carries the
// machine-checkable Reason plus the numbers a front-rank log line needs, so the caller
// refuses with a specific cause instead of a partial launch.
type EPFanoutTopologyError struct {
	Reason    EPFanoutCoverReason
	WorldSize int    // the EP world size N (--expert-parallel N)
	Want      int    // follower ranks that must be covered (N-1)
	Got       int    // follower addresses supplied
	Detail    string // human context (the offending address / positions)
}

func (e *EPFanoutTopologyError) Error() string {
	msg := fmt.Sprintf("model: EP fanout set does not cover the follower ranks (%s): world=%d want %d follower(s), got %d",
		e.Reason, e.WorldSize, e.Want, e.Got)
	if e.Detail != "" {
		msg += " — " + e.Detail
	}
	return msg
}

// ValidateEPFanoutCoverage is the front-rank preflight: it returns nil when
// followerAddrs bijectively covers the N-1 follower ranks of a world of size worldSize,
// and a typed *EPFanoutTopologyError otherwise. It fails closed on the FIRST defect it
// finds, in the order degenerate-world → empty binding → duplicate → cardinality, so the
// most specific, most actionable reason wins (a duplicate is reported as a duplicate even
// when the count also happens to be wrong). It is total and side-effect free — the same
// inputs always yield the same verdict, and it launches nothing.
func ValidateEPFanoutCoverage(worldSize int, followerAddrs []string) error {
	want := worldSize - 1
	got := len(followerAddrs)
	if worldSize < 2 {
		return &EPFanoutTopologyError{
			Reason: EPCoverDegenerateWorld, WorldSize: worldSize, Want: want, Got: got,
			Detail: fmt.Sprintf("world size %d has no follower ranks to fan out to", worldSize),
		}
	}
	// Every follower rank needs a real endpoint (zml: every partitioned axis has a
	// binding). A blank entry is an unbound rank — refuse before the count check so an
	// empty binding is never masked by an otherwise-correct cardinality.
	seen := make(map[string]int, got)
	for i, addr := range followerAddrs {
		trimmed := strings.TrimSpace(addr)
		if trimmed == "" {
			return &EPFanoutTopologyError{
				Reason: EPCoverEmptyAddress, WorldSize: worldSize, Want: want, Got: got,
				Detail: fmt.Sprintf("follower %d has a blank endpoint (rank with no binding)", i),
			}
		}
		if j, dup := seen[trimmed]; dup {
			return &EPFanoutTopologyError{
				Reason: EPCoverDuplicateRank, WorldSize: worldSize, Want: want, Got: got,
				Detail: fmt.Sprintf("endpoint %q addressed twice (followers %d and %d)", trimmed, j, i),
			}
		}
		seen[trimmed] = i
	}
	if got != want {
		return &EPFanoutTopologyError{
			Reason: EPCoverMiscount, WorldSize: worldSize, Want: want, Got: got,
			Detail: fmt.Sprintf("a world of %d needs exactly %d distinct follower endpoint(s)", worldSize, want),
		}
	}
	return nil
}
