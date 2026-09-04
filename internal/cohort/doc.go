// Package cohort provides cohort shrink, quorum agreement, and drift monitoring
// over comm.Group member sets.
//
// A Cohort is a generation-stamped view of a comm.Group. Shrink removes the
// members reported failed, preserves the survivors' existing rank order, and
// returns a new Cohort with generation+1. Agree performs scalar quorum folding:
// present member outputs are reduced with modelroute's vote reducer, while
// absent members remain explicit abstentions against the requested quorum floor.
//
// Honesty caveat: this is not MPI ULFM. Shrink reforms a Go member-set wrapper;
// it does not detect failures, renumber network ranks, or provide communicator
// progress. Agree is a local, deterministic scalar fold; it does not guarantee
// termination or consensus under arbitrary asynchronous failures. Failure means
// a member identity reported dead or absent, not a detected network partition.
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
package cohort
