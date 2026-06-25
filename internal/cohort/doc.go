// Package cohort is a fail-closed cohort shrink and agreement leaf over
// comm.Group.
//
// A Cohort is a generation-stamped view of a comm.Group. Shrink removes the
// members the caller reports failed, preserves the survivors' existing rank
// order, and returns a new Cohort with generation+1. Agree is the scalar quorum
// fold for that cohort: present member outputs are reduced with modelroute's
// vote reducer, while absent members remain explicit abstentions against the
// requested quorum floor.
//
// Honesty caveat: this is NOT MPI ULFM. Shrink reforms a Go member-set wrapper;
// it does not detect failures, renumber network ranks, or provide communicator
// progress. Agree is a local, deterministic scalar fold; it does not guarantee
// termination or consensus under arbitrary asynchronous failures. Failure means
// a member identity the caller reported dead or absent, not a detected network
// partition.
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package cohort
