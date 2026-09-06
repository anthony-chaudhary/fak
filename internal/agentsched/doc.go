// Package agentsched provides prioritized task scheduling and admission governance
// for concurrent agent execution.
//
// Tasks are scheduled across four priority tiers (P0 system through P3 speculative)
// and admitted through four sequential gates: worker concurrency, host resource
// envelope (CPU, memory, file handles, thermals), provider headroom, and lane
// clearance against active repository leases. Under host stress, the governor
// dynamically paces turns, downscales worker concurrency, and sheds speculative load.
package agentsched
