// Package dispatchstatus ports the lease-classification core of tools/dispatch_status.py
// (#1406): given the refs/fak/locks lease records and the current dispatch backlog, it
// classifies each lease LIVE vs EXPIRED, computes its age / TTL / expiry, extracts its
// lane, and — for live leases — whether its file tree overlaps a currently-routed
// issue's lane tree (so it BLOCKS a candidate) or is merely residue.
//
// This is the pure, deterministic summarize core (SummarizeLeases + the tree-overlap
// and lease-time helpers); the git-shelling reader that materializes the records from
// refs/fak/locks is a thin I/O wrapper left as a follow-on. Keeping the classifier pure
// makes the dispatch-observability fold hermetic-testable with no live git state.
package dispatchstatus
