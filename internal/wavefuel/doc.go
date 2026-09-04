// Package wavefuel holds the executable contract for fleet-wave operator receipts.
//
// Invariant: wave fuel accounting is fail-closed and bounded across all fleet wave allocations.
// Guard: operator receipts must never route through stale launcher contracts or unvalidated caps.
// Assumption: deadlines are strictly monotonic and budget overruns immediately reject further turns.
package wavefuel
