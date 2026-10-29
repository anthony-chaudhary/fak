// Package wavefuel holds the executable contract for fleet-wave operator receipts.
//
// Accounting is fail-closed and bounded across all fleet wave allocations: operator
// receipts never route through stale launcher contracts or unvalidated caps, deadlines
// are strictly monotonic, and a budget overrun immediately rejects further turns.
package wavefuel
