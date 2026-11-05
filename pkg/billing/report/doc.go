// Package report is the reporting plumbing of the public billing spine:
// per-tenant usage rollups over UsageRecord v1 and a daily JSONL store
// that is append-safe and replayable.
//
// It is pure mechanism over the record schema defined in pkg/billing: it
// groups, sums, persists, and replays records. It prices nothing and
// decides nothing; no amount enters a rollup unless a consuming authority
// put one in the data.
//
// Unknown-preserving arithmetic is the load-bearing rule: a token axis
// that any contributing record left unreported (nil) comes back
// unreported (Known=false) in the rollup. Summing over the reported values
// and presenting the result as a total would convert "unknown" into
// "known zero", which the record's closed classification exists to
// prevent.
//
// Design source: Plan C Phase 1 item 7 (companion private repository,
// docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
// "Reporting plumbing: per-tenant usage rollup, daily JSONL."
package report
