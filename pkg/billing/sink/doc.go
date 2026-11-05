// Package sink provides the out-of-process record-sink adapter for
// billing.UsageRecord values: a bounded, asynchronous delivery path
// from a serving process to a durable destination outside it, so a
// consuming authority written in any language attaches by reading what
// the sink writes, not by linking against this one.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/billing/sink)
// on purpose. Placement follows two design sources, cited by reference
// only:
//
//   - BOUNDARY.md in the companion private repository keeps the money
//     side of serving private; the neutral record and its transport
//     mechanism are public.
//   - Plan C ("off-the-shelf public spine"), billing-modularization
//     lane, Phase 2 item 9 (companion private repository,
//     docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
//     the optional out-of-process record sink - "bounded ring +
//     flusher" over the same record-sink port - so consuming
//     authorities attach without a compile dependency.
//
// Asynchronous by construction: the hot path never touches the durable
// destination. Write enqueues into a bounded ring and returns; a single
// background flusher goroutine performs every backend call, in batches,
// in order. Nothing durable blocks a waiting client.
//
// Overflow is explicit and typed. RingOptions names the capacity, the
// batch size, the flush interval, and an OverflowPolicy (DropOldest or
// RejectNew); NewRing validates all of them instead of substituting a
// hidden default, and either policy counts every record it discards or
// refuses through a typed counter (Dropped, Rejected), so overflow is
// never silent.
//
// Two backends ship: Memory, an in-memory sink for tests and reference
// use, and File, a JSONL file sink that is append-safe across restarts
// (one record per line, deterministic field order, strict decode on
// read-back via ReadJSONL). A socket or broker backend is the documented
// follow-on seam: implement Backend and pass it to NewRing.
//
// This package is pure MECHANISM. It contains NO pricing and NO policy,
// and no default may encode a commercial choice: records move from
// producer to destination unchanged; what a record means, and what it
// costs, belong to the consuming authority.
package sink
