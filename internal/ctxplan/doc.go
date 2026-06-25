// Package ctxplan is the context PLANNER: it treats the current turn's context as an
// O(1) materialized VIEW over the full, lossless history store, and re-plans that view
// each turn instead of letting the linear transcript grow without bound (or compacting
// it lossily). It is the unbuilt "context-layout compiler" rung of the on-demand-context
// build path (docs/notes/ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md, Step 4): the layer
// that, given a budget and a PREDICTION of what the next turns will reference, OPTIMIZES
// which spans should be resident — rather than executing an authored pipeline (memq) or
// demand-paging one query's working set (cdb/contextq).
//
// # The middle ground between two extremes
//
// A long agent session has two degenerate options today:
//
//	linear      keep the whole transcript resident      -> O(N) tokens, EXACT recall, blows the window
//	compaction  summarize at a cap, drop the originals  -> O(1) tokens, LOSSY recall, irreversible
//
// ctxplan is the third option — the one that keeps BOTH properties the extremes each
// give up:
//
//	planned view  an O(1) resident view + the lossless store behind it, re-planned each turn
//	              -> O(1) tokens, EXACT recall (page any elided span back in), paying only a
//	                 bounded, cheap forecast-MISS rate (a page fault, never a lost fact)
//
// The unbounded transcript is the "core dump" (the term the repo already uses for a
// finished session: recall's manifest+CAS, cdb's debugger view). ctxplan makes the LIVE
// turn the same shape: the resident context is one rendering of a queryable image, so a
// session that would have been 50, 100, 1k, 10k, or 1M linear turns becomes 1 current
// turn + a flexible history the planner re-derives on demand. scaling.go quantifies how
// the resident-token curve bends across that horizon.
//
// The flexible version of that view is a Layout: four independently tunable areas over
// the same lossless store. Base is structural prompt material supplied as pins (system,
// developer, active task); current is the newest entry/entries; recent is the last N
// entries before current; deep is old history reached by relevance or durability. Each
// area declares its own N, token-size cap, and precision (exact, planned, or pointer), and
// the final resident bytes still flow through the same global Budget and trust-gated
// Materialize path. So a user/system can widen recent history, shrink deep history to
// pointers, or force a larger current turn without returning to an unbounded transcript.
//
// # The Postgres-planner correspondence (the lens the design leans on)
//
//	relational analogue          ctxplan
//	-------------------          -------
//	table / relation             the history store (recall manifest / memq cells)
//	query                        the Forecast (predicted reference for the next horizon) + Budget
//	pg_statistic (row stats)     per-cell benefit signals: relevance, learned utility, durability, recency
//	cost constants               Forecast.Weights (how the signals combine)
//	the planner / optimizer      Optimize (a budgeted 0/1 knapsack: maximize benefit s.t. tokens <= W)
//	the chosen plan / access path  Plan.Selected (which spans are resident, in render order)
//	EXPLAIN / EXPLAIN ANALYZE    Plan.Explain (estimated cost+benefit per included/elided span)
//	a materialized view          the rendered O(1) fresh history (Materialize)
//	the buffer pool / working set  the resident view; the backing store is the CAS swap device
//	a page fault                 a forecast MISS -> demand-page the missing span back in (exact, cheap)
//	a prepared statement / plan cache  reusing a Forecast plan across turns (cachemeta.PlanTemplate)
//
// # What ctxplan is NOT
//
// It is a pure planner over a small, self-contained vocabulary: a Span (SAFE metadata
// only — never the bytes of a sealed span) and a Store (Spans + a trust-gated
// Materialize). It computes WHICH spans should be resident and renders the selection
// through the store's gated page-in. It imports NOTHING internal (stdlib only), so it is
// a foundation leaf that builds standalone; an adapter that lowers a memq.Cell, a
// recall.Page, or a cdb.Frame into a Span is a thin, higher-tier follow-on. It registers
// nothing with the frozen ABI, runs no model, and moves no KV bytes. The trust boundary
// stays where it already is: a sealed or tombstoned span is never a candidate (it can
// never be pinned into the resident view — a pin cannot launder poison), exactly as
// recall/cdb/memq enforce.
//
// # Faithful-by-reference, not lossy-by-summary (the honesty gate)
//
// The one property that distinguishes a planned VIEW from COMPACTION is faithfulness:
// every span the planner elides from the resident view stays recoverable — it is moved
// to cold storage with a page-back-in handle (its content address), never destroyed.
// faithful.go turns that into a checkable Witness: a Plan is Faithful iff its
// Selected+Elided sets partition every candidate AND every elided span carries a
// recovery handle. A compaction-style plan that dropped a span with no handle is
// reported UNFAITHFUL. This is the gate that keeps "plan a view" honest against the
// temptation to silently summarize.
//
// # The Span.Bytes cost contract (why the O(1) bound is honest, not advisory)
//
// The planner prices a span's resident cost from Span.Bytes (TokenCost = ceil(Bytes/4));
// the renderer realizes ceil(len(body)/4) from the bytes the store actually pages in. Those
// two token sources agree ONLY while every Store keeps Span.Bytes == len(Materialize(id)) for
// a benign (non-sealed) span — the cost contract a Store implementation MUST hold. A real
// adapter whose Span.Bytes understates the body (a paged-out pointer size, a tokenizer
// estimate) would let the renderer blow past the budget the planner charged. Materialize's
// witness enforces the contract per rendered span: Witness.CostContract falls false and
// CostDiverged names the offender, so a store that hands back more bytes than it advertised
// cannot make the budget accounting fictional SILENTLY. A Store that cannot keep the
// contract makes the O(1) resident bound ADVISORY — the realized resident tokens may exceed
// what the planner charged, and a caller seeing CostContract=false must treat the budget as
// no longer guaranteed (re-plan or shed load), not as a soft warning.
//
// Tier: foundation (1) — see internal/architest. The core planner is a pure algorithm
// over its own Span/Store types and imports nothing internal (stdlib only), exactly like
// the other foundation leaves (answershape, codelint, polymodel). It is off the request
// path and not in the defconfig; the memq/cdb/recall adapters that feed it real stores
// are higher-tier follow-ons that import both this leaf and that mechanism.
package ctxplan
