// Package tooltrend folds a sequence of per-session tool-call buckets into a
// tool-mix + output-shape TREND — "which tools and which response shapes are
// rising or falling across N sessions" (issue #2826, session-analytics C4).
//
// # The fold
//
// The input is an ordered slice of [Bucket] values, each one a labeled group of
// tool calls (typically one agent session / trajectory). [Fold] turns every
// bucket into a [Point] — the bucket's tool MIX (each tool's share of the
// bucket's calls, in [0,1]), its output-SHAPE mix (each response size-class's
// share of calls), and its overall error rate — and then reports the biggest
// MOVERS: the tools and size-classes whose share changed most between the first
// bucket and the last. A [Move] carries the from/to shares, the signed delta,
// and a closed direction ("up" | "down"). The result is deterministic: points
// preserve input order, movers sort by absolute change descending then key
// ascending, so the same buckets always render byte-identically.
//
// The trend is defined as the NET change from the first bucket to the last — the
// earliest session compared to the most recent. It answers "since the start of
// this window, what is the agent reaching for more, and what less", not a
// per-step slope. A window of fewer than two buckets has no first-to-last delta,
// so it yields points with no movers.
//
// # Reuse
//
// Per-bucket tool-mix shares and per-tool error counts are computed by reusing
// internal/toolrollup's [toolrollup.Rollup] fold rather than re-deriving them, so
// a tool's share here is exactly the share the per-tool rollup report shows. The
// input record type is [toolrollup.ToolCall], the same corpus row the rollup and
// the trajectory export use, so one bridge from a trajectory feeds both folds.
//
// # Tier
//
// Tier: composite (2) — see internal/architest. Unlike its foundation siblings
// internal/toolrollup and internal/toolseq (which import nothing internal),
// tooltrend sits one tier up: it imports internal/toolrollup (foundation, 1) to
// reuse the per-tool fold, and nothing else internal, so it never imports
// "upward" and stays off the hot path. A report — or the `fak traj report` front
// door (#2827) — can fold a corpus into a trend without pulling in trajectory
// machinery.
package tooltrend
