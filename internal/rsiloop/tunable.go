package rsiloop

// tunable.go holds the single demo knob the RSI loop optimizes. It lives alone,
// as one unambiguous literal, because the worktree Proposer (worktree.go) rewrites
// THIS line in an isolated copy to propose a candidate — a stable, regex-friendly
// target is the whole reason the constant is not folded into kpi.go.
//
// The loop never edits this file on `main`; it edits the COPY inside a detached
// git worktree, measures the result, and keeps or reverts. `main`'s value is the
// baseline every cycle re-derives from (the "benchmark against latest main" rule).

// DefaultCacheSize is the baseline LRU capacity the kpiprobe reports a hit-rate
// for. It is deliberately below the ReferenceTrace working set so the loop has a
// real, measurable gain to discover (HitRate is monotonically non-decreasing in
// the cache size — see kpi.go). The Proposer's rewrite target is the integer
// literal on the next line; keep the form `DefaultCacheSize = <int>` exact.
// fak:opttarget name=lru-cache-size metric=lru_hit_rate dir=higher sweep=4,5,6,8 measurer=worktree-int
const DefaultCacheSize = 4

// TunableConstName is the identifier the worktree Proposer rewrites. Exported so a
// caller (and the test that guards the rewrite contract) names it once, not as a
// scattered string literal.
const TunableConstName = "DefaultCacheSize"
