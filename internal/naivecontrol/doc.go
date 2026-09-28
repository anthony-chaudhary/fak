// Package naivecontrol records the naive dispatch arm — the 45-character
// `fak issue-orchestrator --top 10 --max-waves 1` — as a measured control group, on
// the same ruler as the orchestrated dispatch loop, so the two can be read side by
// side from one ledger.
//
// Invariant: a ship count is admissible only when git ancestry produced it. A
// worker's or pipeline's claim of a commit is an input to verify, never a count.
//
// Invariant: a metric that could not be read is MISSING_MEASUREMENT and renders as
// UNKNOWN. It never defaults to 0, and an aggregate over runs where any run is
// unmeasured is itself unmeasured (the partial sum is disclosed, not promoted).
//
// Tier: mechanism (2) - see internal/architest. The package imports assumecheck(2)
// for the reachable-and-not-reverted ancestry witness and dispatchtick(2) for the
// subject-cites-#N binding key the orchestrated witness sweep uses; an upward
// import fails the architest gate.
//
// # Why this exists
//
// Every claim in docs/ops-unsticking.md that the naive path out-ships the
// elaborate one is inference until the naive path has a number. The job repo
// relabelled its naive fanout era "human-driven ... not a fair apples-to-apples
// baseline", which excluded the simple path from the comparison while the loop
// was scored against a target. A path with no number cannot be credited when it
// wins, and cannot be defended when it loses.
//
// The same repo recorded "a lying dispatch driver whose SHIPPED/BLOCKED tokens were
// overridden by git-ancestry 5/5 times", which is why nothing here trusts a
// self-reported verdict: every claimed SHA is re-derived from git, and a pick's
// "not shipped" is only asserted after this package's own git scan found nothing.
//
// fak's own DOA incident (internal/dispatchdoa) is why nothing here defaults to 0:
// 350 of 382 spawned workers died and every one wrote `reason: unknown`, which was
// already the largest no-commit bucket, so a total outage read as background noise.
// Here an unread field says so, with its reason, on every surface.
//
// # Shape
//
// The fold (Build, Summarize, Compare, RenderComparison) is pure: no clock, no
// process, no filesystem. Git access is the injected GitRunner, the repo-wide
// exit-code runner shape, so tests script evidence without a repository. The
// ledger reader and appender are the only file I/O, and they stay on the
// caller-supplied path.
package naivecontrol
