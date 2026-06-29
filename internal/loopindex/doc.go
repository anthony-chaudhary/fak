// Package loopindex scores the agentic-coding LOOP — the round an agent (and a
// fleet of agents) runs to go from a task to shipped, verified code — into one
// witnessed number: the loop-index. It is the SPINE of the dev-experience epic
// (#1148, child #1152): every other lever in that epic reports its before/after
// delta against this index.
//
// Tier: foundation (1) — see internal/architest. It is a pure analysis primitive:
// it imports only the standard library, has no I/O, and never touches the request
// path. The impure shell (the `fak loop-index-scorecard` command) discovers each
// stage's witness PROBES from the tracked tree and calls Score; this package owns
// only the deterministic fold.
//
// THE PROBLEM IT MEASURES. "The loop got faster / more self-correcting" is a vibe.
// You cannot 10x what you cannot measure round-over-round. The repo already emits
// per-stage signals (session-observability cost+outcome capture in
// internal/sessionobs, the dos witness verbs, tool-floor repair) but there is no
// single index spanning orient→ship→learn. This scorecard folds the per-stage
// witness state into one grade + a debt count, cross-checked against the tracked
// tree, so a regression reds the gate and an improvement is provable.
//
// THE SIX STAGES. The agentic-coding loop is Orient → Plan → Act → Verify → Ship →
// Learn. Each stage is one rung with a witnessed sub-signal:
//
//	orient  — recall-staleness / context-thrash   (recalled memory re-checked before trust)
//	plan    — collision-priced fan-out coverage    (dos arbitrate as the default pre-dispatch gate)
//	act     — malformed-call repair rate            (a bad tool call repaired-or-refused, not a lost turn)
//	verify  — unwitnessed-done rate                 (a false "done" refused at the STOP seam, not in review)
//	ship    — green-gate latency budget             (the verify loop is fast & incremental, a tracked budget)
//	learn   — session→outcome link coverage         (every session's cost AND outcome captured + consumed)
//
// THE WITNESS-PROBE MODEL. The spine grades, per stage, whether a witnessed signal
// is LOAD-BEARING in the tree, deterministically — never a live runtime metric a
// clean clone could not reproduce. Each stage carries a set of Probes (named checks
// the impure shell ran against the tracked tree). A stage is WIRED iff its keystone
// probes pass (a stage with no keystone witness is still a vibe); its HEALTH is the
// fraction of all its probes that pass. A stage is in DEBT when it is unwired, or
// wired but below its floor. The headline integer is loopindex_debt: the count of
// stages not yet witnessed at their floor.
//
// This is the same discipline as every other fak scorecard (see
// .claude/skills/scorecard): deterministic (no clock, no RNG — two callers with the
// same inputs score identically), and you retire debt by BUILDING the real witness
// for a stage (wire recall re-verify, make collision-priced fan-out the default,
// refuse a false done at the seam), never by weakening a probe.
package loopindex
