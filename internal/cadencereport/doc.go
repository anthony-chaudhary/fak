// Package cadencereport is the consolidated regular-cadence report -- one fold
// over the four cadence dimensions an operator tracks: scores, maturity,
// work-done, and releases.
//
// The repo already reports each dimension, but in three separate places that
// only ever land in an ephemeral CI step summary, and never as a durable trend:
//
//   - SCORES    -- tools/scorecard_control_pane.py folds ~20 scorecards into one
//     portfolio debt against a single pinned baseline (garden.yml, daily).
//   - MATURITY  -- fak maturity folds the feature lifecycle ladder: debt,
//     index, rung distribution, and the ranked next-work backlog.
//   - WORK-DONE -- has no durable cadence report at all; derived here from git
//     (commits + `(fak ` ship-trailer count over a trailing window).
//   - RELEASES  -- tools/release_status.py folds release readiness
//     (release-cadence.yml, every 6h).
//
// Nothing folds those four into ONE "where do we stand this week?" envelope, and
// nothing keeps a durable, append-only HISTORY so the trend is visible across
// weeks rather than against a single point. This package is that fold (the Go
// sibling of internal/gardenbundle): it distills each dimension into a uniform
// row, folds one schema/ok/verdict/finding/reason/next_action control-pane
// envelope, and appends a dated row to a committed JSONL ledger
// (docs/cadence/history.jsonl) so the cadence is durable and trended.
// The ledger also projects each row into an unbounded standing score: bounded
// 0..100 local scores are first normalized against their current difficulty,
// then only the normalized health delta moves the accumulated standing.
//
// The verdict is a REPORT contract, not a second quality gate: the scorecard
// ratchet (ci.yml) already gates debt regressions, so --check here fails ONLY
// when a dimension could not be MEASURED (the report is incomplete) and is OK
// otherwise, surfacing a regressed score or a pending release as an advisory
// line. This mirrors fresh_status's advisory contract.
//
// The pure, tested surface is InterpretScores / InterpretReleases (distill one
// sub-tool payload), Fold/FoldWithMaturity (fold the dimensions into one envelope),
// ParseLedger / ProjectStanding / TrendVsLast (the durable history + per-tick trend), and
// CheckGate (the advisory CI gate). The live runners (collect.go) shell to
// python3 tools/<x>.py and to git exactly as the existing folds do.
package cadencereport
