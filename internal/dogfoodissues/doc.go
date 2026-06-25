// Package dogfoodissues is the backlog bridge from the recent-feature dogfood
// scorecard to a stable, deduplicated GitHub issue per ACTION item.
//
// # Why it exists
//
// The recent-feature dogfood gate intentionally ACCEPTS scorecard ACTION/debt
// when the machine payload is valid: that keeps local/CI dogfood honest without
// letting pre-existing debt block every new feature. The cost of that leniency is
// that scorecard ACTION items can quietly accumulate with nothing tracking them.
// This package is the separate, opt-in bridge that closes the loop: read a dogfood
// report.json, find the scorecard ACTION items, and create OR update exactly one
// stable GitHub issue per item.
//
// # The pure surface
//
// The trustworthy core is entirely pure and hermetically testable:
//
//   - ExtractActionItems parses a report.json fold (a JSON object with a "probes"
//     array) into a []ActionItem. It recognizes two scorecard probes today — the
//     code-slop scorecard and the dogfood-coverage scorecard — and only emits an
//     item when that probe is actually in an ACTION/debt state.
//   - Each item carries a STABLE key (recent-feature-dogfood/<probe>/<finding>).
//     IssueBody renders that key into an HTML-comment marker
//     (<!-- fak-dogfood-action-key:KEY -->) so a later run can find the issue it
//     already opened. MarkerKey reads that marker back out of an existing body.
//   - BuildPlan diffs the freshly extracted items against the existing issues
//     (matched by marker key) and decides create vs update per item.
//
// # The effectful surface
//
// Talking to GitHub is gated behind an explicit --live opt-in on the CLI. The
// default is a dry-run that prints exactly what it WOULD do and never touches the
// network. Only on --live does the command shell out to the `gh` CLI (issue list /
// create / edit). Sync runs the gh subprocess for each planned row through an
// injectable runner so the wiring is testable without gh present.
//
// Tier: composer (3) — a tool-shaped fold that reads a report, derives a plan, and
// (only when asked) composes the external `gh` tool; shells out off the hot path,
// imports nothing internal.
package dogfoodissues
