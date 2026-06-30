// Package issuecatalog is the durable bridge from a reviewable catalog of
// performance-enablement gap rows to a stable, deduplicated GitHub issue per row.
//
// # Why it exists
//
// fak's performance value is largely built but under-shipped: features that are
// default-OFF, blocked by one small wiring step, or never advertised to a
// lowest-common-denominator user. CLAIMS.md records each as an honest fence ("the
// named follow-on", "default OFF", "live-wiring remains"). This package turns a
// curated catalog of those gaps — each row already scoped to the issue contract —
// into tracked, dispatchable issues, and keeps them in sync as the catalog grows.
// It is the same shape as dogfoodissues/learningdebt: fold a source into stable
// keys, dedup by an issue-body marker, gate on the issuecontract, dry-run by
// default, and shell out to `gh` only on --live.
//
// # The pure surface
//
//   - LoadCatalog/ParseCatalog read a JSON array of Row (the reviewable data file,
//     embedded by default, --catalog FILE to override).
//   - Each Row carries a STABLE key. IssueBody renders it into an HTML-comment
//     marker (<!-- fak-issue-catalog-key: KEY -->) and the issuecontract heading
//     grammar, so a later run finds and updates the issue in place and a
//     ReviewIssueDraft re-audit reads it back as dispatchable.
//   - BuildPlan reviews every row against issuecontract, drops the rows that do not
//     name a spine/scope/witness/route as skipped triage rows, then diffs the
//     dispatchable rows against existing issues (matched by marker key) to decide
//     create vs update.
//
// # The effectful surface
//
// Talking to GitHub is gated behind an explicit --live opt-in; the default is a
// dry-run that prints exactly what it WOULD do and never touches the network. On
// --live the command shells out to `gh` through an injectable Runner. Applying
// milestone + labels needs a token with triage scope; a create-only token silently
// drops them, but the issue body carries the same metadata regardless.
//
// Tier: composer (3) — reads a catalog, derives a plan, and (only when asked)
// composes the external `gh` tool off the hot path; imports issuecontract(1).
package issuecatalog
