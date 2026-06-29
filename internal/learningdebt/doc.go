// Package learningdebt is the staleness->backlog bridge from the learning-docs
// scorecard to a cap-bounded, deduplicated GitHub triage issue per HARD teaching
// defect. It is the docs-freshness arm of the fleet loop (epic #1278, issue #1283)
// and the learning-scorecard peer of internal/dogfoodissues.
//
// tools/learning_scorecard.py measures whether the teaching docs actually teach and
// counts the corpus's learning-debt: the HARD defects a reader trips on (an orphan
// lesson unreachable from any front door, a dead link, a stale install pin, an
// uncovered course/topic, a tutorial with no runnable command). The scorecard
// reports that debt but nothing FEEDS it back to the dispatch fleet. This package
// closes the loop: read the scorecard's --json payload, lift each HARD defect, and
// file ONE stable triage issue per defect — without ever storming the tracker.
//
// The pure surface is hermetically testable: ExtractDefects folds a payload into a
// []Defect (each carrying the EXACT doc, a deterministic defect class, the
// scorecard's verbatim finding, and a stable Key). BuildPlan dedups against the
// seen-cache keys AND the existing-issue marker keys, then CAPS the survivors at an
// integer budget. IssueBody stamps the Key into an HTML-comment marker; MarkerKey
// reads it back so a later run never opens a duplicate.
//
// Safe by default: planning mutates nothing — not the tracker, not the seen-cache.
// Only the CLI's --live opt-in shells out to `gh` (FetchExistingIssues / Sync) and
// records filed keys to the gitignored .fak/learning-debt-dispatch/seen.json. The gh
// boundary runs through an injectable Runner so the wiring is testable without gh.
//
// Tier: composer (3) — a tool-shaped fold that reads a scorecard payload, derives a
// plan, and (only when asked) composes the external `gh` tool; shells out off the hot
// path, imports nothing internal. The Go house pattern the de-Python ratchet
// (internal/pythongate) requires instead of a new tools/*.py.
package learningdebt
