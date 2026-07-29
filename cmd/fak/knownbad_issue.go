package main

// Layer E of the result-side false-positive pipeline (#2712 fleet spine): render ONE
// deduped GitHub issue per fleet-correlated known-bad signature. `fak knownbad correlate`
// folds cross-trace fleet observations into candidates (Layer D); when it is asked to file
// issues, each candidate's ledger Record is turned into a create/update plan here and
// materialized through the shared internal/dogfoodissues gh plumbing — the same
// marker-dedup path the dogfood backlog and guardcomplaint appeals already use, so a
// recurring shared failure UPDATES its one tracking issue (bumping an occurrence count)
// instead of opening a duplicate every nightrun.
//
// The signature is already a stable content hash of (reason class, tree globs, failure
// hash), so it is the natural dedup key: one signature ⇔ one issue. These builders are
// pure (Record + Candidate + existing issues in, PlanRow out) so buildKnownBadIssuePlan is
// unit-tested without gh; all impurity (fetch/sync) stays in the correlate verb.

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// knownBadIssueLabel filters correlate-filed issues apart from the dogfood backlog and the
// guardcomplaint appeal channel, so a maintainer can triage shared blockers as their own
// queue.
const knownBadIssueLabel = "known-bad"

// The re-file count is read back out of an existing issue body so an update can increment
// it — a signature that keeps re-correlating across nightruns is a stronger, longer-lived
// blocker than a one-off, and the count makes that legible. The `- occurrences:` line is
// shared marker grammar across the dogfoodissues producers, so its reader (and its pattern)
// live once, in guardcomplaint.OccurrencesOf, rather than being re-declared here.

// knownBadIssueKey is the stable dedup key. The signature already folds (reason class, tree
// globs, failure hash) into one content hash, so it uniquely identifies the shared failure;
// prefixing it namespaces the marker apart from other issue producers that share the
// dogfoodissues marker grammar.
func knownBadIssueKey(rec knownbad.Record) string {
	return "known-bad/" + strings.TrimSpace(rec.Signature)
}

// knownBadIssueTitle renders a self-describing, single-line title: the failure class and
// the affected tree at a glance, so the tracker shows what is broken without opening it.
func knownBadIssueTitle(rec knownbad.Record) string {
	trees := strings.Join(rec.TreeGlobs, ", ")
	if strings.TrimSpace(trees) == "" {
		trees = "(no declared tree)"
	}
	return fmt.Sprintf("known-bad [%s] shared across the fleet — %s", rec.ReasonClass, trees)
}

// knownBadOccurrencesOf is guardcomplaint.OccurrencesOf: absent or unparseable reads as 0,
// so a malformed body restarts the count at 1 on the next file. Both surfaces file through
// the same internal/dogfoodissues plumbing and used to carry byte-identical private readers.
func knownBadOccurrencesOf(body string) int { return guardcomplaint.OccurrencesOf(body) }

// knownBadIssueBody renders the marker-stamped issue body. Line 1 is the dogfoodissues dedup
// marker (so dogfoodissues.Sync's create-verification passes and a re-list matches this
// issue by key); the rest is the evidence a maintainer needs to act — how many DISTINCT
// traces shared the failure, the window, the affected tree, the failure hash, and how the
// signature self-heals if the fix lands quietly.
func knownBadIssueBody(rec knownbad.Record, cand guardrsi.KnownBadCandidate, occurrences int) string {
	if occurrences < 1 {
		occurrences = 1
	}
	key := knownBadIssueKey(rec)
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-dogfood-action-key: %s -->\n", key)
	b.WriteString("# Fleet-correlated known-bad signature\n\n")
	b.WriteString("`fak knownbad correlate` folded cross-trace guard/livelock observations and found ")
	b.WriteString("a single failure that **multiple independent agent traces hit at the same time** — the ")
	b.WriteString("signal that a shared blocker (not a per-agent loop) is in the way.\n\n")
	fmt.Fprintf(&b, "- reason class: `%s`\n", rec.ReasonClass)
	if trees := strings.Join(rec.TreeGlobs, ", "); strings.TrimSpace(trees) != "" {
		fmt.Fprintf(&b, "- affected tree: `%s`\n", trees)
	}
	fmt.Fprintf(&b, "- distinct traces: `%d`\n", cand.DistinctTraces)
	if cand.WindowSecs > 0 {
		fmt.Fprintf(&b, "- correlation window: `%ds`\n", cand.WindowSecs)
	}
	fmt.Fprintf(&b, "- failure hash: `%s`\n", cand.FailureHash)
	fmt.Fprintf(&b, "- signature: `%s`\n", rec.Signature)
	fmt.Fprintf(&b, "- occurrences: `%d`\n", occurrences)
	if ids := knownBadTraceSample(cand.TraceIDs); ids != "" {
		fmt.Fprintf(&b, "- example traces: %s\n", ids)
	}
	b.WriteString("\n## What to do\n\n")
	b.WriteString("Elect one fixer with `fak knownbad claim " + rec.Signature + "` (an exclusive lease over the ")
	b.WriteString("broken tree, so a second agent parks instead of racing a duplicate edit), fix the shared ")
	b.WriteString("cause, then `fak knownbad resolve " + rec.Signature + "` — which requires an independent ")
	b.WriteString("witness (green tests / `dos verify`), not a self-report.\n\n")
	b.WriteString("---\n")
	b.WriteString("Filed by `fak knownbad correlate`. It re-files onto THIS issue in place and bumps the ")
	b.WriteString("occurrence count when the signature re-correlates; the underlying ledger signature carries a ")
	b.WriteString("bounded TTL, so if the fix lands quietly the signature auto-expires and this issue stops ")
	b.WriteString("re-firing.\n")
	return b.String()
}

// knownBadTraceSample renders up to three trace ids inline as code spans, so the body shows
// concrete evidence without pasting an unbounded list.
func knownBadTraceSample(ids []string) string {
	const max = 3
	out := make([]string, 0, max)
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		out = append(out, "`"+id+"`")
		if len(out) == max {
			break
		}
	}
	sample := strings.Join(out, ", ")
	if len(ids) > len(out) && sample != "" {
		sample += fmt.Sprintf(" (+%d more)", len(ids)-len(out))
	}
	return sample
}

// buildKnownBadIssuePlan decides create vs update for one signature against the existing
// issues (matched by the shared dogfoodissues marker key) and computes the escalated
// occurrence count. Pure: no gh, no clock — the correlate verb supplies existing and
// materializes the returned row through dogfoodissues.Sync.
func buildKnownBadIssuePlan(rec knownbad.Record, cand guardrsi.KnownBadCandidate, existing []dogfoodissues.Issue) dogfoodissues.PlanRow {
	key := knownBadIssueKey(rec)
	row := dogfoodissues.PlanRow{
		Action: "create",
		Key:    key,
		Title:  knownBadIssueTitle(rec),
	}
	occurrences := 1
	for _, issue := range existing {
		if dogfoodissues.MarkerKey(issue.Body) != key {
			continue
		}
		row.Action = "update"
		n := issue.Number
		row.Number = &n
		row.State = issue.State
		occurrences = knownBadOccurrencesOf(issue.Body) + 1
		break
	}
	row.Body = knownBadIssueBody(rec, cand, occurrences)
	return row
}
