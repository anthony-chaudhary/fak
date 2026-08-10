package issuepolicy

import "strings"

// TemplateApply is the computed result of turning an issue body whose top
// generated-metadata header still carries unexpanded batch-filer tokens into a
// repaired body. It is produced purely (no gh, no I/O) so the batch repair
// consumer can decide whether a live `gh issue edit` is safe BEFORE shelling
// out — the merge either provably drops no human-authored content (Safe) or it
// refuses with a closed reason (Unsafe) and the fix is left to an operator.
type TemplateApply struct {
	Plan    TemplateRepairPlan // the dry-run plan: detected markers + proposed header
	NewBody string             // the repaired body; meaningful only when Safe
	Safe    bool               // true only when the merge is provably lossless
	Unsafe  string             // closed reason it could not be auto-applied (when !Safe)
}

// Closed set of reasons a template row cannot be auto-repaired into a new body.
// Any of these downgrades the row to propose-only: the corruption is not the
// known top-generated-header shape, so a human/agent applies it via
// `fak issue edit` rather than an unattended sweep guessing at the boundary.
const (
	TemplateUnsafeMarkersSpanSections = "markers-span-multiple-sections"
	TemplateUnsafeUnexpectedProse     = "unexpected-prose-in-header-block"
	TemplateUnsafeMarkerSurvives      = "marker-survives-merge"
	TemplateUnsafeNoHeaderBlock       = "no-generated-header-block"
)

// ApplyTemplateRepair computes the repaired body for an issue whose top
// generated-metadata header block still contains unexpanded batch-filer tokens
// (contract_test.go:195 fixture). It returns ok=false when the body has no such
// markers (nothing to repair) so the caller can skip the row.
//
// Why this is safe to run behind --live: the batch filer leaves its corrupt
// tokens ONLY in the top generated header block — a single "## <heading>"
// followed by "- key: value" bullets — while the human/agent-authored body
// below is intact. The merge therefore replaces exactly [line 0 .. last-marker
// line] with the recomputed ProposedNormalizedHeader and preserves every line
// below verbatim. It fails CLOSED (Safe=false, ok=true) unless all four hold:
//
//   - a marker line was actually located (else TemplateUnsafeMarkerSurvives),
//   - the replaced region contains exactly one ATX heading — more than one
//     means the markers bleed into a second section that could be human content
//     (TemplateUnsafeMarkersSpanSections), zero means the corrupt block is not a
//     clean generated header (TemplateUnsafeNoHeaderBlock),
//   - every non-blank line in the replaced region is a heading, a "- "/"* "
//     bullet, or a marker line — a prose line signals human content in the
//     header block (TemplateUnsafeUnexpectedProse),
//   - no unexpanded marker survives in the merged body
//     (TemplateUnsafeMarkerSurvives), which also catches a marker hiding in the
//     preserved tail.
func ApplyTemplateRepair(d IssueDraft) (TemplateApply, bool) {
	plan, ok := BuildTemplateRepairPlan(d)
	if !ok {
		return TemplateApply{}, false
	}
	body := strings.ReplaceAll(d.Body, "\r\n", "\n")
	lines := strings.Split(body, "\n")

	lastMarker := -1
	for i, line := range lines {
		if unexpandedIssueTemplateMarkerRE.MatchString(line) {
			lastMarker = i
		}
	}
	if lastMarker < 0 {
		return TemplateApply{Plan: plan, Unsafe: TemplateUnsafeMarkerSurvives}, true
	}

	// The replaced region is [0 .. lastMarker]. It must look like exactly one
	// generated header block: one heading, and only heading/bullet/blank/marker
	// lines. Anything else means we cannot prove the region is machine-authored.
	headings := 0
	for i := 0; i <= lastMarker; i++ {
		t := strings.TrimSpace(lines[i])
		switch {
		case t == "":
			// blank — fine
		case isATXHeading(t):
			headings++
		case strings.HasPrefix(t, "- "), strings.HasPrefix(t, "* "):
			// list bullet — fine (markers live inside these too)
		case unexpandedIssueTemplateMarkerRE.MatchString(lines[i]):
			// a marker line that is not a bullet (e.g. a bare token) — fine
		default:
			return TemplateApply{Plan: plan, Unsafe: TemplateUnsafeUnexpectedProse}, true
		}
	}
	if headings == 0 {
		return TemplateApply{Plan: plan, Unsafe: TemplateUnsafeNoHeaderBlock}, true
	}
	if headings > 1 {
		return TemplateApply{Plan: plan, Unsafe: TemplateUnsafeMarkersSpanSections}, true
	}

	// Tail = everything after the last marker line, minus leading blanks so the
	// join controls the single blank-line gap under the new header.
	tailStart := lastMarker + 1
	for tailStart < len(lines) && strings.TrimSpace(lines[tailStart]) == "" {
		tailStart++
	}
	tail := strings.Join(lines[tailStart:], "\n")
	header := strings.TrimRight(plan.ProposedNormalizedHeader, "\n")

	var newBody string
	if strings.TrimSpace(tail) == "" {
		newBody = header + "\n"
	} else {
		newBody = header + "\n\n" + tail
	}
	if len(UnexpandedTemplateMarkers(newBody)) > 0 {
		return TemplateApply{Plan: plan, Unsafe: TemplateUnsafeMarkerSurvives}, true
	}
	return TemplateApply{Plan: plan, NewBody: newBody, Safe: true}, true
}

// isATXHeading reports whether a trimmed line is a Markdown ATX heading
// ("# " through "###### "): one-to-six leading '#' followed by a space/tab.
func isATXHeading(trimmed string) bool {
	n := 0
	for n < len(trimmed) && trimmed[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(trimmed) {
		return false
	}
	return trimmed[n] == ' ' || trimmed[n] == '\t'
}
