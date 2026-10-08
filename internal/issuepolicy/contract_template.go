package issuepolicy

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// HasUnexpandedTemplate reports whether an already-rendered issue body still
// contains PowerShell template tokens from a failed batch filer. Such a body is
// not safe to dispatch: routing metadata may be corrupt even when later
// acceptance sections are intact.
func HasUnexpandedTemplate(body string) bool {
	return unexpandedIssueTemplateRE.MatchString(body)
}

// UnexpandedTemplateMarkers returns the literal corrupt tokens that make a
// generated issue unsafe to dispatch. The markers are ordered by first
// appearance so the first row is the cheapest human triage cue.
func UnexpandedTemplateMarkers(body string) []string {
	matches := unexpandedIssueTemplateMarkerRE.FindAllString(body, -1)
	seen := map[string]bool{}
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		match = strings.TrimSpace(match)
		if match == "" || seen[match] {
			continue
		}
		seen[match] = true
		out = append(out, match)
	}
	return out
}

// BuildTemplateRepairPlan returns a dry-run repair plan for a corrupt generated
// issue body. It does not mutate GitHub; callers decide whether to render the
// plan as JSON, text, or an operator-reviewed edit.
func BuildTemplateRepairPlan(d IssueDraft) (TemplateRepairPlan, bool) {
	markers := UnexpandedTemplateMarkers(d.Body)
	if len(markers) == 0 {
		return TemplateRepairPlan{}, false
	}
	candidate := CandidateFromIssueDraft(d)
	return TemplateRepairPlan{
		Schema:                   TemplateRepairSchema,
		IssueNumber:              d.Number,
		Title:                    strings.TrimSpace(d.Title),
		URL:                      strings.TrimSpace(d.URL),
		Key:                      candidate.Key,
		DetectedMarker:           markers[0],
		DetectedMarkers:          markers,
		ProposedNormalizedHeader: proposedTemplateRepairHeader(d, candidate),
		DryRunOnly:               true,
	}, true
}

func proposedTemplateRepairHeader(d IssueDraft, c Candidate) string {
	labelStream, _ := generationStreamFromLabels(c.Labels)
	stream := strmatch.FirstTrimmed(labelStream, generationStreamFromCandidate(c), generationStreamFromText(d.Body))
	if stream != "" {
		return proposedGenerationHeader(d, c, stream)
	}
	if isManagedContextIssue(d, c) {
		return proposedManagedContextHeader(d, c)
	}
	return proposedGenericIssueHeader(d, c)
}

func proposedGenerationHeader(d IssueDraft, c Candidate, stream string) string {
	parent := strmatch.FirstTrimmed(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef)
	lines := []string{
		"## Generation stream",
		"- Generation: " + stream,
		"- Stream rule: " + generationStreamRule(stream),
		"- Milestone: " + generationMilestone(stream),
	}
	if parent != "" {
		lines = append(lines, "- Parent: "+parent)
	}
	return strings.Join(lines, "\n")
}

func proposedManagedContextHeader(d IssueDraft, c Candidate) string {
	key := issueDraftMarkerKey(d.Body)
	track := strmatch.FirstTrimmed(managedContextTrackFromKey(key), cleanIssueHeaderValue(issueHeaderField(d.Body, "Track")), "needs-track")
	parent := strmatch.FirstTrimmed(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef, "https://github.com/anthony-chaudhary/fak/issues/1570")
	lines := []string{}
	if key != "" {
		lines = append(lines, "<!-- fak-managed-context-key: "+key+" -->")
	}
	title := strings.TrimSpace(d.Title)
	if title == "" {
		title = "managed-context: repair generated issue metadata"
	}
	lines = append(lines,
		"# "+title,
		"",
		"- Program: managed-context",
		"- Track: "+track,
		"- Parent: "+parent,
	)
	return strings.Join(lines, "\n")
}

func proposedGenericIssueHeader(d IssueDraft, c Candidate) string {
	title := strings.TrimSpace(d.Title)
	if title == "" {
		title = "generated issue metadata repair"
	}
	lines := []string{
		"## Issue metadata",
		"- Title: " + title,
	}
	if c.Key != "" {
		lines = append(lines, "- Key: "+c.Key)
	}
	if parent := strmatch.FirstTrimmed(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef); parent != "" {
		lines = append(lines, "- Parent: "+parent)
	}
	return strings.Join(lines, "\n")
}

func issueHeaderField(body, name string) string {
	want := strings.ToLower(strings.TrimSpace(name))
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := trimListPrefix(raw)
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(strings.Trim(key, "`*_ ")))
		if key == want {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func cleanIssueHeaderValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	if strings.Contains(value, "$(") || strings.Contains(value, "$(@{") ||
		strings.Contains(lower, "system.") || strings.Contains(lower, "$source") {
		return ""
	}
	return value
}

func generationStreamRule(stream string) string {
	switch stream {
	case "gen/now":
		return "Now gen: current product/runtime work with direct default-path evidence."
	case "gen/next":
		return "Next gen: near-term foundation that should become agent-runnable after gates, handoffs, and operator visibility exist."
	case "gen/second-next":
		return "Second-next gen: architecture and option work that can promote after next-gen gates prove out."
	case "gen/future":
		return "Future gen: long-horizon research, standards, narratives, and option-value work."
	default:
		return "Generation stream should be re-rendered from labels and issue scope."
	}
}

func generationMilestone(stream string) string {
	switch stream {
	case "gen/now":
		return "Generation G0 - Now"
	case "gen/next":
		return "Generation G1 - Next Gen"
	case "gen/second-next":
		return "Generation G2 - Second Next"
	case "gen/future":
		return "Generation G3 - Future"
	default:
		return "Generation milestone needs operator review"
	}
}
