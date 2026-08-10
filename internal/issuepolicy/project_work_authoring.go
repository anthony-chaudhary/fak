package issuepolicy

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

type ProjectWorkAuthoring struct {
	EstimatePoints     float64
	ParentBaseline     float64
	ContributionPoints float64
	CompletionStandard string
	TargetEnvelope     string
	WitnessedEnvelope  string
}

var projectWorkHeadingRE = regexp.MustCompile(`(?im)^#{2,6}\s+(.+?)\s*$`)

// AppendProjectWorkDefaults adds canonical project-work sections without
// rewriting explicit declarations. Numeric scope is never inferred: callers
// must supply estimate and parent baseline when those sections are absent.
func AppendProjectWorkDefaults(body string, a ProjectWorkAuthoring) (string, error) {
	headings := map[string]bool{}
	for _, m := range projectWorkHeadingRE.FindAllStringSubmatch(body, -1) {
		headings[normalizeHeading(m[1])] = true
	}
	needEstimate := !headings[normalizeHeading("Work estimate")]
	needContribution := !headings[normalizeHeading("Overall completion contribution")] && !headings[normalizeHeading("Scope contribution")]
	needCompletion := !headings[normalizeHeading("Completion standard")]
	if needEstimate && !finitePositive(a.EstimatePoints) {
		return "", fmt.Errorf("project-work authoring requires --estimate-points > 0 when ## Work estimate is absent")
	}
	if needContribution && !finitePositive(a.ParentBaseline) {
		return "", fmt.Errorf("project-work authoring requires --parent-baseline-points > 0 when ## Overall completion contribution is absent")
	}
	contribution := a.ContributionPoints
	if contribution == 0 {
		contribution = a.EstimatePoints
	}
	if needContribution && (!finitePositive(contribution) || contribution > a.ParentBaseline) {
		return "", fmt.Errorf("project-work contribution must be > 0 and <= parent baseline (got %g/%g)", contribution, a.ParentBaseline)
	}
	standard := normalizeCompletionStandard(a.CompletionStandard)
	if !needCompletion {
		sections := markdownSections(body)
		standard = normalizeCompletionStandard(sections[normalizeHeading("Completion standard")])
	}
	if standard == "" {
		standard = "production"
	}
	switch standard {
	case "production", "research", "experiment", "prototype", "demo", "development", "dev", "integrated", "staging":
	default:
		return "", fmt.Errorf("unsupported completion standard %q", a.CompletionStandard)
	}
	out := strings.TrimSpace(body)
	appendSection := func(title, value string) {
		if out != "" {
			out += "\n\n"
		}
		out += "## " + title + "\n" + value
	}
	if needEstimate {
		appendSection("Work estimate", fmt.Sprintf("Estimate: %s points. Uncertainty: operator-supplied estimate; revise through the parent denominator when evidence changes.", formatPoints(a.EstimatePoints)))
	}
	if needContribution {
		appendSection("Overall completion contribution", fmt.Sprintf("Contribution: %s/%s points.", formatPoints(contribution), formatPoints(a.ParentBaseline)))
	}
	if needCompletion {
		appendSection("Completion standard", standard)
	}
	if standard == "production" && !headings[normalizeHeading("Target operating envelope")] && !headings[normalizeHeading("Target envelope")] {
		if strings.TrimSpace(a.TargetEnvelope) == "" {
			return "", fmt.Errorf("production authoring requires --target-envelope when ## Target operating envelope is absent")
		}
		appendSection("Target operating envelope", strings.TrimSpace(a.TargetEnvelope))
	}
	if standard == "production" && !headings[normalizeHeading("Witnessed operating envelope")] && !headings[normalizeHeading("Observed operating envelope")] && !headings[normalizeHeading("Witnessed envelope")] {
		if strings.TrimSpace(a.WitnessedEnvelope) == "" {
			return "", fmt.Errorf("production authoring requires --witnessed-envelope when ## Witnessed operating envelope is absent")
		}
		appendSection("Witnessed operating envelope", strings.TrimSpace(a.WitnessedEnvelope))
	}
	return out + "\n", nil
}

func finitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }
func formatPoints(v float64) string { return fmt.Sprintf("%g", v) }

// BatchProjectWork describes deterministic metadata for a generated child. The
// producer must know the parent denominator; this helper never guesses it.
type BatchProjectWork struct {
	ParentIssue        int
	EstimatePoints     float64
	ParentBaseline     float64
	CompletionStandard string
	TargetEnvelope     string
	WitnessedEnvelope  string
}

// AuthorBatchProjectWork adds canonical project-work sections to a generated
// body, defaulting unqualified maturity to production.
func AuthorBatchProjectWork(body string, in BatchProjectWork) (string, error) {
	if in.ParentIssue <= 0 {
		return "", fmt.Errorf("batch project work: parent issue must be positive")
	}
	if !finitePositive(in.EstimatePoints) || !finitePositive(in.ParentBaseline) || in.EstimatePoints > in.ParentBaseline {
		return "", fmt.Errorf("batch project work: estimate and baseline must be positive and estimate cannot exceed baseline")
	}
	standard := normalizeCompletionStandard(in.CompletionStandard)
	if standard == "" {
		standard = "production"
	}
	return AppendProjectWorkDefaults(body, ProjectWorkAuthoring{
		EstimatePoints: in.EstimatePoints, ContributionPoints: in.EstimatePoints,
		ParentBaseline: in.ParentBaseline, CompletionStandard: standard,
		TargetEnvelope: in.TargetEnvelope, WitnessedEnvelope: in.WitnessedEnvelope,
	})
}
