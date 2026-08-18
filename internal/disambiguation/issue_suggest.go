package disambiguation

import (
	"errors"
	"fmt"
	"strings"
)

const IssueSuggestionSchemaVersion = "fak-disambiguation-issue-suggestion/1"

var ErrIssueSuggestionUnsafe = errors.New("unsafe issue suggestion candidate")
var ErrIssueSuggestionIncomplete = errors.New("incomplete issue suggestion candidate")

type IssueSuggestion struct {
	Schema         string   `json:"schema"`
	Title          string   `json:"title"`
	Problem        string   `json:"problem"`
	DoneCondition  string   `json:"done_condition"`
	AcceptanceGate string   `json:"acceptance_gate"`
	LikelyFiles    []string `json:"likely_files"`
	DedupeQuery    string   `json:"dedupe_query"`
}

func SuggestIssue(finding CoverageFinding) (IssueSuggestion, error) {
	term := strings.TrimSpace(finding.Candidate)
	if term == "" {
		term = strings.TrimSpace(finding.Term)
	}
	if term == "" || strings.TrimSpace(finding.Surface) == "" || strings.TrimSpace(finding.Reason) == "" {
		return IssueSuggestion{}, ErrIssueSuggestionIncomplete
	}
	probe := cloneEntry(publicEntries[0])
	probe.Identity.CanonicalTerm = term
	probe.Identity.Aliases = []string{}
	probe.Definition = "Review uncovered public terminology candidate " + term
	probe.Contrasts = []Contrast{{CanonicalTerm: publicEntries[1].Identity.CanonicalTerm, Explanation: "Candidate requires a reviewed distinction before admission."}}
	probe.Sources[0].Locator = finding.Surface
	if err := validatePublicSafety(probe); err != nil {
		return IssueSuggestion{}, fmt.Errorf("%w: %v", ErrIssueSuggestionUnsafe, err)
	}
	slug := strings.ToLower(strings.Join(strings.Fields(term), "-"))
	return IssueSuggestion{Schema: IssueSuggestionSchemaVersion, Title: "feat(disambiguation): classify " + term, Problem: "The public terminology candidate `" + term + "` is uncovered (" + finding.Reason + ") on `" + finding.Surface + "`.", DoneCondition: "The candidate is reviewed as canonical or incidental through the public disambiguation seam, with its distinction and owner recorded.", AcceptanceGate: "A deterministic fixture resolves the classification and rejects an incompatible conflation.", LikelyFiles: []string{"internal/disambiguation"}, DedupeQuery: "is:issue repo:anthony-chaudhary/fak in:title \"" + slug + "\""}, nil
}

type IssueSuggestionSelfTestReport struct {
	Schema         string          `json:"schema"`
	Suggestion     IssueSuggestion `json:"suggestion"`
	UnsafeRejected bool            `json:"unsafe_rejected"`
	NoAutoFile     bool            `json:"no_auto_file"`
}

func RunIssueSuggestionSelfTest() (IssueSuggestionSelfTestReport, error) {
	good := CoverageFinding{Surface: "internal/example/api.go", Term: "NewTerm", Candidate: "new term", Reason: CoverageReasonMissingClassification}
	suggestion, err := SuggestIssue(good)
	if err != nil {
		return IssueSuggestionSelfTestReport{}, err
	}
	_, unsafeErr := SuggestIssue(CoverageFinding{Surface: "/home/example/secret.go", Term: "SecretTerm", Candidate: "secret term", Reason: CoverageReasonMissingClassification})
	report := IssueSuggestionSelfTestReport{Schema: IssueSuggestionSchemaVersion, Suggestion: suggestion, UnsafeRejected: errors.Is(unsafeErr, ErrIssueSuggestionUnsafe), NoAutoFile: true}
	if !report.UnsafeRejected {
		return report, errors.New("unsafe candidate accepted")
	}
	return report, nil
}
