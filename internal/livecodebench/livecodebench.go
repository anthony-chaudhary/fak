package livecodebench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

const (
	FixtureSchema = "fak.livecodebench.fixture.v1"
	ReportSchema  = "fak.livecodebench.report.v1"
)

var RequiredScenarios = []string{
	"codegeneration",
	"selfrepair",
	"testoutputprediction",
	"codeexecution",
}

type Fixture struct {
	Schema         string        `json:"schema"`
	ReleaseVersion string        `json:"release_version"`
	StartDate      string        `json:"start_date"`
	EndDate        string        `json:"end_date"`
	Items          []FixtureItem `json:"items"`
}

type FixtureItem struct {
	QuestionID string   `json:"question_id"`
	Scenario   string   `json:"scenario"`
	Prompt     string   `json:"prompt"`
	CodeList   []string `json:"code_list,omitempty"`
}

type Report struct {
	Schema             string           `json:"schema"`
	FixtureSchema      string           `json:"fixture_schema"`
	ReleaseVersion     string           `json:"release_version"`
	StartDate          string           `json:"start_date"`
	EndDate            string           `json:"end_date"`
	Questions          int              `json:"questions"`
	Scenarios          []ScenarioReport `json:"scenarios"`
	ResultClaimAllowed bool             `json:"result_claim_allowed"`
	EvidenceClass      string           `json:"evidence_class"`
	PromotionRequired  []string         `json:"promotion_required"`
}

type ScenarioReport struct {
	Scenario  string `json:"scenario"`
	Questions int    `json:"questions"`
}

func Load(r io.Reader) (Fixture, error) {
	var f Fixture
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return Fixture{}, err
	}
	if err := f.Validate(); err != nil {
		return Fixture{}, err
	}
	return f, nil
}

func LoadFile(path string) (Fixture, error) {
	file, err := os.Open(path)
	if err != nil {
		return Fixture{}, err
	}
	defer file.Close()
	return Load(file)
}

func (f Fixture) Validate() error {
	if strings.TrimSpace(f.Schema) != FixtureSchema {
		return fmt.Errorf("livecodebench fixture: schema = %q, want %q", f.Schema, FixtureSchema)
	}
	if strings.TrimSpace(f.ReleaseVersion) == "" {
		return fmt.Errorf("livecodebench fixture: release_version is required")
	}
	if strings.TrimSpace(f.StartDate) == "" || strings.TrimSpace(f.EndDate) == "" {
		return fmt.Errorf("livecodebench fixture: start_date and end_date are required")
	}
	if len(f.Items) == 0 {
		return fmt.Errorf("livecodebench fixture: at least one item is required")
	}
	allowed := scenarioSet(RequiredScenarios)
	seenIDs := map[string]bool{}
	for i, item := range f.Items {
		if strings.TrimSpace(item.QuestionID) == "" {
			return fmt.Errorf("livecodebench fixture: item %d question_id is required", i)
		}
		if seenIDs[item.QuestionID] {
			return fmt.Errorf("livecodebench fixture: duplicate question_id %q", item.QuestionID)
		}
		seenIDs[item.QuestionID] = true
		if !allowed[item.Scenario] {
			return fmt.Errorf("livecodebench fixture: item %q scenario %q is not supported", item.QuestionID, item.Scenario)
		}
	}
	return nil
}

func SmokeReport(f Fixture) Report {
	counts := map[string]int{}
	for _, item := range f.Items {
		counts[item.Scenario]++
	}
	scenarios := make([]ScenarioReport, 0, len(counts))
	for _, name := range RequiredScenarios {
		if counts[name] == 0 {
			continue
		}
		scenarios = append(scenarios, ScenarioReport{Scenario: name, Questions: counts[name]})
	}
	return Report{
		Schema:             ReportSchema,
		FixtureSchema:      f.Schema,
		ReleaseVersion:     f.ReleaseVersion,
		StartDate:          f.StartDate,
		EndDate:            f.EndDate,
		Questions:          len(f.Items),
		Scenarios:          scenarios,
		ResultClaimAllowed: false,
		EvidenceClass:      "fixture-smoke",
		PromotionRequired: []string{
			"official-lcb-runner-grading",
			"release-version-and-date-window-recorded",
			"generation-artifact-digest-recorded",
		},
	}
}

func ValidateSmokeReport(r Report) error {
	if r.Schema != ReportSchema {
		return fmt.Errorf("livecodebench report: schema = %q, want %q", r.Schema, ReportSchema)
	}
	if r.ResultClaimAllowed {
		return fmt.Errorf("livecodebench report: result_claim_allowed must be false for fixture smoke")
	}
	if r.Questions == 0 {
		return fmt.Errorf("livecodebench report: questions must be non-zero")
	}
	seen := map[string]int{}
	for _, s := range r.Scenarios {
		seen[s.Scenario] = s.Questions
	}
	for _, scenario := range RequiredScenarios {
		if seen[scenario] == 0 {
			return fmt.Errorf("livecodebench report: missing scenario %q", scenario)
		}
	}
	return nil
}

func scenarioSet(in []string) map[string]bool {
	out := make(map[string]bool, len(in))
	for _, s := range in {
		out[s] = true
	}
	return out
}

func ScenarioNames(f Fixture) []string {
	seen := map[string]bool{}
	for _, item := range f.Items {
		seen[item.Scenario] = true
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
