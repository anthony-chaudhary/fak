package livecodebench

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// UpstreamProblem is the subset of a LiveCodeBench code_generation dataset row
// (the HuggingFace `livecodebench/code_generation_lite` shape) that a fak Suite
// needs. Upstream stores public_test_cases as a JSON-encoded STRING and
// contest_date as an RFC3339-ish timestamp; Normalize turns both into the
// Suite's native shapes. Fields we do not consume (private_test_cases, which
// upstream ships zlib+base64-encoded, and metadata) are intentionally omitted:
// the fak arm generates from the prompt, and grading runs through lcb_runner.
type UpstreamProblem struct {
	QuestionID      string `json:"question_id"`
	QuestionTitle   string `json:"question_title"`
	QuestionContent string `json:"question_content"`
	Platform        string `json:"platform"`
	Difficulty      string `json:"difficulty"`
	ContestDate     string `json:"contest_date"`
	StarterCode     string `json:"starter_code"`
	PublicTestCases string `json:"public_test_cases"`
}

// hfRowsEnvelope is the HuggingFace datasets-server /rows response shape:
// {"rows":[{"row":{...upstream fields...}}]}. ParseUpstreamRows accepts either
// this envelope or a bare JSON array of UpstreamProblem, so an offline replay
// file and a live datasets-server body normalize through the same path.
type hfRowsEnvelope struct {
	Rows []struct {
		Row UpstreamProblem `json:"row"`
	} `json:"rows"`
}

// ParseUpstreamRows decodes upstream problems from either a bare JSON array of
// UpstreamProblem or a HuggingFace datasets-server {"rows":[{"row":...}]}
// envelope, dispatching on the first non-space byte.
func ParseUpstreamRows(data []byte) ([]UpstreamProblem, error) {
	trimmed := strings.TrimSpace(string(data))
	if strings.HasPrefix(trimmed, "[") {
		var arr []UpstreamProblem
		if err := json.Unmarshal(data, &arr); err != nil {
			return nil, fmt.Errorf("livecodebench normalize: upstream array: %w", err)
		}
		return arr, nil
	}
	var env hfRowsEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("livecodebench normalize: upstream rows envelope: %w", err)
	}
	out := make([]UpstreamProblem, 0, len(env.Rows))
	for _, r := range env.Rows {
		out = append(out, r.Row)
	}
	return out, nil
}

// NormalizeOptions pins the identity a normalized Suite records. Release and the
// provenance dataset id are required; Scenario defaults to codegeneration (the
// scenario code_generation_lite carries). Revision defaults to the resolved
// release when empty. FetchedAt is injected, never read from the wall clock
// here, so a normalized suite is byte-for-byte reproducible in a test or re-run.
type NormalizeOptions struct {
	Release   string
	Scenario  Scenario
	DatasetID string
	Revision  string
	Split     string
	FetchedAt string
	Model     string
}

// Normalize maps upstream LiveCodeBench rows into a release-pinned, sourced
// Suite. It normalizes contest_date to YYYY-MM-DD, lowercases platform and
// difficulty to the Suite's enums, parses the upstream public_test_cases JSON
// string into TestCases, and stamps a provenance header (dataset, revision,
// problem count, contest-date range). The returned Suite is Validated, so a
// normalized suite is always loadable and always carries its provenance.
func Normalize(ups []UpstreamProblem, opts NormalizeOptions) (Suite, error) {
	scenario := opts.Scenario
	if scenario == "" {
		scenario = ScenarioCodeGeneration
	}
	if !KnownScenario(scenario) {
		return Suite{}, fmt.Errorf("livecodebench normalize: scenario %q is not supported", scenario)
	}
	rel, err := ResolveRelease(opts.Release)
	if err != nil {
		return Suite{}, err
	}
	problems := make([]Problem, 0, len(ups))
	var minDate, maxDate string
	for i, u := range ups {
		id := strings.TrimSpace(u.QuestionID)
		if id == "" {
			return Suite{}, fmt.Errorf("livecodebench normalize: upstream row %d has no question_id", i)
		}
		prompt := strings.TrimSpace(u.QuestionContent)
		if prompt == "" {
			return Suite{}, fmt.Errorf("livecodebench normalize: problem %q has no question_content", id)
		}
		date, err := normalizeContestDate(u.ContestDate)
		if err != nil {
			return Suite{}, fmt.Errorf("livecodebench normalize: problem %q: %w", id, err)
		}
		public, err := parseUpstreamTests(u.PublicTestCases)
		if err != nil {
			return Suite{}, fmt.Errorf("livecodebench normalize: problem %q: %w", id, err)
		}
		problems = append(problems, Problem{
			QuestionID:  id,
			Scenario:    scenario,
			Platform:    strings.ToLower(strings.TrimSpace(u.Platform)),
			Difficulty:  strings.ToLower(strings.TrimSpace(u.Difficulty)),
			ContestDate: date,
			Prompt:      prompt,
			StarterCode: u.StarterCode,
			PublicTests: public,
		})
		if date != "" {
			if minDate == "" || date < minDate {
				minDate = date
			}
			if maxDate == "" || date > maxDate {
				maxDate = date
			}
		}
	}
	revision := strings.TrimSpace(opts.Revision)
	if revision == "" {
		revision = rel.Resolved
	}
	suite := Suite{
		Schema:         SuiteSchema,
		Benchmark:      Benchmark,
		Model:          strings.TrimSpace(opts.Model),
		ReleaseVersion: rel.Resolved,
		Provenance: Provenance{
			DatasetID:       strings.TrimSpace(opts.DatasetID),
			Revision:        revision,
			Split:           strings.TrimSpace(opts.Split),
			FetchedAt:       strings.TrimSpace(opts.FetchedAt),
			ProblemCount:    len(problems),
			ContestDateFrom: minDate,
			ContestDateTo:   maxDate,
		},
		Problems: problems,
	}
	if err := suite.Validate(); err != nil {
		return Suite{}, err
	}
	return suite, nil
}

// normalizeContestDate reduces an upstream contest_date to the Suite's
// YYYY-MM-DD form. Upstream carries either a bare date or an RFC3339-ish
// timestamp ("2023-05-20T00:00:00"); an empty date stays empty (undated problems
// are legal but fall outside every contamination window). It errors on an
// unparseable date rather than dropping it silently.
func normalizeContestDate(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", nil
	}
	if len(s) >= 10 {
		if t, err := time.Parse(dateLayout, s[:10]); err == nil {
			return t.Format(dateLayout), nil
		}
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.Format(dateLayout), nil
		}
	}
	return "", fmt.Errorf("contest_date %q is not a recognizable date", raw)
}

// parseUpstreamTests decodes upstream public_test_cases, a JSON-encoded STRING
// holding [{input,output,testtype}]. An empty string yields no tests (some rows
// ship none). testtype is carried through as-is (stdin | functional).
func parseUpstreamTests(raw string) ([]TestCase, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	var upstream []struct {
		Input    string `json:"input"`
		Output   string `json:"output"`
		TestType string `json:"testtype"`
	}
	if err := json.Unmarshal([]byte(s), &upstream); err != nil {
		return nil, fmt.Errorf("public_test_cases is not decodable JSON: %w", err)
	}
	out := make([]TestCase, 0, len(upstream))
	for _, t := range upstream {
		out = append(out, TestCase{Input: t.Input, Output: t.Output, TestType: t.TestType})
	}
	return out, nil
}
