package livecodebench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

const (
	SuiteSchema  = "fak.livecodebench-suite.v1"
	ReportSchema = "fak.livecodebench-report.v1"

	Benchmark = "livecodebench"
)

// Evidence classes, weakest to strongest. Only the official lcb_runner
// grading can back a claimable result.
const (
	EvidenceFixtureSmoke      = "fixture-smoke"
	EvidenceLocalUngraded     = "local-ungraded"
	EvidenceOfficialLCBRunner = "official-lcb-runner-graded"
)

// Scenario is one of the four upstream LiveCodeBench scenarios.
type Scenario string

const (
	ScenarioCodeGeneration       Scenario = "codegeneration"
	ScenarioSelfRepair           Scenario = "selfrepair"
	ScenarioTestOutputPrediction Scenario = "testoutputprediction"
	ScenarioCodeExecution        Scenario = "codeexecution"
)

func KnownScenario(s Scenario) bool {
	return scenarioSet(RequiredScenarios)[string(s)]
}

var knownPlatforms = map[string]bool{
	"leetcode":   true,
	"atcoder":    true,
	"codeforces": true,
}

var knownDifficulties = map[string]bool{
	"easy":   true,
	"medium": true,
	"hard":   true,
}

var knownArms = map[string]bool{
	"raw": true,
	"fak": true,
}

// Suite is a normalized, release-pinned LiveCodeBench problem set.
type Suite struct {
	Schema         string     `json:"schema"`
	Benchmark      string     `json:"benchmark"`
	Model          string     `json:"model,omitempty"`
	ReleaseVersion string     `json:"release_version"`
	Provenance     Provenance `json:"provenance"`
	Problems       []Problem  `json:"problems"`
}

// Provenance pins where a normalized Suite came from, so a reader can trace a
// result back to an exact dataset version and problem window instead of an
// implicit "the benchmark". A normalized suite MUST carry it (see
// Suite.Validate) — an unsourced suite cannot back an honest result, and its
// ProblemCount must equal the number of problems it ships so the header cannot
// drift from the body.
type Provenance struct {
	DatasetID       string `json:"dataset_id"`                  // HF dataset, e.g. "livecodebench/code_generation_lite"
	Revision        string `json:"revision"`                    // dataset config/revision the rows were read at (e.g. "release_v2")
	Split           string `json:"split,omitempty"`             // dataset split, e.g. "test"
	FetchedAt       string `json:"fetched_at,omitempty"`        // RFC3339; injected by the caller, never read from the wall clock in pure code
	ProblemCount    int    `json:"problem_count"`               // number of problems the suite carries; must equal len(Problems)
	ContestDateFrom string `json:"contest_date_from,omitempty"` // YYYY-MM-DD, oldest contest_date in the suite
	ContestDateTo   string `json:"contest_date_to,omitempty"`   // YYYY-MM-DD, newest contest_date in the suite
}

// validate checks a Provenance against the problem count it heads. dataset_id and
// revision are required (an unsourced or unpinned suite is refused), the count
// must match, and any recorded dates must be well-formed.
func (p Provenance) validate(problems int) error {
	if strings.TrimSpace(p.DatasetID) == "" {
		return fmt.Errorf("livecodebench suite: provenance.dataset_id is required (an unsourced suite cannot back a result)")
	}
	if strings.TrimSpace(p.Revision) == "" {
		return fmt.Errorf("livecodebench suite: provenance.revision is required (pin the dataset config/revision, never implicit)")
	}
	if p.ProblemCount != problems {
		return fmt.Errorf("livecodebench suite: provenance.problem_count %d does not match problems %d", p.ProblemCount, problems)
	}
	if p.FetchedAt != "" {
		if _, err := time.Parse(time.RFC3339, p.FetchedAt); err != nil {
			return fmt.Errorf("livecodebench suite: provenance.fetched_at %q is not RFC3339", p.FetchedAt)
		}
	}
	for _, d := range []struct{ name, val string }{
		{"contest_date_from", p.ContestDateFrom},
		{"contest_date_to", p.ContestDateTo},
	} {
		if d.val == "" {
			continue
		}
		if _, err := time.Parse(dateLayout, d.val); err != nil {
			return fmt.Errorf("livecodebench suite: provenance.%s %q is not YYYY-MM-DD", d.name, d.val)
		}
	}
	return nil
}

// Problem carries the LCB-native fields for one benchmark question.
type Problem struct {
	QuestionID   string     `json:"question_id"`
	Scenario     Scenario   `json:"scenario"`
	Platform     string     `json:"platform,omitempty"`
	Difficulty   string     `json:"difficulty,omitempty"`
	ContestDate  string     `json:"contest_date,omitempty"`
	Prompt       string     `json:"prompt"`
	StarterCode  string     `json:"starter_code,omitempty"`
	PublicTests  []TestCase `json:"public_test_cases,omitempty"`
	PrivateTests []TestCase `json:"private_test_cases,omitempty"`
}

// TestCase mirrors the upstream test-case shape (testtype: stdin | functional).
type TestCase struct {
	Input    string `json:"input"`
	Output   string `json:"output"`
	TestType string `json:"testtype,omitempty"`
}

// Report is a run report over a Suite. Pass-rate fields cannot be carried
// without an evidence class, and ResultClaimAllowed cannot be true unless the
// evidence class is EvidenceOfficialLCBRunner — see Validate.
type Report struct {
	Schema         string `json:"schema"`
	GeneratedAt    string `json:"generated_at"`
	Benchmark      string `json:"benchmark"`
	Model          string `json:"model,omitempty"`
	ReleaseVersion string `json:"release_version"`
	StartDate      string `json:"start_date,omitempty"`
	EndDate        string `json:"end_date,omitempty"`
	EvidenceClass  string `json:"evidence_class,omitempty"`
	// WindowScore carries the contamination-free date-windowed rate alongside the
	// full-set rate when a run scopes results to a contest-date window (see
	// WindowedPassAtK); nil when the run was not date-windowed.
	WindowScore *WindowScores `json:"window_score,omitempty"`
	Arms        []ArmResult   `json:"arms,omitempty"`
	// Problems are the per-problem verdict rows: each links a question_id to its
	// verdict and the evidence id that backs it, so every row in a rendered
	// report traces to gradeable evidence. Empty for a summary-only report.
	Problems              []ProblemVerdict `json:"problems,omitempty"`
	Summary               Summary          `json:"summary"`
	PromotionRequirements []string         `json:"promotion_requirements,omitempty"`
	ResultClaimAllowed    bool             `json:"result_claim_allowed"`
	ClaimBoundary         string           `json:"claim_boundary,omitempty"`
}

// ArmResult is one (arm, scenario) result cell: raw vs fak, per scenario.
type ArmResult struct {
	Arm         string   `json:"arm"`
	Scenario    Scenario `json:"scenario"`
	Problems    int      `json:"problems"`
	Generations int      `json:"generations"`
	Graded      int      `json:"graded"`
	Pass1       float64  `json:"pass_1,omitempty"`
	Pass5       float64  `json:"pass_5,omitempty"`
}

// Summary folds the suite shape a report ran over.
type Summary struct {
	Problems  int              `json:"problems"`
	Graded    int              `json:"graded"`
	Scenarios []ScenarioReport `json:"scenarios,omitempty"`
}

func LoadSuite(r io.Reader) (Suite, error) {
	var s Suite
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return Suite{}, fmt.Errorf("livecodebench suite: %w", err)
	}
	if err := s.Validate(); err != nil {
		return Suite{}, err
	}
	return s, nil
}

func LoadSuiteFile(path string) (Suite, error) {
	file, err := os.Open(path)
	if err != nil {
		return Suite{}, err
	}
	defer file.Close()
	return LoadSuite(file)
}

func (s Suite) Validate() error {
	if s.Schema != SuiteSchema {
		return fmt.Errorf("livecodebench suite: schema = %q, want %q", s.Schema, SuiteSchema)
	}
	if s.Benchmark != Benchmark {
		return fmt.Errorf("livecodebench suite: benchmark = %q, want %q", s.Benchmark, Benchmark)
	}
	if strings.TrimSpace(s.ReleaseVersion) == "" {
		return fmt.Errorf("livecodebench suite: release_version is required (pin release_vN, never implicit)")
	}
	if len(s.Problems) == 0 {
		return fmt.Errorf("livecodebench suite: at least one problem is required")
	}
	if err := s.Provenance.validate(len(s.Problems)); err != nil {
		return err
	}
	seen := map[string]bool{}
	for i, p := range s.Problems {
		if strings.TrimSpace(p.QuestionID) == "" {
			return fmt.Errorf("livecodebench suite: problem %d question_id is required", i)
		}
		if !KnownScenario(p.Scenario) {
			return fmt.Errorf("livecodebench suite: problem %q scenario %q is not supported", p.QuestionID, p.Scenario)
		}
		key := string(p.Scenario) + "\x00" + p.QuestionID
		if seen[key] {
			return fmt.Errorf("livecodebench suite: duplicate question_id %q for scenario %q", p.QuestionID, p.Scenario)
		}
		seen[key] = true
		if strings.TrimSpace(p.Prompt) == "" {
			return fmt.Errorf("livecodebench suite: problem %q prompt is required", p.QuestionID)
		}
		if p.Platform != "" && !knownPlatforms[p.Platform] {
			return fmt.Errorf("livecodebench suite: problem %q platform %q is not supported", p.QuestionID, p.Platform)
		}
		if p.Difficulty != "" && !knownDifficulties[p.Difficulty] {
			return fmt.Errorf("livecodebench suite: problem %q difficulty %q is not supported", p.QuestionID, p.Difficulty)
		}
		if p.ContestDate != "" {
			if _, err := time.Parse("2006-01-02", p.ContestDate); err != nil {
				return fmt.Errorf("livecodebench suite: problem %q contest_date %q is not YYYY-MM-DD", p.QuestionID, p.ContestDate)
			}
		}
	}
	return nil
}

// NewReport scaffolds an honest, unpromoted report over a suite: local
// evidence only, no result claim, and the promotion requirements spelled out.
func NewReport(s Suite, generatedAt time.Time) Report {
	counts := map[string]int{}
	for _, p := range s.Problems {
		counts[string(p.Scenario)]++
	}
	scenarios := make([]ScenarioReport, 0, len(counts))
	for _, name := range RequiredScenarios {
		if counts[name] == 0 {
			continue
		}
		scenarios = append(scenarios, ScenarioReport{Scenario: name, Questions: counts[name]})
	}
	return Report{
		Schema:                ReportSchema,
		GeneratedAt:           generatedAt.UTC().Format(time.RFC3339),
		Benchmark:             s.Benchmark,
		Model:                 s.Model,
		ReleaseVersion:        s.ReleaseVersion,
		EvidenceClass:         EvidenceLocalUngraded,
		Summary:               Summary{Problems: len(s.Problems), Scenarios: scenarios},
		PromotionRequirements: PromotionRequirements(),
		ResultClaimAllowed:    false,
		ClaimBoundary: "Local generations only: the same saved generations must be graded by the " +
			"official lcb_runner checker before any pass-rate is claimable.",
	}
}

func (r Report) Validate() error {
	if r.Schema != ReportSchema {
		return fmt.Errorf("livecodebench report: schema = %q, want %q", r.Schema, ReportSchema)
	}
	if r.Benchmark != Benchmark {
		return fmt.Errorf("livecodebench report: benchmark = %q, want %q", r.Benchmark, Benchmark)
	}
	if strings.TrimSpace(r.ReleaseVersion) == "" {
		return fmt.Errorf("livecodebench report: release_version is required")
	}
	if strings.TrimSpace(r.GeneratedAt) == "" {
		return fmt.Errorf("livecodebench report: generated_at is required")
	}
	if _, err := time.Parse(time.RFC3339, r.GeneratedAt); err != nil {
		return fmt.Errorf("livecodebench report: generated_at %q is not RFC3339", r.GeneratedAt)
	}
	results := false
	for i, arm := range r.Arms {
		if !knownArms[arm.Arm] {
			return fmt.Errorf("livecodebench report: arm %d name %q is not raw|fak", i, arm.Arm)
		}
		if !KnownScenario(arm.Scenario) {
			return fmt.Errorf("livecodebench report: arm %q scenario %q is not supported", arm.Arm, arm.Scenario)
		}
		if arm.Graded > arm.Generations {
			return fmt.Errorf("livecodebench report: arm %q/%s graded %d exceeds generations %d", arm.Arm, arm.Scenario, arm.Graded, arm.Generations)
		}
		if arm.Pass1 < 0 || arm.Pass1 > 1 || arm.Pass5 < 0 || arm.Pass5 > 1 {
			return fmt.Errorf("livecodebench report: arm %q/%s pass rates must be within [0,1]", arm.Arm, arm.Scenario)
		}
		if arm.Graded > 0 || arm.Pass1 != 0 || arm.Pass5 != 0 {
			results = true
		}
	}
	if results && strings.TrimSpace(r.EvidenceClass) == "" {
		return fmt.Errorf("livecodebench report: pass-rate/result fields require a named evidence_class")
	}
	if r.ResultClaimAllowed && r.EvidenceClass != EvidenceOfficialLCBRunner {
		return fmt.Errorf("livecodebench report: result_claim_allowed requires evidence_class %q, have %q", EvidenceOfficialLCBRunner, r.EvidenceClass)
	}
	if r.ResultClaimAllowed && !results {
		return fmt.Errorf("livecodebench report: result_claim_allowed requires graded results")
	}
	for i, pv := range r.Problems {
		if strings.TrimSpace(pv.QuestionID) == "" {
			return fmt.Errorf("livecodebench report: problem row %d question_id is required", i)
		}
		if pv.Scenario != "" && !KnownScenario(pv.Scenario) {
			return fmt.Errorf("livecodebench report: problem row %q scenario %q is not supported", pv.QuestionID, pv.Scenario)
		}
		if pv.Arm != "" && !knownArms[pv.Arm] {
			return fmt.Errorf("livecodebench report: problem row %q arm %q is not raw|fak", pv.QuestionID, pv.Arm)
		}
		if !knownVerdicts[pv.Verdict] {
			return fmt.Errorf("livecodebench report: problem row %q verdict %q is not %s|%s|%s", pv.QuestionID, pv.Verdict, VerdictUngraded, VerdictPass, VerdictFail)
		}
		if strings.TrimSpace(pv.EvidenceID) == "" {
			return fmt.Errorf("livecodebench report: problem row %q evidence_id is required (every verdict must trace to evidence)", pv.QuestionID)
		}
	}
	return nil
}
