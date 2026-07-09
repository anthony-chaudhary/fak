package livecodebench

import (
	"fmt"
	"strings"
)

// RunReportSchema tags the artifact a fixture run writes.
const RunReportSchema = "fak.livecodebench.run.v1"

// RunConfig captures the lcb_runner.runner.main-parity knobs a fak-native
// LiveCodeBench run is invoked with. Each field mirrors exactly one upstream
// flag so a fak run and a raw `python -m lcb_runner.runner.main` run are
// configured the same way:
//
//	Model       -> --model
//	Scenario    -> --scenario
//	Evaluate    -> --evaluate
//	Release     -> --release_version
//	N           -> -n
//	Temperature -> --temperature
//	UseCache    -> --use_cache
type RunConfig struct {
	Model          string   `json:"model"`
	Scenario       Scenario `json:"scenario"`
	Evaluate       bool     `json:"evaluate"`
	ReleaseVersion string   `json:"release_version"`
	N              int      `json:"n"`
	Temperature    float64  `json:"temperature"`
	UseCache       bool     `json:"use_cache"`
}

// RunReport is the end-to-end artifact a fixture run writes. It records the
// resolved lcb_runner-parity config and the per-scenario question breakdown of
// the committed fixture. It is a fixture smoke, never a claimable score:
// ResultClaimAllowed stays false and promotion still requires the official
// lcb_runner grading (the same honesty fence SmokeReport carries).
type RunReport struct {
	Schema             string           `json:"schema"`
	Config             RunConfig        `json:"config"`
	ReleaseVersion     string           `json:"release_version"`
	StartDate          string           `json:"start_date"`
	EndDate            string           `json:"end_date"`
	Questions          int              `json:"questions"`
	Scenarios          []ScenarioReport `json:"scenarios"`
	Evaluated          bool             `json:"evaluated"`
	ResultClaimAllowed bool             `json:"result_claim_allowed"`
	EvidenceClass      string           `json:"evidence_class"`
	PromotionRequired  []string         `json:"promotion_required"`
}

// BuildRunReport runs the committed fixture end-to-end under cfg: it scopes the
// fixture to the requested scenario (as lcb_runner runs one scenario per
// invocation) and emits a result-claim-gated report. The release defaults to
// the fixture's pinned release when cfg leaves it empty.
func BuildRunReport(f Fixture, cfg RunConfig) RunReport {
	scen := string(cfg.Scenario)
	questions := 0
	for _, item := range f.Items {
		if item.Scenario == scen {
			questions++
		}
	}
	scenarios := make([]ScenarioReport, 0, 1)
	if questions > 0 {
		scenarios = append(scenarios, ScenarioReport{Scenario: scen, Questions: questions})
	}
	release := strings.TrimSpace(cfg.ReleaseVersion)
	if release == "" {
		release = f.ReleaseVersion
	}
	return RunReport{
		Schema:             RunReportSchema,
		Config:             cfg,
		ReleaseVersion:     release,
		StartDate:          f.StartDate,
		EndDate:            f.EndDate,
		Questions:          questions,
		Scenarios:          scenarios,
		Evaluated:          cfg.Evaluate,
		ResultClaimAllowed: false,
		EvidenceClass:      EvidenceFixtureSmoke,
		PromotionRequired:  PromotionRequirements(),
	}
}

// ValidateRunReport enforces the honesty fence on a run report: a fixture run
// can never promote itself into a claimable score, and it must have scored at
// least one question of the requested scenario.
func ValidateRunReport(r RunReport) error {
	if r.Schema != RunReportSchema {
		return fmt.Errorf("livecodebench run report: schema = %q, want %q", r.Schema, RunReportSchema)
	}
	if r.ResultClaimAllowed {
		return fmt.Errorf("livecodebench run report: result_claim_allowed must be false for a fixture run")
	}
	if r.Questions == 0 {
		return fmt.Errorf("livecodebench run report: no questions for scenario %q in the fixture", r.Config.Scenario)
	}
	return nil
}
