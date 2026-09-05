package shipgate

const (
	// AdoptionReportSchema versions the committed adoption artifact.
	AdoptionReportSchema = "fak.shipgate.crossaudit-adoption/v1"

	crossAuditCalibrationVersion = "issue-resolution-audit/v2"

	crossAuditReceiptFreshnessNanos int64 = 14 * 24 * 60 * 60 * 1_000_000_000
)

var crossAuditCalibratedFamilies = []string{"claude", "gpt"}

// DefaultCrossAuditPolicy returns the baseline policy calibrated against evaluated models.
func DefaultCrossAuditPolicy() CrossAuditPolicy {
	return CrossAuditPolicy{
		RequiredCalibrationVersion: crossAuditCalibrationVersion,
		CalibratedAuditorFamilies:  append([]string(nil), crossAuditCalibratedFamilies...),
		MaxReceiptAgeNanos:         crossAuditReceiptFreshnessNanos,
		Prereqs: Prerequisites{
			CalibratedAuditorFamilies: len(crossAuditCalibratedFamilies),
			MinIndependent:            2,
			DogfoodGreen:              false,
		},
	}
}

// AdoptionStage represents the operational stage of cross-audit enforcement.
type AdoptionStage string

// AdoptionStage constants indicate operational deployment stages.
const (
	StageDryRun  AdoptionStage = "dry-run-report-only"
	StageEnforce AdoptionStage = "enforce-fail-closed"
)

// CalibrationEvidence tracks prerequisite calibration status.
type CalibrationEvidence struct {
	Source             string   `json:"source"`
	PromptVersion      string   `json:"prompt_version"`
	CalibratedFamilies []string `json:"calibrated_families"`
	FamilyCount        int      `json:"family_count"`
	MinIndependent     int      `json:"min_independent"`
	Met                bool     `json:"met"`
}

// DogfoodEvidence records dogfood deployment status for policy gating.
type DogfoodEvidence struct {
	Source   string `json:"source"`
	Grade    string `json:"grade"`
	DarkLoop bool   `json:"dark_loop"`
	Green    bool   `json:"green"`
	Note     string `json:"note"`
}

// AdoptionReport documents policy prerequisites and enforcement readiness.
type AdoptionReport struct {
	Schema           string              `json:"schema"`
	Issue            int                 `json:"issue"`
	Epic             int                 `json:"epic"`
	Calibration      CalibrationEvidence `json:"calibration"`
	Dogfood          DogfoodEvidence     `json:"dogfood"`
	PrereqsMet       bool                `json:"prereqs_met"`
	RecommendedStage AdoptionStage       `json:"recommended_stage"`
	Rationale        string              `json:"rationale"`
}

// CrossAuditAdoptionReport builds the adoption report from a policy's measured prerequisite evidence.
func CrossAuditAdoptionReport(pol CrossAuditPolicy) AdoptionReport {
	met := pol.Prereqs.Met()
	rep := AdoptionReport{
		Schema: AdoptionReportSchema,
		Issue:  3860,
		Epic:   3846,
		Calibration: CalibrationEvidence{
			Source:             "experiments/crossaudit-calibration-3854/report.json",
			PromptVersion:      pol.RequiredCalibrationVersion,
			CalibratedFamilies: append([]string(nil), pol.CalibratedAuditorFamilies...),
			FamilyCount:        pol.Prereqs.CalibratedAuditorFamilies,
			MinIndependent:     pol.Prereqs.MinIndependent,
			Met:                pol.Prereqs.CalibratedAuditorFamilies >= pol.Prereqs.MinIndependent && pol.Prereqs.MinIndependent >= 2,
		},
		Dogfood: DogfoodEvidence{
			Source:   "experiments/crossaudit-dogfood-3859/scorecard.json",
			Grade:    "F",
			DarkLoop: true,
			Green:    pol.Prereqs.DogfoodGreen,
			Note:     "background audit loop not running (dark_loop=true); 1 auditor provider unavailable",
		},
		PrereqsMet: met,
	}
	if met {
		rep.RecommendedStage = StageEnforce
		rep.Rationale = "calibration (>=2 independent families) and a green dogfood rollout are both satisfied; the high-risk closure gate may fail closed by default."
	} else {
		rep.RecommendedStage = StageDryRun
		rep.Rationale = "calibration is satisfied (2 independent calibrated families) but the dogfood rollout is not green (live audit loop dark); the gate ships enforcement-capable but stays in dry-run until dogfood is green — no default-on enablement while prerequisites are unmet."
	}
	return rep
}
