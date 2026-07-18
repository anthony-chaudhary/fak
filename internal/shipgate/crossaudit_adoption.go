package shipgate

// crossaudit_adoption.go — the calibrated policy instance for the #3860 high-risk
// closure gate, plus the committed ADOPTION REPORT that records the measured
// prerequisite evidence and the resulting staged-enablement recommendation.
//
// The report is the issue's named witness: "a committed adoption report showing
// prerequisite calibration and dogfood thresholds satisfied before default-on." It is
// generated from the same measured evidence the policy is built from, so the committed
// artifact (testdata/crossaudit_adoption_report.json) can never silently drift from the
// code — the test regenerates it and compares.
//
// The bound evidence, verbatim from the two closed blocker issues:
//   - #3854 calibration (experiments/crossaudit-calibration-3854/report.json):
//     prompt_version "issue-resolution-audit/v2"; TWO independent families reached
//     status "calibrated" — anthropic/claude and openai/gpt; the local open-weight arm
//     is "not-yet" (no authenticated sanctioned-node bridge this session).
//   - #3859 dogfood (experiments/crossaudit-dogfood-3859/scorecard.json): grade F,
//     ok=false, dark_loop=true — the live background audit loop is not running and one
//     provider is unavailable.
//
// Because the dogfood loop is dark, the prerequisites for DEFAULT-ON enforcement are
// NOT met, so the recommended stage is DRY-RUN (report-only). Shipping the
// enforcement-capable policy while keeping it in dry-run is exactly the issue's
// out-of-scope rule "no enablement when calibration/dogfood prerequisites are not met",
// and its closure binding "no default-on change lands in the same commit as
// uncalibrated plumbing" — this commit flips nothing on.

const (
	// AdoptionReportSchema versions the committed adoption artifact.
	AdoptionReportSchema = "fak.shipgate.crossaudit-adoption/v1"

	// crossAuditCalibrationVersion is the calibrated prompt/policy version from #3854.
	crossAuditCalibrationVersion = "issue-resolution-audit/v2"

	// crossAuditReceiptFreshnessNanos bounds how old an audit receipt may be and still
	// open the gate: 14 days. A receipt older than the window is STALE (re-audit).
	crossAuditReceiptFreshnessNanos int64 = 14 * 24 * 60 * 60 * 1_000_000_000
)

// crossAuditCalibratedFamilies is the calibrated auditor allowlist from #3854 — the
// two independent families that reached status "calibrated". It is data, not a
// hard-coded monopoly: adding a third calibrated family is an allowlist edit.
var crossAuditCalibratedFamilies = []string{"claude", "gpt"}

// DefaultCrossAuditPolicy is the calibrated #3860 policy built from the measured
// #3854/#3859 evidence. Its Prereqs.Met() is false while the dogfood loop is dark, so
// AdjudicateClosure runs in DRY-RUN until an operator advances the stage on green
// dogfood evidence.
func DefaultCrossAuditPolicy() CrossAuditPolicy {
	return CrossAuditPolicy{
		RequiredCalibrationVersion: crossAuditCalibrationVersion,
		CalibratedAuditorFamilies:  append([]string(nil), crossAuditCalibratedFamilies...),
		MaxReceiptAgeNanos:         crossAuditReceiptFreshnessNanos,
		Prereqs: Prerequisites{
			CalibratedAuditorFamilies: len(crossAuditCalibratedFamilies), // 2 (#3854)
			MinIndependent:            2,
			DogfoodGreen:              false, // #3859 scorecard: grade F, dark_loop=true
		},
	}
}

// AdoptionStage is the staged-enablement recommendation.
type AdoptionStage string

const (
	StageDryRun  AdoptionStage = "dry-run-report-only" // enforcement disabled; report what it would do
	StageEnforce AdoptionStage = "enforce-fail-closed" // default-on; block on missing/invalid receipts
)

// CalibrationEvidence records the #3854 prerequisite as adopted by the policy.
type CalibrationEvidence struct {
	Source             string   `json:"source"`
	PromptVersion      string   `json:"prompt_version"`
	CalibratedFamilies []string `json:"calibrated_families"`
	FamilyCount        int      `json:"family_count"`
	MinIndependent     int      `json:"min_independent"`
	Met                bool     `json:"met"`
}

// DogfoodEvidence records the #3859 prerequisite as read from the rollout scorecard.
type DogfoodEvidence struct {
	Source   string `json:"source"`
	Grade    string `json:"grade"`
	DarkLoop bool   `json:"dark_loop"`
	Green    bool   `json:"green"`
	Note     string `json:"note"`
}

// AdoptionReport is the committed witness that the prerequisite evidence was evaluated
// before any default-on decision. It is fully derived from DefaultCrossAuditPolicy so
// the artifact cannot drift from the code that enforces the gate.
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

// CrossAuditAdoptionReport builds the adoption report from a policy's measured
// prerequisite evidence. A policy whose prerequisites are met recommends enforcement;
// otherwise it recommends dry-run and names the missing prerequisite.
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
