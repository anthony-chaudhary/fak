package assumecheck

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// session_bridge.go — read-only bridge from ctxplan's session assumption ledger
// to assumecheck's closed outcome vocabulary (#3828, epic #3818 C9).
//
// DEDUP NOTE:
// Rather than re-implement a separate session ledger or duplicate the scoring
// mechanism, this bridge directly consumes the shipped per-session assumption
// ledger from internal/ctxplan.AssessAssumptions and ctxplan.AssumptionReport
// (#1570, #1616). The scoring, confidence normalization, and action derivation
// belong to ctxplan (#1616); this file is purely a read-only translation bridge
// that maps ctxplan's session assessments onto assumecheck's LevelSession
// assumptions (source-tagged "managed-context") and the closed Outcome vocabulary
// (OutcomeHolds, OutcomeStale, OutcomeUnverifiable, OutcomeViolated).

// SessionAssumptionSource is the provenance tag for session assumptions ingested
// from the managed-context subsystem (#1570).
const SessionAssumptionSource = "managed-context"

// AssessedAssumption binds an assumption to its mapped outcome, reason, and source
// from a session-level assessment (e.g. from ctxplan's managed-context ledger).
type AssessedAssumption struct {
	Assumption Assumption `json:"assumption"`
	Outcome    Outcome    `json:"outcome"`
	Reason     string     `json:"reason"`
	Source     string     `json:"source"`
}

// Verdict converts the assessed assumption to an assumecheck.Verdict.
func (a AssessedAssumption) Verdict() Verdict {
	return Verdict{
		AssumptionID: a.Assumption.ID,
		Level:        a.Assumption.Level,
		Witness:      a.Assumption.WitnessKind,
		Outcome:      a.Outcome,
		Reason:       a.Reason,
	}
}

// BlocksReliance reports whether the outcome prevents a caller from relying on
// this assumption (i.e. anything other than OutcomeHolds).
func (a AssessedAssumption) BlocksReliance() bool {
	return a.Outcome != OutcomeHolds
}

// MapOutcome maps one ctxplan.AssumptionAssessment and the report-level effect-safe gate
// onto the shared assumecheck.Outcome vocabulary:
//   - stale -> OutcomeStale
//   - holds / high confidence (AssumptionUse) -> OutcomeHolds
//   - unknown below policy -> OutcomeUnverifiable
//   - low-confidence below policy -> OutcomeViolated if not EffectSafe, else OutcomeUnverifiable
func MapOutcome(assessment ctxplan.AssumptionAssessment, effectSafe bool) (Outcome, string) {
	// 1. Stale assumptions map to OutcomeStale.
	if assessment.Source == ctxplan.AssumptionStale || assessment.Action == ctxplan.AssumptionRefresh {
		reason := assessment.Reason
		if reason == "" {
			reason = "stale assumption must refresh its source before effects"
		}
		return OutcomeStale, reason
	}

	// 2. Holds / high confidence (cleared policy threshold) maps to OutcomeHolds.
	if assessment.Action == ctxplan.AssumptionUse {
		reason := assessment.Reason
		if reason == "" {
			reason = "assumption cleared confidence threshold and holds"
		}
		return OutcomeHolds, reason
	}

	// 3. Low-confidence or unknown below policy:
	// If unknown/unverifiable -> OutcomeUnverifiable.
	// If not EffectSafe -> OutcomeViolated (otherwise OutcomeUnverifiable).
	switch assessment.Source {
	case ctxplan.AssumptionUserStated, ctxplan.AssumptionWitnessed, ctxplan.AssumptionInferred:
		if !effectSafe {
			reason := assessment.Reason
			if reason == "" {
				reason = "assumption below confidence threshold and plan is not effect-safe"
			}
			return OutcomeViolated, reason
		}
		reason := assessment.Reason
		if reason == "" {
			reason = "assumption below confidence threshold: unverifiable"
		}
		return OutcomeUnverifiable, reason
	default:
		reason := assessment.Reason
		if reason == "" {
			reason = "unknown assumption cannot be verified"
		}
		return OutcomeUnverifiable, reason
	}
}

// MapAssessment maps one ctxplan.AssumptionAssessment into an AssessedAssumption.
func MapAssessment(assessment ctxplan.AssumptionAssessment, effectSafe bool) AssessedAssumption {
	outcome, reason := MapOutcome(assessment, effectSafe)
	id := strings.TrimSpace(assessment.Key)
	if id == "" {
		id = "session-assumption"
	}
	statement := strings.TrimSpace(assessment.Statement)
	if statement == "" {
		statement = id
	}
	refusalReason := outcome.RefusalReason()
	if refusalReason == "" && outcome != OutcomeHolds {
		refusalReason = "ASSUMPTION_VIOLATED"
	}
	assumption := Assumption{
		ID:              id,
		Owner:           SessionAssumptionSource,
		Statement:       statement,
		Level:           LevelSession,
		WitnessKind:     WitnessSessionReport,
		RefusalReason:   refusalReason,
		ConfidenceClass: string(assessment.Source),
		WitnessStatus:   WitnessWired,
	}
	return AssessedAssumption{
		Assumption: assumption,
		Outcome:    outcome,
		Reason:     reason,
		Source:     SessionAssumptionSource,
	}
}

// IngestAssumptionReport ingests an AssumptionReport from ctxplan, mapping its assessments
// into AssessedAssumption records tagged with LevelSession and source "managed-context".
func IngestAssumptionReport(report ctxplan.AssumptionReport) []AssessedAssumption {
	if len(report.Assessments) == 0 {
		return nil
	}
	out := make([]AssessedAssumption, 0, len(report.Assessments))
	for _, assessment := range report.Assessments {
		out = append(out, MapAssessment(assessment, report.EffectSafe))
	}
	return out
}

// IngestSessionReport is an alias for IngestAssumptionReport.
func IngestSessionReport(report ctxplan.AssumptionReport) []AssessedAssumption {
	return IngestAssumptionReport(report)
}

// SessionBridge is an alias for IngestAssumptionReport.
func SessionBridge(report ctxplan.AssumptionReport) []AssessedAssumption {
	return IngestAssumptionReport(report)
}

// IngestSessionAssumptions assesses raw ctxplan assumptions using the given policy and maps
// the resulting report into AssessedAssumption records.
func IngestSessionAssumptions(assumptions []ctxplan.Assumption, policy ctxplan.AssumptionPolicy) []AssessedAssumption {
	report := ctxplan.AssessAssumptions(assumptions, policy)
	return IngestAssumptionReport(report)
}

// IngestSessionAssumptionsDefault assesses raw ctxplan assumptions using ctxplan's default
// policy and maps the resulting report into AssessedAssumption records.
func IngestSessionAssumptionsDefault(assumptions []ctxplan.Assumption) []AssessedAssumption {
	return IngestSessionAssumptions(assumptions, ctxplan.DefaultAssumptionPolicy())
}

// AssumptionsFromReport extracts the []Assumption slice from an AssumptionReport,
// tagged with LevelSession and source "managed-context".
func AssumptionsFromReport(report ctxplan.AssumptionReport) []Assumption {
	assessed := IngestAssumptionReport(report)
	if len(assessed) == 0 {
		return nil
	}
	out := make([]Assumption, len(assessed))
	for i, a := range assessed {
		out[i] = a.Assumption
	}
	return out
}

// VerdictsFromReport extracts the []Verdict slice from an AssumptionReport.
func VerdictsFromReport(report ctxplan.AssumptionReport) []Verdict {
	assessed := IngestAssumptionReport(report)
	if len(assessed) == 0 {
		return nil
	}
	out := make([]Verdict, len(assessed))
	for i, a := range assessed {
		out[i] = a.Verdict()
	}
	return out
}
