package modelroute

import (
	"encoding/json"
	"fmt"
	"os"
)

const CrossAuditCalibrationRunSchema = "fak-crossaudit-calibration-run/v1"

type CalibrationArmSpec struct {
	Name         string        `json:"name"`
	Auditor      AuditIdentity `json:"auditor"`
	ReceiptFile  string        `json:"receipt_file,omitempty"`
	Status       string        `json:"status"`
	NotYetReason string        `json:"not_yet_reason,omitempty"`
}

type CalibrationRunManifest struct {
	Schema                string                   `json:"schema"`
	Corpus                string                   `json:"corpus"`
	MaxSamplesPerArm      int                      `json:"max_samples_per_arm"`
	MaxCostMicrosUSD      int64                    `json:"max_cost_micros_usd"`
	HoldoutPairPrefixes   []string                 `json:"holdout_pair_prefixes"`
	HighSeverityClasses   []AccidentalFailureClass `json:"high_severity_classes"`
	MinHighSeverityRecall float64                  `json:"min_high_severity_recall"`
	MaxFalsePositiveRate  float64                  `json:"max_false_positive_rate"`
	Arms                  []CalibrationArmSpec     `json:"arms"`
}

func (m CalibrationRunManifest) Validate() error {
	if m.Schema != CrossAuditCalibrationRunSchema || m.Corpus != AccidentalCorpusManifestSchema {
		return fmt.Errorf("modelroute: calibration run manifest schema/corpus is invalid")
	}
	if m.MaxSamplesPerArm <= 0 || m.MaxCostMicrosUSD < 0 || len(m.Arms) == 0 {
		return fmt.Errorf("modelroute: calibration run manifest bounds are invalid")
	}
	if m.MinHighSeverityRecall < 0 || m.MinHighSeverityRecall > 1 || m.MaxFalsePositiveRate < 0 || m.MaxFalsePositiveRate > 1 {
		return fmt.Errorf("modelroute: calibration thresholds must be within 0..1")
	}
	seen := map[string]bool{}
	for _, a := range m.Arms {
		if a.Name == "" || seen[a.Name] {
			return fmt.Errorf("modelroute: duplicate/empty calibration arm %q", a.Name)
		}
		seen[a.Name] = true
		if a.Auditor.Provider == "" || a.Auditor.Family == "" || a.Auditor.Model == "" || a.Auditor.WeightsRevision == "" || a.Auditor.ReasoningPosture == "" || a.Auditor.Driver == "" {
			return fmt.Errorf("modelroute: calibration arm %s provenance incomplete", a.Name)
		}
		if a.Status != "available" && a.Status != "not-yet" {
			return fmt.Errorf("modelroute: calibration arm %s status %q invalid", a.Name, a.Status)
		}
		if a.Status == "not-yet" && a.NotYetReason == "" {
			return fmt.Errorf("modelroute: calibration arm %s missing not-yet reason", a.Name)
		}
	}
	return nil
}

func LoadCalibrationObservations(path string) ([]CalibrationObservation, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []CalibrationObservation
	if err := json.Unmarshal(b, &rows); err != nil {
		return nil, err
	}
	return rows, nil
}

func AccidentalCalibrationTruth() ([]CalibrationTruth, error) {
	fixtures := AccidentalCorpus()
	truth := make([]CalibrationTruth, 0, len(fixtures))
	for _, f := range fixtures {
		b, err := f.Bundle()
		if err != nil {
			return nil, err
		}
		truth = append(truth, CalibrationTruth{ID: f.ID, Class: f.Class, Corrupt: f.Corrupt, BundleDigest: b.BundleDigest})
	}
	return truth, nil
}

func DeriveCrossAuditCalibration(m CalibrationRunManifest, observations map[string][]CalibrationObservation) (CrossAuditCalibrationReport, error) {
	if err := m.Validate(); err != nil {
		return CrossAuditCalibrationReport{}, err
	}
	truth, err := AccidentalCalibrationTruth()
	if err != nil {
		return CrossAuditCalibrationReport{}, err
	}
	var all []CalibrationObservation
	for _, arm := range m.Arms {
		if arm.Status == "not-yet" {
			continue
		}
		rows := observations[arm.Name]
		if len(rows) > m.MaxSamplesPerArm {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: arm %s exceeds sample cap", arm.Name)
		}
		for _, row := range rows {
			if calibrationAuditorKey(row.Auditor) != calibrationAuditorKey(arm.Auditor) {
				return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: arm %s observation identity mismatch", arm.Name)
			}
		}
		all = append(all, rows...)
	}
	report, err := BuildCrossAuditCalibrationReport(truth, all)
	if err != nil {
		return CrossAuditCalibrationReport{}, err
	}
	observed := map[string]bool{}
	for _, a := range report.Arms {
		observed[calibrationAuditorKey(a.Auditor)] = true
	}
	var total int64
	for _, a := range report.Arms {
		total += a.Metrics.CostMicrosUSD
	}
	if total > m.MaxCostMicrosUSD {
		return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: calibration cost %d exceeds cap %d", total, m.MaxCostMicrosUSD)
	}
	for _, spec := range m.Arms {
		if spec.Status == "not-yet" {
			report.Arms = append(report.Arms, CalibrationArm{Auditor: spec.Auditor, Status: "not-yet", NotYetReason: spec.NotYetReason})
		} else if !observed[calibrationAuditorKey(spec.Auditor)] {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: available arm %s has no observations", spec.Name)
		}
	}
	return report, nil
}

// CalibrationReportFromJSON is a small exported, deterministic report-derivation
// seam for experiment harnesses. It keeps benchmark scripts out of internal
// implementation details while preserving the same validation as the Go tests.
func CalibrationReportFromJSON(manifestJSON []byte, observationJSON map[string][]byte) ([]byte, error) {
	var manifest CalibrationRunManifest
	if err := json.Unmarshal(manifestJSON, &manifest); err != nil {
		return nil, err
	}
	rows := map[string][]CalibrationObservation{}
	for arm, raw := range observationJSON {
		var r []CalibrationObservation
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, err
		}
		rows[arm] = r
	}
	report, err := DeriveCrossAuditCalibration(manifest, rows)
	if err != nil {
		return nil, err
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
