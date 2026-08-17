package disambiguation

import "bytes"

type CommittedFreshnessVerdict string

const (
	CommittedFreshnessClean        CommittedFreshnessVerdict = "committed-clean"
	CommittedFreshnessOverlayDrift CommittedFreshnessVerdict = "own-overlay-drift"
	CommittedFreshnessUnavailable  CommittedFreshnessVerdict = "unavailable"
)

type CommittedFreshnessReport struct {
	Schema         string                    `json:"schema"`
	Verdict        CommittedFreshnessVerdict `json:"verdict"`
	CommittedClean bool                      `json:"committed_clean"`
	OverlayDrift   bool                      `json:"overlay_drift"`
	ProbeAvailable bool                      `json:"probe_available"`
	Reason         string                    `json:"reason,omitempty"`
}

func EvaluateCommittedFreshness(committed, overlay []byte, probeErr error) CommittedFreshnessReport {
	report := CommittedFreshnessReport{Schema: "fak-disambiguation-committed-freshness/1"}
	if probeErr != nil {
		report.Verdict = CommittedFreshnessUnavailable
		report.Reason = probeErr.Error()
		return report
	}
	report.ProbeAvailable = true
	report.CommittedClean = true
	report.OverlayDrift = !bytes.Equal(committed, overlay)
	if report.OverlayDrift {
		report.Verdict = CommittedFreshnessOverlayDrift
	} else {
		report.Verdict = CommittedFreshnessClean
	}
	return report
}
