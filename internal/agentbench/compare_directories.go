package agentbench

import "errors"

type comparisonMetricVerdict struct {
	Status string       `json:"status"`
	Reason string       `json:"reason,omitempty"`
	Metric PairedMetric `json:"metric"`
}

type comparisonArmObservation struct {
	ArmID                string                `json:"arm_id"`
	QualifiedConcurrency int                   `json:"qualified_concurrency"`
	Replicates           []ComparisonReplicate `json:"replicates"`
}

type compareDirectoryReceipt struct {
	Schema            string                             `json:"schema"`
	Status            string                             `json:"status"`
	Reason            string                             `json:"reason,omitempty"`
	Baseline          comparisonArmArtifact              `json:"baseline"`
	Candidate         comparisonArmArtifact              `json:"candidate"`
	BaselineObserved  comparisonArmObservation           `json:"baseline_observed"`
	CandidateObserved comparisonArmObservation           `json:"candidate_observed"`
	Comparison        ComparisonResult                   `json:"comparison"`
	MetricVerdicts    map[string]comparisonMetricVerdict `json:"metric_verdicts,omitempty"`
}

func compareDirectories(baselineDir, candidateDir string) (compareDirectoryReceipt, error) {
	r := compareDirectoryReceipt{Schema: "fak.agentbench.comparison.v1", Status: "INCONCLUSIVE"}
	a, aa, baselineErr := readComparisonArm(baselineDir)
	r.Baseline = aa
	r.BaselineObserved = comparisonArmObservation{ArmID: a.ArmID, QualifiedConcurrency: a.QualifiedConcurrency, Replicates: a.Replicates}
	b, bb, candidateErr := readComparisonArm(candidateDir)
	r.Candidate = bb
	r.CandidateObserved = comparisonArmObservation{ArmID: b.ArmID, QualifiedConcurrency: b.QualifiedConcurrency, Replicates: b.Replicates}
	for _, readErr := range []error{baselineErr, candidateErr} {
		if readErr == nil {
			continue
		}
		if errors.Is(readErr, errComparisonInconclusive) {
			r.Reason = readErr.Error()
			return r, nil
		}
		return r, readErr
	}
	if !comparisonEnvelopeMatches(aa, bb) {
		r.Reason = "comparison envelope identity mismatch"
		return r, nil
	}
	pairedContexts := make(map[int]string, len(aa.Replicates))
	for _, replicate := range aa.Replicates {
		pairedContexts[replicate.Pair] = replicate.ContextManifestSHA256
	}
	for _, replicate := range bb.Replicates {
		if pairedContexts[replicate.Pair] == "" || pairedContexts[replicate.Pair] != replicate.ContextManifestSHA256 {
			r.Reason = "paired context manifest or campaign namespace mismatch"
			return r, nil
		}
	}
	result, err := ComparePaired(a, b)
	if err != nil {
		return r, err
	}
	r.Comparison = result
	if result.Status != "COMPARABLE" {
		r.Reason = result.Reason
		return r, nil
	}
	r.MetricVerdicts = map[string]comparisonMetricVerdict{}
	noisy := false
	for name, m := range result.Metrics {
		v := comparisonMetricVerdict{Metric: m}
		if m.BootstrapHighMillis < 0 {
			v.Status = "CANDIDATE_LOWER"
		} else if m.BootstrapLowMillis > 0 {
			v.Status = "BASELINE_LOWER"
		} else {
			v.Status = "NOISY"
			noisy = true
		}
		r.MetricVerdicts[name] = v
	}
	if noisy {
		r.Status = "NOISY_INCONCLUSIVE"
		r.Reason = "at least one paired metric includes zero"
	} else {
		r.Status = "COMPARABLE"
		r.Reason = "per-metric directions only; no overall winner"
	}
	return r, nil
}
